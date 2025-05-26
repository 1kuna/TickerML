#!/usr/bin/env python3
"""
Inference service for Raspberry Pi
Loads quantized ONNX model and makes predictions every 5 minutes
"""

import sqlite3
import numpy as np
import pandas as pd
import onnxruntime as ort
import logging
from datetime import datetime, timedelta
from pathlib import Path
import json

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration
DB_PATH = Path(__file__).parent.parent / "data" / "db" / "crypto_data.db"
MODEL_PATH = Path(__file__).parent.parent / "models" / "onnx" / "crypto_transformer_quantized.onnx"
CONFIG_PATH = Path(__file__).parent.parent / "config" / "config.yaml"
SYMBOLS = ["BTCUSDT", "ETHUSDT"]
SEQUENCE_LENGTH = 60  # 60 minutes of history

def init_predictions_table():
    """Initialize predictions table in database"""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp INTEGER NOT NULL,
                symbol TEXT NOT NULL,
                prediction_5min REAL NOT NULL,
                prediction_10min REAL NOT NULL,
                prediction_30min REAL NOT NULL,
                confidence_up REAL NOT NULL,
                confidence_down REAL NOT NULL,
                model_version TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(timestamp, symbol)
            )
        """)
        
        # Create index for faster queries
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_pred_timestamp_symbol 
            ON predictions(timestamp, symbol)
        """)
        
        conn.commit()
        conn.close()
        logger.info("Predictions table initialized")
        
    except sqlite3.Error as e:
        logger.error(f"Error initializing predictions table: {e}")
        raise

def load_onnx_model():
    """Load quantized ONNX model"""
    try:
        if not MODEL_PATH.exists():
            logger.error(f"Model not found at {MODEL_PATH}")
            return None
            
        # Create ONNX Runtime session with CPU provider
        providers = ['CPUExecutionProvider']
        session = ort.InferenceSession(str(MODEL_PATH), providers=providers)
        
        logger.info(f"Loaded ONNX model from {MODEL_PATH}")
        logger.info(f"Model inputs: {[input.name for input in session.get_inputs()]}")
        logger.info(f"Model outputs: {[output.name for output in session.get_outputs()]}")
        
        return session
        
    except Exception as e:
        logger.error(f"Error loading ONNX model: {e}")
        return None

def get_latest_features(symbol, sequence_length=60):
    """Get latest features for a symbol from database"""
    try:
        conn = sqlite3.connect(DB_PATH)
        
        # Get latest timestamp
        latest_query = "SELECT MAX(timestamp) FROM ohlcv WHERE symbol = ?"
        latest_timestamp = pd.read_sql_query(latest_query, conn, params=(symbol,)).iloc[0, 0]
        
        if latest_timestamp is None:
            logger.warning(f"No data found for {symbol}")
            return None
        
        # Get last N minutes of data
        start_timestamp = latest_timestamp - (sequence_length * 60 * 1000)  # Convert to milliseconds
        
        query = """
            SELECT timestamp, open, high, low, close, volume
            FROM ohlcv 
            WHERE symbol = ? AND timestamp >= ? AND timestamp <= ?
            ORDER BY timestamp ASC
        """
        
        df = pd.read_sql_query(query, conn, params=(symbol, start_timestamp, latest_timestamp))
        conn.close()
        
        if len(df) < sequence_length:
            logger.warning(f"Insufficient data for {symbol}: {len(df)} < {sequence_length}")
            return None
        
        # Take last sequence_length records
        df = df.tail(sequence_length).copy()
        
        # Basic feature engineering (placeholder - should match training features)
        features = compute_basic_features(df)
        
        return features
        
    except Exception as e:
        logger.error(f"Error getting features for {symbol}: {e}")
        return None

def compute_basic_features(df):
    """Compute basic technical indicators (placeholder implementation)"""
    try:
        # Normalize prices
        df['price_norm'] = (df['close'] - df['close'].mean()) / df['close'].std()
        
        # Simple moving averages
        df['sma_5'] = df['close'].rolling(5).mean()
        df['sma_10'] = df['close'].rolling(10).mean()
        
        # Price changes
        df['price_change'] = df['close'].pct_change()
        df['volume_change'] = df['volume'].pct_change()
        
        # Volatility
        df['volatility'] = df['price_change'].rolling(10).std()
        
        # High-low spread
        df['hl_spread'] = (df['high'] - df['low']) / df['close']
        
        # Select features (should match training feature set)
        feature_columns = [
            'price_norm', 'price_change', 'volume_change', 
            'volatility', 'hl_spread'
        ]
        
        # Fill NaN values
        for col in feature_columns:
            df[col] = df[col].fillna(0)
        
        # Return as numpy array
        features = df[feature_columns].values
        
        return features.astype(np.float32)
        
    except Exception as e:
        logger.error(f"Error computing features: {e}")
        return None

def make_prediction(session, features):
    """Make prediction using ONNX model"""
    try:
        # Reshape features for model input [batch_size, sequence_length, feature_dim]
        input_data = features.reshape(1, *features.shape)
        
        # Get input name
        input_name = session.get_inputs()[0].name
        
        # Run inference
        outputs = session.run(None, {input_name: input_data})
        
        # Parse outputs (assuming regression and classification heads)
        regression_output = outputs[0][0]  # [price_5min, price_10min, price_30min]
        classification_output = outputs[1][0]  # [down_prob, up_prob]
        
        # Apply softmax to classification output
        exp_scores = np.exp(classification_output)
        probabilities = exp_scores / np.sum(exp_scores)
        
        return {
            'prediction_5min': float(regression_output[0]),
            'prediction_10min': float(regression_output[1]),
            'prediction_30min': float(regression_output[2]),
            'confidence_down': float(probabilities[0]),
            'confidence_up': float(probabilities[1])
        }
        
    except Exception as e:
        logger.error(f"Error making prediction: {e}")
        return None

def store_prediction(symbol, timestamp, prediction):
    """Store prediction in database"""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT OR REPLACE INTO predictions 
            (timestamp, symbol, prediction_5min, prediction_10min, prediction_30min,
             confidence_up, confidence_down, model_version)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            timestamp,
            symbol,
            prediction['prediction_5min'],
            prediction['prediction_10min'],
            prediction['prediction_30min'],
            prediction['confidence_up'],
            prediction['confidence_down'],
            'v1.0'  # Model version
        ))
        
        conn.commit()
        conn.close()
        
        logger.info(f"Stored prediction for {symbol} at {datetime.fromtimestamp(timestamp/1000)}")
        logger.info(f"  5min: {prediction['prediction_5min']:.4f}, "
                   f"10min: {prediction['prediction_10min']:.4f}, "
                   f"30min: {prediction['prediction_30min']:.4f}")
        logger.info(f"  Confidence - Up: {prediction['confidence_up']:.3f}, "
                   f"Down: {prediction['confidence_down']:.3f}")
        
    except sqlite3.Error as e:
        logger.error(f"Error storing prediction for {symbol}: {e}")

def main():
    """Main inference function"""
    logger.info("Starting inference service")
    
    # Initialize predictions table
    init_predictions_table()
    
    # Load ONNX model
    session = load_onnx_model()
    if session is None:
        logger.error("Failed to load model, exiting")
        return
    
    # Get current timestamp
    current_timestamp = int(datetime.now().timestamp() * 1000)
    
    # Make predictions for each symbol
    for symbol in SYMBOLS:
        logger.info(f"Making prediction for {symbol}")
        
        # Get latest features
        features = get_latest_features(symbol, SEQUENCE_LENGTH)
        if features is None:
            logger.warning(f"Skipping {symbol} due to insufficient data")
            continue
        
        # Make prediction
        prediction = make_prediction(session, features)
        if prediction is None:
            logger.warning(f"Failed to make prediction for {symbol}")
            continue
        
        # Store prediction
        store_prediction(symbol, current_timestamp, prediction)
    
    logger.info("Inference completed")

if __name__ == "__main__":
    main() 