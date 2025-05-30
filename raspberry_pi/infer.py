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
import yaml
import pickle
import signal
import os

# Global shutdown flag
shutdown_requested = False

# Signal handler
def handle_shutdown_signal(signum, frame):
    global shutdown_requested
    logger.info(f"Received signal {signum}. Requesting shutdown...")
    shutdown_requested = True

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# --- Configuration Loading ---
DEFAULT_DB_PATH = Path(__file__).parent.parent / "data" / "db" / "crypto_ohlcv.db"
DEFAULT_SYMBOLS = ["BTCUSDT", "ETHUSDT"] # Used if config or metadata fails
DEFAULT_MODELS_PATH = Path(__file__).parent.parent / "models"
DEFAULT_ONNX_MODELS_PATH = "onnx"
DEFAULT_CHECKPOINTS_PATH = "checkpoints"
DEFAULT_ONNX_MODEL_NAME = "crypto_transformer_quantized.onnx"
DEFAULT_METADATA_NAME = "model_metadata.json"
DEFAULT_SCALER_NAME = "feature_scaler.pkl"
DEFAULT_SEQUENCE_LENGTH = 60  # Used if metadata loading fails
DEFAULT_FEATURE_COLUMNS = [] # Used if metadata loading fails

def load_app_config():
    """Loads configuration from YAML file, with fallbacks to defaults."""
    config_file_path = Path(__file__).parent.parent / "config" / "config.yaml"
    
    cfg = {
        "db_path": str(DEFAULT_DB_PATH),
        "symbols": DEFAULT_SYMBOLS,
        "models_path": str(DEFAULT_MODELS_PATH),
        "onnx_models_path": DEFAULT_ONNX_MODELS_PATH,
        "checkpoints_path": DEFAULT_CHECKPOINTS_PATH,
        "onnx_model_name": DEFAULT_ONNX_MODEL_NAME,
        "metadata_name": DEFAULT_METADATA_NAME,
        "scaler_name": DEFAULT_SCALER_NAME,
    }

    if not config_file_path.exists():
        logger.warning(f"Config file not found at {config_file_path}. Using default values for all settings.")
        return cfg

    try:
        with open(config_file_path, 'r') as f:
            yaml_config = yaml.safe_load(f)

        if yaml_config:
            cfg["db_path"] = yaml_config.get("database", {}).get("ohlcv_path", cfg["db_path"])
            cfg["symbols"] = yaml_config.get("data", {}).get("symbols", cfg["symbols"])
            cfg["models_path"] = yaml_config.get("paths", {}).get("models", cfg["models_path"])
            cfg["onnx_models_path"] = yaml_config.get("paths", {}).get("onnx", cfg["onnx_models_path"]) # Note: 'onnx' is a sub-path within 'models'
            cfg["checkpoints_path"] = yaml_config.get("paths", {}).get("checkpoints", cfg["checkpoints_path"]) # Note: 'checkpoints' is a sub-path
            
            # Log loaded config values or which ones are using defaults
            for key, default_val in [("db_path", str(DEFAULT_DB_PATH)), ("symbols", DEFAULT_SYMBOLS), 
                                     ("models_path", str(DEFAULT_MODELS_PATH)), ("onnx_models_path", DEFAULT_ONNX_MODELS_PATH),
                                     ("checkpoints_path", DEFAULT_CHECKPOINTS_PATH)]:
                if cfg[key] == default_val and (yaml_config.get("database",{}).get("ohlcv_path") if key == "db_path" else \
                                                yaml_config.get("data",{}).get("symbols") if key == "symbols" else \
                                                yaml_config.get("paths",{}).get(key.replace("_path",""), None)) is not None : # Check if it was explicitly set to default
                    logger.info(f"Loaded {key} from config: {cfg[key]}")
                elif cfg[key] == default_val:
                     logger.warning(f"{key} not found in config or its parent key is missing. Using default: {cfg[key]}")
                else:
                    logger.info(f"Loaded {key} from config: {cfg[key]}")
        else:
            logger.warning(f"Config file {config_file_path} is empty. Using default values for all settings.")
            
    except yaml.YAMLError as e:
        logger.error(f"Error parsing config file {config_file_path}: {e}. Using default values for all settings.")
    except Exception as e:
        logger.error(f"Unexpected error loading config file {config_file_path}: {e}. Using default values for all settings.")
        
    return cfg

app_config = load_app_config()

# Set up global paths and configurations from app_config
DB_PATH = Path(app_config['db_path'])
SYMBOLS = app_config['symbols'] # For iteration
MODELS_BASE_PATH = Path(app_config['models_path'])
MODEL_PATH = MODELS_BASE_PATH / app_config['onnx_models_path'] / app_config['onnx_model_name']
METADATA_PATH = MODELS_BASE_PATH / app_config['onnx_models_path'] / app_config['metadata_name']
SCALER_PATH = MODELS_BASE_PATH / app_config['checkpoints_path'] / app_config['scaler_name']

# Load Model Metadata
SEQUENCE_LENGTH = DEFAULT_SEQUENCE_LENGTH
LOADED_FEATURE_COLUMNS = DEFAULT_FEATURE_COLUMNS
try:
    if METADATA_PATH.exists():
        with open(METADATA_PATH, 'r') as f:
            metadata = json.load(f)
        SEQUENCE_LENGTH = metadata.get('sequence_length', DEFAULT_SEQUENCE_LENGTH)
        LOADED_FEATURE_COLUMNS = metadata.get('feature_columns', DEFAULT_FEATURE_COLUMNS)
        logger.info(f"Loaded model metadata from {METADATA_PATH}")
        logger.info(f"  Sequence Length: {SEQUENCE_LENGTH}")
        logger.info(f"  Feature Columns: {LOADED_FEATURE_COLUMNS}")
        if SEQUENCE_LENGTH == DEFAULT_SEQUENCE_LENGTH and 'sequence_length' not in metadata :
            logger.warning("  sequence_length not found in metadata, using default.")
        if LOADED_FEATURE_COLUMNS == DEFAULT_FEATURE_COLUMNS and 'feature_columns' not in metadata:
            logger.warning("  feature_columns not found in metadata, using default.")
    else:
        logger.error(f"Model metadata file not found at {METADATA_PATH}. Using default sequence length and feature columns.")
except Exception as e:
    logger.error(f"Error loading model metadata from {METADATA_PATH}: {e}. Using defaults.")

# Load Feature Scaler
feature_scaler = None
try:
    if SCALER_PATH.exists():
        with open(SCALER_PATH, 'rb') as f:
            feature_scaler = pickle.load(f)
        logger.info(f"Successfully loaded feature scaler from {SCALER_PATH}")
    else:
        logger.error(f"Feature scaler file not found at {SCALER_PATH}. Scaler will not be used.")
except Exception as e:
    logger.error(f"Error loading feature scaler from {SCALER_PATH}: {e}. Scaler will not be used.")

# Technical Analysis library imports
import ta
from ta.utils import dropna
from ta.volatility import BollingerBands, average_true_range
from ta.momentum import RSIIndicator, StochOscillator, WilliamsRIndicator, ROCIndicator # Adjusted based on pc/features.py actual usage: rsi, stoch, stoch_signal, williams_r, roc
from ta.volume import MFIIndicator # money_flow_index
from ta.trend import MACD, SMAIndicator, EMAIndicator, CCIIndicator # macd, macd_signal, macd_diff, sma, ema, cci

def init_predictions_table():
    """Initialize predictions table in database"""
    conn = None
    try:
        conn = sqlite3.connect(str(DB_PATH))
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
        logger.info("Predictions table initialized")
        
    except sqlite3.Error as e:
        logger.error(f"Error initializing predictions table: {e}")
        # Potentially re-raise if critical, or handle to allow startup to continue if possible
        raise 
    finally:
        if conn:
            conn.close()

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
    conn = None
    try:
        conn = sqlite3.connect(str(DB_PATH)) # Use str(DB_PATH)
        
        # Get latest timestamp
        latest_query = "SELECT MAX(timestamp) FROM ohlcv WHERE symbol = ?"
        # Execute query using cursor for better error handling and resource management
        cursor = conn.cursor()
        cursor.execute(latest_query, (symbol,))
        result = cursor.fetchone()
        
        if result is None or result[0] is None:
            logger.warning(f"No data found for {symbol} in ohlcv table.")
            # conn.close() already handled by finally
            return None
        latest_timestamp = result[0]
        
        # Get last N minutes of data
        start_timestamp = latest_timestamp - (sequence_length * 60 * 1000)  # Convert to milliseconds
        
        query = """
            SELECT timestamp, open, high, low, close, volume
            FROM ohlcv 
            WHERE symbol = ? AND timestamp >= ? AND timestamp <= ?
            ORDER BY timestamp ASC
        """
        # Using pandas read_sql_query, which should ideally manage its own connection if one isn't passed,
        # but since we pass one, we must manage it.
        df = pd.read_sql_query(query, conn, params=(symbol, start_timestamp, latest_timestamp))
        # conn.close() will be handled by finally block
        
        if len(df) < sequence_length:
            logger.warning(f"Insufficient data for {symbol}: {len(df)} < {sequence_length}")
            # conn.close() already handled by finally
            return None
        
        # Take last sequence_length records
        df = df.tail(sequence_length).copy()
        
        features_np = calculate_and_scale_features(df, feature_scaler, LOADED_FEATURE_COLUMNS)
        
        return features_np # This is now a NumPy array
        
    except Exception as e:
        logger.error(f"Error getting features for {symbol}: {e}")
        return None
    finally:
        if conn:
            conn.close()

def calculate_and_scale_features(df_ohlcv, scaler, expected_feature_columns):
    """
    Calculates technical indicators based on expected_feature_columns,
    matches logic from pc/features.py, handles NaNs, and scales the features.
    """
    if not expected_feature_columns:
        logger.error("LOADED_FEATURE_COLUMNS is empty. Cannot calculate features.")
        return np.array([]).reshape(len(df_ohlcv), 0).astype(np.float32)

    features_df = df_ohlcv.copy()

    # Calculate features one by one based on LOADED_FEATURE_COLUMNS
    # This ensures intermediate features like 'price_change' are available if needed by others.
    
    # Pre-calculate 'price_change' if it's a dependency for volatility columns, or if it's a feature itself.
    if any('price_change' in col or 'volatility' in col for col in expected_feature_columns):
        features_df['price_change'] = features_df['close'].pct_change()

    calculated_ta_features = pd.DataFrame(index=features_df.index)

    for col_name in expected_feature_columns:
        if col_name in calculated_ta_features.columns: # Already calculated (e.g. as part of a group like BBands)
            continue
        try:
            if col_name == 'price_change':
                # Already computed if needed, or compute now
                if 'price_change' not in features_df.columns:
                     features_df['price_change'] = features_df['close'].pct_change()
                calculated_ta_features[col_name] = features_df['price_change']
            elif col_name == 'volume_change':
                calculated_ta_features[col_name] = features_df['volume'].pct_change()
            elif col_name == 'high_low_pct':
                calculated_ta_features[col_name] = (features_df['high'] - features_df['low']) / features_df['close']
            elif col_name == 'open_close_pct':
                calculated_ta_features[col_name] = (features_df['close'] - features_df['open']) / features_df['open']
            
            # Moving Averages
            elif col_name == 'sma_5':
                calculated_ta_features[col_name] = SMAIndicator(features_df['close'], window=5).sma_indicator()
            elif col_name == 'sma_10':
                calculated_ta_features[col_name] = SMAIndicator(features_df['close'], window=10).sma_indicator()
            elif col_name == 'sma_20':
                calculated_ta_features[col_name] = SMAIndicator(features_df['close'], window=20).sma_indicator()
            elif col_name == 'sma_50':
                calculated_ta_features[col_name] = SMAIndicator(features_df['close'], window=50).sma_indicator()
            elif col_name == 'ema_12':
                calculated_ta_features[col_name] = EMAIndicator(features_df['close'], window=12).ema_indicator()
            elif col_name == 'ema_26':
                calculated_ta_features[col_name] = EMAIndicator(features_df['close'], window=26).ema_indicator()
            
            # MACD
            elif col_name == 'macd': # MACD Line
                calculated_ta_features[col_name] = MACD(features_df['close']).macd()
            elif col_name == 'macd_signal':
                calculated_ta_features[col_name] = MACD(features_df['close']).macd_signal()
            elif col_name == 'macd_histogram': # MACD Difference/Histogram
                calculated_ta_features[col_name] = MACD(features_df['close']).macd_diff()
            
            # RSI
            elif col_name == 'rsi':
                calculated_ta_features[col_name] = RSIIndicator(features_df['close'], window=14).rsi()
            
            # Bollinger Bands
            elif col_name.startswith('bb_'):
                bb_window = 20 # from pc/features.py
                bb_std = 2    # from pc/features.py
                bb_indicator = BollingerBands(features_df['close'], window=bb_window, window_dev=bb_std)
                if 'bb_upper' not in calculated_ta_features.columns : calculated_ta_features['bb_upper'] = bb_indicator.bollinger_hband()
                if 'bb_middle' not in calculated_ta_features.columns : calculated_ta_features['bb_middle'] = bb_indicator.bollinger_mavg()
                if 'bb_lower' not in calculated_ta_features.columns : calculated_ta_features['bb_lower'] = bb_indicator.bollinger_lband()
                if col_name == 'bb_width': # This needs bb_upper, bb_lower, bb_middle
                     calculated_ta_features[col_name] = (calculated_ta_features['bb_upper'] - calculated_ta_features['bb_lower']) / calculated_ta_features['bb_middle']
                elif col_name == 'bb_position': # This needs close, bb_lower, bb_upper
                     calculated_ta_features[col_name] = (features_df['close'] - calculated_ta_features['bb_lower']) / (calculated_ta_features['bb_upper'] - calculated_ta_features['bb_lower'])
                # Ensure the specific requested bb_ column is present
                if col_name not in calculated_ta_features and col_name in ['bb_upper', 'bb_middle', 'bb_lower']:
                     # This case should not be hit if logic is correct, but as a safeguard:
                     logger.warning(f"BBand component {col_name} was not pre-calculated as expected.")
                     # calculated_ta_features[col_name] will be filled by the one of the if conditions above.

            # Stochastic Oscillator
            elif col_name == 'stoch_k':
                calculated_ta_features[col_name] = StochOscillator(features_df['high'], features_df['low'], features_df['close']).stoch()
            elif col_name == 'stoch_d':
                calculated_ta_features[col_name] = StochOscillator(features_df['high'], features_df['low'], features_df['close']).stoch_signal()
            
            # Williams %R
            elif col_name == 'williams_r':
                calculated_ta_features[col_name] = WilliamsRIndicator(features_df['high'], features_df['low'], features_df['close']).williams_r()
            
            # ATR
            elif col_name == 'atr':
                calculated_ta_features[col_name] = average_true_range(features_df['high'], features_df['low'], features_df['close'])

            # CCI
            elif col_name == 'cci':
                calculated_ta_features[col_name] = CCIIndicator(features_df['high'], features_df['low'], features_df['close'], window=20).cci()
            
            # MFI
            elif col_name == 'mfi':
                calculated_ta_features[col_name] = MFIIndicator(features_df['high'], features_df['low'], features_df['close'], features_df['volume'], window=14).money_flow_index()

            # ROC
            elif col_name == 'roc':
                calculated_ta_features[col_name] = ROCIndicator(features_df['close'], window=12).roc()

            # Volatility (price_change dependent)
            elif col_name == 'volatility_5':
                calculated_ta_features[col_name] = features_df['price_change'].rolling(5).std()
            elif col_name == 'volatility_10':
                calculated_ta_features[col_name] = features_df['price_change'].rolling(10).std()
            elif col_name == 'volatility_20':
                calculated_ta_features[col_name] = features_df['price_change'].rolling(20).std()

            # Price Position
            elif col_name == 'price_position_5':
                min_5 = features_df['close'].rolling(5).min()
                max_5 = features_df['close'].rolling(5).max()
                calculated_ta_features[col_name] = (features_df['close'] - min_5) / (max_5 - min_5)
            elif col_name == 'price_position_20':
                min_20 = features_df['close'].rolling(20).min()
                max_20 = features_df['close'].rolling(20).max()
                calculated_ta_features[col_name] = (features_df['close'] - min_20) / (max_20 - min_20)

            # Momentum
            elif col_name == 'momentum_5':
                calculated_ta_features[col_name] = features_df['close'] / features_df['close'].shift(5) - 1
            elif col_name == 'momentum_10':
                calculated_ta_features[col_name] = features_df['close'] / features_df['close'].shift(10) - 1
            elif col_name == 'momentum_20':
                calculated_ta_features[col_name] = features_df['close'] / features_df['close'].shift(20) - 1

            # Sentiment Features (Placeholder)
            elif col_name.startswith('sentiment_'):
                logger.warning(f"Feature '{col_name}' is sentiment-related and cannot be calculated in this script. Using placeholder 0.0.")
                calculated_ta_features[col_name] = 0.0
            
            # If a feature from LOADED_FEATURE_COLUMNS is not handled above
            elif col_name not in calculated_ta_features.columns:
                logger.warning(f"Unknown or unhandled feature '{col_name}' in LOADED_FEATURE_COLUMNS. Using placeholder 0.0.")
                calculated_ta_features[col_name] = 0.0
        
        except Exception as e:
            logger.error(f"Error calculating feature '{col_name}': {e}. Using placeholder 0.0.")
            calculated_ta_features[col_name] = 0.0


    # Ensure all expected columns exist, even if calculation failed and they became 0.0
    for col in expected_feature_columns:
        if col not in calculated_ta_features.columns:
            logger.error(f"Feature '{col}' was expected but not calculated. Adding placeholder 0.0.")
            calculated_ta_features[col] = 0.0
            
    # Select only the features required by the model, in the correct order
    final_features_df = calculated_ta_features[expected_feature_columns].copy()
    
    # Fill NaNs: ffill and bfill for internal NaNs, then 0 for any remaining (e.g., at the very start)
    final_features_df = final_features_df.fillna(method='ffill').fillna(method='bfill').fillna(0.0)
    
    # Replace any infinities with 0 as well (e.g. from division by zero if a rolling window had all same values for min/max)
    final_features_df.replace([np.inf, -np.inf], 0.0, inplace=True)

    if scaler:
        try:
            scaled_features_np = scaler.transform(final_features_df)
            logger.info("Features successfully scaled.")
        except Exception as e:
            logger.error(f"Error during feature scaling: {e}. Using unscaled features.")
            scaled_features_np = final_features_df.values
    else:
        logger.warning("Feature scaler is not loaded. Using unscaled features.")
        scaled_features_np = final_features_df.values
        
    return scaled_features_np.astype(np.float32)

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
    conn = None
    try:
        conn = sqlite3.connect(str(DB_PATH)) # Use str(DB_PATH)
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
        
        logger.info(f"Stored prediction for {symbol} at {datetime.fromtimestamp(timestamp/1000)}")
        logger.info(f"  5min: {prediction['prediction_5min']:.4f}, "
                   f"10min: {prediction['prediction_10min']:.4f}, "
                   f"30min: {prediction['prediction_30min']:.4f}")
        logger.info(f"  Confidence - Up: {prediction['confidence_up']:.3f}, "
                   f"Down: {prediction['confidence_down']:.3f}")
        
    except sqlite3.Error as e:
        logger.error(f"Error storing prediction for {symbol}: {e}")
    finally:
        if conn:
            conn.close()

def main():
    """Main inference function"""
    global shutdown_requested
    logger.info("Starting inference service")

    try:
        # Initialize predictions table
        try:
            init_predictions_table()
        except Exception as e:
            logger.error(f"Failed to initialize predictions table: {e}. Exiting.")
            return

        # Load ONNX model
        session = load_onnx_model()
        if session is None:
            logger.error("Failed to load model, exiting")
            return
        
        if shutdown_requested:
            logger.info("Shutdown requested before inference loop.")
            return

        # Get current timestamp (once for all symbols in this run)
        current_timestamp = int(datetime.now().timestamp() * 1000)
        
        # Make predictions for each symbol
        for symbol in SYMBOLS:
            if shutdown_requested:
                logger.info("Shutdown requested during symbol processing loop.")
                break
            logger.info(f"Making prediction for {symbol}")
            
            # Get latest features
            features = get_latest_features(symbol, SEQUENCE_LENGTH)
            if features is None:
                logger.warning(f"Skipping {symbol} due to insufficient data")
                continue
            
            if shutdown_requested: # Check again after potentially long feature calculation
                logger.info("Shutdown requested after feature calculation.")
                break

            # Make prediction
            prediction = make_prediction(session, features)
            if prediction is None:
                logger.warning(f"Failed to make prediction for {symbol}")
                continue
            
            if shutdown_requested: # Check again after potentially long prediction
                logger.info("Shutdown requested after model prediction.")
                break

            # Store prediction
            store_prediction(symbol, current_timestamp, prediction)
        
        if not shutdown_requested:
            logger.info("Inference completed for all symbols.")

    except KeyboardInterrupt:
        logger.info("KeyboardInterrupt received. Requesting shutdown...")
        shutdown_requested = True
    except Exception as e:
        logger.error(f"An unexpected error occurred in main: {e}", exc_info=True)
    finally:
        if shutdown_requested:
            logger.info("Inference service shutdown process initiated.")
        else:
            logger.info("Inference service run completed normally.")

if __name__ == "__main__":
    # Register signal handlers
    signal.signal(signal.SIGINT, handle_shutdown_signal)
    signal.signal(signal.SIGTERM, handle_shutdown_signal)
    
    main() 