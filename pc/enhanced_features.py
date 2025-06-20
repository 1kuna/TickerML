#!/usr/bin/env python3
"""
Enhanced Feature Engineering with Microstructure Integration
Combines traditional technical indicators with advanced market microstructure features
"""

import pandas as pd
import numpy as np
import ta
import sqlite3
import logging
from datetime import datetime, timedelta
from pathlib import Path
import yaml
import os
from typing import Dict, List, Optional, Tuple
import json

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration
DATA_PATH = Path(__file__).parent.parent / "data" / "dumps"
OUTPUT_PATH = Path(__file__).parent.parent / "data" / "features"
CONFIG_PATH = Path(__file__).parent.parent / "config" / "config.yaml"
DB_PATH = Path(__file__).parent.parent / "data" / "db" / "crypto_data.db"

# Ensure output directory exists
OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

class MicrostructureFeatureEngine:
    """Advanced feature engineering with market microstructure integration"""
    
    def __init__(self, config_path: str = None):
        """Initialize the feature engine"""
        self.config = self._load_config(config_path or CONFIG_PATH)
        self.db_path = self.config.get('database', {}).get('path', str(DB_PATH))
        
    def _load_config(self, config_path: Path) -> Dict:
        """Load configuration from config.yaml"""
        try:
            if config_path.exists():
                with open(config_path, 'r') as f:
                    return yaml.safe_load(f)
            else:
                logger.warning("Config file not found, using defaults")
                return {}
        except Exception as e:
            logger.error(f"Error loading config: {e}")
            return {}
    
    def load_order_book_data(self, symbol: str, hours: int = 24) -> Optional[pd.DataFrame]:
        """Load order book snapshots from database"""
        try:
            conn = sqlite3.connect(self.db_path)
            
            # Calculate time range
            end_time = datetime.now()
            start_time = end_time - timedelta(hours=hours)
            start_timestamp = start_time.timestamp()
            
            query = '''
                SELECT timestamp, symbol, bids, asks, mid_price, spread_bps,
                       imbalance, microprice, depth_5, depth_10
                FROM order_books 
                WHERE symbol = ? AND timestamp >= ?
                ORDER BY timestamp ASC
            '''
            
            df = pd.read_sql_query(query, conn, params=(symbol, start_timestamp))
            conn.close()
            
            if len(df) == 0:
                logger.warning(f"No order book data found for {symbol}")
                return None
                
            # Convert timestamp to datetime
            df['datetime'] = pd.to_datetime(df['timestamp'], unit='s')
            
            logger.info(f"Loaded {len(df)} order book snapshots for {symbol}")
            return df
            
        except Exception as e:
            logger.error(f"Error loading order book data for {symbol}: {e}")
            return None
    
    def load_trade_data(self, symbol: str, hours: int = 24) -> Optional[pd.DataFrame]:
        """Load trade data from database"""
        try:
            conn = sqlite3.connect(self.db_path)
            
            # Calculate time range
            end_time = datetime.now()
            start_time = end_time - timedelta(hours=hours)
            start_timestamp = start_time.timestamp()
            
            query = '''
                SELECT timestamp, symbol, price, quantity, side, trade_id
                FROM trades 
                WHERE symbol = ? AND timestamp >= ?
                ORDER BY timestamp ASC
            '''
            
            df = pd.read_sql_query(query, conn, params=(symbol, start_timestamp))
            conn.close()
            
            if len(df) == 0:
                logger.warning(f"No trade data found for {symbol}")
                return None
                
            # Convert timestamp to datetime
            df['datetime'] = pd.to_datetime(df['timestamp'], unit='s')
            
            logger.info(f"Loaded {len(df)} trades for {symbol}")
            return df
            
        except Exception as e:
            logger.error(f"Error loading trade data for {symbol}: {e}")
            return None
    
    def compute_microstructure_features(self, order_book_df: pd.DataFrame, 
                                      trade_df: pd.DataFrame) -> pd.DataFrame:
        """Compute advanced microstructure features"""
        try:
            features = []
            
            # Resample order book data to 1-minute intervals
            ob_resampled = order_book_df.set_index('datetime').resample('1min').agg({
                'mid_price': 'last',
                'spread_bps': 'mean',
                'imbalance': 'mean',
                'microprice': 'last',
                'depth_5': 'mean',
                'depth_10': 'mean'
            }).dropna()
            
            # Resample trade data to 1-minute intervals
            if len(trade_df) > 0:
                trade_df['buy_volume'] = trade_df.apply(
                    lambda x: x['quantity'] if x['side'] == 'buy' else 0, axis=1
                )
                trade_df['sell_volume'] = trade_df.apply(
                    lambda x: x['quantity'] if x['side'] == 'sell' else 0, axis=1
                )
                
                trade_resampled = trade_df.set_index('datetime').resample('1min').agg({
                    'price': ['first', 'last', 'min', 'max'],
                    'quantity': 'sum',
                    'buy_volume': 'sum',
                    'sell_volume': 'sum'
                }).dropna()
                
                # Flatten column names
                trade_resampled.columns = ['_'.join(col).strip() for col in trade_resampled.columns]
                
                # Calculate trade flow imbalance
                trade_resampled['trade_flow_imbalance'] = (
                    (trade_resampled['buy_volume_sum'] - trade_resampled['sell_volume_sum']) /
                    (trade_resampled['buy_volume_sum'] + trade_resampled['sell_volume_sum'] + 1e-10)
                )
                
                # Calculate VWAP
                trade_resampled['vwap'] = (
                    trade_df.set_index('datetime').resample('1min').apply(
                        lambda x: (x['price'] * x['quantity']).sum() / x['quantity'].sum()
                        if x['quantity'].sum() > 0 else np.nan
                    )
                )
            else:
                logger.warning("No trade data available for microstructure features")
                return None
            
            # Merge order book and trade features
            microstructure_df = pd.merge(
                ob_resampled, trade_resampled, 
                left_index=True, right_index=True, how='inner'
            )
            
            if len(microstructure_df) == 0:
                logger.warning("No overlapping order book and trade data")
                return None
            
            # Calculate additional microstructure features
            microstructure_df['microprice_mid_diff'] = (
                microstructure_df['microprice'] - microstructure_df['mid_price']
            )
            
            microstructure_df['vwap_mid_diff'] = (
                microstructure_df['vwap'] - microstructure_df['mid_price']
            )
            
            # Rolling features
            for window in [5, 10, 15]:
                microstructure_df[f'imbalance_ma_{window}'] = (
                    microstructure_df['imbalance'].rolling(window).mean()
                )
                microstructure_df[f'spread_ma_{window}'] = (
                    microstructure_df['spread_bps'].rolling(window).mean()
                )
                microstructure_df[f'trade_flow_ma_{window}'] = (
                    microstructure_df['trade_flow_imbalance'].rolling(window).mean()
                )
            
            logger.info(f"Computed microstructure features: {microstructure_df.shape}")
            return microstructure_df
            
        except Exception as e:
            logger.error(f"Error computing microstructure features: {e}")
            return None
    
    def compute_traditional_features(self, price_df: pd.DataFrame) -> pd.DataFrame:
        """Compute traditional technical indicators"""
        try:
            if len(price_df) < 20:  # Need minimum data for technical indicators
                logger.warning(f"Insufficient data for technical indicators: {len(price_df)} rows")
                return price_df
            
            df = price_df.copy()
            
            # Price-based indicators
            df['sma_5'] = ta.trend.sma_indicator(df['price_last'], window=5)
            df['sma_10'] = ta.trend.sma_indicator(df['price_last'], window=10)
            df['sma_20'] = ta.trend.sma_indicator(df['price_last'], window=20)
            
            df['ema_5'] = ta.trend.ema_indicator(df['price_last'], window=5)
            df['ema_10'] = ta.trend.ema_indicator(df['price_last'], window=10)
            df['ema_20'] = ta.trend.ema_indicator(df['price_last'], window=20)
            
            # RSI
            df['rsi'] = ta.momentum.rsi(df['price_last'], window=14)
            
            # MACD
            df['macd'] = ta.trend.macd_diff(df['price_last'])
            df['macd_signal'] = ta.trend.macd_signal(df['price_last'])
            
            # Bollinger Bands
            bb = ta.volatility.BollingerBands(df['price_last'])
            df['bb_upper'] = bb.bollinger_hband()
            df['bb_lower'] = bb.bollinger_lband()
            df['bb_width'] = bb.bollinger_wband()
            
            # Volume indicators
            if 'quantity_sum' in df.columns:
                df['volume_sma'] = ta.volume.sma_ease_of_movement(
                    df['price_max'], df['price_min'], df['quantity_sum']
                )
            
            # Momentum indicators
            df['price_change'] = df['price_last'].pct_change()
            df['price_change_5'] = df['price_last'].pct_change(periods=5)
            df['price_change_10'] = df['price_last'].pct_change(periods=10)
            
            # Volatility
            df['volatility_5'] = df['price_change'].rolling(5).std()
            df['volatility_10'] = df['price_change'].rolling(10).std()
            
            logger.info(f"Computed traditional technical indicators")
            return df
            
        except Exception as e:
            logger.error(f"Error computing traditional features: {e}")
            return price_df
    
    def create_target_variables(self, df: pd.DataFrame, price_col: str = 'mid_price') -> pd.DataFrame:
        """Create target variables for prediction"""
        try:
            # Future price changes
            df['target_5min'] = df[price_col].shift(-5).pct_change()
            df['target_10min'] = df[price_col].shift(-10).pct_change()
            df['target_30min'] = df[price_col].shift(-30).pct_change()
            
            # Direction classification
            df['direction_5min'] = (df['target_5min'] > 0).astype(int)
            df['direction_10min'] = (df['target_10min'] > 0).astype(int)
            df['direction_30min'] = (df['target_30min'] > 0).astype(int)
            
            logger.info("Created target variables")
            return df
            
        except Exception as e:
            logger.error(f"Error creating target variables: {e}")
            return df
    
    def engineer_features(self, symbol: str, hours: int = 24) -> Optional[pd.DataFrame]:
        """Main feature engineering pipeline"""
        try:
            logger.info(f"Starting enhanced feature engineering for {symbol}")
            
            # Load data
            order_book_df = self.load_order_book_data(symbol, hours)
            trade_df = self.load_trade_data(symbol, hours)
            
            if order_book_df is None:
                logger.error(f"No order book data available for {symbol}")
                return None
            
            if trade_df is None:
                logger.warning(f"No trade data available for {symbol}")
                # Use order book data only
                microstructure_df = order_book_df.set_index('datetime').resample('1min').agg({
                    'mid_price': 'last',
                    'spread_bps': 'mean',
                    'imbalance': 'mean',
                    'microprice': 'last'
                }).dropna()
                
                # Add basic price columns for traditional indicators
                microstructure_df['price_last'] = microstructure_df['mid_price']
                microstructure_df['price_first'] = microstructure_df['mid_price']
                microstructure_df['price_min'] = microstructure_df['mid_price']
                microstructure_df['price_max'] = microstructure_df['mid_price']
            else:
                # Compute full microstructure features
                microstructure_df = self.compute_microstructure_features(order_book_df, trade_df)
                
            if microstructure_df is None:
                logger.error(f"Failed to compute microstructure features for {symbol}")
                return None
            
            # Compute traditional technical indicators
            enhanced_df = self.compute_traditional_features(microstructure_df)
            
            # Create target variables
            final_df = self.create_target_variables(enhanced_df)
            
            # Reset index to get datetime as column
            final_df = final_df.reset_index()
            
            # Clean data
            final_df = final_df.replace([np.inf, -np.inf], np.nan)
            
            logger.info(f"Enhanced feature engineering completed for {symbol}")
            logger.info(f"Final shape: {final_df.shape}")
            logger.info(f"Missing values: {final_df.isnull().sum().sum()}")
            
            return final_df
            
        except Exception as e:
            logger.error(f"Error in feature engineering for {symbol}: {e}")
            return None
    
    def save_features(self, df: pd.DataFrame, symbol: str, format: str = 'both') -> bool:
        """Save features to file"""
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            base_filename = f"{symbol}_enhanced_features_{timestamp}"
            
            if format in ['csv', 'both']:
                csv_path = OUTPUT_PATH / f"{base_filename}.csv"
                df.to_csv(csv_path, index=False)
                logger.info(f"Saved CSV features to {csv_path}")
            
            if format in ['pkl', 'both']:
                pkl_path = OUTPUT_PATH / f"{base_filename}.pkl"
                df.to_pickle(pkl_path)
                logger.info(f"Saved pickle features to {pkl_path}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error saving features: {e}")
            return False

def main():
    """Main execution function"""
    logger.info("Starting enhanced feature engineering pipeline")
    
    # Initialize feature engine
    engine = MicrostructureFeatureEngine()
    
    # Process symbols
    symbols = ['BTCUSD', 'ETHUSD']  # Use USD pairs for Binance.US
    
    for symbol in symbols:
        logger.info(f"Processing {symbol}...")
        
        # Engineer features
        features_df = engine.engineer_features(symbol, hours=24)
        
        if features_df is not None:
            # Save features
            engine.save_features(features_df, symbol)
            
            # Print summary
            logger.info(f"Feature summary for {symbol}:")
            logger.info(f"  Shape: {features_df.shape}")
            logger.info(f"  Date range: {features_df['datetime'].min()} to {features_df['datetime'].max()}")
            logger.info(f"  Missing values: {features_df.isnull().sum().sum()}")
        else:
            logger.error(f"Failed to engineer features for {symbol}")
    
    logger.info("Enhanced feature engineering pipeline completed")

if __name__ == "__main__":
    main()