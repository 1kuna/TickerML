#!/usr/bin/env python3
"""
Data Validation Layer for TickerML.
Provides gap detection, quality checks, and data integrity validation.
"""

import logging
import sqlite3
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/data_validator.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class DataValidator:
    """Validates data quality and detects gaps in market data."""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """Initialize the data validator."""
        self.config = self._load_config(config_path)
        self.db_path = self.config.get('database', {}).get('path', 'data/db/crypto_data.db')
        self.symbols = self.config.get('binance', {}).get('symbols', ['BTCUSDT', 'ETHUSDT'])
        
        # Validation thresholds
        self.max_gap_seconds = 300  # 5 minutes
        self.min_price_change_pct = 0.0001  # 0.01%
        self.max_price_change_pct = 0.50  # 50%
        self.min_volume = 0.0001
        self.max_spread_pct = 0.10  # 10%
        
        # Statistical thresholds
        self.outlier_z_score = 3.0
        self.min_data_points = 10
        
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from YAML file."""
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            logger.warning(f"Config file {config_path} not found, using defaults")
            return {
                'binance': {
                    'symbols': ['BTCUSDT', 'ETHUSDT']
                },
                'database': {
                    'path': 'data/db/crypto_data.db'
                }
            }
    
    def validate_ohlcv_data(self, symbol: str, hours: int = 24) -> Dict:
        """Validate OHLCV data for a symbol."""
        try:
            conn = sqlite3.connect(self.db_path)
            
            # Get recent OHLCV data
            time_threshold = time.time() - (hours * 3600)
            query = '''
                SELECT timestamp, open, high, low, close, volume
                FROM ohlcv
                WHERE symbol = ? AND timestamp >= ?
                ORDER BY timestamp ASC
            '''
            
            df = pd.read_sql_query(query, conn, params=(symbol, time_threshold))
            conn.close()
            
            if df.empty:
                return {
                    'symbol': symbol,
                    'valid': False,
                    'error': 'No data found',
                    'data_points': 0
                }
            
            # Initialize validation results
            validation_results = {
                'symbol': symbol,
                'valid': True,
                'data_points': len(df),
                'time_range': {
                    'start': datetime.fromtimestamp(df['timestamp'].min()).isoformat(),
                    'end': datetime.fromtimestamp(df['timestamp'].max()).isoformat()
                },
                'issues': []
            }
            
            # Check for time gaps
            time_gaps = self._check_time_gaps(df, validation_results)
            
            # Check for price anomalies
            price_anomalies = self._check_price_anomalies(df, validation_results)
            
            # Check for volume anomalies
            volume_anomalies = self._check_volume_anomalies(df, validation_results)
            
            # Check OHLC consistency
            ohlc_consistency = self._check_ohlc_consistency(df, validation_results)
            
            # Check for duplicate timestamps
            duplicate_timestamps = self._check_duplicate_timestamps(df, validation_results)
            
            # Overall validation status
            validation_results['valid'] = len(validation_results['issues']) == 0
            validation_results['gap_count'] = time_gaps
            validation_results['price_anomaly_count'] = price_anomalies
            validation_results['volume_anomaly_count'] = volume_anomalies
            validation_results['ohlc_error_count'] = ohlc_consistency
            validation_results['duplicate_count'] = duplicate_timestamps
            
            return validation_results
            
        except Exception as e:
            logger.error(f"Failed to validate OHLCV data for {symbol}: {e}")
            return {
                'symbol': symbol,
                'valid': False,
                'error': str(e),
                'data_points': 0
            }
    
    def _check_time_gaps(self, df: pd.DataFrame, results: Dict) -> int:
        """Check for time gaps in the data."""
        if len(df) < 2:
            return 0
        
        # Calculate time differences
        time_diffs = np.diff(df['timestamp'].values)
        
        # Find gaps larger than threshold
        gaps = time_diffs > self.max_gap_seconds
        gap_count = np.sum(gaps)
        
        if gap_count > 0:
            gap_indices = np.where(gaps)[0]
            for idx in gap_indices:
                gap_size = time_diffs[idx]
                results['issues'].append({
                    'type': 'time_gap',
                    'severity': 'high' if gap_size > 3600 else 'medium',
                    'description': f"Data gap of {gap_size:.1f} seconds",
                    'timestamp': df.iloc[idx + 1]['timestamp'],
                    'gap_seconds': gap_size
                })
        
        return gap_count
    
    def _check_price_anomalies(self, df: pd.DataFrame, results: Dict) -> int:
        """Check for price anomalies."""
        if len(df) < self.min_data_points:
            return 0
        
        anomaly_count = 0
        
        # Check for negative or zero prices
        for col in ['open', 'high', 'low', 'close']:
            negative_prices = df[col] <= 0
            if negative_prices.any():
                anomaly_count += negative_prices.sum()
                results['issues'].append({
                    'type': 'invalid_price',
                    'severity': 'critical',
                    'description': f"Negative or zero {col} prices found",
                    'count': negative_prices.sum()
                })
        
        # Check for extreme price changes
        for col in ['open', 'high', 'low', 'close']:
            price_changes = df[col].pct_change().abs()
            extreme_changes = price_changes > self.max_price_change_pct
            
            if extreme_changes.any():
                anomaly_count += extreme_changes.sum()
                max_change = price_changes.max()
                results['issues'].append({
                    'type': 'extreme_price_change',
                    'severity': 'high',
                    'description': f"Extreme {col} price change: {max_change:.2%}",
                    'count': extreme_changes.sum(),
                    'max_change_pct': max_change
                })
        
        # Check for price outliers using Z-score
        for col in ['open', 'high', 'low', 'close']:
            if len(df) >= self.min_data_points:
                z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
                outliers = z_scores > self.outlier_z_score
                
                if outliers.any():
                    anomaly_count += outliers.sum()
                    results['issues'].append({
                        'type': 'price_outlier',
                        'severity': 'medium',
                        'description': f"Price outliers detected in {col}",
                        'count': outliers.sum(),
                        'max_z_score': z_scores.max()
                    })
        
        return anomaly_count
    
    def _check_volume_anomalies(self, df: pd.DataFrame, results: Dict) -> int:
        """Check for volume anomalies."""
        if len(df) < self.min_data_points:
            return 0
        
        anomaly_count = 0
        
        # Check for negative or zero volume
        invalid_volume = df['volume'] <= self.min_volume
        if invalid_volume.any():
            anomaly_count += invalid_volume.sum()
            results['issues'].append({
                'type': 'invalid_volume',
                'severity': 'high',
                'description': "Invalid volume values found",
                'count': invalid_volume.sum()
            })
        
        # Check for volume outliers
        if len(df) >= self.min_data_points:
            # Use log transformation for volume outlier detection
            log_volume = np.log(df['volume'] + 1e-8)
            z_scores = np.abs((log_volume - log_volume.mean()) / log_volume.std())
            outliers = z_scores > self.outlier_z_score
            
            if outliers.any():
                anomaly_count += outliers.sum()
                results['issues'].append({
                    'type': 'volume_outlier',
                    'severity': 'medium',
                    'description': "Volume outliers detected",
                    'count': outliers.sum(),
                    'max_z_score': z_scores.max()
                })
        
        return anomaly_count
    
    def _check_ohlc_consistency(self, df: pd.DataFrame, results: Dict) -> int:
        """Check OHLC price consistency."""
        error_count = 0
        
        # Check that high >= low
        high_low_errors = df['high'] < df['low']
        if high_low_errors.any():
            error_count += high_low_errors.sum()
            results['issues'].append({
                'type': 'ohlc_consistency',
                'severity': 'critical',
                'description': "High price less than low price",
                'count': high_low_errors.sum()
            })
        
        # Check that high >= open and high >= close
        high_open_errors = df['high'] < df['open']
        high_close_errors = df['high'] < df['close']
        
        if high_open_errors.any():
            error_count += high_open_errors.sum()
            results['issues'].append({
                'type': 'ohlc_consistency',
                'severity': 'critical',
                'description': "High price less than open price",
                'count': high_open_errors.sum()
            })
        
        if high_close_errors.any():
            error_count += high_close_errors.sum()
            results['issues'].append({
                'type': 'ohlc_consistency',
                'severity': 'critical',
                'description': "High price less than close price",
                'count': high_close_errors.sum()
            })
        
        # Check that low <= open and low <= close
        low_open_errors = df['low'] > df['open']
        low_close_errors = df['low'] > df['close']
        
        if low_open_errors.any():
            error_count += low_open_errors.sum()
            results['issues'].append({
                'type': 'ohlc_consistency',
                'severity': 'critical',
                'description': "Low price greater than open price",
                'count': low_open_errors.sum()
            })
        
        if low_close_errors.any():
            error_count += low_close_errors.sum()
            results['issues'].append({
                'type': 'ohlc_consistency',
                'severity': 'critical',
                'description': "Low price greater than close price",
                'count': low_close_errors.sum()
            })
        
        return error_count
    
    def _check_duplicate_timestamps(self, df: pd.DataFrame, results: Dict) -> int:
        """Check for duplicate timestamps."""
        duplicates = df.duplicated(subset=['timestamp'])
        duplicate_count = duplicates.sum()
        
        if duplicate_count > 0:
            results['issues'].append({
                'type': 'duplicate_timestamp',
                'severity': 'medium',
                'description': "Duplicate timestamps found",
                'count': duplicate_count
            })
        
        return duplicate_count
    
    def validate_order_book_data(self, symbol: str, hours: int = 1) -> Dict:
        """Validate order book data for a symbol."""
        try:
            conn = sqlite3.connect(self.db_path)
            
            # Get recent order book data
            time_threshold = time.time() - (hours * 3600)
            query = '''
                SELECT timestamp, bids, asks, mid_price, spread
                FROM order_books
                WHERE symbol = ? AND timestamp >= ?
                ORDER BY timestamp ASC
            '''
            
            df = pd.read_sql_query(query, conn, params=(symbol, time_threshold))
            conn.close()
            
            if df.empty:
                return {
                    'symbol': symbol,
                    'valid': False,
                    'error': 'No order book data found',
                    'data_points': 0
                }
            
            validation_results = {
                'symbol': symbol,
                'valid': True,
                'data_points': len(df),
                'time_range': {
                    'start': datetime.fromtimestamp(df['timestamp'].min()).isoformat(),
                    'end': datetime.fromtimestamp(df['timestamp'].max()).isoformat()
                },
                'issues': []
            }
            
            # Check for gaps in order book updates
            self._check_time_gaps(df, validation_results)
            
            # Check spread consistency
            self._check_spread_consistency(df, validation_results)
            
            # Check mid-price consistency
            self._check_mid_price_consistency(df, validation_results)
            
            validation_results['valid'] = len(validation_results['issues']) == 0
            
            return validation_results
            
        except Exception as e:
            logger.error(f"Failed to validate order book data for {symbol}: {e}")
            return {
                'symbol': symbol,
                'valid': False,
                'error': str(e),
                'data_points': 0
            }
    
    def _check_spread_consistency(self, df: pd.DataFrame, results: Dict):
        """Check order book spread consistency."""
        if 'spread' not in df.columns:
            return
        
        # Check for negative spreads
        negative_spreads = df['spread'] < 0
        if negative_spreads.any():
            results['issues'].append({
                'type': 'negative_spread',
                'severity': 'critical',
                'description': "Negative spreads detected",
                'count': negative_spreads.sum()
            })
        
        # Check for extremely wide spreads
        if len(df) >= self.min_data_points:
            spread_pct = df['spread'] / df['mid_price']
            wide_spreads = spread_pct > self.max_spread_pct
            
            if wide_spreads.any():
                results['issues'].append({
                    'type': 'wide_spread',
                    'severity': 'high',
                    'description': f"Extremely wide spreads detected (>{self.max_spread_pct:.1%})",
                    'count': wide_spreads.sum(),
                    'max_spread_pct': spread_pct.max()
                })
    
    def _check_mid_price_consistency(self, df: pd.DataFrame, results: Dict):
        """Check mid-price consistency."""
        if 'mid_price' not in df.columns:
            return
        
        # Check for negative or zero mid-prices
        invalid_mid_price = df['mid_price'] <= 0
        if invalid_mid_price.any():
            results['issues'].append({
                'type': 'invalid_mid_price',
                'severity': 'critical',
                'description': "Invalid mid-price values",
                'count': invalid_mid_price.sum()
            })
    
    def validate_trades_data(self, symbol: str, hours: int = 1) -> Dict:
        """Validate trades data for a symbol."""
        try:
            conn = sqlite3.connect(self.db_path)
            
            # Get recent trades data
            time_threshold = time.time() - (hours * 3600)
            query = '''
                SELECT timestamp, trade_id, price, quantity, is_buyer_maker
                FROM trades
                WHERE symbol = ? AND timestamp >= ?
                ORDER BY timestamp ASC
            '''
            
            df = pd.read_sql_query(query, conn, params=(symbol, time_threshold))
            conn.close()
            
            if df.empty:
                return {
                    'symbol': symbol,
                    'valid': False,
                    'error': 'No trades data found',
                    'data_points': 0
                }
            
            validation_results = {
                'symbol': symbol,
                'valid': True,
                'data_points': len(df),
                'time_range': {
                    'start': datetime.fromtimestamp(df['timestamp'].min()).isoformat(),
                    'end': datetime.fromtimestamp(df['timestamp'].max()).isoformat()
                },
                'issues': []
            }
            
            # Check for duplicate trade IDs
            duplicate_trades = df.duplicated(subset=['trade_id'])
            if duplicate_trades.any():
                validation_results['issues'].append({
                    'type': 'duplicate_trade_id',
                    'severity': 'medium',
                    'description': "Duplicate trade IDs found",
                    'count': duplicate_trades.sum()
                })
            
            # Check for invalid prices
            invalid_prices = df['price'] <= 0
            if invalid_prices.any():
                validation_results['issues'].append({
                    'type': 'invalid_trade_price',
                    'severity': 'critical',
                    'description': "Invalid trade prices",
                    'count': invalid_prices.sum()
                })
            
            # Check for invalid quantities
            invalid_quantities = df['quantity'] <= 0
            if invalid_quantities.any():
                validation_results['issues'].append({
                    'type': 'invalid_trade_quantity',
                    'severity': 'critical',
                    'description': "Invalid trade quantities",
                    'count': invalid_quantities.sum()
                })
            
            validation_results['valid'] = len(validation_results['issues']) == 0
            
            return validation_results
            
        except Exception as e:
            logger.error(f"Failed to validate trades data for {symbol}: {e}")
            return {
                'symbol': symbol,
                'valid': False,
                'error': str(e),
                'data_points': 0
            }
    
    def generate_validation_report(self, hours: int = 24) -> Dict:
        """Generate a comprehensive validation report for all symbols."""
        report = {
            'timestamp': datetime.now().isoformat(),
            'validation_period_hours': hours,
            'symbols': {},
            'summary': {
                'total_symbols': len(self.symbols),
                'valid_symbols': 0,
                'invalid_symbols': 0,
                'total_issues': 0
            }
        }
        
        for symbol in self.symbols:
            logger.info(f"Validating data for {symbol}...")
            
            # Validate OHLCV data
            ohlcv_validation = self.validate_ohlcv_data(symbol, hours)
            
            # Validate order book data (if available)
            orderbook_validation = self.validate_order_book_data(symbol, min(hours, 1))
            
            # Validate trades data (if available)
            trades_validation = self.validate_trades_data(symbol, min(hours, 1))
            
            # Combine results
            symbol_report = {
                'ohlcv': ohlcv_validation,
                'orderbook': orderbook_validation,
                'trades': trades_validation,
                'overall_valid': (ohlcv_validation.get('valid', False) and
                                orderbook_validation.get('valid', True) and
                                trades_validation.get('valid', True))
            }
            
            report['symbols'][symbol] = symbol_report
            
            # Update summary
            if symbol_report['overall_valid']:
                report['summary']['valid_symbols'] += 1
            else:
                report['summary']['invalid_symbols'] += 1
            
            # Count issues
            for validation in [ohlcv_validation, orderbook_validation, trades_validation]:
                if 'issues' in validation:
                    report['summary']['total_issues'] += len(validation['issues'])
        
        return report
    
    def get_data_quality_metrics(self, symbol: str, hours: int = 24) -> Dict:
        """Get data quality metrics for a symbol."""
        try:
            conn = sqlite3.connect(self.db_path)
            
            # Get data completeness metrics
            time_threshold = time.time() - (hours * 3600)
            
            # OHLCV completeness
            ohlcv_query = '''
                SELECT COUNT(*) as count,
                       MIN(timestamp) as min_time,
                       MAX(timestamp) as max_time
                FROM ohlcv
                WHERE symbol = ? AND timestamp >= ?
            '''
            
            ohlcv_result = conn.execute(ohlcv_query, (symbol, time_threshold)).fetchone()
            
            # Expected data points (assuming 1-minute intervals)
            expected_points = hours * 60
            actual_points = ohlcv_result[0] if ohlcv_result else 0
            completeness_pct = (actual_points / expected_points) * 100 if expected_points > 0 else 0
            
            conn.close()
            
            return {
                'symbol': symbol,
                'time_period_hours': hours,
                'expected_data_points': expected_points,
                'actual_data_points': actual_points,
                'completeness_percentage': completeness_pct,
                'data_quality_score': min(completeness_pct, 100.0)
            }
            
        except Exception as e:
            logger.error(f"Failed to get data quality metrics for {symbol}: {e}")
            return {
                'symbol': symbol,
                'error': str(e),
                'data_quality_score': 0.0
            }

def main():
    """Main function to run data validation."""
    validator = DataValidator()
    
    # Generate validation report
    report = validator.generate_validation_report(hours=24)
    
    # Print summary
    print(f"\n{'='*60}")
    print("DATA VALIDATION REPORT")
    print(f"{'='*60}")
    print(f"Validation Time: {report['timestamp']}")
    print(f"Period: {report['validation_period_hours']} hours")
    print(f"Total Symbols: {report['summary']['total_symbols']}")
    print(f"Valid Symbols: {report['summary']['valid_symbols']}")
    print(f"Invalid Symbols: {report['summary']['invalid_symbols']}")
    print(f"Total Issues: {report['summary']['total_issues']}")
    
    # Print detailed results for each symbol
    for symbol, symbol_report in report['symbols'].items():
        print(f"\n{symbol}:")
        print(f"  Overall Valid: {symbol_report['overall_valid']}")
        
        for data_type, validation in symbol_report.items():
            if data_type == 'overall_valid':
                continue
            
            if validation.get('valid') is not None:
                print(f"  {data_type.upper()}: {'✓' if validation['valid'] else '✗'} "
                      f"({validation.get('data_points', 0)} points)")
                
                if validation.get('issues'):
                    for issue in validation['issues'][:3]:  # Show first 3 issues
                        print(f"    - {issue['description']} ({issue['severity']})")
                    
                    if len(validation['issues']) > 3:
                        print(f"    ... and {len(validation['issues']) - 3} more issues")

if __name__ == "__main__":
    # Create logs directory if it doesn't exist
    import os
    os.makedirs('logs', exist_ok=True)
    
    main()