#!/usr/bin/env python3
"""
Daily ETL export script for Raspberry Pi
Exports last 24 hours of crypto data to CSV files
"""

import sqlite3
import pandas as pd
import logging
from datetime import datetime, timedelta
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration
DB_PATH = Path(__file__).parent.parent / "data" / "db" / "crypto_data.db"
DUMPS_PATH = Path(__file__).parent.parent / "data" / "dumps"
SYMBOLS = ["BTCUSD", "ETHUSD"]  # Updated to USD pairs

def export_daily_data():
    """Export last 24 hours of data to CSV files"""
    try:
        # Ensure dumps directory exists
        DUMPS_PATH.mkdir(parents=True, exist_ok=True)
        
        # Calculate time range (last 24 hours)
        end_time = datetime.now()
        start_time = end_time - timedelta(days=1)
        
        # Convert to millisecond timestamps
        start_timestamp = int(start_time.timestamp() * 1000)
        end_timestamp = int(end_time.timestamp() * 1000)
        
        logger.info(f"Exporting data from {start_time} to {end_time}")
        
        # Connect to database
        conn = sqlite3.connect(DB_PATH)
        
        # Export data for each symbol
        for symbol in SYMBOLS:
            query = """
                SELECT timestamp, symbol, open, high, low, close, volume, created_at
                FROM ohlcv 
                WHERE symbol = ? AND timestamp >= ? AND timestamp <= ?
                ORDER BY timestamp ASC
            """
            
            df = pd.read_sql_query(query, conn, params=(symbol, start_timestamp, end_timestamp))
            
            if df.empty:
                logger.warning(f"No data found for {symbol} in the last 24 hours")
                continue
            
            # Convert timestamp to readable datetime
            df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
            
            # Generate filename with date
            date_str = end_time.strftime('%Y-%m-%d')
            filename = f"{symbol}_{date_str}.csv"
            filepath = DUMPS_PATH / filename
            
            # Export to CSV
            df.to_csv(filepath, index=False)
            logger.info(f"Exported {len(df)} records for {symbol} to {filename}")
        
        # Export combined data
        combined_query = """
            SELECT timestamp, symbol, open, high, low, close, volume, created_at
            FROM ohlcv 
            WHERE timestamp >= ? AND timestamp <= ?
            ORDER BY timestamp ASC, symbol ASC
        """
        
        combined_df = pd.read_sql_query(combined_query, conn, params=(start_timestamp, end_timestamp))
        
        if not combined_df.empty:
            combined_df['datetime'] = pd.to_datetime(combined_df['timestamp'], unit='ms')
            combined_filename = f"combined_{date_str}.csv"
            combined_filepath = DUMPS_PATH / combined_filename
            combined_df.to_csv(combined_filepath, index=False)
            logger.info(f"Exported {len(combined_df)} combined records to {combined_filename}")
        
        conn.close()
        
        # Optional: Clean up old dumps (keep last 7 days)
        cleanup_old_dumps()
        
        logger.info("ETL export completed successfully")
        
    except Exception as e:
        logger.error(f"Error during ETL export: {e}")
        raise

def cleanup_old_dumps(days_to_keep=7):
    """Remove CSV dumps older than specified days"""
    try:
        cutoff_date = datetime.now() - timedelta(days=days_to_keep)
        
        for csv_file in DUMPS_PATH.glob("*.csv"):
            # Extract date from filename
            try:
                # Assuming format: SYMBOL_YYYY-MM-DD.csv or combined_YYYY-MM-DD.csv
                date_part = csv_file.stem.split('_')[-1]
                file_date = datetime.strptime(date_part, '%Y-%m-%d')
                
                if file_date < cutoff_date:
                    csv_file.unlink()
                    logger.info(f"Removed old dump: {csv_file.name}")
                    
            except (ValueError, IndexError):
                # Skip files that don't match expected format
                continue
                
    except Exception as e:
        logger.warning(f"Error during cleanup: {e}")

def get_data_summary():
    """Get summary statistics of the database"""
    try:
        conn = sqlite3.connect(DB_PATH)
        
        # Get record counts by symbol
        summary_query = """
            SELECT 
                symbol,
                COUNT(*) as record_count,
                MIN(datetime(timestamp/1000, 'unixepoch')) as earliest_record,
                MAX(datetime(timestamp/1000, 'unixepoch')) as latest_record
            FROM ohlcv 
            GROUP BY symbol
        """
        
        summary_df = pd.read_sql_query(summary_query, conn)
        logger.info("Database summary:")
        for _, row in summary_df.iterrows():
            logger.info(f"  {row['symbol']}: {row['record_count']} records "
                       f"({row['earliest_record']} to {row['latest_record']})")
        
        conn.close()
        
    except Exception as e:
        logger.warning(f"Error getting data summary: {e}")

def main():
    """Main ETL export function"""
    logger.info("Starting daily ETL export")
    
    # Check if database exists
    if not DB_PATH.exists():
        logger.error(f"Database not found at {DB_PATH}")
        return
    
    # Get data summary
    get_data_summary()
    
    # Export data
    export_daily_data()
    
    logger.info("ETL export process completed")

if __name__ == "__main__":
    main() 