#!/usr/bin/env python3
"""
Minute-level crypto data harvester for Raspberry Pi
Fetches BTC/USDT and ETH/USDT data from Binance API every minute
"""

import sqlite3
import requests
import time
import logging
from datetime import datetime
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration
DB_PATH = Path(__file__).parent.parent / "data" / "db" / "crypto_data.db"
BINANCE_API_BASE = "https://api.binance.com/api/v3"
SYMBOLS = ["BTCUSDT", "ETHUSDT"]

def init_database():
    """Initialize SQLite database with OHLCV table"""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS ohlcv (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp INTEGER NOT NULL,
            symbol TEXT NOT NULL,
            open REAL NOT NULL,
            high REAL NOT NULL,
            low REAL NOT NULL,
            close REAL NOT NULL,
            volume REAL NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(timestamp, symbol)
        )
    """)
    
    # Create index for faster queries
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_timestamp_symbol 
        ON ohlcv(timestamp, symbol)
    """)
    
    conn.commit()
    conn.close()
    logger.info("Database initialized")

def fetch_kline_data(symbol):
    """Fetch latest 1-minute kline data from Binance"""
    try:
        url = f"{BINANCE_API_BASE}/klines"
        params = {
            'symbol': symbol,
            'interval': '1m',
            'limit': 1
        }
        
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        if not data:
            logger.warning(f"No data received for {symbol}")
            return None
            
        kline = data[0]
        return {
            'timestamp': int(kline[0]),  # Open time
            'open': float(kline[1]),
            'high': float(kline[2]),
            'low': float(kline[3]),
            'close': float(kline[4]),
            'volume': float(kline[5])
        }
        
    except requests.RequestException as e:
        logger.error(f"Error fetching data for {symbol}: {e}")
        return None
    except (ValueError, IndexError) as e:
        logger.error(f"Error parsing data for {symbol}: {e}")
        return None

def store_data(symbol, kline_data):
    """Store kline data in SQLite database"""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT OR IGNORE INTO ohlcv 
            (timestamp, symbol, open, high, low, close, volume)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            kline_data['timestamp'],
            symbol,
            kline_data['open'],
            kline_data['high'],
            kline_data['low'],
            kline_data['close'],
            kline_data['volume']
        ))
        
        if cursor.rowcount > 0:
            logger.info(f"Stored data for {symbol} at {datetime.fromtimestamp(kline_data['timestamp']/1000)}")
        else:
            logger.debug(f"Data already exists for {symbol} at {datetime.fromtimestamp(kline_data['timestamp']/1000)}")
            
        conn.commit()
        conn.close()
        
    except sqlite3.Error as e:
        logger.error(f"Database error storing {symbol}: {e}")

def main():
    """Main harvesting function"""
    logger.info("Starting crypto data harvest")
    
    # Initialize database
    init_database()
    
    # Fetch data for each symbol
    for symbol in SYMBOLS:
        logger.info(f"Fetching data for {symbol}")
        kline_data = fetch_kline_data(symbol)
        
        if kline_data:
            store_data(symbol, kline_data)
        else:
            logger.warning(f"Failed to fetch data for {symbol}")
    
    logger.info("Harvest completed")

if __name__ == "__main__":
    main() 