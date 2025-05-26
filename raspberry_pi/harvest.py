#!/usr/bin/env python3
"""
Minute-level crypto data harvester for Raspberry Pi
Fetches BTC/USD and ETH/USD data with Binance.US primary, CoinGecko fallback
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

# API Configuration - Primary: Binance.US, Fallback: CoinGecko
BINANCE_US_API_BASE = "https://api.binance.us/api/v3"
COINGECKO_API_BASE = "https://api.coingecko.com/api/v3"

# Symbol mappings for different APIs
BINANCE_SYMBOLS = ["BTCUSD", "ETHUSD"]  # Binance.US format
COINGECKO_SYMBOLS = {
    "BTCUSD": "bitcoin",
    "ETHUSD": "ethereum"
}

# Standardized symbols for database storage
SYMBOLS = ["BTCUSD", "ETHUSD"]

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

def fetch_kline_data_binance_us(symbol):
    """Fetch latest 1-minute kline data from Binance.US"""
    try:
        url = f"{BINANCE_US_API_BASE}/klines"
        params = {
            'symbol': symbol,
            'interval': '1m',
            'limit': 1
        }
        
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        if not data:
            logger.warning(f"No Binance.US data received for {symbol}")
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
        logger.warning(f"Binance.US API error for {symbol}: {e}")
        return None
    except (ValueError, IndexError) as e:
        logger.warning(f"Binance.US data parsing error for {symbol}: {e}")
        return None

def fetch_kline_data_coingecko(symbol):
    """Fetch current price data from CoinGecko as fallback"""
    try:
        if symbol not in COINGECKO_SYMBOLS:
            logger.error(f"Symbol {symbol} not supported by CoinGecko fallback")
            return None
            
        coin_id = COINGECKO_SYMBOLS[symbol]
        url = f"{COINGECKO_API_BASE}/simple/price"
        params = {
            'ids': coin_id,
            'vs_currencies': 'usd',
            'include_24hr_change': 'true'
        }
        
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        if coin_id not in data:
            logger.warning(f"No CoinGecko data received for {symbol}")
            return None
            
        price_data = data[coin_id]
        current_price = float(price_data['usd'])
        
        # CoinGecko doesn't provide OHLCV for current minute, so we simulate it
        # This is a limitation but provides basic price data as fallback
        current_timestamp = int(time.time() * 1000)
        
        return {
            'timestamp': current_timestamp,
            'open': current_price,  # Using current price for all OHLC
            'high': current_price,
            'low': current_price,
            'close': current_price,
            'volume': 0.0  # CoinGecko simple API doesn't provide volume
        }
        
    except requests.RequestException as e:
        logger.warning(f"CoinGecko API error for {symbol}: {e}")
        return None
    except (ValueError, KeyError) as e:
        logger.warning(f"CoinGecko data parsing error for {symbol}: {e}")
        return None

def fetch_kline_data(symbol):
    """Fetch kline data with Binance.US primary, CoinGecko fallback"""
    # Try Binance.US first
    data = fetch_kline_data_binance_us(symbol)
    if data:
        logger.info(f"✅ {symbol} data from Binance.US: ${data['close']:.2f}")
        return data
    
    # Fallback to CoinGecko
    logger.info(f"🔄 Falling back to CoinGecko for {symbol}")
    data = fetch_kline_data_coingecko(symbol)
    if data:
        logger.info(f"✅ {symbol} data from CoinGecko: ${data['close']:.2f} (fallback)")
        return data
    
    logger.error(f"❌ Failed to fetch data for {symbol} from all sources")
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
    logger.info("Starting crypto data harvest (Binance.US + CoinGecko fallback)")
    
    # Initialize database
    init_database()
    
    # Fetch data for each symbol
    success_count = 0
    for symbol in SYMBOLS:
        logger.info(f"Fetching data for {symbol}")
        kline_data = fetch_kline_data(symbol)
        
        if kline_data:
            store_data(symbol, kline_data)
            success_count += 1
        else:
            logger.warning(f"Failed to fetch data for {symbol}")
    
    logger.info(f"Harvest completed: {success_count}/{len(SYMBOLS)} symbols successful")

if __name__ == "__main__":
    main() 