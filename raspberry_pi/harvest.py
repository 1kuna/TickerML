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
import yaml

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Default Configuration
DEFAULT_DB_PATH = Path(__file__).parent.parent / "data" / "db" / "crypto_data.db"
DEFAULT_BINANCE_US_API_BASE = "https://api.binance.us/api/v3"
DEFAULT_SYMBOLS = ["BTCUSD", "ETHUSD"]

# Fallback API Configuration (remains hardcoded for now)
COINGECKO_API_BASE = "https://api.coingecko.com/api/v3"
COINGECKO_SYMBOLS = {
    "BTCUSD": "bitcoin",
    "ETHUSD": "ethereum"
}

def load_config():
    """Loads configuration from YAML file, with fallbacks to defaults."""
    config_path = Path(__file__).parent.parent / "config" / "config.yaml"
    
    # Initialize with defaults
    config_values = {
        "db_path": str(DEFAULT_DB_PATH),
        "binance_api_base": DEFAULT_BINANCE_US_API_BASE,
        "symbols": DEFAULT_SYMBOLS,
    }

    if not config_path.exists():
        logger.warning(f"Config file not found at {config_path}. Using default values.")
        return config_values

    try:
        with open(config_path, 'r') as f:
            yaml_config = yaml.safe_load(f)

        if yaml_config:
            config_values["db_path"] = yaml_config.get("database", {}).get("path", DEFAULT_DB_PATH)
            if config_values["db_path"] is DEFAULT_DB_PATH:
                 logger.warning("database.path not found in config. Using default.")
            else:
                logger.info(f"Loaded database.path from config: {config_values['db_path']}")

            config_values["binance_api_base"] = yaml_config.get("data", {}).get("binance_api_base", DEFAULT_BINANCE_US_API_BASE)
            if config_values["binance_api_base"] == DEFAULT_BINANCE_US_API_BASE and not (yaml_config.get("data", {}).get("binance_api_base") == DEFAULT_BINANCE_US_API_BASE) :
                 logger.warning("data.binance_api_base not found in config. Using default.")
            else:
                logger.info(f"Loaded data.binance_api_base from config: {config_values['binance_api_base']}")
            
            config_values["symbols"] = yaml_config.get("data", {}).get("symbols", DEFAULT_SYMBOLS)
            if config_values["symbols"] == DEFAULT_SYMBOLS and not (yaml_config.get("data", {}).get("symbols") == DEFAULT_SYMBOLS) :
                 logger.warning("data.symbols not found in config. Using default.")
            else:
                logger.info(f"Loaded data.symbols from config: {config_values['symbols']}")

        else:
            logger.warning(f"Config file {config_path} is empty. Using default values.")
            
    except yaml.YAMLError as e:
        logger.error(f"Error parsing config file {config_path}: {e}. Using default values.")
    except Exception as e:
        logger.error(f"Unexpected error loading config file {config_path}: {e}. Using default values.")
        
    return config_values

# Load configuration
config = load_config()

DB_PATH = Path(config['db_path'])
BINANCE_US_API_BASE = config['binance_api_base']
SYMBOLS = config['symbols']
# BINANCE_SYMBOLS will use the same list as SYMBOLS from config for Binance.US
BINANCE_SYMBOLS = config['symbols'] 

def init_database():
    """Initialize SQLite database with OHLCV table"""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    
    conn = sqlite3.connect(str(DB_PATH))
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
    logger.info(f"Database initialized at {DB_PATH}")

def fetch_kline_data_binance_us(symbol):
    """Fetch latest 1-minute kline data from Binance.US"""
    try:
        url = f"{BINANCE_US_API_BASE}/klines" # Uses global BINANCE_US_API_BASE
        params = {
            'symbol': symbol, # Symbol should be one of BINANCE_SYMBOLS
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
    # Note: COINGECKO_SYMBOLS and COINGECKO_API_BASE remain hardcoded as per requirements
    try:
        if symbol not in COINGECKO_SYMBOLS: # Check against hardcoded COINGECKO_SYMBOLS
            logger.error(f"Symbol {symbol} not supported by CoinGecko fallback")
            return None
            
        coin_id = COINGECKO_SYMBOLS[symbol] # Uses hardcoded COINGECKO_SYMBOLS
        url = f"{COINGECKO_API_BASE}/simple/price" # Uses hardcoded COINGECKO_API_BASE
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
        conn = sqlite3.connect(str(DB_PATH)) # Uses global DB_PATH
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
    # Iterate over SYMBOLS from config
    for symbol_idx, symbol_name in enumerate(SYMBOLS):
        logger.info(f"Fetching data for {symbol_name} (from config)")
        
        # Determine the correct symbol format for Binance.US API call
        # This assumes that BINANCE_SYMBOLS from config matches the order of SYMBOLS
        binance_api_symbol = BINANCE_SYMBOLS[symbol_idx] if symbol_idx < len(BINANCE_SYMBOLS) else symbol_name
        logger.info(f"Using API symbol {binance_api_symbol} for Binance.US call.")

        kline_data = fetch_kline_data(binance_api_symbol) # Pass the API-specific symbol
        
        if kline_data:
            store_data(symbol_name, kline_data) # Store with the standardized symbol name
            success_count += 1
        else:
            logger.warning(f"Failed to fetch data for {symbol_name}")
    
    logger.info(f"Harvest completed: {success_count}/{len(SYMBOLS)} symbols successful")

if __name__ == "__main__":
    main() 