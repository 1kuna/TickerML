#!/usr/bin/env python3
"""
Minute-level crypto data harvester for Raspberry Pi
Fetches BTC/USD and ETH/USD data with Binance.US primary, CoinGecko fallback
Automatically backfills missing historical data up to 7 days, then continues live
"""

import sqlite3
import requests
import time
import logging
import signal
import os
from datetime import datetime, timedelta
from pathlib import Path
import yaml

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

# Default Configuration
DEFAULT_DB_PATH = Path(__file__).parent.parent / "data" / "db" / "crypto_ohlcv.db"
DEFAULT_BINANCE_US_API_BASE = "https://api.binance.us/api/v3"
DEFAULT_SYMBOLS = ["BTCUSD", "ETHUSD"]

# Fallback API Configuration (remains hardcoded for now)
COINGECKO_API_BASE = "https://api.coingecko.com/api/v3"
COINGECKO_SYMBOLS = {
    "BTCUSD": "bitcoin",
    "ETHUSD": "ethereum"
}

# Rate limiting and backfill configuration
BINANCE_RATE_LIMIT_PER_MINUTE = 1200
BATCH_SIZE = 500
MAX_BACKFILL_DAYS = 7  # Automatically backfill up to 7 days

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
            config_values["db_path"] = yaml_config.get("database", {}).get("ohlcv_path", DEFAULT_DB_PATH)
            if config_values["db_path"] is DEFAULT_DB_PATH:
                 logger.warning("database.ohlcv_path not found in config. Using default.")
            else:
                logger.info(f"Loaded database.ohlcv_path from config: {config_values['db_path']}")

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
    conn = None # Initialize conn to None
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
        
    except sqlite3.Error as e:
        logger.error(f"Database error storing {symbol}: {e}")
    finally:
        if conn:
            conn.close()

def fetch_historical_klines_binance_us(symbol, start_time, end_time, limit=500):
    """Fetch historical kline data from Binance.US with rate limiting"""
    try:
        url = f"{BINANCE_US_API_BASE}/klines"
        params = {
            'symbol': symbol,
            'interval': '1m',
            'startTime': start_time,
            'endTime': end_time,
            'limit': limit
        }
        
        # Rate limiting: ensure we don't exceed limits
        time.sleep(60 / BINANCE_RATE_LIMIT_PER_MINUTE)
        
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        
        data = response.json()
        if not data:
            return []
        
        klines = []
        for kline in data:
            klines.append({
                'timestamp': int(kline[0]),
                'open': float(kline[1]),
                'high': float(kline[2]),
                'low': float(kline[3]),
                'close': float(kline[4]),
                'volume': float(kline[5])
            })
        
        return klines
        
    except requests.RequestException:
        return []
    except (ValueError, IndexError):
        return []

def store_historical_data_batch(symbol, klines_data):
    """Store multiple kline records in batch"""
    if not klines_data:
        return 0
    
    conn = None # Initialize conn to None
    try:
        conn = sqlite3.connect(str(DB_PATH))
        cursor = conn.cursor()
        
        records = []
        for kline in klines_data:
            records.append((
                kline['timestamp'],
                symbol,
                kline['open'],
                kline['high'],
                kline['low'],
                kline['close'],
                kline['volume']
            ))
        
        cursor.executemany("""
            INSERT OR IGNORE INTO ohlcv 
            (timestamp, symbol, open, high, low, close, volume)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, records)
        
        inserted_count = cursor.rowcount
        conn.commit()
        
        return inserted_count
        
    except sqlite3.Error as e: # It's good practice to log the error
        logger.error(f"Database error storing historical batch for {symbol}: {e}")
        return 0
    finally:
        if conn:
            conn.close()

def check_and_backfill(symbol):
    """Check for missing data and backfill automatically up to MAX_BACKFILL_DAYS"""
    conn = None # Initialize conn to None
    try:
        conn = sqlite3.connect(str(DB_PATH))
        cursor = conn.cursor()
        
        # Check if we have any data for this symbol
        cursor.execute("SELECT MIN(timestamp), MAX(timestamp) FROM ohlcv WHERE symbol = ?", (symbol,))
        result = cursor.fetchone()
        # No conn.close() here, keep it open for the duration of this function logic if needed, or close and reopen.
        # For simplicity here, we'll close it if we return early, or it will be closed in finally.
        
        if not result[0]:  # No data exists
            logger.info(f"No existing data for {symbol}, starting {MAX_BACKFILL_DAYS}-day backfill")
            if conn: # Close connection if we are about to make many API calls before next DB op
                conn.close()
                conn = None
            end_time = datetime.now()
            start_time = end_time - timedelta(days=MAX_BACKFILL_DAYS)
        else:
            # Check if we have recent data (within last hour)
            latest_timestamp = result[1]
            latest_datetime = datetime.fromtimestamp(latest_timestamp / 1000)
            now = datetime.now()
            
            if (now - latest_datetime).total_seconds() < 3600:  # Less than 1 hour old
                logger.info(f"Recent data exists for {symbol}, skipping backfill")
                # conn was opened, ensure it's closed before returning
                if conn:
                    conn.close()
                    conn = None
                return 0
            
            # Fill gap from latest data to now
            logger.info(f"Gap detected for {symbol}, backfilling from {latest_datetime}")
            start_time = latest_datetime
            end_time = now
            if conn: # Close connection if we are about to make many API calls before next DB op
                conn.close()
                conn = None
        
        # Perform backfill
        total_inserted = 0
        current_time = start_time
        
        while current_time < end_time and not shutdown_requested: # Check shutdown_requested
            batch_end = min(current_time + timedelta(hours=8), end_time)  # 8 hours per batch
            
            start_ts = int(current_time.timestamp() * 1000)
            end_ts = int(batch_end.timestamp() * 1000)
            
            historical_data = fetch_historical_klines_binance_us(symbol, start_ts, end_ts, BATCH_SIZE)
            
            if historical_data:
                inserted = store_historical_data_batch(symbol, historical_data) # This function now handles its own conn
                total_inserted += inserted
                if inserted > 0:
                    logger.info(f"Backfilled {inserted} records for {symbol}")
            
            current_time = batch_end
            if shutdown_requested:
                logger.info(f"Shutdown requested during backfill for {symbol}. Processed up to {current_time}.")
                break
        
        if total_inserted > 0:
            logger.info(f"✅ Backfill completed for {symbol}: {total_inserted} records")
        
        return total_inserted
        
    except Exception as e:
        logger.error(f"Backfill error for {symbol}: {e}")
        return 0
    finally:
        if conn: # Ensure connection opened at the start of check_and_backfill is closed
            conn.close()

def main():
    """Main function: auto-backfill missing data, then live harvest"""
    global shutdown_requested
    logger.info("Starting crypto data harvest with automatic backfill")
    
    try:
        # Initialize database
        init_database()
        
        # Auto-backfill missing historical data for each symbol
        logger.info("Checking for missing historical data...")
        total_backfilled = 0
        for symbol in SYMBOLS:
            if shutdown_requested:
                logger.info("Shutdown requested before backfill loop could complete.")
                break
            backfilled = check_and_backfill(symbol) # This function now checks shutdown_requested
            total_backfilled += backfilled
        
        if not shutdown_requested:
            if total_backfilled > 0:
                logger.info(f"📈 Historical backfill completed: {total_backfilled} total records")
            else:
                logger.info("✅ No historical backfill needed, data is up to date")
        
        # Now do live harvest
        if not shutdown_requested:
            logger.info("Starting live data harvest...")
            success_count = 0
            for symbol_idx, symbol_name in enumerate(SYMBOLS):
                if shutdown_requested:
                    logger.info("Shutdown requested during live harvest.")
                    break
                logger.info(f"Fetching live data for {symbol_name}")
                
                binance_api_symbol = BINANCE_SYMBOLS[symbol_idx] if symbol_idx < len(BINANCE_SYMBOLS) else symbol_name
                kline_data = fetch_kline_data(binance_api_symbol)

                if kline_data:
                    store_data(symbol_name, kline_data) # This function now handles its own conn
                    success_count += 1
                else:
                    logger.warning(f"Failed to fetch live data for {symbol_name}")
            
            if not shutdown_requested:
                logger.info(f"🎯 Live harvest completed: {success_count}/{len(SYMBOLS)} symbols successful")

    except KeyboardInterrupt:
        logger.info("KeyboardInterrupt received. Requesting shutdown...")
        shutdown_requested = True
    except Exception as e:
        logger.error(f"An unexpected error occurred in main: {e}", exc_info=True)
    finally:
        if shutdown_requested:
            logger.info("Shutdown process initiated in main. Exiting.")
        else:
            logger.info("Main process completed normally.")

if __name__ == "__main__":
    # Register signal handlers
    signal.signal(signal.SIGINT, handle_shutdown_signal)
    signal.signal(signal.SIGTERM, handle_shutdown_signal)
    
    main() 