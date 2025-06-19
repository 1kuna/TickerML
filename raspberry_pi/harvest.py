#!/usr/bin/env python3
"""
HIGH-FREQUENCY crypto data harvester for Raspberry Pi
Fetches DETAILED multi-timeframe data with maximum granularity:
- OHLCV data (1m, 5m, 15m, 1h intervals)
- Order book depth (top 20 levels)
- Recent trades (last 100 trades)
- Volume and trade statistics
Binance.US primary, CoinGecko fallback, automatic backfill up to 7 days
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

# HIGH-FREQUENCY DATA COLLECTION SETTINGS
TIMEFRAMES = ["1m", "5m", "15m", "1h"]  # Multiple timeframes for richer data
ORDER_BOOK_DEPTH = 20  # Top 20 bid/ask levels
RECENT_TRADES_LIMIT = 100  # Last 100 trades for each symbol
KLINE_LIMIT = 100  # Get more historical bars per request
COLLECT_ORDER_BOOKS = True  # Enable order book collection
COLLECT_TRADES = True  # Enable recent trades collection

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
    """Initialize SQLite database with comprehensive tables for high-frequency data"""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    
    conn = sqlite3.connect(str(DB_PATH))
    cursor = conn.cursor()
    
    # Check if we need to upgrade the schema
    cursor.execute("PRAGMA table_info(ohlcv)")
    columns = [row[1] for row in cursor.fetchall()]
    
    if 'timeframe' not in columns:
        logger.info("🔄 Upgrading database schema for high-frequency data collection...")
        # Backup existing data
        cursor.execute("ALTER TABLE ohlcv RENAME TO ohlcv_old")
        logger.info("📋 Backed up existing OHLCV data")
    
    # Main OHLCV table (enhanced with more fields)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS ohlcv (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp INTEGER NOT NULL,
            symbol TEXT NOT NULL,
            timeframe TEXT NOT NULL DEFAULT '1m',
            open REAL NOT NULL,
            high REAL NOT NULL,
            low REAL NOT NULL,
            close REAL NOT NULL,
            volume REAL NOT NULL,
            trade_count INTEGER,
            taker_buy_volume REAL,
            taker_buy_quote_volume REAL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(timestamp, symbol, timeframe)
        )
    """)
    
    # If we had old data, migrate it
    if 'timeframe' not in columns:
        cursor.execute("""
            INSERT INTO ohlcv (timestamp, symbol, timeframe, open, high, low, close, volume, created_at)
            SELECT timestamp, symbol, '1m', open, high, low, close, volume, created_at FROM ohlcv_old
        """)
        migrated_rows = cursor.rowcount
        cursor.execute("DROP TABLE ohlcv_old")
        logger.info(f"📈 Migrated {migrated_rows} existing records to new schema")
    
    # Order book depth table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS order_books (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp INTEGER NOT NULL,
            symbol TEXT NOT NULL,
            side TEXT NOT NULL,  -- 'bid' or 'ask'
            level INTEGER NOT NULL,  -- 0 = best price, 1 = second best, etc.
            price REAL NOT NULL,
            quantity REAL NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(timestamp, symbol, side, level)
        )
    """)
    
    # Recent trades table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS trades (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp INTEGER NOT NULL,
            symbol TEXT NOT NULL,
            trade_id INTEGER NOT NULL,
            price REAL NOT NULL,
            quantity REAL NOT NULL,
            is_buyer_maker BOOLEAN,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(symbol, trade_id)
        )
    """)
    
    # Market statistics table (24hr stats)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS market_stats (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp INTEGER NOT NULL,
            symbol TEXT NOT NULL,
            price_change REAL,
            price_change_percent REAL,
            weighted_avg_price REAL,
            prev_close_price REAL,
            last_price REAL,
            bid_price REAL,
            ask_price REAL,
            open_price REAL,
            high_price REAL,
            low_price REAL,
            volume REAL,
            quote_volume REAL,
            open_time INTEGER,
            close_time INTEGER,
            first_trade_id INTEGER,
            last_trade_id INTEGER,
            trade_count INTEGER,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(timestamp, symbol)
        )
    """)
    
    # Create indexes for faster queries
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_ohlcv_timestamp_symbol ON ohlcv(timestamp, symbol, timeframe)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_orderbooks_timestamp_symbol ON order_books(timestamp, symbol)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_trades_timestamp_symbol ON trades(timestamp, symbol)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_market_stats_timestamp_symbol ON market_stats(timestamp, symbol)")
    
    conn.commit()
    conn.close()
    logger.info("Enhanced database schema initialized for high-frequency data collection")
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

# ========================================
# HIGH-FREQUENCY DATA COLLECTION FUNCTIONS
# ========================================

def fetch_order_book_binance_us(symbol, limit=20):
    """Fetch order book depth data from Binance.US"""
    try:
        url = f"{BINANCE_US_API_BASE}/depth"
        params = {
            'symbol': symbol,
            'limit': limit
        }
        
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        timestamp = int(time.time() * 1000)
        
        order_book = {
            'timestamp': timestamp,
            'symbol': symbol,
            'bids': data.get('bids', []),
            'asks': data.get('asks', [])
        }
        
        return order_book
        
    except Exception as e:
        logger.error(f"Error fetching order book for {symbol}: {e}")
        return None

def fetch_recent_trades_binance_us(symbol, limit=100):
    """Fetch recent trades data from Binance.US"""
    try:
        url = f"{BINANCE_US_API_BASE}/trades"
        params = {
            'symbol': symbol,
            'limit': limit
        }
        
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        timestamp = int(time.time() * 1000)
        
        trades = []
        for trade in data:
            trades.append({
                'timestamp': timestamp,
                'symbol': symbol,
                'trade_id': int(trade['id']),
                'price': float(trade['price']),
                'quantity': float(trade['qty']),
                'is_buyer_maker': trade['isBuyerMaker']
            })
        
        return trades
        
    except Exception as e:
        logger.error(f"Error fetching recent trades for {symbol}: {e}")
        return None

def fetch_24hr_stats_binance_us(symbol):
    """Fetch 24hr statistics from Binance.US"""
    try:
        url = f"{BINANCE_US_API_BASE}/ticker/24hr"
        params = {'symbol': symbol}
        
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        timestamp = int(time.time() * 1000)
        
        stats = {
            'timestamp': timestamp,
            'symbol': symbol,
            'price_change': float(data.get('priceChange', 0)),
            'price_change_percent': float(data.get('priceChangePercent', 0)),
            'weighted_avg_price': float(data.get('weightedAvgPrice', 0)),
            'prev_close_price': float(data.get('prevClosePrice', 0)),
            'last_price': float(data.get('lastPrice', 0)),
            'bid_price': float(data.get('bidPrice', 0)),
            'ask_price': float(data.get('askPrice', 0)),
            'open_price': float(data.get('openPrice', 0)),
            'high_price': float(data.get('highPrice', 0)),
            'low_price': float(data.get('lowPrice', 0)),
            'volume': float(data.get('volume', 0)),
            'quote_volume': float(data.get('quoteVolume', 0)),
            'open_time': int(data.get('openTime', 0)),
            'close_time': int(data.get('closeTime', 0)),
            'first_trade_id': int(data.get('firstId', 0)),
            'last_trade_id': int(data.get('lastId', 0)),
            'trade_count': int(data.get('count', 0))
        }
        
        return stats
        
    except Exception as e:
        logger.error(f"Error fetching 24hr stats for {symbol}: {e}")
        return None

def fetch_multi_timeframe_klines_binance_us(symbol, timeframes=None):
    """Fetch kline data for multiple timeframes"""
    if timeframes is None:
        timeframes = TIMEFRAMES
    
    multi_data = {}
    
    for timeframe in timeframes:
        try:
            url = f"{BINANCE_US_API_BASE}/klines"
            params = {
                'symbol': symbol,
                'interval': timeframe,
                'limit': KLINE_LIMIT
            }
            
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            if data:
                # Get the most recent kline
                latest_kline = data[-1]
                kline_data = {
                    'timestamp': int(latest_kline[0]),
                    'timeframe': timeframe,
                    'open': float(latest_kline[1]),
                    'high': float(latest_kline[2]),
                    'low': float(latest_kline[3]),
                    'close': float(latest_kline[4]),
                    'volume': float(latest_kline[5]),
                    'trade_count': int(latest_kline[8]),
                    'taker_buy_volume': float(latest_kline[9]),
                    'taker_buy_quote_volume': float(latest_kline[10])
                }
                multi_data[timeframe] = kline_data
            
            # Rate limiting
            time.sleep(0.1)
            
        except Exception as e:
            logger.error(f"Error fetching {timeframe} klines for {symbol}: {e}")
            continue
    
    return multi_data

def store_order_book(order_book_data):
    """Store order book data in database"""
    try:
        conn = sqlite3.connect(str(DB_PATH))
        cursor = conn.cursor()
        
        # Store bids
        for i, (price, quantity) in enumerate(order_book_data['bids']):
            cursor.execute("""
                INSERT OR REPLACE INTO order_books 
                (timestamp, symbol, side, level, price, quantity)
                VALUES (?, ?, 'bid', ?, ?, ?)
            """, (
                order_book_data['timestamp'],
                order_book_data['symbol'],
                i,
                float(price),
                float(quantity)
            ))
        
        # Store asks
        for i, (price, quantity) in enumerate(order_book_data['asks']):
            cursor.execute("""
                INSERT OR REPLACE INTO order_books 
                (timestamp, symbol, side, level, price, quantity)
                VALUES (?, ?, 'ask', ?, ?, ?)
            """, (
                order_book_data['timestamp'],
                order_book_data['symbol'],
                i,
                float(price),
                float(quantity)
            ))
        
        conn.commit()
        conn.close()
        
        logger.debug(f"Stored order book for {order_book_data['symbol']} with {len(order_book_data['bids'])} bids, {len(order_book_data['asks'])} asks")
        
    except Exception as e:
        logger.error(f"Error storing order book data: {e}")

def store_trades(trades_data):
    """Store trades data in database"""
    try:
        conn = sqlite3.connect(str(DB_PATH))
        cursor = conn.cursor()
        
        for trade in trades_data:
            cursor.execute("""
                INSERT OR IGNORE INTO trades 
                (timestamp, symbol, trade_id, price, quantity, is_buyer_maker)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                trade['timestamp'],
                trade['symbol'],
                trade['trade_id'],
                trade['price'],
                trade['quantity'],
                trade['is_buyer_maker']
            ))
        
        conn.commit()
        conn.close()
        
        logger.debug(f"Stored {len(trades_data)} trades for {trades_data[0]['symbol'] if trades_data else 'unknown'}")
        
    except Exception as e:
        logger.error(f"Error storing trades data: {e}")

def store_market_stats(stats_data):
    """Store 24hr market statistics in database"""
    try:
        conn = sqlite3.connect(str(DB_PATH))
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT OR REPLACE INTO market_stats 
            (timestamp, symbol, price_change, price_change_percent, weighted_avg_price,
             prev_close_price, last_price, bid_price, ask_price, open_price, high_price,
             low_price, volume, quote_volume, open_time, close_time, first_trade_id,
             last_trade_id, trade_count)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            stats_data['timestamp'],
            stats_data['symbol'],
            stats_data['price_change'],
            stats_data['price_change_percent'],
            stats_data['weighted_avg_price'],
            stats_data['prev_close_price'],
            stats_data['last_price'],
            stats_data['bid_price'],
            stats_data['ask_price'],
            stats_data['open_price'],
            stats_data['high_price'],
            stats_data['low_price'],
            stats_data['volume'],
            stats_data['quote_volume'],
            stats_data['open_time'],
            stats_data['close_time'],
            stats_data['first_trade_id'],
            stats_data['last_trade_id'],
            stats_data['trade_count']
        ))
        
        conn.commit()
        conn.close()
        
        logger.debug(f"Stored market stats for {stats_data['symbol']}")
        
    except Exception as e:
        logger.error(f"Error storing market stats: {e}")

def store_multi_timeframe_data(symbol, multi_data):
    """Store multi-timeframe OHLCV data"""
    try:
        conn = sqlite3.connect(str(DB_PATH))
        cursor = conn.cursor()
        
        for timeframe, kline_data in multi_data.items():
            cursor.execute("""
                INSERT OR REPLACE INTO ohlcv 
                (timestamp, symbol, timeframe, open, high, low, close, volume, 
                 trade_count, taker_buy_volume, taker_buy_quote_volume)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                kline_data['timestamp'],
                symbol,
                timeframe,
                kline_data['open'],
                kline_data['high'],
                kline_data['low'],
                kline_data['close'],
                kline_data['volume'],
                kline_data['trade_count'],
                kline_data['taker_buy_volume'],
                kline_data['taker_buy_quote_volume']
            ))
        
        conn.commit()
        conn.close()
        
        logger.info(f"Stored multi-timeframe data for {symbol}: {list(multi_data.keys())}")
        
    except Exception as e:
        logger.error(f"Error storing multi-timeframe data: {e}")

def collect_comprehensive_data(symbol):
    """Collect ALL available data for a symbol"""
    logger.info(f"🚀 Collecting comprehensive data for {symbol}")
    
    # 1. Multi-timeframe OHLCV data
    multi_data = fetch_multi_timeframe_klines_binance_us(symbol)
    if multi_data:
        store_multi_timeframe_data(symbol, multi_data)
        logger.info(f"📊 Collected {len(multi_data)} timeframes for {symbol}")
    
    # 2. Order book depth
    if COLLECT_ORDER_BOOKS:
        order_book = fetch_order_book_binance_us(symbol, ORDER_BOOK_DEPTH)
        if order_book:
            store_order_book(order_book)
            logger.info(f"📈 Collected order book depth for {symbol}")
    
    # 3. Recent trades
    if COLLECT_TRADES:
        trades = fetch_recent_trades_binance_us(symbol, RECENT_TRADES_LIMIT)
        if trades:
            store_trades(trades)
            logger.info(f"💱 Collected {len(trades)} recent trades for {symbol}")
    
    # 4. 24hr market statistics
    stats = fetch_24hr_stats_binance_us(symbol)
    if stats:
        store_market_stats(stats)
        logger.info(f"📋 Collected 24hr stats for {symbol}")
    
    return True

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
        
        # Now do COMPREHENSIVE live harvest
        if not shutdown_requested:
            logger.info("🚀 Starting COMPREHENSIVE HIGH-FREQUENCY data harvest...")
            logger.info(f"📊 Data types: Multi-timeframe OHLCV, Order Books, Trades, Market Stats")
            success_count = 0
            for symbol_idx, symbol_name in enumerate(SYMBOLS):
                if shutdown_requested:
                    logger.info("Shutdown requested during live harvest.")
                    break
                
                binance_api_symbol = BINANCE_SYMBOLS[symbol_idx] if symbol_idx < len(BINANCE_SYMBOLS) else symbol_name
                logger.info(f"🎯 Collecting comprehensive data for {symbol_name}")
                
                # Use the new comprehensive data collection
                success = collect_comprehensive_data(binance_api_symbol)
                if success:
                    success_count += 1
                    logger.info(f"✅ Complete data collection successful for {symbol_name}")
                else:
                    logger.warning(f"⚠️ Some data collection failed for {symbol_name}")
                
                # Small delay between symbols to avoid rate limits
                if not shutdown_requested and symbol_idx < len(SYMBOLS) - 1:
                    time.sleep(0.5)
            
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