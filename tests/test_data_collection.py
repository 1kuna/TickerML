#!/usr/bin/env python3
"""
Test script for the crypto data collection pipeline
Tests each component individually and end-to-end
"""

import sys
import sqlite3
import requests
import time
import logging
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

import yaml

# Configuration
# Adjust PROJECT_ROOT to be the actual root of the project (two levels up from tests/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent 
CONFIG_FILE_PATH = PROJECT_ROOT / "config" / "config.yaml"

# --- Configuration Helper ---
def load_test_config():
    """Loads test configurations from config.yaml, with fallbacks."""
    # Defaults based on original hardcoded values
    defaults = {
        "binance_api_base": "https://api.binance.us/api/v3",
        "db_path": "data/db/crypto_ohlcv.db", # Relative to project root
        "dumps_path": "data/dumps",          # Relative to project root
        "symbols": ["BTCUSD", "ETHUSD"]
    }
    
    loaded_cfg = defaults.copy()

    if not CONFIG_FILE_PATH.exists():
        logger.warning(f"Config file not found at {CONFIG_FILE_PATH}. Using default test configurations.")
        return loaded_cfg

    try:
        with open(CONFIG_FILE_PATH, 'r') as f:
            yaml_config = yaml.safe_load(f)
        
        if yaml_config:
            loaded_cfg["binance_api_base"] = yaml_config.get("data", {}).get("binance_api_base", defaults["binance_api_base"])
            loaded_cfg["db_path"] = yaml_config.get("database", {}).get("path", defaults["db_path"])
            loaded_cfg["dumps_path"] = yaml_config.get("paths", {}).get("data_dumps", defaults["dumps_path"])
            loaded_cfg["symbols"] = yaml_config.get("data", {}).get("symbols", defaults["symbols"])

            for key in defaults.keys():
                if loaded_cfg[key] == defaults[key] and \
                   ( (key == "binance_api_base" and yaml_config.get("data", {}).get("binance_api_base") is None) or \
                     (key == "db_path" and yaml_config.get("database", {}).get("path") is None) or \
                     (key == "dumps_path" and yaml_config.get("paths", {}).get("data_dumps") is None) or \
                     (key == "symbols" and yaml_config.get("data", {}).get("symbols") is None) ):
                    logger.warning(f"Key '{key}' not found or misconfigured in {CONFIG_FILE_PATH}. Using default: {defaults[key]}")
                else:
                    logger.info(f"Loaded '{key}' from config: {loaded_cfg[key]}")
        else:
            logger.warning(f"Config file {CONFIG_FILE_PATH} is empty. Using default test configurations.")
            loaded_cfg = defaults.copy()
            
    except Exception as e:
        logger.error(f"Error loading test configurations from {CONFIG_FILE_PATH}: {e}. Using defaults.")
        loaded_cfg = defaults.copy()
        
    return loaded_cfg

# Load configurations
config_values = load_test_config()

# Define global variables from loaded config
# Paths from config are relative to project root, so use PROJECT_ROOT directly.
DB_PATH = PROJECT_ROOT / config_values['db_path']
DUMPS_PATH = PROJECT_ROOT / config_values['dumps_path']
BINANCE_US_API_BASE = config_values['binance_api_base']
SYMBOLS = config_values['symbols']


def test_api_connection():
    """Test 1: Verify API connectivity (Binance.US + CoinGecko fallback)"""
    logger.info("🔍 Test 1: Testing API connections...")
    
    binance_us_success = False
    coingecko_success = False
    
    # Test Binance.US
    try:
        # Test basic ping
        response = requests.get(f"{BINANCE_US_API_BASE}/ping", timeout=10)
        response.raise_for_status()
        logger.info("✅ Binance.US API ping successful")
        
        # Test klines endpoint
        params = {
            'symbol': 'BTCUSD',
            'interval': '1m',
            'limit': 1
        }
        response = requests.get(f"{BINANCE_US_API_BASE}/klines", params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        if data and len(data) > 0:
            logger.info(f"✅ Binance.US klines data received: OHLCV = {data[0][1:6]}")
            binance_us_success = True
        else:
            logger.warning("⚠️  No Binance.US klines data received")
            
    except Exception as e:
        logger.warning(f"⚠️  Binance.US API failed: {e}")
    
    # Test CoinGecko fallback
    try:
        response = requests.get("https://api.coingecko.com/api/v3/simple/price?ids=bitcoin,ethereum&vs_currencies=usd", timeout=10)
        response.raise_for_status()
        data = response.json()
        if 'bitcoin' in data and 'ethereum' in data:
            logger.info(f"✅ CoinGecko API working: BTC=${data['bitcoin']['usd']}, ETH=${data['ethereum']['usd']}")
            coingecko_success = True
        else:
            logger.warning("⚠️  CoinGecko API response incomplete")
    except Exception as e:
        logger.warning(f"⚠️  CoinGecko API failed: {e}")
    
    if binance_us_success:
        logger.info("✅ Primary API (Binance.US) is working")
        return True
    elif coingecko_success:
        logger.info("✅ Fallback API (CoinGecko) is working")
        return True
    else:
        logger.error("❌ All APIs failed")
        return False

def test_database_setup():
    """Test 2: Verify database creation and operations"""
    logger.info("🔍 Test 2: Testing database setup...")
    
    try:
        # Add raspberry_pi to sys.path if not already there
        # PROJECT_ROOT is now the actual project root.
        raspberry_pi_path = str(PROJECT_ROOT / "raspberry_pi")
        if raspberry_pi_path not in sys.path:
            sys.path.insert(0, raspberry_pi_path)
        from harvest import init_database, store_data
        
        # Initialize database (uses DB_PATH from loaded config)
        init_database() 
        logger.info("✅ Database initialized successfully")
        
        # Test data insertion
        test_data = {
            'timestamp': int(time.time() * 1000),
            'open': 50000.0,
            'high': 50100.0,
            'low': 49900.0,
            'close': 50050.0,
            'volume': 1000.0
        }
        
        store_data("TESTUSD", test_data)
        logger.info("✅ Test data stored successfully")
        
        # Verify data was stored
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM ohlcv WHERE symbol = 'TESTUSD'")
        count = cursor.fetchone()[0]
        conn.close()
        
        if count > 0:
            logger.info(f"✅ Data verification successful: {count} records found")
            return True
        else:
            logger.error("❌ No test data found in database")
            return False
            
    except Exception as e:
        logger.error(f"❌ Database test failed: {e}")
        return False

def test_data_harvesting():
    """Test 3: Test actual data harvesting from Binance"""
    logger.info("🔍 Test 3: Testing data harvesting...")
    
    try:
        # PROJECT_ROOT is now the actual project root.
        raspberry_pi_path = str(PROJECT_ROOT / "raspberry_pi")
        if raspberry_pi_path not in sys.path:
            sys.path.insert(0, raspberry_pi_path)
        from harvest import fetch_kline_data, store_data
        
        success_count = 0
        
        for symbol in SYMBOLS:
            logger.info(f"   Testing {symbol}...")
            
            # Fetch data
            kline_data = fetch_kline_data(symbol)
            
            if kline_data:
                logger.info(f"   ✅ {symbol} data fetched: Close=${kline_data['close']:.2f}")
                
                # Store data
                store_data(symbol, kline_data)
                logger.info(f"   ✅ {symbol} data stored")
                success_count += 1
            else:
                logger.error(f"   ❌ Failed to fetch {symbol} data")
        
        if success_count == len(SYMBOLS):
            logger.info("✅ All symbols harvested successfully")
            return True
        else:
            logger.warning(f"⚠️  Only {success_count}/{len(SYMBOLS)} symbols successful")
            return False
            
    except Exception as e:
        logger.error(f"❌ Data harvesting test failed: {e}")
        return False

def test_data_export():
    """Test 4: Test CSV export functionality"""
    logger.info("🔍 Test 4: Testing data export...")
    
    try:
        # PROJECT_ROOT is now the actual project root.
        raspberry_pi_path = str(PROJECT_ROOT / "raspberry_pi")
        if raspberry_pi_path not in sys.path:
            sys.path.insert(0, raspberry_pi_path)
        from export_etl import export_daily_data
        
        # Run export (uses DUMPS_PATH and DB_PATH from loaded config)
        export_daily_data()
        
        # Check if CSV files were created
        csv_files = list(DUMPS_PATH.glob("*.csv"))
        
        if csv_files:
            logger.info(f"✅ Export successful: {len(csv_files)} CSV files created")
            for csv_file in csv_files:
                df = pd.read_csv(csv_file)
                logger.info(f"   📄 {csv_file.name}: {len(df)} records")
            return True
        else:
            logger.error("❌ No CSV files found after export")
            return False
            
    except Exception as e:
        logger.error(f"❌ Data export test failed: {e}")
        return False

def test_database_queries():
    """Test 5: Test database query performance"""
    logger.info("🔍 Test 5: Testing database queries...")
    
    try:
        conn = sqlite3.connect(DB_PATH)
        
        # Test basic count
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM ohlcv")
        total_records = cursor.fetchone()[0]
        logger.info(f"✅ Total records in database: {total_records}")
        
        # Test symbol-specific queries
        for symbol in SYMBOLS:
            cursor.execute("SELECT COUNT(*) FROM ohlcv WHERE symbol = ?", (symbol,))
            symbol_count = cursor.fetchone()[0]
            logger.info(f"   📊 {symbol}: {symbol_count} records")
        
        # Test recent data query (last hour)
        one_hour_ago = int((datetime.now() - timedelta(hours=1)).timestamp() * 1000)
        cursor.execute("""
            SELECT symbol, COUNT(*) as count, 
                   MIN(timestamp) as min_time, 
                   MAX(timestamp) as max_time
            FROM ohlcv 
            WHERE timestamp >= ? AND symbol IN ('BTCUSD', 'ETHUSD')
            GROUP BY symbol
        """, (one_hour_ago,))
        
        recent_data = cursor.fetchall()
        if recent_data:
            logger.info("✅ Recent data query successful:")
            for row in recent_data:
                symbol, count, min_time, max_time = row
                min_dt = datetime.fromtimestamp(min_time/1000)
                max_dt = datetime.fromtimestamp(max_time/1000)
                logger.info(f"   📈 {symbol}: {count} records from {min_dt} to {max_dt}")
        else:
            logger.warning("⚠️  No recent data found (last hour)")
        
        conn.close()
        return True
        
    except Exception as e:
        logger.error(f"❌ Database query test failed: {e}")
        return False

def test_dashboard_data():
    """Test 6: Test data availability for dashboard"""
    logger.info("🔍 Test 6: Testing dashboard data availability...")
    
    try:
        conn = sqlite3.connect(DB_PATH)
        
        # Test data for last 24 hours (dashboard default)
        end_time = datetime.now()
        start_time = end_time - timedelta(hours=24)
        start_timestamp = int(start_time.timestamp() * 1000)
        end_timestamp = int(end_time.timestamp() * 1000)
        
        for symbol in SYMBOLS:
            query = """
                SELECT COUNT(*) as count,
                       MIN(close) as min_price,
                       MAX(close) as max_price,
                       AVG(close) as avg_price
                FROM ohlcv 
                WHERE symbol = ? AND timestamp >= ? AND timestamp <= ?
            """
            
            df = pd.read_sql_query(query, conn, params=(symbol, start_timestamp, end_timestamp))
            
            if not df.empty and df.iloc[0]['count'] > 0:
                row = df.iloc[0]
                logger.info(f"✅ {symbol} dashboard data ready:")
                logger.info(f"   📊 Records: {row['count']}")
                logger.info(f"   💰 Price range: ${row['min_price']:.2f} - ${row['max_price']:.2f}")
                logger.info(f"   📈 Average: ${row['avg_price']:.2f}")
            else:
                logger.warning(f"⚠️  {symbol}: No data available for dashboard (last 24h)")
        
        conn.close()
        return True
        
    except Exception as e:
        logger.error(f"❌ Dashboard data test failed: {e}")
        return False

def run_continuous_test(duration_minutes=5):
    """Test 7: Run continuous data collection for specified duration"""
    logger.info(f"🔍 Test 7: Running continuous collection for {duration_minutes} minutes...")
    
    try:
        # PROJECT_ROOT is now the actual project root.
        raspberry_pi_path = str(PROJECT_ROOT / "raspberry_pi")
        if raspberry_pi_path not in sys.path:
            sys.path.insert(0, raspberry_pi_path)
        from harvest import fetch_kline_data, store_data
        
        start_time = time.time()
        end_time = start_time + (duration_minutes * 60)
        collection_count = 0
        
        while time.time() < end_time:
            logger.info(f"   Collection cycle {collection_count + 1}...")
            
            cycle_success = 0
            for symbol in SYMBOLS:
                kline_data = fetch_kline_data(symbol)
                if kline_data:
                    store_data(symbol, kline_data)
                    cycle_success += 1
            
            if cycle_success == len(SYMBOLS):
                collection_count += 1
                logger.info(f"   ✅ Cycle {collection_count} successful")
            else:
                logger.warning(f"   ⚠️  Cycle {collection_count + 1} partial success: {cycle_success}/{len(SYMBOLS)}")
            
            # Wait 60 seconds (simulate cron minute interval)
            time.sleep(60)
        
        logger.info(f"✅ Continuous test completed: {collection_count} successful cycles")
        return True
        
    except KeyboardInterrupt:
        logger.info("⏹️  Continuous test interrupted by user")
        return True
    except Exception as e:
        logger.error(f"❌ Continuous test failed: {e}")
        return False

def main():
    """Run all tests"""
    logger.info("🚀 Starting TickerML Data Collection Pipeline Tests")
    logger.info("=" * 60)
    
    tests = [
        ("API Connection", test_api_connection),
        ("Database Setup", test_database_setup),
        ("Data Harvesting", test_data_harvesting),
        ("Data Export", test_data_export),
        ("Database Queries", test_database_queries),
        ("Dashboard Data", test_dashboard_data),
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        logger.info("")
        try:
            results[test_name] = test_func()
        except Exception as e:
            logger.error(f"❌ {test_name} failed with exception: {e}")
            results[test_name] = False
    
    # Summary
    logger.info("")
    logger.info("📋 TEST SUMMARY")
    logger.info("=" * 60)
    
    passed = 0
    total = len(tests)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        logger.info(f"{status} - {test_name}")
        if result:
            passed += 1
    
    logger.info("")
    logger.info(f"🎯 Results: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 All tests passed! Data collection pipeline is ready.")
        
        # Offer continuous test
        try:
            response = input("\n🔄 Run continuous collection test? (y/N): ").strip().lower()
            if response in ['y', 'yes']:
                duration = input("Duration in minutes (default 5): ").strip()
                duration = int(duration) if duration.isdigit() else 5
                run_continuous_test(duration)
        except KeyboardInterrupt:
            logger.info("\n👋 Test session ended by user")
    else:
        logger.warning("⚠️  Some tests failed. Please check the logs above.")
        logger.info("\n💡 Common fixes:")
        logger.info("   - Ensure internet connection for API tests")
        logger.info("   - Check file permissions for database/CSV creation")
        logger.info("   - Verify directory structure exists")
        logger.info("   - Run: mkdir -p data/db data/dumps logs")

if __name__ == "__main__":
    main() 