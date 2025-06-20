#!/usr/bin/env python3
"""
Funding Rate Monitor for TickerML.
Tracks perpetuals funding rates across exchanges - critical for cost calculation as rates can be up to 1% daily!
Funding rates reset every 8 hours (00:00, 08:00, 16:00 UTC).
"""

import asyncio
import json
import logging
import sqlite3
import time
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import aiohttp
import pandas as pd
import yaml
from dataclasses import dataclass

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/funding_monitor.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class FundingRate:
    """Represents a funding rate for a trading pair."""
    timestamp: float
    exchange: str
    symbol: str
    funding_rate: float
    next_funding_time: float
    mark_price: float
    index_price: Optional[float] = None
    funding_countdown: Optional[int] = None

class FundingMonitor:
    """Monitors funding rates across multiple exchanges."""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """Initialize the funding rate monitor."""
        self.config = self._load_config(config_path)
        self.db_path = self.config.get('database', {}).get('path', 'data/db/crypto_data.db')
        self.symbols = self.config.get('binance', {}).get('symbols', ['BTCUSDT', 'ETHUSDT'])
        
        # Exchange configurations
        self.exchanges = {
            'binance': {
                'base_url': 'https://fapi.binance.com',
                'funding_endpoint': '/fapi/v1/premiumIndex',
                'enabled': True
            },
            # Note: Binance.US doesn't support futures/perpetuals
            # Adding other exchanges for future expansion
            'bybit': {
                'base_url': 'https://api.bybit.com',
                'funding_endpoint': '/v2/public/tickers',
                'enabled': False  # Disable for now
            }
        }
        
        # Funding rate thresholds
        self.high_funding_threshold = 0.01  # 1% daily (very high)
        self.medium_funding_threshold = 0.005  # 0.5% daily (high)
        self.funding_interval_hours = 8  # Standard funding interval
        
        # Initialize database
        self._init_database()
        
        # Session for HTTP requests
        self.session: Optional[aiohttp.ClientSession] = None
        self.running = False
    
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
    
    def _init_database(self):
        """Initialize the database with funding rates table."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Create funding rates table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS funding_rates (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp REAL NOT NULL,
                    exchange TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    funding_rate REAL NOT NULL,
                    next_funding_time REAL NOT NULL,
                    mark_price REAL NOT NULL,
                    index_price REAL,
                    funding_countdown INTEGER,
                    daily_rate REAL,
                    annualized_rate REAL,
                    UNIQUE(timestamp, exchange, symbol)
                )
            ''')
            
            # Create funding rate alerts table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS funding_alerts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp REAL NOT NULL,
                    exchange TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    funding_rate REAL NOT NULL,
                    alert_level TEXT NOT NULL,
                    message TEXT NOT NULL
                )
            ''')
            
            # Create indexes for efficient queries
            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_funding_rates_symbol_timestamp 
                ON funding_rates(symbol, timestamp)
            ''')
            
            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_funding_rates_exchange_timestamp 
                ON funding_rates(exchange, timestamp)
            ''')
            
            conn.commit()
            conn.close()
            logger.info("Funding rates database initialized successfully")
            
        except Exception as e:
            logger.error(f"Database initialization failed: {e}")
            raise
    
    async def _fetch_binance_funding_rates(self) -> List[FundingRate]:
        """Fetch funding rates from Binance Futures API."""
        try:
            exchange_config = self.exchanges['binance']
            url = f"{exchange_config['base_url']}{exchange_config['funding_endpoint']}"
            
            async with self.session.get(url) as response:
                if response.status != 200:
                    logger.error(f"Binance API error: {response.status}")
                    return []
                
                data = await response.json()
                
                funding_rates = []
                current_time = time.time()
                
                for item in data:
                    symbol = item.get('symbol', '')
                    
                    # Filter for our tracked symbols
                    if symbol not in self.symbols:
                        continue
                    
                    try:
                        funding_rate = FundingRate(
                            timestamp=current_time,
                            exchange='binance',
                            symbol=symbol,
                            funding_rate=float(item.get('lastFundingRate', 0)),
                            next_funding_time=float(item.get('nextFundingTime', 0)) / 1000,  # Convert to seconds
                            mark_price=float(item.get('markPrice', 0)),
                            index_price=float(item.get('indexPrice', 0)),
                            funding_countdown=int(item.get('countDownHour', 0)) if 'countDownHour' in item else None
                        )
                        
                        funding_rates.append(funding_rate)
                        
                    except (ValueError, KeyError) as e:
                        logger.error(f"Error parsing funding rate for {symbol}: {e}")
                        continue
                
                logger.info(f"Fetched {len(funding_rates)} funding rates from Binance")
                return funding_rates
                
        except Exception as e:
            logger.error(f"Failed to fetch Binance funding rates: {e}")
            return []
    
    def _calculate_daily_rate(self, funding_rate: float) -> float:
        """Calculate daily funding rate from 8-hour rate."""
        return funding_rate * (24 / self.funding_interval_hours)
    
    def _calculate_annualized_rate(self, funding_rate: float) -> float:
        """Calculate annualized funding rate."""
        daily_rate = self._calculate_daily_rate(funding_rate)
        return daily_rate * 365
    
    def _assess_funding_level(self, daily_rate: float) -> str:
        """Assess the funding rate level."""
        abs_rate = abs(daily_rate)
        
        if abs_rate >= self.high_funding_threshold:
            return 'critical'
        elif abs_rate >= self.medium_funding_threshold:
            return 'high'
        elif abs_rate >= 0.001:  # 0.1% daily
            return 'medium'
        else:
            return 'low'
    
    async def _store_funding_rates(self, funding_rates: List[FundingRate]):
        """Store funding rates in the database."""
        if not funding_rates:
            return
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            funding_data = []
            alerts = []
            
            for rate in funding_rates:
                daily_rate = self._calculate_daily_rate(rate.funding_rate)
                annualized_rate = self._calculate_annualized_rate(rate.funding_rate)
                funding_level = self._assess_funding_level(daily_rate)
                
                # Prepare funding rate data
                funding_data.append((
                    rate.timestamp,
                    rate.exchange,
                    rate.symbol,
                    rate.funding_rate,
                    rate.next_funding_time,
                    rate.mark_price,
                    rate.index_price,
                    rate.funding_countdown,
                    daily_rate,
                    annualized_rate
                ))
                
                # Generate alerts for high funding rates
                if funding_level in ['high', 'critical']:
                    direction = 'Long' if daily_rate > 0 else 'Short'
                    message = (f"{direction} positions pay {abs(daily_rate)*100:.3f}% daily "
                             f"funding on {rate.symbol} ({rate.exchange})")
                    
                    alerts.append((
                        rate.timestamp,
                        rate.exchange,
                        rate.symbol,
                        rate.funding_rate,
                        funding_level,
                        message
                    ))
            
            # Insert funding rates
            cursor.executemany('''
                INSERT OR REPLACE INTO funding_rates 
                (timestamp, exchange, symbol, funding_rate, next_funding_time, 
                 mark_price, index_price, funding_countdown, daily_rate, annualized_rate)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', funding_data)
            
            # Insert alerts
            if alerts:
                cursor.executemany('''
                    INSERT INTO funding_alerts 
                    (timestamp, exchange, symbol, funding_rate, alert_level, message)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', alerts)
            
            conn.commit()
            conn.close()
            
            logger.info(f"Stored {len(funding_data)} funding rates and {len(alerts)} alerts")
            
        except Exception as e:
            logger.error(f"Failed to store funding rates: {e}")
    
    async def collect_funding_rates(self):
        """Collect funding rates from all enabled exchanges."""
        all_funding_rates = []
        
        # Collect from Binance
        if self.exchanges['binance']['enabled']:
            binance_rates = await self._fetch_binance_funding_rates()
            all_funding_rates.extend(binance_rates)
        
        # Store all rates
        await self._store_funding_rates(all_funding_rates)
        
        return all_funding_rates
    
    async def start_monitoring(self, interval_minutes: int = 15):
        """Start continuous funding rate monitoring."""
        logger.info("Starting funding rate monitoring...")
        self.running = True
        
        # Create HTTP session
        self.session = aiohttp.ClientSession()
        
        try:
            while self.running:
                try:
                    # Collect funding rates
                    rates = await self.collect_funding_rates()
                    
                    # Log summary
                    if rates:
                        high_rates = [r for r in rates if self._assess_funding_level(
                            self._calculate_daily_rate(r.funding_rate)) in ['high', 'critical']]
                        
                        if high_rates:
                            logger.warning(f"High funding rates detected: {len(high_rates)} symbols")
                            for rate in high_rates:
                                daily_rate = self._calculate_daily_rate(rate.funding_rate)
                                logger.warning(f"{rate.symbol}: {daily_rate*100:.3f}% daily on {rate.exchange}")
                    
                    # Wait for next collection
                    await asyncio.sleep(interval_minutes * 60)
                    
                except KeyboardInterrupt:
                    break
                except Exception as e:
                    logger.error(f"Error in monitoring loop: {e}")
                    await asyncio.sleep(60)  # Wait 1 minute on error
                    
        finally:
            # Cleanup
            if self.session:
                await self.session.close()
            
            self.running = False
            logger.info("Funding rate monitoring stopped")
    
    def stop_monitoring(self):
        """Stop funding rate monitoring."""
        self.running = False
    
    def get_current_funding_rates(self, symbol: Optional[str] = None) -> pd.DataFrame:
        """Get current funding rates for symbols."""
        try:
            conn = sqlite3.connect(self.db_path)
            
            if symbol:
                query = '''
                    SELECT * FROM funding_rates
                    WHERE symbol = ?
                    ORDER BY timestamp DESC
                    LIMIT 1
                '''
                params = (symbol,)
            else:
                query = '''
                    SELECT fr1.* FROM funding_rates fr1
                    INNER JOIN (
                        SELECT exchange, symbol, MAX(timestamp) as max_timestamp
                        FROM funding_rates
                        GROUP BY exchange, symbol
                    ) fr2 ON fr1.exchange = fr2.exchange 
                           AND fr1.symbol = fr2.symbol 
                           AND fr1.timestamp = fr2.max_timestamp
                '''
                params = ()
            
            df = pd.read_sql_query(query, conn, params=params)
            conn.close()
            
            return df
            
        except Exception as e:
            logger.error(f"Failed to get current funding rates: {e}")
            return pd.DataFrame()
    
    def get_funding_history(self, symbol: str, hours: int = 168) -> pd.DataFrame:  # 1 week default
        """Get funding rate history for a symbol."""
        try:
            conn = sqlite3.connect(self.db_path)
            
            time_threshold = time.time() - (hours * 3600)
            
            query = '''
                SELECT timestamp, exchange, symbol, funding_rate, daily_rate, 
                       annualized_rate, mark_price, next_funding_time
                FROM funding_rates
                WHERE symbol = ? AND timestamp >= ?
                ORDER BY timestamp DESC
            '''
            
            df = pd.read_sql_query(query, conn, params=(symbol, time_threshold))
            conn.close()
            
            # Convert timestamps to datetime
            if not df.empty:
                df['datetime'] = pd.to_datetime(df['timestamp'], unit='s')
                df['next_funding_datetime'] = pd.to_datetime(df['next_funding_time'], unit='s')
            
            return df
            
        except Exception as e:
            logger.error(f"Failed to get funding history for {symbol}: {e}")
            return pd.DataFrame()
    
    def get_high_funding_alerts(self, hours: int = 24) -> pd.DataFrame:
        """Get recent high funding rate alerts."""
        try:
            conn = sqlite3.connect(self.db_path)
            
            time_threshold = time.time() - (hours * 3600)
            
            query = '''
                SELECT timestamp, exchange, symbol, funding_rate, alert_level, message
                FROM funding_alerts
                WHERE timestamp >= ?
                ORDER BY timestamp DESC
            '''
            
            df = pd.read_sql_query(query, conn, params=(time_threshold,))
            conn.close()
            
            if not df.empty:
                df['datetime'] = pd.to_datetime(df['timestamp'], unit='s')
            
            return df
            
        except Exception as e:
            logger.error(f"Failed to get funding alerts: {e}")
            return pd.DataFrame()
    
    def calculate_funding_cost(self, symbol: str, position_size: float, 
                             hold_hours: int = 8) -> Dict:
        """Calculate estimated funding cost for a position."""
        try:
            # Get current funding rate
            current_rates = self.get_current_funding_rates(symbol)
            
            if current_rates.empty:
                return {
                    'symbol': symbol,
                    'error': 'No funding rate data available'
                }
            
            rate_data = current_rates.iloc[0]
            funding_rate = rate_data['funding_rate']
            daily_rate = rate_data['daily_rate']
            
            # Calculate costs
            funding_periods = hold_hours / self.funding_interval_hours
            total_funding_rate = funding_rate * funding_periods
            estimated_cost = abs(position_size * total_funding_rate)
            cost_percentage = abs(total_funding_rate) * 100
            
            # Determine if position pays or receives funding
            if funding_rate > 0:
                long_pays = True
                short_pays = False
            else:
                long_pays = False
                short_pays = True
            
            return {
                'symbol': symbol,
                'position_size': position_size,
                'hold_hours': hold_hours,
                'funding_periods': funding_periods,
                'current_funding_rate': funding_rate,
                'daily_rate': daily_rate,
                'total_funding_cost': estimated_cost,
                'cost_percentage': cost_percentage,
                'long_pays': long_pays,
                'short_pays': short_pays,
                'exchange': rate_data['exchange'],
                'next_funding_time': datetime.fromtimestamp(rate_data['next_funding_time']).isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to calculate funding cost: {e}")
            return {
                'symbol': symbol,
                'error': str(e)
            }
    
    def get_funding_summary(self) -> Dict:
        """Get a summary of current funding rates across all symbols."""
        try:
            current_rates = self.get_current_funding_rates()
            
            if current_rates.empty:
                return {
                    'total_symbols': 0,
                    'message': 'No funding rate data available'
                }
            
            summary = {
                'timestamp': datetime.now().isoformat(),
                'total_symbols': len(current_rates),
                'exchanges': current_rates['exchange'].unique().tolist(),
                'rate_statistics': {
                    'avg_daily_rate': current_rates['daily_rate'].mean(),
                    'max_daily_rate': current_rates['daily_rate'].max(),
                    'min_daily_rate': current_rates['daily_rate'].min(),
                    'high_rate_count': len(current_rates[current_rates['daily_rate'].abs() >= self.medium_funding_threshold])
                },
                'symbol_details': []
            }
            
            for _, row in current_rates.iterrows():
                daily_rate = row['daily_rate']
                level = self._assess_funding_level(daily_rate)
                
                summary['symbol_details'].append({
                    'symbol': row['symbol'],
                    'exchange': row['exchange'],
                    'funding_rate': row['funding_rate'],
                    'daily_rate': daily_rate,
                    'daily_rate_percent': daily_rate * 100,
                    'level': level,
                    'next_funding_time': datetime.fromtimestamp(row['next_funding_time']).isoformat()
                })
            
            return summary
            
        except Exception as e:
            logger.error(f"Failed to get funding summary: {e}")
            return {'error': str(e)}

async def main():
    """Main function to run the funding monitor."""
    monitor = FundingMonitor()
    
    try:
        # Test single collection
        logger.info("Testing funding rate collection...")
        rates = await monitor.collect_funding_rates()
        
        if rates:
            print("\nCurrent Funding Rates:")
            for rate in rates:
                daily_rate = monitor._calculate_daily_rate(rate.funding_rate)
                print(f"{rate.symbol}: {daily_rate*100:.3f}% daily ({rate.exchange})")
        
        # Get funding summary
        summary = monitor.get_funding_summary()
        print(f"\nFunding Summary: {json.dumps(summary, indent=2)}")
        
        # Start continuous monitoring (uncomment to run continuously)
        # await monitor.start_monitoring(interval_minutes=15)
        
    except KeyboardInterrupt:
        logger.info("Shutting down funding monitor...")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
    finally:
        monitor.stop_monitoring()

if __name__ == "__main__":
    # Create logs directory if it doesn't exist
    import os
    os.makedirs('logs', exist_ok=True)
    
    # Run the monitor
    asyncio.run(main())