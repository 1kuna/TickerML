#!/usr/bin/env python3
"""
Production Database Setup
Creates enhanced SQLite schema for production deployment
"""

import os
import sys
import logging
import sqlite3
from pathlib import Path
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ProductionDatabaseSetup:
    """Setup production-ready SQLite schema with TimescaleDB-equivalent structure"""
    
    def __init__(self):
        self.project_root = Path(__file__).parent.parent
        self.db_path = self.project_root / "data" / "db" / "crypto_data.db"
        self.conn = None
        
    def connect(self):
        """Connect to SQLite database"""
        try:
            # Ensure directory exists
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            
            self.conn = sqlite3.connect(str(self.db_path))
            
            # Enable WAL mode for better concurrency
            self.conn.execute("PRAGMA journal_mode=WAL")
            self.conn.execute("PRAGMA synchronous=NORMAL")
            self.conn.execute("PRAGMA cache_size=10000")
            self.conn.execute("PRAGMA temp_store=memory")
            
            logger.info(f"Connected to production database: {self.db_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error connecting to database: {e}")
            return False
    
    def create_enhanced_schema(self):
        """Create production-ready schema with all necessary tables"""
        try:
            cursor = self.conn.cursor()
            
            # Check if OHLCV table exists and add columns if needed
            cursor.execute("PRAGMA table_info(ohlcv)")
            columns = {row[1] for row in cursor.fetchall()}
            
            if 'exchange' not in columns:
                cursor.execute("ALTER TABLE ohlcv ADD COLUMN exchange TEXT DEFAULT 'binance'")
            if 'timeframe' not in columns:
                cursor.execute("ALTER TABLE ohlcv ADD COLUMN timeframe TEXT DEFAULT '1m'")
            if 'volume_usd' not in columns:
                cursor.execute("ALTER TABLE ohlcv ADD COLUMN volume_usd REAL")
            if 'trade_count' not in columns:
                cursor.execute("ALTER TABLE ohlcv ADD COLUMN trade_count INTEGER")
            if 'vwap' not in columns:
                cursor.execute("ALTER TABLE ohlcv ADD COLUMN vwap REAL")
            
            # Order book snapshots
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS order_books (
                    timestamp TEXT NOT NULL,
                    exchange TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    sequence INTEGER,
                    bids TEXT NOT NULL,  -- JSON array
                    asks TEXT NOT NULL,  -- JSON array
                    mid_price REAL,
                    spread REAL,
                    spread_bps REAL,
                    imbalance REAL,
                    bid_volume REAL,
                    ask_volume REAL,
                    local_timestamp TEXT DEFAULT (datetime('now')),
                    PRIMARY KEY (timestamp, exchange, symbol)
                )
            """)
            
            # Individual trades
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS trades (
                    timestamp TEXT NOT NULL,
                    exchange TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    trade_id TEXT,
                    price REAL NOT NULL,
                    quantity REAL NOT NULL,
                    side TEXT NOT NULL,  -- 'buy' or 'sell'
                    is_maker INTEGER DEFAULT 0,
                    value_usd REAL,
                    local_timestamp TEXT DEFAULT (datetime('now'))
                )
            """)
            
            # Portfolio state tracking
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS portfolio_state (
                    timestamp TEXT PRIMARY KEY,
                    cash_balance REAL NOT NULL,
                    total_value REAL NOT NULL,
                    positions TEXT,  -- JSON
                    daily_pnl REAL,
                    total_pnl REAL,
                    max_drawdown REAL,
                    sharpe_ratio REAL,
                    win_rate REAL,
                    trade_count INTEGER DEFAULT 0
                )
            """)
            
            # Trade executions
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS trade_executions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    action TEXT NOT NULL,  -- 'buy', 'sell', 'hold'
                    quantity REAL NOT NULL,
                    price REAL NOT NULL,
                    value_usd REAL NOT NULL,
                    fees REAL DEFAULT 0,
                    slippage REAL DEFAULT 0,
                    confidence REAL,
                    signal_source TEXT,
                    portfolio_impact TEXT,  -- JSON
                    execution_latency_ms INTEGER,
                    queue_position INTEGER
                )
            """)
            
            # News and sentiment data
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS news_articles (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    published_at TEXT,
                    source TEXT,
                    title TEXT NOT NULL,
                    description TEXT,
                    url TEXT,
                    sentiment_score REAL,
                    sentiment_category TEXT,
                    keywords TEXT,  -- JSON
                    collection_timestamp TEXT DEFAULT (datetime('now'))
                )
            """)
            
            # Hourly sentiment aggregates
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS sentiment_hourly (
                    timestamp TEXT PRIMARY KEY,
                    article_count INTEGER NOT NULL,
                    avg_sentiment REAL NOT NULL,
                    sentiment_std REAL,
                    market_regime TEXT,
                    top_keywords TEXT,  -- JSON
                    sources TEXT  -- JSON
                )
            """)
            
            # Model predictions
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS model_predictions (
                    timestamp TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    model_version TEXT NOT NULL,
                    prediction_type TEXT NOT NULL,  -- 'price', 'action', 'risk'
                    prediction_value REAL,
                    confidence REAL,
                    horizon_minutes INTEGER,
                    features_used TEXT,  -- JSON
                    model_metadata TEXT,  -- JSON
                    PRIMARY KEY (timestamp, symbol, model_version, prediction_type)
                )
            """)
            
            # Model features
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS model_features (
                    timestamp TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    features TEXT NOT NULL,  -- JSON
                    feature_version TEXT DEFAULT 'v1',
                    generation_time_ms INTEGER,
                    PRIMARY KEY (timestamp, symbol, feature_version)
                )
            """)
            
            # Funding rates (critical for perpetuals)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS funding_rates (
                    timestamp TEXT NOT NULL,
                    exchange TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    funding_rate REAL NOT NULL,
                    funding_time TEXT NOT NULL,
                    next_funding_time TEXT NOT NULL,
                    mark_price REAL,
                    index_price REAL,
                    PRIMARY KEY (timestamp, exchange, symbol)
                )
            """)
            
            # Arbitrage opportunities
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS arbitrage_opportunities (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    buy_exchange TEXT NOT NULL,
                    sell_exchange TEXT NOT NULL,
                    buy_price REAL NOT NULL,
                    sell_price REAL NOT NULL,
                    profit_absolute REAL NOT NULL,
                    profit_percentage REAL NOT NULL,
                    profit_after_fees REAL NOT NULL,
                    estimated_quantity REAL,
                    latency_risk REAL,
                    executed INTEGER DEFAULT 0
                )
            """)
            
            self.conn.commit()
            logger.info("Created enhanced production schema")
            return True
            
        except Exception as e:
            logger.error(f"Error creating schema: {e}")
            return False
    
    def create_indexes(self):
        """Create performance indexes"""
        try:
            cursor = self.conn.cursor()
            
            # Get list of tables that exist
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            existing_tables = {row[0] for row in cursor.fetchall()}
            
            # Only create indexes for tables that exist
            index_definitions = {
                'ohlcv': [
                    "CREATE INDEX IF NOT EXISTS idx_ohlcv_symbol_time_new ON ohlcv(symbol, timestamp DESC)",
                    "CREATE INDEX IF NOT EXISTS idx_ohlcv_exchange_symbol ON ohlcv(exchange, symbol)"
                ],
                'order_books': [
                    "CREATE INDEX IF NOT EXISTS idx_orderbook_symbol_time ON order_books(symbol, timestamp DESC)",
                    "CREATE INDEX IF NOT EXISTS idx_orderbook_exchange_time ON order_books(exchange, timestamp DESC)"
                ],
                'trades': [
                    "CREATE INDEX IF NOT EXISTS idx_trades_symbol_time ON trades(symbol, timestamp DESC)",
                    "CREATE INDEX IF NOT EXISTS idx_trades_side ON trades(side)"
                ],
                'portfolio_state': [
                    "CREATE INDEX IF NOT EXISTS idx_portfolio_time ON portfolio_state(timestamp DESC)"
                ],
                'trade_executions': [
                    "CREATE INDEX IF NOT EXISTS idx_executions_symbol ON trade_executions(symbol)",
                    "CREATE INDEX IF NOT EXISTS idx_executions_time ON trade_executions(timestamp DESC)"
                ],
                'news_articles': [
                    "CREATE INDEX IF NOT EXISTS idx_news_time ON news_articles(timestamp DESC)",
                    "CREATE INDEX IF NOT EXISTS idx_news_sentiment ON news_articles(sentiment_score)"
                ],
                'model_predictions': [
                    "CREATE INDEX IF NOT EXISTS idx_predictions_symbol_time ON model_predictions(symbol, timestamp DESC)",
                    "CREATE INDEX IF NOT EXISTS idx_predictions_type ON model_predictions(prediction_type)"
                ],
                'model_features': [
                    "CREATE INDEX IF NOT EXISTS idx_features_symbol_time ON model_features(symbol, timestamp DESC)"
                ],
                'funding_rates': [
                    "CREATE INDEX IF NOT EXISTS idx_funding_exchange_symbol ON funding_rates(exchange, symbol)",
                    "CREATE INDEX IF NOT EXISTS idx_funding_time ON funding_rates(timestamp DESC)"
                ],
                'arbitrage_opportunities': [
                    "CREATE INDEX IF NOT EXISTS idx_arbitrage_symbol_time ON arbitrage_opportunities(symbol, timestamp DESC)",
                    "CREATE INDEX IF NOT EXISTS idx_arbitrage_profit ON arbitrage_opportunities(profit_percentage DESC)"
                ]
            }
            
            created_count = 0
            for table_name, indexes in index_definitions.items():
                if table_name in existing_tables:
                    for index_sql in indexes:
                        try:
                            cursor.execute(index_sql)
                            created_count += 1
                        except Exception as e:
                            logger.warning(f"Could not create index for {table_name}: {e}")
            
            self.conn.commit()
            logger.info(f"Created {created_count} performance indexes")
            return True
            
        except Exception as e:
            logger.error(f"Error creating indexes: {e}")
            return False
    
    def create_views(self):
        """Create useful views for analysis"""
        try:
            cursor = self.conn.cursor()
            
            # Latest prices view
            cursor.execute("""
                CREATE VIEW IF NOT EXISTS latest_prices AS
                SELECT 
                    symbol, 
                    close as price, 
                    timestamp, 
                    volume,
                    exchange
                FROM ohlcv o1
                WHERE timestamp = (
                    SELECT MAX(timestamp) 
                    FROM ohlcv o2 
                    WHERE o2.symbol = o1.symbol AND o2.exchange = o1.exchange
                )
                ORDER BY symbol
            """)
            
            # Portfolio performance view
            cursor.execute("""
                CREATE VIEW IF NOT EXISTS portfolio_performance AS
                SELECT 
                    date(timestamp) as date,
                    MAX(total_value) as high_value,
                    MIN(total_value) as low_value,
                    AVG(daily_pnl) as avg_daily_pnl,
                    MAX(max_drawdown) as max_drawdown,
                    AVG(sharpe_ratio) as avg_sharpe,
                    SUM(trade_count) as total_trades
                FROM portfolio_state
                GROUP BY date(timestamp)
                ORDER BY date DESC
            """)
            
            # Recent trading activity
            cursor.execute("""
                CREATE VIEW IF NOT EXISTS recent_trades AS
                SELECT 
                    symbol,
                    action,
                    quantity,
                    price,
                    value_usd,
                    confidence,
                    timestamp
                FROM trade_executions
                ORDER BY timestamp DESC
                LIMIT 100
            """)
            
            # Market sentiment summary
            cursor.execute("""
                CREATE VIEW IF NOT EXISTS sentiment_summary AS
                SELECT 
                    timestamp,
                    avg_sentiment,
                    market_regime,
                    article_count
                FROM sentiment_hourly
                ORDER BY timestamp DESC
                LIMIT 24
            """)
            
            self.conn.commit()
            logger.info("Created analysis views")
            return True
            
        except Exception as e:
            logger.error(f"Error creating views: {e}")
            return False
    
    def verify_setup(self):
        """Verify database setup"""
        try:
            cursor = self.conn.cursor()
            
            # Check tables exist
            cursor.execute("""
                SELECT name FROM sqlite_master 
                WHERE type='table' AND name NOT LIKE 'sqlite_%'
                ORDER BY name
            """)
            
            tables = cursor.fetchall()
            expected_tables = {
                'ohlcv', 'order_books', 'trades', 'portfolio_state',
                'trade_executions', 'news_articles', 'sentiment_hourly',
                'model_predictions', 'model_features', 'funding_rates',
                'arbitrage_opportunities'
            }
            
            actual_tables = {table[0] for table in tables}
            
            if expected_tables.issubset(actual_tables):
                logger.info(f"All {len(expected_tables)} tables created successfully")
                
                # Check views
                cursor.execute("""
                    SELECT name FROM sqlite_master 
                    WHERE type='view'
                    ORDER BY name
                """)
                views = cursor.fetchall()
                logger.info(f"Created {len(views)} views: {[v[0] for v in views]}")
                
                return True
            else:
                missing = expected_tables - actual_tables
                logger.error(f"Missing tables: {missing}")
                return False
                
        except Exception as e:
            logger.error(f"Error verifying setup: {e}")
            return False
    
    def setup(self):
        """Run complete database setup"""
        if not self.connect():
            return False
        
        try:
            if not self.create_enhanced_schema():
                return False
            
            if not self.create_indexes():
                return False
            
            if not self.create_views():
                return False
            
            if not self.verify_setup():
                return False
            
            logger.info("Production database setup completed successfully!")
            return True
            
        finally:
            if self.conn:
                self.conn.close()


def main():
    """Main entry point"""
    setup = ProductionDatabaseSetup()
    
    success = setup.setup()
    
    if not success:
        sys.exit(1)
    
    logger.info("Database setup script completed successfully!")


if __name__ == "__main__":
    main()