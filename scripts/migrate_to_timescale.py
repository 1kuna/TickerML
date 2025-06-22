#!/usr/bin/env python3
"""
SQLite to TimescaleDB Migration Script
Migrates existing SQLite data to TimescaleDB for production deployment
"""

import os
import sys
import logging
import sqlite3
import psycopg2
import pandas as pd
from datetime import datetime, timedelta
import yaml
from pathlib import Path
import argparse

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TimescaleMigration:
    """Handle migration from SQLite to TimescaleDB"""
    
    def __init__(self, config_path: str = None):
        self.project_root = Path(__file__).parent.parent
        
        # Load TimescaleDB configuration
        if not config_path:
            config_path = self.project_root / "config" / "timescale_config.yaml"
        
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        self.timescale_config = config['connection']
        self.sqlite_path = self.project_root / "data" / "db" / "crypto_data.db"
        
        # Database connections
        self.sqlite_conn = None
        self.timescale_conn = None
        
    def connect_databases(self):
        """Connect to both SQLite and TimescaleDB"""
        try:
            # Connect to SQLite
            if not self.sqlite_path.exists():
                logger.error(f"SQLite database not found at {self.sqlite_path}")
                return False
            
            self.sqlite_conn = sqlite3.connect(str(self.sqlite_path))
            logger.info("Connected to SQLite database")
            
            # Connect to TimescaleDB
            self.timescale_conn = psycopg2.connect(
                host=self.timescale_config['host'],
                port=self.timescale_config['port'],
                database=self.timescale_config['database'],
                user=self.timescale_config['username'],
                password=os.getenv(self.timescale_config['password_env_var'], 'tickerml_pass')
            )
            logger.info("Connected to TimescaleDB")
            
            return True
            
        except Exception as e:
            logger.error(f"Error connecting to databases: {e}")
            return False
    
    def setup_timescale_schema(self):
        """Create TimescaleDB schema and hypertables"""
        try:
            cursor = self.timescale_conn.cursor()
            
            # Enable TimescaleDB extension
            cursor.execute("CREATE EXTENSION IF NOT EXISTS timescaledb;")
            
            # Create schema
            schema_sql = """
            -- OHLCV data with enhanced structure
            CREATE TABLE IF NOT EXISTS ohlcv (
                timestamp TIMESTAMPTZ NOT NULL,
                symbol VARCHAR(20) NOT NULL,
                exchange VARCHAR(20) DEFAULT 'binance',
                open NUMERIC(20,8) NOT NULL,
                high NUMERIC(20,8) NOT NULL,
                low NUMERIC(20,8) NOT NULL,
                close NUMERIC(20,8) NOT NULL,
                volume NUMERIC(20,8) NOT NULL,
                volume_usd NUMERIC(20,8),
                trade_count INTEGER,
                collection_timestamp TIMESTAMPTZ DEFAULT NOW()
            );
            
            -- Order book snapshots
            CREATE TABLE IF NOT EXISTS order_books (
                timestamp TIMESTAMPTZ NOT NULL,
                exchange VARCHAR(20) NOT NULL,
                symbol VARCHAR(20) NOT NULL,
                sequence BIGINT,
                bids JSONB NOT NULL,
                asks JSONB NOT NULL,
                mid_price NUMERIC(20,8),
                spread NUMERIC(20,8),
                imbalance NUMERIC(10,6),
                bid_volume NUMERIC(20,8),
                ask_volume NUMERIC(20,8),
                local_timestamp TIMESTAMPTZ DEFAULT NOW()
            );
            
            -- Individual trades
            CREATE TABLE IF NOT EXISTS trades (
                timestamp TIMESTAMPTZ NOT NULL,
                exchange VARCHAR(20) NOT NULL,
                symbol VARCHAR(20) NOT NULL,
                trade_id VARCHAR(50),
                price NUMERIC(20,8) NOT NULL,
                quantity NUMERIC(20,8) NOT NULL,
                side VARCHAR(10) NOT NULL,  -- 'buy' or 'sell'
                is_maker BOOLEAN DEFAULT FALSE,
                value_usd NUMERIC(20,8),
                local_timestamp TIMESTAMPTZ DEFAULT NOW()
            );
            
            -- Portfolio state tracking
            CREATE TABLE IF NOT EXISTS portfolio_state (
                timestamp TIMESTAMPTZ NOT NULL,
                cash_balance NUMERIC(20,8) NOT NULL,
                total_value NUMERIC(20,8) NOT NULL,
                positions JSONB,
                daily_pnl NUMERIC(20,8),
                max_drawdown NUMERIC(10,4),
                sharpe_ratio NUMERIC(10,4),
                win_rate NUMERIC(10,4),
                trade_count INTEGER DEFAULT 0
            );
            
            -- Trade executions
            CREATE TABLE IF NOT EXISTS trade_executions (
                id SERIAL PRIMARY KEY,
                timestamp TIMESTAMPTZ NOT NULL,
                symbol VARCHAR(20) NOT NULL,
                action VARCHAR(10) NOT NULL,  -- 'buy', 'sell'
                quantity NUMERIC(20,8) NOT NULL,
                price NUMERIC(20,8) NOT NULL,
                value_usd NUMERIC(20,8) NOT NULL,
                fees NUMERIC(20,8) DEFAULT 0,
                slippage NUMERIC(10,6) DEFAULT 0,
                confidence NUMERIC(10,4),
                signal_source VARCHAR(50),
                portfolio_impact JSONB
            );
            
            -- News and sentiment data
            CREATE TABLE IF NOT EXISTS news_articles (
                id SERIAL PRIMARY KEY,
                timestamp TIMESTAMPTZ NOT NULL,
                published_at TIMESTAMPTZ,
                source VARCHAR(100),
                title TEXT NOT NULL,
                description TEXT,
                url TEXT,
                sentiment_score NUMERIC(10,4),
                sentiment_category VARCHAR(20),
                keywords JSONB,
                collection_timestamp TIMESTAMPTZ DEFAULT NOW()
            );
            
            -- Hourly sentiment aggregates
            CREATE TABLE IF NOT EXISTS sentiment_hourly (
                timestamp TIMESTAMPTZ NOT NULL,
                article_count INTEGER NOT NULL,
                avg_sentiment NUMERIC(10,4) NOT NULL,
                sentiment_std NUMERIC(10,4),
                market_regime VARCHAR(20),
                top_keywords JSONB,
                sources JSONB
            );
            
            -- Model predictions and features
            CREATE TABLE IF NOT EXISTS model_features (
                timestamp TIMESTAMPTZ NOT NULL,
                symbol VARCHAR(20) NOT NULL,
                features JSONB NOT NULL,
                feature_version VARCHAR(20) DEFAULT 'v1',
                generation_time_ms INTEGER
            );
            
            CREATE TABLE IF NOT EXISTS model_predictions (
                timestamp TIMESTAMPTZ NOT NULL,
                symbol VARCHAR(20) NOT NULL,
                model_version VARCHAR(50) NOT NULL,
                prediction_type VARCHAR(20) NOT NULL,  -- 'price', 'action', 'risk'
                prediction_value NUMERIC(20,8),
                confidence NUMERIC(10,4),
                horizon_minutes INTEGER,
                features_used JSONB,
                model_metadata JSONB
            );
            """
            
            cursor.execute(schema_sql)
            
            # Create hypertables
            hypertables = [
                ("ohlcv", "timestamp"),
                ("order_books", "timestamp"),
                ("trades", "timestamp"),
                ("portfolio_state", "timestamp"),
                ("trade_executions", "timestamp"),
                ("news_articles", "timestamp"),
                ("sentiment_hourly", "timestamp"),
                ("model_features", "timestamp"),
                ("model_predictions", "timestamp")
            ]
            
            for table_name, time_column in hypertables:
                try:
                    cursor.execute(f"SELECT create_hypertable('{table_name}', '{time_column}', if_not_exists => TRUE);")
                    logger.info(f"Created hypertable: {table_name}")
                except Exception as e:
                    logger.warning(f"Hypertable {table_name} may already exist: {e}")
            
            # Create indexes for performance
            indexes = [
                "CREATE INDEX IF NOT EXISTS idx_ohlcv_symbol_time ON ohlcv (symbol, timestamp DESC);",
                "CREATE INDEX IF NOT EXISTS idx_orderbooks_symbol_time ON order_books (symbol, timestamp DESC);",
                "CREATE INDEX IF NOT EXISTS idx_trades_symbol_time ON trades (symbol, timestamp DESC);",
                "CREATE INDEX IF NOT EXISTS idx_trades_side ON trades (side);",
                "CREATE INDEX IF NOT EXISTS idx_portfolio_time ON portfolio_state (timestamp DESC);",
                "CREATE INDEX IF NOT EXISTS idx_executions_symbol ON trade_executions (symbol);",
                "CREATE INDEX IF NOT EXISTS idx_news_sentiment ON news_articles (sentiment_score);",
                "CREATE INDEX IF NOT EXISTS idx_features_symbol_time ON model_features (symbol, timestamp DESC);"
            ]
            
            for index_sql in indexes:
                cursor.execute(index_sql)
            
            # Set up data retention policies
            retention_policies = [
                ("ohlcv", "30 days"),
                ("order_books", "7 days"),  # High frequency data
                ("trades", "14 days"),
                ("news_articles", "90 days"),
                ("model_features", "7 days"),
                ("model_predictions", "30 days")
            ]
            
            for table_name, retention_period in retention_policies:
                try:
                    cursor.execute(f"SELECT add_retention_policy('{table_name}', INTERVAL '{retention_period}');")
                    logger.info(f"Added retention policy for {table_name}: {retention_period}")
                except Exception as e:
                    logger.warning(f"Retention policy for {table_name} may already exist: {e}")
            
            self.timescale_conn.commit()
            logger.info("TimescaleDB schema created successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error setting up TimescaleDB schema: {e}")
            self.timescale_conn.rollback()
            return False
    
    def migrate_ohlcv_data(self):
        """Migrate OHLCV data from SQLite to TimescaleDB"""
        try:
            logger.info("Migrating OHLCV data...")
            
            # Read from SQLite
            sqlite_cursor = self.sqlite_conn.cursor()
            sqlite_cursor.execute("SELECT * FROM ohlcv ORDER BY timestamp")
            
            # Get column names
            columns = [description[0] for description in sqlite_cursor.description]
            
            # Batch insert to TimescaleDB
            timescale_cursor = self.timescale_conn.cursor()
            
            batch_size = 1000
            batch = []
            total_rows = 0
            
            for row in sqlite_cursor:
                # Convert row to dict
                row_dict = dict(zip(columns, row))
                
                # Convert timestamp
                if isinstance(row_dict['timestamp'], str):
                    row_dict['timestamp'] = datetime.fromisoformat(row_dict['timestamp'].replace('Z', '+00:00'))
                
                batch.append(row_dict)
                
                if len(batch) >= batch_size:
                    self._insert_ohlcv_batch(timescale_cursor, batch)
                    total_rows += len(batch)
                    batch = []
                    
                    if total_rows % 10000 == 0:
                        logger.info(f"Migrated {total_rows} OHLCV rows")
            
            # Insert remaining batch
            if batch:
                self._insert_ohlcv_batch(timescale_cursor, batch)
                total_rows += len(batch)
            
            self.timescale_conn.commit()
            logger.info(f"Migrated {total_rows} OHLCV rows successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error migrating OHLCV data: {e}")
            self.timescale_conn.rollback()
            return False
    
    def _insert_ohlcv_batch(self, cursor, batch):
        """Insert batch of OHLCV data"""
        insert_sql = """
        INSERT INTO ohlcv (timestamp, symbol, open, high, low, close, volume, exchange, collection_timestamp)
        VALUES (%(timestamp)s, %(symbol)s, %(open)s, %(high)s, %(low)s, %(close)s, %(volume)s, 'binance', NOW())
        ON CONFLICT (timestamp, symbol) DO NOTHING
        """
        
        cursor.executemany(insert_sql, batch)
    
    def migrate_predictions_data(self):
        """Migrate predictions data"""
        try:
            logger.info("Migrating predictions data...")
            
            sqlite_cursor = self.sqlite_conn.cursor()
            sqlite_cursor.execute("SELECT * FROM predictions ORDER BY timestamp")
            
            columns = [description[0] for description in sqlite_cursor.description]
            timescale_cursor = self.timescale_conn.cursor()
            
            batch_size = 1000
            batch = []
            total_rows = 0
            
            for row in sqlite_cursor:
                row_dict = dict(zip(columns, row))
                
                # Convert timestamp
                if isinstance(row_dict['timestamp'], str):
                    row_dict['timestamp'] = datetime.fromisoformat(row_dict['timestamp'].replace('Z', '+00:00'))
                
                # Create separate entries for different prediction types
                predictions = [
                    {
                        'timestamp': row_dict['timestamp'],
                        'symbol': row_dict['symbol'],
                        'model_version': 'legacy_v1',
                        'prediction_type': 'price_5m',
                        'prediction_value': row_dict.get('prediction_5m'),
                        'confidence': 0.5,  # Default confidence
                        'horizon_minutes': 5
                    },
                    {
                        'timestamp': row_dict['timestamp'],
                        'symbol': row_dict['symbol'],
                        'model_version': 'legacy_v1',
                        'prediction_type': 'price_10m',
                        'prediction_value': row_dict.get('prediction_10m'),
                        'confidence': 0.5,
                        'horizon_minutes': 10
                    },
                    {
                        'timestamp': row_dict['timestamp'],
                        'symbol': row_dict['symbol'],
                        'model_version': 'legacy_v1',
                        'prediction_type': 'direction',
                        'prediction_value': 1 if row_dict.get('direction') == 'up' else 0,
                        'confidence': 0.5,
                        'horizon_minutes': 30
                    }
                ]
                
                batch.extend(predictions)
                
                if len(batch) >= batch_size:
                    self._insert_predictions_batch(timescale_cursor, batch)
                    total_rows += len(batch)
                    batch = []
                    
                    if total_rows % 10000 == 0:
                        logger.info(f"Migrated {total_rows} prediction rows")
            
            # Insert remaining batch
            if batch:
                self._insert_predictions_batch(timescale_cursor, batch)
                total_rows += len(batch)
            
            self.timescale_conn.commit()
            logger.info(f"Migrated {total_rows} prediction rows successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error migrating predictions data: {e}")
            self.timescale_conn.rollback()
            return False
    
    def _insert_predictions_batch(self, cursor, batch):
        """Insert batch of predictions data"""
        insert_sql = """
        INSERT INTO model_predictions (timestamp, symbol, model_version, prediction_type, 
                                     prediction_value, confidence, horizon_minutes)
        VALUES (%(timestamp)s, %(symbol)s, %(model_version)s, %(prediction_type)s,
                %(prediction_value)s, %(confidence)s, %(horizon_minutes)s)
        ON CONFLICT DO NOTHING
        """
        
        cursor.executemany(insert_sql, batch)
    
    def migrate_sentiment_data(self):
        """Migrate sentiment data"""
        try:
            logger.info("Migrating sentiment data...")
            
            sqlite_cursor = self.sqlite_conn.cursor()
            
            # Check if sentiment_data table exists
            sqlite_cursor.execute("""
                SELECT name FROM sqlite_master 
                WHERE type='table' AND name='sentiment_data'
            """)
            
            if not sqlite_cursor.fetchone():
                logger.info("No sentiment data table found in SQLite")
                return True
            
            sqlite_cursor.execute("SELECT * FROM sentiment_data ORDER BY timestamp")
            columns = [description[0] for description in sqlite_cursor.description]
            timescale_cursor = self.timescale_conn.cursor()
            
            batch = []
            total_rows = 0
            
            for row in sqlite_cursor:
                row_dict = dict(zip(columns, row))
                
                # Convert timestamp
                if isinstance(row_dict['timestamp'], str):
                    row_dict['timestamp'] = datetime.fromisoformat(row_dict['timestamp'].replace('Z', '+00:00'))
                
                # Map to new schema
                sentiment_row = {
                    'timestamp': row_dict['timestamp'],
                    'article_count': row_dict.get('news_count', 0),
                    'avg_sentiment': row_dict.get('sentiment_score', 0),
                    'sentiment_std': 0.1,  # Default
                    'market_regime': 'neutral'  # Default
                }
                
                batch.append(sentiment_row)
                
                if len(batch) >= 1000:
                    self._insert_sentiment_batch(timescale_cursor, batch)
                    total_rows += len(batch)
                    batch = []
            
            # Insert remaining
            if batch:
                self._insert_sentiment_batch(timescale_cursor, batch)
                total_rows += len(batch)
            
            self.timescale_conn.commit()
            logger.info(f"Migrated {total_rows} sentiment rows successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error migrating sentiment data: {e}")
            self.timescale_conn.rollback()
            return False
    
    def _insert_sentiment_batch(self, cursor, batch):
        """Insert batch of sentiment data"""
        insert_sql = """
        INSERT INTO sentiment_hourly (timestamp, article_count, avg_sentiment, sentiment_std, market_regime)
        VALUES (%(timestamp)s, %(article_count)s, %(avg_sentiment)s, %(sentiment_std)s, %(market_regime)s)
        ON CONFLICT DO NOTHING
        """
        
        cursor.executemany(insert_sql, batch)
    
    def verify_migration(self):
        """Verify migration was successful"""
        try:
            logger.info("Verifying migration...")
            
            timescale_cursor = self.timescale_conn.cursor()
            
            # Check row counts
            tables_to_check = [
                'ohlcv',
                'model_predictions',
                'sentiment_hourly'
            ]
            
            for table in tables_to_check:
                timescale_cursor.execute(f"SELECT COUNT(*) FROM {table}")
                count = timescale_cursor.fetchone()[0]
                logger.info(f"{table}: {count} rows")
            
            # Check data quality
            timescale_cursor.execute("""
                SELECT symbol, COUNT(*), MIN(timestamp), MAX(timestamp)
                FROM ohlcv 
                GROUP BY symbol
                ORDER BY symbol
            """)
            
            logger.info("OHLCV data summary:")
            for row in timescale_cursor.fetchall():
                logger.info(f"  {row[0]}: {row[1]} rows, {row[2]} to {row[3]}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error verifying migration: {e}")
            return False
    
    def create_views(self):
        """Create useful views for analysis"""
        try:
            cursor = self.timescale_conn.cursor()
            
            views_sql = """
            -- Latest prices view
            CREATE OR REPLACE VIEW latest_prices AS
            SELECT DISTINCT ON (symbol) 
                symbol, close as price, timestamp, volume
            FROM ohlcv 
            ORDER BY symbol, timestamp DESC;
            
            -- Daily portfolio performance
            CREATE OR REPLACE VIEW daily_portfolio_stats AS
            SELECT 
                DATE(timestamp) as date,
                MAX(total_value) as high_value,
                MIN(total_value) as low_value,
                (SELECT total_value FROM portfolio_state p2 
                 WHERE DATE(p2.timestamp) = DATE(portfolio_state.timestamp)
                 ORDER BY p2.timestamp LIMIT 1) as open_value,
                (SELECT total_value FROM portfolio_state p2 
                 WHERE DATE(p2.timestamp) = DATE(portfolio_state.timestamp)
                 ORDER BY p2.timestamp DESC LIMIT 1) as close_value,
                AVG(daily_pnl) as avg_daily_pnl,
                MAX(max_drawdown) as max_drawdown
            FROM portfolio_state
            GROUP BY DATE(timestamp)
            ORDER BY date DESC;
            
            -- Hourly trading volume by symbol
            CREATE OR REPLACE VIEW hourly_volume AS
            SELECT 
                DATE_TRUNC('hour', timestamp) as hour,
                symbol,
                SUM(volume) as total_volume,
                COUNT(*) as trade_count,
                AVG(close) as avg_price
            FROM ohlcv
            GROUP BY DATE_TRUNC('hour', timestamp), symbol
            ORDER BY hour DESC, symbol;
            
            -- Recent sentiment trends
            CREATE OR REPLACE VIEW recent_sentiment AS
            SELECT 
                timestamp,
                avg_sentiment,
                market_regime,
                article_count,
                LAG(avg_sentiment) OVER (ORDER BY timestamp) as prev_sentiment
            FROM sentiment_hourly
            WHERE timestamp >= NOW() - INTERVAL '7 days'
            ORDER BY timestamp DESC;
            """
            
            cursor.execute(views_sql)
            self.timescale_conn.commit()
            logger.info("Created analysis views")
            return True
            
        except Exception as e:
            logger.error(f"Error creating views: {e}")
            return False
    
    def migrate(self, verify_only: bool = False):
        """Run complete migration"""
        if not self.connect_databases():
            return False
        
        try:
            if not verify_only:
                # Setup schema
                if not self.setup_timescale_schema():
                    return False
                
                # Migrate data
                if not self.migrate_ohlcv_data():
                    return False
                
                if not self.migrate_predictions_data():
                    return False
                
                if not self.migrate_sentiment_data():
                    return False
                
                # Create views
                if not self.create_views():
                    return False
            
            # Verify migration
            if not self.verify_migration():
                return False
            
            logger.info("Migration completed successfully!")
            return True
            
        finally:
            if self.sqlite_conn:
                self.sqlite_conn.close()
            if self.timescale_conn:
                self.timescale_conn.close()


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='Migrate TickerML data to TimescaleDB')
    parser.add_argument('--config', help='Path to TimescaleDB config file')
    parser.add_argument('--verify-only', action='store_true', help='Only verify existing migration')
    
    args = parser.parse_args()
    
    migration = TimescaleMigration(args.config)
    
    success = migration.migrate(verify_only=args.verify_only)
    
    if not success:
        sys.exit(1)
    
    logger.info("Migration script completed successfully!")


if __name__ == "__main__":
    main()