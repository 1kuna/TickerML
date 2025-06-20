#!/usr/bin/env python3
"""
Real-time trade stream collection for TickerML.
Collects individual trade data (price, volume, side) from Binance.US WebSocket.
"""

import asyncio
import json
import logging
import signal
import sqlite3
import sys
import time
from datetime import datetime
from typing import Dict, List, Optional

import aiohttp
import pandas as pd
import yaml

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/trade_stream.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class TradeStreamCollector:
    """Collects real-time trade data from Binance.US WebSocket streams."""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """Initialize the trade stream collector."""
        self.config = self._load_config(config_path)
        self.session: Optional[aiohttp.ClientSession] = None
        self.websocket: Optional[aiohttp.ClientWebSocketResponse] = None
        self.db_path = self.config.get('database', {}).get('path', 'data/db/crypto_data.db')
        self.symbols = self.config.get('binance', {}).get('symbols', ['BTCUSDT', 'ETHUSDT'])
        self.base_url = 'wss://stream.binance.us:9443/ws/'
        self.running = False
        self.trade_buffer: List[Dict] = []
        self.buffer_size = 100
        self.last_flush_time = time.time()
        self.flush_interval = 5.0  # Flush every 5 seconds
        
        # Initialize database
        self._init_database()
        
        # Set up signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from YAML file."""
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            logger.warning(f"Config file {config_path} not found, using defaults")
            return {
                'binance': {
                    'symbols': ['BTCUSDT', 'ETHUSDT'],
                    'base_url': 'https://api.binance.us'
                },
                'database': {
                    'path': 'data/db/crypto_data.db'
                }
            }
    
    def _init_database(self):
        """Initialize the database with trades table."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Create trades table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS trades (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp REAL NOT NULL,
                    symbol TEXT NOT NULL,
                    trade_id INTEGER NOT NULL,
                    price REAL NOT NULL,
                    quantity REAL NOT NULL,
                    is_buyer_maker BOOLEAN NOT NULL,
                    event_time REAL NOT NULL,
                    UNIQUE(symbol, trade_id)
                )
            ''')
            
            # Create index for efficient queries
            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_trades_symbol_timestamp 
                ON trades(symbol, timestamp)
            ''')
            
            conn.commit()
            conn.close()
            logger.info("Database initialized successfully")
            
        except Exception as e:
            logger.error(f"Database initialization failed: {e}")
            raise
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully."""
        logger.info(f"Received signal {signum}, shutting down...")
        self.running = False
    
    async def _flush_trades_to_db(self):
        """Flush accumulated trades to database."""
        if not self.trade_buffer:
            return
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Prepare batch insert
            trade_data = [
                (
                    trade['timestamp'],
                    trade['symbol'],
                    trade['trade_id'],
                    trade['price'],
                    trade['quantity'],
                    trade['is_buyer_maker'],
                    trade['event_time']
                )
                for trade in self.trade_buffer
            ]
            
            cursor.executemany('''
                INSERT OR IGNORE INTO trades 
                (timestamp, symbol, trade_id, price, quantity, is_buyer_maker, event_time)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', trade_data)
            
            conn.commit()
            conn.close()
            
            logger.info(f"Flushed {len(self.trade_buffer)} trades to database")
            self.trade_buffer.clear()
            self.last_flush_time = time.time()
            
        except Exception as e:
            logger.error(f"Failed to flush trades to database: {e}")
    
    def _process_trade_message(self, data: Dict):
        """Process individual trade message from WebSocket."""
        try:
            # Parse trade data
            trade = {
                'timestamp': time.time(),
                'symbol': data['s'],
                'trade_id': int(data['t']),
                'price': float(data['p']),
                'quantity': float(data['q']),
                'is_buyer_maker': data['m'],  # True if buyer is market maker
                'event_time': float(data['T']) / 1000.0  # Convert to seconds
            }
            
            self.trade_buffer.append(trade)
            
            # Flush if buffer is full or time interval exceeded
            current_time = time.time()
            if (len(self.trade_buffer) >= self.buffer_size or 
                current_time - self.last_flush_time >= self.flush_interval):
                asyncio.create_task(self._flush_trades_to_db())
            
        except (KeyError, ValueError) as e:
            logger.error(f"Failed to process trade message: {e}")
    
    async def _connect_websocket(self):
        """Connect to Binance.US WebSocket trade streams."""
        try:
            # Create stream names for all symbols
            streams = [f"{symbol.lower()}@trade" for symbol in self.symbols]
            stream_url = self.base_url + "/".join(streams)
            
            logger.info(f"Connecting to WebSocket: {stream_url}")
            
            self.session = aiohttp.ClientSession()
            self.websocket = await self.session.ws_connect(stream_url)
            
            logger.info("WebSocket connection established")
            return True
            
        except Exception as e:
            logger.error(f"Failed to connect to WebSocket: {e}")
            return False
    
    async def _handle_websocket_messages(self):
        """Handle incoming WebSocket messages."""
        try:
            async for msg in self.websocket:
                if msg.type == aiohttp.WSMsgType.TEXT:
                    try:
                        data = json.loads(msg.data)
                        
                        # Handle single stream or combined stream format
                        if 'stream' in data:
                            # Combined stream format
                            self._process_trade_message(data['data'])
                        else:
                            # Single stream format
                            self._process_trade_message(data)
                            
                    except json.JSONDecodeError as e:
                        logger.error(f"Failed to parse JSON message: {e}")
                        
                elif msg.type == aiohttp.WSMsgType.ERROR:
                    logger.error(f"WebSocket error: {self.websocket.exception()}")
                    break
                    
                elif msg.type == aiohttp.WSMsgType.CLOSE:
                    logger.info("WebSocket connection closed")
                    break
                    
        except Exception as e:
            logger.error(f"Error handling WebSocket messages: {e}")
    
    async def _reconnect_websocket(self):
        """Reconnect WebSocket with exponential backoff."""
        reconnect_delay = 1
        max_delay = 60
        
        while self.running:
            try:
                logger.info(f"Attempting to reconnect in {reconnect_delay} seconds...")
                await asyncio.sleep(reconnect_delay)
                
                if await self._connect_websocket():
                    logger.info("WebSocket reconnected successfully")
                    return True
                    
                # Exponential backoff
                reconnect_delay = min(reconnect_delay * 2, max_delay)
                
            except Exception as e:
                logger.error(f"Reconnection failed: {e}")
                
        return False
    
    async def start_collection(self):
        """Start the trade stream collection."""
        logger.info("Starting trade stream collection...")
        self.running = True
        
        while self.running:
            try:
                # Connect to WebSocket
                if not await self._connect_websocket():
                    logger.error("Failed to establish WebSocket connection")
                    await asyncio.sleep(5)
                    continue
                
                # Handle messages
                await self._handle_websocket_messages()
                
                # Connection lost, attempt to reconnect
                if self.running:
                    await self._cleanup_connection()
                    if not await self._reconnect_websocket():
                        break
                        
            except KeyboardInterrupt:
                logger.info("Received keyboard interrupt, shutting down...")
                break
            except Exception as e:
                logger.error(f"Unexpected error in collection loop: {e}")
                await asyncio.sleep(5)
        
        # Final cleanup
        await self._cleanup_connection()
        
        # Flush any remaining trades
        if self.trade_buffer:
            await self._flush_trades_to_db()
        
        logger.info("Trade stream collection stopped")
    
    async def _cleanup_connection(self):
        """Clean up WebSocket connection and session."""
        try:
            if self.websocket:
                await self.websocket.close()
                self.websocket = None
                
            if self.session:
                await self.session.close()
                self.session = None
                
        except Exception as e:
            logger.error(f"Error cleaning up connection: {e}")
    
    def get_recent_trades(self, symbol: str, limit: int = 100) -> pd.DataFrame:
        """Get recent trades for a symbol."""
        try:
            conn = sqlite3.connect(self.db_path)
            
            query = '''
                SELECT timestamp, symbol, trade_id, price, quantity, is_buyer_maker, event_time
                FROM trades
                WHERE symbol = ?
                ORDER BY timestamp DESC
                LIMIT ?
            '''
            
            df = pd.read_sql_query(query, conn, params=(symbol, limit))
            conn.close()
            
            return df
            
        except Exception as e:
            logger.error(f"Failed to get recent trades: {e}")
            return pd.DataFrame()
    
    def get_trade_statistics(self, symbol: str, hours: int = 24) -> Dict:
        """Get trade statistics for a symbol over the last N hours."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Calculate time threshold
            time_threshold = time.time() - (hours * 3600)
            
            # Get trade statistics
            cursor.execute('''
                SELECT 
                    COUNT(*) as trade_count,
                    SUM(quantity) as total_volume,
                    AVG(price) as avg_price,
                    MIN(price) as min_price,
                    MAX(price) as max_price,
                    SUM(CASE WHEN is_buyer_maker = 0 THEN quantity ELSE 0 END) as buy_volume,
                    SUM(CASE WHEN is_buyer_maker = 1 THEN quantity ELSE 0 END) as sell_volume
                FROM trades
                WHERE symbol = ? AND timestamp >= ?
            ''', (symbol, time_threshold))
            
            result = cursor.fetchone()
            conn.close()
            
            if result and result[0] > 0:
                return {
                    'symbol': symbol,
                    'trade_count': result[0],
                    'total_volume': result[1],
                    'avg_price': result[2],
                    'min_price': result[3],
                    'max_price': result[4],
                    'buy_volume': result[5],
                    'sell_volume': result[6],
                    'buy_sell_ratio': result[5] / result[6] if result[6] > 0 else 0,
                    'hours': hours
                }
            else:
                return {'symbol': symbol, 'trade_count': 0, 'hours': hours}
                
        except Exception as e:
            logger.error(f"Failed to get trade statistics: {e}")
            return {'symbol': symbol, 'error': str(e)}

async def main():
    """Main function to run the trade stream collector."""
    collector = TradeStreamCollector()
    
    try:
        await collector.start_collection()
    except KeyboardInterrupt:
        logger.info("Shutting down trade stream collector...")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
    finally:
        # Ensure cleanup
        await collector._cleanup_connection()

if __name__ == "__main__":
    # Create logs directory if it doesn't exist
    import os
    os.makedirs('logs', exist_ok=True)
    
    # Run the collector
    asyncio.run(main())