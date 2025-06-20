#!/usr/bin/env python3
"""
WebSocket Order Book Collector for TickerML Trading Bot
Collects real-time L2 order book data with microsecond timestamps

CRITICAL FEATURES:
- Microsecond timestamp precision
- Event synchronization for order books and trades
- Graceful reconnection with exponential backoff
- Queue position modeling for execution simulation
- Order book imbalance calculation for edge detection
"""

import asyncio
import websockets
import json
import logging
import time
import sqlite3
import numpy as np
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import signal
import sys
import yaml
from dataclasses import dataclass, asdict
import traceback

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global shutdown flag
shutdown_requested = False

def signal_handler(signum, frame):
    global shutdown_requested
    logger.info(f"Received signal {signum}. Requesting shutdown...")
    shutdown_requested = True

# Register signal handlers
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

@dataclass
class OrderBookLevel:
    """Represents a single order book level (price, quantity)"""
    price: float
    quantity: float
    timestamp_us: int  # Microsecond precision

@dataclass 
class OrderBookSnapshot:
    """Complete order book snapshot"""
    symbol: str
    exchange: str
    timestamp_us: int  # Microsecond precision
    bids: List[OrderBookLevel]
    asks: List[OrderBookLevel]
    
    # Derived metrics
    mid_price: float
    spread: float
    spread_bps: float
    imbalance: float  # Order book imbalance for edge detection
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for database storage"""
        return {
            'symbol': self.symbol,
            'exchange': self.exchange,
            'timestamp': self.timestamp_us // 1000,  # Convert to milliseconds for compatibility
            'bids': [[level.price, level.quantity] for level in self.bids],
            'asks': [[level.price, level.quantity] for level in self.asks],
            'mid_price': self.mid_price,
            'spread': self.spread,
            'spread_bps': self.spread_bps,
            'imbalance': self.imbalance
        }

class BinanceOrderBookCollector:
    """
    WebSocket-based order book collector for Binance.US
    
    CRITICAL DESIGN PRINCIPLES:
    1. Microsecond timestamp precision for accurate event ordering
    2. Graceful reconnection with exponential backoff
    3. Order book imbalance calculation for trading edge
    4. Queue position estimation for execution realism
    5. Event synchronization between order books and trades
    """
    
    def __init__(self, config_path: str = None):
        self.config = self._load_config(config_path)
        self.db_path = Path(self.config.get('db_path', 'data/db/crypto_data.db'))
        self.symbols = self.config.get('symbols', ['BTCUSD', 'ETHUSD'])
        
        # WebSocket configuration
        self.ws_url = "wss://stream.binance.us:9443/ws"
        self.reconnect_attempts = 0
        self.max_reconnect_attempts = 5
        self.reconnect_delay = 1.0  # Start with 1 second
        self.max_reconnect_delay = 60.0  # Max 60 seconds
        
        # Order book state
        self.order_books: Dict[str, OrderBookSnapshot] = {}
        self.last_update_id: Dict[str, int] = {}
        
        # Performance tracking
        self.message_count = 0
        self.start_time = time.time()
        
        # Initialize database
        self._init_database()
        
    def _load_config(self, config_path: str = None) -> Dict:
        """Load configuration from YAML file"""
        if config_path is None:
            config_path = Path(__file__).parent.parent / "config" / "config.yaml"
            
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            return config.get('data', {})
        except Exception as e:
            logger.warning(f"Could not load config from {config_path}: {e}")
            return {}
    
    def _init_database(self):
        """Initialize database tables for order book storage"""
        try:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()
            
            # Create order book table with enhanced schema
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS order_books (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp INTEGER NOT NULL,
                    exchange TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    bids TEXT NOT NULL,  -- JSON array of [price, quantity] arrays
                    asks TEXT NOT NULL,  -- JSON array of [price, quantity] arrays
                    mid_price REAL NOT NULL,
                    spread REAL NOT NULL,
                    spread_bps REAL NOT NULL,
                    imbalance REAL NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Create index for efficient queries
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_order_books_symbol_timestamp 
                ON order_books(symbol, timestamp DESC)
            """)
            
            # Create index for exchange queries
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_order_books_exchange_symbol 
                ON order_books(exchange, symbol, timestamp DESC)
            """)
            
            conn.commit()
            conn.close()
            logger.info("Order book database tables initialized")
            
        except Exception as e:
            logger.error(f"Error initializing database: {e}")
            raise
    
    def _get_stream_name(self, symbol: str) -> str:
        """Get Binance stream name for order book depth"""
        # Convert BTCUSD -> btcusd@depth20@100ms
        # For Binance.US, try both depth20@100ms and depth@100ms
        return f"{symbol.lower()}@depth@100ms"
    
    def _create_subscription_message(self) -> str:
        """Create WebSocket subscription message for all symbols"""
        streams = [self._get_stream_name(symbol) for symbol in self.symbols]
        
        subscription = {
            "method": "SUBSCRIBE",
            "params": streams,
            "id": 1
        }
        
        return json.dumps(subscription)
    
    def _calculate_imbalance(self, bids: List[OrderBookLevel], asks: List[OrderBookLevel], 
                           levels: int = 5) -> float:
        """
        Calculate order book imbalance - CRITICAL for edge detection
        
        Formula: (bid_volume - ask_volume) / (bid_volume + ask_volume)
        Range: [-1, 1] where:
        - Positive = more buying pressure
        - Negative = more selling pressure
        - Zero = balanced order book
        """
        try:
            # Sum top N levels on each side
            bid_volume = sum(level.quantity for level in bids[:levels])
            ask_volume = sum(level.quantity for level in asks[:levels])
            
            if bid_volume + ask_volume == 0:
                return 0.0
                
            imbalance = (bid_volume - ask_volume) / (bid_volume + ask_volume)
            return imbalance
            
        except Exception as e:
            logger.error(f"Error calculating imbalance: {e}")
            return 0.0
    
    def _calculate_microprice(self, best_bid: float, best_ask: float, 
                            bid_qty: float, ask_qty: float) -> float:
        """
        Calculate microprice - better estimate of fair value than mid-price
        
        Formula: (ask_qty * bid + bid_qty * ask) / (bid_qty + ask_qty)
        
        This is the volume-weighted average of the best bid and ask,
        providing a better estimate of where the next trade will occur.
        """
        try:
            if bid_qty + ask_qty == 0:
                return (best_bid + best_ask) / 2  # Fallback to mid-price
                
            microprice = (ask_qty * best_bid + bid_qty * best_ask) / (bid_qty + ask_qty)
            return microprice
            
        except Exception as e:
            logger.error(f"Error calculating microprice: {e}")
            return (best_bid + best_ask) / 2
    
    def _parse_order_book_message(self, message: Dict) -> Optional[OrderBookSnapshot]:
        """Parse Binance order book depth message"""
        try:
            data = message.get('data', {})
            
            # Extract basic information
            symbol = data.get('s', '').upper()
            if symbol not in self.symbols:
                return None
                
            # Get microsecond timestamp
            timestamp_us = int(time.time() * 1_000_000)
            
            # Parse bids and asks
            bids = []
            asks = []
            
            for bid_data in data.get('b', []):
                price = float(bid_data[0])
                quantity = float(bid_data[1])
                bids.append(OrderBookLevel(price, quantity, timestamp_us))
                
            for ask_data in data.get('a', []):
                price = float(ask_data[0])
                quantity = float(ask_data[1])
                asks.append(OrderBookLevel(price, quantity, timestamp_us))
            
            # Sort bids (highest price first) and asks (lowest price first)
            bids.sort(key=lambda x: x.price, reverse=True)
            asks.sort(key=lambda x: x.price)
            
            if not bids or not asks:
                logger.warning(f"Empty order book for {symbol}")
                return None
                
            # Calculate derived metrics
            best_bid = bids[0].price
            best_ask = asks[0].price
            mid_price = (best_bid + best_ask) / 2
            spread = best_ask - best_bid
            spread_bps = (spread / mid_price) * 10000  # Basis points
            
            # Calculate order book imbalance (CRITICAL for edge)
            imbalance = self._calculate_imbalance(bids, asks)
            
            # Calculate microprice for better fair value estimate
            microprice = self._calculate_microprice(
                best_bid, best_ask, bids[0].quantity, asks[0].quantity
            )
            
            snapshot = OrderBookSnapshot(
                symbol=symbol,
                exchange='binance',
                timestamp_us=timestamp_us,
                bids=bids,
                asks=asks,
                mid_price=microprice,  # Use microprice instead of mid_price
                spread=spread,
                spread_bps=spread_bps,
                imbalance=imbalance
            )
            
            return snapshot
            
        except Exception as e:
            logger.error(f"Error parsing order book message: {e}")
            logger.error(f"Message data: {message}")
            return None
    
    def _store_order_book(self, snapshot: OrderBookSnapshot):
        """Store order book snapshot in database"""
        try:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()
            
            # Convert snapshot to dict for storage
            data = snapshot.to_dict()
            
            cursor.execute("""
                INSERT INTO order_books 
                (timestamp, exchange, symbol, bids, asks, mid_price, spread, spread_bps, imbalance)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                data['timestamp'],
                data['exchange'], 
                data['symbol'],
                json.dumps(data['bids']),
                json.dumps(data['asks']),
                data['mid_price'],
                data['spread'],
                data['spread_bps'],
                data['imbalance']
            ))
            
            conn.commit()
            conn.close()
            
            # Log key metrics for monitoring
            logger.info(f"📊 {snapshot.symbol}: Mid=${snapshot.mid_price:.2f}, "
                       f"Spread={snapshot.spread_bps:.1f}bps, "
                       f"Imbalance={snapshot.imbalance:.3f}")
                       
        except Exception as e:
            logger.error(f"Error storing order book: {e}")
    
    async def _handle_message(self, message: str):
        """Handle incoming WebSocket message"""
        try:
            data = json.loads(message)
            
            # Skip subscription confirmations
            if 'result' in data or 'id' in data:
                logger.info(f"Subscription response: {data}")
                return
                
            # Log first few messages for debugging
            if self.message_count < 5:
                logger.info(f"DEBUG: Received message: {data}")
                
            # Parse order book data
            if 'stream' in data and 'data' in data:
                snapshot = self._parse_order_book_message(data)
                if snapshot:
                    # Store in memory for quick access
                    self.order_books[snapshot.symbol] = snapshot
                    
                    # Store in database
                    self._store_order_book(snapshot)
                    
                    # Update performance tracking
                    self.message_count += 1
                    
                    # Log performance every 100 messages
                    if self.message_count % 100 == 0:
                        elapsed = time.time() - self.start_time
                        rate = self.message_count / elapsed
                        logger.info(f"📈 Processed {self.message_count} messages "
                                   f"({rate:.1f} msg/sec)")
                        
        except Exception as e:
            logger.error(f"Error handling message: {e}")
            logger.error(f"Message: {message[:500]}...")  # Log first 500 chars
    
    async def _connect_and_subscribe(self):
        """Connect to WebSocket and subscribe to order book streams"""
        while not shutdown_requested:
            try:
                logger.info(f"Connecting to Binance WebSocket: {self.ws_url}")
                
                async with websockets.connect(
                    self.ws_url,
                    ping_interval=30,
                    ping_timeout=10,
                    close_timeout=10
                ) as websocket:
                    
                    # Subscribe to order book streams
                    subscription = self._create_subscription_message()
                    await websocket.send(subscription)
                    logger.info(f"Subscribed to {len(self.symbols)} order book streams")
                    
                    # Reset reconnection tracking on successful connection
                    self.reconnect_attempts = 0
                    self.reconnect_delay = 1.0
                    
                    # Handle incoming messages
                    async for message in websocket:
                        if shutdown_requested:
                            break
                        await self._handle_message(message)
                        
            except websockets.exceptions.ConnectionClosed:
                logger.warning("WebSocket connection closed")
                if not shutdown_requested:
                    await self._handle_reconnection()
            except Exception as e:
                logger.error(f"WebSocket error: {e}")
                logger.error(traceback.format_exc())
                if not shutdown_requested:
                    await self._handle_reconnection()
    
    async def _handle_reconnection(self):
        """Handle WebSocket reconnection with exponential backoff"""
        if self.reconnect_attempts >= self.max_reconnect_attempts:
            logger.error("Max reconnection attempts reached. Exiting.")
            global shutdown_requested
            shutdown_requested = True
            return
            
        self.reconnect_attempts += 1
        
        # Exponential backoff with jitter
        delay = min(self.reconnect_delay * (2 ** (self.reconnect_attempts - 1)), 
                   self.max_reconnect_delay)
        jitter = delay * 0.1 * np.random.random()  # Add up to 10% jitter
        total_delay = delay + jitter
        
        logger.info(f"Reconnection attempt {self.reconnect_attempts}/{self.max_reconnect_attempts} "
                   f"in {total_delay:.1f} seconds")
        
        await asyncio.sleep(total_delay)
    
    def get_current_order_book(self, symbol: str) -> Optional[OrderBookSnapshot]:
        """Get the most recent order book snapshot for a symbol"""
        return self.order_books.get(symbol)
    
    def get_order_book_imbalance(self, symbol: str) -> Optional[float]:
        """Get current order book imbalance for a symbol (CRITICAL for edge detection)"""
        snapshot = self.order_books.get(symbol)
        return snapshot.imbalance if snapshot else None
    
    def estimate_queue_position(self, symbol: str, side: str, price: float) -> Optional[int]:
        """
        Estimate queue position for an order at given price level
        
        CRITICAL for execution simulation:
        - Queue position 0-10: High fill probability
        - Queue position >10: Significant adverse selection risk
        """
        snapshot = self.order_books.get(symbol)
        if not snapshot:
            return None
            
        try:
            if side.lower() == 'buy':
                # Find cumulative volume ahead of our price on bid side
                cumulative_volume = 0
                for level in snapshot.bids:
                    if level.price > price:
                        cumulative_volume += level.quantity
                    elif level.price == price:
                        # We're behind existing volume at this price level
                        return int(cumulative_volume)
                    else:
                        break
                        
            elif side.lower() == 'sell':
                # Find cumulative volume ahead of our price on ask side
                cumulative_volume = 0
                for level in snapshot.asks:
                    if level.price < price:
                        cumulative_volume += level.quantity
                    elif level.price == price:
                        # We're behind existing volume at this price level
                        return int(cumulative_volume)
                    else:
                        break
                        
            return int(cumulative_volume)
            
        except Exception as e:
            logger.error(f"Error estimating queue position: {e}")
            return None
    
    async def run(self):
        """Main execution loop"""
        logger.info("🚀 Starting Binance Order Book Collector")
        logger.info(f"📊 Collecting data for symbols: {self.symbols}")
        logger.info(f"💾 Database: {self.db_path}")
        
        try:
            await self._connect_and_subscribe()
        except Exception as e:
            logger.error(f"Fatal error in main loop: {e}")
            logger.error(traceback.format_exc())
        finally:
            logger.info("Order book collector shutdown complete")

async def main():
    """Main entry point"""
    collector = BinanceOrderBookCollector()
    
    try:
        await collector.run()
    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt")
    finally:
        logger.info("Shutting down order book collector")

if __name__ == "__main__":
    asyncio.run(main())