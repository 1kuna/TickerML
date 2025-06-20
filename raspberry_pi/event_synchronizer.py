#!/usr/bin/env python3
"""
Event Synchronization System for TickerML.
Synchronizes order book and trade events that arrive separately, replaying them in timestamp order.
Critical for preventing false patterns from incorrect event ordering.
"""

import asyncio
import heapq
import json
import logging
import sqlite3
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Union
from enum import Enum

import numpy as np
import pandas as pd
import yaml

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/event_synchronizer.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class EventType(Enum):
    """Types of market events."""
    ORDER_BOOK = "orderbook"
    TRADE = "trade"
    OHLCV = "ohlcv"

@dataclass
class MarketEvent:
    """Represents a market event with timestamp and data."""
    timestamp: float
    event_type: EventType
    symbol: str
    data: Dict
    event_id: Optional[str] = None
    sequence_number: Optional[int] = None
    
    def __lt__(self, other):
        """For heap ordering by timestamp."""
        return self.timestamp < other.timestamp

class EventSynchronizer:
    """Synchronizes market events by timestamp to prevent false patterns."""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """Initialize the event synchronizer."""
        self.config = self._load_config(config_path)
        self.db_path = self.config.get('database', {}).get('path', 'data/db/crypto_data.db')
        self.symbols = self.config.get('binance', {}).get('symbols', ['BTCUSDT', 'ETHUSDT'])
        
        # Event queues and synchronization
        self.event_queue = []  # Min heap for timestamp ordering
        self.pending_events = defaultdict(deque)  # Per-symbol event buffers
        self.last_processed_time = defaultdict(float)  # Last processed timestamp per symbol
        
        # Synchronization parameters
        self.max_delay_ms = 1000  # Maximum delay to wait for events (1 second)
        self.buffer_size = 1000   # Maximum events in buffer per symbol
        self.replay_window_ms = 5000  # Time window for event replay (5 seconds)
        
        # Statistics
        self.stats = {
            'events_processed': 0,
            'events_reordered': 0,
            'events_dropped': 0,
            'symbols_processed': set(),
            'last_sync_time': time.time()
        }
        
        # Threading
        self.running = False
        self.sync_thread = None
        self.lock = threading.RLock()
        
        # Event callbacks
        self.event_callbacks = {
            EventType.ORDER_BOOK: [],
            EventType.TRADE: [],
            EventType.OHLCV: []
        }
        
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
    
    def register_callback(self, event_type: EventType, callback):
        """Register a callback for specific event types."""
        self.event_callbacks[event_type].append(callback)
        logger.info(f"Registered callback for {event_type.value} events")
    
    def add_event(self, timestamp: float, event_type: EventType, symbol: str, 
                  data: Dict, event_id: Optional[str] = None, 
                  sequence_number: Optional[int] = None):
        """Add an event to the synchronization queue."""
        with self.lock:
            event = MarketEvent(
                timestamp=timestamp,
                event_type=event_type,
                symbol=symbol,
                data=data,
                event_id=event_id,
                sequence_number=sequence_number
            )
            
            # Add to pending events for this symbol
            self.pending_events[symbol].append(event)
            
            # Maintain buffer size limit
            if len(self.pending_events[symbol]) > self.buffer_size:
                dropped_event = self.pending_events[symbol].popleft()
                self.stats['events_dropped'] += 1
                logger.warning(f"Dropped event for {symbol} due to buffer overflow")
            
            # Trigger synchronization
            self._trigger_synchronization()
    
    def _trigger_synchronization(self):
        """Trigger event synchronization."""
        current_time = time.time()
        
        # Check if any events are ready to be processed
        for symbol in self.symbols:
            if symbol not in self.pending_events or not self.pending_events[symbol]:
                continue
            
            # Get the oldest event for this symbol
            oldest_event = self.pending_events[symbol][0]
            event_age_ms = (current_time - oldest_event.timestamp) * 1000
            
            # Process if event is old enough or buffer is full
            if (event_age_ms > self.max_delay_ms or 
                len(self.pending_events[symbol]) >= self.buffer_size):
                self._process_pending_events(symbol)
    
    def _process_pending_events(self, symbol: str):
        """Process pending events for a symbol in timestamp order."""
        if symbol not in self.pending_events or not self.pending_events[symbol]:
            return
        
        # Convert deque to list and sort by timestamp
        events = list(self.pending_events[symbol])
        events.sort(key=lambda e: e.timestamp)
        
        # Check for reordering
        original_order = [e.timestamp for e in self.pending_events[symbol]]
        sorted_order = [e.timestamp for e in events]
        
        if original_order != sorted_order:
            self.stats['events_reordered'] += 1
            logger.debug(f"Reordered {len(events)} events for {symbol}")
        
        # Process events in timestamp order
        for event in events:
            self._dispatch_event(event)
            self.stats['events_processed'] += 1
            self.stats['symbols_processed'].add(symbol)
            self.last_processed_time[symbol] = event.timestamp
        
        # Clear processed events
        self.pending_events[symbol].clear()
    
    def _dispatch_event(self, event: MarketEvent):
        """Dispatch a synchronized event to registered callbacks."""
        try:
            # Call registered callbacks
            for callback in self.event_callbacks[event.event_type]:
                try:
                    callback(event)
                except Exception as e:
                    logger.error(f"Error in callback for {event.event_type.value}: {e}")
            
            # Store event in database
            self._store_synchronized_event(event)
            
        except Exception as e:
            logger.error(f"Error dispatching event: {e}")
    
    def _store_synchronized_event(self, event: MarketEvent):
        """Store synchronized event in database."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Create synchronized events table if it doesn't exist
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS synchronized_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp REAL NOT NULL,
                    event_type TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    data TEXT NOT NULL,
                    event_id TEXT,
                    sequence_number INTEGER,
                    processed_at REAL NOT NULL
                )
            ''')
            
            # Insert event
            cursor.execute('''
                INSERT INTO synchronized_events 
                (timestamp, event_type, symbol, data, event_id, sequence_number, processed_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                event.timestamp,
                event.event_type.value,
                event.symbol,
                json.dumps(event.data),
                event.event_id,
                event.sequence_number,
                time.time()
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Failed to store synchronized event: {e}")
    
    def start_synchronization(self):
        """Start the event synchronization thread."""
        if self.running:
            logger.warning("Synchronization already running")
            return
        
        self.running = True
        self.sync_thread = threading.Thread(target=self._synchronization_loop)
        self.sync_thread.daemon = True
        self.sync_thread.start()
        
        logger.info("Event synchronization started")
    
    def stop_synchronization(self):
        """Stop the event synchronization thread."""
        self.running = False
        
        if self.sync_thread:
            self.sync_thread.join(timeout=5.0)
        
        # Process any remaining events
        for symbol in self.symbols:
            self._process_pending_events(symbol)
        
        logger.info("Event synchronization stopped")
    
    def _synchronization_loop(self):
        """Main synchronization loop."""
        while self.running:
            try:
                with self.lock:
                    # Process events for all symbols
                    for symbol in self.symbols:
                        self._process_pending_events(symbol)
                    
                    # Update statistics
                    self.stats['last_sync_time'] = time.time()
                
                # Sleep briefly to avoid busy waiting
                time.sleep(0.01)  # 10ms
                
            except Exception as e:
                logger.error(f"Error in synchronization loop: {e}")
                time.sleep(1.0)
    
    def replay_historical_events(self, symbol: str, start_time: float, 
                                end_time: float) -> List[MarketEvent]:
        """Replay historical events in timestamp order."""
        try:
            conn = sqlite3.connect(self.db_path)
            
            # Get historical events from all relevant tables
            events = []
            
            # Get order book events (if table exists)
            try:
                orderbook_query = '''
                    SELECT timestamp, 'orderbook' as event_type, symbol, 
                           json_object(
                               'bids', bids,
                               'asks', asks,
                               'mid_price', mid_price,
                               'spread', spread
                           ) as data
                    FROM order_books
                    WHERE symbol = ? AND timestamp BETWEEN ? AND ?
                '''
                orderbook_df = pd.read_sql_query(orderbook_query, conn, 
                                                params=(symbol, start_time, end_time))
                
                for _, row in orderbook_df.iterrows():
                    events.append(MarketEvent(
                        timestamp=row['timestamp'],
                        event_type=EventType.ORDER_BOOK,
                        symbol=row['symbol'],
                        data=json.loads(row['data'])
                    ))
            except:
                pass  # Table might not exist
            
            # Get trade events (if table exists)
            try:
                trades_query = '''
                    SELECT timestamp, 'trade' as event_type, symbol,
                           json_object(
                               'trade_id', trade_id,
                               'price', price,
                               'quantity', quantity,
                               'is_buyer_maker', is_buyer_maker
                           ) as data
                    FROM trades
                    WHERE symbol = ? AND timestamp BETWEEN ? AND ?
                '''
                trades_df = pd.read_sql_query(trades_query, conn, 
                                            params=(symbol, start_time, end_time))
                
                for _, row in trades_df.iterrows():
                    events.append(MarketEvent(
                        timestamp=row['timestamp'],
                        event_type=EventType.TRADE,
                        symbol=row['symbol'],
                        data=json.loads(row['data'])
                    ))
            except:
                pass  # Table might not exist
            
            # Get OHLCV events
            try:
                ohlcv_query = '''
                    SELECT timestamp, 'ohlcv' as event_type, symbol,
                           json_object(
                               'open', open,
                               'high', high,
                               'low', low,
                               'close', close,
                               'volume', volume
                           ) as data
                    FROM ohlcv
                    WHERE symbol = ? AND timestamp BETWEEN ? AND ?
                '''
                ohlcv_df = pd.read_sql_query(ohlcv_query, conn, 
                                           params=(symbol, start_time, end_time))
                
                for _, row in ohlcv_df.iterrows():
                    events.append(MarketEvent(
                        timestamp=row['timestamp'],
                        event_type=EventType.OHLCV,
                        symbol=row['symbol'],
                        data=json.loads(row['data'])
                    ))
            except:
                pass  # Table might not exist
            
            conn.close()
            
            # Sort events by timestamp
            events.sort(key=lambda e: e.timestamp)
            
            logger.info(f"Retrieved {len(events)} historical events for {symbol}")
            return events
            
        except Exception as e:
            logger.error(f"Failed to replay historical events: {e}")
            return []
    
    def get_synchronization_stats(self) -> Dict:
        """Get synchronization statistics."""
        with self.lock:
            return {
                'events_processed': self.stats['events_processed'],
                'events_reordered': self.stats['events_reordered'],
                'events_dropped': self.stats['events_dropped'],
                'symbols_processed': list(self.stats['symbols_processed']),
                'pending_events': {
                    symbol: len(queue) for symbol, queue in self.pending_events.items()
                },
                'last_sync_time': datetime.fromtimestamp(self.stats['last_sync_time']).isoformat(),
                'is_running': self.running
            }
    
    def validate_event_ordering(self, symbol: str, hours: int = 1) -> Dict:
        """Validate that events are properly ordered in the database."""
        try:
            conn = sqlite3.connect(self.db_path)
            
            time_threshold = time.time() - (hours * 3600)
            
            # Check synchronized events ordering
            query = '''
                SELECT timestamp, event_type, COUNT(*) as count
                FROM synchronized_events
                WHERE symbol = ? AND timestamp >= ?
                ORDER BY timestamp ASC
            '''
            
            df = pd.read_sql_query(query, conn, params=(symbol, time_threshold))
            conn.close()
            
            if df.empty:
                return {
                    'symbol': symbol,
                    'valid': True,
                    'message': 'No synchronized events found',
                    'event_count': 0
                }
            
            # Check for timestamp ordering
            timestamps = df['timestamp'].values
            is_ordered = np.all(timestamps[1:] >= timestamps[:-1])
            
            # Calculate statistics
            total_events = df['count'].sum()
            event_types = df['event_type'].value_counts().to_dict()
            
            return {
                'symbol': symbol,
                'valid': is_ordered,
                'event_count': total_events,
                'event_types': event_types,
                'time_span_seconds': timestamps[-1] - timestamps[0] if len(timestamps) > 1 else 0,
                'ordering_violations': 0 if is_ordered else np.sum(timestamps[1:] < timestamps[:-1])
            }
            
        except Exception as e:
            logger.error(f"Failed to validate event ordering for {symbol}: {e}")
            return {
                'symbol': symbol,
                'valid': False,
                'error': str(e)
            }

# Example usage and testing functions
def test_event_synchronizer():
    """Test the event synchronizer with sample data."""
    synchronizer = EventSynchronizer()
    
    # Test callback function
    def test_callback(event: MarketEvent):
        print(f"Received {event.event_type.value} event for {event.symbol} at {event.timestamp}")
    
    # Register callbacks
    synchronizer.register_callback(EventType.ORDER_BOOK, test_callback)
    synchronizer.register_callback(EventType.TRADE, test_callback)
    
    # Start synchronization
    synchronizer.start_synchronization()
    
    # Add test events (out of order)
    current_time = time.time()
    
    # Add events out of chronological order to test synchronization
    synchronizer.add_event(
        timestamp=current_time + 2,
        event_type=EventType.TRADE,
        symbol='BTCUSDT',
        data={'price': 50000, 'quantity': 1.0, 'side': 'buy'}
    )
    
    synchronizer.add_event(
        timestamp=current_time + 1,
        event_type=EventType.ORDER_BOOK,
        symbol='BTCUSDT',
        data={'bids': [[49900, 10]], 'asks': [[50100, 5]]}
    )
    
    synchronizer.add_event(
        timestamp=current_time + 3,
        event_type=EventType.TRADE,
        symbol='BTCUSDT',
        data={'price': 50050, 'quantity': 0.5, 'side': 'sell'}
    )
    
    # Wait for processing
    time.sleep(2)
    
    # Print statistics
    stats = synchronizer.get_synchronization_stats()
    print(f"Synchronization stats: {stats}")
    
    # Stop synchronization
    synchronizer.stop_synchronization()

def main():
    """Main function for testing."""
    test_event_synchronizer()

if __name__ == "__main__":
    # Create logs directory if it doesn't exist
    import os
    os.makedirs('logs', exist_ok=True)
    
    main()