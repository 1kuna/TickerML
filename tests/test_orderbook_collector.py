#!/usr/bin/env python3
"""
Comprehensive test suite for WebSocket order book collection
Tests real-time data collection, microstructure features, and data quality
"""

import unittest
import sqlite3
import time
import json
import threading
from unittest.mock import Mock, patch, MagicMock
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from raspberry_pi.orderbook_collector import BinanceOrderBookCollector, OrderBookLevel, OrderBookSnapshot
import logging

# Configure test logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TestOrderBookCollector(unittest.TestCase):
    """Test suite for order book collector"""
    
    def setUp(self):
        """Set up test environment"""
        self.test_db_path = "test_orderbook.db"
        self.test_symbols = ['BTCUSD', 'ETHUSD']
        
        # Clean up any existing test database
        try:
            import os
            if os.path.exists(self.test_db_path):
                os.remove(self.test_db_path)
        except:
            pass
        
        # Initialize collector with test database
        self.collector = BinanceOrderBookCollector()
        # Override the database path for testing
        self.collector.db_path = Path(self.test_db_path)
        self.collector.symbols = self.test_symbols
        # Re-initialize database with test path
        self.collector._init_database()
    
    def tearDown(self):
        """Clean up test environment"""
        try:
            self.collector.stop()
            import os
            if os.path.exists(self.test_db_path):
                os.remove(self.test_db_path)
        except:
            pass
    
    def test_database_initialization(self):
        """Test database table creation"""
        conn = sqlite3.connect(self.test_db_path)
        cursor = conn.cursor()
        
        # Check if order_books table exists
        cursor.execute("""
            SELECT name FROM sqlite_master 
            WHERE type='table' AND name='order_books'
        """)
        
        result = cursor.fetchone()
        self.assertIsNotNone(result, "order_books table should be created")
        
        # Check table schema
        cursor.execute("PRAGMA table_info(order_books)")
        columns = [row[1] for row in cursor.fetchall()]
        
        expected_columns = [
            'timestamp', 'symbol', 'bids', 'asks', 'mid_price', 
            'spread_bps', 'imbalance', 'microprice', 'depth_5', 'depth_10'
        ]
        
        for col in expected_columns:
            self.assertIn(col, columns, f"Column {col} should exist in order_books table")
        
        conn.close()
    
    def test_order_book_level_creation(self):
        """Test OrderBookLevel data structure"""
        timestamp = int(time.time() * 1_000_000)
        level = OrderBookLevel(100.5, 1.5, timestamp)
        
        self.assertEqual(level.price, 100.5)
        self.assertEqual(level.quantity, 1.5)
        self.assertEqual(level.timestamp, timestamp)
    
    def test_order_book_snapshot_creation(self):
        """Test OrderBookSnapshot creation with microstructure features"""
        timestamp = int(time.time() * 1_000_000)
        
        # Create test bids and asks
        bids = [
            OrderBookLevel(100.0, 1.0, timestamp),
            OrderBookLevel(99.9, 1.5, timestamp),
            OrderBookLevel(99.8, 2.0, timestamp)
        ]
        
        asks = [
            OrderBookLevel(100.1, 1.2, timestamp),
            OrderBookLevel(100.2, 1.8, timestamp),
            OrderBookLevel(100.3, 2.2, timestamp)
        ]
        
        snapshot = OrderBookSnapshot(
            symbol='BTCUSD',
            timestamp=timestamp,
            bids=bids,
            asks=asks,
            mid_price=100.05,
            spread_bps=10.0,
            imbalance=0.1,
            microprice=100.04,
            depth_5=10.5,
            depth_10=20.0
        )
        
        self.assertEqual(snapshot.symbol, 'BTCUSD')
        self.assertEqual(snapshot.timestamp, timestamp)
        self.assertEqual(len(snapshot.bids), 3)
        self.assertEqual(len(snapshot.asks), 3)
        self.assertEqual(snapshot.mid_price, 100.05)
        self.assertAlmostEqual(snapshot.imbalance, 0.1, places=2)
    
    def test_imbalance_calculation(self):
        """Test order book imbalance calculation"""
        timestamp = int(time.time() * 1_000_000)
        
        # Test balanced order book
        bids = [OrderBookLevel(100.0, 1.0, timestamp) for _ in range(5)]
        asks = [OrderBookLevel(100.1, 1.0, timestamp) for _ in range(5)]
        
        imbalance = self.collector._calculate_imbalance(bids, asks, levels=5)
        self.assertAlmostEqual(imbalance, 0.0, places=2, msg="Balanced order book should have 0 imbalance")
        
        # Test buy pressure (more bid volume)
        bids = [OrderBookLevel(100.0, 2.0, timestamp) for _ in range(5)]
        asks = [OrderBookLevel(100.1, 1.0, timestamp) for _ in range(5)]
        
        imbalance = self.collector._calculate_imbalance(bids, asks, levels=5)
        self.assertGreater(imbalance, 0, "More bid volume should give positive imbalance")
        
        # Test sell pressure (more ask volume)
        bids = [OrderBookLevel(100.0, 1.0, timestamp) for _ in range(5)]
        asks = [OrderBookLevel(100.1, 2.0, timestamp) for _ in range(5)]
        
        imbalance = self.collector._calculate_imbalance(bids, asks, levels=5)
        self.assertLess(imbalance, 0, "More ask volume should give negative imbalance")
    
    def test_microprice_calculation(self):
        """Test microprice calculation"""
        # Test with equal quantities (should equal mid-price)
        microprice = self.collector._calculate_microprice(100.0, 100.2, 1.0, 1.0)
        expected_mid = (100.0 + 100.2) / 2
        self.assertAlmostEqual(microprice, expected_mid, places=2)
        
        # Test with more bid quantity (should be closer to bid)
        microprice = self.collector._calculate_microprice(100.0, 100.2, 2.0, 1.0)
        self.assertLess(microprice, expected_mid, "More bid quantity should pull microprice toward bid")
        
        # Test with more ask quantity (should be closer to ask)
        microprice = self.collector._calculate_microprice(100.0, 100.2, 1.0, 2.0)
        self.assertGreater(microprice, expected_mid, "More ask quantity should pull microprice toward ask")
    
    def test_queue_position_estimation(self):
        """Test queue position estimation for execution simulation"""
        # Store test order book data
        test_data = {
            'timestamp': time.time(),
            'symbol': 'BTCUSD',
            'bids': json.dumps([[100.0, 5.0], [99.9, 3.0], [99.8, 2.0]]),
            'asks': json.dumps([[100.1, 4.0], [100.2, 3.5], [100.3, 2.5]]),
            'mid_price': 100.05,
            'spread_bps': 10.0,
            'imbalance': 0.1,
            'microprice': 100.04,
            'depth_5': 10.0,
            'depth_10': 20.0
        }
        
        # Insert test data
        conn = sqlite3.connect(self.test_db_path)
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO order_books (timestamp, symbol, bids, asks, mid_price, 
                                   spread_bps, imbalance, microprice, depth_5, depth_10)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', tuple(test_data.values()))
        conn.commit()
        conn.close()
        
        # Test queue position estimation
        queue_pos = self.collector.estimate_queue_position('BTCUSD', 'buy', 100.0)
        self.assertIsInstance(queue_pos, int, "Queue position should be an integer")
        self.assertGreaterEqual(queue_pos, 0, "Queue position should be non-negative")
        
        # Test for price not in order book
        queue_pos = self.collector.estimate_queue_position('BTCUSD', 'buy', 99.0)
        self.assertIsNone(queue_pos, "Should return None for price not in order book")
    
    def test_data_storage(self):
        """Test order book data storage"""
        timestamp = time.time()
        
        # Create test snapshot
        bids = [OrderBookLevel(100.0, 1.0, int(timestamp * 1_000_000))]
        asks = [OrderBookLevel(100.1, 1.0, int(timestamp * 1_000_000))]
        
        snapshot = OrderBookSnapshot(
            symbol='BTCUSD',
            timestamp=int(timestamp * 1_000_000),
            bids=bids,
            asks=asks,
            mid_price=100.05,
            spread_bps=10.0,
            imbalance=0.0,
            microprice=100.05,
            depth_5=2.0,
            depth_10=4.0
        )
        
        # Store snapshot
        success = self.collector._store_order_book_snapshot(snapshot)
        self.assertTrue(success, "Should successfully store order book snapshot")
        
        # Verify data was stored
        conn = sqlite3.connect(self.test_db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM order_books WHERE symbol = 'BTCUSD'")
        count = cursor.fetchone()[0]
        conn.close()
        
        self.assertEqual(count, 1, "Should have stored one order book snapshot")
    
    def test_spread_calculation(self):
        """Test spread calculation in basis points"""
        # Test normal spread
        best_bid = 100.0
        best_ask = 100.1
        mid_price = 100.05
        spread = best_ask - best_bid
        spread_bps = (spread / mid_price) * 10000
        
        self.assertAlmostEqual(spread_bps, 99.95, places=1, 
                              msg="Spread calculation should be accurate")
        
        # Test tight spread
        best_bid = 100.00
        best_ask = 100.01
        mid_price = 100.005
        spread = best_ask - best_bid
        spread_bps = (spread / mid_price) * 10000
        
        self.assertAlmostEqual(spread_bps, 10.0, places=1,
                              msg="Tight spread calculation should be accurate")
    
    def test_depth_calculation(self):
        """Test order book depth calculation"""
        timestamp = int(time.time() * 1_000_000)
        
        # Create order book with known volumes
        bids = [
            OrderBookLevel(100.0, 1.0, timestamp),
            OrderBookLevel(99.9, 2.0, timestamp),
            OrderBookLevel(99.8, 3.0, timestamp),
            OrderBookLevel(99.7, 4.0, timestamp),
            OrderBookLevel(99.6, 5.0, timestamp)
        ]
        
        asks = [
            OrderBookLevel(100.1, 1.5, timestamp),
            OrderBookLevel(100.2, 2.5, timestamp),
            OrderBookLevel(100.3, 3.5, timestamp),
            OrderBookLevel(100.4, 4.5, timestamp),
            OrderBookLevel(100.5, 5.5, timestamp)
        ]
        
        # Calculate depth
        depth_5_bids = sum(bid.quantity for bid in bids[:5])
        depth_5_asks = sum(ask.quantity for ask in asks[:5])
        total_depth_5 = depth_5_bids + depth_5_asks
        
        expected_depth_5 = 15.0 + 17.5  # 32.5
        self.assertEqual(total_depth_5, expected_depth_5, 
                        "Depth calculation should sum correctly")
    
    @patch('websocket.WebSocketApp')
    def test_websocket_connection(self, mock_websocket):
        """Test WebSocket connection setup"""
        mock_ws = Mock()
        mock_websocket.return_value = mock_ws
        
        # Test connection
        self.collector._connect()
        
        # Verify WebSocket was created with correct URL
        mock_websocket.assert_called_once()
        args, kwargs = mock_websocket.call_args
        self.assertIn('wss://stream.binance.us:9443/ws', args[0])
        
        # Verify callback functions are set
        self.assertIn('on_open', kwargs)
        self.assertIn('on_message', kwargs)
        self.assertIn('on_error', kwargs)
        self.assertIn('on_close', kwargs)
    
    def test_subscription_message_format(self):
        """Test WebSocket subscription message format"""
        subscription = self.collector._create_subscription_message()
        message = json.loads(subscription)
        
        # Verify message structure
        self.assertIn('method', message)
        self.assertIn('params', message)
        self.assertIn('id', message)
        
        self.assertEqual(message['method'], 'SUBSCRIBE')
        self.assertIsInstance(message['params'], list)
        
        # Verify all symbols are included
        streams = message['params']
        for symbol in self.test_symbols:
            symbol_stream = f"{symbol.lower()}@depth20@100ms"
            self.assertIn(symbol_stream, streams)
    
    def test_error_handling(self):
        """Test error handling in various scenarios"""
        # Test invalid order book message
        invalid_message = {"invalid": "data"}
        snapshot = self.collector._parse_order_book_message(invalid_message)
        self.assertIsNone(snapshot, "Should handle invalid message gracefully")
        
        # Test empty bids/asks
        empty_message = {
            "data": {
                "s": "BTCUSD",
                "b": [],
                "a": []
            }
        }
        snapshot = self.collector._parse_order_book_message(empty_message)
        self.assertIsNone(snapshot, "Should handle empty order book gracefully")
        
        # Test imbalance calculation with zero volume
        result = self.collector._calculate_imbalance([], [], levels=5)
        self.assertEqual(result, 0.0, "Should handle empty order book levels")

class TestOrderBookIntegration(unittest.TestCase):
    """Integration tests for order book collector"""
    
    def setUp(self):
        """Set up integration test environment"""
        self.test_db_path = "test_integration.db"
        
        # Clean up any existing test database
        try:
            import os
            if os.path.exists(self.test_db_path):
                os.remove(self.test_db_path)
        except:
            pass
    
    def tearDown(self):
        """Clean up integration test environment"""
        try:
            import os
            if os.path.exists(self.test_db_path):
                os.remove(self.test_db_path)
        except:
            pass
    
    def test_data_flow_integration(self):
        """Test complete data flow from WebSocket to database"""
        collector = BinanceOrderBookCollector()
        collector.db_path = Path(self.test_db_path)
        collector.symbols = ['BTCUSD']
        collector._init_database()
        
        # Create mock order book message
        mock_message = {
            "data": {
                "s": "BTCUSD",
                "b": [["100.0", "1.0"], ["99.9", "2.0"]],
                "a": [["100.1", "1.5"], ["100.2", "2.5"]]
            }
        }
        
        # Process message
        snapshot = collector._parse_order_book_message(mock_message)
        self.assertIsNotNone(snapshot, "Should create snapshot from mock message")
        
        # Store snapshot
        success = collector._store_order_book_snapshot(snapshot)
        self.assertTrue(success, "Should store snapshot successfully")
        
        # Verify data in database
        conn = sqlite3.connect(self.test_db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM order_books WHERE symbol = 'BTCUSD'")
        row = cursor.fetchone()
        conn.close()
        
        self.assertIsNotNone(row, "Should find stored data in database")
        
        # Verify microstructure features were calculated
        stored_imbalance = row[6]  # imbalance column
        stored_microprice = row[7]  # microprice column
        
        self.assertIsNotNone(stored_imbalance, "Should have calculated imbalance")
        self.assertIsNotNone(stored_microprice, "Should have calculated microprice")

def run_tests():
    """Run all order book collector tests"""
    logger.info("Starting order book collector test suite")
    
    # Create test suite
    suite = unittest.TestSuite()
    
    # Add unit tests
    suite.addTest(unittest.makeSuite(TestOrderBookCollector))
    suite.addTest(unittest.makeSuite(TestOrderBookIntegration))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    tests_run = result.testsRun
    failures = len(result.failures)
    errors = len(result.errors)
    
    logger.info(f"Order Book Collector Tests Complete:")
    logger.info(f"  Tests Run: {tests_run}")
    logger.info(f"  Failures: {failures}")
    logger.info(f"  Errors: {errors}")
    logger.info(f"  Success Rate: {((tests_run - failures - errors) / tests_run * 100):.1f}%")
    
    return result.wasSuccessful()

if __name__ == "__main__":
    success = run_tests()
    exit(0 if success else 1)