#!/usr/bin/env python3
"""
Test suite for newly implemented TickerML components.
Tests: Trade Stream, Event Synchronizer, Data Validator, Funding Monitor, Paper Trader
"""

import asyncio
import json
import logging
import os
import sqlite3
import sys
import time
import unittest
from unittest.mock import Mock, patch, MagicMock

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from raspberry_pi.trade_stream import TradeStreamCollector
from raspberry_pi.event_synchronizer import EventSynchronizer, EventType, MarketEvent
from raspberry_pi.data_validator import DataValidator
from raspberry_pi.funding_monitor import FundingMonitor
from raspberry_pi.paper_trader import PaperTradingEngine, OrderSide, OrderType

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TestTradeStream(unittest.TestCase):
    """Test the Trade Stream Collector."""
    
    def setUp(self):
        """Set up test environment."""
        self.test_db = 'test_crypto_data.db'
        self.collector = TradeStreamCollector()
        self.collector.db_path = self.test_db
        self.collector._init_database()
    
    def tearDown(self):
        """Clean up test environment."""
        if os.path.exists(self.test_db):
            os.remove(self.test_db)
    
    def test_database_initialization(self):
        """Test database initialization."""
        conn = sqlite3.connect(self.test_db)
        cursor = conn.cursor()
        
        # Check if trades table exists
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='trades'")
        result = cursor.fetchone()
        self.assertIsNotNone(result)
        
        conn.close()
    
    def test_process_trade_message(self):
        """Test trade message processing."""
        # Mock trade data
        trade_data = {
            's': 'BTCUSDT',
            't': 12345,
            'p': '50000.0',
            'q': '1.0',
            'm': False,
            'T': int(time.time() * 1000)
        }
        
        # Process the message
        self.collector._process_trade_message(trade_data)
        
        # Check if trade was added to buffer
        self.assertEqual(len(self.collector.trade_buffer), 1)
        
        trade = self.collector.trade_buffer[0]
        self.assertEqual(trade['symbol'], 'BTCUSDT')
        self.assertEqual(trade['price'], 50000.0)
        self.assertEqual(trade['quantity'], 1.0)
    
    def test_get_trade_statistics(self):
        """Test trade statistics calculation."""
        # Insert test data
        conn = sqlite3.connect(self.test_db)
        cursor = conn.cursor()
        
        current_time = time.time()
        test_trades = [
            (current_time, 'BTCUSDT', 1, 50000.0, 1.0, False, current_time),
            (current_time, 'BTCUSDT', 2, 50100.0, 0.5, True, current_time),
        ]
        
        cursor.executemany('''
            INSERT INTO trades (timestamp, symbol, trade_id, price, quantity, is_buyer_maker, event_time)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', test_trades)
        
        conn.commit()
        conn.close()
        
        # Get statistics
        stats = self.collector.get_trade_statistics('BTCUSDT', 1)
        
        self.assertEqual(stats['trade_count'], 2)
        self.assertEqual(stats['total_volume'], 1.5)
        self.assertEqual(stats['buy_volume'], 1.0)
        self.assertEqual(stats['sell_volume'], 0.5)

class TestEventSynchronizer(unittest.TestCase):
    """Test the Event Synchronizer."""
    
    def setUp(self):
        """Set up test environment."""
        self.test_db = 'test_crypto_data.db'
        self.synchronizer = EventSynchronizer()
        self.synchronizer.db_path = self.test_db
        
        # Create test database
        conn = sqlite3.connect(self.test_db)
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS synchronized_events (
                id INTEGER PRIMARY KEY,
                timestamp REAL,
                event_type TEXT,
                symbol TEXT,
                data TEXT,
                event_id TEXT,
                sequence_number INTEGER,
                processed_at REAL
            )
        ''')
        conn.commit()
        conn.close()
    
    def tearDown(self):
        """Clean up test environment."""
        if os.path.exists(self.test_db):
            os.remove(self.test_db)
    
    def test_event_ordering(self):
        """Test event ordering functionality."""
        events_received = []
        
        def test_callback(event):
            events_received.append(event)
        
        # Register callback
        self.synchronizer.register_callback(EventType.TRADE, test_callback)
        
        # Add events out of order
        current_time = time.time()
        
        self.synchronizer.add_event(
            timestamp=current_time + 2,
            event_type=EventType.TRADE,
            symbol='BTCUSDT',
            data={'price': 50000, 'quantity': 1.0}
        )
        
        self.synchronizer.add_event(
            timestamp=current_time + 1,
            event_type=EventType.TRADE,
            symbol='BTCUSDT',
            data={'price': 49900, 'quantity': 0.5}
        )
        
        # Process events
        self.synchronizer._process_pending_events('BTCUSDT')
        
        # Check that events were processed in timestamp order
        self.assertEqual(len(events_received), 2)
        self.assertLess(events_received[0].timestamp, events_received[1].timestamp)
    
    def test_synchronization_stats(self):
        """Test synchronization statistics."""
        # Add some events
        current_time = time.time()
        self.synchronizer.add_event(current_time, EventType.TRADE, 'BTCUSDT', {})
        
        # Get stats
        stats = self.synchronizer.get_synchronization_stats()
        
        self.assertIn('events_processed', stats)
        self.assertIn('pending_events', stats)
        self.assertIn('is_running', stats)

class TestDataValidator(unittest.TestCase):
    """Test the Data Validator."""
    
    def setUp(self):
        """Set up test environment."""
        self.test_db = 'test_crypto_data.db'
        self.validator = DataValidator()
        self.validator.db_path = self.test_db
        
        # Create test database with sample data
        conn = sqlite3.connect(self.test_db)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS ohlcv (
                timestamp REAL,
                symbol TEXT,
                open REAL,
                high REAL,
                low REAL,
                close REAL,
                volume REAL
            )
        ''')
        
        # Insert test data
        current_time = time.time()
        test_data = [
            (current_time - 300, 'BTCUSDT', 50000, 50100, 49900, 50050, 1.5),
            (current_time - 240, 'BTCUSDT', 50050, 50200, 50000, 50150, 2.0),
            (current_time - 180, 'BTCUSDT', 50150, 50300, 50100, 50250, 1.8),
        ]
        
        cursor.executemany('''
            INSERT INTO ohlcv (timestamp, symbol, open, high, low, close, volume)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', test_data)
        
        conn.commit()
        conn.close()
    
    def tearDown(self):
        """Clean up test environment."""
        if os.path.exists(self.test_db):
            os.remove(self.test_db)
    
    def test_ohlcv_validation(self):
        """Test OHLCV data validation."""
        result = self.validator.validate_ohlcv_data('BTCUSDT', 1)
        
        self.assertTrue(result['valid'])
        self.assertEqual(result['data_points'], 3)
        self.assertEqual(len(result['issues']), 0)
    
    def test_invalid_ohlc_data(self):
        """Test validation with invalid OHLC data."""
        # Insert invalid data (high < low)
        conn = sqlite3.connect(self.test_db)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO ohlcv (timestamp, symbol, open, high, low, close, volume)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (time.time(), 'BTCUSDT', 50000, 49900, 50100, 50050, 1.0))
        
        conn.commit()
        conn.close()
        
        result = self.validator.validate_ohlcv_data('BTCUSDT', 1)
        
        self.assertFalse(result['valid'])
        self.assertGreater(len(result['issues']), 0)
    
    def test_data_quality_metrics(self):
        """Test data quality metrics calculation."""
        metrics = self.validator.get_data_quality_metrics('BTCUSDT', 1)
        
        self.assertIn('completeness_percentage', metrics)
        self.assertIn('data_quality_score', metrics)
        self.assertEqual(metrics['symbol'], 'BTCUSDT')

class TestFundingMonitor(unittest.TestCase):
    """Test the Funding Monitor."""
    
    def setUp(self):
        """Set up test environment."""
        self.test_db = 'test_crypto_data.db'
        self.monitor = FundingMonitor()
        self.monitor.db_path = self.test_db
        self.monitor._init_database()
    
    def tearDown(self):
        """Clean up test environment."""
        if os.path.exists(self.test_db):
            os.remove(self.test_db)
    
    def test_database_initialization(self):
        """Test funding monitor database initialization."""
        conn = sqlite3.connect(self.test_db)
        cursor = conn.cursor()
        
        # Check if funding_rates table exists
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='funding_rates'")
        result = cursor.fetchone()
        self.assertIsNotNone(result)
        
        conn.close()
    
    def test_daily_rate_calculation(self):
        """Test daily funding rate calculation."""
        # Test with 0.01% 8-hour rate (should be 0.03% daily)
        funding_rate = 0.0001
        daily_rate = self.monitor._calculate_daily_rate(funding_rate)
        
        expected_daily = funding_rate * (24 / 8)
        self.assertAlmostEqual(daily_rate, expected_daily, places=6)
    
    def test_funding_level_assessment(self):
        """Test funding level assessment."""
        # Test critical level (1% daily)
        level = self.monitor._assess_funding_level(0.01)
        self.assertEqual(level, 'critical')
        
        # Test high level (0.5% daily)
        level = self.monitor._assess_funding_level(0.005)
        self.assertEqual(level, 'high')
        
        # Test low level (0.05% daily)
        level = self.monitor._assess_funding_level(0.0005)
        self.assertEqual(level, 'low')
    
    def test_funding_cost_calculation(self):
        """Test funding cost calculation."""
        # Insert test funding rate data
        conn = sqlite3.connect(self.test_db)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO funding_rates 
            (timestamp, exchange, symbol, funding_rate, next_funding_time, 
             mark_price, daily_rate, annualized_rate)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            time.time(), 'binance', 'BTCUSDT', 0.0001, time.time() + 3600,
            50000.0, 0.0003, 0.1095
        ))
        
        conn.commit()
        conn.close()
        
        # Calculate funding cost
        cost = self.monitor.calculate_funding_cost('BTCUSDT', 1000.0, 8)
        
        self.assertEqual(cost['symbol'], 'BTCUSDT')
        self.assertEqual(cost['position_size'], 1000.0)
        self.assertIn('total_funding_cost', cost)

class TestPaperTrader(unittest.TestCase):
    """Test the Paper Trading Engine."""
    
    def setUp(self):
        """Set up test environment."""
        self.test_db = 'test_crypto_data.db'
        self.engine = PaperTradingEngine()
        self.engine.set_db_path(self.test_db)
        
        # Create test price data
        conn = sqlite3.connect(self.test_db)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS ohlcv (
                timestamp REAL,
                symbol TEXT,
                open REAL,
                high REAL,
                low REAL,
                close REAL,
                volume REAL
            )
        ''')
        
        # Insert current price data
        cursor.execute('''
            INSERT INTO ohlcv (timestamp, symbol, open, high, low, close, volume)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (time.time(), 'BTCUSDT', 50000, 50100, 49900, 50000, 1.0))
        
        conn.commit()
        conn.close()
    
    def tearDown(self):
        """Clean up test environment."""
        if os.path.exists(self.test_db):
            os.remove(self.test_db)
    
    def test_initial_portfolio_state(self):
        """Test initial portfolio state."""
        self.assertEqual(self.engine.cash_balance, 10000.0)
        self.assertEqual(len(self.engine.positions), 0)
        self.assertEqual(self.engine.total_trades, 0)
    
    def test_place_buy_order(self):
        """Test placing a buy order."""
        # Use smaller quantity that won't exceed risk limits (0.05 BTC ≈ $2500)
        order_id = self.engine.place_order('BTCUSDT', OrderSide.BUY, 0.05, OrderType.MARKET)
        
        self.assertIsNotNone(order_id)
        self.assertNotEqual(order_id, "")  # Order should not be rejected
        self.assertIn(order_id, self.engine.orders)
        
        # Check if cash balance decreased
        self.assertLess(self.engine.cash_balance, 10000.0)
        
        # Check if position was created
        self.assertIn('BTCUSDT', self.engine.positions)
    
    def test_place_sell_order(self):
        """Test placing a sell order."""
        # First place a buy order (smaller quantity to stay within risk limits)
        buy_order_id = self.engine.place_order('BTCUSDT', OrderSide.BUY, 0.05, OrderType.MARKET)
        self.assertNotEqual(buy_order_id, "")  # Should succeed
        
        # Then place a sell order
        sell_order_id = self.engine.place_order('BTCUSDT', OrderSide.SELL, 0.02, OrderType.MARKET)
        
        self.assertIsNotNone(sell_order_id)
        self.assertNotEqual(sell_order_id, "")  # Should not be rejected
        
        # Check if position was reduced
        position = self.engine.positions['BTCUSDT']
        self.assertAlmostEqual(position.quantity, 0.03, places=6)  # 0.05 - 0.02
    
    def test_portfolio_value_calculation(self):
        """Test portfolio value calculation."""
        # Place a buy order (smaller quantity to stay within risk limits)
        order_id = self.engine.place_order('BTCUSDT', OrderSide.BUY, 0.05, OrderType.MARKET)
        self.assertNotEqual(order_id, "")  # Should succeed
        
        # Calculate portfolio value
        portfolio_value = self.engine._calculate_portfolio_value()
        
        # Should be approximately equal to starting balance (minus fees)
        self.assertGreater(portfolio_value, 9900.0)  # Account for fees
        self.assertLess(portfolio_value, 10000.0)
    
    def test_position_size_calculation(self):
        """Test position size calculation."""
        position_size = self.engine._calculate_position_size('BTCUSDT', 1.0)
        
        # Should be based on max position percentage
        expected_max_value = 10000.0 * 0.25  # 25% of portfolio
        expected_quantity = expected_max_value / 50000.0  # At $50,000 per BTC
        
        self.assertAlmostEqual(position_size, expected_quantity, places=6)
    
    def test_portfolio_summary(self):
        """Test portfolio summary generation."""
        # Place some orders (ensure it stays within risk limits)
        order_id = self.engine.place_order('BTCUSDT', OrderSide.BUY, 0.05, OrderType.MARKET)
        self.assertNotEqual(order_id, "")  # Should succeed
        
        summary = self.engine.get_portfolio_summary()
        
        self.assertIn('cash_balance', summary)
        self.assertIn('total_value', summary)
        self.assertIn('positions', summary)
        self.assertIn('total_trades', summary)
        self.assertEqual(summary['position_count'], 1)

class TestIntegration(unittest.TestCase):
    """Integration tests for all components."""
    
    def setUp(self):
        """Set up integration test environment."""
        self.test_db = 'test_crypto_data.db'
        
        # Initialize all components
        self.trade_stream = TradeStreamCollector()
        self.trade_stream.db_path = self.test_db
        self.trade_stream._init_database()
        
        self.event_sync = EventSynchronizer()
        self.event_sync.db_path = self.test_db
        
        self.data_validator = DataValidator()
        self.data_validator.db_path = self.test_db
        
        self.funding_monitor = FundingMonitor()
        self.funding_monitor.db_path = self.test_db
        self.funding_monitor._init_database()
        
        self.paper_trader = PaperTradingEngine()
        self.paper_trader.db_path = self.test_db
        self.paper_trader._init_database()
    
    def tearDown(self):
        """Clean up integration test environment."""
        if os.path.exists(self.test_db):
            os.remove(self.test_db)
    
    def test_database_compatibility(self):
        """Test that all components can work with the same database."""
        conn = sqlite3.connect(self.test_db)
        cursor = conn.cursor()
        
        # Check that all required tables exist
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [row[0] for row in cursor.fetchall()]
        
        expected_tables = [
            'trades', 'funding_rates', 'portfolio_state', 
            'paper_orders', 'paper_trades'
        ]
        
        for table in expected_tables:
            self.assertIn(table, tables)
        
        conn.close()
    
    def test_data_flow_integration(self):
        """Test data flow between components."""
        # Simulate trade data
        trade_data = {
            's': 'BTCUSDT',
            't': 12345,
            'p': '50000.0',
            'q': '1.0',
            'm': False,
            'T': int(time.time() * 1000)
        }
        
        # Process trade through trade stream
        self.trade_stream._process_trade_message(trade_data)
        
        # Flush to database
        asyncio.run(self.trade_stream._flush_trades_to_db())
        
        # Validate data
        validation_result = self.data_validator.validate_trades_data('BTCUSDT', 1)
        
        # Should have valid data
        self.assertTrue(validation_result.get('valid', False))
        self.assertGreater(validation_result.get('data_points', 0), 0)

def run_tests():
    """Run all tests and generate a summary report."""
    print("=" * 80)
    print("TickerML New Components Test Suite")
    print("=" * 80)
    
    # Create test suite
    test_classes = [
        TestTradeStream,
        TestEventSynchronizer,
        TestDataValidator,
        TestFundingMonitor,
        TestPaperTrader,
        TestIntegration
    ]
    
    total_tests = 0
    passed_tests = 0
    failed_tests = 0
    
    for test_class in test_classes:
        print(f"\n{'='*20} {test_class.__name__} {'='*20}")
        
        suite = unittest.TestLoader().loadTestsFromTestCase(test_class)
        runner = unittest.TextTestRunner(verbosity=2)
        result = runner.run(suite)
        
        total_tests += result.testsRun
        passed_tests += result.testsRun - len(result.failures) - len(result.errors)
        failed_tests += len(result.failures) + len(result.errors)
    
    # Print summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    print(f"Total Tests: {total_tests}")
    print(f"Passed: {passed_tests}")
    print(f"Failed: {failed_tests}")
    print(f"Success Rate: {(passed_tests/total_tests)*100:.1f}%")
    
    if failed_tests == 0:
        print("🎉 ALL TESTS PASSED!")
    else:
        print(f"⚠️  {failed_tests} tests failed. Please review the failures above.")
    
    return failed_tests == 0

if __name__ == "__main__":
    # Create logs directory if it doesn't exist
    os.makedirs('logs', exist_ok=True)
    
    # Run the tests
    success = run_tests()
    sys.exit(0 if success else 1)