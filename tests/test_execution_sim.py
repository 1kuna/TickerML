#!/usr/bin/env python3
"""
Comprehensive test suite for Advanced Execution Simulator
Tests FIFO queue modeling, toxic fill detection, and realistic execution
"""

import unittest
import tempfile
import os
import sys
import sqlite3
import json
import time
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from raspberry_pi.execution_simulator import (
    AdvancedExecutionSimulator, ExchangeType, OrderBookLevel, ExecutionResult
)

class TestAdvancedExecutionSimulator(unittest.TestCase):
    """Test advanced execution simulation functionality"""
    
    def setUp(self):
        """Set up test fixtures"""
        # Create temporary database with test data
        self.temp_db = tempfile.NamedTemporaryFile(suffix='.db', delete=False)
        self.temp_db.close()
        
        # Initialize simulator
        self.simulator = AdvancedExecutionSimulator(self.temp_db.name)
        
        # Create test data
        self._create_test_data()
    
    def tearDown(self):
        """Clean up test fixtures"""
        try:
            os.unlink(self.temp_db.name)
        except:
            pass
    
    def _create_test_data(self):
        """Create test order book and OHLCV data"""
        conn = sqlite3.connect(self.temp_db.name)
        cursor = conn.cursor()
        
        # Create OHLCV table with test data
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
        
        # Add test OHLCV data
        current_time = time.time()
        test_data = [
            (current_time - 300, 'BTCUSD', 50000, 51000, 49500, 50500, 100),
            (current_time - 240, 'BTCUSD', 50500, 51200, 50000, 50800, 120),
            (current_time - 180, 'BTCUSD', 50800, 51500, 50300, 51000, 90),
            (current_time - 120, 'BTCUSD', 51000, 51300, 50700, 51200, 110),
            (current_time - 60, 'BTCUSD', 51200, 51800, 51000, 51500, 130),
        ]
        
        cursor.executemany('''
            INSERT INTO ohlcv (timestamp, symbol, open, high, low, close, volume)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', test_data)
        
        # Create order books table with test data
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS order_books (
                timestamp REAL,
                symbol TEXT,
                bids TEXT,
                asks TEXT,
                mid_price REAL,
                spread_bps REAL,
                imbalance REAL
            )
        ''')
        
        # Add test order book data
        test_bids = [[51450, 2.5], [51440, 1.8], [51430, 3.2], [51420, 2.1], [51410, 1.5]]
        test_asks = [[51460, 1.9], [51470, 2.3], [51480, 2.8], [51490, 1.7], [51500, 2.0]]
        
        cursor.execute('''
            INSERT INTO order_books (timestamp, symbol, bids, asks, mid_price, spread_bps, imbalance)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (current_time - 30, 'BTCUSD', json.dumps(test_bids), json.dumps(test_asks), 
              51455, 19.4, 0.12))
        
        conn.commit()
        conn.close()
    
    def test_order_book_retrieval(self):
        """Test order book data retrieval"""
        order_book = self.simulator.get_current_order_book('BTCUSD')
        
        # Should retrieve order book or create synthetic one
        self.assertIsNotNone(order_book)
        self.assertIn('bids', order_book)
        self.assertIn('asks', order_book)
        self.assertIn('mid_price', order_book)
        
        # Bids and asks should be lists
        self.assertIsInstance(order_book['bids'], list)
        self.assertIsInstance(order_book['asks'], list)
        
        # Should have reasonable prices
        if order_book['bids']:
            self.assertGreater(order_book['bids'][0][0], 1000)  # Reasonable BTC price
        if order_book['asks']:
            self.assertGreater(order_book['asks'][0][0], 1000)  # Reasonable BTC price
    
    def test_synthetic_order_book_creation(self):
        """Test synthetic order book creation from OHLCV"""
        # Test with symbol that has OHLCV but no order book data
        order_book = self.simulator.get_current_order_book('BTCUSD')
        
        self.assertIsNotNone(order_book)
        
        # Should have 'synthetic' flag if created from OHLCV
        if order_book.get('synthetic'):
            self.assertTrue(order_book['synthetic'])
            
            # Should have reasonable bid-ask spread
            if order_book['bids'] and order_book['asks']:
                spread = order_book['asks'][0][0] - order_book['bids'][0][0]
                spread_pct = spread / order_book['mid_price']
                self.assertLess(spread_pct, 0.01)  # Less than 1% spread
    
    def test_cumulative_volume_calculation(self):
        """Test cumulative volume calculation for queue position"""
        test_levels = [[51450, 2.5], [51440, 1.8], [51430, 3.2]]
        
        ob_levels = self.simulator.calculate_cumulative_volumes(test_levels)
        
        # Should have proper cumulative volumes
        self.assertEqual(len(ob_levels), 3)
        self.assertEqual(ob_levels[0].cumulative_quantity, 2.5)
        self.assertEqual(ob_levels[1].cumulative_quantity, 4.3)  # 2.5 + 1.8
        self.assertEqual(ob_levels[2].cumulative_quantity, 7.5)  # 4.3 + 3.2
    
    def test_queue_position_estimation(self):
        """Test FIFO queue position estimation"""
        test_levels = [
            OrderBookLevel(51450, 2.5, 2.5),
            OrderBookLevel(51440, 1.8, 4.3),
            OrderBookLevel(51430, 3.2, 7.5)
        ]
        
        # Test queue position for order at existing price level
        queue_pos = self.simulator.estimate_queue_position(test_levels, 51440, 'buy')
        self.assertEqual(queue_pos, 4)  # Should be behind 2.5 + 1.8 = 4.3, rounded to 4
        
        # Test queue position for order at best price
        queue_pos_best = self.simulator.estimate_queue_position(test_levels, 51450, 'buy')
        self.assertEqual(queue_pos_best, 2)  # Should be behind 2.5, rounded to 2
    
    def test_toxic_fill_detection(self):
        """Test toxic fill detection (queue position > 10)"""
        # Create levels with high cumulative volume
        high_volume_levels = [
            OrderBookLevel(51450, 15.0, 15.0),  # High volume at best price
            OrderBookLevel(51440, 5.0, 20.0),
            OrderBookLevel(51430, 3.0, 23.0)
        ]
        
        queue_pos = self.simulator.estimate_queue_position(high_volume_levels, 51450, 'buy')
        
        # Should detect high queue position
        self.assertGreater(queue_pos, self.simulator.toxic_fill_threshold)
    
    def test_market_order_execution(self):
        """Test market order execution simulation"""
        result = self.simulator.simulate_order_execution(
            symbol='BTCUSD',
            side='buy',
            quantity=0.01,
            order_type='market'
        )
        
        # Should return proper execution result
        self.assertIsInstance(result, ExecutionResult)
        self.assertGreaterEqual(result.filled_quantity, 0)
        self.assertGreaterEqual(result.avg_fill_price, 0)
        self.assertGreaterEqual(result.latency_ms, 0)
        
        # Remaining quantity should be original minus filled
        expected_remaining = 0.01 - result.filled_quantity
        self.assertAlmostEqual(result.remaining_quantity, expected_remaining, places=6)
    
    def test_limit_order_execution(self):
        """Test limit order execution simulation"""
        # Get current order book to set reasonable limit price
        order_book = self.simulator.get_current_order_book('BTCUSD')
        
        if order_book and order_book['asks']:
            limit_price = order_book['asks'][0][0] - 50  # Below best ask
            
            result = self.simulator.simulate_order_execution(
                symbol='BTCUSD',
                side='buy',
                quantity=0.01,
                order_type='limit',
                price=limit_price
            )
            
            # Should return proper execution result
            self.assertIsInstance(result, ExecutionResult)
            self.assertGreaterEqual(result.filled_quantity, 0)
    
    def test_exchange_latency_simulation(self):
        """Test exchange-specific latency simulation"""
        # Test different exchanges
        exchanges = [
            ExchangeType.BINANCE_US,
            ExchangeType.COINBASE,
            ExchangeType.KRAKEN,
            ExchangeType.KUCOIN
        ]
        
        for exchange in exchanges:
            latency = self.simulator.simulate_exchange_latency(exchange)
            
            # Should return reasonable latency
            self.assertGreater(latency, 0)
            self.assertLess(latency, 10000)  # Less than 10 seconds
            
            # Should be within expected range for exchange
            min_lat, max_lat = self.simulator.exchange_latencies[exchange]
            self.assertGreaterEqual(latency, min_lat)
            self.assertLessEqual(latency, max_lat * 2)  # Allow for network spikes
    
    def test_market_impact_calculation(self):
        """Test market impact calculation"""
        order_book = self.simulator.get_current_order_book('BTCUSD')
        
        if order_book:
            # Test small order (should have minimal impact)
            small_impact = self.simulator.calculate_market_impact(0.001, 'buy', order_book)
            
            # Test large order (should have larger impact)
            large_impact = self.simulator.calculate_market_impact(1.0, 'buy', order_book)
            
            # Large order should have more impact
            self.assertGreaterEqual(abs(large_impact), abs(small_impact))
            
            # Impact should be reasonable (not excessive)
            mid_price = order_book['mid_price']
            small_impact_pct = abs(small_impact / mid_price)
            large_impact_pct = abs(large_impact / mid_price)
            
            self.assertLess(small_impact_pct, 0.01)  # Less than 1% for small order
            self.assertLess(large_impact_pct, 0.10)  # Less than 10% for large order
    
    def test_partial_fills_simulation(self):
        """Test partial fills simulation"""
        order_book = self.simulator.get_current_order_book('BTCUSD')
        
        if order_book and order_book['bids']:
            # Create mock order book levels
            levels = self.simulator.calculate_cumulative_volumes(order_book['bids'])
            
            if levels:
                # Test partial fills for large order
                fills = self.simulator.simulate_partial_fills(
                    order_quantity=10.0,  # Large order
                    queue_position=1,
                    order_price=levels[0].price,
                    levels=levels,
                    side='sell',
                    is_synthetic=order_book.get('synthetic', False)
                )
                
                # Should return list of fills
                self.assertIsInstance(fills, list)
                
                # Each fill should be a tuple of (quantity, price)
                for fill in fills:
                    self.assertEqual(len(fill), 2)
                    self.assertGreater(fill[0], 0)  # Positive quantity
                    self.assertGreater(fill[1], 0)  # Positive price
    
    def test_execution_result_logging(self):
        """Test execution result logging"""
        # Simulate an order
        result = self.simulator.simulate_order_execution(
            symbol='BTCUSD',
            side='buy',
            quantity=0.01,
            order_type='market'
        )
        
        # Test logging (should not raise exception)
        try:
            self.simulator.log_execution_metrics(result, 'TEST_ORDER_001')
        except Exception as e:
            self.fail(f"Logging execution metrics failed: {e}")

class TestExecutionSimulatorRealism(unittest.TestCase):
    """Test execution simulator realism features"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.temp_db = tempfile.NamedTemporaryFile(suffix='.db', delete=False)
        self.temp_db.close()
        self.simulator = AdvancedExecutionSimulator(self.temp_db.name)
    
    def tearDown(self):
        """Clean up test fixtures"""
        try:
            os.unlink(self.temp_db.name)
        except:
            pass
    
    def test_toxic_fill_threshold(self):
        """Test toxic fill threshold configuration"""
        # Threshold should be reasonable
        self.assertGreaterEqual(self.simulator.toxic_fill_threshold, 5)
        self.assertLessEqual(self.simulator.toxic_fill_threshold, 20)
        
        # Max queue position should be higher than toxic threshold
        self.assertGreater(self.simulator.max_queue_position, self.simulator.toxic_fill_threshold)
    
    def test_market_impact_parameters(self):
        """Test market impact parameters are reasonable"""
        # Impact coefficient should be small
        self.assertLess(self.simulator.impact_coefficient, 0.01)
        self.assertGreater(self.simulator.impact_coefficient, 0.00001)
        
        # Min/max impact should be reasonable
        self.assertLess(self.simulator.min_impact_bps, 10)  # Less than 10 bps minimum
        self.assertGreater(self.simulator.max_impact_bps, 10)  # More than 10 bps maximum
        self.assertLess(self.simulator.max_impact_bps, 1000)  # Less than 10% maximum
    
    def test_exchange_latency_profiles(self):
        """Test exchange latency profiles are realistic"""
        for exchange, (min_lat, max_lat) in self.simulator.exchange_latencies.items():
            # Minimum latency should be reasonable
            self.assertGreaterEqual(min_lat, 10)   # At least 10ms
            self.assertLessEqual(min_lat, 1000)    # At most 1 second
            
            # Maximum latency should be higher than minimum
            self.assertGreater(max_lat, min_lat)
            
            # Maximum latency should be reasonable
            self.assertLessEqual(max_lat, 5000)    # At most 5 seconds

def run_execution_simulator_tests():
    """Run all execution simulator tests"""
    print("Running Advanced Execution Simulator Tests...")
    
    # Create test suite
    suite = unittest.TestSuite()
    
    # Add test cases
    suite.addTest(unittest.makeSuite(TestAdvancedExecutionSimulator))
    suite.addTest(unittest.makeSuite(TestExecutionSimulatorRealism))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print(f"\nExecution Simulator Tests Complete:")
    print(f"  Tests Run: {result.testsRun}")
    print(f"  Failures: {len(result.failures)}")
    print(f"  Errors: {len(result.errors)}")
    print(f"  Success Rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%")
    
    return len(result.failures) == 0 and len(result.errors) == 0

if __name__ == "__main__":
    success = run_execution_simulator_tests()
    if not success:
        sys.exit(1)