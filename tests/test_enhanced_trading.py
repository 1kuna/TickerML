#!/usr/bin/env python3
"""
Test suite for enhanced paper trading with queue modeling and risk management
"""

import unittest
import sys
import tempfile
import os
from pathlib import Path
import sqlite3
import time

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from raspberry_pi.paper_trader import PaperTradingEngine, OrderSide, OrderType
from raspberry_pi.execution_simulator import AdvancedExecutionSimulator
from raspberry_pi.risk_manager import AdvancedRiskManager

class TestEnhancedTradingSystem(unittest.TestCase):
    """Test suite for enhanced trading system"""
    
    def setUp(self):
        """Set up test environment"""
        # Create temporary database
        self.test_db = tempfile.mktemp(suffix='.db')
        
        # Create test configuration
        self.test_config = {
            'database': {'path': self.test_db},
            'binance': {'symbols': ['BTCUSD', 'ETHUSD']}
        }
        
        # Initialize components
        self.execution_simulator = AdvancedExecutionSimulator(db_path=self.test_db)
        self.risk_manager = AdvancedRiskManager(db_path=self.test_db)
        
    def tearDown(self):
        """Clean up test environment"""
        try:
            if os.path.exists(self.test_db):
                os.remove(self.test_db)
        except:
            pass
    
    def create_test_order_book_data(self):
        """Create sample order book data for testing"""
        try:
            conn = sqlite3.connect(self.test_db)
            cursor = conn.cursor()
            
            # Create order_books table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS order_books (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp INTEGER NOT NULL,
                    exchange TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    bids TEXT NOT NULL,
                    asks TEXT NOT NULL,
                    mid_price REAL NOT NULL,
                    spread REAL NOT NULL,
                    spread_bps REAL NOT NULL,
                    imbalance REAL NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Insert sample order book data
            current_time = time.time()
            sample_data = {
                'timestamp': int(current_time),
                'exchange': 'binance_us',
                'symbol': 'BTCUSD',
                'bids': '[["50000.0", "1.0"], ["49900.0", "2.0"], ["49800.0", "1.5"]]',
                'asks': '[["50100.0", "1.2"], ["50200.0", "1.8"], ["50300.0", "2.0"]]',
                'mid_price': 50050.0,
                'spread': 100.0,
                'spread_bps': 20.0,
                'imbalance': 0.1
            }
            
            cursor.execute('''
                INSERT INTO order_books (timestamp, exchange, symbol, bids, asks, 
                                       mid_price, spread, spread_bps, imbalance)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                sample_data['timestamp'],
                sample_data['exchange'],
                sample_data['symbol'],
                sample_data['bids'],
                sample_data['asks'],
                sample_data['mid_price'],
                sample_data['spread'],
                sample_data['spread_bps'],
                sample_data['imbalance']
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            print(f"Error creating test order book data: {e}")
    
    def create_test_price_data(self):
        """Create sample price data for correlation analysis"""
        try:
            conn = sqlite3.connect(self.test_db)
            cursor = conn.cursor()
            
            # Create ohlcv table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS ohlcv (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp INTEGER NOT NULL,
                    symbol TEXT NOT NULL,
                    open REAL NOT NULL,
                    high REAL NOT NULL,
                    low REAL NOT NULL,
                    close REAL NOT NULL,
                    volume REAL NOT NULL
                )
            ''')
            
            # Insert sample price data for correlation analysis
            import random
            current_time = int(time.time())
            
            for symbol in ['BTCUSD', 'ETHUSD']:
                base_price = 50000 if symbol == 'BTCUSD' else 3000
                
                for i in range(50):  # 50 data points
                    timestamp = current_time - (i * 3600)  # Hourly data
                    
                    # Add some correlation between assets
                    price_change = random.uniform(-0.05, 0.05)
                    if symbol == 'ETHUSD':
                        price_change *= 1.2  # ETH is more volatile
                    
                    price = base_price * (1 + price_change)
                    
                    cursor.execute('''
                        INSERT INTO ohlcv (timestamp, symbol, open, high, low, close, volume)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        timestamp,
                        symbol,
                        price * 0.995,  # open
                        price * 1.01,   # high
                        price * 0.99,   # low
                        price,          # close
                        random.uniform(100, 1000)  # volume
                    ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            print(f"Error creating test price data: {e}")
    
    def test_execution_simulator_with_order_book(self):
        """Test execution simulator with order book data"""
        self.create_test_order_book_data()
        
        # Test market buy order
        result = self.execution_simulator.simulate_order_execution(
            symbol='BTCUSD',
            side='buy',
            quantity=0.1,
            order_type='market'
        )
        
        self.assertGreater(result.filled_quantity, 0, "Should fill some quantity")
        self.assertGreater(result.avg_fill_price, 0, "Should have execution price")
        self.assertGreaterEqual(result.queue_position, 0, "Queue position should be non-negative")
        self.assertGreater(result.latency_ms, 0, "Should have simulated latency")
    
    def test_risk_manager_position_limits(self):
        """Test risk manager position limits"""
        self.create_test_price_data()
        
        # Test portfolio with reasonable position
        positions = {
            'BTCUSD': {'quantity': 0.05, 'market_value': 2500, 'side': 'long'},
            'total_value': 10000
        }
        
        symbols = ['BTCUSD', 'ETHUSD']
        
        # Should allow reasonable position
        allowed, reason, max_size = self.risk_manager.check_position_limits(
            'ETHUSD', 1000, positions, symbols
        )
        
        self.assertTrue(allowed, f"Should allow reasonable position: {reason}")
        
        # Should reject oversized position
        allowed, reason, max_size = self.risk_manager.check_position_limits(
            'ETHUSD', 5000, positions, symbols  # 50% of portfolio
        )
        
        self.assertFalse(allowed, "Should reject oversized position")
        self.assertLess(max_size, 5000, "Should suggest smaller position size")
    
    def test_risk_assessment(self):
        """Test comprehensive risk assessment"""
        self.create_test_price_data()
        
        # Test balanced portfolio
        positions = {
            'BTCUSD': {'quantity': 0.05, 'market_value': 2000, 'side': 'long'},
            'ETHUSD': {'quantity': 1.0, 'market_value': 2000, 'side': 'long'},
            'total_value': 10000
        }
        
        symbols = ['BTCUSD', 'ETHUSD']
        
        risk_metrics = self.risk_manager.assess_portfolio_risk(positions, symbols)
        
        self.assertIsNotNone(risk_metrics, "Should return risk metrics")
        self.assertGreaterEqual(risk_metrics.total_exposure, 0, "Total exposure should be non-negative")
        self.assertLessEqual(risk_metrics.total_exposure, 1, "Total exposure should not exceed 100%")
        self.assertGreaterEqual(risk_metrics.portfolio_heat, 0, "Portfolio heat should be non-negative")
    
    def test_volatility_regime_detection(self):
        """Test volatility regime detection"""
        self.create_test_price_data()
        
        symbols = ['BTCUSD', 'ETHUSD']
        regime = self.risk_manager.detect_volatility_regime(symbols)
        
        self.assertIsInstance(regime, self.risk_manager.VolatilityRegime)
    
    def test_correlation_calculation(self):
        """Test correlation matrix calculation"""
        self.create_test_price_data()
        
        symbols = ['BTCUSD', 'ETHUSD']
        correlation_matrix = self.risk_manager.calculate_correlation_matrix(symbols)
        
        self.assertEqual(correlation_matrix.shape, (2, 2), "Should return 2x2 correlation matrix")
        self.assertAlmostEqual(correlation_matrix[0, 0], 1.0, places=2, 
                              msg="Diagonal should be 1.0")
        self.assertAlmostEqual(correlation_matrix[1, 1], 1.0, places=2,
                              msg="Diagonal should be 1.0")
    
    def test_integrated_trading_system(self):
        """Test integrated trading system with all enhancements"""
        self.create_test_order_book_data()
        self.create_test_price_data()
        
        # Create paper trading engine with real configuration
        config_file = tempfile.mktemp(suffix='.yaml')
        with open(config_file, 'w') as f:
            f.write(f"""
database:
  path: {self.test_db}
binance:
  symbols: ["BTCUSD", "ETHUSD"]
""")
        
        try:
            # Initialize paper trading engine
            engine = PaperTradingEngine(config_path=config_file)
            
            # Test that components are initialized
            self.assertIsNotNone(engine.execution_simulator, "Execution simulator should be initialized")
            self.assertIsNotNone(engine.risk_manager, "Risk manager should be initialized")
            
            # Test order placement (should work with order book data)
            order_id = engine.place_order('BTCUSD', OrderSide.BUY, 0.01, OrderType.MARKET)
            
            # Order might be rejected due to risk limits or lack of data, but should not crash
            self.assertIsInstance(order_id, str, "Should return order ID string")
            
        finally:
            if os.path.exists(config_file):
                os.remove(config_file)
    
    def test_position_risk_adjustment(self):
        """Test position size adjustment based on correlation"""
        self.create_test_price_data()
        
        # Portfolio with existing correlated position
        positions = {
            'BTCUSD': {'quantity': 0.1, 'market_value': 5000, 'side': 'long'},
            'total_value': 10000
        }
        
        symbols = ['BTCUSD', 'ETHUSD']
        
        # Test correlation-based position adjustment
        base_size = 2000
        adjusted_size = self.risk_manager.calculate_position_risk_adjustment(
            'ETHUSD', base_size, positions, symbols
        )
        
        self.assertLessEqual(adjusted_size, base_size, 
                           "Adjusted size should not exceed base size")
        self.assertGreater(adjusted_size, 0, 
                          "Adjusted size should be positive")

def run_enhanced_trading_tests():
    """Run all enhanced trading tests"""
    print("Running Enhanced Trading System Tests...")
    
    suite = unittest.TestLoader().loadTestsFromTestCase(TestEnhancedTradingSystem)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    print(f"\nEnhanced Trading Tests Complete:")
    print(f"  Tests Run: {result.testsRun}")
    print(f"  Failures: {len(result.failures)}")
    print(f"  Errors: {len(result.errors)}")
    print(f"  Success Rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%")
    
    return result.wasSuccessful()

if __name__ == "__main__":
    success = run_enhanced_trading_tests()
    exit(0 if success else 1)