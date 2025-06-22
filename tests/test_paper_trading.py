#!/usr/bin/env python3
"""
Comprehensive test suite for Paper Trading Engine
Tests portfolio management, order execution, and risk controls
"""

import unittest
import tempfile
import os
import sys
import time
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from raspberry_pi.paper_trader import (
    PaperTradingEngine, OrderType, OrderSide, OrderStatus, PositionSide
)
from raspberry_pi.execution_simulator import AdvancedExecutionSimulator
from raspberry_pi.risk_manager import AdvancedRiskManager

class TestPaperTradingEngine(unittest.TestCase):
    """Test paper trading engine functionality"""
    
    def setUp(self):
        """Set up test fixtures"""
        # Create temporary database
        self.temp_db = tempfile.NamedTemporaryFile(suffix='.db', delete=False)
        self.temp_db.close()
        
        self.engine = PaperTradingEngine()
        self.engine.set_db_path(self.temp_db.name)
        
    def tearDown(self):
        """Clean up test fixtures"""
        try:
            os.unlink(self.temp_db.name)
        except:
            pass
    
    def test_initial_portfolio_state(self):
        """Test initial portfolio configuration"""
        self.assertEqual(self.engine.starting_balance, 10000.0)
        self.assertEqual(self.engine.cash_balance, 10000.0)
        self.assertEqual(len(self.engine.positions), 0)
        self.assertEqual(self.engine.total_trades, 0)
        self.assertEqual(self.engine.winning_trades, 0)
    
    def test_portfolio_value_calculation(self):
        """Test portfolio value calculation"""
        # Initial value should equal cash balance
        initial_value = self.engine._calculate_portfolio_value()
        self.assertEqual(initial_value, 10000.0)
        
        # Add a mock position
        from raspberry_pi.paper_trader import Position, PositionSide
        self.engine.positions['BTCUSD'] = Position(
            symbol='BTCUSD',
            side=PositionSide.LONG,
            quantity=0.1,
            entry_price=50000.0,
            current_price=52000.0,
            unrealized_pnl=200.0,
            realized_pnl=0.0,
            timestamp=time.time()
        )
        
        # Portfolio value should include position value
        portfolio_value = self.engine._calculate_portfolio_value()
        self.assertGreater(portfolio_value, 10000.0)
    
    def test_position_sizing(self):
        """Test position sizing calculations"""
        # Test with 100% signal strength
        position_size = self.engine._calculate_position_size('BTCUSD', 1.0)
        
        # Should be reasonable percentage of portfolio
        portfolio_value = self.engine._calculate_portfolio_value()
        position_pct = (position_size * 50000) / portfolio_value  # Assuming BTC at $50k
        self.assertLessEqual(position_pct, self.engine.max_position_pct)
        
        # Test with lower signal strength
        small_position = self.engine._calculate_position_size('BTCUSD', 0.5)
        self.assertLess(small_position, position_size)
    
    def test_risk_management_integration(self):
        """Test integration with risk management system"""
        # Risk manager should be initialized
        self.assertIsInstance(self.engine.risk_manager, AdvancedRiskManager)
        
        # Test risk limits are configured
        self.assertEqual(self.engine.max_position_pct, 0.25)
        self.assertEqual(self.engine.max_drawdown_pct, 0.25)
        self.assertEqual(self.engine.stop_loss_pct, 0.05)
    
    def test_order_placement_validation(self):
        """Test order placement with validation"""
        # Place a reasonable order
        order_id = self.engine.place_order(
            'BTCUSD', OrderSide.BUY, 0.01, OrderType.MARKET
        )
        
        # Should return valid order ID or empty string if rejected
        self.assertIsInstance(order_id, str)
        
        # Try to place order larger than available cash
        large_order_id = self.engine.place_order(
            'BTCUSD', OrderSide.BUY, 1000, OrderType.MARKET  # Huge order
        )
        
        # Should be rejected (empty string) due to insufficient cash
        self.assertEqual(large_order_id, "")
    
    def test_portfolio_summary(self):
        """Test portfolio summary generation"""
        summary = self.engine.get_portfolio_summary()
        
        # Check required fields
        required_fields = [
            'timestamp', 'cash_balance', 'total_value', 'total_pnl',
            'positions', 'max_drawdown', 'win_rate', 'total_trades'
        ]
        
        for field in required_fields:
            self.assertIn(field, summary)
        
        # Check data types
        self.assertIsInstance(summary['cash_balance'], float)
        self.assertIsInstance(summary['total_value'], float)
        self.assertIsInstance(summary['positions'], dict)
        self.assertIsInstance(summary['total_trades'], int)
    
    def test_stop_loss_and_take_profit_levels(self):
        """Test stop loss and take profit configuration"""
        # Check stop loss percentage is reasonable
        self.assertLessEqual(self.engine.stop_loss_pct, 0.10)  # Max 10%
        self.assertGreaterEqual(self.engine.stop_loss_pct, 0.01)  # Min 1%
        
        # Check take profit percentage
        self.assertLessEqual(self.engine.take_profit_pct, 0.20)  # Max 20%
        self.assertGreaterEqual(self.engine.take_profit_pct, 0.02)  # Min 2%
        
        # Take profit should be larger than stop loss for positive risk/reward
        self.assertGreater(self.engine.take_profit_pct, self.engine.stop_loss_pct)
    
    def test_execution_simulator_integration(self):
        """Test integration with execution simulator"""
        # Execution simulator should be initialized
        self.assertIsInstance(self.engine.execution_simulator, AdvancedExecutionSimulator)
        
        # Should have proper database path
        self.assertEqual(self.engine.execution_simulator.db_path, self.temp_db.name)
    
    def test_commission_and_fees(self):
        """Test commission and fee calculations"""
        # Commission rate should be reasonable for crypto
        self.assertLessEqual(self.engine.commission_rate, 0.01)  # Max 1%
        self.assertGreaterEqual(self.engine.commission_rate, 0.0001)  # Min 0.01%
        
        # Slippage factor should be small
        self.assertLessEqual(self.engine.slippage_factor, 0.01)  # Max 1%
        self.assertGreaterEqual(self.engine.slippage_factor, 0.0001)  # Min 0.01%
    
    def test_minimum_order_size(self):
        """Test minimum order size enforcement"""
        # Should have reasonable minimum order size
        self.assertGreaterEqual(self.engine.min_order_size, 1.0)  # At least $1
        self.assertLessEqual(self.engine.min_order_size, 50.0)  # At most $50
        
        # Test position sizing respects minimum
        small_signal_size = self.engine._calculate_position_size('BTCUSD', 0.001)  # Tiny signal
        
        # If position value would be below minimum, should return 0
        position_value = small_signal_size * 50000  # Assuming BTC at $50k
        if position_value > 0:
            self.assertGreaterEqual(position_value, self.engine.min_order_size)

class TestPaperTradingRiskControls(unittest.TestCase):
    """Test risk control mechanisms in paper trading"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.temp_db = tempfile.NamedTemporaryFile(suffix='.db', delete=False)
        self.temp_db.close()
        
        self.engine = PaperTradingEngine()
        self.engine.set_db_path(self.temp_db.name)
    
    def tearDown(self):
        """Clean up test fixtures"""
        try:
            os.unlink(self.temp_db.name)
        except:
            pass
    
    def test_maximum_drawdown_protection(self):
        """Test maximum drawdown protection"""
        # Max drawdown should be reasonable
        self.assertLessEqual(self.engine.max_drawdown_pct, 0.30)  # Max 30%
        self.assertGreaterEqual(self.engine.max_drawdown_pct, 0.10)  # Min 10%
        
        # High water mark should be initialized to starting balance
        self.assertEqual(self.engine.high_water_mark, self.engine.starting_balance)
    
    def test_position_concentration_limits(self):
        """Test position concentration limits"""
        # Max position percentage should be reasonable
        self.assertLessEqual(self.engine.max_position_pct, 0.50)  # Max 50%
        self.assertGreaterEqual(self.engine.max_position_pct, 0.10)  # Min 10%
        
        # Should prevent excessive concentration
        self.assertLessEqual(self.engine.max_position_pct, 0.25)  # Recommended max 25%
    
    def test_portfolio_heat_calculation(self):
        """Test portfolio heat (total exposure) calculation"""
        # Test with risk manager integration
        portfolio_data = self.engine._get_portfolio_for_risk_check()
        
        # Should return proper dictionary structure
        self.assertIsInstance(portfolio_data, dict)
        self.assertIn('total_value', portfolio_data)
        
        # Total value should match portfolio value
        expected_value = self.engine._calculate_portfolio_value()
        self.assertEqual(portfolio_data['total_value'], expected_value)
    
    def test_sharpe_ratio_calculation(self):
        """Test Sharpe ratio calculation"""
        # Should handle empty returns gracefully
        sharpe = self.engine._calculate_sharpe_ratio()
        self.assertIsInstance(sharpe, float)
        
        # Should be 0 for no trading history
        self.assertEqual(sharpe, 0.0)

def run_paper_trading_tests():
    """Run all paper trading tests"""
    print("Running Paper Trading Engine Tests...")
    
    # Create test suite
    suite = unittest.TestSuite()
    
    # Add test cases
    suite.addTest(unittest.makeSuite(TestPaperTradingEngine))
    suite.addTest(unittest.makeSuite(TestPaperTradingRiskControls))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print(f"\nPaper Trading Tests Complete:")
    print(f"  Tests Run: {result.testsRun}")
    print(f"  Failures: {len(result.failures)}")
    print(f"  Errors: {len(result.errors)}")
    print(f"  Success Rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%")
    
    return len(result.failures) == 0 and len(result.errors) == 0

if __name__ == "__main__":
    success = run_paper_trading_tests()
    if not success:
        sys.exit(1)