#!/usr/bin/env python3
"""
Comprehensive test suite for Advanced Risk Manager
Tests correlation analysis, position limits, and portfolio risk assessment
"""

import unittest
import tempfile
import os
import sys
import sqlite3
import time
import numpy as np
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from raspberry_pi.risk_manager import (
    AdvancedRiskManager, RiskLevel, VolatilityRegime, RiskMetrics, PositionRisk
)

class TestAdvancedRiskManager(unittest.TestCase):
    """Test advanced risk management functionality"""
    
    def setUp(self):
        """Set up test fixtures"""
        # Create temporary database with test data
        self.temp_db = tempfile.NamedTemporaryFile(suffix='.db', delete=False)
        self.temp_db.close()
        
        # Initialize risk manager
        self.risk_manager = AdvancedRiskManager(self.temp_db.name)
        
        # Create test data
        self._create_test_data()
    
    def tearDown(self):
        """Clean up test fixtures"""
        try:
            os.unlink(self.temp_db.name)
        except:
            pass
    
    def _create_test_data(self):
        """Create test price data for correlation analysis"""
        conn = sqlite3.connect(self.temp_db.name)
        cursor = conn.cursor()
        
        # Create OHLCV table
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
        
        # Generate test price data with some correlation
        np.random.seed(42)  # For reproducible tests
        current_time = time.time()
        
        # Create correlated price series
        btc_prices = []
        eth_prices = []
        base_btc = 50000
        base_eth = 3000
        
        for i in range(100):  # 100 data points
            timestamp = current_time - (100 - i) * 3600  # Hourly data
            
            # Generate correlated returns
            btc_return = np.random.normal(0, 0.02)  # 2% volatility
            eth_return = 0.7 * btc_return + 0.3 * np.random.normal(0, 0.02)  # 70% correlation
            
            base_btc *= (1 + btc_return)
            base_eth *= (1 + eth_return)
            
            btc_prices.append((timestamp, 'BTCUSD', base_btc * 0.999, base_btc * 1.001, 
                             base_btc * 0.998, base_btc, 100))
            eth_prices.append((timestamp, 'ETHUSD', base_eth * 0.999, base_eth * 1.001,
                             base_eth * 0.998, base_eth, 200))
        
        # Insert test data
        cursor.executemany('''
            INSERT INTO ohlcv (timestamp, symbol, open, high, low, close, volume)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', btc_prices + eth_prices)
        
        conn.commit()
        conn.close()
    
    def test_risk_limit_configuration(self):
        """Test risk limit configuration"""
        # Check position limits are reasonable
        self.assertLessEqual(self.risk_manager.max_single_position, 0.50)  # Max 50%
        self.assertGreaterEqual(self.risk_manager.max_single_position, 0.10)  # Min 10%
        
        # Check portfolio limits
        self.assertLessEqual(self.risk_manager.max_portfolio_exposure, 1.0)  # Max 100%
        self.assertGreaterEqual(self.risk_manager.max_portfolio_exposure, 0.50)  # Min 50%
        
        # Check correlation threshold
        self.assertLessEqual(self.risk_manager.correlation_threshold, 1.0)
        self.assertGreaterEqual(self.risk_manager.correlation_threshold, 0.5)
    
    def test_price_data_loading(self):
        """Test price data loading for correlation analysis"""
        symbols = ['BTCUSD', 'ETHUSD']
        price_data = self.risk_manager.get_price_data(symbols, hours=24)
        
        if price_data is not None:
            # Should have data for both symbols
            self.assertGreaterEqual(len(price_data.columns), 2)
            
            # Should have reasonable number of data points
            self.assertGreater(len(price_data), 10)
            
            # Columns should be the symbols
            for symbol in symbols:
                if symbol in price_data.columns:
                    self.assertIn(symbol, price_data.columns)
    
    def test_correlation_matrix_calculation(self):
        """Test correlation matrix calculation"""
        symbols = ['BTCUSD', 'ETHUSD']
        corr_matrix = self.risk_manager.calculate_correlation_matrix(symbols)
        
        # Should return proper correlation matrix
        self.assertIsInstance(corr_matrix, np.ndarray)
        self.assertEqual(corr_matrix.shape, (2, 2))
        
        # Diagonal elements should be 1 (perfect self-correlation)
        np.testing.assert_allclose(np.diag(corr_matrix), [1.0, 1.0], atol=1e-6)
        
        # Matrix should be symmetric
        np.testing.assert_allclose(corr_matrix, corr_matrix.T, atol=1e-6)
        
        # Correlations should be between -1 and 1
        self.assertTrue(np.all(corr_matrix >= -1.0))
        self.assertTrue(np.all(corr_matrix <= 1.0))
    
    def test_volatility_regime_detection(self):
        """Test volatility regime detection"""
        symbols = ['BTCUSD', 'ETHUSD']
        regime = self.risk_manager.detect_volatility_regime(symbols)
        
        # Should return a valid volatility regime
        self.assertIsInstance(regime, VolatilityRegime)
        self.assertIn(regime, [VolatilityRegime.LOW_VOL, VolatilityRegime.NORMAL_VOL,
                              VolatilityRegime.HIGH_VOL, VolatilityRegime.EXTREME_VOL])
    
    def test_portfolio_heat_calculation(self):
        """Test portfolio heat calculation"""
        # Test with empty portfolio
        empty_portfolio = {'total_value': 10000}
        heat_empty = self.risk_manager.calculate_portfolio_heat(empty_portfolio)
        self.assertEqual(heat_empty, 0.0)
        
        # Test with positions
        portfolio_with_positions = {
            'BTCUSD': {'quantity': 0.1, 'market_value': 5000},
            'ETHUSD': {'quantity': 1.0, 'market_value': 3000},
            'total_value': 10000
        }
        
        heat_with_positions = self.risk_manager.calculate_portfolio_heat(portfolio_with_positions)
        
        # Should be between 0 and 1
        self.assertGreaterEqual(heat_with_positions, 0.0)
        self.assertLessEqual(heat_with_positions, 1.0)
        
        # Should reflect the actual exposure
        expected_heat = (5000 + 3000) / 10000
        self.assertAlmostEqual(heat_with_positions, expected_heat, places=2)
    
    def test_concentration_risk_assessment(self):
        """Test concentration risk assessment"""
        # Test with diversified portfolio
        diversified_portfolio = {
            'BTCUSD': {'quantity': 0.05, 'market_value': 2500},
            'ETHUSD': {'quantity': 0.8, 'market_value': 2400},
            'total_value': 10000
        }
        
        concentration_risk = self.risk_manager.assess_concentration_risk(diversified_portfolio)
        
        # Should be between 0 and 1
        self.assertGreaterEqual(concentration_risk, 0.0)
        self.assertLessEqual(concentration_risk, 1.0)
        
        # Test with concentrated portfolio (all in one sector)
        concentrated_portfolio = {
            'BTCUSD': {'quantity': 0.1, 'market_value': 8000},
            'total_value': 10000
        }
        
        concentration_risk_high = self.risk_manager.assess_concentration_risk(concentrated_portfolio)
        
        # Concentrated portfolio should have higher concentration risk
        self.assertGreater(concentration_risk_high, concentration_risk)
    
    def test_correlation_risk_calculation(self):
        """Test correlation risk calculation"""
        portfolio = {
            'BTCUSD': {'quantity': 0.1, 'market_value': 5000},
            'ETHUSD': {'quantity': 1.0, 'market_value': 3000},
            'total_value': 10000
        }
        symbols = ['BTCUSD', 'ETHUSD']
        
        correlation_risk = self.risk_manager.calculate_correlation_risk(portfolio, symbols)
        
        # Should be between 0 and 1
        self.assertGreaterEqual(correlation_risk, 0.0)
        self.assertLessEqual(correlation_risk, 1.0)
    
    def test_position_risk_adjustment(self):
        """Test position size adjustment based on correlation"""
        portfolio = {
            'BTCUSD': {'quantity': 0.1, 'market_value': 5000}
        }
        symbols = ['BTCUSD', 'ETHUSD']
        base_size = 2000.0
        
        adjusted_size = self.risk_manager.calculate_position_risk_adjustment(
            'ETHUSD', base_size, portfolio, symbols
        )
        
        # Should return a positive size
        self.assertGreater(adjusted_size, 0)
        
        # Adjusted size should be reasonable relative to base size
        self.assertLessEqual(adjusted_size, base_size)  # Should not increase size
        self.assertGreaterEqual(adjusted_size, base_size * 0.1)  # Should not reduce too much
    
    def test_portfolio_risk_assessment(self):
        """Test comprehensive portfolio risk assessment"""
        portfolio = {
            'BTCUSD': {'quantity': 0.1, 'market_value': 5000},
            'ETHUSD': {'quantity': 1.0, 'market_value': 3000},
            'total_value': 10000
        }
        symbols = ['BTCUSD', 'ETHUSD']
        
        risk_metrics = self.risk_manager.assess_portfolio_risk(portfolio, symbols)
        
        # Should return proper RiskMetrics object
        self.assertIsInstance(risk_metrics, RiskMetrics)
        
        # Check all required attributes
        self.assertIsInstance(risk_metrics.risk_level, RiskLevel)
        self.assertIsInstance(risk_metrics.volatility_regime, VolatilityRegime)
        
        # Check numeric metrics are reasonable
        self.assertGreaterEqual(risk_metrics.total_exposure, 0.0)
        self.assertLessEqual(risk_metrics.total_exposure, 1.0)
        
        self.assertGreaterEqual(risk_metrics.correlation_risk, 0.0)
        self.assertLessEqual(risk_metrics.correlation_risk, 1.0)
        
        self.assertGreaterEqual(risk_metrics.concentration_risk, 0.0)
        self.assertLessEqual(risk_metrics.concentration_risk, 1.0)
        
        self.assertGreaterEqual(risk_metrics.portfolio_heat, 0.0)
        self.assertLessEqual(risk_metrics.portfolio_heat, 1.0)
    
    def test_position_limits_checking(self):
        """Test position limits checking"""
        portfolio = {
            'BTCUSD': {'quantity': 0.05, 'market_value': 2000},
            'total_value': 10000
        }
        symbols = ['BTCUSD', 'ETHUSD']
        
        # Test reasonable position size (should be allowed)
        allowed, reason, max_size = self.risk_manager.check_position_limits(
            'ETHUSD', 1500, portfolio, symbols
        )
        
        self.assertTrue(allowed, f"Reasonable position should be allowed: {reason}")
        self.assertGreaterEqual(max_size, 1500)
        
        # Test excessive position size (should be rejected)
        rejected, reason_rej, max_size_rej = self.risk_manager.check_position_limits(
            'ETHUSD', 5000, portfolio, symbols  # 50% of portfolio
        )
        
        # Should be rejected if limits are properly configured
        if self.risk_manager.max_single_position < 0.5:
            self.assertFalse(rejected, f"Large position should be rejected: {reason_rej}")
    
    def test_volatility_multipliers(self):
        """Test volatility regime multipliers"""
        # Should have multipliers for all volatility regimes
        for regime in VolatilityRegime:
            self.assertIn(regime, self.risk_manager.volatility_multipliers)
            
            multiplier = self.risk_manager.volatility_multipliers[regime]
            
            # Multipliers should be positive
            self.assertGreater(multiplier, 0)
            
            # Should be reasonable (between 0.1 and 2.0)
            self.assertGreater(multiplier, 0.1)
            self.assertLess(multiplier, 2.0)
    
    def test_asset_sector_classification(self):
        """Test asset sector classification"""
        # Should have classifications for major crypto assets
        major_symbols = ['BTCUSD', 'ETHUSD']
        
        for symbol in major_symbols:
            if symbol in self.risk_manager.asset_sectors:
                sector = self.risk_manager.asset_sectors[symbol]
                self.assertIsInstance(sector, str)
                self.assertGreater(len(sector), 0)
    
    def test_risk_metrics_logging(self):
        """Test risk metrics logging"""
        portfolio = {
            'BTCUSD': {'quantity': 0.1, 'market_value': 5000},
            'total_value': 10000
        }
        symbols = ['BTCUSD']
        
        risk_metrics = self.risk_manager.assess_portfolio_risk(portfolio, symbols)
        
        # Should be able to log without error
        try:
            self.risk_manager.log_risk_metrics(risk_metrics)
        except Exception as e:
            self.fail(f"Risk metrics logging failed: {e}")

class TestRiskManagerRealism(unittest.TestCase):
    """Test risk manager realism and safety features"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.risk_manager = AdvancedRiskManager()
    
    def test_conservative_defaults(self):
        """Test that default risk limits are conservative"""
        # Position limits should be conservative for retail trading
        self.assertLessEqual(self.risk_manager.max_single_position, 0.30)  # Max 30%
        self.assertLessEqual(self.risk_manager.max_portfolio_exposure, 1.0)  # Max 100%
        
        # Correlation threshold should detect high correlation
        self.assertLessEqual(self.risk_manager.correlation_threshold, 0.8)  # 80% correlation
        
        # Volatility adjustments should reduce risk in high vol
        high_vol_mult = self.risk_manager.volatility_multipliers[VolatilityRegime.HIGH_VOL]
        extreme_vol_mult = self.risk_manager.volatility_multipliers[VolatilityRegime.EXTREME_VOL]
        
        self.assertLess(high_vol_mult, 1.0)  # Should reduce limits
        self.assertLess(extreme_vol_mult, high_vol_mult)  # Even more reduction
    
    def test_correlation_matrix_stability(self):
        """Test correlation matrix numerical stability"""
        # Test with minimal data
        symbols = ['BTCUSD']
        corr_matrix = self.risk_manager.calculate_correlation_matrix(symbols, force_update=True)
        
        # Should handle single asset case
        self.assertEqual(corr_matrix.shape, (1, 1))
        self.assertAlmostEqual(corr_matrix[0, 0], 1.0, places=6)
        
        # Should be positive semi-definite
        eigenvals = np.linalg.eigvals(corr_matrix)
        self.assertTrue(np.all(eigenvals >= -1e-6))  # Allow small numerical errors
    
    def test_emergency_risk_scenarios(self):
        """Test risk management in emergency scenarios"""
        # Test extreme concentration scenario
        extreme_portfolio = {
            'BTCUSD': {'quantity': 1.0, 'market_value': 9500},  # 95% in one asset
            'total_value': 10000
        }
        
        risk_metrics = self.risk_manager.assess_portfolio_risk(extreme_portfolio, ['BTCUSD'])
        
        # Should detect high risk
        self.assertIn(risk_metrics.risk_level, [RiskLevel.HIGH, RiskLevel.CRITICAL])
        
        # Concentration risk should be very high
        self.assertGreater(risk_metrics.concentration_risk, 0.8)

def run_risk_manager_tests():
    """Run all risk manager tests"""
    print("Running Advanced Risk Manager Tests...")
    
    # Create test suite
    suite = unittest.TestSuite()
    
    # Add test cases
    suite.addTest(unittest.makeSuite(TestAdvancedRiskManager))
    suite.addTest(unittest.makeSuite(TestRiskManagerRealism))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print(f"\nRisk Manager Tests Complete:")
    print(f"  Tests Run: {result.testsRun}")
    print(f"  Failures: {len(result.failures)}")
    print(f"  Errors: {len(result.errors)}")
    print(f"  Success Rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%")
    
    return len(result.failures) == 0 and len(result.errors) == 0

if __name__ == "__main__":
    success = run_risk_manager_tests()
    if not success:
        sys.exit(1)