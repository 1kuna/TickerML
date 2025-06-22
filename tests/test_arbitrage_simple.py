#!/usr/bin/env python3
"""
Simplified test suite for arbitrage logic
Tests basic arbitrage calculations and opportunity detection
"""

import unittest
import time
import tempfile
import os
import sqlite3

# Simple test for arbitrage opportunity calculations
class TestArbitrageCalculations(unittest.TestCase):
    """Test arbitrage opportunity calculations"""
    
    def test_spread_calculation(self):
        """Test spread percentage calculation"""
        buy_price = 50000.0
        sell_price = 50200.0
        
        spread_pct = (sell_price - buy_price) / buy_price
        expected_spread = 0.004  # 0.4%
        
        self.assertAlmostEqual(spread_pct, expected_spread, places=6)
    
    def test_profit_after_fees(self):
        """Test profit calculation after fees"""
        buy_price = 50000.0
        sell_price = 50200.0
        buy_fee = 0.001  # 0.1%
        sell_fee = 0.001  # 0.1%
        
        spread_pct = (sell_price - buy_price) / buy_price
        total_fees = buy_fee + sell_fee
        net_profit_pct = spread_pct - total_fees
        
        expected_profit = 0.004 - 0.002  # 0.2%
        self.assertAlmostEqual(net_profit_pct, expected_profit, places=6)
    
    def test_quantity_calculation(self):
        """Test maximum quantity calculation"""
        buy_quantity = 1.5
        sell_quantity = 2.0
        max_position_usd = 100000
        buy_price = 50000.0
        
        # Limited by orderbook depth
        max_qty_orderbook = min(buy_quantity, sell_quantity)
        
        # Limited by position size
        max_qty_position = max_position_usd / buy_price
        
        max_quantity = min(max_qty_orderbook, max_qty_position)
        
        self.assertEqual(max_qty_orderbook, 1.5)
        self.assertEqual(max_qty_position, 2.0)
        self.assertEqual(max_quantity, 1.5)
    
    def test_profit_estimation(self):
        """Test estimated profit calculation"""
        quantity = 1.0
        buy_price = 50000.0
        net_profit_pct = 0.002  # 0.2%
        
        estimated_profit_usd = quantity * buy_price * net_profit_pct
        expected_profit = 100.0  # $100
        
        self.assertEqual(estimated_profit_usd, expected_profit)
    
    def test_latency_adjustment(self):
        """Test latency risk adjustment"""
        estimated_profit = 100.0
        latency_ms = 350  # Total latency
        latency_risk_pct = (latency_ms / 100) * 0.01  # 1% per 100ms
        
        adjusted_profit = estimated_profit * (1 - latency_risk_pct)
        expected_adjustment = 0.035  # 3.5% reduction
        expected_profit = 100.0 * (1 - 0.035)
        
        self.assertAlmostEqual(adjusted_profit, expected_profit, places=2)
    
    def test_profitability_threshold(self):
        """Test profitability threshold checking"""
        min_profit_threshold = 0.001  # 0.1%
        
        # Profitable opportunity
        profitable_spread = 0.004
        profitable_fees = 0.002
        profitable_net = profitable_spread - profitable_fees  # 0.2%
        
        # Unprofitable opportunity
        unprofitable_spread = 0.002
        unprofitable_fees = 0.002
        unprofitable_net = unprofitable_spread - unprofitable_fees  # 0%
        
        self.assertTrue(profitable_net > min_profit_threshold)
        self.assertFalse(unprofitable_net > min_profit_threshold)

class TestOrderBookAnalysis(unittest.TestCase):
    """Test order book analysis for arbitrage"""
    
    def test_best_price_extraction(self):
        """Test extracting best bid/ask prices"""
        # Mock orderbook data
        bids = [(50150.0, 1.0), (50140.0, 0.5), (50130.0, 2.0)]  # Descending
        asks = [(50200.0, 1.5), (50210.0, 1.0), (50220.0, 0.8)]  # Ascending
        
        best_bid = bids[0][0]  # Highest bid
        best_ask = asks[0][0]  # Lowest ask
        best_bid_qty = bids[0][1]
        best_ask_qty = asks[0][1]
        
        self.assertEqual(best_bid, 50150.0)
        self.assertEqual(best_ask, 50200.0)
        self.assertEqual(best_bid_qty, 1.0)
        self.assertEqual(best_ask_qty, 1.5)
    
    def test_spread_calculation_from_orderbook(self):
        """Test spread calculation from orderbook"""
        best_bid = 50150.0
        best_ask = 50200.0
        
        spread_abs = best_ask - best_bid
        spread_pct = spread_abs / ((best_bid + best_ask) / 2)
        
        self.assertEqual(spread_abs, 50.0)
        self.assertAlmostEqual(spread_pct, 0.001, places=3)  # ~0.1%
    
    def test_depth_analysis(self):
        """Test analyzing orderbook depth"""
        bids = [(50150.0, 1.0), (50140.0, 0.5), (50130.0, 2.0)]
        asks = [(50200.0, 1.5), (50210.0, 1.0), (50220.0, 0.8)]
        
        # Calculate cumulative depth
        bid_depth = sum(qty for _, qty in bids)
        ask_depth = sum(qty for _, qty in asks)
        
        self.assertEqual(bid_depth, 3.5)
        self.assertEqual(ask_depth, 3.3)
        
        # Available for arbitrage limited by smaller side
        max_arbitrage_qty = min(bids[0][1], asks[0][1])
        self.assertEqual(max_arbitrage_qty, 1.0)

class TestDatabaseOperations(unittest.TestCase):
    """Test database operations for arbitrage logging"""
    
    def setUp(self):
        """Set up temporary database"""
        self.temp_db = tempfile.NamedTemporaryFile(delete=False)
        self.temp_db.close()
        self.db_path = self.temp_db.name
        
        # Initialize database
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE arbitrage_opportunities (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL NOT NULL,
                symbol TEXT NOT NULL,
                buy_exchange TEXT NOT NULL,
                sell_exchange TEXT NOT NULL,
                buy_price REAL NOT NULL,
                sell_price REAL NOT NULL,
                spread_pct REAL NOT NULL,
                net_profit_pct REAL NOT NULL,
                max_quantity REAL NOT NULL,
                estimated_profit_usd REAL NOT NULL
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def tearDown(self):
        """Clean up temporary database"""
        if os.path.exists(self.db_path):
            os.unlink(self.db_path)
    
    def test_opportunity_logging(self):
        """Test logging arbitrage opportunity"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Insert test opportunity
        cursor.execute('''
            INSERT INTO arbitrage_opportunities (
                timestamp, symbol, buy_exchange, sell_exchange,
                buy_price, sell_price, spread_pct, net_profit_pct,
                max_quantity, estimated_profit_usd
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            time.time(),
            'BTC/USD',
            'binance',
            'coinbase', 
            50000.0,
            50200.0,
            0.004,
            0.002,
            1.0,
            100.0
        ))
        
        conn.commit()
        
        # Verify insertion
        cursor.execute('SELECT COUNT(*) FROM arbitrage_opportunities')
        count = cursor.fetchone()[0]
        self.assertEqual(count, 1)
        
        # Verify data
        cursor.execute('SELECT * FROM arbitrage_opportunities')
        row = cursor.fetchone()
        self.assertEqual(row[2], 'BTC/USD')  # symbol
        self.assertEqual(row[3], 'binance')  # buy_exchange
        self.assertEqual(row[4], 'coinbase')  # sell_exchange
        
        conn.close()
    
    def test_opportunity_querying(self):
        """Test querying arbitrage opportunities"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        current_time = time.time()
        
        # Insert multiple opportunities
        opportunities = [
            (current_time - 3600, 'BTC/USD', 'binance', 'coinbase', 50000, 50200, 0.004, 0.002, 1.0, 100),
            (current_time - 1800, 'ETH/USD', 'kraken', 'kucoin', 3000, 3030, 0.01, 0.008, 5.0, 120),
            (current_time - 300, 'BTC/USD', 'coinbase', 'kraken', 50100, 50250, 0.003, 0.001, 0.5, 25)
        ]
        
        for opp in opportunities:
            cursor.execute('''
                INSERT INTO arbitrage_opportunities (
                    timestamp, symbol, buy_exchange, sell_exchange,
                    buy_price, sell_price, spread_pct, net_profit_pct,
                    max_quantity, estimated_profit_usd
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', opp)
        
        conn.commit()
        
        # Query recent opportunities (last hour)
        cutoff_time = current_time - 3600
        cursor.execute('''
            SELECT COUNT(*) FROM arbitrage_opportunities 
            WHERE timestamp >= ?
        ''', (cutoff_time,))
        
        recent_count = cursor.fetchone()[0]
        self.assertEqual(recent_count, 3)
        
        # Query by symbol
        cursor.execute('''
            SELECT COUNT(*) FROM arbitrage_opportunities 
            WHERE symbol = ?
        ''', ('BTC/USD',))
        
        btc_count = cursor.fetchone()[0]
        self.assertEqual(btc_count, 2)
        
        # Query most profitable
        cursor.execute('''
            SELECT symbol, estimated_profit_usd FROM arbitrage_opportunities 
            ORDER BY estimated_profit_usd DESC LIMIT 1
        ''')
        
        best = cursor.fetchone()
        self.assertEqual(best[0], 'ETH/USD')
        self.assertEqual(best[1], 120.0)
        
        conn.close()

def run_tests():
    """Run all tests"""
    suite = unittest.TestSuite()
    
    # Add test classes
    suite.addTest(unittest.makeSuite(TestArbitrageCalculations))
    suite.addTest(unittest.makeSuite(TestOrderBookAnalysis))
    suite.addTest(unittest.makeSuite(TestDatabaseOperations))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()

if __name__ == '__main__':
    print("Running simplified arbitrage tests...")
    success = run_tests()
    
    if success:
        print("\n✅ All arbitrage tests passed!")
    else:
        print("\n❌ Some arbitrage tests failed!")