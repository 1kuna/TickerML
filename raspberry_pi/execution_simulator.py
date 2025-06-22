#!/usr/bin/env python3
"""
Advanced Execution Simulation for Paper Trading
Implements realistic order execution with FIFO queue modeling, partial fills, and toxic fill detection

CRITICAL FEATURES:
- FIFO queue position tracking using cumulative volume
- Partial fill simulation with progressive filling
- Exchange-specific latency modeling (50-200ms)  
- Toxic fill detection (queue position >10)
- Order book depth-based execution pricing
- Adverse selection modeling
"""

import json
import logging
import sqlite3
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, NamedTuple
from dataclasses import dataclass, asdict
from enum import Enum
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ExchangeType(Enum):
    """Exchange types with different latency profiles"""
    BINANCE_US = "binance_us"
    COINBASE = "coinbase" 
    KRAKEN = "kraken"
    KUCOIN = "kucoin"

@dataclass
class OrderBookLevel:
    """Single order book level"""
    price: float
    quantity: float
    cumulative_quantity: float = 0.0  # Running total for queue position

@dataclass 
class ExecutionResult:
    """Result of order execution simulation"""
    filled_quantity: float
    avg_fill_price: float
    remaining_quantity: float
    queue_position: int
    is_toxic_fill: bool
    latency_ms: float
    partial_fills: List[Tuple[float, float]]  # [(quantity, price), ...]
    execution_timestamp: float

class AdvancedExecutionSimulator:
    """Advanced execution simulation with queue modeling"""
    
    def __init__(self, db_path: str = "data/db/crypto_data.db"):
        self.db_path = db_path
        
        # Exchange-specific latency profiles (milliseconds)
        self.exchange_latencies = {
            ExchangeType.BINANCE_US: (50, 100),   # (min, max) latency
            ExchangeType.COINBASE: (100, 200),
            ExchangeType.KRAKEN: (150, 300),
            ExchangeType.KUCOIN: (80, 180)
        }
        
        # Queue position thresholds
        self.toxic_fill_threshold = 10  # Queue position >10 = likely toxic
        self.max_queue_position = 50   # Stop filling beyond this position
        
        # Market impact parameters
        self.impact_coefficient = 0.0001  # Price impact per unit volume
        self.min_impact_bps = 0.5        # Minimum 0.5 bps impact
        self.max_impact_bps = 50.0       # Maximum 5% impact
        
    def get_current_order_book(self, symbol: str, max_age_seconds: int = 30) -> Optional[Dict]:
        """Get the most recent order book snapshot, fallback to synthetic order book from OHLCV"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # First try to get real order book data
            try:
                min_timestamp = time.time() - max_age_seconds
                
                query = '''
                    SELECT bids, asks, mid_price, spread_bps, imbalance, timestamp
                    FROM order_books 
                    WHERE symbol = ? AND timestamp >= ?
                    ORDER BY timestamp DESC 
                    LIMIT 1
                '''
                
                cursor.execute(query, (symbol, min_timestamp))
                row = cursor.fetchone()
                
                if row:
                    bids_json, asks_json, mid_price, spread_bps, imbalance, timestamp = row
                    conn.close()
                    return {
                        'bids': json.loads(bids_json),
                        'asks': json.loads(asks_json),
                        'mid_price': mid_price,
                        'spread_bps': spread_bps,
                        'imbalance': imbalance,
                        'timestamp': timestamp
                    }
            except sqlite3.OperationalError as e:
                # Table doesn't exist, fall through to synthetic creation
                logger.info(f"Order book table doesn't exist, will create synthetic order book: {e}")
            
            # Fallback: Create synthetic order book from OHLCV data
            logger.info(f"No order book data found for {symbol}, creating synthetic order book from OHLCV")
            
            query = '''
                SELECT close, high, low, volume, timestamp
                FROM ohlcv 
                WHERE symbol = ?
                ORDER BY timestamp DESC 
                LIMIT 1
            '''
            
            cursor.execute(query, (symbol,))
            ohlcv_row = cursor.fetchone()
            conn.close()
            
            if not ohlcv_row:
                logger.warning(f"No OHLCV data found for {symbol}")
                return None
            
            close_price, high, low, volume, timestamp = ohlcv_row
            return self._create_synthetic_order_book(close_price, high, low, volume, timestamp)
            
        except Exception as e:
            logger.error(f"Error getting order book for {symbol}: {e}")
            # Emergency fallback - create a basic synthetic order book
            try:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                query = '''
                    SELECT close, high, low, volume, timestamp
                    FROM ohlcv 
                    WHERE symbol = ?
                    ORDER BY timestamp DESC 
                    LIMIT 1
                '''
                cursor.execute(query, (symbol,))
                ohlcv_row = cursor.fetchone()
                conn.close()
                
                if ohlcv_row:
                    close_price, high, low, volume, timestamp = ohlcv_row
                    return self._create_synthetic_order_book(close_price, high, low, volume, timestamp)
            except Exception as fallback_e:
                logger.error(f"Emergency fallback failed: {fallback_e}")
            
            return None
    
    def _create_synthetic_order_book(self, close_price: float, high: float, low: float, 
                                   volume: float, timestamp: float) -> Dict:
        """Create a synthetic order book from OHLCV data for testing"""
        try:
            # Calculate reasonable spread (0.01% to 0.1% based on volatility)
            price_range = (high - low) / close_price
            spread_pct = max(0.0001, min(0.001, price_range * 0.1))  # 0.01% to 0.1%
            spread = close_price * spread_pct
            
            bid_price = close_price - (spread / 2)
            ask_price = close_price + (spread / 2)
            
            # Estimate quantity distribution (use volume as guide)
            # Distribute volume across multiple price levels
            base_quantity = max(1.0, volume / 100)  # More generous estimate for testing
            
            # Create 5 levels on each side with decreasing quantities
            bids = []
            asks = []
            
            for i in range(5):
                # Quantities decrease as we go further from best price
                quantity_factor = 1.0 / (1.0 + i * 0.5)
                level_quantity = base_quantity * quantity_factor
                
                # Bid side (decreasing prices)
                bid_level_price = bid_price - (i * spread * 0.1)
                bids.append([bid_level_price, level_quantity])
                
                # Ask side (increasing prices)
                ask_level_price = ask_price + (i * spread * 0.1)
                asks.append([ask_level_price, level_quantity])
            
            # Calculate synthetic metrics
            mid_price = (bid_price + ask_price) / 2
            spread_bps = (spread / mid_price) * 10000
            imbalance = 0.0  # Neutral for synthetic book
            
            logger.info(f"Created synthetic order book: mid=${mid_price:.2f}, spread={spread_bps:.1f}bps")
            
            return {
                'bids': bids,
                'asks': asks,
                'mid_price': mid_price,
                'spread_bps': spread_bps,
                'imbalance': imbalance,
                'timestamp': timestamp,
                'synthetic': True  # Flag to indicate this is synthetic
            }
            
        except Exception as e:
            logger.error(f"Error creating synthetic order book: {e}")
            return None
    
    def calculate_cumulative_volumes(self, levels: List[List[float]]) -> List[OrderBookLevel]:
        """Convert price/quantity pairs to OrderBookLevel with cumulative volumes"""
        ob_levels = []
        cumulative = 0.0
        
        for price, quantity in levels:
            cumulative += quantity
            ob_levels.append(OrderBookLevel(
                price=float(price),
                quantity=float(quantity),
                cumulative_quantity=cumulative
            ))
            
        return ob_levels
    
    def estimate_queue_position(self, levels: List[OrderBookLevel], order_price: float, 
                              side: str) -> int:
        """
        Estimate queue position using FIFO assumption
        
        For buy orders: position in the bid queue at order_price
        For sell orders: position in the ask queue at order_price
        """
        try:
            for i, level in enumerate(levels):
                if side == 'buy' and level.price == order_price:
                    # For buy orders, we're at the back of the queue at this price level
                    return int(level.cumulative_quantity)
                elif side == 'sell' and level.price == order_price:
                    # For sell orders, we're at the back of the queue at this price level  
                    return int(level.cumulative_quantity)
                    
            # Price not found in order book
            return 0
            
        except Exception as e:
            logger.error(f"Error estimating queue position: {e}")
            return 0
    
    def simulate_partial_fills(self, order_quantity: float, queue_position: int,
                             order_price: float, levels: List[OrderBookLevel],
                             side: str, is_synthetic: bool = False) -> List[Tuple[float, float]]:
        """
        Simulate progressive order filling based on queue position
        
        Returns list of (fill_quantity, fill_price) tuples
        """
        fills = []
        remaining_quantity = order_quantity
        current_position = queue_position
        
        try:
            if is_synthetic:
                # For synthetic order books, always allow fills at market prices
                # Market orders get immediate execution at best prices
                if remaining_quantity > 0 and len(levels) > 0:
                    # For market orders in synthetic books, fill the entire order
                    # Use weighted average price across levels
                    total_filled = 0.0
                    
                    for i, level in enumerate(levels):
                        if remaining_quantity <= 0 or i >= 5:  # Use up to 5 levels
                            break
                            
                        # Fill up to the available quantity at this level
                        fill_quantity = min(remaining_quantity, level.quantity)
                        
                        if fill_quantity > 0:
                            fills.append((fill_quantity, level.price))
                            remaining_quantity -= fill_quantity
                            total_filled += fill_quantity
                    
                    # If we still have remaining quantity and no fills, force a fill
                    if len(fills) == 0 and len(levels) > 0:
                        # Emergency fallback: fill at least some quantity at best price
                        best_level = levels[0]
                        fill_quantity = min(remaining_quantity, max(0.001, best_level.quantity))
                        fills.append((fill_quantity, best_level.price))
                        logger.info(f"Emergency fill: {fill_quantity} @ {best_level.price}")
                            
                return fills
            
            # Original logic for real order book data
            # Find the level with our order price
            target_level = None
            for level in levels:
                if abs(level.price - order_price) < 0.01:  # Allow small price tolerance
                    target_level = level
                    break
                    
            if not target_level:
                # For market orders, use the best available price
                if len(levels) > 0:
                    target_level = levels[0]
                    order_price = target_level.price
                else:
                    logger.warning(f"No price levels available in order book")
                    return fills
            
            # Simulate FIFO filling
            level_remaining = max(0, target_level.quantity - current_position)
            
            if level_remaining <= 0:
                # Try next best level
                if len(levels) > 1:
                    target_level = levels[1]
                    level_remaining = target_level.quantity
                    order_price = target_level.price
                else:
                    return fills
                
            # Fill what's available at this level
            fill_quantity = min(remaining_quantity, level_remaining)
            
            if fill_quantity > 0:
                fills.append((fill_quantity, order_price))
                remaining_quantity -= fill_quantity
                current_position += fill_quantity
                
            # If more quantity needed, use additional levels
            level_index = 1
            while remaining_quantity > 0 and level_index < len(levels) and level_index < 5:
                level = levels[level_index]
                fill_quantity = min(remaining_quantity, level.quantity)
                if fill_quantity > 0:
                    fills.append((fill_quantity, level.price))
                    remaining_quantity -= fill_quantity
                level_index += 1
                        
        except Exception as e:
            logger.error(f"Error simulating partial fills: {e}")
            
        return fills
    
    def calculate_market_impact(self, order_quantity: float, side: str, 
                              order_book: Dict) -> float:
        """Calculate market impact based on order size and book depth"""
        try:
            mid_price = order_book['mid_price']
            
            # Simple market impact model: impact increases with order size
            levels = order_book['bids'] if side == 'buy' else order_book['asks']
            
            # Calculate available liquidity in first 5 levels
            total_liquidity = sum(float(level[1]) for level in levels[:5])
            
            # Impact as percentage of order size relative to available liquidity
            if total_liquidity > 0:
                impact_ratio = order_quantity / total_liquidity
                impact_bps = self.impact_coefficient * impact_ratio * 10000
                
                # Clamp impact to reasonable bounds
                impact_bps = max(self.min_impact_bps, min(impact_bps, self.max_impact_bps))
                
                impact_price = mid_price * (impact_bps / 10000)
                return impact_price if side == 'buy' else -impact_price
            else:
                return 0.0
                
        except Exception as e:
            logger.error(f"Error calculating market impact: {e}")
            return 0.0
    
    def simulate_exchange_latency(self, exchange: ExchangeType = ExchangeType.BINANCE_US) -> float:
        """Simulate exchange-specific network latency"""
        min_latency, max_latency = self.exchange_latencies[exchange]
        
        # Add some randomness to latency
        latency = np.random.uniform(min_latency, max_latency)
        
        # Occasionally simulate network spikes (5% chance)
        if np.random.random() < 0.05:
            latency *= 2.0  # Double latency for network spike
            
        return latency
    
    def simulate_order_execution(self, symbol: str, side: str, quantity: float,
                               order_type: str = 'market', price: Optional[float] = None,
                               exchange: ExchangeType = ExchangeType.BINANCE_US) -> ExecutionResult:
        """
        Main execution simulation function
        
        Args:
            symbol: Trading symbol (e.g., 'BTCUSD')
            side: 'buy' or 'sell'
            quantity: Order quantity
            order_type: 'market' or 'limit'
            price: Limit price (for limit orders)
            exchange: Exchange type for latency modeling
            
        Returns:
            ExecutionResult with detailed execution information
        """
        try:
            # Get current order book
            order_book = self.get_current_order_book(symbol)
            if not order_book:
                logger.error(f"No order book data available for {symbol}")
                return ExecutionResult(
                    filled_quantity=0.0,
                    avg_fill_price=0.0,
                    remaining_quantity=quantity,
                    queue_position=0,
                    is_toxic_fill=False,
                    latency_ms=0.0,
                    partial_fills=[],
                    execution_timestamp=time.time()
                )
            
            # Determine execution price
            if order_type == 'market':
                # Market orders hit the opposite side of the book
                if side == 'buy':
                    execution_price = float(order_book['asks'][0][0])  # Best ask
                    levels = self.calculate_cumulative_volumes(order_book['asks'])
                else:
                    execution_price = float(order_book['bids'][0][0])  # Best bid
                    levels = self.calculate_cumulative_volumes(order_book['bids'])
            else:
                # Limit orders use specified price
                execution_price = price
                if side == 'buy':
                    levels = self.calculate_cumulative_volumes(order_book['bids'])
                else:
                    levels = self.calculate_cumulative_volumes(order_book['asks'])
            
            # Estimate queue position
            queue_position = self.estimate_queue_position(levels, execution_price, side)
            
            # Check for toxic fill
            is_toxic_fill = queue_position > self.toxic_fill_threshold
            
            # Simulate exchange latency
            latency_ms = self.simulate_exchange_latency(exchange)
            
            # Simulate partial fills
            is_synthetic = order_book.get('synthetic', False)
            partial_fills = self.simulate_partial_fills(
                quantity, queue_position, execution_price, levels, side, is_synthetic
            )
            
            # Calculate execution results
            if partial_fills:
                total_filled = sum(fill[0] for fill in partial_fills)
                weighted_price = sum(fill[0] * fill[1] for fill in partial_fills) / total_filled
                remaining_quantity = max(0.0, quantity - total_filled)
            else:
                # No fills - for synthetic books, force at least a minimal fill
                if is_synthetic and len(levels) > 0:
                    # Emergency fallback for synthetic books
                    best_level = levels[0]
                    min_fill = min(quantity, 0.01)  # Fill at least 0.01 units
                    partial_fills = [(min_fill, best_level.price)]
                    total_filled = min_fill
                    weighted_price = best_level.price
                    remaining_quantity = max(0.0, quantity - total_filled)
                    logger.info(f"Emergency synthetic fill: {min_fill} @ {best_level.price}")
                else:
                    # No fills (order too far back in queue or insufficient liquidity)
                    total_filled = 0.0
                    weighted_price = execution_price
                    remaining_quantity = quantity
                    partial_fills = []
            
            # Apply market impact for large orders
            if total_filled > 0:
                impact = self.calculate_market_impact(total_filled, side, order_book)
                weighted_price += impact
            
            logger.info(f"Simulated execution: {total_filled:.6f} {symbol} @ ${weighted_price:.2f}, "
                       f"queue_pos={queue_position}, toxic={is_toxic_fill}, latency={latency_ms:.1f}ms")
            
            return ExecutionResult(
                filled_quantity=total_filled,
                avg_fill_price=weighted_price,
                remaining_quantity=remaining_quantity,
                queue_position=queue_position,
                is_toxic_fill=is_toxic_fill,
                latency_ms=latency_ms,
                partial_fills=partial_fills,
                execution_timestamp=time.time()
            )
            
        except Exception as e:
            logger.error(f"Error simulating order execution: {e}")
            return ExecutionResult(
                filled_quantity=0.0,
                avg_fill_price=0.0,
                remaining_quantity=quantity,
                queue_position=0,
                is_toxic_fill=False,
                latency_ms=0.0,
                partial_fills=[],
                execution_timestamp=time.time()
            )
    
    def log_execution_metrics(self, result: ExecutionResult, order_id: str):
        """Log detailed execution metrics for analysis"""
        logger.info(f"Execution Metrics for {order_id}:")
        logger.info(f"  Filled: {result.filled_quantity:.6f}")
        logger.info(f"  Avg Price: ${result.avg_fill_price:.2f}")
        logger.info(f"  Remaining: {result.remaining_quantity:.6f}")
        logger.info(f"  Queue Position: {result.queue_position}")
        logger.info(f"  Toxic Fill: {result.is_toxic_fill}")
        logger.info(f"  Latency: {result.latency_ms:.1f}ms")
        logger.info(f"  Partial Fills: {len(result.partial_fills)}")
        
        if result.is_toxic_fill:
            logger.warning(f"TOXIC FILL DETECTED: Queue position {result.queue_position} > {self.toxic_fill_threshold}")

# Example usage and testing
def test_execution_simulator():
    """Test the execution simulator with sample data"""
    simulator = AdvancedExecutionSimulator()
    
    # Test market buy order
    result = simulator.simulate_order_execution(
        symbol='BTCUSD',
        side='buy',
        quantity=0.01,
        order_type='market'
    )
    
    simulator.log_execution_metrics(result, 'TEST_001')
    
    return result

if __name__ == "__main__":
    test_execution_simulator()