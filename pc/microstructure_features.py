#!/usr/bin/env python3
"""
Advanced Microstructure Features
Implements VPIN, Kyle's Lambda, and other institutional-grade market microstructure indicators
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
from collections import deque
import logging
from scipy import stats
import sqlite3

logger = logging.getLogger(__name__)

@dataclass
class Trade:
    """Trade data structure"""
    timestamp: float
    price: float
    volume: float
    side: str  # 'buy' or 'sell'
    exchange: str = ""
    
@dataclass
class OrderBookSnapshot:
    """Order book snapshot"""
    timestamp: float
    bids: List[Tuple[float, float]]  # [(price, volume), ...]
    asks: List[Tuple[float, float]]  # [(price, volume), ...]
    
@dataclass
class MicrostructureMetrics:
    """Comprehensive microstructure metrics"""
    timestamp: float
    symbol: str
    
    # VPIN metrics
    vpin: float
    vpin_confidence: float
    
    # Kyle's Lambda metrics
    kyle_lambda: float
    price_impact_score: float
    
    # Order flow metrics
    order_flow_imbalance: float
    volume_weighted_imbalance: float
    
    # Toxicity metrics
    trade_flow_toxicity: float
    informed_trading_probability: float

class VPINCalculator:
    """
    Volume-synchronized Probability of Informed Trading (VPIN)
    
    VPIN measures the probability that trades are informed (based on private information)
    by analyzing volume imbalances in volume-time buckets rather than calendar time.
    
    Reference: Easley, D., López de Prado, M., & O'Hara, M. (2012). 
    "The volume clock: Insights into the high-frequency paradigm"
    """
    
    def __init__(self, volume_bucket_size: float = 1000, num_buckets: int = 50):
        """
        Initialize VPIN calculator
        
        Args:
            volume_bucket_size: Target volume per bucket (e.g., 1000 shares)
            num_buckets: Number of volume buckets to maintain for VPIN calculation
        """
        self.volume_bucket_size = volume_bucket_size
        self.num_buckets = num_buckets
        self.volume_buckets = deque(maxlen=num_buckets)
        self.current_bucket = {"buy_volume": 0, "sell_volume": 0, "timestamp": 0}
        
    def add_trade(self, trade: Trade) -> Optional[float]:
        """
        Add trade and calculate VPIN if bucket is complete
        
        Args:
            trade: Trade to add
            
        Returns:
            VPIN value if bucket completed, None otherwise
        """
        # Classify trade direction (use trade.side if available, otherwise use tick rule)
        if trade.side in ['buy', 'sell']:
            side = trade.side
        else:
            # Fallback to tick rule classification
            side = self._classify_trade_direction(trade)
        
        # Add to current bucket
        if side == 'buy':
            self.current_bucket["buy_volume"] += trade.volume
        else:
            self.current_bucket["sell_volume"] += trade.volume
        
        self.current_bucket["timestamp"] = trade.timestamp
        
        # Check if bucket is full
        total_volume = self.current_bucket["buy_volume"] + self.current_bucket["sell_volume"]
        
        if total_volume >= self.volume_bucket_size:
            # Complete current bucket
            self.volume_buckets.append(self.current_bucket.copy())
            
            # Start new bucket with overflow
            overflow_buy = max(0, self.current_bucket["buy_volume"] - self.volume_bucket_size/2)
            overflow_sell = max(0, self.current_bucket["sell_volume"] - self.volume_bucket_size/2)
            
            self.current_bucket = {
                "buy_volume": overflow_buy,
                "sell_volume": overflow_sell,
                "timestamp": trade.timestamp
            }
            
            # Calculate VPIN if we have enough buckets
            if len(self.volume_buckets) >= self.num_buckets:
                return self._calculate_vpin()
        
        return None
    
    def _classify_trade_direction(self, trade: Trade) -> str:
        """
        Classify trade direction using tick rule (simplified)
        In practice, would use more sophisticated classification
        """
        # This is a placeholder - in practice you'd compare to mid-price or use other methods
        return 'buy'  # Simplified for demonstration
    
    def _calculate_vpin(self) -> float:
        """Calculate VPIN over the available volume buckets"""
        if len(self.volume_buckets) < self.num_buckets:
            return 0.0
        
        total_imbalance = 0
        total_volume = 0
        
        for bucket in self.volume_buckets:
            buy_vol = bucket["buy_volume"]
            sell_vol = bucket["sell_volume"]
            volume_imbalance = abs(buy_vol - sell_vol)
            bucket_volume = buy_vol + sell_vol
            
            total_imbalance += volume_imbalance
            total_volume += bucket_volume
        
        if total_volume == 0:
            return 0.0
        
        # VPIN is the average absolute volume imbalance
        vpin = total_imbalance / total_volume
        
        return min(vpin, 1.0)  # Cap at 1.0
    
    def get_vpin_confidence(self) -> float:
        """Calculate confidence in VPIN estimate based on volume dispersion"""
        if len(self.volume_buckets) < 10:
            return 0.0
        
        volumes = [bucket["buy_volume"] + bucket["sell_volume"] for bucket in self.volume_buckets]
        cv = np.std(volumes) / np.mean(volumes) if np.mean(volumes) > 0 else 1.0
        
        # Lower coefficient of variation = higher confidence
        confidence = max(0, 1.0 - cv)
        return confidence

class KyleLambdaCalculator:
    """
    Kyle's Lambda - measures permanent price impact per unit of order flow
    
    Lambda represents the adverse selection component of the bid-ask spread
    and indicates how much prices move permanently per unit of unexpected order flow.
    
    Reference: Kyle, A. S. (1985). "Continuous auctions and insider trading"
    """
    
    def __init__(self, window_minutes: int = 60):
        """
        Initialize Kyle's Lambda calculator
        
        Args:
            window_minutes: Time window for regression analysis
        """
        self.window_minutes = window_minutes
        self.trade_data = deque(maxlen=10000)  # Store recent trades
        self.price_data = deque(maxlen=10000)  # Store price changes
        
    def add_data(self, trades: List[Trade], mid_price_change: float) -> Optional[float]:
        """
        Add trade data and calculate Kyle's Lambda
        
        Args:
            trades: List of trades in the interval
            mid_price_change: Change in mid-price over the interval
            
        Returns:
            Kyle's Lambda if sufficient data, None otherwise
        """
        # Calculate signed order flow
        signed_order_flow = 0
        for trade in trades:
            sign = 1 if trade.side == 'buy' else -1
            signed_order_flow += sign * trade.volume
        
        # Store data point
        data_point = {
            'timestamp': trades[-1].timestamp if trades else 0,
            'signed_order_flow': signed_order_flow,
            'price_change': mid_price_change
        }
        
        self.trade_data.append(data_point)
        
        # Calculate Kyle's Lambda using regression
        return self._calculate_kyle_lambda()
    
    def _calculate_kyle_lambda(self) -> Optional[float]:
        """Calculate Kyle's Lambda using OLS regression"""
        if len(self.trade_data) < 30:  # Need minimum observations
            return None
        
        # Prepare regression data
        cutoff_time = self.trade_data[-1]['timestamp'] - (self.window_minutes * 60)
        recent_data = [d for d in self.trade_data if d['timestamp'] >= cutoff_time]
        
        if len(recent_data) < 20:
            return None
        
        # Extract variables for regression: price_change = lambda * signed_order_flow + error
        y = np.array([d['price_change'] for d in recent_data])
        x = np.array([d['signed_order_flow'] for d in recent_data])
        
        # Remove zero order flow observations
        non_zero_mask = x != 0
        if np.sum(non_zero_mask) < 10:
            return None
        
        y = y[non_zero_mask]
        x = x[non_zero_mask]
        
        # Run regression
        try:
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
            
            # Kyle's Lambda is the slope (price impact per unit order flow)
            # Only return if statistically significant
            if p_value < 0.05 and abs(r_value) > 0.1:
                return abs(slope)  # Take absolute value
            
        except Exception as e:
            logger.warning(f"Kyle's Lambda calculation error: {e}")
        
        return None
    
    def get_price_impact_score(self, recent_kyle_lambda: float) -> float:
        """
        Convert Kyle's Lambda to a standardized price impact score (0-1)
        
        Args:
            recent_kyle_lambda: Recent Kyle's Lambda value
            
        Returns:
            Price impact score between 0 and 1
        """
        if recent_kyle_lambda is None:
            return 0.5  # Neutral score
        
        # Historical lambdas for normalization
        historical_lambdas = [d.get('kyle_lambda', 0) for d in self.trade_data if 'kyle_lambda' in d]
        
        if len(historical_lambdas) < 10:
            return 0.5
        
        # Z-score normalization
        mean_lambda = np.mean(historical_lambdas)
        std_lambda = np.std(historical_lambdas)
        
        if std_lambda == 0:
            return 0.5
        
        z_score = (recent_kyle_lambda - mean_lambda) / std_lambda
        
        # Convert to 0-1 scale using sigmoid
        score = 1 / (1 + np.exp(-z_score))
        
        return score

class OrderFlowAnalyzer:
    """
    Analyzes order flow patterns and imbalances
    """
    
    def __init__(self, window_size: int = 100):
        """
        Initialize order flow analyzer
        
        Args:
            window_size: Number of recent trades to analyze
        """
        self.window_size = window_size
        self.recent_trades = deque(maxlen=window_size)
        
    def add_trade(self, trade: Trade) -> Dict[str, float]:
        """
        Add trade and calculate order flow metrics
        
        Args:
            trade: Trade to add
            
        Returns:
            Dictionary of order flow metrics
        """
        self.recent_trades.append(trade)
        
        if len(self.recent_trades) < 10:
            return self._empty_metrics()
        
        # Calculate metrics
        metrics = {}
        
        # Order flow imbalance
        metrics['order_flow_imbalance'] = self._calculate_order_flow_imbalance()
        
        # Volume-weighted imbalance
        metrics['volume_weighted_imbalance'] = self._calculate_volume_weighted_imbalance()
        
        # Trade flow toxicity
        metrics['trade_flow_toxicity'] = self._calculate_trade_flow_toxicity()
        
        return metrics
    
    def _calculate_order_flow_imbalance(self) -> float:
        """Calculate simple order flow imbalance"""
        buy_count = sum(1 for trade in self.recent_trades if trade.side == 'buy')
        sell_count = len(self.recent_trades) - buy_count
        total_count = buy_count + sell_count
        
        if total_count == 0:
            return 0.0
        
        imbalance = (buy_count - sell_count) / total_count
        return imbalance
    
    def _calculate_volume_weighted_imbalance(self) -> float:
        """Calculate volume-weighted order flow imbalance"""
        buy_volume = sum(trade.volume for trade in self.recent_trades if trade.side == 'buy')
        sell_volume = sum(trade.volume for trade in self.recent_trades if trade.side == 'sell')
        total_volume = buy_volume + sell_volume
        
        if total_volume == 0:
            return 0.0
        
        imbalance = (buy_volume - sell_volume) / total_volume
        return imbalance
    
    def _calculate_trade_flow_toxicity(self) -> float:
        """
        Calculate trade flow toxicity - measure of how much trade flow predicts price changes
        Simplified implementation using recent price impact correlation
        """
        if len(self.recent_trades) < 20:
            return 0.0
        
        # Calculate price changes and signed volumes
        price_changes = []
        signed_volumes = []
        
        for i in range(1, len(self.recent_trades)):
            prev_trade = self.recent_trades[i-1]
            curr_trade = self.recent_trades[i]
            
            price_change = (curr_trade.price - prev_trade.price) / prev_trade.price
            sign = 1 if curr_trade.side == 'buy' else -1
            signed_volume = sign * curr_trade.volume
            
            price_changes.append(price_change)
            signed_volumes.append(signed_volume)
        
        if len(price_changes) < 10:
            return 0.0
        
        # Calculate correlation between signed volume and subsequent price changes
        try:
            correlation = np.corrcoef(signed_volumes[:-1], price_changes[1:])[0, 1]
            return abs(correlation) if not np.isnan(correlation) else 0.0
        except:
            return 0.0
    
    def _empty_metrics(self) -> Dict[str, float]:
        """Return empty metrics dictionary"""
        return {
            'order_flow_imbalance': 0.0,
            'volume_weighted_imbalance': 0.0,
            'trade_flow_toxicity': 0.0
        }

class MicrostructureFeatureEngine:
    """
    Main engine for calculating all microstructure features
    """
    
    def __init__(self, symbol: str):
        """
        Initialize feature engine for a symbol
        
        Args:
            symbol: Trading symbol (e.g., 'BTC/USD')
        """
        self.symbol = symbol
        self.vpin_calculator = VPINCalculator()
        self.kyle_calculator = KyleLambdaCalculator()
        self.order_flow_analyzer = OrderFlowAnalyzer()
        
        self.recent_trades = deque(maxlen=1000)
        self.recent_orderbooks = deque(maxlen=100)
        
    def add_trade(self, trade: Trade) -> Optional[MicrostructureMetrics]:
        """
        Add trade data and calculate features if ready
        
        Args:
            trade: Trade data
            
        Returns:
            MicrostructureMetrics if calculation ready, None otherwise
        """
        self.recent_trades.append(trade)
        
        # Calculate VPIN
        vpin = self.vpin_calculator.add_trade(trade)
        vpin_confidence = self.vpin_calculator.get_vpin_confidence()
        
        # Calculate order flow metrics
        order_flow_metrics = self.order_flow_analyzer.add_trade(trade)
        
        # Calculate Kyle's Lambda (need price change data)
        kyle_lambda = None
        price_impact_score = 0.5
        
        if len(self.recent_trades) >= 2:
            # Simple price change calculation
            prev_price = self.recent_trades[-2].price
            curr_price = self.recent_trades[-1].price
            price_change = (curr_price - prev_price) / prev_price
            
            kyle_lambda = self.kyle_calculator.add_data([trade], price_change)
            if kyle_lambda:
                price_impact_score = self.kyle_calculator.get_price_impact_score(kyle_lambda)
        
        # Only return metrics if we have meaningful data
        if vpin is not None and len(self.recent_trades) >= 50:
            return MicrostructureMetrics(
                timestamp=trade.timestamp,
                symbol=self.symbol,
                vpin=vpin,
                vpin_confidence=vpin_confidence,
                kyle_lambda=kyle_lambda or 0.0,
                price_impact_score=price_impact_score,
                order_flow_imbalance=order_flow_metrics['order_flow_imbalance'],
                volume_weighted_imbalance=order_flow_metrics['volume_weighted_imbalance'],
                trade_flow_toxicity=order_flow_metrics['trade_flow_toxicity'],
                informed_trading_probability=vpin if vpin else 0.0
            )
        
        return None
    
    def add_orderbook(self, orderbook: OrderBookSnapshot) -> None:
        """
        Add order book data for additional context
        
        Args:
            orderbook: Order book snapshot
        """
        self.recent_orderbooks.append(orderbook)
    
    def get_latest_metrics(self) -> Optional[MicrostructureMetrics]:
        """Get the most recent microstructure metrics"""
        if not self.recent_trades:
            return None
        
        latest_trade = self.recent_trades[-1]
        return self.add_trade(latest_trade)

# Database integration
class MicrostructureDataLogger:
    """Logger for microstructure features to database"""
    
    def __init__(self, db_path: str = "data/db/microstructure_features.db"):
        self.db_path = db_path
        self._init_database()
    
    def _init_database(self) -> None:
        """Initialize database tables"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS microstructure_features (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL NOT NULL,
                symbol TEXT NOT NULL,
                vpin REAL,
                vpin_confidence REAL,
                kyle_lambda REAL,
                price_impact_score REAL,
                order_flow_imbalance REAL,
                volume_weighted_imbalance REAL,
                trade_flow_toxicity REAL,
                informed_trading_probability REAL
            )
        ''')
        
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_microstructure_timestamp 
            ON microstructure_features(timestamp);
        ''')
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_microstructure_symbol 
            ON microstructure_features(symbol);
        ''')
        
        conn.commit()
        conn.close()
    
    def log_metrics(self, metrics: MicrostructureMetrics) -> None:
        """Log metrics to database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO microstructure_features (
                timestamp, symbol, vpin, vpin_confidence, kyle_lambda,
                price_impact_score, order_flow_imbalance, volume_weighted_imbalance,
                trade_flow_toxicity, informed_trading_probability
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            metrics.timestamp,
            metrics.symbol,
            metrics.vpin,
            metrics.vpin_confidence,
            metrics.kyle_lambda,
            metrics.price_impact_score,
            metrics.order_flow_imbalance,
            metrics.volume_weighted_imbalance,
            metrics.trade_flow_toxicity,
            metrics.informed_trading_probability
        ))
        
        conn.commit()
        conn.close()

# Example usage
def example_usage():
    """Example usage of microstructure features"""
    import time
    import random
    
    # Initialize feature engine
    engine = MicrostructureFeatureEngine("BTC/USD")
    logger_db = MicrostructureDataLogger()
    
    # Simulate trade data
    base_price = 50000
    current_time = time.time()
    
    for i in range(1000):
        # Simulate random trade
        price_change = random.gauss(0, 0.001)  # 0.1% std price changes
        base_price *= (1 + price_change)
        
        trade = Trade(
            timestamp=current_time + i,
            price=base_price,
            volume=random.uniform(0.1, 10.0),
            side=random.choice(['buy', 'sell'])
        )
        
        # Add trade to engine
        metrics = engine.add_trade(trade)
        
        if metrics:
            print(f"VPIN: {metrics.vpin:.4f}, Kyle's Lambda: {metrics.kyle_lambda:.6f}, "
                  f"Order Flow Imbalance: {metrics.order_flow_imbalance:.4f}")
            
            # Log to database
            logger_db.log_metrics(metrics)

if __name__ == "__main__":
    example_usage()