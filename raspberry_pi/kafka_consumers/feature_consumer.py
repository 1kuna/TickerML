#!/usr/bin/env python3
"""
Kafka Consumer for Feature Engineering
Consumes order book and trade data to generate trading features
"""

import json
import logging
import asyncio
from typing import Dict, Any, List, Optional, Deque
from datetime import datetime, timedelta
from collections import defaultdict, deque
import yaml
from kafka import KafkaConsumer
from kafka.errors import KafkaError
import coloredlogs
import pandas as pd
import numpy as np

# Local imports
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from pc.enhanced_features import EnhancedFeatureEngineer

class FeatureConsumer:
    """Consumes market data and produces trading features"""
    
    def __init__(self, config_path: str = 'config/kafka_config.yaml'):
        """Initialize Kafka consumer for feature generation"""
        self.logger = logging.getLogger(self.__class__.__name__)
        coloredlogs.install(level='INFO', logger=self.logger)
        
        # Load configuration
        with open(config_path, 'r') as f:
            kafka_config = yaml.safe_load(f)
            
        self.config = kafka_config['kafka']
        
        # Initialize consumers
        self.orderbook_consumer = KafkaConsumer(
            self.config['topics']['orderbooks'],
            bootstrap_servers=self.config['brokers'],
            value_deserializer=lambda m: json.loads(m.decode('utf-8')),
            key_deserializer=lambda m: m.decode('utf-8') if m else None,
            auto_offset_reset='latest',
            enable_auto_commit=True,
            group_id='feature-generator',
            max_poll_interval_ms=300000  # 5 minutes
        )
        
        self.trade_consumer = KafkaConsumer(
            self.config['topics']['trades'],
            bootstrap_servers=self.config['brokers'],
            value_deserializer=lambda m: json.loads(m.decode('utf-8')),
            key_deserializer=lambda m: m.decode('utf-8') if m else None,
            auto_offset_reset='latest',
            enable_auto_commit=True,
            group_id='feature-generator'
        )
        
        # Feature engineer
        self.feature_engineer = EnhancedFeatureEngineer()
        
        # Data buffers for feature calculation
        self.orderbook_buffer: Dict[str, Deque[Dict]] = defaultdict(lambda: deque(maxlen=100))
        self.trade_buffer: Dict[str, Deque[Dict]] = defaultdict(lambda: deque(maxlen=1000))
        
        # Feature cache
        self.feature_cache: Dict[str, Dict] = {}
        self.last_feature_update: Dict[str, datetime] = {}
        
        # Feature update interval (seconds)
        self.feature_update_interval = 5
        
        self.logger.info("Initialized FeatureConsumer")
    
    def process_orderbook(self, message: Dict[str, Any]):
        """Process order book message"""
        try:
            symbol = message.get('symbol')
            if not symbol:
                return
            
            # Add to buffer
            self.orderbook_buffer[symbol].append({
                'timestamp': message['timestamp'],
                'bids': message['bids'],
                'asks': message['asks'],
                'mid_price': message['metadata']['mid_price'],
                'spread': message['metadata']['spread'],
                'imbalance': message['metadata']['imbalance']
            })
            
            self.logger.debug(f"Processed order book for {symbol}")
            
        except Exception as e:
            self.logger.error(f"Error processing order book: {e}")
    
    def process_trade(self, message: Dict[str, Any]):
        """Process trade message"""
        try:
            symbol = message.get('symbol')
            if not symbol:
                return
            
            # Add to buffer
            self.trade_buffer[symbol].append({
                'timestamp': message['timestamp'],
                'price': message['price'],
                'quantity': message['quantity'],
                'side': message['side'],
                'value_usd': message['metadata']['value_usd']
            })
            
            self.logger.debug(f"Processed trade for {symbol}")
            
        except Exception as e:
            self.logger.error(f"Error processing trade: {e}")
    
    def calculate_features(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Calculate features for a symbol"""
        try:
            # Check if we have enough data
            if len(self.orderbook_buffer[symbol]) < 10 or len(self.trade_buffer[symbol]) < 10:
                return None
            
            # Convert buffers to DataFrames
            orderbook_df = pd.DataFrame(list(self.orderbook_buffer[symbol]))
            trade_df = pd.DataFrame(list(self.trade_buffer[symbol]))
            
            # Parse timestamps
            orderbook_df['timestamp'] = pd.to_datetime(orderbook_df['timestamp'])
            trade_df['timestamp'] = pd.to_datetime(trade_df['timestamp'])
            
            # Sort by timestamp
            orderbook_df = orderbook_df.sort_values('timestamp')
            trade_df = trade_df.sort_values('timestamp')
            
            # Calculate microstructure features
            features = {}
            
            # Order book features
            features.update(self._calculate_orderbook_features(orderbook_df))
            
            # Trade features
            features.update(self._calculate_trade_features(trade_df))
            
            # Combined features
            features.update(self._calculate_combined_features(orderbook_df, trade_df))
            
            # Add metadata
            features['symbol'] = symbol
            features['timestamp'] = datetime.utcnow().isoformat()
            features['data_points'] = {
                'orderbook_count': len(orderbook_df),
                'trade_count': len(trade_df)
            }
            
            return features
            
        except Exception as e:
            self.logger.error(f"Error calculating features for {symbol}: {e}")
            return None
    
    def _calculate_orderbook_features(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calculate order book specific features"""
        features = {}
        
        # Recent values
        recent = df.tail(20)
        
        # Price movements
        features['mid_price_return_1m'] = (recent['mid_price'].iloc[-1] / recent['mid_price'].iloc[0] - 1) if len(recent) > 1 else 0
        features['mid_price_volatility'] = recent['mid_price'].pct_change().std() if len(recent) > 2 else 0
        
        # Spread dynamics
        features['spread_mean'] = recent['spread'].mean()
        features['spread_std'] = recent['spread'].std()
        features['spread_current'] = recent['spread'].iloc[-1] if not recent.empty else 0
        
        # Imbalance features
        features['imbalance_mean'] = recent['imbalance'].mean()
        features['imbalance_std'] = recent['imbalance'].std()
        features['imbalance_current'] = recent['imbalance'].iloc[-1] if not recent.empty else 0
        
        # Trend features
        if len(recent) >= 5:
            features['imbalance_trend'] = np.polyfit(range(len(recent)), recent['imbalance'], 1)[0]
            features['spread_trend'] = np.polyfit(range(len(recent)), recent['spread'], 1)[0]
        else:
            features['imbalance_trend'] = 0
            features['spread_trend'] = 0
        
        return features
    
    def _calculate_trade_features(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calculate trade specific features"""
        features = {}
        
        # Recent trades
        recent = df.tail(100)
        
        if recent.empty:
            return features
        
        # Volume features
        features['trade_volume_mean'] = recent['quantity'].mean()
        features['trade_volume_std'] = recent['quantity'].std()
        features['trade_count_1m'] = len(recent)
        
        # Price impact
        features['price_range'] = recent['price'].max() - recent['price'].min()
        features['price_std'] = recent['price'].std()
        
        # Flow features
        buy_trades = recent[recent['side'] == 'buy']
        sell_trades = recent[recent['side'] == 'sell']
        
        buy_volume = buy_trades['quantity'].sum() if not buy_trades.empty else 0
        sell_volume = sell_trades['quantity'].sum() if not sell_trades.empty else 0
        total_volume = buy_volume + sell_volume
        
        features['buy_volume_ratio'] = buy_volume / total_volume if total_volume > 0 else 0.5
        features['trade_flow_imbalance'] = (buy_volume - sell_volume) / total_volume if total_volume > 0 else 0
        
        # Value features
        features['avg_trade_size_usd'] = recent['value_usd'].mean()
        features['total_value_usd'] = recent['value_usd'].sum()
        
        # VWAP calculation
        if total_volume > 0:
            features['vwap'] = (recent['price'] * recent['quantity']).sum() / total_volume
            current_price = recent['price'].iloc[-1]
            features['price_vs_vwap'] = (current_price - features['vwap']) / features['vwap']
        else:
            features['vwap'] = 0
            features['price_vs_vwap'] = 0
        
        return features
    
    def _calculate_combined_features(self, orderbook_df: pd.DataFrame, trade_df: pd.DataFrame) -> Dict[str, float]:
        """Calculate combined order book and trade features"""
        features = {}
        
        if orderbook_df.empty or trade_df.empty:
            return features
        
        # Align timestamps (find closest orderbook for each trade)
        recent_trades = trade_df.tail(50)
        recent_orderbooks = orderbook_df.tail(50)
        
        if len(recent_trades) < 5 or len(recent_orderbooks) < 5:
            return features
        
        # Market impact estimation
        price_changes = []
        for _, trade in recent_trades.iterrows():
            # Find closest orderbook before and after trade
            before_ob = recent_orderbooks[recent_orderbooks['timestamp'] <= trade['timestamp']]
            after_ob = recent_orderbooks[recent_orderbooks['timestamp'] > trade['timestamp']]
            
            if not before_ob.empty and not after_ob.empty:
                before_price = before_ob['mid_price'].iloc[-1]
                after_price = after_ob['mid_price'].iloc[0]
                price_change = (after_price - before_price) / before_price
                
                # Adjust for trade direction
                if trade['side'] == 'sell':
                    price_change = -price_change
                    
                price_changes.append(price_change)
        
        if price_changes:
            features['market_impact_mean'] = np.mean(price_changes)
            features['market_impact_std'] = np.std(price_changes)
        else:
            features['market_impact_mean'] = 0
            features['market_impact_std'] = 0
        
        # Liquidity estimation
        recent_ob = recent_orderbooks.tail(10)
        recent_trades_value = recent_trades['value_usd'].sum()
        avg_spread = recent_ob['spread'].mean()
        
        if avg_spread > 0:
            features['liquidity_score'] = recent_trades_value / avg_spread
        else:
            features['liquidity_score'] = 0
        
        return features
    
    def should_update_features(self, symbol: str) -> bool:
        """Check if features should be updated for symbol"""
        last_update = self.last_feature_update.get(symbol)
        if not last_update:
            return True
        
        time_since_update = (datetime.utcnow() - last_update).total_seconds()
        return time_since_update >= self.feature_update_interval
    
    async def run(self):
        """Main consumer loop"""
        self.logger.info("Starting feature consumer...")
        
        try:
            while True:
                # Poll order book messages
                orderbook_messages = self.orderbook_consumer.poll(timeout_ms=100)
                for topic_partition, messages in orderbook_messages.items():
                    for message in messages:
                        self.process_orderbook(message.value)
                
                # Poll trade messages
                trade_messages = self.trade_consumer.poll(timeout_ms=100)
                for topic_partition, messages in trade_messages.items():
                    for message in messages:
                        self.process_trade(message.value)
                
                # Update features for symbols that need it
                for symbol in set(list(self.orderbook_buffer.keys()) + list(self.trade_buffer.keys())):
                    if self.should_update_features(symbol):
                        features = self.calculate_features(symbol)
                        if features:
                            self.feature_cache[symbol] = features
                            self.last_feature_update[symbol] = datetime.utcnow()
                            self.logger.debug(f"Updated features for {symbol}")
                
                # Small sleep to prevent busy waiting
                await asyncio.sleep(0.1)
                
        except KeyboardInterrupt:
            self.logger.info("Shutting down feature consumer...")
        except Exception as e:
            self.logger.error(f"Consumer error: {e}")
        finally:
            self.orderbook_consumer.close()
            self.trade_consumer.close()
            self.logger.info("Feature consumer stopped")
    
    def get_latest_features(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get latest features for a symbol"""
        return self.feature_cache.get(symbol)
    
    def health_check(self) -> Dict[str, Any]:
        """Check consumer health"""
        return {
            'consumer_active': True,
            'symbols_tracked': len(self.feature_cache),
            'orderbook_buffer_sizes': {k: len(v) for k, v in self.orderbook_buffer.items()},
            'trade_buffer_sizes': {k: len(v) for k, v in self.trade_buffer.items()},
            'feature_cache_age': {
                k: (datetime.utcnow() - v).total_seconds() 
                for k, v in self.last_feature_update.items()
            }
        }


def main():
    """Main entry point"""
    consumer = FeatureConsumer()
    asyncio.run(consumer.run())


if __name__ == "__main__":
    main()