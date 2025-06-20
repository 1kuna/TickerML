#!/usr/bin/env python3
"""
Kafka Producer for Trade Stream Data
Streams real-time individual trades to Kafka topics
"""

import json
import logging
import asyncio
from typing import Dict, Any, List
from datetime import datetime
import yaml
from kafka import KafkaProducer
from kafka.errors import KafkaError
import coloredlogs

# Local imports
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from raspberry_pi.trade_stream import TradeStreamCollector

class TradeProducer:
    """Produces trade data to Kafka topics"""
    
    def __init__(self, config_path: str = 'config/kafka_config.yaml'):
        """Initialize Kafka producer for trade data"""
        self.logger = logging.getLogger(self.__class__.__name__)
        coloredlogs.install(level='INFO', logger=self.logger)
        
        # Load configuration
        with open(config_path, 'r') as f:
            kafka_config = yaml.safe_load(f)
            
        self.config = kafka_config['kafka']
        self.topic = self.config['topics']['trades']
        
        # Initialize Kafka producer with optimizations for high-frequency data
        self.producer = KafkaProducer(
            bootstrap_servers=self.config['brokers'],
            value_serializer=lambda v: json.dumps(v).encode('utf-8'),
            key_serializer=lambda k: k.encode('utf-8') if k else None,
            compression_type='lz4',  # Faster compression for real-time
            batch_size=32768,       # 32KB batches
            linger_ms=10,          # Very low latency
            acks=1,                # Leader acknowledgment only for speed
            retries=3,
            max_in_flight_requests_per_connection=10,
            buffer_memory=33554432  # 32MB buffer for bursts
        )
        
        # Initialize trade stream collector
        self.collector = TradeStreamCollector()
        
        # Trade aggregation for analytics
        self.trade_buffer: List[Dict[str, Any]] = []
        self.buffer_start_time = datetime.utcnow()
        
        self.logger.info(f"Initialized TradeProducer with topic: {self.topic}")
    
    def _create_message(self, trade_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create Kafka message from trade data"""
        return {
            'timestamp': trade_data.get('timestamp'),
            'exchange': trade_data.get('exchange', 'binance'),
            'symbol': trade_data.get('symbol'),
            'trade_id': trade_data.get('trade_id'),
            'price': float(trade_data.get('price', 0)),
            'quantity': float(trade_data.get('quantity', 0)),
            'side': trade_data.get('side'),  # 'buy' or 'sell'
            'maker': trade_data.get('is_maker', False),
            'local_timestamp': datetime.utcnow().isoformat(),
            'metadata': {
                'value_usd': float(trade_data.get('price', 0)) * float(trade_data.get('quantity', 0)),
                'sequence': trade_data.get('sequence')
            }
        }
    
    def _aggregate_trades(self, window_seconds: int = 1) -> Dict[str, Any]:
        """Aggregate trades over a time window for analytics"""
        if not self.trade_buffer:
            return None
            
        current_time = datetime.utcnow()
        window_duration = (current_time - self.buffer_start_time).total_seconds()
        
        if window_duration < window_seconds:
            return None
            
        # Calculate aggregates
        buy_volume = sum(t['quantity'] for t in self.trade_buffer if t['side'] == 'buy')
        sell_volume = sum(t['quantity'] for t in self.trade_buffer if t['side'] == 'sell')
        total_volume = buy_volume + sell_volume
        
        buy_value = sum(t['metadata']['value_usd'] for t in self.trade_buffer if t['side'] == 'buy')
        sell_value = sum(t['metadata']['value_usd'] for t in self.trade_buffer if t['side'] == 'sell')
        
        vwap = sum(t['price'] * t['quantity'] for t in self.trade_buffer) / total_volume if total_volume > 0 else 0
        
        aggregates = {
            'timestamp': current_time.isoformat(),
            'window_seconds': window_seconds,
            'trade_count': len(self.trade_buffer),
            'buy_volume': buy_volume,
            'sell_volume': sell_volume,
            'total_volume': total_volume,
            'buy_value_usd': buy_value,
            'sell_value_usd': sell_value,
            'flow_imbalance': (buy_volume - sell_volume) / total_volume if total_volume > 0 else 0,
            'vwap': vwap,
            'high': max(t['price'] for t in self.trade_buffer),
            'low': min(t['price'] for t in self.trade_buffer),
            'trades_per_second': len(self.trade_buffer) / window_duration
        }
        
        # Clear buffer
        self.trade_buffer = []
        self.buffer_start_time = current_time
        
        return aggregates
    
    def send_trade(self, trade_data: Dict[str, Any]) -> bool:
        """Send trade data to Kafka"""
        try:
            # Create message
            message = self._create_message(trade_data)
            
            # Add to buffer for aggregation
            self.trade_buffer.append(message)
            
            # Create key for partitioning (exchange:symbol)
            key = f"{message['exchange']}:{message['symbol']}"
            
            # Send individual trade
            self.producer.send(
                self.topic,
                key=key,
                value=message
            )
            
            # Check if we should send aggregates
            aggregates = self._aggregate_trades()
            if aggregates:
                self.producer.send(
                    f"{self.topic}-aggregates",
                    key=key,
                    value=aggregates
                )
                self.logger.debug(f"Sent trade aggregates: {aggregates['trade_count']} trades")
            
            return True
            
        except KafkaError as e:
            self.logger.error(f"Failed to send trade data: {e}")
            return False
        except Exception as e:
            self.logger.error(f"Unexpected error sending trade: {e}")
            return False
    
    async def run(self):
        """Main loop to collect and produce trade data"""
        self.logger.info("Starting trade producer...")
        
        # Set up trade callback
        def trade_callback(data: Dict[str, Any]):
            """Callback for trade updates"""
            success = self.send_trade(data)
            if not success:
                self.logger.warning(f"Failed to send trade for {data.get('symbol')}")
        
        # Connect collector callback
        self.collector.trade_callback = trade_callback
        
        try:
            # Start collecting
            await self.collector.connect()
            
            # Periodic aggregate flush
            async def flush_aggregates():
                while True:
                    await asyncio.sleep(1)
                    aggregates = self._aggregate_trades()
                    if aggregates:
                        key = f"{self.collector.exchange}:aggregates"
                        self.producer.send(
                            f"{self.topic}-aggregates",
                            key=key,
                            value=aggregates
                        )
            
            # Run aggregate flusher
            asyncio.create_task(flush_aggregates())
            
            # Keep running
            while True:
                await asyncio.sleep(1)
                
        except KeyboardInterrupt:
            self.logger.info("Shutting down trade producer...")
        except Exception as e:
            self.logger.error(f"Producer error: {e}")
        finally:
            # Clean shutdown
            await self.collector.disconnect()
            self.producer.flush()
            self.producer.close()
            self.logger.info("Trade producer stopped")
    
    def health_check(self) -> Dict[str, Any]:
        """Check producer health"""
        metrics = self.producer.metrics()
        
        return {
            'producer_active': True,
            'buffer_size': len(self.trade_buffer),
            'buffer_duration': (datetime.utcnow() - self.buffer_start_time).total_seconds(),
            'record_send_rate': metrics.get('record-send-rate', 0),
            'record_error_rate': metrics.get('record-error-rate', 0),
            'batch_size_avg': metrics.get('batch-size-avg', 0),
            'compression_rate_avg': metrics.get('compression-rate-avg', 0),
            'collector_connected': self.collector.connected if hasattr(self.collector, 'connected') else False
        }


def main():
    """Main entry point"""
    producer = TradeProducer()
    asyncio.run(producer.run())


if __name__ == "__main__":
    main()