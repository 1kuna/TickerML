#!/usr/bin/env python3
"""
Kafka Producer for Order Book Data
Streams real-time order book updates to Kafka topics
"""

import json
import logging
import asyncio
from typing import Dict, Any, Optional
from datetime import datetime
import yaml
from kafka import KafkaProducer
from kafka.errors import KafkaError
import coloredlogs

# Local imports
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from raspberry_pi.orderbook_collector import OrderBookCollector

class OrderBookProducer:
    """Produces order book data to Kafka topics"""
    
    def __init__(self, config_path: str = 'config/kafka_config.yaml'):
        """Initialize Kafka producer for order book data"""
        self.logger = logging.getLogger(self.__class__.__name__)
        coloredlogs.install(level='INFO', logger=self.logger)
        
        # Load configuration
        with open(config_path, 'r') as f:
            kafka_config = yaml.safe_load(f)
            
        self.config = kafka_config['kafka']
        self.topic = self.config['topics']['orderbooks']
        
        # Initialize Kafka producer
        self.producer = KafkaProducer(
            bootstrap_servers=self.config['brokers'],
            value_serializer=lambda v: json.dumps(v).encode('utf-8'),
            key_serializer=lambda k: k.encode('utf-8') if k else None,
            compression_type='gzip',
            batch_size=16384,  # 16KB batches
            linger_ms=100,     # Wait up to 100ms for batching
            acks='all',        # Wait for all replicas
            retries=3,
            max_in_flight_requests_per_connection=5
        )
        
        # Initialize order book collector
        self.collector = OrderBookCollector()
        
        self.logger.info(f"Initialized OrderBookProducer with topic: {self.topic}")
    
    def _create_message(self, orderbook_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create Kafka message from order book data"""
        return {
            'timestamp': datetime.utcnow().isoformat(),
            'exchange': orderbook_data.get('exchange', 'binance'),
            'symbol': orderbook_data.get('symbol'),
            'bids': orderbook_data.get('bids', []),
            'asks': orderbook_data.get('asks', []),
            'sequence': orderbook_data.get('sequence'),
            'local_timestamp': orderbook_data.get('local_timestamp'),
            'metadata': {
                'bid_volume': sum(float(bid[1]) for bid in orderbook_data.get('bids', [])),
                'ask_volume': sum(float(ask[1]) for ask in orderbook_data.get('asks', [])),
                'spread': self._calculate_spread(orderbook_data),
                'mid_price': self._calculate_mid_price(orderbook_data),
                'imbalance': self._calculate_imbalance(orderbook_data)
            }
        }
    
    def _calculate_spread(self, orderbook: Dict[str, Any]) -> Optional[float]:
        """Calculate bid-ask spread"""
        bids = orderbook.get('bids', [])
        asks = orderbook.get('asks', [])
        
        if bids and asks:
            best_bid = float(bids[0][0])
            best_ask = float(asks[0][0])
            return best_ask - best_bid
        return None
    
    def _calculate_mid_price(self, orderbook: Dict[str, Any]) -> Optional[float]:
        """Calculate mid price"""
        bids = orderbook.get('bids', [])
        asks = orderbook.get('asks', [])
        
        if bids and asks:
            best_bid = float(bids[0][0])
            best_ask = float(asks[0][0])
            return (best_bid + best_ask) / 2
        return None
    
    def _calculate_imbalance(self, orderbook: Dict[str, Any]) -> Optional[float]:
        """Calculate order book imbalance"""
        bids = orderbook.get('bids', [:5])  # Top 5 levels
        asks = orderbook.get('asks', [:5])
        
        if bids and asks:
            bid_volume = sum(float(bid[1]) for bid in bids)
            ask_volume = sum(float(ask[1]) for ask in asks)
            total_volume = bid_volume + ask_volume
            
            if total_volume > 0:
                return (bid_volume - ask_volume) / total_volume
        return None
    
    def send_orderbook(self, orderbook_data: Dict[str, Any]) -> bool:
        """Send order book data to Kafka"""
        try:
            # Create message
            message = self._create_message(orderbook_data)
            
            # Create key for partitioning (exchange:symbol)
            key = f"{message['exchange']}:{message['symbol']}"
            
            # Send to Kafka
            future = self.producer.send(
                self.topic,
                key=key,
                value=message
            )
            
            # Wait for confirmation (optional, for debugging)
            # record_metadata = future.get(timeout=10)
            # self.logger.debug(f"Sent to {record_metadata.topic}:{record_metadata.partition} @ {record_metadata.offset}")
            
            return True
            
        except KafkaError as e:
            self.logger.error(f"Failed to send order book data: {e}")
            return False
        except Exception as e:
            self.logger.error(f"Unexpected error sending data: {e}")
            return False
    
    async def run(self):
        """Main loop to collect and produce order book data"""
        self.logger.info("Starting order book producer...")
        
        # Set up order book callback
        def orderbook_callback(data: Dict[str, Any]):
            """Callback for order book updates"""
            success = self.send_orderbook(data)
            if success:
                self.logger.debug(f"Sent order book for {data.get('symbol')}")
            else:
                self.logger.warning(f"Failed to send order book for {data.get('symbol')}")
        
        # Connect collector callback
        self.collector.orderbook_callback = orderbook_callback
        
        try:
            # Start collecting
            await self.collector.connect()
            
            # Keep running
            while True:
                await asyncio.sleep(1)
                
        except KeyboardInterrupt:
            self.logger.info("Shutting down order book producer...")
        except Exception as e:
            self.logger.error(f"Producer error: {e}")
        finally:
            # Clean shutdown
            await self.collector.disconnect()
            self.producer.flush()
            self.producer.close()
            self.logger.info("Order book producer stopped")
    
    def health_check(self) -> Dict[str, Any]:
        """Check producer health"""
        metrics = self.producer.metrics()
        
        return {
            'producer_active': True,
            'buffer_usage': metrics.get('buffer-available-bytes', 0),
            'record_send_rate': metrics.get('record-send-rate', 0),
            'record_error_rate': metrics.get('record-error-rate', 0),
            'request_latency_avg': metrics.get('request-latency-avg', 0),
            'collector_connected': self.collector.connected if hasattr(self.collector, 'connected') else False
        }


def main():
    """Main entry point"""
    producer = OrderBookProducer()
    asyncio.run(producer.run())


if __name__ == "__main__":
    main()