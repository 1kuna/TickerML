#!/usr/bin/env python3
"""
Multi-Exchange Support Module
Provides unified interface for trading across multiple cryptocurrency exchanges
"""

from .base import (
    ExchangeInterface,
    ExchangeConfig,
    OrderBook,
    Trade,
    Order,
    Balance,
    OrderType,
    OrderSide,
    OrderStatus
)

from .binance import BinanceExchange
from .coinbase import CoinbaseExchange
from .kraken import KrakenExchange
from .kucoin import KuCoinExchange

# Exchange registry
EXCHANGES = {
    'binance': BinanceExchange,
    'coinbase': CoinbaseExchange,
    'kraken': KrakenExchange,
    'kucoin': KuCoinExchange,
}

def create_exchange(exchange_name: str, config: ExchangeConfig) -> ExchangeInterface:
    """
    Factory function to create exchange instances
    
    Args:
        exchange_name: Name of the exchange ('binance', 'coinbase', 'kraken', 'kucoin')
        config: Exchange configuration
        
    Returns:
        ExchangeInterface: Configured exchange instance
        
    Raises:
        ValueError: If exchange name is not supported
    """
    if exchange_name.lower() not in EXCHANGES:
        available = ', '.join(EXCHANGES.keys())
        raise ValueError(f"Unsupported exchange: {exchange_name}. Available: {available}")
    
    exchange_class = EXCHANGES[exchange_name.lower()]
    return exchange_class(config)

__all__ = [
    'ExchangeInterface',
    'ExchangeConfig', 
    'OrderBook',
    'Trade',
    'Order',
    'Balance',
    'OrderType',
    'OrderSide',
    'OrderStatus',
    'BinanceExchange',
    'CoinbaseExchange', 
    'KrakenExchange',
    'KuCoinExchange',
    'EXCHANGES',
    'create_exchange'
]