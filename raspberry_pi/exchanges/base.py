#!/usr/bin/env python3
"""
Base Exchange Interface for Multi-Exchange Support
Provides abstract base class for all exchange implementations
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any, Callable
from enum import Enum
import asyncio
import logging
from datetime import datetime
import numpy as np

logger = logging.getLogger(__name__)

class OrderType(Enum):
    """Order types supported across exchanges"""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"

class OrderSide(Enum):
    """Order side"""
    BUY = "buy"
    SELL = "sell"

class OrderStatus(Enum):
    """Order status"""
    PENDING = "pending"
    OPEN = "open"
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    EXPIRED = "expired"

@dataclass
class OrderBook:
    """Unified order book representation"""
    exchange: str
    symbol: str
    timestamp: float
    bids: List[Tuple[float, float]]  # [(price, quantity), ...]
    asks: List[Tuple[float, float]]  # [(price, quantity), ...]
    sequence: Optional[int] = None    # For order book updates
    
    @property
    def mid_price(self) -> float:
        """Calculate mid price"""
        if self.bids and self.asks:
            return (self.bids[0][0] + self.asks[0][0]) / 2
        return 0.0
    
    @property
    def spread(self) -> float:
        """Calculate bid-ask spread"""
        if self.bids and self.asks:
            return self.asks[0][0] - self.bids[0][0]
        return float('inf')
    
    @property
    def imbalance(self) -> float:
        """Calculate order book imbalance"""
        if not self.bids or not self.asks:
            return 0.0
        
        bid_volume = sum(qty for _, qty in self.bids[:5])
        ask_volume = sum(qty for _, qty in self.asks[:5])
        total_volume = bid_volume + ask_volume
        
        if total_volume == 0:
            return 0.0
        
        return (bid_volume - ask_volume) / total_volume

@dataclass
class Trade:
    """Unified trade representation"""
    exchange: str
    symbol: str
    timestamp: float
    price: float
    quantity: float
    side: OrderSide
    trade_id: Optional[str] = None
    
@dataclass
class Order:
    """Unified order representation"""
    exchange: str
    symbol: str
    order_id: str
    client_order_id: Optional[str]
    timestamp: float
    type: OrderType
    side: OrderSide
    price: Optional[float]
    quantity: float
    filled_quantity: float
    status: OrderStatus
    fee: Optional[float] = None
    fee_currency: Optional[str] = None

@dataclass
class Balance:
    """Account balance"""
    currency: str
    free: float      # Available balance
    locked: float    # In orders
    total: float     # free + locked

@dataclass
class ExchangeConfig:
    """Exchange configuration"""
    name: str
    api_key: Optional[str] = None
    api_secret: Optional[str] = None
    passphrase: Optional[str] = None  # For exchanges like Coinbase
    testnet: bool = False
    rate_limit: int = 10  # Requests per second
    ws_url: Optional[str] = None
    rest_url: Optional[str] = None
    
class ExchangeInterface(ABC):
    """Abstract base class for all exchange implementations"""
    
    def __init__(self, config: ExchangeConfig):
        self.config = config
        self.name = config.name
        self._ws_connected = False
        self._rest_session = None
        self._ws_connection = None
        self._orderbook_callbacks: Dict[str, List[Callable]] = {}
        self._trade_callbacks: Dict[str, List[Callable]] = {}
        
    @abstractmethod
    async def connect(self) -> None:
        """Initialize connection to exchange"""
        pass
    
    @abstractmethod
    async def disconnect(self) -> None:
        """Close all connections"""
        pass
    
    @abstractmethod
    async def get_symbols(self) -> List[str]:
        """Get list of available trading symbols"""
        pass
    
    @abstractmethod
    async def get_orderbook(self, symbol: str, depth: int = 20) -> OrderBook:
        """Get current order book snapshot"""
        pass
    
    @abstractmethod
    async def subscribe_orderbook(self, symbol: str, callback: Callable[[OrderBook], None]) -> None:
        """Subscribe to real-time order book updates"""
        pass
    
    @abstractmethod
    async def unsubscribe_orderbook(self, symbol: str) -> None:
        """Unsubscribe from order book updates"""
        pass
    
    @abstractmethod
    async def subscribe_trades(self, symbol: str, callback: Callable[[Trade], None]) -> None:
        """Subscribe to real-time trades"""
        pass
    
    @abstractmethod
    async def unsubscribe_trades(self, symbol: str) -> None:
        """Unsubscribe from trades"""
        pass
    
    @abstractmethod
    async def get_balance(self) -> Dict[str, Balance]:
        """Get account balances"""
        pass
    
    @abstractmethod
    async def place_order(
        self,
        symbol: str,
        side: OrderSide,
        order_type: OrderType,
        quantity: float,
        price: Optional[float] = None,
        client_order_id: Optional[str] = None
    ) -> Order:
        """Place an order"""
        pass
    
    @abstractmethod
    async def cancel_order(self, symbol: str, order_id: str) -> bool:
        """Cancel an order"""
        pass
    
    @abstractmethod
    async def get_order(self, symbol: str, order_id: str) -> Order:
        """Get order status"""
        pass
    
    @abstractmethod
    async def get_open_orders(self, symbol: Optional[str] = None) -> List[Order]:
        """Get all open orders"""
        pass
    
    @abstractmethod
    async def get_fees(self, symbol: str) -> Dict[str, float]:
        """Get trading fees for symbol"""
        pass
    
    @abstractmethod
    async def get_server_time(self) -> float:
        """Get exchange server time"""
        pass
    
    # Helper methods common to all exchanges
    
    def normalize_symbol(self, symbol: str) -> str:
        """Normalize symbol format for the exchange"""
        # Default implementation, can be overridden
        return symbol.upper().replace('/', '').replace('-', '')
    
    def denormalize_symbol(self, symbol: str) -> str:
        """Convert from normalized format to exchange format"""
        # Default implementation, can be overridden
        if 'USDT' in symbol:
            base = symbol.replace('USDT', '')
            return f"{base}/USDT"
        elif 'USD' in symbol:
            base = symbol.replace('USD', '')
            return f"{base}/USD"
        return symbol
    
    async def get_ticker(self, symbol: str) -> Dict[str, float]:
        """Get ticker data (can be implemented using orderbook)"""
        orderbook = await self.get_orderbook(symbol, depth=1)
        return {
            'bid': orderbook.bids[0][0] if orderbook.bids else 0,
            'ask': orderbook.asks[0][0] if orderbook.asks else 0,
            'mid': orderbook.mid_price,
            'spread': orderbook.spread,
            'timestamp': orderbook.timestamp
        }
    
    def register_orderbook_callback(self, symbol: str, callback: Callable[[OrderBook], None]) -> None:
        """Register callback for orderbook updates"""
        if symbol not in self._orderbook_callbacks:
            self._orderbook_callbacks[symbol] = []
        self._orderbook_callbacks[symbol].append(callback)
    
    def register_trade_callback(self, symbol: str, callback: Callable[[Trade], None]) -> None:
        """Register callback for trade updates"""
        if symbol not in self._trade_callbacks:
            self._trade_callbacks[symbol] = []
        self._trade_callbacks[symbol].append(callback)
    
    async def _emit_orderbook_update(self, symbol: str, orderbook: OrderBook) -> None:
        """Emit orderbook update to all registered callbacks"""
        if symbol in self._orderbook_callbacks:
            for callback in self._orderbook_callbacks[symbol]:
                try:
                    await asyncio.create_task(asyncio.coroutine(callback)(orderbook))
                except Exception as e:
                    logger.error(f"Error in orderbook callback: {e}")
    
    async def _emit_trade_update(self, symbol: str, trade: Trade) -> None:
        """Emit trade update to all registered callbacks"""
        if symbol in self._trade_callbacks:
            for callback in self._trade_callbacks[symbol]:
                try:
                    await asyncio.create_task(asyncio.coroutine(callback)(trade))
                except Exception as e:
                    logger.error(f"Error in trade callback: {e}")
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}', connected={self._ws_connected})"