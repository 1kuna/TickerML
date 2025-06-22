#!/usr/bin/env python3
"""
Coinbase Exchange Implementation
Supports Coinbase Advanced Trade API (formerly Coinbase Pro)
"""

import asyncio
import aiohttp
import websockets
import json
import time
import hmac
import hashlib
import base64
from typing import Dict, List, Optional, Callable, Any
import logging
from datetime import datetime

from .base import (
    ExchangeInterface, ExchangeConfig, OrderBook, Trade, Order, Balance,
    OrderType, OrderSide, OrderStatus
)

logger = logging.getLogger(__name__)

class CoinbaseExchange(ExchangeInterface):
    """Coinbase exchange implementation using Advanced Trade API"""
    
    def __init__(self, config: ExchangeConfig):
        super().__init__(config)
        
        # Set URLs
        if config.testnet:
            self.rest_url = "https://api-public.sandbox.pro.coinbase.com"
            self.ws_url = "wss://ws-feed-public.sandbox.pro.coinbase.com"
        else:
            self.rest_url = config.rest_url or "https://api.coinbase.com"
            self.ws_url = config.ws_url or "wss://ws-feed.exchange.coinbase.com"
        
        self._orderbook_cache: Dict[str, Dict] = {}  # Full orderbook state
        self._ws_connection = None
        self._ws_subscriptions = set()
        
    async def connect(self) -> None:
        """Initialize connection to Coinbase"""
        if not self._rest_session:
            self._rest_session = aiohttp.ClientSession()
        
        # Test connectivity
        try:
            await self._make_request("GET", "/api/v3/brokerage/time")
            logger.info(f"Connected to Coinbase ({self.rest_url})")
        except Exception as e:
            logger.error(f"Failed to connect to Coinbase: {e}")
            raise
    
    async def disconnect(self) -> None:
        """Close all connections"""
        # Close WebSocket
        if self._ws_connection:
            await self._ws_connection.close()
            self._ws_connection = None
        
        # Close REST session
        if self._rest_session:
            await self._rest_session.close()
            self._rest_session = None
        
        logger.info("Disconnected from Coinbase")
    
    async def get_symbols(self) -> List[str]:
        """Get list of available trading symbols"""
        data = await self._make_request("GET", "/api/v3/brokerage/products")
        symbols = []
        
        for product in data['products']:
            if product['trading_disabled'] is False:
                # Coinbase uses format like "BTC-USD"
                symbol = product['product_id'].replace('-', '/')
                symbols.append(symbol)
        
        return symbols
    
    async def get_orderbook(self, symbol: str, depth: int = 20) -> OrderBook:
        """Get current order book snapshot"""
        coinbase_symbol = self._to_coinbase_symbol(symbol)
        
        params = {
            'product_id': coinbase_symbol,
            'limit': min(depth, 999)  # Coinbase max
        }
        
        data = await self._make_request("GET", "/api/v3/brokerage/product_book", params=params)
        
        # Convert to OrderBook format
        bids = [(float(bid['price']), float(bid['size'])) for bid in data['pricebook']['bids']]
        asks = [(float(ask['price']), float(ask['size'])) for ask in data['pricebook']['asks']]
        
        return OrderBook(
            exchange=self.name,
            symbol=symbol,
            timestamp=time.time(),
            bids=bids[:depth],
            asks=asks[:depth],
            sequence=int(data['pricebook']['time'])
        )
    
    async def subscribe_orderbook(self, symbol: str, callback: Callable[[OrderBook], None]) -> None:
        """Subscribe to real-time order book updates"""
        self.register_orderbook_callback(symbol, callback)
        
        coinbase_symbol = self._to_coinbase_symbol(symbol)
        
        # Start WebSocket if not already running
        if not self._ws_connection:
            await self._start_websocket()
        
        # Subscribe to level2 channel
        subscribe_msg = {
            "type": "subscribe",
            "product_ids": [coinbase_symbol],
            "channels": ["level2_batch"]
        }
        
        await self._ws_connection.send(json.dumps(subscribe_msg))
        self._ws_subscriptions.add((symbol, 'orderbook'))
        
        logger.info(f"Subscribed to {symbol} orderbook on Coinbase")
    
    async def unsubscribe_orderbook(self, symbol: str) -> None:
        """Unsubscribe from order book updates"""
        coinbase_symbol = self._to_coinbase_symbol(symbol)
        
        if self._ws_connection:
            unsubscribe_msg = {
                "type": "unsubscribe",
                "product_ids": [coinbase_symbol],
                "channels": ["level2_batch"]
            }
            await self._ws_connection.send(json.dumps(unsubscribe_msg))
        
        self._ws_subscriptions.discard((symbol, 'orderbook'))
        
        if symbol in self._orderbook_callbacks:
            del self._orderbook_callbacks[symbol]
        
        logger.info(f"Unsubscribed from {symbol} orderbook on Coinbase")
    
    async def subscribe_trades(self, symbol: str, callback: Callable[[Trade], None]) -> None:
        """Subscribe to real-time trades"""
        self.register_trade_callback(symbol, callback)
        
        coinbase_symbol = self._to_coinbase_symbol(symbol)
        
        # Start WebSocket if not already running
        if not self._ws_connection:
            await self._start_websocket()
        
        # Subscribe to matches channel
        subscribe_msg = {
            "type": "subscribe",
            "product_ids": [coinbase_symbol],
            "channels": ["matches"]
        }
        
        await self._ws_connection.send(json.dumps(subscribe_msg))
        self._ws_subscriptions.add((symbol, 'trades'))
        
        logger.info(f"Subscribed to {symbol} trades on Coinbase")
    
    async def unsubscribe_trades(self, symbol: str) -> None:
        """Unsubscribe from trades"""
        coinbase_symbol = self._to_coinbase_symbol(symbol)
        
        if self._ws_connection:
            unsubscribe_msg = {
                "type": "unsubscribe",
                "product_ids": [coinbase_symbol],
                "channels": ["matches"]
            }
            await self._ws_connection.send(json.dumps(unsubscribe_msg))
        
        self._ws_subscriptions.discard((symbol, 'trades'))
        
        if symbol in self._trade_callbacks:
            del self._trade_callbacks[symbol]
        
        logger.info(f"Unsubscribed from {symbol} trades on Coinbase")
    
    async def get_balance(self) -> Dict[str, Balance]:
        """Get account balances"""
        data = await self._make_request("GET", "/api/v3/brokerage/accounts", signed=True)
        
        balances = {}
        for account in data['accounts']:
            currency = account['currency']
            available = float(account['available_balance']['value'])
            hold = float(account['hold']['value']) if 'hold' in account else 0
            
            if available > 0 or hold > 0:
                balances[currency] = Balance(
                    currency=currency,
                    free=available,
                    locked=hold,
                    total=available + hold
                )
        
        return balances
    
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
        coinbase_symbol = self._to_coinbase_symbol(symbol)
        
        order_config = {
            "product_id": coinbase_symbol,
            "side": side.value.upper(),
            "order_configuration": {}
        }
        
        if client_order_id:
            order_config["client_order_id"] = client_order_id
        
        # Configure order based on type
        if order_type == OrderType.MARKET:
            if side == OrderSide.BUY:
                # For market buys, Coinbase uses quote_size (USD amount)
                # We need to estimate this from current price
                ticker = await self.get_ticker(symbol)
                quote_size = quantity * ticker['ask']
                order_config["order_configuration"]["market_market_ioc"] = {
                    "quote_size": str(quote_size)
                }
            else:
                # For market sells, use base_size
                order_config["order_configuration"]["market_market_ioc"] = {
                    "base_size": str(quantity)
                }
        
        elif order_type == OrderType.LIMIT:
            if price is None:
                raise ValueError("Price required for limit orders")
            order_config["order_configuration"]["limit_limit_gtc"] = {
                "base_size": str(quantity),
                "limit_price": str(price)
            }
        
        elif order_type == OrderType.STOP:
            if price is None:
                raise ValueError("Stop price required for stop orders")
            order_config["order_configuration"]["stop_limit_stop_limit_gtc"] = {
                "base_size": str(quantity),
                "stop_price": str(price),
                "limit_price": str(price * 0.95 if side == OrderSide.SELL else price * 1.05)
            }
        
        data = await self._make_request("POST", "/api/v3/brokerage/orders", json=order_config, signed=True)
        
        return self._parse_order(data['order'], symbol)
    
    async def cancel_order(self, symbol: str, order_id: str) -> bool:
        """Cancel an order"""
        try:
            order_ids = {"order_ids": [order_id]}
            await self._make_request("POST", "/api/v3/brokerage/orders/batch_cancel", json=order_ids, signed=True)
            return True
        except Exception as e:
            logger.error(f"Failed to cancel order {order_id}: {e}")
            return False
    
    async def get_order(self, symbol: str, order_id: str) -> Order:
        """Get order status"""
        data = await self._make_request("GET", f"/api/v3/brokerage/orders/historical/{order_id}", signed=True)
        return self._parse_order(data['order'], symbol)
    
    async def get_open_orders(self, symbol: Optional[str] = None) -> List[Order]:
        """Get all open orders"""
        params = {"order_status": "OPEN"}
        
        if symbol:
            params["product_id"] = self._to_coinbase_symbol(symbol)
        
        data = await self._make_request("GET", "/api/v3/brokerage/orders/historical/batch", params=params, signed=True)
        
        orders = []
        for order_data in data.get('orders', []):
            # Convert Coinbase symbol to normalized
            coinbase_symbol = order_data['product_id']
            normalized_symbol = coinbase_symbol.replace('-', '/')
            orders.append(self._parse_order(order_data, normalized_symbol))
        
        return orders
    
    async def get_fees(self, symbol: str) -> Dict[str, float]:
        """Get trading fees for symbol"""
        # Coinbase Advanced Trade has tiered fees
        # These are approximate values; actual fees depend on volume
        return {
            'maker': 0.004,  # 0.4%
            'taker': 0.006   # 0.6%
        }
    
    async def get_server_time(self) -> float:
        """Get exchange server time"""
        data = await self._make_request("GET", "/api/v3/brokerage/time")
        return float(data['epochSeconds'])
    
    # Helper methods
    
    def _to_coinbase_symbol(self, symbol: str) -> str:
        """Convert normalized symbol to Coinbase format"""
        return symbol.replace('/', '-')
    
    def _parse_order_status(self, status: str) -> OrderStatus:
        """Parse Coinbase order status"""
        mapping = {
            'OPEN': OrderStatus.OPEN,
            'FILLED': OrderStatus.FILLED,
            'CANCELLED': OrderStatus.CANCELLED,
            'EXPIRED': OrderStatus.EXPIRED,
            'FAILED': OrderStatus.REJECTED
        }
        return mapping.get(status, OrderStatus.PENDING)
    
    def _parse_order(self, data: Dict, symbol: str) -> Order:
        """Parse order data from Coinbase"""
        # Extract quantity and filled quantity based on order configuration
        config = data.get('order_configuration', {})
        quantity = 0
        price = None
        
        # Determine order type and extract relevant data
        if 'market_market_ioc' in config:
            order_type = OrderType.MARKET
            if 'base_size' in config['market_market_ioc']:
                quantity = float(config['market_market_ioc']['base_size'])
            elif 'quote_size' in config['market_market_ioc']:
                # For market buy orders, we have quote size
                quantity = float(config['market_market_ioc']['quote_size'])
        
        elif 'limit_limit_gtc' in config or 'limit_limit_fok' in config:
            order_type = OrderType.LIMIT
            limit_config = config.get('limit_limit_gtc', config.get('limit_limit_fok', {}))
            quantity = float(limit_config.get('base_size', 0))
            price = float(limit_config.get('limit_price', 0))
        
        elif 'stop_limit_stop_limit_gtc' in config:
            order_type = OrderType.STOP_LIMIT
            stop_config = config['stop_limit_stop_limit_gtc']
            quantity = float(stop_config.get('base_size', 0))
            price = float(stop_config.get('stop_price', 0))
        else:
            order_type = OrderType.LIMIT
        
        return Order(
            exchange=self.name,
            symbol=symbol,
            order_id=data['order_id'],
            client_order_id=data.get('client_order_id'),
            timestamp=datetime.fromisoformat(data['created_time'].replace('Z', '+00:00')).timestamp(),
            type=order_type,
            side=OrderSide.BUY if data['side'] == 'BUY' else OrderSide.SELL,
            price=price,
            quantity=quantity,
            filled_quantity=float(data.get('filled_size', 0)),
            status=self._parse_order_status(data['status'])
        )
    
    async def _start_websocket(self) -> None:
        """Start WebSocket connection and handler"""
        async def ws_handler():
            try:
                async with websockets.connect(self.ws_url) as websocket:
                    self._ws_connection = websocket
                    logger.info("Connected to Coinbase WebSocket")
                    
                    while self._ws_connection:
                        try:
                            message = await asyncio.wait_for(websocket.recv(), timeout=30)
                            data = json.loads(message)
                            
                            await self._handle_ws_message(data)
                            
                        except asyncio.TimeoutError:
                            # Send ping
                            await websocket.ping()
                        except Exception as e:
                            logger.error(f"WebSocket error: {e}")
                            break
                            
            except Exception as e:
                logger.error(f"WebSocket connection error: {e}")
            finally:
                self._ws_connection = None
        
        asyncio.create_task(ws_handler())
    
    async def _handle_ws_message(self, data: Dict) -> None:
        """Handle WebSocket message"""
        msg_type = data.get('type')
        
        if msg_type == 'l2update':
            # Level 2 orderbook update
            symbol = data['product_id'].replace('-', '/')
            
            # Initialize cache if needed
            if symbol not in self._orderbook_cache:
                # Need to get snapshot first
                orderbook = await self.get_orderbook(symbol)
                self._orderbook_cache[symbol] = {
                    'bids': {str(bid[0]): bid[1] for bid in orderbook.bids},
                    'asks': {str(ask[0]): ask[1] for ask in orderbook.asks}
                }
            
            # Apply updates
            cache = self._orderbook_cache[symbol]
            
            for change in data['changes']:
                side, price, size = change
                price = str(float(price))
                size = float(size)
                
                if side == 'buy':
                    if size == 0:
                        cache['bids'].pop(price, None)
                    else:
                        cache['bids'][price] = size
                else:
                    if size == 0:
                        cache['asks'].pop(price, None)
                    else:
                        cache['asks'][price] = size
            
            # Convert to OrderBook
            bids = sorted([(float(p), v) for p, v in cache['bids'].items()], reverse=True)[:20]
            asks = sorted([(float(p), v) for p, v in cache['asks'].items()])[:20]
            
            orderbook = OrderBook(
                exchange=self.name,
                symbol=symbol,
                timestamp=datetime.fromisoformat(data['time'].replace('Z', '+00:00')).timestamp(),
                bids=bids,
                asks=asks
            )
            
            await self._emit_orderbook_update(symbol, orderbook)
        
        elif msg_type == 'match':
            # Trade update
            symbol = data['product_id'].replace('-', '/')
            
            trade = Trade(
                exchange=self.name,
                symbol=symbol,
                timestamp=datetime.fromisoformat(data['time'].replace('Z', '+00:00')).timestamp(),
                price=float(data['price']),
                quantity=float(data['size']),
                side=OrderSide.BUY if data['side'] == 'buy' else OrderSide.SELL,
                trade_id=str(data['trade_id'])
            )
            
            await self._emit_trade_update(symbol, trade)
    
    async def _make_request(
        self,
        method: str,
        endpoint: str,
        params: Optional[Dict] = None,
        json: Optional[Dict] = None,
        signed: bool = False
    ) -> Any:
        """Make HTTP request to Coinbase API"""
        if not self._rest_session:
            self._rest_session = aiohttp.ClientSession()
        
        url = f"{self.rest_url}{endpoint}"
        
        headers = {
            'Content-Type': 'application/json'
        }
        
        if signed:
            if not self.config.api_key or not self.config.api_secret:
                raise ValueError("API key and secret required for signed requests")
            
            # Coinbase uses JWT for authentication in Advanced Trade API
            # For simplicity, using the legacy API key method here
            # In production, implement proper JWT signing
            
            timestamp = str(int(time.time()))
            
            # Create signature
            if json:
                body = json.dumps(json, separators=(',', ':'))
            else:
                body = ''
            
            message = timestamp + method + endpoint + body
            
            signature = hmac.new(
                base64.b64decode(self.config.api_secret),
                message.encode('utf-8'),
                hashlib.sha256
            ).digest()
            
            headers.update({
                'CB-ACCESS-KEY': self.config.api_key,
                'CB-ACCESS-SIGN': base64.b64encode(signature).decode(),
                'CB-ACCESS-TIMESTAMP': timestamp,
            })
            
            if self.config.passphrase:
                headers['CB-ACCESS-PASSPHRASE'] = self.config.passphrase
        
        async with self._rest_session.request(
            method,
            url,
            params=params,
            json=json,
            headers=headers
        ) as response:
            data = await response.json()
            
            if response.status != 200:
                error_msg = data.get('message', 'Unknown error')
                raise Exception(f"Coinbase API error: {error_msg}")
            
            return data