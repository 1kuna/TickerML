#!/usr/bin/env python3
"""
KuCoin Exchange Implementation
Supports KuCoin REST and WebSocket APIs
"""

import asyncio
import aiohttp
import websockets
import json
import time
import hmac
import hashlib
import base64
import uuid
from typing import Dict, List, Optional, Callable, Any, Tuple
import logging

from .base import (
    ExchangeInterface, ExchangeConfig, OrderBook, Trade, Order, Balance,
    OrderType, OrderSide, OrderStatus
)

logger = logging.getLogger(__name__)

class KuCoinExchange(ExchangeInterface):
    """KuCoin exchange implementation"""
    
    def __init__(self, config: ExchangeConfig):
        super().__init__(config)
        
        # Set URLs
        if config.testnet:
            self.rest_url = "https://openapi-sandbox.kucoin.com"
        else:
            self.rest_url = config.rest_url or "https://api.kucoin.com"
        
        self._ws_token = None
        self._ws_endpoint = None
        self._ws_connection = None
        self._orderbook_cache: Dict[str, Dict] = {}
        self._ping_interval = 30  # KuCoin requires ping every 30 seconds
        
    async def connect(self) -> None:
        """Initialize connection to KuCoin"""
        if not self._rest_session:
            self._rest_session = aiohttp.ClientSession()
        
        # Test connectivity
        try:
            await self._make_request("GET", "/api/v1/timestamp")
            logger.info(f"Connected to KuCoin ({self.rest_url})")
        except Exception as e:
            logger.error(f"Failed to connect to KuCoin: {e}")
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
        
        logger.info("Disconnected from KuCoin")
    
    async def get_symbols(self) -> List[str]:
        """Get list of available trading symbols"""
        data = await self._make_request("GET", "/api/v1/symbols")
        
        symbols = []
        for symbol_info in data['data']:
            if symbol_info['enableTrading']:
                # Convert to normalized format
                base = symbol_info['baseCurrency']
                quote = symbol_info['quoteCurrency']
                symbols.append(f"{base}/{quote}")
        
        return symbols
    
    async def get_orderbook(self, symbol: str, depth: int = 20) -> OrderBook:
        """Get current order book snapshot"""
        kucoin_symbol = self._to_kucoin_symbol(symbol)
        
        # KuCoin has different depth levels: 20, 100
        kucoin_depth = 100 if depth > 20 else 20
        
        data = await self._make_request("GET", f"/api/v1/market/orderbook/level2_{kucoin_depth}", params={
            'symbol': kucoin_symbol
        })
        
        orderbook_data = data['data']
        
        # Convert to OrderBook format
        bids = [(float(bid[0]), float(bid[1])) for bid in orderbook_data['bids']]
        asks = [(float(ask[0]), float(ask[1])) for ask in orderbook_data['asks']]
        
        return OrderBook(
            exchange=self.name,
            symbol=symbol,
            timestamp=int(orderbook_data['time']) / 1000,
            bids=bids[:depth],
            asks=asks[:depth],
            sequence=int(orderbook_data['sequence'])
        )
    
    async def subscribe_orderbook(self, symbol: str, callback: Callable[[OrderBook], None]) -> None:
        """Subscribe to real-time order book updates"""
        self.register_orderbook_callback(symbol, callback)
        
        kucoin_symbol = self._to_kucoin_symbol(symbol)
        
        # Start WebSocket if not already running
        if not self._ws_connection:
            await self._start_websocket()
        
        # Subscribe to level2 channel
        subscribe_msg = {
            "id": str(int(time.time() * 1000)),
            "type": "subscribe",
            "topic": f"/market/level2:{kucoin_symbol}",
            "response": True
        }
        
        await self._ws_connection.send(json.dumps(subscribe_msg))
        logger.info(f"Subscribed to {symbol} orderbook on KuCoin")
    
    async def unsubscribe_orderbook(self, symbol: str) -> None:
        """Unsubscribe from order book updates"""
        kucoin_symbol = self._to_kucoin_symbol(symbol)
        
        if self._ws_connection:
            unsubscribe_msg = {
                "id": str(int(time.time() * 1000)),
                "type": "unsubscribe",
                "topic": f"/market/level2:{kucoin_symbol}",
                "response": True
            }
            await self._ws_connection.send(json.dumps(unsubscribe_msg))
        
        if symbol in self._orderbook_callbacks:
            del self._orderbook_callbacks[symbol]
        
        logger.info(f"Unsubscribed from {symbol} orderbook on KuCoin")
    
    async def subscribe_trades(self, symbol: str, callback: Callable[[Trade], None]) -> None:
        """Subscribe to real-time trades"""
        self.register_trade_callback(symbol, callback)
        
        kucoin_symbol = self._to_kucoin_symbol(symbol)
        
        # Start WebSocket if not already running
        if not self._ws_connection:
            await self._start_websocket()
        
        # Subscribe to match channel
        subscribe_msg = {
            "id": str(int(time.time() * 1000)),
            "type": "subscribe",
            "topic": f"/market/match:{kucoin_symbol}",
            "response": True
        }
        
        await self._ws_connection.send(json.dumps(subscribe_msg))
        logger.info(f"Subscribed to {symbol} trades on KuCoin")
    
    async def unsubscribe_trades(self, symbol: str) -> None:
        """Unsubscribe from trades"""
        kucoin_symbol = self._to_kucoin_symbol(symbol)
        
        if self._ws_connection:
            unsubscribe_msg = {
                "id": str(int(time.time() * 1000)),
                "type": "unsubscribe",
                "topic": f"/market/match:{kucoin_symbol}",
                "response": True
            }
            await self._ws_connection.send(json.dumps(unsubscribe_msg))
        
        if symbol in self._trade_callbacks:
            del self._trade_callbacks[symbol]
        
        logger.info(f"Unsubscribed from {symbol} trades on KuCoin")
    
    async def get_balance(self) -> Dict[str, Balance]:
        """Get account balances"""
        data = await self._make_request("GET", "/api/v1/accounts", signed=True)
        
        balances = {}
        for account in data['data']:
            currency = account['currency']
            balance = float(account['balance'])
            available = float(account['available'])
            holds = float(account['holds'])
            
            if balance > 0:
                balances[currency] = Balance(
                    currency=currency,
                    free=available,
                    locked=holds,
                    total=balance
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
        kucoin_symbol = self._to_kucoin_symbol(symbol)
        
        order_data = {
            'symbol': kucoin_symbol,
            'side': side.value,
            'type': self._to_kucoin_order_type(order_type),
            'size': str(quantity)
        }
        
        if client_order_id:
            order_data['clientOid'] = client_order_id
        else:
            order_data['clientOid'] = str(uuid.uuid4())
        
        if order_type in [OrderType.LIMIT, OrderType.STOP_LIMIT]:
            if price is None:
                raise ValueError("Price required for limit orders")
            order_data['price'] = str(price)
        
        if order_type == OrderType.STOP:
            if price is None:
                raise ValueError("Stop price required for stop orders")
            order_data['stopPrice'] = str(price)
        
        data = await self._make_request("POST", "/api/v1/orders", json=order_data, signed=True)
        
        return Order(
            exchange=self.name,
            symbol=symbol,
            order_id=data['data']['orderId'],
            client_order_id=order_data['clientOid'],
            timestamp=time.time(),
            type=order_type,
            side=side,
            price=price,
            quantity=quantity,
            filled_quantity=0,
            status=OrderStatus.PENDING
        )
    
    async def cancel_order(self, symbol: str, order_id: str) -> bool:
        """Cancel an order"""
        try:
            await self._make_request("DELETE", f"/api/v1/orders/{order_id}", signed=True)
            return True
        except Exception as e:
            logger.error(f"Failed to cancel order {order_id}: {e}")
            return False
    
    async def get_order(self, symbol: str, order_id: str) -> Order:
        """Get order status"""
        data = await self._make_request("GET", f"/api/v1/orders/{order_id}", signed=True)
        return self._parse_order(data['data'], symbol)
    
    async def get_open_orders(self, symbol: Optional[str] = None) -> List[Order]:
        """Get all open orders"""
        params = {'status': 'active'}
        
        if symbol:
            params['symbol'] = self._to_kucoin_symbol(symbol)
        
        data = await self._make_request("GET", "/api/v1/orders", params=params, signed=True)
        
        orders = []
        for order_data in data['data']['items']:
            # Convert KuCoin symbol to normalized
            kucoin_symbol = order_data['symbol']
            normalized_symbol = self._from_kucoin_symbol(kucoin_symbol)
            orders.append(self._parse_order(order_data, normalized_symbol))
        
        return orders
    
    async def get_fees(self, symbol: str) -> Dict[str, float]:
        """Get trading fees for symbol"""
        # Get base fee rate (this may require additional API calls for precise fees)
        data = await self._make_request("GET", "/api/v1/base-fee", signed=True)
        
        return {
            'maker': float(data['data']['makerFeeRate']),
            'taker': float(data['data']['takerFeeRate'])
        }
    
    async def get_server_time(self) -> float:
        """Get exchange server time"""
        data = await self._make_request("GET", "/api/v1/timestamp")
        return int(data['data']) / 1000
    
    # Helper methods
    
    def _to_kucoin_symbol(self, symbol: str) -> str:
        """Convert normalized symbol to KuCoin format"""
        return symbol.replace('/', '-')
    
    def _from_kucoin_symbol(self, kucoin_symbol: str) -> str:
        """Convert KuCoin symbol to normalized format"""
        return kucoin_symbol.replace('-', '/')
    
    def _to_kucoin_order_type(self, order_type: OrderType) -> str:
        """Convert order type to KuCoin format"""
        mapping = {
            OrderType.MARKET: 'market',
            OrderType.LIMIT: 'limit',
            OrderType.STOP: 'market',  # KuCoin uses market for stop orders
            OrderType.STOP_LIMIT: 'limit'
        }
        return mapping.get(order_type, 'limit')
    
    def _parse_order_status(self, status: str) -> OrderStatus:
        """Parse KuCoin order status"""
        mapping = {
            'active': OrderStatus.OPEN,
            'done': OrderStatus.FILLED,
            'cancelled': OrderStatus.CANCELLED
        }
        return mapping.get(status, OrderStatus.PENDING)
    
    def _parse_order(self, data: Dict, symbol: str) -> Order:
        """Parse order data from KuCoin"""
        return Order(
            exchange=self.name,
            symbol=symbol,
            order_id=data['id'],
            client_order_id=data.get('clientOid'),
            timestamp=int(data['createdAt']) / 1000,
            type=self._parse_order_type(data['type']),
            side=OrderSide.BUY if data['side'] == 'buy' else OrderSide.SELL,
            price=float(data['price']) if data['price'] else None,
            quantity=float(data['size']),
            filled_quantity=float(data['dealSize']),
            status=self._parse_order_status(data['isActive'])
        )
    
    def _parse_order_type(self, kucoin_type: str) -> OrderType:
        """Parse KuCoin order type"""
        mapping = {
            'market': OrderType.MARKET,
            'limit': OrderType.LIMIT
        }
        return mapping.get(kucoin_type, OrderType.LIMIT)
    
    async def _get_ws_token(self) -> Tuple[str, str]:
        """Get WebSocket token and endpoint"""
        data = await self._make_request("POST", "/api/v1/bullet-public")
        
        token = data['data']['token']
        endpoint = data['data']['instanceServers'][0]['endpoint']
        
        return token, endpoint
    
    async def _start_websocket(self) -> None:
        """Start WebSocket connection and handler"""
        # Get token and endpoint
        self._ws_token, self._ws_endpoint = await self._get_ws_token()
        
        # Connect to WebSocket
        ws_url = f"{self._ws_endpoint}?token={self._ws_token}&[connectId={uuid.uuid4()}]"
        
        async def ws_handler():
            try:
                async with websockets.connect(ws_url) as websocket:
                    self._ws_connection = websocket
                    logger.info("Connected to KuCoin WebSocket")
                    
                    # Start ping task
                    ping_task = asyncio.create_task(self._ping_loop())
                    
                    while self._ws_connection:
                        try:
                            message = await asyncio.wait_for(websocket.recv(), timeout=60)
                            data = json.loads(message)
                            
                            await self._handle_ws_message(data)
                            
                        except asyncio.TimeoutError:
                            logger.warning("WebSocket timeout")
                            break
                        except Exception as e:
                            logger.error(f"WebSocket error: {e}")
                            break
                    
                    ping_task.cancel()
                    
            except Exception as e:
                logger.error(f"WebSocket connection error: {e}")
            finally:
                self._ws_connection = None
        
        asyncio.create_task(ws_handler())
    
    async def _ping_loop(self) -> None:
        """Send ping messages to keep connection alive"""
        while self._ws_connection:
            try:
                ping_msg = {
                    "id": str(int(time.time() * 1000)),
                    "type": "ping"
                }
                await self._ws_connection.send(json.dumps(ping_msg))
                await asyncio.sleep(self._ping_interval)
            except Exception as e:
                logger.error(f"Ping error: {e}")
                break
    
    async def _handle_ws_message(self, data: Dict) -> None:
        """Handle WebSocket message"""
        msg_type = data.get('type')
        
        if msg_type == 'message':
            topic = data.get('topic', '')
            subject = data.get('subject', '')
            
            if topic.startswith('/market/level2:'):
                # Orderbook update
                symbol_part = topic.split(':')[1]
                symbol = self._from_kucoin_symbol(symbol_part)
                await self._handle_orderbook_update(symbol, data['data'])
            
            elif topic.startswith('/market/match:'):
                # Trade update
                symbol_part = topic.split(':')[1]
                symbol = self._from_kucoin_symbol(symbol_part)
                await self._handle_trade_update(symbol, data['data'])
    
    async def _handle_orderbook_update(self, symbol: str, data: Dict) -> None:
        """Handle orderbook update"""
        # KuCoin sends incremental updates
        # For full implementation, need to maintain orderbook state
        
        changes = data.get('changes', {})
        timestamp = int(data['time']) / 1000
        
        bids = []
        asks = []
        
        # Parse changes
        if 'bids' in changes:
            bids = [(float(bid[0]), float(bid[1])) for bid in changes['bids'] if float(bid[1]) > 0]
        
        if 'asks' in changes:
            asks = [(float(ask[0]), float(ask[1])) for ask in changes['asks'] if float(ask[1]) > 0]
        
        # For simplicity, getting fresh snapshot
        # In production, should maintain incremental state
        if bids or asks:
            try:
                orderbook = await self.get_orderbook(symbol, 20)
                await self._emit_orderbook_update(symbol, orderbook)
            except Exception as e:
                logger.error(f"Error getting orderbook snapshot: {e}")
    
    async def _handle_trade_update(self, symbol: str, data: Dict) -> None:
        """Handle trade update"""
        trade = Trade(
            exchange=self.name,
            symbol=symbol,
            timestamp=int(data['time']) / 1000,
            price=float(data['price']),
            quantity=float(data['size']),
            side=OrderSide.BUY if data['side'] == 'buy' else OrderSide.SELL,
            trade_id=data.get('tradeId')
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
        """Make HTTP request to KuCoin API"""
        if not self._rest_session:
            self._rest_session = aiohttp.ClientSession()
        
        url = f"{self.rest_url}{endpoint}"
        
        headers = {
            'Content-Type': 'application/json'
        }
        
        if signed:
            if not all([self.config.api_key, self.config.api_secret, self.config.passphrase]):
                raise ValueError("API key, secret, and passphrase required for signed requests")
            
            # KuCoin signature requirements
            timestamp = str(int(time.time() * 1000))
            
            # Create signature
            if json:
                body = json.dumps(json, separators=(',', ':'))
            else:
                body = ''
            
            # Query string for GET requests
            query_string = ''
            if params:
                from urllib.parse import urlencode
                query_string = '?' + urlencode(params)
            
            str_to_sign = timestamp + method + endpoint + query_string + body
            
            signature = base64.b64encode(
                hmac.new(
                    self.config.api_secret.encode('utf-8'),
                    str_to_sign.encode('utf-8'),
                    hashlib.sha256
                ).digest()
            ).decode()
            
            # Passphrase signature
            passphrase = base64.b64encode(
                hmac.new(
                    self.config.api_secret.encode('utf-8'),
                    self.config.passphrase.encode('utf-8'),
                    hashlib.sha256
                ).digest()
            ).decode()
            
            headers.update({
                'KC-API-KEY': self.config.api_key,
                'KC-API-SIGN': signature,
                'KC-API-TIMESTAMP': timestamp,
                'KC-API-PASSPHRASE': passphrase,
                'KC-API-KEY-VERSION': '2'
            })
        
        async with self._rest_session.request(
            method,
            url,
            params=params,
            json=json,
            headers=headers
        ) as response:
            data = await response.json()
            
            if response.status != 200 or data.get('code') != '200000':
                error_msg = data.get('msg', 'Unknown error')
                raise Exception(f"KuCoin API error: {error_msg}")
            
            return data