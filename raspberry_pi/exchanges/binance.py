#!/usr/bin/env python3
"""
Binance Exchange Implementation
Supports Binance.US and Binance International
"""

import asyncio
import aiohttp
import websockets
import json
import time
import hmac
import hashlib
from typing import Dict, List, Optional, Callable, Any
from urllib.parse import urlencode
import logging

from .base import (
    ExchangeInterface, ExchangeConfig, OrderBook, Trade, Order, Balance,
    OrderType, OrderSide, OrderStatus
)

logger = logging.getLogger(__name__)

class BinanceExchange(ExchangeInterface):
    """Binance exchange implementation"""
    
    def __init__(self, config: ExchangeConfig):
        super().__init__(config)
        
        # Set URLs based on testnet or region
        if config.testnet:
            self.rest_url = "https://testnet.binance.vision/api"
            self.ws_url = "wss://testnet.binance.vision/ws"
        else:
            # Use Binance.US for US users
            if config.rest_url and "binance.us" in config.rest_url:
                self.rest_url = "https://api.binance.us/api"
                self.ws_url = "wss://stream.binance.us:9443/ws"
            else:
                self.rest_url = config.rest_url or "https://api.binance.com/api"
                self.ws_url = config.ws_url or "wss://stream.binance.com:9443/ws"
        
        self._orderbook_cache: Dict[str, OrderBook] = {}
        self._orderbook_ws_connections: Dict[str, Any] = {}
        self._trade_ws_connections: Dict[str, Any] = {}
        
    async def connect(self) -> None:
        """Initialize connection to Binance"""
        if not self._rest_session:
            self._rest_session = aiohttp.ClientSession()
        
        # Test connectivity
        try:
            await self._make_request("GET", "/v3/ping")
            logger.info(f"Connected to Binance ({self.rest_url})")
        except Exception as e:
            logger.error(f"Failed to connect to Binance: {e}")
            raise
    
    async def disconnect(self) -> None:
        """Close all connections"""
        # Close WebSocket connections
        for ws in list(self._orderbook_ws_connections.values()):
            await ws.close()
        for ws in list(self._trade_ws_connections.values()):
            await ws.close()
        
        self._orderbook_ws_connections.clear()
        self._trade_ws_connections.clear()
        
        # Close REST session
        if self._rest_session:
            await self._rest_session.close()
            self._rest_session = None
        
        logger.info("Disconnected from Binance")
    
    async def get_symbols(self) -> List[str]:
        """Get list of available trading symbols"""
        data = await self._make_request("GET", "/v3/exchangeInfo")
        symbols = []
        
        for symbol_info in data['symbols']:
            if symbol_info['status'] == 'TRADING':
                # Convert to normalized format
                base = symbol_info['baseAsset']
                quote = symbol_info['quoteAsset']
                symbols.append(f"{base}/{quote}")
        
        return symbols
    
    async def get_orderbook(self, symbol: str, depth: int = 20) -> OrderBook:
        """Get current order book snapshot"""
        binance_symbol = self._to_binance_symbol(symbol)
        
        params = {
            'symbol': binance_symbol,
            'limit': min(depth, 5000)  # Binance max is 5000
        }
        
        data = await self._make_request("GET", "/v3/depth", params=params)
        
        # Convert to OrderBook format
        bids = [(float(price), float(qty)) for price, qty in data['bids']]
        asks = [(float(price), float(qty)) for price, qty in data['asks']]
        
        return OrderBook(
            exchange=self.name,
            symbol=symbol,
            timestamp=time.time(),
            bids=bids,
            asks=asks,
            sequence=data.get('lastUpdateId')
        )
    
    async def subscribe_orderbook(self, symbol: str, callback: Callable[[OrderBook], None]) -> None:
        """Subscribe to real-time order book updates"""
        self.register_orderbook_callback(symbol, callback)
        
        binance_symbol = self._to_binance_symbol(symbol).lower()
        stream_name = f"{binance_symbol}@depth20@100ms"
        
        # Create WebSocket connection for this symbol
        ws_url = f"{self.ws_url}/{stream_name}"
        
        async def orderbook_handler():
            try:
                async with websockets.connect(ws_url) as websocket:
                    self._orderbook_ws_connections[symbol] = websocket
                    logger.info(f"Subscribed to {symbol} orderbook on Binance")
                    
                    while symbol in self._orderbook_ws_connections:
                        try:
                            message = await asyncio.wait_for(websocket.recv(), timeout=30)
                            data = json.loads(message)
                            
                            # Convert to OrderBook format
                            bids = [(float(price), float(qty)) for price, qty in data['bids']]
                            asks = [(float(price), float(qty)) for price, qty in data['asks']]
                            
                            orderbook = OrderBook(
                                exchange=self.name,
                                symbol=symbol,
                                timestamp=data['E'] / 1000,  # Event time in seconds
                                bids=bids,
                                asks=asks,
                                sequence=data['u']  # Final update ID
                            )
                            
                            # Update cache and emit
                            self._orderbook_cache[symbol] = orderbook
                            await self._emit_orderbook_update(symbol, orderbook)
                            
                        except asyncio.TimeoutError:
                            # Send ping to keep connection alive
                            await websocket.ping()
                        except Exception as e:
                            logger.error(f"Error in orderbook handler for {symbol}: {e}")
                            break
                            
            except Exception as e:
                logger.error(f"WebSocket connection error for {symbol}: {e}")
            finally:
                if symbol in self._orderbook_ws_connections:
                    del self._orderbook_ws_connections[symbol]
        
        # Start handler task
        asyncio.create_task(orderbook_handler())
    
    async def unsubscribe_orderbook(self, symbol: str) -> None:
        """Unsubscribe from order book updates"""
        if symbol in self._orderbook_ws_connections:
            ws = self._orderbook_ws_connections[symbol]
            await ws.close()
            del self._orderbook_ws_connections[symbol]
        
        if symbol in self._orderbook_callbacks:
            del self._orderbook_callbacks[symbol]
        
        logger.info(f"Unsubscribed from {symbol} orderbook on Binance")
    
    async def subscribe_trades(self, symbol: str, callback: Callable[[Trade], None]) -> None:
        """Subscribe to real-time trades"""
        self.register_trade_callback(symbol, callback)
        
        binance_symbol = self._to_binance_symbol(symbol).lower()
        stream_name = f"{binance_symbol}@trade"
        
        # Create WebSocket connection for trades
        ws_url = f"{self.ws_url}/{stream_name}"
        
        async def trade_handler():
            try:
                async with websockets.connect(ws_url) as websocket:
                    self._trade_ws_connections[symbol] = websocket
                    logger.info(f"Subscribed to {symbol} trades on Binance")
                    
                    while symbol in self._trade_ws_connections:
                        try:
                            message = await asyncio.wait_for(websocket.recv(), timeout=30)
                            data = json.loads(message)
                            
                            # Convert to Trade format
                            trade = Trade(
                                exchange=self.name,
                                symbol=symbol,
                                timestamp=data['T'] / 1000,  # Trade time in seconds
                                price=float(data['p']),
                                quantity=float(data['q']),
                                side=OrderSide.BUY if data['m'] else OrderSide.SELL,
                                trade_id=str(data['t'])
                            )
                            
                            await self._emit_trade_update(symbol, trade)
                            
                        except asyncio.TimeoutError:
                            await websocket.ping()
                        except Exception as e:
                            logger.error(f"Error in trade handler for {symbol}: {e}")
                            break
                            
            except Exception as e:
                logger.error(f"WebSocket connection error for {symbol} trades: {e}")
            finally:
                if symbol in self._trade_ws_connections:
                    del self._trade_ws_connections[symbol]
        
        # Start handler task
        asyncio.create_task(trade_handler())
    
    async def unsubscribe_trades(self, symbol: str) -> None:
        """Unsubscribe from trades"""
        if symbol in self._trade_ws_connections:
            ws = self._trade_ws_connections[symbol]
            await ws.close()
            del self._trade_ws_connections[symbol]
        
        if symbol in self._trade_callbacks:
            del self._trade_callbacks[symbol]
        
        logger.info(f"Unsubscribed from {symbol} trades on Binance")
    
    async def get_balance(self) -> Dict[str, Balance]:
        """Get account balances"""
        data = await self._make_request("GET", "/v3/account", signed=True)
        
        balances = {}
        for balance_data in data['balances']:
            currency = balance_data['asset']
            free = float(balance_data['free'])
            locked = float(balance_data['locked'])
            
            if free > 0 or locked > 0:
                balances[currency] = Balance(
                    currency=currency,
                    free=free,
                    locked=locked,
                    total=free + locked
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
        binance_symbol = self._to_binance_symbol(symbol)
        
        params = {
            'symbol': binance_symbol,
            'side': side.value.upper(),
            'type': self._to_binance_order_type(order_type),
            'quantity': quantity,
            'timestamp': int(time.time() * 1000)
        }
        
        if client_order_id:
            params['newClientOrderId'] = client_order_id
        
        if order_type in [OrderType.LIMIT, OrderType.STOP_LIMIT]:
            if price is None:
                raise ValueError("Price required for limit orders")
            params['price'] = price
            params['timeInForce'] = 'GTC'  # Good Till Cancelled
        
        data = await self._make_request("POST", "/v3/order", params=params, signed=True)
        
        return self._parse_order(data, symbol)
    
    async def cancel_order(self, symbol: str, order_id: str) -> bool:
        """Cancel an order"""
        binance_symbol = self._to_binance_symbol(symbol)
        
        params = {
            'symbol': binance_symbol,
            'orderId': order_id,
            'timestamp': int(time.time() * 1000)
        }
        
        try:
            await self._make_request("DELETE", "/v3/order", params=params, signed=True)
            return True
        except Exception as e:
            logger.error(f"Failed to cancel order {order_id}: {e}")
            return False
    
    async def get_order(self, symbol: str, order_id: str) -> Order:
        """Get order status"""
        binance_symbol = self._to_binance_symbol(symbol)
        
        params = {
            'symbol': binance_symbol,
            'orderId': order_id,
            'timestamp': int(time.time() * 1000)
        }
        
        data = await self._make_request("GET", "/v3/order", params=params, signed=True)
        return self._parse_order(data, symbol)
    
    async def get_open_orders(self, symbol: Optional[str] = None) -> List[Order]:
        """Get all open orders"""
        params = {'timestamp': int(time.time() * 1000)}
        
        if symbol:
            params['symbol'] = self._to_binance_symbol(symbol)
        
        data = await self._make_request("GET", "/v3/openOrders", params=params, signed=True)
        
        orders = []
        for order_data in data:
            # Determine normalized symbol
            binance_symbol = order_data['symbol']
            normalized_symbol = self._from_binance_symbol(binance_symbol)
            orders.append(self._parse_order(order_data, normalized_symbol))
        
        return orders
    
    async def get_fees(self, symbol: str) -> Dict[str, float]:
        """Get trading fees for symbol"""
        # Binance doesn't provide fee info per symbol in a simple way
        # Using account info which has general fee tier
        data = await self._make_request("GET", "/v3/account", signed=True)
        
        # Binance fees (these are typical values, actual fees depend on VIP level and BNB usage)
        return {
            'maker': data.get('makerCommission', 10) / 10000,  # Convert from basis points
            'taker': data.get('takerCommission', 10) / 10000
        }
    
    async def get_server_time(self) -> float:
        """Get exchange server time"""
        data = await self._make_request("GET", "/v3/time")
        return data['serverTime'] / 1000  # Convert to seconds
    
    # Helper methods
    
    def _to_binance_symbol(self, symbol: str) -> str:
        """Convert normalized symbol to Binance format"""
        # Remove any separators and uppercase
        return symbol.replace('/', '').replace('-', '').upper()
    
    def _from_binance_symbol(self, binance_symbol: str) -> str:
        """Convert Binance symbol to normalized format"""
        # Try to identify quote currency
        for quote in ['USDT', 'USDC', 'BUSD', 'USD', 'BTC', 'ETH', 'BNB']:
            if binance_symbol.endswith(quote):
                base = binance_symbol[:-len(quote)]
                return f"{base}/{quote}"
        return binance_symbol
    
    def _to_binance_order_type(self, order_type: OrderType) -> str:
        """Convert order type to Binance format"""
        mapping = {
            OrderType.MARKET: 'MARKET',
            OrderType.LIMIT: 'LIMIT',
            OrderType.STOP: 'STOP_LOSS',
            OrderType.STOP_LIMIT: 'STOP_LOSS_LIMIT'
        }
        return mapping.get(order_type, 'LIMIT')
    
    def _parse_order_status(self, status: str) -> OrderStatus:
        """Parse Binance order status"""
        mapping = {
            'NEW': OrderStatus.OPEN,
            'PARTIALLY_FILLED': OrderStatus.PARTIALLY_FILLED,
            'FILLED': OrderStatus.FILLED,
            'CANCELED': OrderStatus.CANCELLED,
            'REJECTED': OrderStatus.REJECTED,
            'EXPIRED': OrderStatus.EXPIRED
        }
        return mapping.get(status, OrderStatus.OPEN)
    
    def _parse_order(self, data: Dict, symbol: str) -> Order:
        """Parse order data from Binance"""
        return Order(
            exchange=self.name,
            symbol=symbol,
            order_id=str(data['orderId']),
            client_order_id=data.get('clientOrderId'),
            timestamp=data['time'] / 1000 if 'time' in data else time.time(),
            type=self._parse_order_type(data['type']),
            side=OrderSide.BUY if data['side'] == 'BUY' else OrderSide.SELL,
            price=float(data['price']) if data['price'] != '0' else None,
            quantity=float(data['origQty']),
            filled_quantity=float(data['executedQty']),
            status=self._parse_order_status(data['status'])
        )
    
    def _parse_order_type(self, binance_type: str) -> OrderType:
        """Parse Binance order type"""
        mapping = {
            'MARKET': OrderType.MARKET,
            'LIMIT': OrderType.LIMIT,
            'STOP_LOSS': OrderType.STOP,
            'STOP_LOSS_LIMIT': OrderType.STOP_LIMIT
        }
        return mapping.get(binance_type, OrderType.LIMIT)
    
    async def _make_request(
        self,
        method: str,
        endpoint: str,
        params: Optional[Dict] = None,
        signed: bool = False
    ) -> Any:
        """Make HTTP request to Binance API"""
        if not self._rest_session:
            self._rest_session = aiohttp.ClientSession()
        
        url = f"{self.rest_url}{endpoint}"
        
        if signed:
            if not self.config.api_key or not self.config.api_secret:
                raise ValueError("API key and secret required for signed requests")
            
            # Add timestamp if not present
            if params is None:
                params = {}
            if 'timestamp' not in params:
                params['timestamp'] = int(time.time() * 1000)
            
            # Create signature
            query_string = urlencode(params)
            signature = hmac.new(
                self.config.api_secret.encode('utf-8'),
                query_string.encode('utf-8'),
                hashlib.sha256
            ).hexdigest()
            
            params['signature'] = signature
        
        headers = {}
        if self.config.api_key:
            headers['X-MBX-APIKEY'] = self.config.api_key
        
        async with self._rest_session.request(
            method,
            url,
            params=params if method == "GET" else None,
            data=params if method != "GET" else None,
            headers=headers
        ) as response:
            data = await response.json()
            
            if response.status != 200:
                error_msg = data.get('msg', 'Unknown error')
                raise Exception(f"Binance API error: {error_msg}")
            
            return data