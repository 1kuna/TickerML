#!/usr/bin/env python3
"""
Kraken Exchange Implementation
Supports Kraken REST and WebSocket APIs
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
from urllib.parse import urlencode

from .base import (
    ExchangeInterface, ExchangeConfig, OrderBook, Trade, Order, Balance,
    OrderType, OrderSide, OrderStatus
)

logger = logging.getLogger(__name__)

class KrakenExchange(ExchangeInterface):
    """Kraken exchange implementation"""
    
    def __init__(self, config: ExchangeConfig):
        super().__init__(config)
        
        # Set URLs
        self.rest_url = config.rest_url or "https://api.kraken.com"
        self.ws_url = config.ws_url or "wss://ws.kraken.com"
        
        self._orderbook_cache: Dict[str, Dict] = {}
        self._ws_connection = None
        self._channel_map = {}  # Map channel IDs to symbols
        self._asset_pairs = {}  # Cache for asset pair info
        
    async def connect(self) -> None:
        """Initialize connection to Kraken"""
        if not self._rest_session:
            self._rest_session = aiohttp.ClientSession()
        
        # Test connectivity and load asset pairs
        try:
            await self._make_request("GET", "/0/public/Time")
            await self._load_asset_pairs()
            logger.info(f"Connected to Kraken ({self.rest_url})")
        except Exception as e:
            logger.error(f"Failed to connect to Kraken: {e}")
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
        
        logger.info("Disconnected from Kraken")
    
    async def get_symbols(self) -> List[str]:
        """Get list of available trading symbols"""
        if not self._asset_pairs:
            await self._load_asset_pairs()
        
        symbols = []
        for pair_info in self._asset_pairs.values():
            # Convert to normalized format
            symbols.append(pair_info['normalized'])
        
        return symbols
    
    async def get_orderbook(self, symbol: str, depth: int = 20) -> OrderBook:
        """Get current order book snapshot"""
        kraken_symbol = self._to_kraken_symbol(symbol)
        
        params = {
            'pair': kraken_symbol,
            'count': min(depth, 500)  # Kraken max
        }
        
        data = await self._make_request("GET", "/0/public/Depth", params=params)
        
        # Kraken returns data under the symbol name
        orderbook_data = list(data['result'].values())[0]
        
        # Convert to OrderBook format
        bids = [(float(bid[0]), float(bid[1])) for bid in orderbook_data['bids']]
        asks = [(float(ask[0]), float(ask[1])) for ask in orderbook_data['asks']]
        
        return OrderBook(
            exchange=self.name,
            symbol=symbol,
            timestamp=time.time(),
            bids=bids[:depth],
            asks=asks[:depth]
        )
    
    async def subscribe_orderbook(self, symbol: str, callback: Callable[[OrderBook], None]) -> None:
        """Subscribe to real-time order book updates"""
        self.register_orderbook_callback(symbol, callback)
        
        kraken_symbol = self._to_kraken_symbol(symbol)
        
        # Start WebSocket if not already running
        if not self._ws_connection:
            await self._start_websocket()
        
        # Subscribe to book channel
        subscribe_msg = {
            "event": "subscribe",
            "pair": [kraken_symbol],
            "subscription": {
                "name": "book",
                "depth": 25
            }
        }
        
        await self._ws_connection.send(json.dumps(subscribe_msg))
        logger.info(f"Subscribed to {symbol} orderbook on Kraken")
    
    async def unsubscribe_orderbook(self, symbol: str) -> None:
        """Unsubscribe from order book updates"""
        kraken_symbol = self._to_kraken_symbol(symbol)
        
        if self._ws_connection:
            unsubscribe_msg = {
                "event": "unsubscribe",
                "pair": [kraken_symbol],
                "subscription": {
                    "name": "book"
                }
            }
            await self._ws_connection.send(json.dumps(unsubscribe_msg))
        
        if symbol in self._orderbook_callbacks:
            del self._orderbook_callbacks[symbol]
        
        logger.info(f"Unsubscribed from {symbol} orderbook on Kraken")
    
    async def subscribe_trades(self, symbol: str, callback: Callable[[Trade], None]) -> None:
        """Subscribe to real-time trades"""
        self.register_trade_callback(symbol, callback)
        
        kraken_symbol = self._to_kraken_symbol(symbol)
        
        # Start WebSocket if not already running
        if not self._ws_connection:
            await self._start_websocket()
        
        # Subscribe to trade channel
        subscribe_msg = {
            "event": "subscribe",
            "pair": [kraken_symbol],
            "subscription": {
                "name": "trade"
            }
        }
        
        await self._ws_connection.send(json.dumps(subscribe_msg))
        logger.info(f"Subscribed to {symbol} trades on Kraken")
    
    async def unsubscribe_trades(self, symbol: str) -> None:
        """Unsubscribe from trades"""
        kraken_symbol = self._to_kraken_symbol(symbol)
        
        if self._ws_connection:
            unsubscribe_msg = {
                "event": "unsubscribe",
                "pair": [kraken_symbol],
                "subscription": {
                    "name": "trade"
                }
            }
            await self._ws_connection.send(json.dumps(unsubscribe_msg))
        
        if symbol in self._trade_callbacks:
            del self._trade_callbacks[symbol]
        
        logger.info(f"Unsubscribed from {symbol} trades on Kraken")
    
    async def get_balance(self) -> Dict[str, Balance]:
        """Get account balances"""
        data = await self._make_request("POST", "/0/private/Balance", signed=True)
        
        balances = {}
        for currency, balance_str in data['result'].items():
            balance = float(balance_str)
            if balance > 0:
                # Kraken doesn't separate free/locked in balance endpoint
                # Would need to check open orders for locked amounts
                balances[currency] = Balance(
                    currency=currency,
                    free=balance,
                    locked=0,
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
        kraken_symbol = self._to_kraken_symbol(symbol)
        
        params = {
            'pair': kraken_symbol,
            'type': side.value,
            'ordertype': self._to_kraken_order_type(order_type),
            'volume': str(quantity)
        }
        
        if order_type in [OrderType.LIMIT, OrderType.STOP_LIMIT]:
            if price is None:
                raise ValueError("Price required for limit orders")
            params['price'] = str(price)
        
        if client_order_id:
            params['userref'] = client_order_id
        
        data = await self._make_request("POST", "/0/private/AddOrder", params=params, signed=True)
        
        # Kraken returns order ID in txid
        order_ids = data['result']['txid']
        order_id = order_ids[0] if order_ids else None
        
        return Order(
            exchange=self.name,
            symbol=symbol,
            order_id=order_id,
            client_order_id=client_order_id,
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
        params = {'txid': order_id}
        
        try:
            await self._make_request("POST", "/0/private/CancelOrder", params=params, signed=True)
            return True
        except Exception as e:
            logger.error(f"Failed to cancel order {order_id}: {e}")
            return False
    
    async def get_order(self, symbol: str, order_id: str) -> Order:
        """Get order status"""
        params = {'txid': order_id}
        data = await self._make_request("POST", "/0/private/QueryOrders", params=params, signed=True)
        
        order_data = data['result'][order_id]
        return self._parse_order(order_id, order_data, symbol)
    
    async def get_open_orders(self, symbol: Optional[str] = None) -> List[Order]:
        """Get all open orders"""
        data = await self._make_request("POST", "/0/private/OpenOrders", signed=True)
        
        orders = []
        for order_id, order_data in data['result']['open'].items():
            # Get symbol from order data
            kraken_symbol = order_data['descr']['pair']
            normalized_symbol = self._from_kraken_symbol(kraken_symbol)
            
            if symbol is None or normalized_symbol == symbol:
                orders.append(self._parse_order(order_id, order_data, normalized_symbol))
        
        return orders
    
    async def get_fees(self, symbol: str) -> Dict[str, float]:
        """Get trading fees for symbol"""
        kraken_symbol = self._to_kraken_symbol(symbol)
        
        # Get fee info - this requires an order volume check
        # Using typical Kraken fees
        return {
            'maker': 0.0016,  # 0.16%
            'taker': 0.0026   # 0.26%
        }
    
    async def get_server_time(self) -> float:
        """Get exchange server time"""
        data = await self._make_request("GET", "/0/public/Time")
        return float(data['result']['unixtime'])
    
    # Helper methods
    
    async def _load_asset_pairs(self) -> None:
        """Load asset pair information from Kraken"""
        data = await self._make_request("GET", "/0/public/AssetPairs")
        
        for pair_name, pair_info in data['result'].items():
            # Skip .d (dark pool) pairs
            if '.d' in pair_name:
                continue
            
            # Get base and quote currencies
            base = pair_info['base']
            quote = pair_info['quote']
            
            # Convert to normalized format
            normalized = f"{base}/{quote}"
            
            self._asset_pairs[pair_name] = {
                'base': base,
                'quote': quote,
                'normalized': normalized,
                'wsname': pair_info.get('wsname', pair_name)
            }
    
    def _to_kraken_symbol(self, symbol: str) -> str:
        """Convert normalized symbol to Kraken format"""
        # Find matching asset pair
        for pair_name, pair_info in self._asset_pairs.items():
            if pair_info['normalized'] == symbol:
                return pair_name
        
        # Fallback: try to construct
        parts = symbol.split('/')
        if len(parts) == 2:
            base, quote = parts
            # Kraken uses specific naming conventions
            return f"{base}{quote}"
        
        return symbol.replace('/', '')
    
    def _from_kraken_symbol(self, kraken_symbol: str) -> str:
        """Convert Kraken symbol to normalized format"""
        if kraken_symbol in self._asset_pairs:
            return self._asset_pairs[kraken_symbol]['normalized']
        
        # Fallback parsing
        # This is approximate - Kraken symbols can be complex
        if kraken_symbol.endswith('USD'):
            base = kraken_symbol[:-3]
            return f"{base}/USD"
        elif kraken_symbol.endswith('USDT'):
            base = kraken_symbol[:-4]
            return f"{base}/USDT"
        elif kraken_symbol.endswith('EUR'):
            base = kraken_symbol[:-3]
            return f"{base}/EUR"
        elif kraken_symbol.endswith('BTC'):
            base = kraken_symbol[:-3]
            return f"{base}/BTC"
        
        return kraken_symbol
    
    def _to_kraken_order_type(self, order_type: OrderType) -> str:
        """Convert order type to Kraken format"""
        mapping = {
            OrderType.MARKET: 'market',
            OrderType.LIMIT: 'limit',
            OrderType.STOP: 'stop-loss',
            OrderType.STOP_LIMIT: 'stop-loss-limit'
        }
        return mapping.get(order_type, 'limit')
    
    def _parse_order_status(self, status: str) -> OrderStatus:
        """Parse Kraken order status"""
        mapping = {
            'pending': OrderStatus.PENDING,
            'open': OrderStatus.OPEN,
            'closed': OrderStatus.FILLED,
            'canceled': OrderStatus.CANCELLED,
            'expired': OrderStatus.EXPIRED
        }
        return mapping.get(status, OrderStatus.PENDING)
    
    def _parse_order(self, order_id: str, data: Dict, symbol: str) -> Order:
        """Parse order data from Kraken"""
        desc = data['descr']
        
        return Order(
            exchange=self.name,
            symbol=symbol,
            order_id=order_id,
            client_order_id=data.get('userref'),
            timestamp=float(data['opentm']),
            type=self._parse_order_type(desc['ordertype']),
            side=OrderSide.BUY if desc['type'] == 'buy' else OrderSide.SELL,
            price=float(desc['price']) if desc['price'] != '0' else None,
            quantity=float(data['vol']),
            filled_quantity=float(data['vol_exec']),
            status=self._parse_order_status(data['status'])
        )
    
    def _parse_order_type(self, kraken_type: str) -> OrderType:
        """Parse Kraken order type"""
        mapping = {
            'market': OrderType.MARKET,
            'limit': OrderType.LIMIT,
            'stop-loss': OrderType.STOP,
            'stop-loss-limit': OrderType.STOP_LIMIT
        }
        return mapping.get(kraken_type, OrderType.LIMIT)
    
    async def _start_websocket(self) -> None:
        """Start WebSocket connection and handler"""
        async def ws_handler():
            try:
                async with websockets.connect(self.ws_url) as websocket:
                    self._ws_connection = websocket
                    logger.info("Connected to Kraken WebSocket")
                    
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
    
    async def _handle_ws_message(self, data) -> None:
        """Handle WebSocket message"""
        if isinstance(data, dict):
            # System messages
            if 'event' in data:
                if data['event'] == 'subscriptionStatus':
                    if 'channelID' in data:
                        # Map channel ID to symbol
                        symbol = data.get('pair', '')
                        if symbol:
                            normalized = self._from_kraken_symbol(symbol)
                            self._channel_map[data['channelID']] = normalized
                            logger.info(f"Mapped channel {data['channelID']} to {normalized}")
                
        elif isinstance(data, list) and len(data) >= 4:
            # Data message: [channelID, data, channelName, pair]
            channel_id = data[0]
            message_data = data[1]
            channel_name = data[2]
            pair = data[3] if len(data) > 3 else None
            
            symbol = self._channel_map.get(channel_id)
            if not symbol and pair:
                symbol = self._from_kraken_symbol(pair)
                self._channel_map[channel_id] = symbol
            
            if not symbol:
                return
            
            if channel_name == 'book':
                await self._handle_orderbook_update(symbol, message_data)
            elif channel_name == 'trade':
                await self._handle_trade_update(symbol, message_data)
    
    async def _handle_orderbook_update(self, symbol: str, data: Dict) -> None:
        """Handle orderbook update"""
        # Kraken sends full snapshot or incremental updates
        timestamp = time.time()
        
        bids = []
        asks = []
        
        # Parse orderbook data
        if 'bs' in data:  # Bid snapshot
            bids = [(float(bid[0]), float(bid[1])) for bid in data['bs']]
        if 'as' in data:  # Ask snapshot
            asks = [(float(ask[0]), float(ask[1])) for ask in data['as']]
        
        if 'b' in data:  # Bid updates
            bids = [(float(bid[0]), float(bid[1])) for bid in data['b']]
        if 'a' in data:  # Ask updates
            asks = [(float(ask[0]), float(ask[1])) for ask in data['a']]
        
        if bids or asks:
            # For incremental updates, we should maintain full orderbook state
            # For simplicity, treating as snapshots here
            orderbook = OrderBook(
                exchange=self.name,
                symbol=symbol,
                timestamp=timestamp,
                bids=sorted(bids, reverse=True)[:25],
                asks=sorted(asks)[:25]
            )
            
            await self._emit_orderbook_update(symbol, orderbook)
    
    async def _handle_trade_update(self, symbol: str, trades: List) -> None:
        """Handle trade update"""
        for trade_data in trades:
            # Kraken trade format: [price, volume, time, side, orderType, misc]
            trade = Trade(
                exchange=self.name,
                symbol=symbol,
                timestamp=float(trade_data[2]),
                price=float(trade_data[0]),
                quantity=float(trade_data[1]),
                side=OrderSide.BUY if trade_data[3] == 'b' else OrderSide.SELL
            )
            
            await self._emit_trade_update(symbol, trade)
    
    async def _make_request(
        self,
        method: str,
        endpoint: str,
        params: Optional[Dict] = None,
        signed: bool = False
    ) -> Any:
        """Make HTTP request to Kraken API"""
        if not self._rest_session:
            self._rest_session = aiohttp.ClientSession()
        
        url = f"{self.rest_url}{endpoint}"
        
        headers = {
            'User-Agent': 'TickerML/1.0'
        }
        
        if signed:
            if not self.config.api_key or not self.config.api_secret:
                raise ValueError("API key and secret required for signed requests")
            
            # Add nonce
            if params is None:
                params = {}
            params['nonce'] = str(int(time.time() * 1000000))
            
            # Create signature
            postdata = urlencode(params)
            encoded = (str(params['nonce']) + postdata).encode('utf-8')
            message = endpoint.encode('utf-8') + hashlib.sha256(encoded).digest()
            
            signature = hmac.new(
                base64.b64decode(self.config.api_secret),
                message,
                hashlib.sha512
            )
            
            headers.update({
                'API-Key': self.config.api_key,
                'API-Sign': base64.b64encode(signature.digest()).decode()
            })
        
        async with self._rest_session.request(
            method,
            url,
            params=params if method == "GET" else None,
            data=params if method != "GET" else None,
            headers=headers
        ) as response:
            data = await response.json()
            
            if 'error' in data and data['error']:
                error_msg = ', '.join(data['error'])
                raise Exception(f"Kraken API error: {error_msg}")
            
            return data