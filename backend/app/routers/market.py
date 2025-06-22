"""
Market data router for real-time and historical market information
"""

from fastapi import APIRouter, Depends, HTTPException, Query
from typing import Dict, List, Optional
from pydantic import BaseModel
from datetime import datetime, timedelta
from pathlib import Path
import sqlite3
import pandas as pd
import logging
import json

from app.routers.auth import get_current_user, User

router = APIRouter()
logger = logging.getLogger(__name__)

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
DB_PATH = PROJECT_ROOT / "data" / "db" / "crypto_data.db"

# Pydantic models
class MarketData(BaseModel):
    symbol: str
    exchange: str
    price: float
    volume_24h: float
    change_24h: float
    change_percent_24h: float
    high_24h: float
    low_24h: float
    last_updated: datetime

class OHLCV(BaseModel):
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float

class OrderBookEntry(BaseModel):
    price: float
    quantity: float
    count: Optional[int] = None

class OrderBook(BaseModel):
    symbol: str
    exchange: str
    bids: List[OrderBookEntry]
    asks: List[OrderBookEntry]
    mid_price: float
    spread: float
    spread_percent: float
    last_updated: datetime

class Trade(BaseModel):
    id: str
    timestamp: datetime
    symbol: str
    exchange: str
    price: float
    quantity: float
    side: str  # buy/sell
    is_maker: Optional[bool] = None

class TradingPair(BaseModel):
    symbol: str
    base_asset: str
    quote_asset: str
    price_precision: int
    quantity_precision: int
    min_quantity: float
    max_quantity: float
    supported_exchanges: List[str]

class MarketSummary(BaseModel):
    total_symbols: int
    total_exchanges: int
    total_24h_volume: float
    top_gainers: List[MarketData]
    top_losers: List[MarketData]
    most_active: List[MarketData]

# Helper functions
def get_db_connection():
    """Get database connection"""
    return sqlite3.connect(DB_PATH, check_same_thread=False)

def calculate_24h_change(symbol: str, exchange: str) -> tuple:
    """Calculate 24h price change and percentage"""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    try:
        # Get current price
        cursor.execute("""
            SELECT close FROM ohlcv
            WHERE symbol = ? AND exchange = ?
            ORDER BY timestamp DESC LIMIT 1
        """, (symbol, exchange))
        current_price = cursor.fetchone()
        
        if not current_price:
            return 0, 0
        current_price = current_price[0]
        
        # Get price 24h ago
        cutoff_time = datetime.now().timestamp() * 1000 - (24 * 3600 * 1000)
        cursor.execute("""
            SELECT close FROM ohlcv
            WHERE symbol = ? AND exchange = ? AND timestamp <= ?
            ORDER BY timestamp DESC LIMIT 1
        """, (symbol, exchange, cutoff_time))
        old_price = cursor.fetchone()
        
        if not old_price:
            return 0, 0
        old_price = old_price[0]
        
        change = current_price - old_price
        change_percent = (change / old_price) * 100 if old_price > 0 else 0
        
        conn.close()
        return change, change_percent
        
    except Exception as e:
        conn.close()
        logger.error(f"Error calculating 24h change: {e}")
        return 0, 0

# API Endpoints
@router.get("/prices")
async def get_market_prices(
    symbols: Optional[str] = Query(None, description="Comma-separated list of symbols"),
    exchange: Optional[str] = Query(None, description="Filter by exchange"),
    current_user: User = Depends(get_current_user)
) -> List[MarketData]:
    """Get current market prices"""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    try:
        # Build query
        query = """
            SELECT DISTINCT o1.symbol, o1.exchange, o1.close, o1.volume, o1.high, o1.low, o1.timestamp
            FROM ohlcv o1
            INNER JOIN (
                SELECT symbol, exchange, MAX(timestamp) as max_timestamp
                FROM ohlcv
                WHERE timestamp > ?
                GROUP BY symbol, exchange
            ) o2 ON o1.symbol = o2.symbol AND o1.exchange = o2.exchange AND o1.timestamp = o2.max_timestamp
        """
        
        params = [datetime.now().timestamp() * 1000 - 86400000]  # Last 24 hours
        
        if exchange:
            query += " WHERE o1.exchange = ?"
            params.append(exchange)
        
        if symbols:
            symbol_list = [s.strip() for s in symbols.split(',')]
            if exchange:
                query += " AND o1.symbol IN ({})".format(','.join(['?'] * len(symbol_list)))
            else:
                query += " WHERE o1.symbol IN ({})".format(','.join(['?'] * len(symbol_list)))
            params.extend(symbol_list)
        
        cursor.execute(query, params)
        
        market_data = []
        for row in cursor.fetchall():
            symbol, exchange, price, volume, high, low, timestamp = row
            
            # Calculate 24h change
            change, change_percent = calculate_24h_change(symbol, exchange)
            
            market_data.append(MarketData(
                symbol=symbol,
                exchange=exchange,
                price=price,
                volume_24h=volume,
                change_24h=change,
                change_percent_24h=change_percent,
                high_24h=high,
                low_24h=low,
                last_updated=datetime.fromtimestamp(timestamp / 1000)
            ))
        
        conn.close()
        return market_data
        
    except Exception as e:
        conn.close()
        logger.error(f"Failed to get market prices: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve market prices")

@router.get("/ohlcv/{symbol}")
async def get_ohlcv_data(
    symbol: str,
    exchange: Optional[str] = Query(None),
    interval: str = Query("1h", description="Time interval (1m, 5m, 1h, 1d)"),
    start_time: Optional[datetime] = None,
    end_time: Optional[datetime] = None,
    limit: int = Query(100, ge=1, le=1000),
    current_user: User = Depends(get_current_user)
) -> List[OHLCV]:
    """Get OHLCV data for a symbol"""
    conn = get_db_connection()
    
    try:
        # Set default time range if not provided
        if not end_time:
            end_time = datetime.now()
        if not start_time:
            start_time = end_time - timedelta(days=7)
        
        query = """
            SELECT timestamp, open, high, low, close, volume
            FROM ohlcv
            WHERE symbol = ? AND timestamp BETWEEN ? AND ?
        """
        params = [symbol, start_time.timestamp() * 1000, end_time.timestamp() * 1000]
        
        if exchange:
            query += " AND exchange = ?"
            params.append(exchange)
        
        query += " ORDER BY timestamp ASC LIMIT ?"
        params.append(limit)
        
        df = pd.read_sql_query(query, conn, params=params)
        conn.close()
        
        if df.empty:
            return []
        
        ohlcv_data = []
        for _, row in df.iterrows():
            ohlcv_data.append(OHLCV(
                timestamp=datetime.fromtimestamp(row['timestamp'] / 1000),
                open=row['open'],
                high=row['high'],
                low=row['low'],
                close=row['close'],
                volume=row['volume']
            ))
        
        return ohlcv_data
        
    except Exception as e:
        conn.close()
        logger.error(f"Failed to get OHLCV data: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve OHLCV data")

@router.get("/orderbook/{symbol}")
async def get_order_book(
    symbol: str,
    exchange: str,
    depth: int = Query(20, ge=1, le=100),
    current_user: User = Depends(get_current_user)
) -> OrderBook:
    """Get current order book for a symbol"""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    try:
        # Get latest order book data
        cursor.execute("""
            SELECT bids, asks, mid_price, spread, timestamp
            FROM order_books
            WHERE symbol = ? AND exchange = ?
            ORDER BY timestamp DESC LIMIT 1
        """, (symbol, exchange))
        
        row = cursor.fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="Order book not found")
        
        bids_json, asks_json, mid_price, spread, timestamp = row
        
        # Parse JSON data
        bids_data = json.loads(bids_json) if bids_json else []
        asks_data = json.loads(asks_json) if asks_json else []
        
        # Convert to OrderBookEntry objects (limit by depth)
        bids = [OrderBookEntry(price=bid[0], quantity=bid[1]) for bid in bids_data[:depth]]
        asks = [OrderBookEntry(price=ask[0], quantity=ask[1]) for ask in asks_data[:depth]]
        
        spread_percent = (spread / mid_price) * 100 if mid_price > 0 else 0
        
        conn.close()
        
        return OrderBook(
            symbol=symbol,
            exchange=exchange,
            bids=bids,
            asks=asks,
            mid_price=mid_price,
            spread=spread,
            spread_percent=spread_percent,
            last_updated=datetime.fromtimestamp(timestamp / 1000)
        )
        
    except Exception as e:
        conn.close()
        logger.error(f"Failed to get order book: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve order book")

@router.get("/trades/{symbol}")
async def get_recent_trades(
    symbol: str,
    exchange: Optional[str] = Query(None),
    limit: int = Query(50, ge=1, le=500),
    current_user: User = Depends(get_current_user)
) -> List[Trade]:
    """Get recent trades for a symbol"""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    try:
        query = """
            SELECT id, timestamp, symbol, exchange, price, quantity, side
            FROM trades
            WHERE symbol = ?
        """
        params = [symbol]
        
        if exchange:
            query += " AND exchange = ?"
            params.append(exchange)
        
        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)
        
        cursor.execute(query, params)
        
        trades = []
        for row in cursor.fetchall():
            trades.append(Trade(
                id=str(row[0]),
                timestamp=datetime.fromtimestamp(row[1] / 1000),
                symbol=row[2],
                exchange=row[3],
                price=row[4],
                quantity=row[5],
                side=row[6]
            ))
        
        conn.close()
        return trades
        
    except Exception as e:
        conn.close()
        logger.error(f"Failed to get trades: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve trades")

@router.get("/symbols")
async def get_trading_pairs(
    current_user: User = Depends(get_current_user)
) -> List[TradingPair]:
    """Get list of supported trading pairs"""
    # Mock data for supported trading pairs
    return [
        TradingPair(
            symbol="BTCUSD",
            base_asset="BTC",
            quote_asset="USD",
            price_precision=2,
            quantity_precision=8,
            min_quantity=0.0001,
            max_quantity=100.0,
            supported_exchanges=["binance", "coinbase", "kraken"]
        ),
        TradingPair(
            symbol="ETHUSD",
            base_asset="ETH",
            quote_asset="USD",
            price_precision=2,
            quantity_precision=6,
            min_quantity=0.001,
            max_quantity=1000.0,
            supported_exchanges=["binance", "coinbase", "kraken"]
        ),
        TradingPair(
            symbol="BTCUSDT",
            base_asset="BTC",
            quote_asset="USDT",
            price_precision=2,
            quantity_precision=8,
            min_quantity=0.0001,
            max_quantity=100.0,
            supported_exchanges=["binance", "kucoin"]
        ),
        TradingPair(
            symbol="ETHUSDT",
            base_asset="ETH",
            quote_asset="USDT",
            price_precision=2,
            quantity_precision=6,
            min_quantity=0.001,
            max_quantity=1000.0,
            supported_exchanges=["binance", "kucoin"]
        )
    ]

@router.get("/summary")
async def get_market_summary(
    current_user: User = Depends(get_current_user)
) -> MarketSummary:
    """Get market summary with statistics"""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    try:
        # Get basic statistics
        cursor.execute("SELECT COUNT(DISTINCT symbol) FROM ohlcv")
        total_symbols = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(DISTINCT exchange) FROM ohlcv")
        total_exchanges = cursor.fetchone()[0]
        
        # Get 24h volume
        cutoff_time = datetime.now().timestamp() * 1000 - (24 * 3600 * 1000)
        cursor.execute("""
            SELECT SUM(volume) FROM ohlcv
            WHERE timestamp > ?
        """, (cutoff_time,))
        total_volume = cursor.fetchone()[0] or 0
        
        # Get top gainers, losers, and most active (simplified)
        # In a real implementation, you'd calculate these from price changes
        mock_gainers = [
            MarketData(
                symbol="ETHUSD",
                exchange="binance",
                price=2250.75,
                volume_24h=987654.32,
                change_24h=156.23,
                change_percent_24h=7.48,
                high_24h=2280.00,
                low_24h=2100.00,
                last_updated=datetime.now()
            )
        ]
        
        mock_losers = [
            MarketData(
                symbol="BTCUSD",
                exchange="coinbase",
                price=43200.50,
                volume_24h=1234567.89,
                change_24h=-1200.00,
                change_percent_24h=-2.7,
                high_24h=44500.00,
                low_24h=42800.00,
                last_updated=datetime.now()
            )
        ]
        
        mock_active = [
            MarketData(
                symbol="BTCUSD",
                exchange="binance",
                price=43500.50,
                volume_24h=2345678.90,
                change_24h=500.00,
                change_percent_24h=1.16,
                high_24h=44000.00,
                low_24h=43000.00,
                last_updated=datetime.now()
            )
        ]
        
        conn.close()
        
        return MarketSummary(
            total_symbols=total_symbols,
            total_exchanges=total_exchanges,
            total_24h_volume=total_volume,
            top_gainers=mock_gainers,
            top_losers=mock_losers,
            most_active=mock_active
        )
        
    except Exception as e:
        conn.close()
        logger.error(f"Failed to get market summary: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve market summary")

@router.get("/exchanges")
async def get_exchanges(
    current_user: User = Depends(get_current_user)
) -> List[Dict]:
    """Get list of supported exchanges"""
    return [
        {
            "id": "binance",
            "name": "Binance.US",
            "status": "active",
            "fee_rate": 0.001,
            "supported_symbols": ["BTCUSD", "ETHUSD", "BTCUSDT", "ETHUSDT"],
            "api_status": "connected"
        },
        {
            "id": "coinbase",
            "name": "Coinbase",
            "status": "active",
            "fee_rate": 0.005,
            "supported_symbols": ["BTCUSD", "ETHUSD"],
            "api_status": "connected"
        },
        {
            "id": "kraken",
            "name": "Kraken",
            "status": "active",
            "fee_rate": 0.0026,
            "supported_symbols": ["BTCUSD", "ETHUSD"],
            "api_status": "connected"
        },
        {
            "id": "kucoin",
            "name": "KuCoin",
            "status": "maintenance",
            "fee_rate": 0.001,
            "supported_symbols": ["BTCUSDT", "ETHUSDT"],
            "api_status": "disconnected"
        }
    ]