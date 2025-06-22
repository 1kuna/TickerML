"""
WebSocket endpoints for real-time data streaming
"""

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Depends, Query
from typing import List, Dict, Set
import json
import asyncio
import logging
from datetime import datetime
import sqlite3
from pathlib import Path

from app.routers.auth import get_current_user, User

router = APIRouter()
logger = logging.getLogger(__name__)

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
DB_PATH = PROJECT_ROOT / "data" / "db" / "crypto_data.db"

class ConnectionManager:
    """Manages WebSocket connections"""
    
    def __init__(self):
        self.active_connections: Dict[str, List[WebSocket]] = {
            "market": [],
            "portfolio": [],
            "trades": [],
            "orderbook": [],
            "alerts": []
        }
        self.subscriptions: Dict[WebSocket, Set[str]] = {}
    
    async def connect(self, websocket: WebSocket, channel: str):
        """Accept new WebSocket connection"""
        await websocket.accept()
        if channel in self.active_connections:
            self.active_connections[channel].append(websocket)
            if websocket not in self.subscriptions:
                self.subscriptions[websocket] = set()
            self.subscriptions[websocket].add(channel)
            logger.info(f"WebSocket connected to {channel}")
    
    def disconnect(self, websocket: WebSocket):
        """Remove WebSocket connection"""
        # Remove from all channels
        for channel, connections in self.active_connections.items():
            if websocket in connections:
                connections.remove(websocket)
        
        # Remove subscriptions
        if websocket in self.subscriptions:
            del self.subscriptions[websocket]
        
        logger.info("WebSocket disconnected")
    
    async def send_personal_message(self, message: str, websocket: WebSocket):
        """Send message to specific connection"""
        await websocket.send_text(message)
    
    async def broadcast(self, message: str, channel: str):
        """Broadcast message to all connections in a channel"""
        if channel in self.active_connections:
            # Send to all connections in parallel
            tasks = []
            for connection in self.active_connections[channel]:
                tasks.append(connection.send_text(message))
            
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)

# Global connection manager
manager = ConnectionManager()

# Background tasks for data streaming
async def stream_market_data():
    """Stream market data to connected clients"""
    while True:
        try:
            # Get latest market data from database
            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT symbol, close as price, volume, timestamp
                FROM ohlcv
                WHERE timestamp > ?
                GROUP BY symbol
                ORDER BY timestamp DESC
                LIMIT 10
            """, (datetime.now().timestamp() * 1000 - 60000,))  # Last minute
            
            market_data = []
            for row in cursor.fetchall():
                market_data.append({
                    "symbol": row[0],
                    "price": row[1],
                    "volume": row[2],
                    "timestamp": row[3]
                })
            
            conn.close()
            
            if market_data:
                message = json.dumps({
                    "type": "market_update",
                    "data": market_data,
                    "timestamp": datetime.now().isoformat()
                })
                await manager.broadcast(message, "market")
            
        except Exception as e:
            logger.error(f"Error streaming market data: {e}")
        
        await asyncio.sleep(5)  # Update every 5 seconds

async def stream_portfolio_updates():
    """Stream portfolio updates to connected clients"""
    while True:
        try:
            # Get latest portfolio data
            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT timestamp, cash_balance, total_value, daily_pnl
                FROM portfolio_state
                ORDER BY timestamp DESC
                LIMIT 1
            """)
            
            row = cursor.fetchone()
            if row:
                portfolio_data = {
                    "timestamp": row[0],
                    "cash_balance": row[1],
                    "total_value": row[2],
                    "daily_pnl": row[3]
                }
                
                message = json.dumps({
                    "type": "portfolio_update",
                    "data": portfolio_data,
                    "timestamp": datetime.now().isoformat()
                })
                await manager.broadcast(message, "portfolio")
            
            conn.close()
            
        except Exception as e:
            logger.error(f"Error streaming portfolio data: {e}")
        
        await asyncio.sleep(10)  # Update every 10 seconds

# Start background tasks
background_tasks = []

@router.on_event("startup")
async def startup_event():
    """Start background streaming tasks"""
    background_tasks.append(asyncio.create_task(stream_market_data()))
    background_tasks.append(asyncio.create_task(stream_portfolio_updates()))
    logger.info("WebSocket background tasks started")

@router.on_event("shutdown")
async def shutdown_event():
    """Cancel background tasks"""
    for task in background_tasks:
        task.cancel()
    logger.info("WebSocket background tasks cancelled")

# WebSocket endpoints
@router.websocket("/ws/market")
async def websocket_market_data(websocket: WebSocket):
    """WebSocket endpoint for real-time market data"""
    await manager.connect(websocket, "market")
    try:
        while True:
            # Keep connection alive and handle incoming messages
            data = await websocket.receive_text()
            
            # Handle subscription requests
            message = json.loads(data)
            if message.get("action") == "subscribe":
                symbols = message.get("symbols", [])
                # Store subscription preferences (not implemented in this example)
                await websocket.send_text(json.dumps({
                    "type": "subscription_confirmed",
                    "symbols": symbols
                }))
            
    except WebSocketDisconnect:
        manager.disconnect(websocket)

@router.websocket("/ws/portfolio")
async def websocket_portfolio(websocket: WebSocket):
    """WebSocket endpoint for portfolio updates"""
    await manager.connect(websocket, "portfolio")
    try:
        while True:
            # Keep connection alive
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(websocket)

@router.websocket("/ws/trades")
async def websocket_trades(websocket: WebSocket):
    """WebSocket endpoint for real-time trade updates"""
    await manager.connect(websocket, "trades")
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(websocket)

@router.websocket("/ws/orderbook")
async def websocket_orderbook(
    websocket: WebSocket,
    symbol: str = Query("BTCUSD"),
    depth: int = Query(20, ge=1, le=100)
):
    """WebSocket endpoint for order book updates"""
    await manager.connect(websocket, "orderbook")
    
    try:
        # Send initial configuration
        await websocket.send_text(json.dumps({
            "type": "config",
            "symbol": symbol,
            "depth": depth
        }))
        
        while True:
            # In production, stream actual order book data
            # For now, just keep connection alive
            await websocket.receive_text()
            
    except WebSocketDisconnect:
        manager.disconnect(websocket)

@router.websocket("/ws/alerts")
async def websocket_alerts(websocket: WebSocket):
    """WebSocket endpoint for system alerts"""
    await manager.connect(websocket, "alerts")
    try:
        # Send welcome message
        await websocket.send_text(json.dumps({
            "type": "connected",
            "message": "Connected to alerts stream"
        }))
        
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(websocket)

# HTTP endpoint to send alerts
@router.post("/broadcast/alert")
async def broadcast_alert(
    alert: Dict,
    current_user: User = Depends(get_current_user)
):
    """Broadcast an alert to all connected clients"""
    message = json.dumps({
        "type": "alert",
        "data": alert,
        "timestamp": datetime.now().isoformat()
    })
    await manager.broadcast(message, "alerts")
    return {"status": "broadcast", "connections": len(manager.active_connections["alerts"])}