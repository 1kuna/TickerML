"""
Trading control router for paper trading management
"""

from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from typing import Dict, List, Optional
from datetime import datetime, timedelta
from pydantic import BaseModel
import sqlite3
import json
import yaml
from pathlib import Path
import logging
from enum import Enum

from app.routers.auth import get_current_user, require_role, User

router = APIRouter()
logger = logging.getLogger(__name__)

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"
RISK_CONFIG_PATH = PROJECT_ROOT / "config" / "risk_limits.yaml"
DB_PATH = PROJECT_ROOT / "data" / "db" / "crypto_data.db"

# Enums
class OrderSide(str, Enum):
    BUY = "buy"
    SELL = "sell"

class OrderType(str, Enum):
    MARKET = "market"
    LIMIT = "limit"

class TradingStatus(str, Enum):
    ACTIVE = "active"
    PAUSED = "paused"
    STOPPED = "stopped"

# Pydantic models
class TradingConfig(BaseModel):
    initial_balance: float = 10000
    max_position_size: float = 0.25
    max_positions: int = 5
    enable_stop_loss: bool = True
    stop_loss_percent: float = 0.05
    enable_take_profit: bool = True
    take_profit_percent: float = 0.10
    trading_pairs: List[str] = ["BTCUSD", "ETHUSD"]
    exchanges: List[str] = ["binance", "coinbase"]

class OrderRequest(BaseModel):
    symbol: str
    side: OrderSide
    type: OrderType
    quantity: float
    price: Optional[float] = None  # For limit orders

class TradingSettings(BaseModel):
    risk_limits: Dict[str, float]
    position_sizing: Dict[str, float]
    allowed_symbols: List[str]
    trading_hours: Optional[Dict[str, str]] = None
    circuit_breaker: Dict[str, float]

class TradeHistory(BaseModel):
    id: int
    timestamp: datetime
    symbol: str
    side: str
    quantity: float
    price: float
    pnl: Optional[float]
    commission: float
    exchange: str

class Position(BaseModel):
    symbol: str
    quantity: float
    entry_price: float
    current_price: float
    unrealized_pnl: float
    realized_pnl: float
    position_value: float
    weight_percent: float

# Helper functions
def load_config() -> Dict:
    """Load configuration from YAML"""
    try:
        with open(CONFIG_PATH, 'r') as f:
            return yaml.safe_load(f)
    except Exception as e:
        logger.error(f"Failed to load config: {e}")
        return {}

def load_risk_config() -> Dict:
    """Load risk configuration"""
    try:
        with open(RISK_CONFIG_PATH, 'r') as f:
            return yaml.safe_load(f)
    except Exception as e:
        logger.error(f"Failed to load risk config: {e}")
        return {}

def get_db_connection():
    """Get database connection"""
    return sqlite3.connect(DB_PATH, check_same_thread=False)

def get_trading_status() -> TradingStatus:
    """Get current trading status"""
    # In production, check actual paper trader status
    # For now, return mock status
    return TradingStatus.ACTIVE

def get_current_positions(conn) -> List[Dict]:
    """Get current open positions from database"""
    cursor = conn.cursor()
    cursor.execute("""
        SELECT symbol, quantity, entry_price
        FROM positions
        WHERE status = 'open'
    """)
    
    positions = []
    for row in cursor.fetchall():
        positions.append({
            "symbol": row[0],
            "quantity": row[1],
            "entry_price": row[2]
        })
    
    return positions

# API Endpoints
@router.get("/status")
async def get_trading_status_endpoint(
    current_user: User = Depends(get_current_user)
) -> Dict:
    """Get current trading status and summary"""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    try:
        # Get portfolio state
        cursor.execute("""
            SELECT cash_balance, total_value, daily_pnl, max_drawdown
            FROM portfolio_state
            ORDER BY timestamp DESC
            LIMIT 1
        """)
        portfolio = cursor.fetchone()
        
        if portfolio:
            cash_balance, total_value, daily_pnl, max_drawdown = portfolio
        else:
            cash_balance = 10000
            total_value = 10000
            daily_pnl = 0
            max_drawdown = 0
        
        # Get open positions
        positions = get_current_positions(conn)
        
        # Get today's trades
        today_start = datetime.now().replace(hour=0, minute=0, second=0).timestamp() * 1000
        cursor.execute("""
            SELECT COUNT(*), SUM(pnl)
            FROM paper_trades
            WHERE timestamp > ?
        """, (today_start,))
        trades_today, pnl_today = cursor.fetchone()
        
        conn.close()
        
        return {
            "status": get_trading_status().value,
            "portfolio": {
                "cash_balance": cash_balance,
                "total_value": total_value,
                "daily_pnl": daily_pnl,
                "max_drawdown": max_drawdown,
                "positions_count": len(positions),
                "positions_value": total_value - cash_balance
            },
            "today": {
                "trades_count": trades_today or 0,
                "pnl": pnl_today or 0
            },
            "risk_utilization": {
                "position_count": f"{len(positions)}/5",
                "capital_deployed": f"{((total_value - cash_balance) / total_value * 100):.1f}%",
                "drawdown": f"{(max_drawdown * 100):.1f}%"
            }
        }
        
    except Exception as e:
        conn.close()
        logger.error(f"Failed to get trading status: {e}")
        raise HTTPException(status_code=500, detail="Failed to get trading status")

@router.post("/start")
async def start_trading(
    config: TradingConfig,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(require_role("trader"))
) -> Dict[str, str]:
    """Start paper trading with configuration"""
    # Validate configuration
    if config.initial_balance <= 0:
        raise HTTPException(status_code=400, detail="Initial balance must be positive")
    
    if config.max_position_size > 1 or config.max_position_size <= 0:
        raise HTTPException(status_code=400, detail="Max position size must be between 0 and 1")
    
    # Save trading configuration
    trading_config = {
        "paper_trading": {
            "enabled": True,
            "initial_balance": config.initial_balance,
            "max_position_size": config.max_position_size,
            "max_positions": config.max_positions,
            "stop_loss": {
                "enabled": config.enable_stop_loss,
                "percent": config.stop_loss_percent
            },
            "take_profit": {
                "enabled": config.enable_take_profit,
                "percent": config.take_profit_percent
            },
            "trading_pairs": config.trading_pairs,
            "exchanges": config.exchanges
        }
    }
    
    # In production, this would start the actual paper trader
    # For now, just return success
    return {
        "status": "started",
        "message": "Paper trading started successfully",
        "config": trading_config
    }

@router.post("/stop")
async def stop_trading(
    current_user: User = Depends(require_role("trader"))
) -> Dict[str, str]:
    """Stop paper trading gracefully"""
    # In production, this would:
    # 1. Close all open positions at market
    # 2. Cancel all pending orders
    # 3. Stop the paper trader process
    
    return {
        "status": "stopped",
        "message": "Paper trading stopped. All positions closed.",
        "final_balance": 10234.56  # Mock value
    }

@router.post("/pause")
async def pause_trading(
    current_user: User = Depends(require_role("trader"))
) -> Dict[str, str]:
    """Pause paper trading (keep positions open)"""
    return {
        "status": "paused",
        "message": "Paper trading paused. Positions remain open."
    }

@router.post("/resume")
async def resume_trading(
    current_user: User = Depends(require_role("trader"))
) -> Dict[str, str]:
    """Resume paper trading"""
    return {
        "status": "resumed",
        "message": "Paper trading resumed."
    }

@router.post("/orders")
async def place_manual_order(
    order: OrderRequest,
    current_user: User = Depends(require_role("trader"))
) -> Dict:
    """Place manual order (for testing)"""
    # Validate order
    if order.quantity <= 0:
        raise HTTPException(status_code=400, detail="Quantity must be positive")
    
    if order.type == OrderType.LIMIT and order.price is None:
        raise HTTPException(status_code=400, detail="Price required for limit orders")
    
    # In production, this would send order to paper trader
    # For now, return mock response
    return {
        "order_id": "ORD-12345",
        "status": "submitted",
        "symbol": order.symbol,
        "side": order.side,
        "type": order.type,
        "quantity": order.quantity,
        "price": order.price,
        "timestamp": datetime.now().isoformat()
    }

@router.get("/settings")
async def get_trading_settings(
    current_user: User = Depends(get_current_user)
) -> TradingSettings:
    """Get current risk limits and trading settings"""
    risk_config = load_risk_config()
    
    return TradingSettings(
        risk_limits={
            "max_position_size": risk_config.get("max_position_size_percent", 25) / 100,
            "max_drawdown": risk_config.get("max_drawdown_percent", 25) / 100,
            "daily_loss_limit": risk_config.get("daily_loss_limit_percent", 5) / 100,
            "position_limit": risk_config.get("max_positions", 5)
        },
        position_sizing={
            "kelly_fraction": risk_config.get("position_sizing", {}).get("kelly_fraction", 0.25),
            "min_position": risk_config.get("position_sizing", {}).get("min_position_percent", 1) / 100,
            "max_position": risk_config.get("position_sizing", {}).get("max_position_percent", 25) / 100
        },
        allowed_symbols=risk_config.get("allowed_symbols", ["BTCUSD", "ETHUSD"]),
        circuit_breaker={
            "max_loss_per_minute": risk_config.get("circuit_breaker", {}).get("max_loss_per_minute_percent", 2) / 100,
            "volatility_halt_threshold": risk_config.get("circuit_breaker", {}).get("volatility_halt_threshold", 0.1),
            "correlation_threshold": risk_config.get("risk_management", {}).get("correlation_threshold", 0.7)
        }
    )

@router.put("/settings")
async def update_trading_settings(
    settings: TradingSettings,
    current_user: User = Depends(require_role("admin"))
) -> Dict[str, str]:
    """Update trading settings and risk limits"""
    # Convert settings back to config format
    risk_config = {
        "max_position_size_percent": settings.risk_limits["max_position_size"] * 100,
        "max_drawdown_percent": settings.risk_limits["max_drawdown"] * 100,
        "daily_loss_limit_percent": settings.risk_limits["daily_loss_limit"] * 100,
        "max_positions": int(settings.risk_limits["position_limit"]),
        "position_sizing": {
            "kelly_fraction": settings.position_sizing["kelly_fraction"],
            "min_position_percent": settings.position_sizing["min_position"] * 100,
            "max_position_percent": settings.position_sizing["max_position"] * 100
        },
        "allowed_symbols": settings.allowed_symbols,
        "circuit_breaker": {
            "max_loss_per_minute_percent": settings.circuit_breaker["max_loss_per_minute"] * 100,
            "volatility_halt_threshold": settings.circuit_breaker["volatility_halt_threshold"]
        },
        "risk_management": {
            "correlation_threshold": settings.circuit_breaker["correlation_threshold"]
        }
    }
    
    # Save configuration
    try:
        with open(RISK_CONFIG_PATH, 'w') as f:
            yaml.dump(risk_config, f, default_flow_style=False)
        
        return {
            "status": "updated",
            "message": "Trading settings updated successfully"
        }
    except Exception as e:
        logger.error(f"Failed to save settings: {e}")
        raise HTTPException(status_code=500, detail="Failed to save settings")

@router.get("/positions")
async def get_positions(
    current_user: User = Depends(get_current_user)
) -> List[Position]:
    """Get current open positions"""
    conn = get_db_connection()
    
    try:
        positions = []
        
        # Get portfolio total for weight calculation
        cursor = conn.cursor()
        cursor.execute("SELECT total_value FROM portfolio_state ORDER BY timestamp DESC LIMIT 1")
        total_value = cursor.fetchone()[0] if cursor.fetchone() else 10000
        
        # Mock positions (in production, query actual positions)
        mock_positions = [
            {
                "symbol": "BTCUSD",
                "quantity": 0.1,
                "entry_price": 42000,
                "current_price": 43500
            },
            {
                "symbol": "ETHUSD",
                "quantity": 2.5,
                "entry_price": 2200,
                "current_price": 2150
            }
        ]
        
        for pos in mock_positions:
            position_value = pos["quantity"] * pos["current_price"]
            unrealized_pnl = pos["quantity"] * (pos["current_price"] - pos["entry_price"])
            
            positions.append(Position(
                symbol=pos["symbol"],
                quantity=pos["quantity"],
                entry_price=pos["entry_price"],
                current_price=pos["current_price"],
                unrealized_pnl=unrealized_pnl,
                realized_pnl=0,
                position_value=position_value,
                weight_percent=(position_value / total_value) * 100
            ))
        
        conn.close()
        return positions
        
    except Exception as e:
        conn.close()
        logger.error(f"Failed to get positions: {e}")
        raise HTTPException(status_code=500, detail="Failed to get positions")

@router.get("/trades/history")
async def get_trade_history(
    limit: int = 50,
    offset: int = 0,
    current_user: User = Depends(get_current_user)
) -> List[TradeHistory]:
    """Get historical trades"""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    try:
        cursor.execute("""
            SELECT id, timestamp, symbol, side, quantity, price, pnl, commission, exchange
            FROM paper_trades
            ORDER BY timestamp DESC
            LIMIT ? OFFSET ?
        """, (limit, offset))
        
        trades = []
        for row in cursor.fetchall():
            trades.append(TradeHistory(
                id=row[0],
                timestamp=datetime.fromtimestamp(row[1] / 1000),
                symbol=row[2],
                side=row[3],
                quantity=row[4],
                price=row[5],
                pnl=row[6],
                commission=row[7],
                exchange=row[8] if len(row) > 8 else "unknown"
            ))
        
        conn.close()
        return trades
        
    except Exception as e:
        conn.close()
        logger.error(f"Failed to get trade history: {e}")
        raise HTTPException(status_code=500, detail="Failed to get trade history")