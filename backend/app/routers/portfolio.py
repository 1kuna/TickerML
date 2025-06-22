"""
Portfolio tracking and analytics router
"""

from fastapi import APIRouter, Depends, HTTPException, Query
from typing import Dict, List, Optional
from datetime import datetime, timedelta
from pydantic import BaseModel
import sqlite3
import pandas as pd
import numpy as np
from pathlib import Path
import logging

from app.routers.auth import get_current_user, User

router = APIRouter()
logger = logging.getLogger(__name__)

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
DB_PATH = PROJECT_ROOT / "data" / "db" / "crypto_data.db"

# Pydantic models
class PortfolioSnapshot(BaseModel):
    timestamp: datetime
    cash_balance: float
    total_value: float
    positions_value: float
    daily_pnl: float
    daily_return: float
    total_return: float
    max_drawdown: float
    positions_count: int

class PerformanceMetrics(BaseModel):
    total_return: float
    total_return_percent: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    max_drawdown_duration_days: int
    win_rate: float
    profit_factor: float
    avg_win: float
    avg_loss: float
    best_trade: float
    worst_trade: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    avg_trade_duration_hours: float
    risk_adjusted_return: float

class PositionHistory(BaseModel):
    symbol: str
    entry_time: datetime
    exit_time: Optional[datetime]
    entry_price: float
    exit_price: Optional[float]
    quantity: float
    side: str
    pnl: Optional[float]
    pnl_percent: Optional[float]
    duration_hours: Optional[float]
    max_profit: float
    max_loss: float

class RiskMetrics(BaseModel):
    current_var: float  # Value at Risk
    current_cvar: float  # Conditional VaR
    beta: float
    correlation_matrix: Dict[str, Dict[str, float]]
    position_concentration: Dict[str, float]
    sector_exposure: Dict[str, float]
    volatility_30d: float
    downside_deviation: float
    risk_score: float  # 0-100

# Helper functions
def get_db_connection():
    """Get database connection"""
    return sqlite3.connect(DB_PATH, check_same_thread=False)

def calculate_sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.02) -> float:
    """Calculate Sharpe ratio"""
    if len(returns) < 2:
        return 0.0
    
    excess_returns = returns - risk_free_rate / 252  # Daily risk-free rate
    return np.sqrt(252) * excess_returns.mean() / excess_returns.std() if excess_returns.std() > 0 else 0.0

def calculate_sortino_ratio(returns: pd.Series, risk_free_rate: float = 0.02) -> float:
    """Calculate Sortino ratio (downside deviation)"""
    if len(returns) < 2:
        return 0.0
    
    excess_returns = returns - risk_free_rate / 252
    downside_returns = excess_returns[excess_returns < 0]
    
    if len(downside_returns) == 0:
        return float('inf')
    
    downside_std = downside_returns.std()
    return np.sqrt(252) * excess_returns.mean() / downside_std if downside_std > 0 else 0.0

def calculate_max_drawdown(values: pd.Series) -> tuple:
    """Calculate maximum drawdown and duration"""
    if len(values) < 2:
        return 0.0, 0
    
    cummax = values.expanding().max()
    drawdown = (values - cummax) / cummax
    
    max_dd = drawdown.min()
    
    # Calculate duration
    in_drawdown = drawdown < 0
    drawdown_periods = in_drawdown.astype(int).groupby(in_drawdown.ne(in_drawdown.shift()).cumsum())
    
    max_duration = 0
    for _, period in drawdown_periods:
        if period.iloc[0] == 1:  # In drawdown
            duration = len(period)
            max_duration = max(max_duration, duration)
    
    return abs(max_dd), max_duration

# API Endpoints
@router.get("/snapshot")
async def get_portfolio_snapshot(
    current_user: User = Depends(get_current_user)
) -> PortfolioSnapshot:
    """Get current portfolio snapshot"""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    try:
        # Get latest portfolio state
        cursor.execute("""
            SELECT timestamp, cash_balance, total_value, daily_pnl, max_drawdown
            FROM portfolio_state
            ORDER BY timestamp DESC
            LIMIT 1
        """)
        
        row = cursor.fetchone()
        if not row:
            # Return default values if no data
            return PortfolioSnapshot(
                timestamp=datetime.now(),
                cash_balance=10000,
                total_value=10000,
                positions_value=0,
                daily_pnl=0,
                daily_return=0,
                total_return=0,
                max_drawdown=0,
                positions_count=0
            )
        
        timestamp, cash_balance, total_value, daily_pnl, max_drawdown = row
        
        # Get initial value for total return calculation
        cursor.execute("""
            SELECT total_value FROM portfolio_state
            ORDER BY timestamp ASC
            LIMIT 1
        """)
        initial_value = cursor.fetchone()[0] if cursor.fetchone() else 10000
        
        # Count open positions
        cursor.execute("SELECT COUNT(*) FROM positions WHERE status = 'open'")
        positions_count = cursor.fetchone()[0]
        
        positions_value = total_value - cash_balance
        daily_return = daily_pnl / (total_value - daily_pnl) if total_value > daily_pnl else 0
        total_return = (total_value - initial_value) / initial_value if initial_value > 0 else 0
        
        conn.close()
        
        return PortfolioSnapshot(
            timestamp=datetime.fromtimestamp(timestamp / 1000),
            cash_balance=cash_balance,
            total_value=total_value,
            positions_value=positions_value,
            daily_pnl=daily_pnl,
            daily_return=daily_return,
            total_return=total_return,
            max_drawdown=max_drawdown,
            positions_count=positions_count
        )
        
    except Exception as e:
        conn.close()
        logger.error(f"Failed to get portfolio snapshot: {e}")
        raise HTTPException(status_code=500, detail="Failed to get portfolio snapshot")

@router.get("/history")
async def get_portfolio_history(
    hours: int = Query(24, ge=1, le=720),
    current_user: User = Depends(get_current_user)
) -> Dict:
    """Get portfolio value history"""
    conn = get_db_connection()
    
    try:
        cutoff = datetime.now().timestamp() * 1000 - (hours * 3600 * 1000)
        
        query = """
            SELECT timestamp, total_value, cash_balance, daily_pnl
            FROM portfolio_state
            WHERE timestamp > ?
            ORDER BY timestamp ASC
        """
        
        df = pd.read_sql_query(query, conn, params=[cutoff])
        conn.close()
        
        if df.empty:
            return {
                "timestamps": [],
                "portfolio_values": [],
                "cash_balances": [],
                "positions_values": [],
                "daily_pnl": []
            }
        
        # Calculate positions value
        df['positions_value'] = df['total_value'] - df['cash_balance']
        
        return {
            "timestamps": df['timestamp'].tolist(),
            "portfolio_values": df['total_value'].tolist(),
            "cash_balances": df['cash_balance'].tolist(),
            "positions_values": df['positions_value'].tolist(),
            "daily_pnl": df['daily_pnl'].tolist()
        }
        
    except Exception as e:
        conn.close()
        logger.error(f"Failed to get portfolio history: {e}")
        raise HTTPException(status_code=500, detail="Failed to get portfolio history")

@router.get("/performance")
async def get_performance_metrics(
    days: int = Query(30, ge=1, le=365),
    current_user: User = Depends(get_current_user)
) -> PerformanceMetrics:
    """Get detailed performance metrics"""
    conn = get_db_connection()
    
    try:
        cutoff = datetime.now().timestamp() * 1000 - (days * 86400 * 1000)
        
        # Get portfolio history
        portfolio_query = """
            SELECT timestamp, total_value, daily_pnl
            FROM portfolio_state
            WHERE timestamp > ?
            ORDER BY timestamp ASC
        """
        portfolio_df = pd.read_sql_query(portfolio_query, conn, params=[cutoff])
        
        if portfolio_df.empty or len(portfolio_df) < 2:
            return PerformanceMetrics(
                total_return=0, total_return_percent=0, sharpe_ratio=0,
                sortino_ratio=0, max_drawdown=0, max_drawdown_duration_days=0,
                win_rate=0, profit_factor=0, avg_win=0, avg_loss=0,
                best_trade=0, worst_trade=0, total_trades=0,
                winning_trades=0, losing_trades=0, avg_trade_duration_hours=0,
                risk_adjusted_return=0
            )
        
        # Calculate returns
        portfolio_df['returns'] = portfolio_df['total_value'].pct_change()
        returns = portfolio_df['returns'].dropna()
        
        # Calculate metrics
        initial_value = portfolio_df['total_value'].iloc[0]
        final_value = portfolio_df['total_value'].iloc[-1]
        total_return = final_value - initial_value
        total_return_percent = (total_return / initial_value) * 100
        
        sharpe = calculate_sharpe_ratio(returns)
        sortino = calculate_sortino_ratio(returns)
        max_dd, dd_duration = calculate_max_drawdown(portfolio_df['total_value'])
        
        # Get trade statistics
        trades_query = """
            SELECT pnl, (exit_time - entry_time) as duration
            FROM paper_trades
            WHERE timestamp > ? AND pnl IS NOT NULL
        """
        trades_df = pd.read_sql_query(trades_query, conn, params=[cutoff])
        
        if not trades_df.empty:
            winning_trades = trades_df[trades_df['pnl'] > 0]
            losing_trades = trades_df[trades_df['pnl'] < 0]
            
            win_rate = len(winning_trades) / len(trades_df) if len(trades_df) > 0 else 0
            
            total_wins = winning_trades['pnl'].sum() if len(winning_trades) > 0 else 0
            total_losses = abs(losing_trades['pnl'].sum()) if len(losing_trades) > 0 else 0
            profit_factor = total_wins / total_losses if total_losses > 0 else float('inf')
            
            avg_win = winning_trades['pnl'].mean() if len(winning_trades) > 0 else 0
            avg_loss = losing_trades['pnl'].mean() if len(losing_trades) > 0 else 0
            
            best_trade = trades_df['pnl'].max()
            worst_trade = trades_df['pnl'].min()
            
            avg_duration = trades_df['duration'].mean() / 3600000 if len(trades_df) > 0 else 0  # Convert to hours
        else:
            win_rate = profit_factor = avg_win = avg_loss = 0
            best_trade = worst_trade = avg_duration = 0
            winning_trades = losing_trades = pd.DataFrame()
        
        # Risk-adjusted return
        volatility = returns.std() * np.sqrt(252)
        risk_adjusted_return = (total_return_percent / 100) / volatility if volatility > 0 else 0
        
        conn.close()
        
        return PerformanceMetrics(
            total_return=total_return,
            total_return_percent=total_return_percent,
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            max_drawdown=max_dd,
            max_drawdown_duration_days=dd_duration,
            win_rate=win_rate,
            profit_factor=profit_factor,
            avg_win=avg_win,
            avg_loss=avg_loss,
            best_trade=best_trade,
            worst_trade=worst_trade,
            total_trades=len(trades_df),
            winning_trades=len(winning_trades),
            losing_trades=len(losing_trades),
            avg_trade_duration_hours=avg_duration,
            risk_adjusted_return=risk_adjusted_return
        )
        
    except Exception as e:
        conn.close()
        logger.error(f"Failed to calculate performance metrics: {e}")
        raise HTTPException(status_code=500, detail="Failed to calculate performance metrics")

@router.get("/positions/history")
async def get_position_history(
    days: int = Query(30, ge=1, le=365),
    symbol: Optional[str] = None,
    current_user: User = Depends(get_current_user)
) -> List[PositionHistory]:
    """Get historical positions"""
    conn = get_db_connection()
    
    try:
        cutoff = datetime.now().timestamp() * 1000 - (days * 86400 * 1000)
        
        query = """
            SELECT symbol, entry_time, exit_time, entry_price, exit_price,
                   quantity, side, pnl, max_profit, max_loss
            FROM positions
            WHERE entry_time > ?
        """
        params = [cutoff]
        
        if symbol:
            query += " AND symbol = ?"
            params.append(symbol)
        
        query += " ORDER BY entry_time DESC"
        
        cursor = conn.cursor()
        cursor.execute(query, params)
        
        positions = []
        for row in cursor.fetchall():
            entry_time = datetime.fromtimestamp(row[1] / 1000)
            exit_time = datetime.fromtimestamp(row[2] / 1000) if row[2] else None
            
            duration_hours = None
            if exit_time:
                duration_hours = (row[2] - row[1]) / 3600000
            
            pnl_percent = None
            if row[7] is not None and row[3] > 0:
                pnl_percent = (row[7] / (row[5] * row[3])) * 100
            
            positions.append(PositionHistory(
                symbol=row[0],
                entry_time=entry_time,
                exit_time=exit_time,
                entry_price=row[3],
                exit_price=row[4],
                quantity=row[5],
                side=row[6],
                pnl=row[7],
                pnl_percent=pnl_percent,
                duration_hours=duration_hours,
                max_profit=row[8] if len(row) > 8 else 0,
                max_loss=row[9] if len(row) > 9 else 0
            ))
        
        conn.close()
        return positions
        
    except Exception as e:
        conn.close()
        logger.error(f"Failed to get position history: {e}")
        raise HTTPException(status_code=500, detail="Failed to get position history")

@router.get("/risk")
async def get_risk_metrics(
    current_user: User = Depends(get_current_user)
) -> RiskMetrics:
    """Get current risk metrics"""
    conn = get_db_connection()
    
    try:
        # Get recent returns for risk calculations
        query = """
            SELECT timestamp, total_value
            FROM portfolio_state
            WHERE timestamp > ?
            ORDER BY timestamp ASC
        """
        cutoff = datetime.now().timestamp() * 1000 - (30 * 86400 * 1000)  # 30 days
        
        df = pd.read_sql_query(query, conn, params=[cutoff])
        
        if len(df) < 2:
            return RiskMetrics(
                current_var=0, current_cvar=0, beta=0,
                correlation_matrix={}, position_concentration={},
                sector_exposure={}, volatility_30d=0,
                downside_deviation=0, risk_score=0
            )
        
        # Calculate returns
        df['returns'] = df['total_value'].pct_change().dropna()
        returns = df['returns'].dropna()
        
        # Calculate VaR and CVaR (95% confidence)
        var_95 = np.percentile(returns, 5)
        cvar_95 = returns[returns <= var_95].mean()
        
        # Calculate volatility
        volatility_30d = returns.std() * np.sqrt(252)
        
        # Downside deviation
        negative_returns = returns[returns < 0]
        downside_deviation = negative_returns.std() * np.sqrt(252) if len(negative_returns) > 0 else 0
        
        # Mock correlation matrix (in production, calculate actual correlations)
        correlation_matrix = {
            "BTCUSD": {"BTCUSD": 1.0, "ETHUSD": 0.85},
            "ETHUSD": {"BTCUSD": 0.85, "ETHUSD": 1.0}
        }
        
        # Mock position concentration
        position_concentration = {
            "BTCUSD": 0.6,
            "ETHUSD": 0.4
        }
        
        # Mock sector exposure
        sector_exposure = {
            "Large Cap": 0.6,
            "DeFi": 0.4
        }
        
        # Calculate risk score (0-100, higher is riskier)
        risk_factors = [
            volatility_30d * 100,  # Volatility component
            abs(var_95) * 1000,    # VaR component
            (1 - min(position_concentration.values())) * 50,  # Concentration risk
            abs(df['total_value'].iloc[-1] - 10000) / 100  # Drawdown component
        ]
        risk_score = min(100, sum(risk_factors) / len(risk_factors))
        
        conn.close()
        
        return RiskMetrics(
            current_var=var_95,
            current_cvar=cvar_95,
            beta=0.9,  # Mock beta
            correlation_matrix=correlation_matrix,
            position_concentration=position_concentration,
            sector_exposure=sector_exposure,
            volatility_30d=volatility_30d,
            downside_deviation=downside_deviation,
            risk_score=risk_score
        )
        
    except Exception as e:
        conn.close()
        logger.error(f"Failed to calculate risk metrics: {e}")
        raise HTTPException(status_code=500, detail="Failed to calculate risk metrics")

@router.get("/daily-summary")
async def get_daily_summary(
    date: Optional[str] = None,
    current_user: User = Depends(get_current_user)
) -> Dict:
    """Get daily trading summary"""
    conn = get_db_connection()
    
    try:
        # Parse date or use today
        if date:
            target_date = datetime.strptime(date, "%Y-%m-%d")
        else:
            target_date = datetime.now()
        
        # Get start and end of day timestamps
        start_of_day = target_date.replace(hour=0, minute=0, second=0, microsecond=0)
        end_of_day = target_date.replace(hour=23, minute=59, second=59, microsecond=999999)
        
        start_ts = start_of_day.timestamp() * 1000
        end_ts = end_of_day.timestamp() * 1000
        
        # Get portfolio values at start and end of day
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT total_value FROM portfolio_state
            WHERE timestamp >= ? AND timestamp <= ?
            ORDER BY timestamp ASC LIMIT 1
        """, (start_ts, end_ts))
        start_value = cursor.fetchone()
        start_value = start_value[0] if start_value else None
        
        cursor.execute("""
            SELECT total_value, cash_balance FROM portfolio_state
            WHERE timestamp >= ? AND timestamp <= ?
            ORDER BY timestamp DESC LIMIT 1
        """, (start_ts, end_ts))
        end_data = cursor.fetchone()
        end_value = end_data[0] if end_data else None
        end_cash = end_data[1] if end_data else None
        
        # Get trades for the day
        cursor.execute("""
            SELECT COUNT(*), SUM(pnl), AVG(pnl),
                   SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END) as wins,
                   SUM(CASE WHEN pnl < 0 THEN 1 ELSE 0 END) as losses
            FROM paper_trades
            WHERE timestamp >= ? AND timestamp <= ?
        """, (start_ts, end_ts))
        
        trade_stats = cursor.fetchone()
        
        conn.close()
        
        # Calculate daily metrics
        daily_pnl = (end_value - start_value) if start_value and end_value else 0
        daily_return = (daily_pnl / start_value * 100) if start_value else 0
        
        return {
            "date": target_date.strftime("%Y-%m-%d"),
            "starting_value": start_value,
            "ending_value": end_value,
            "cash_balance": end_cash,
            "daily_pnl": daily_pnl,
            "daily_return_percent": daily_return,
            "trades": {
                "total": trade_stats[0] or 0,
                "winning": trade_stats[3] or 0,
                "losing": trade_stats[4] or 0,
                "total_pnl": trade_stats[1] or 0,
                "avg_pnl": trade_stats[2] or 0,
                "win_rate": (trade_stats[3] / trade_stats[0] * 100) if trade_stats[0] else 0
            }
        }
        
    except Exception as e:
        conn.close()
        logger.error(f"Failed to get daily summary: {e}")
        raise HTTPException(status_code=500, detail="Failed to get daily summary")