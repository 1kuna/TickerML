#!/usr/bin/env python3
"""
Paper Trading Engine for TickerML.
Production-grade portfolio management with realistic execution simulation.
Implements virtual $10,000 starting balance with comprehensive risk management.
"""

import json
import logging
import signal
import sqlite3
import time
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum

import numpy as np
import pandas as pd
import yaml

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/paper_trader.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class OrderType(Enum):
    """Order types for paper trading."""
    MARKET = "market"
    LIMIT = "limit"
    STOP_LOSS = "stop_loss"
    TAKE_PROFIT = "take_profit"

class OrderSide(Enum):
    """Order sides."""
    BUY = "buy"
    SELL = "sell"

class OrderStatus(Enum):
    """Order status."""
    PENDING = "pending"
    FILLED = "filled"
    PARTIALLY_FILLED = "partially_filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"

class PositionSide(Enum):
    """Position sides."""
    LONG = "long"
    SHORT = "short"
    FLAT = "flat"

@dataclass
class Order:
    """Represents a trading order."""
    order_id: str
    timestamp: float
    symbol: str
    side: OrderSide
    order_type: OrderType
    quantity: float
    price: Optional[float] = None
    stop_price: Optional[float] = None
    status: OrderStatus = OrderStatus.PENDING
    filled_quantity: float = 0.0
    avg_fill_price: float = 0.0
    commission: float = 0.0
    
@dataclass
class Position:
    """Represents a trading position."""
    symbol: str
    side: PositionSide
    quantity: float
    entry_price: float
    current_price: float
    unrealized_pnl: float
    realized_pnl: float
    timestamp: float
    
@dataclass
class Portfolio:
    """Represents portfolio state."""
    cash_balance: float
    total_value: float
    positions: Dict[str, Position]
    daily_pnl: float
    total_pnl: float
    max_drawdown: float
    win_rate: float
    sharpe_ratio: float
    total_trades: int
    winning_trades: int
    timestamp: float

class PaperTradingEngine:
    """Advanced paper trading engine with realistic execution simulation."""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """Initialize the paper trading engine."""
        self.config = self._load_config(config_path)
        self.db_path = self.config.get('database', {}).get('path', 'data/db/crypto_data.db')
        
        # Initial portfolio settings
        self.starting_balance = 10000.0  # $10,000 starting balance
        self.cash_balance = self.starting_balance
        self.positions: Dict[str, Position] = {}
        self.orders: Dict[str, Order] = {}
        self.trade_history: List[Dict] = []
        
        # Risk management parameters
        self.max_position_pct = 0.25  # Max 25% of portfolio per position
        self.max_drawdown_pct = 0.25  # Max 25% portfolio loss
        self.stop_loss_pct = 0.05     # 5% stop loss
        self.take_profit_pct = 0.10   # 10% take profit
        
        # Trading parameters
        self.commission_rate = 0.001  # 0.1% commission
        self.slippage_factor = 0.0005  # 0.05% slippage
        self.min_order_size = 10.0    # Minimum $10 order
        
        # Performance tracking
        self.portfolio_history: List[Portfolio] = []
        self.high_water_mark = self.starting_balance
        self.max_drawdown = 0.0
        self.total_trades = 0
        self.winning_trades = 0
        
        # Threading
        self.running = False
        self.engine_thread = None
        self.lock = threading.RLock()
        
        # Initialize database
        self._init_database()
        
        # Load existing state if available
        self._load_portfolio_state()
        
        # Set up signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from YAML file."""
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            logger.warning(f"Config file {config_path} not found, using defaults")
            return {
                'binance': {
                    'symbols': ['BTCUSDT', 'ETHUSDT']
                },
                'database': {
                    'path': 'data/db/crypto_data.db'
                }
            }
    
    def _init_database(self):
        """Initialize database tables for paper trading."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Portfolio state table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS portfolio_state (
                    timestamp REAL PRIMARY KEY,
                    cash_balance REAL NOT NULL,
                    total_value REAL NOT NULL,
                    positions TEXT NOT NULL,
                    daily_pnl REAL NOT NULL,
                    total_pnl REAL NOT NULL,
                    max_drawdown REAL NOT NULL,
                    win_rate REAL NOT NULL,
                    sharpe_ratio REAL NOT NULL,
                    total_trades INTEGER NOT NULL,
                    winning_trades INTEGER NOT NULL
                )
            ''')
            
            # Orders table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS paper_orders (
                    order_id TEXT PRIMARY KEY,
                    timestamp REAL NOT NULL,
                    symbol TEXT NOT NULL,
                    side TEXT NOT NULL,
                    order_type TEXT NOT NULL,
                    quantity REAL NOT NULL,
                    price REAL,
                    stop_price REAL,
                    status TEXT NOT NULL,
                    filled_quantity REAL NOT NULL,
                    avg_fill_price REAL NOT NULL,
                    commission REAL NOT NULL
                )
            ''')
            
            # Trades table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS paper_trades (
                    trade_id TEXT PRIMARY KEY,
                    timestamp REAL NOT NULL,
                    symbol TEXT NOT NULL,
                    side TEXT NOT NULL,
                    quantity REAL NOT NULL,
                    price REAL NOT NULL,
                    commission REAL NOT NULL,
                    pnl REAL NOT NULL,
                    portfolio_value REAL NOT NULL
                )
            ''')
            
            # Performance metrics table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS performance_metrics (
                    timestamp REAL PRIMARY KEY,
                    portfolio_value REAL NOT NULL,
                    daily_return REAL NOT NULL,
                    sharpe_ratio REAL NOT NULL,
                    max_drawdown REAL NOT NULL,
                    win_rate REAL NOT NULL,
                    profit_factor REAL NOT NULL,
                    total_trades INTEGER NOT NULL
                )
            ''')
            
            conn.commit()
            conn.close()
            logger.info("Paper trading database initialized successfully")
            
        except Exception as e:
            logger.error(f"Database initialization failed: {e}")
            raise
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully."""
        logger.info(f"Received signal {signum}, shutting down paper trader...")
        self.stop_trading()
    
    def _load_portfolio_state(self):
        """Load existing portfolio state from database."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Get latest portfolio state
            cursor.execute('''
                SELECT * FROM portfolio_state 
                ORDER BY timestamp DESC LIMIT 1
            ''')
            
            result = cursor.fetchone()
            if result:
                columns = [desc[0] for desc in cursor.description]
                portfolio_data = dict(zip(columns, result))
                
                self.cash_balance = portfolio_data['cash_balance']
                self.total_trades = portfolio_data['total_trades']
                self.winning_trades = portfolio_data['winning_trades']
                self.max_drawdown = portfolio_data['max_drawdown']
                
                # Load positions
                positions_json = portfolio_data['positions']
                if positions_json:
                    positions_data = json.loads(positions_json)
                    for symbol, pos_data in positions_data.items():
                        # Convert string back to enum
                        pos_data['side'] = PositionSide(pos_data['side'])
                        self.positions[symbol] = Position(**pos_data)
                
                logger.info(f"Loaded portfolio state: ${self.cash_balance:.2f} cash, "
                           f"{len(self.positions)} positions")
            
            conn.close()
            
        except Exception as e:
            logger.error(f"Failed to load portfolio state: {e}")
    
    def _save_portfolio_state(self):
        """Save current portfolio state to database."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Calculate current metrics
            total_value = self._calculate_portfolio_value()
            daily_pnl = self._calculate_daily_pnl()
            total_pnl = total_value - self.starting_balance
            win_rate = self.winning_trades / max(self.total_trades, 1)
            sharpe_ratio = self._calculate_sharpe_ratio()
            
            # Serialize positions (handle enum serialization)
            positions_data = {}
            for symbol, position in self.positions.items():
                pos_dict = asdict(position)
                pos_dict['side'] = position.side.value  # Convert enum to string
                positions_data[symbol] = pos_dict
            
            positions_json = json.dumps(positions_data)
            
            # Insert portfolio state
            cursor.execute('''
                INSERT OR REPLACE INTO portfolio_state 
                (timestamp, cash_balance, total_value, positions, daily_pnl, 
                 total_pnl, max_drawdown, win_rate, sharpe_ratio, total_trades, winning_trades)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                time.time(), self.cash_balance, total_value, positions_json,
                daily_pnl, total_pnl, self.max_drawdown, win_rate, sharpe_ratio,
                self.total_trades, self.winning_trades
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Failed to save portfolio state: {e}")
    
    def _get_current_price(self, symbol: str) -> Optional[float]:
        """Get current market price for a symbol."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Get latest price from OHLCV data
            cursor.execute('''
                SELECT close FROM ohlcv 
                WHERE symbol = ? 
                ORDER BY timestamp DESC LIMIT 1
            ''', (symbol,))
            
            result = cursor.fetchone()
            conn.close()
            
            if result:
                return float(result[0])
            else:
                logger.warning(f"No price data found for {symbol}")
                return None
                
        except Exception as e:
            logger.error(f"Failed to get current price for {symbol}: {e}")
            return None
    
    def _calculate_portfolio_value(self) -> float:
        """Calculate total portfolio value including positions."""
        total_value = self.cash_balance
        
        for symbol, position in self.positions.items():
            current_price = self._get_current_price(symbol)
            if current_price:
                position_value = position.quantity * current_price
                if position.side == PositionSide.SHORT:
                    position_value = -position_value
                total_value += position_value
        
        return total_value
    
    def _calculate_daily_pnl(self) -> float:
        """Calculate daily P&L."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Get portfolio value from 24 hours ago
            yesterday = time.time() - 86400
            cursor.execute('''
                SELECT total_value FROM portfolio_state 
                WHERE timestamp >= ? 
                ORDER BY timestamp ASC LIMIT 1
            ''', (yesterday,))
            
            result = cursor.fetchone()
            conn.close()
            
            if result:
                previous_value = result[0]
                current_value = self._calculate_portfolio_value()
                return current_value - previous_value
            else:
                return 0.0
                
        except Exception as e:
            logger.error(f"Failed to calculate daily P&L: {e}")
            return 0.0
    
    def _calculate_sharpe_ratio(self, periods: int = 252) -> float:
        """Calculate Sharpe ratio based on recent returns."""
        try:
            conn = sqlite3.connect(self.db_path)
            
            # Get recent portfolio values
            query = '''
                SELECT total_value FROM portfolio_state 
                ORDER BY timestamp DESC LIMIT ?
            '''
            
            df = pd.read_sql_query(query, conn, params=(periods + 1,))
            conn.close()
            
            if len(df) < 2:
                return 0.0
            
            # Calculate daily returns
            returns = df['total_value'].pct_change().dropna()
            
            if len(returns) == 0 or returns.std() == 0:
                return 0.0
            
            # Annualized Sharpe ratio (assuming daily data)
            return (returns.mean() * 252) / (returns.std() * np.sqrt(252))
            
        except Exception as e:
            logger.error(f"Failed to calculate Sharpe ratio: {e}")
            return 0.0
    
    def _calculate_position_size(self, symbol: str, signal_strength: float = 1.0) -> float:
        """Calculate position size based on risk management rules."""
        portfolio_value = self._calculate_portfolio_value()
        max_position_value = portfolio_value * self.max_position_pct
        
        # Apply signal strength (0.0 to 1.0)
        position_value = max_position_value * signal_strength
        
        # Get current price
        current_price = self._get_current_price(symbol)
        if not current_price:
            return 0.0
        
        # Calculate quantity
        quantity = position_value / current_price
        
        # Apply minimum order size
        if position_value < self.min_order_size:
            return 0.0
        
        return quantity
    
    def _simulate_market_impact(self, symbol: str, quantity: float, side: OrderSide) -> float:
        """Simulate market impact and slippage."""
        current_price = self._get_current_price(symbol)
        if not current_price:
            return 0.0
        
        # Basic slippage model
        slippage = self.slippage_factor * current_price
        
        if side == OrderSide.BUY:
            return current_price + slippage
        else:
            return current_price - slippage
    
    def place_order(self, symbol: str, side: OrderSide, quantity: float,
                   order_type: OrderType = OrderType.MARKET, 
                   price: Optional[float] = None,
                   stop_price: Optional[float] = None) -> str:
        """Place a paper trading order."""
        with self.lock:
            try:
                # Generate order ID
                order_id = f"{symbol}_{int(time.time() * 1000)}"
                
                # Validate order
                if quantity <= 0:
                    logger.error("Invalid quantity: must be positive")
                    return ""
                
                # Check available balance for buy orders
                if side == OrderSide.BUY:
                    required_cash = quantity * (price or self._get_current_price(symbol) or 0)
                    if required_cash > self.cash_balance:
                        logger.error(f"Insufficient cash: need ${required_cash:.2f}, have ${self.cash_balance:.2f}")
                        return ""
                
                # Create order
                order = Order(
                    order_id=order_id,
                    timestamp=time.time(),
                    symbol=symbol,
                    side=side,
                    order_type=order_type,
                    quantity=quantity,
                    price=price,
                    stop_price=stop_price
                )
                
                self.orders[order_id] = order
                
                # Execute market orders immediately
                if order_type == OrderType.MARKET:
                    self._execute_order(order_id)
                
                # Store order in database
                self._store_order(order)
                
                logger.info(f"Placed {order_type.value} {side.value} order: {quantity} {symbol}")
                return order_id
                
            except Exception as e:
                logger.error(f"Failed to place order: {e}")
                return ""
    
    def _execute_order(self, order_id: str) -> bool:
        """Execute a paper trading order."""
        try:
            order = self.orders.get(order_id)
            if not order or order.status != OrderStatus.PENDING:
                return False
            
            # Get execution price
            if order.order_type == OrderType.MARKET:
                execution_price = self._simulate_market_impact(order.symbol, order.quantity, order.side)
            else:
                execution_price = order.price
            
            if not execution_price:
                order.status = OrderStatus.REJECTED
                return False
            
            # Calculate commission
            commission = order.quantity * execution_price * self.commission_rate
            
            # Execute the trade
            if order.side == OrderSide.BUY:
                self._execute_buy(order, execution_price, commission)
            else:
                self._execute_sell(order, execution_price, commission)
            
            # Update order status
            order.status = OrderStatus.FILLED
            order.filled_quantity = order.quantity
            order.avg_fill_price = execution_price
            order.commission = commission
            
            # Store trade
            self._store_trade(order, execution_price, commission)
            
            # Update portfolio state
            self._save_portfolio_state()
            
            logger.info(f"Executed order {order_id}: {order.quantity} {order.symbol} @ ${execution_price:.2f}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to execute order {order_id}: {e}")
            return False
    
    def _execute_buy(self, order: Order, execution_price: float, commission: float):
        """Execute a buy order."""
        total_cost = order.quantity * execution_price + commission
        
        # Update cash balance
        self.cash_balance -= total_cost
        
        # Update or create position
        if order.symbol in self.positions:
            position = self.positions[order.symbol]
            if position.side == PositionSide.LONG:
                # Add to existing long position
                total_quantity = position.quantity + order.quantity
                position.entry_price = ((position.entry_price * position.quantity) + 
                                      (execution_price * order.quantity)) / total_quantity
                position.quantity = total_quantity
            elif position.side == PositionSide.SHORT:
                # Reduce short position
                if order.quantity >= position.quantity:
                    # Close short and potentially open long
                    remaining_quantity = order.quantity - position.quantity
                    realized_pnl = position.quantity * (position.entry_price - execution_price)
                    position.realized_pnl += realized_pnl
                    
                    if remaining_quantity > 0:
                        position.side = PositionSide.LONG
                        position.quantity = remaining_quantity
                        position.entry_price = execution_price
                    else:
                        position.side = PositionSide.FLAT
                        position.quantity = 0
                else:
                    # Partially close short position
                    position.quantity -= order.quantity
                    realized_pnl = order.quantity * (position.entry_price - execution_price)
                    position.realized_pnl += realized_pnl
        else:
            # Create new long position
            self.positions[order.symbol] = Position(
                symbol=order.symbol,
                side=PositionSide.LONG,
                quantity=order.quantity,
                entry_price=execution_price,
                current_price=execution_price,
                unrealized_pnl=0.0,
                realized_pnl=0.0,
                timestamp=time.time()
            )
    
    def _execute_sell(self, order: Order, execution_price: float, commission: float):
        """Execute a sell order."""
        total_proceeds = order.quantity * execution_price - commission
        
        # Update cash balance
        self.cash_balance += total_proceeds
        
        # Update or create position
        if order.symbol in self.positions:
            position = self.positions[order.symbol]
            if position.side == PositionSide.LONG:
                # Reduce long position
                if order.quantity >= position.quantity:
                    # Close long and potentially open short
                    remaining_quantity = order.quantity - position.quantity
                    realized_pnl = position.quantity * (execution_price - position.entry_price)
                    position.realized_pnl += realized_pnl
                    
                    if remaining_quantity > 0:
                        position.side = PositionSide.SHORT
                        position.quantity = remaining_quantity
                        position.entry_price = execution_price
                    else:
                        position.side = PositionSide.FLAT
                        position.quantity = 0
                else:
                    # Partially close long position
                    position.quantity -= order.quantity
                    realized_pnl = order.quantity * (execution_price - position.entry_price)
                    position.realized_pnl += realized_pnl
            elif position.side == PositionSide.SHORT:
                # Add to existing short position
                total_quantity = position.quantity + order.quantity
                position.entry_price = ((position.entry_price * position.quantity) + 
                                      (execution_price * order.quantity)) / total_quantity
                position.quantity = total_quantity
        else:
            # Create new short position
            self.positions[order.symbol] = Position(
                symbol=order.symbol,
                side=PositionSide.SHORT,
                quantity=order.quantity,
                entry_price=execution_price,
                current_price=execution_price,
                unrealized_pnl=0.0,
                realized_pnl=0.0,
                timestamp=time.time()
            )
        
        # Track trade statistics
        self.total_trades += 1
        if total_proceeds > 0:  # Simplified win condition
            self.winning_trades += 1
    
    def _store_order(self, order: Order):
        """Store order in database."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT OR REPLACE INTO paper_orders 
                (order_id, timestamp, symbol, side, order_type, quantity, price, 
                 stop_price, status, filled_quantity, avg_fill_price, commission)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                order.order_id, order.timestamp, order.symbol, order.side.value,
                order.order_type.value, order.quantity, order.price, order.stop_price,
                order.status.value, order.filled_quantity, order.avg_fill_price, order.commission
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Failed to store order: {e}")
    
    def _store_trade(self, order: Order, execution_price: float, commission: float):
        """Store executed trade in database."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Calculate P&L for this trade
            if order.side == OrderSide.SELL and order.symbol in self.positions:
                position = self.positions[order.symbol]
                if position.side == PositionSide.LONG:
                    pnl = order.quantity * (execution_price - position.entry_price)
                else:
                    pnl = 0.0  # Simplification for now
            else:
                pnl = 0.0
            
            trade_id = f"{order.order_id}_trade"
            cursor.execute('''
                INSERT INTO paper_trades 
                (trade_id, timestamp, symbol, side, quantity, price, commission, pnl, portfolio_value)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                trade_id, order.timestamp, order.symbol, order.side.value,
                order.quantity, execution_price, commission, pnl, self._calculate_portfolio_value()
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Failed to store trade: {e}")
    
    def update_positions(self):
        """Update position values with current market prices."""
        with self.lock:
            for symbol, position in self.positions.items():
                current_price = self._get_current_price(symbol)
                if current_price:
                    position.current_price = current_price
                    
                    # Calculate unrealized P&L
                    if position.side == PositionSide.LONG:
                        position.unrealized_pnl = position.quantity * (current_price - position.entry_price)
                    elif position.side == PositionSide.SHORT:
                        position.unrealized_pnl = position.quantity * (position.entry_price - current_price)
    
    def get_portfolio_summary(self) -> Dict:
        """Get current portfolio summary."""
        self.update_positions()
        total_value = self._calculate_portfolio_value()
        
        return {
            'timestamp': time.time(),
            'cash_balance': self.cash_balance,
            'total_value': total_value,
            'total_pnl': total_value - self.starting_balance,
            'total_pnl_pct': ((total_value - self.starting_balance) / self.starting_balance) * 100,
            'daily_pnl': self._calculate_daily_pnl(),
            'positions': {
                symbol: asdict(position) for symbol, position in self.positions.items()
                if position.side != PositionSide.FLAT
            },
            'position_count': len([p for p in self.positions.values() if p.side != PositionSide.FLAT]),
            'max_drawdown': self.max_drawdown,
            'win_rate': self.winning_trades / max(self.total_trades, 1),
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'sharpe_ratio': self._calculate_sharpe_ratio()
        }
    
    def start_trading(self):
        """Start the paper trading engine."""
        if self.running:
            logger.warning("Paper trading engine already running")
            return
        
        self.running = True
        self.engine_thread = threading.Thread(target=self._trading_loop)
        self.engine_thread.daemon = True
        self.engine_thread.start()
        
        logger.info("Paper trading engine started")
    
    def stop_trading(self):
        """Stop the paper trading engine."""
        self.running = False
        
        if self.engine_thread:
            self.engine_thread.join(timeout=5.0)
        
        # Save final state
        self._save_portfolio_state()
        
        logger.info("Paper trading engine stopped")
    
    def _trading_loop(self):
        """Main trading loop."""
        while self.running:
            try:
                # Update positions with current prices
                self.update_positions()
                
                # Check for stop losses and take profits
                self._check_risk_management()
                
                # Save portfolio state periodically
                self._save_portfolio_state()
                
                # Sleep for a bit
                time.sleep(60)  # Update every minute
                
            except Exception as e:
                logger.error(f"Error in trading loop: {e}")
                time.sleep(60)
    
    def _check_risk_management(self):
        """Check and enforce risk management rules."""
        with self.lock:
            for symbol, position in self.positions.items():
                if position.side == PositionSide.FLAT:
                    continue
                
                current_price = position.current_price
                entry_price = position.entry_price
                
                # Check stop loss
                if position.side == PositionSide.LONG:
                    pnl_pct = (current_price - entry_price) / entry_price
                    if pnl_pct <= -self.stop_loss_pct:
                        logger.warning(f"Stop loss triggered for {symbol}: {pnl_pct:.2%}")
                        self.place_order(symbol, OrderSide.SELL, position.quantity, OrderType.MARKET)
                    elif pnl_pct >= self.take_profit_pct:
                        logger.info(f"Take profit triggered for {symbol}: {pnl_pct:.2%}")
                        self.place_order(symbol, OrderSide.SELL, position.quantity, OrderType.MARKET)
                
                elif position.side == PositionSide.SHORT:
                    pnl_pct = (entry_price - current_price) / entry_price
                    if pnl_pct <= -self.stop_loss_pct:
                        logger.warning(f"Stop loss triggered for {symbol}: {pnl_pct:.2%}")
                        self.place_order(symbol, OrderSide.BUY, position.quantity, OrderType.MARKET)
                    elif pnl_pct >= self.take_profit_pct:
                        logger.info(f"Take profit triggered for {symbol}: {pnl_pct:.2%}")
                        self.place_order(symbol, OrderSide.BUY, position.quantity, OrderType.MARKET)
            
            # Check overall portfolio drawdown
            total_value = self._calculate_portfolio_value()
            if total_value > self.high_water_mark:
                self.high_water_mark = total_value
            
            current_drawdown = (self.high_water_mark - total_value) / self.high_water_mark
            if current_drawdown > self.max_drawdown:
                self.max_drawdown = current_drawdown
            
            # Emergency stop if drawdown exceeds limit
            if current_drawdown > self.max_drawdown_pct:
                logger.critical(f"Maximum drawdown exceeded: {current_drawdown:.2%}")
                # Close all positions
                for symbol, position in self.positions.items():
                    if position.side == PositionSide.LONG:
                        self.place_order(symbol, OrderSide.SELL, position.quantity, OrderType.MARKET)
                    elif position.side == PositionSide.SHORT:
                        self.place_order(symbol, OrderSide.BUY, position.quantity, OrderType.MARKET)

def main():
    """Main function for testing the paper trading engine."""
    engine = PaperTradingEngine()
    
    try:
        # Start the trading engine
        engine.start_trading()
        
        # Example: Place a test order
        engine.place_order('BTCUSDT', OrderSide.BUY, 0.001, OrderType.MARKET)
        
        # Print portfolio summary
        summary = engine.get_portfolio_summary()
        print(f"Portfolio Summary: {json.dumps(summary, indent=2)}")
        
        # Run for a while (in practice, this would run continuously)
        time.sleep(10)
        
    except KeyboardInterrupt:
        logger.info("Shutting down paper trader...")
    finally:
        engine.stop_trading()

if __name__ == "__main__":
    # Create logs directory if it doesn't exist
    import os
    os.makedirs('logs', exist_ok=True)
    
    main()