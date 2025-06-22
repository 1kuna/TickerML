#!/usr/bin/env python3
"""
Cross-Exchange Arbitrage Monitor
Detects and tracks arbitrage opportunities across multiple exchanges
Implements real-time price difference monitoring with fee calculation
"""

import asyncio
import logging
import time
from typing import Dict, List, Optional, Tuple, NamedTuple, Set
from dataclasses import dataclass, field
from collections import defaultdict, deque
import json
import sqlite3
from datetime import datetime, timedelta

from .exchanges import ExchangeInterface, ExchangeConfig, create_exchange, OrderBook, Trade

logger = logging.getLogger(__name__)

@dataclass
class ArbitrageOpportunity:
    """Represents a potential arbitrage opportunity"""
    symbol: str
    buy_exchange: str
    sell_exchange: str
    buy_price: float
    sell_price: float
    spread_pct: float
    net_profit_pct: float  # After fees
    max_quantity: float
    estimated_profit_usd: float
    timestamp: float
    buy_fees: Dict[str, float]
    sell_fees: Dict[str, float]
    
    @property
    def is_profitable(self) -> bool:
        """Check if opportunity is profitable after fees"""
        return self.net_profit_pct > 0.001  # At least 0.1% profit

@dataclass  
class ExchangeState:
    """Tracks state of an exchange"""
    name: str
    exchange: ExchangeInterface
    connected: bool = False
    last_update: float = 0
    orderbooks: Dict[str, OrderBook] = field(default_factory=dict)
    fees: Dict[str, Dict[str, float]] = field(default_factory=dict)
    balances: Dict[str, float] = field(default_factory=dict)
    latency_ms: float = 100  # Estimated latency
    
class ArbitrageMonitor:
    """
    Cross-exchange arbitrage monitor
    
    Features:
    - Real-time price monitoring across exchanges
    - Fee calculation with maker/taker rates
    - Latency-aware opportunity detection
    - Transfer time considerations
    - Risk-adjusted profit calculations
    """
    
    def __init__(self, config_file: str = "config/exchanges_config.yaml"):
        self.config_file = config_file
        self.exchanges: Dict[str, ExchangeState] = {}
        self.symbols: Set[str] = set()
        self.opportunities: deque = deque(maxlen=1000)  # Recent opportunities
        self.min_profit_pct = 0.005  # Minimum 0.5% profit threshold
        self.max_position_usd = 1000  # Maximum position size for arbitrage
        self.update_interval = 1.0  # Update frequency in seconds
        
        # Profitability tracking
        self.opportunity_history: Dict[str, List[ArbitrageOpportunity]] = defaultdict(list)
        self.execution_latency = {
            'binance': 75,    # ms
            'coinbase': 150,  # ms  
            'kraken': 200,    # ms
            'kucoin': 120     # ms
        }
        
        # Database for logging
        self.db_path = "data/db/arbitrage_opportunities.db"
        self._init_database()
        
    def _init_database(self) -> None:
        """Initialize SQLite database for opportunity logging"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS arbitrage_opportunities (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL NOT NULL,
                symbol TEXT NOT NULL,
                buy_exchange TEXT NOT NULL,
                sell_exchange TEXT NOT NULL,
                buy_price REAL NOT NULL,
                sell_price REAL NOT NULL,
                spread_pct REAL NOT NULL,
                net_profit_pct REAL NOT NULL,
                max_quantity REAL NOT NULL,
                estimated_profit_usd REAL NOT NULL,
                executed BOOLEAN DEFAULT FALSE,
                notes TEXT
            )
        ''')
        
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_timestamp ON arbitrage_opportunities(timestamp);
        ''')
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_symbol ON arbitrage_opportunities(symbol);
        ''')
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_profit ON arbitrage_opportunities(net_profit_pct);
        ''')
        
        conn.commit()
        conn.close()
        
    async def add_exchange(self, name: str, config: ExchangeConfig) -> None:
        """Add exchange to monitoring"""
        try:
            exchange = create_exchange(name, config)
            await exchange.connect()
            
            self.exchanges[name] = ExchangeState(
                name=name,
                exchange=exchange,
                connected=True,
                latency_ms=self.execution_latency.get(name, 150)
            )
            
            logger.info(f"Added exchange: {name}")
            
        except Exception as e:
            logger.error(f"Failed to add exchange {name}: {e}")
            raise
    
    async def remove_exchange(self, name: str) -> None:
        """Remove exchange from monitoring"""
        if name in self.exchanges:
            exchange_state = self.exchanges[name]
            await exchange_state.exchange.disconnect()
            del self.exchanges[name]
            logger.info(f"Removed exchange: {name}")
    
    def add_symbol(self, symbol: str) -> None:
        """Add symbol to monitor for arbitrage"""
        self.symbols.add(symbol)
        logger.info(f"Added symbol for monitoring: {symbol}")
    
    async def start_monitoring(self) -> None:
        """Start monitoring all exchanges for arbitrage opportunities"""
        if not self.exchanges:
            raise ValueError("No exchanges configured")
        
        if not self.symbols:
            raise ValueError("No symbols configured for monitoring")
        
        logger.info(f"Starting arbitrage monitoring for {len(self.symbols)} symbols across {len(self.exchanges)} exchanges")
        
        # Start orderbook subscriptions
        tasks = []
        for symbol in self.symbols:
            for exchange_name, exchange_state in self.exchanges.items():
                task = asyncio.create_task(
                    self._subscribe_to_symbol(exchange_state, symbol)
                )
                tasks.append(task)
        
        # Start monitoring loop
        tasks.append(asyncio.create_task(self._monitoring_loop()))
        
        # Wait for all tasks
        await asyncio.gather(*tasks, return_exceptions=True)
    
    async def _subscribe_to_symbol(self, exchange_state: ExchangeState, symbol: str) -> None:
        """Subscribe to orderbook updates for a symbol on an exchange"""
        try:
            # Load fees for this symbol
            fees = await exchange_state.exchange.get_fees(symbol)
            exchange_state.fees[symbol] = fees
            
            # Subscribe to orderbook
            await exchange_state.exchange.subscribe_orderbook(
                symbol, 
                lambda ob: self._on_orderbook_update(exchange_state.name, ob)
            )
            
            logger.info(f"Subscribed to {symbol} on {exchange_state.name}")
            
        except Exception as e:
            logger.error(f"Failed to subscribe to {symbol} on {exchange_state.name}: {e}")
    
    async def _on_orderbook_update(self, exchange_name: str, orderbook: OrderBook) -> None:
        """Handle orderbook update"""
        if exchange_name in self.exchanges:
            exchange_state = self.exchanges[exchange_name]
            exchange_state.orderbooks[orderbook.symbol] = orderbook
            exchange_state.last_update = time.time()
            
            # Trigger arbitrage check for this symbol
            await self._check_arbitrage_for_symbol(orderbook.symbol)
    
    async def _monitoring_loop(self) -> None:
        """Main monitoring loop"""
        while True:
            try:
                # Update exchange balances periodically
                await self._update_balances()
                
                # Clean old opportunities
                self._clean_old_opportunities()
                
                # Log statistics
                await self._log_statistics()
                
                await asyncio.sleep(self.update_interval)
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(5)
    
    async def _check_arbitrage_for_symbol(self, symbol: str) -> None:
        """Check for arbitrage opportunities for a specific symbol"""
        # Get all exchange orderbooks for this symbol
        orderbooks = {}
        for exchange_name, exchange_state in self.exchanges.items():
            if symbol in exchange_state.orderbooks:
                orderbooks[exchange_name] = exchange_state.orderbooks[symbol]
        
        if len(orderbooks) < 2:
            return  # Need at least 2 exchanges
        
        # Check all exchange pairs
        exchange_names = list(orderbooks.keys())
        for i in range(len(exchange_names)):
            for j in range(i + 1, len(exchange_names)):
                buy_exchange = exchange_names[i]
                sell_exchange = exchange_names[j]
                
                # Check both directions
                await self._check_arbitrage_pair(
                    symbol, buy_exchange, sell_exchange, 
                    orderbooks[buy_exchange], orderbooks[sell_exchange]
                )
                await self._check_arbitrage_pair(
                    symbol, sell_exchange, buy_exchange,
                    orderbooks[sell_exchange], orderbooks[buy_exchange]
                )
    
    async def _check_arbitrage_pair(
        self, 
        symbol: str, 
        buy_exchange: str, 
        sell_exchange: str,
        buy_orderbook: OrderBook, 
        sell_orderbook: OrderBook
    ) -> None:
        """Check arbitrage opportunity between two specific exchanges"""
        if not buy_orderbook.asks or not sell_orderbook.bids:
            return
        
        # Get best prices
        buy_price = buy_orderbook.asks[0][0]  # Best ask (buy price)
        sell_price = sell_orderbook.bids[0][0]  # Best bid (sell price)
        
        # Calculate raw spread
        spread_pct = (sell_price - buy_price) / buy_price
        
        if spread_pct <= 0:
            return  # No positive spread
        
        # Get fees
        buy_fees = self.exchanges[buy_exchange].fees.get(symbol, {'taker': 0.001})
        sell_fees = self.exchanges[sell_exchange].fees.get(symbol, {'taker': 0.001})
        
        # Calculate net profit after fees
        buy_fee_pct = buy_fees.get('taker', 0.001)
        sell_fee_pct = sell_fees.get('taker', 0.001)
        total_fees_pct = buy_fee_pct + sell_fee_pct
        
        net_profit_pct = spread_pct - total_fees_pct
        
        if net_profit_pct < self.min_profit_pct:
            return  # Not profitable enough
        
        # Calculate maximum quantity (limited by orderbook depth and position size)
        buy_qty = buy_orderbook.asks[0][1]
        sell_qty = sell_orderbook.bids[0][1]
        max_quantity = min(buy_qty, sell_qty)
        
        # Limit by maximum position size
        max_qty_by_position = self.max_position_usd / buy_price
        max_quantity = min(max_quantity, max_qty_by_position)
        
        # Calculate estimated profit
        estimated_profit_usd = max_quantity * buy_price * net_profit_pct
        
        # Adjust for latency risk
        total_latency = (
            self.exchanges[buy_exchange].latency_ms + 
            self.exchanges[sell_exchange].latency_ms
        )
        
        # Reduce profit estimate by latency risk (1% per 100ms of latency)
        latency_risk_pct = (total_latency / 100) * 0.01
        estimated_profit_usd *= (1 - latency_risk_pct)
        
        # Create opportunity
        opportunity = ArbitrageOpportunity(
            symbol=symbol,
            buy_exchange=buy_exchange,
            sell_exchange=sell_exchange,
            buy_price=buy_price,
            sell_price=sell_price,
            spread_pct=spread_pct,
            net_profit_pct=net_profit_pct,
            max_quantity=max_quantity,
            estimated_profit_usd=estimated_profit_usd,
            timestamp=time.time(),
            buy_fees=buy_fees,
            sell_fees=sell_fees
        )
        
        if opportunity.is_profitable:
            self.opportunities.append(opportunity)
            self.opportunity_history[symbol].append(opportunity)
            
            # Log to database
            await self._log_opportunity(opportunity)
            
            logger.info(
                f"Arbitrage opportunity: {symbol} "
                f"Buy {buy_exchange}@{buy_price:.6f} -> "
                f"Sell {sell_exchange}@{sell_price:.6f} "
                f"Profit: {net_profit_pct:.3f}% "
                f"Est: ${estimated_profit_usd:.2f}"
            )
    
    async def _log_opportunity(self, opportunity: ArbitrageOpportunity) -> None:
        """Log opportunity to database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO arbitrage_opportunities (
                    timestamp, symbol, buy_exchange, sell_exchange,
                    buy_price, sell_price, spread_pct, net_profit_pct,
                    max_quantity, estimated_profit_usd
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                opportunity.timestamp,
                opportunity.symbol,
                opportunity.buy_exchange,
                opportunity.sell_exchange,
                opportunity.buy_price,
                opportunity.sell_price,
                opportunity.spread_pct,
                opportunity.net_profit_pct,
                opportunity.max_quantity,
                opportunity.estimated_profit_usd
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Failed to log opportunity: {e}")
    
    async def _update_balances(self) -> None:
        """Update balances for all exchanges"""
        for exchange_name, exchange_state in self.exchanges.items():
            try:
                if exchange_state.connected:
                    balances = await exchange_state.exchange.get_balance()
                    exchange_state.balances = {
                        currency: balance.total 
                        for currency, balance in balances.items()
                    }
            except Exception as e:
                logger.error(f"Failed to update balances for {exchange_name}: {e}")
    
    def _clean_old_opportunities(self) -> None:
        """Remove old opportunities from history"""
        cutoff_time = time.time() - 3600  # Keep 1 hour of history
        
        for symbol in self.opportunity_history:
            self.opportunity_history[symbol] = [
                opp for opp in self.opportunity_history[symbol]
                if opp.timestamp > cutoff_time
            ]
    
    async def _log_statistics(self) -> None:
        """Log monitoring statistics"""
        if len(self.opportunities) == 0:
            return
        
        recent_opportunities = [
            opp for opp in self.opportunities
            if time.time() - opp.timestamp < 300  # Last 5 minutes
        ]
        
        if recent_opportunities:
            avg_profit = sum(opp.net_profit_pct for opp in recent_opportunities) / len(recent_opportunities)
            max_profit = max(opp.net_profit_pct for opp in recent_opportunities)
            
            logger.info(
                f"Arbitrage stats (5min): {len(recent_opportunities)} opportunities, "
                f"avg profit: {avg_profit:.3f}%, max profit: {max_profit:.3f}%"
            )
    
    def get_recent_opportunities(self, minutes: int = 60) -> List[ArbitrageOpportunity]:
        """Get opportunities from the last N minutes"""
        cutoff = time.time() - (minutes * 60)
        return [opp for opp in self.opportunities if opp.timestamp > cutoff]
    
    def get_best_opportunity(self) -> Optional[ArbitrageOpportunity]:
        """Get the most profitable recent opportunity"""
        recent = self.get_recent_opportunities(5)  # Last 5 minutes
        if not recent:
            return None
        
        return max(recent, key=lambda opp: opp.estimated_profit_usd)
    
    async def execute_arbitrage(self, opportunity: ArbitrageOpportunity) -> bool:
        """
        Execute an arbitrage opportunity
        WARNING: This is a basic implementation. Production use requires:
        - Portfolio risk management
        - Real-time inventory checks
        - Partial fill handling
        - Error recovery
        """
        logger.warning("execute_arbitrage called - This is not implemented for safety")
        logger.info(f"Would execute: {opportunity}")
        
        # In a real implementation, this would:
        # 1. Check current balances
        # 2. Validate opportunity is still valid
        # 3. Place simultaneous orders on both exchanges
        # 4. Monitor execution and handle partial fills
        # 5. Update portfolio state
        # 6. Handle transfer between exchanges if needed
        
        return False
    
    async def shutdown(self) -> None:
        """Shutdown monitoring and disconnect all exchanges"""
        logger.info("Shutting down arbitrage monitor...")
        
        for exchange_name, exchange_state in self.exchanges.items():
            try:
                await exchange_state.exchange.disconnect()
            except Exception as e:
                logger.error(f"Error disconnecting {exchange_name}: {e}")
        
        self.exchanges.clear()
        logger.info("Arbitrage monitor shutdown complete")

# Example usage
async def main():
    """Example usage of ArbitrageMonitor"""
    monitor = ArbitrageMonitor()
    
    # Add exchanges
    binance_config = ExchangeConfig(
        name="binance",
        rest_url="https://api.binance.us/api",
        ws_url="wss://stream.binance.us:9443/ws"
    )
    
    coinbase_config = ExchangeConfig(
        name="coinbase",
        rest_url="https://api.coinbase.com"
    )
    
    try:
        await monitor.add_exchange("binance", binance_config)
        await monitor.add_exchange("coinbase", coinbase_config)
        
        # Add symbols to monitor
        monitor.add_symbol("BTC/USD")
        monitor.add_symbol("ETH/USD")
        
        # Start monitoring
        await monitor.start_monitoring()
        
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    except Exception as e:
        logger.error(f"Error: {e}")
    finally:
        await monitor.shutdown()

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())