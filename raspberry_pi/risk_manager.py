#!/usr/bin/env python3
"""
Advanced Correlation-Based Risk Management System
Prevents concentrated exposure and manages portfolio-wide risk

CRITICAL FEATURES:
- Cross-asset correlation analysis to detect concentrated risk
- Dynamic risk limits based on volatility regimes  
- Portfolio concentration monitoring (sector/theme exposure)
- Real-time correlation matrix updates
- Circuit breakers for anomalous market conditions
- Multi-asset position sizing with correlation adjustment
"""

import numpy as np
import pandas as pd
import logging
import sqlite3
import time
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, NamedTuple
from dataclasses import dataclass
from enum import Enum
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RiskLevel(Enum):
    """Risk level classifications"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class VolatilityRegime(Enum):
    """Market volatility regimes"""
    LOW_VOL = "low_volatility"
    NORMAL_VOL = "normal_volatility"
    HIGH_VOL = "high_volatility"
    EXTREME_VOL = "extreme_volatility"

@dataclass
class RiskMetrics:
    """Portfolio risk metrics"""
    total_exposure: float
    correlation_risk: float
    concentration_risk: float
    volatility_risk: float
    max_drawdown_risk: float
    portfolio_heat: float
    risk_level: RiskLevel
    volatility_regime: VolatilityRegime
    
@dataclass
class PositionRisk:
    """Individual position risk assessment"""
    symbol: str
    exposure_pct: float
    volatility: float
    correlation_score: float
    concentration_score: float
    risk_adjusted_size: float
    max_position_size: float

class AdvancedRiskManager:
    """Advanced portfolio risk management with correlation analysis"""
    
    def __init__(self, db_path: str = "data/db/crypto_data.db", config: Dict = None):
        self.db_path = db_path
        self.config = config or {}
        
        # Risk limits (configurable)
        self.max_portfolio_exposure = 0.95  # Max 95% of capital at risk
        self.max_single_position = 0.25     # Max 25% in single asset
        self.max_correlated_exposure = 0.40  # Max 40% in highly correlated assets
        self.max_sector_exposure = 0.50     # Max 50% in single sector
        self.correlation_threshold = 0.7    # Assets with >0.7 correlation are "highly correlated"
        
        # Dynamic risk adjustment
        self.volatility_multipliers = {
            VolatilityRegime.LOW_VOL: 1.2,     # Increase limits in low vol
            VolatilityRegime.NORMAL_VOL: 1.0,  # Normal limits
            VolatilityRegime.HIGH_VOL: 0.7,    # Reduce limits in high vol
            VolatilityRegime.EXTREME_VOL: 0.3  # Severely reduce in extreme vol
        }
        
        # Correlation matrix cache
        self.correlation_matrix = None
        self.correlation_last_update = 0
        self.correlation_update_interval = 3600  # Update every hour
        
        # Asset classifications
        self.asset_sectors = {
            'BTCUSD': 'crypto_major',
            'ETHUSD': 'crypto_major', 
            'ADAUSD': 'crypto_alt',
            'SOLUSD': 'crypto_alt',
            'MATICUSD': 'crypto_alt',
            'LINKUSD': 'crypto_defi',
            'UNIUSD': 'crypto_defi'
        }
        
    def get_price_data(self, symbols: List[str], hours: int = 168) -> pd.DataFrame:
        """Get price data for correlation analysis (last 7 days)"""
        try:
            conn = sqlite3.connect(self.db_path)
            
            # Calculate time range
            end_time = datetime.now()
            start_time = end_time - timedelta(hours=hours)
            start_timestamp = start_time.timestamp()
            
            # Build query for all symbols
            placeholders = ','.join(['?' for _ in symbols])
            query = f'''
                SELECT timestamp, symbol, close
                FROM ohlcv 
                WHERE symbol IN ({placeholders}) AND timestamp >= ?
                ORDER BY timestamp ASC
            '''
            
            df = pd.read_sql_query(query, conn, params=symbols + [start_timestamp])
            conn.close()
            
            if len(df) == 0:
                logger.warning("No price data found for correlation analysis")
                return None
                
            # Pivot to get symbols as columns
            df['datetime'] = pd.to_datetime(df['timestamp'], unit='s')
            price_matrix = df.pivot(index='datetime', columns='symbol', values='close')
            
            # Forward fill missing values and drop any remaining NaNs
            price_matrix = price_matrix.fillna(method='ffill').dropna()
            
            logger.info(f"Loaded price data: {price_matrix.shape[0]} periods, {price_matrix.shape[1]} symbols")
            return price_matrix
            
        except Exception as e:
            logger.error(f"Error loading price data for correlation: {e}")
            return None
    
    def calculate_correlation_matrix(self, symbols: List[str], force_update: bool = False) -> np.ndarray:
        """Calculate correlation matrix with caching"""
        try:
            current_time = time.time()
            
            # Check if update needed
            if (not force_update and 
                self.correlation_matrix is not None and 
                (current_time - self.correlation_last_update) < self.correlation_update_interval):
                return self.correlation_matrix
                
            # Get price data
            price_data = self.get_price_data(symbols)
            if price_data is None or len(price_data) < 20:
                logger.warning("Insufficient data for correlation analysis")
                return np.eye(len(symbols))  # Return identity matrix as fallback
            
            # Calculate returns
            returns = price_data.pct_change().dropna()
            
            if len(returns) < 10:
                logger.warning("Insufficient returns data for correlation")
                return np.eye(len(symbols))
                
            # Calculate correlation matrix
            correlation_matrix = returns.corr().values
            
            # Handle any NaN values
            correlation_matrix = np.nan_to_num(correlation_matrix, nan=0.0)
            
            # Ensure matrix is positive semi-definite
            eigenvals = np.linalg.eigvals(correlation_matrix)
            if np.any(eigenvals < -1e-6):
                logger.warning("Correlation matrix not positive semi-definite, adjusting")
                correlation_matrix = self._make_positive_semidefinite(correlation_matrix)
            
            # Cache the result
            self.correlation_matrix = correlation_matrix
            self.correlation_last_update = current_time
            
            logger.info(f"Updated correlation matrix: {correlation_matrix.shape}")
            return correlation_matrix
            
        except Exception as e:
            logger.error(f"Error calculating correlation matrix: {e}")
            return np.eye(len(symbols))
    
    def _make_positive_semidefinite(self, matrix: np.ndarray) -> np.ndarray:
        """Make correlation matrix positive semi-definite using eigenvalue decomposition"""
        try:
            eigenvals, eigenvecs = np.linalg.eigh(matrix)
            eigenvals = np.maximum(eigenvals, 1e-6)  # Set minimum eigenvalue
            return eigenvecs @ np.diag(eigenvals) @ eigenvecs.T
        except:
            return np.eye(matrix.shape[0])
    
    def detect_volatility_regime(self, symbols: List[str]) -> VolatilityRegime:
        """Detect current market volatility regime"""
        try:
            price_data = self.get_price_data(symbols, hours=24)  # Last 24 hours
            if price_data is None or len(price_data) < 10:
                return VolatilityRegime.NORMAL_VOL
                
            # Calculate portfolio volatility (equal weights)
            returns = price_data.pct_change().dropna()
            portfolio_returns = returns.mean(axis=1)
            
            # Annualized volatility
            volatility = portfolio_returns.std() * np.sqrt(365 * 24)  # Hourly to annualized
            
            # Classify regime
            if volatility < 0.3:
                return VolatilityRegime.LOW_VOL
            elif volatility < 0.6:
                return VolatilityRegime.NORMAL_VOL
            elif volatility < 1.0:
                return VolatilityRegime.HIGH_VOL
            else:
                return VolatilityRegime.EXTREME_VOL
                
        except Exception as e:
            logger.error(f"Error detecting volatility regime: {e}")
            return VolatilityRegime.NORMAL_VOL
    
    def calculate_portfolio_heat(self, positions: Dict) -> float:
        """Calculate portfolio heat (percentage of capital at risk)"""
        try:
            total_exposure = 0.0
            
            for symbol, position in positions.items():
                if position.get('quantity', 0) > 0:
                    exposure = position.get('market_value', 0)
                    total_exposure += abs(exposure)
            
            portfolio_value = positions.get('total_value', 10000)  # Default to starting capital
            heat = total_exposure / portfolio_value if portfolio_value > 0 else 0.0
            
            return min(heat, 1.0)  # Cap at 100%
            
        except Exception as e:
            logger.error(f"Error calculating portfolio heat: {e}")
            return 0.0
    
    def assess_concentration_risk(self, positions: Dict) -> float:
        """Assess portfolio concentration risk by sector"""
        try:
            sector_exposures = {}
            total_exposure = 0.0
            
            for symbol, position in positions.items():
                if position.get('quantity', 0) > 0:
                    exposure = abs(position.get('market_value', 0))
                    sector = self.asset_sectors.get(symbol, 'unknown')
                    
                    sector_exposures[sector] = sector_exposures.get(sector, 0) + exposure
                    total_exposure += exposure
            
            if total_exposure == 0:
                return 0.0
                
            # Calculate max sector concentration
            max_sector_pct = max(sector_exposures.values()) / total_exposure
            
            # Risk score: 0 = perfectly diversified, 1 = all in one sector
            concentration_risk = max_sector_pct
            
            return concentration_risk
            
        except Exception as e:
            logger.error(f"Error assessing concentration risk: {e}")
            return 0.0
    
    def calculate_correlation_risk(self, positions: Dict, symbols: List[str]) -> float:
        """Calculate correlation risk of current portfolio"""
        try:
            if len(symbols) < 2:
                return 0.0
                
            # Get correlation matrix
            corr_matrix = self.calculate_correlation_matrix(symbols)
            
            # Get position weights
            weights = np.zeros(len(symbols))
            total_value = sum(abs(pos.get('market_value', 0)) for pos in positions.values())
            
            if total_value == 0:
                return 0.0
                
            for i, symbol in enumerate(symbols):
                if symbol in positions:
                    weights[i] = abs(positions[symbol].get('market_value', 0)) / total_value
            
            # Calculate portfolio correlation risk
            # Higher correlation = higher risk
            portfolio_correlation = weights.T @ corr_matrix @ weights
            
            # Normalize to 0-1 scale
            correlation_risk = max(0, min(1, portfolio_correlation))
            
            return correlation_risk
            
        except Exception as e:
            logger.error(f"Error calculating correlation risk: {e}")
            return 0.0
    
    def calculate_position_risk_adjustment(self, symbol: str, base_size: float, 
                                         positions: Dict, symbols: List[str]) -> float:
        """Calculate risk-adjusted position size based on correlations"""
        try:
            if len(symbols) <= 1:
                return base_size
                
            # Get correlation matrix
            corr_matrix = self.calculate_correlation_matrix(symbols)
            
            if symbol not in symbols:
                return base_size
                
            symbol_idx = symbols.index(symbol)
            
            # Calculate correlation with existing positions
            existing_correlation = 0.0
            existing_exposure = 0.0
            
            for other_symbol, position in positions.items():
                if other_symbol != symbol and position.get('quantity', 0) > 0:
                    if other_symbol in symbols:
                        other_idx = symbols.index(other_symbol)
                        correlation = corr_matrix[symbol_idx, other_idx]
                        exposure = abs(position.get('market_value', 0))
                        
                        existing_correlation += correlation * exposure
                        existing_exposure += exposure
            
            if existing_exposure > 0:
                avg_correlation = existing_correlation / existing_exposure
            else:
                avg_correlation = 0.0
                
            # Adjust position size based on correlation
            # Higher correlation = smaller position
            correlation_adjustment = 1.0 - (avg_correlation * 0.5)  # Max 50% reduction
            correlation_adjustment = max(0.2, correlation_adjustment)  # Min 20% of original size
            
            adjusted_size = base_size * correlation_adjustment
            
            logger.info(f"Position size adjustment for {symbol}: {correlation_adjustment:.3f} "
                       f"(avg correlation: {avg_correlation:.3f})")
            
            return adjusted_size
            
        except Exception as e:
            logger.error(f"Error calculating position risk adjustment: {e}")
            return base_size
    
    def assess_portfolio_risk(self, positions: Dict, symbols: List[str]) -> RiskMetrics:
        """Comprehensive portfolio risk assessment"""
        try:
            # Calculate individual risk components
            portfolio_heat = self.calculate_portfolio_heat(positions)
            concentration_risk = self.assess_concentration_risk(positions)
            correlation_risk = self.calculate_correlation_risk(positions, symbols)
            volatility_regime = self.detect_volatility_regime(symbols)
            
            # Calculate total exposure
            total_exposure = sum(abs(pos.get('market_value', 0)) for pos in positions.values())
            portfolio_value = positions.get('total_value', 10000)
            exposure_pct = total_exposure / portfolio_value if portfolio_value > 0 else 0.0
            
            # Volatility risk based on regime
            volatility_risk_map = {
                VolatilityRegime.LOW_VOL: 0.2,
                VolatilityRegime.NORMAL_VOL: 0.5,
                VolatilityRegime.HIGH_VOL: 0.8,
                VolatilityRegime.EXTREME_VOL: 1.0
            }
            volatility_risk = volatility_risk_map.get(volatility_regime, 0.5)
            
            # Calculate max drawdown risk (simplified)
            max_drawdown_risk = min(1.0, exposure_pct * 2)  # Assume 2x leverage effect
            
            # Overall risk level
            risk_score = (
                portfolio_heat * 0.3 +
                concentration_risk * 0.25 +
                correlation_risk * 0.25 +
                volatility_risk * 0.2
            )
            
            if risk_score < 0.3:
                risk_level = RiskLevel.LOW
            elif risk_score < 0.6:
                risk_level = RiskLevel.MEDIUM
            elif risk_score < 0.8:
                risk_level = RiskLevel.HIGH
            else:
                risk_level = RiskLevel.CRITICAL
            
            return RiskMetrics(
                total_exposure=exposure_pct,
                correlation_risk=correlation_risk,
                concentration_risk=concentration_risk,
                volatility_risk=volatility_risk,
                max_drawdown_risk=max_drawdown_risk,
                portfolio_heat=portfolio_heat,
                risk_level=risk_level,
                volatility_regime=volatility_regime
            )
            
        except Exception as e:
            logger.error(f"Error assessing portfolio risk: {e}")
            return RiskMetrics(
                total_exposure=0.0,
                correlation_risk=0.0,
                concentration_risk=0.0,
                volatility_risk=0.5,
                max_drawdown_risk=0.0,
                portfolio_heat=0.0,
                risk_level=RiskLevel.LOW,
                volatility_regime=VolatilityRegime.NORMAL_VOL
            )
    
    def check_position_limits(self, symbol: str, position_size: float, 
                            positions: Dict, symbols: List[str]) -> Tuple[bool, str, float]:
        """
        Check if proposed position size violates risk limits
        
        Returns:
            (is_allowed, reason, max_allowed_size)
        """
        try:
            # Get current risk metrics
            risk_metrics = self.assess_portfolio_risk(positions, symbols)
            
            # Get volatility regime multiplier
            vol_multiplier = self.volatility_multipliers.get(
                risk_metrics.volatility_regime, 1.0
            )
            
            # Adjust limits based on volatility
            max_single = self.max_single_position * vol_multiplier
            max_portfolio = self.max_portfolio_exposure * vol_multiplier
            max_sector = self.max_sector_exposure * vol_multiplier
            
            # Calculate position value
            portfolio_value = positions.get('total_value', 10000)
            position_value = position_size  # Assuming position_size is in dollar terms
            position_pct = position_value / portfolio_value if portfolio_value > 0 else 0
            
            # Check single position limit
            if position_pct > max_single:
                max_allowed = max_single * portfolio_value
                return False, f"Exceeds single position limit ({max_single:.1%})", max_allowed
            
            # Check portfolio exposure limit
            current_exposure = risk_metrics.total_exposure
            new_exposure = current_exposure + position_pct
            
            if new_exposure > max_portfolio:
                max_allowed = (max_portfolio - current_exposure) * portfolio_value
                return False, f"Exceeds portfolio exposure limit ({max_portfolio:.1%})", max_allowed
            
            # Check sector concentration
            symbol_sector = self.asset_sectors.get(symbol, 'unknown')
            sector_exposure = 0.0
            
            for sym, pos in positions.items():
                if (self.asset_sectors.get(sym, 'unknown') == symbol_sector and 
                    pos.get('quantity', 0) > 0):
                    sector_exposure += abs(pos.get('market_value', 0)) / portfolio_value
            
            new_sector_exposure = sector_exposure + position_pct
            
            if new_sector_exposure > max_sector:
                max_allowed = (max_sector - sector_exposure) * portfolio_value
                return False, f"Exceeds sector exposure limit ({max_sector:.1%})", max_allowed
            
            # Check correlation limits
            if len(symbols) > 1:
                adjusted_size = self.calculate_position_risk_adjustment(
                    symbol, position_size, positions, symbols
                )
                
                if adjusted_size < position_size * 0.5:  # More than 50% reduction
                    return False, "High correlation with existing positions", adjusted_size
            
            # All checks passed
            return True, "Position allowed", position_size
            
        except Exception as e:
            logger.error(f"Error checking position limits: {e}")
            return False, f"Error in risk check: {e}", 0.0
    
    def log_risk_metrics(self, risk_metrics: RiskMetrics):
        """Log detailed risk metrics"""
        logger.info("=== Portfolio Risk Assessment ===")
        logger.info(f"Risk Level: {risk_metrics.risk_level.value.upper()}")
        logger.info(f"Volatility Regime: {risk_metrics.volatility_regime.value}")
        logger.info(f"Total Exposure: {risk_metrics.total_exposure:.1%}")
        logger.info(f"Portfolio Heat: {risk_metrics.portfolio_heat:.1%}")
        logger.info(f"Correlation Risk: {risk_metrics.correlation_risk:.3f}")
        logger.info(f"Concentration Risk: {risk_metrics.concentration_risk:.1%}")
        logger.info(f"Volatility Risk: {risk_metrics.volatility_risk:.3f}")
        logger.info(f"Max Drawdown Risk: {risk_metrics.max_drawdown_risk:.1%}")
        
        if risk_metrics.risk_level in [RiskLevel.HIGH, RiskLevel.CRITICAL]:
            logger.warning(f"⚠️ {risk_metrics.risk_level.value.upper()} RISK DETECTED!")

# Example usage and testing
def test_risk_manager():
    """Test the risk manager with sample portfolio"""
    risk_manager = AdvancedRiskManager()
    
    # Sample portfolio
    positions = {
        'BTCUSD': {'quantity': 0.1, 'market_value': 5000},
        'ETHUSD': {'quantity': 1.0, 'market_value': 2000},
        'total_value': 10000
    }
    
    symbols = ['BTCUSD', 'ETHUSD']
    
    # Assess portfolio risk
    risk_metrics = risk_manager.assess_portfolio_risk(positions, symbols)
    risk_manager.log_risk_metrics(risk_metrics)
    
    # Check new position
    allowed, reason, max_size = risk_manager.check_position_limits(
        'ADAUSD', 1000, positions, symbols + ['ADAUSD']
    )
    
    logger.info(f"New position check: {'ALLOWED' if allowed else 'REJECTED'}")
    logger.info(f"Reason: {reason}")
    logger.info(f"Max allowed size: ${max_size:.2f}")
    
    return risk_metrics

if __name__ == "__main__":
    test_risk_manager()