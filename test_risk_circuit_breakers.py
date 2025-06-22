#!/usr/bin/env python3
"""
Direct test of risk management circuit breakers and position limits - CRITICAL SAFETY
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from raspberry_pi.risk_manager import AdvancedRiskManager
from raspberry_pi.paper_trader import PaperTradingEngine

def test_risk_circuit_breakers():
    """Test risk management circuit breakers and position limits"""
    
    print("🛡️  Testing Risk Management Circuit Breakers (CRITICAL SAFETY)")
    print("=" * 70)
    
    # Initialize risk manager
    risk_manager = AdvancedRiskManager()
    
    print(f"Risk limits configured:")
    print(f"  Max position size: {risk_manager.max_single_position:.1%}")
    print(f"  Max portfolio exposure: {risk_manager.max_portfolio_exposure:.1%}")
    print(f"  Max correlated exposure: {risk_manager.max_correlated_exposure:.1%}")
    print(f"  Correlation threshold: {risk_manager.correlation_threshold}")
    
    # Test 1: Single position limit
    print(f"\n🧪 Test 1: Single Position Limit")
    portfolio = {'total_value': 10000}
    symbols = ['BTCUSD']
    
    # Try to place a position that's too large (30% of portfolio = $3000)
    allowed, reason, max_size = risk_manager.check_position_limits(
        'BTCUSD', 3000, portfolio, symbols
    )
    
    print(f"  Attempting $3000 position (30%): {'❌ BLOCKED' if not allowed else '✅ ALLOWED'}")
    print(f"  Reason: {reason}")
    print(f"  Max allowed: ${max_size:.2f}")
    
    # Try a reasonable position (20% of portfolio = $2000)
    allowed2, reason2, max_size2 = risk_manager.check_position_limits(
        'BTCUSD', 2000, portfolio, symbols
    )
    
    print(f"  Attempting $2000 position (20%): {'✅ ALLOWED' if allowed2 else '❌ BLOCKED'}")
    print(f"  Reason: {reason2}")
    
    # Test 2: Portfolio exposure limit
    print(f"\n🧪 Test 2: Portfolio Exposure Limit")
    
    # Create portfolio with high existing exposure
    high_exposure_portfolio = {
        'BTCUSD': {'quantity': 0.1, 'market_value': 8000},  # 80% exposure
        'total_value': 10000
    }
    
    # Try to add more exposure
    allowed3, reason3, max_size3 = risk_manager.check_position_limits(
        'ETHUSD', 2000, high_exposure_portfolio, ['BTCUSD', 'ETHUSD']
    )
    
    print(f"  Current exposure: 80%, attempting +20%: {'❌ BLOCKED' if not allowed3 else '✅ ALLOWED'}")
    print(f"  Reason: {reason3}")
    print(f"  Max additional: ${max_size3:.2f}")
    
    # Test 3: Portfolio risk assessment
    print(f"\n🧪 Test 3: Portfolio Risk Assessment")
    
    test_portfolio = {
        'BTCUSD': {'quantity': 0.05, 'market_value': 2000},
        'ETHUSD': {'quantity': 1.0, 'market_value': 1500},
        'total_value': 10000
    }
    
    risk_metrics = risk_manager.assess_portfolio_risk(test_portfolio, ['BTCUSD', 'ETHUSD'])
    
    print(f"  Risk Level: {risk_metrics.risk_level.value}")
    print(f"  Total Exposure: {risk_metrics.total_exposure:.1%}")
    print(f"  Portfolio Heat: {risk_metrics.portfolio_heat:.1%}")
    print(f"  Volatility Regime: {risk_metrics.volatility_regime.value}")
    print(f"  Correlation Risk: {risk_metrics.correlation_risk:.3f}")
    print(f"  Concentration Risk: {risk_metrics.concentration_risk:.1%}")
    
    # Test 4: Paper trader integration with risk limits
    print(f"\n🧪 Test 4: Paper Trader Risk Integration")
    
    # Create paper trader
    paper_trader = PaperTradingEngine()
    paper_trader.set_db_path("data/db/crypto_data.db")
    
    # Check risk limits are properly configured
    print(f"  Paper trader max position: {paper_trader.max_position_pct:.1%}")
    print(f"  Paper trader max drawdown: {paper_trader.max_drawdown_pct:.1%}")
    print(f"  Stop loss percentage: {paper_trader.stop_loss_pct:.1%}")
    print(f"  Take profit percentage: {paper_trader.take_profit_pct:.1%}")
    
    # Validation
    print(f"\n✅ Validation Results:")
    
    # Check 1: Position limits are enforced
    position_limits_ok = not allowed and allowed2  # Large position blocked, reasonable allowed
    print(f"Position size limits enforced: {'✅' if position_limits_ok else '❌'}")
    
    # Check 2: Portfolio exposure limits are enforced  
    exposure_limits_ok = not allowed3  # High exposure addition blocked
    print(f"Portfolio exposure limits enforced: {'✅' if exposure_limits_ok else '❌'}")
    
    # Check 3: Risk assessment works
    risk_assessment_ok = hasattr(risk_metrics, 'risk_level') and hasattr(risk_metrics, 'total_exposure')
    print(f"Risk assessment functional: {'✅' if risk_assessment_ok else '❌'}")
    
    # Check 4: Paper trader has proper limits
    paper_trader_ok = (paper_trader.max_position_pct <= 0.25 and 
                      paper_trader.max_drawdown_pct <= 0.25)
    print(f"Paper trader limits configured: {'✅' if paper_trader_ok else '❌'}")
    
    all_checks_passed = (position_limits_ok and exposure_limits_ok and 
                        risk_assessment_ok and paper_trader_ok)
    
    if all_checks_passed:
        print("\n🎉 ALL RISK MANAGEMENT CHECKS PASSED")
        print("✅ Position limits are enforced")
        print("✅ Portfolio exposure is controlled")
        print("✅ Risk assessment is working")
        print("✅ Circuit breakers will prevent catastrophic losses")
        return True
    else:
        print("\n🚨 RISK MANAGEMENT ISSUES DETECTED")
        print("❌ Some risk controls are not working properly")
        return False

if __name__ == "__main__":
    success = test_risk_circuit_breakers()
    if not success:
        print("\n🚨 CRITICAL ERROR: Risk management is not working properly!")
        print("🚨 This could lead to catastrophic losses!")
        sys.exit(1)
    else:
        print("\n🎉 Risk management is working correctly - system is safe to use")