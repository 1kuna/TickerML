#!/usr/bin/env python3
"""
Exchange Integration Validation Test
Tests all 4 exchange implementations for completeness and API compatibility
"""

import asyncio
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from raspberry_pi.exchanges import (
    create_exchange, ExchangeConfig, EXCHANGES,
    BinanceExchange, CoinbaseExchange, KrakenExchange, KuCoinExchange
)
from raspberry_pi.exchanges.base import ExchangeInterface

def test_exchange_factory():
    """Test exchange factory and registry"""
    
    print("🏭 Testing Exchange Factory and Registry")
    print("=" * 50)
    
    # Test 1: Exchange registry completeness
    print(f"\n📋 Test 1: Exchange Registry")
    expected_exchanges = ['binance', 'coinbase', 'kraken', 'kucoin']
    
    print(f"Expected exchanges: {expected_exchanges}")
    print(f"Registered exchanges: {list(EXCHANGES.keys())}")
    
    registry_complete = all(exchange in EXCHANGES for exchange in expected_exchanges)
    print(f"Registry complete: {'✅' if registry_complete else '❌'}")
    
    # Test 2: Factory function
    print(f"\n🏗️ Test 2: Factory Function")
    
    factory_tests = []
    for exchange_name in expected_exchanges:
        try:
            config = ExchangeConfig(name=exchange_name)
            exchange = create_exchange(exchange_name, config)
            
            is_interface = isinstance(exchange, ExchangeInterface)
            factory_tests.append((exchange_name, is_interface, exchange.__class__.__name__))
            print(f"  {exchange_name}: {'✅' if is_interface else '❌'} ({exchange.__class__.__name__})")
            
        except Exception as e:
            factory_tests.append((exchange_name, False, str(e)))
            print(f"  {exchange_name}: ❌ ({e})")
    
    factory_success = all(test[1] for test in factory_tests)
    print(f"Factory working: {'✅' if factory_success else '❌'}")
    
    return registry_complete and factory_success

def test_interface_compliance():
    """Test that all exchanges implement the required interface methods"""
    
    print("\n🔌 Testing Interface Compliance")
    print("=" * 40)
    
    # Required abstract methods from ExchangeInterface
    required_methods = [
        'connect', 'disconnect', 'get_symbols', 'get_orderbook',
        'subscribe_orderbook', 'unsubscribe_orderbook', 'subscribe_trades',
        'unsubscribe_trades', 'get_balance', 'place_order', 'cancel_order',
        'get_order', 'get_open_orders', 'get_fees', 'get_server_time'
    ]
    
    exchange_classes = [BinanceExchange, CoinbaseExchange, KrakenExchange, KuCoinExchange]
    
    compliance_results = {}
    
    for exchange_class in exchange_classes:
        exchange_name = exchange_class.__name__.replace('Exchange', '').lower()
        print(f"\n📋 {exchange_name.title()} Exchange:")
        
        missing_methods = []
        for method in required_methods:
            if not hasattr(exchange_class, method):
                missing_methods.append(method)
            else:
                # Check if method is properly implemented (not just inherited abstract)
                method_obj = getattr(exchange_class, method)
                if hasattr(method_obj, '__isabstractmethod__') and method_obj.__isabstractmethod__:
                    missing_methods.append(f"{method} (abstract)")
        
        if missing_methods:
            print(f"  Missing methods: {missing_methods}")
            compliance_results[exchange_name] = False
        else:
            print(f"  All required methods implemented: ✅")
            compliance_results[exchange_name] = True
    
    all_compliant = all(compliance_results.values())
    print(f"\nAll exchanges compliant: {'✅' if all_compliant else '❌'}")
    
    return all_compliant

def test_configuration_handling():
    """Test exchange configuration handling"""
    
    print("\n⚙️ Testing Configuration Handling")
    print("=" * 35)
    
    config_tests = {}
    
    for exchange_name in ['binance', 'coinbase', 'kraken', 'kucoin']:
        print(f"\n🔧 {exchange_name.title()} Configuration:")
        
        try:
            # Test basic config
            config = ExchangeConfig(
                name=exchange_name,
                testnet=True,
                rate_limit=5
            )
            
            exchange = create_exchange(exchange_name, config)
            
            # Check config is properly set
            config_valid = (
                exchange.config.name == exchange_name and
                exchange.config.testnet == True and
                exchange.config.rate_limit == 5 and
                exchange.name == exchange_name
            )
            
            print(f"  Basic config: {'✅' if config_valid else '❌'}")
            
            # Test with API credentials (should not fail even if invalid)
            config_with_creds = ExchangeConfig(
                name=exchange_name,
                api_key="test_key",
                api_secret="test_secret",
                passphrase="test_passphrase" if exchange_name == 'coinbase' else None
            )
            
            exchange_with_creds = create_exchange(exchange_name, config_with_creds)
            creds_handled = (
                exchange_with_creds.config.api_key == "test_key" and
                exchange_with_creds.config.api_secret == "test_secret"
            )
            
            print(f"  Credential handling: {'✅' if creds_handled else '❌'}")
            
            config_tests[exchange_name] = config_valid and creds_handled
            
        except Exception as e:
            print(f"  Configuration failed: ❌ ({e})")
            config_tests[exchange_name] = False
    
    all_configs_ok = all(config_tests.values())
    print(f"\nAll configurations valid: {'✅' if all_configs_ok else '❌'}")
    
    return all_configs_ok

def test_symbol_normalization():
    """Test symbol normalization across exchanges"""
    
    print("\n🔤 Testing Symbol Normalization")
    print("=" * 35)
    
    test_symbols = [
        'BTC/USDT', 'ETH/USD', 'BTC/USD', 'ETH/USDT'
    ]
    
    normalization_tests = {}
    
    for exchange_name in ['binance', 'coinbase', 'kraken', 'kucoin']:
        print(f"\n📝 {exchange_name.title()} Symbol Normalization:")
        
        try:
            config = ExchangeConfig(name=exchange_name)
            exchange = create_exchange(exchange_name, config)
            
            normalization_working = True
            
            for symbol in test_symbols:
                # Test normalization (if method exists)
                if hasattr(exchange, 'normalize_symbol'):
                    normalized = exchange.normalize_symbol(symbol)
                    print(f"  {symbol} -> {normalized}")
                else:
                    print(f"  No normalize_symbol method")
                
                # Test exchange-specific conversion methods
                conversion_methods = [
                    '_to_binance_symbol', '_to_coinbase_symbol', 
                    '_to_kraken_symbol', '_to_kucoin_symbol'
                ]
                
                found_method = False
                for method_name in conversion_methods:
                    if hasattr(exchange, method_name):
                        try:
                            converted = getattr(exchange, method_name)(symbol)
                            print(f"    {method_name}: {symbol} -> {converted}")
                            found_method = True
                            break
                        except Exception as e:
                            print(f"    {method_name} failed: {e}")
                
                if not found_method:
                    print(f"    No conversion method found")
                    normalization_working = False
            
            normalization_tests[exchange_name] = normalization_working
            print(f"  Normalization working: {'✅' if normalization_working else '❌'}")
            
        except Exception as e:
            print(f"  Normalization test failed: ❌ ({e})")
            normalization_tests[exchange_name] = False
    
    all_normalizations_ok = all(normalization_tests.values())
    print(f"\nAll normalizations working: {'✅' if all_normalizations_ok else '❌'}")
    
    return all_normalizations_ok

async def test_connection_methods():
    """Test connection methods (without actually connecting)"""
    
    print("\n🔗 Testing Connection Methods")
    print("=" * 30)
    
    connection_tests = {}
    
    for exchange_name in ['binance', 'coinbase', 'kraken', 'kucoin']:
        print(f"\n🌐 {exchange_name.title()} Connection Methods:")
        
        try:
            config = ExchangeConfig(name=exchange_name, testnet=True)
            exchange = create_exchange(exchange_name, config)
            
            # Test that connection methods exist and are callable
            connect_exists = hasattr(exchange, 'connect') and callable(exchange.connect)
            disconnect_exists = hasattr(exchange, 'disconnect') and callable(exchange.disconnect)
            
            print(f"  connect() method: {'✅' if connect_exists else '❌'}")
            print(f"  disconnect() method: {'✅' if disconnect_exists else '❌'}")
            
            # Test URL configuration
            rest_url_set = hasattr(exchange, 'rest_url') and exchange.rest_url
            ws_url_set = hasattr(exchange, 'ws_url') and getattr(exchange, 'ws_url', None)
            
            print(f"  REST URL configured: {'✅' if rest_url_set else '❌'}")
            print(f"  WebSocket URL configured: {'✅' if ws_url_set else '❌'}")
            
            if rest_url_set:
                print(f"    REST URL: {exchange.rest_url}")
            if ws_url_set:
                print(f"    WS URL: {getattr(exchange, 'ws_url', 'Not set')}")
            
            connection_tests[exchange_name] = (
                connect_exists and disconnect_exists and rest_url_set
            )
            
        except Exception as e:
            print(f"  Connection test failed: ❌ ({e})")
            connection_tests[exchange_name] = False
    
    all_connections_ok = all(connection_tests.values())
    print(f"\nAll connection methods ready: {'✅' if all_connections_ok else '❌'}")
    
    return all_connections_ok

def test_order_types_and_enums():
    """Test order types and enum handling"""
    
    print("\n📋 Testing Order Types and Enums")
    print("=" * 35)
    
    from raspberry_pi.exchanges.base import OrderType, OrderSide, OrderStatus
    
    enum_tests = {}
    
    for exchange_name in ['binance', 'coinbase', 'kraken', 'kucoin']:
        print(f"\n🏷️ {exchange_name.title()} Enum Handling:")
        
        try:
            config = ExchangeConfig(name=exchange_name)
            exchange = create_exchange(exchange_name, config)
            
            # Test order type conversion methods
            order_type_methods = [
                '_to_binance_order_type', '_to_coinbase_order_type',
                '_to_kraken_order_type', '_to_kucoin_order_type'
            ]
            
            conversion_working = False
            for method_name in order_type_methods:
                if hasattr(exchange, method_name):
                    try:
                        # Test with different order types
                        market_result = getattr(exchange, method_name)(OrderType.MARKET)
                        limit_result = getattr(exchange, method_name)(OrderType.LIMIT)
                        
                        print(f"  {method_name}:")
                        print(f"    MARKET -> {market_result}")
                        print(f"    LIMIT -> {limit_result}")
                        
                        conversion_working = True
                        break
                    except Exception as e:
                        print(f"  {method_name} failed: {e}")
            
            if not conversion_working:
                print(f"  No order type conversion method found")
            
            # Test status parsing methods
            status_methods = [
                '_parse_order_status', '_parse_order_type'
            ]
            
            status_parsing = False
            for method_name in status_methods:
                if hasattr(exchange, method_name):
                    print(f"  {method_name}: exists")
                    status_parsing = True
            
            enum_tests[exchange_name] = conversion_working or status_parsing
            print(f"  Enum handling: {'✅' if enum_tests[exchange_name] else '❌'}")
            
        except Exception as e:
            print(f"  Enum test failed: ❌ ({e})")
            enum_tests[exchange_name] = False
    
    all_enums_ok = all(enum_tests.values())
    print(f"\nAll enum handling working: {'✅' if all_enums_ok else '❌'}")
    
    return all_enums_ok

def run_exchange_validation():
    """Run all exchange validation tests"""
    
    print("🔍 Exchange Integration Validation")
    print("=" * 70)
    
    tests = [
        ("Exchange Factory", test_exchange_factory),
        ("Interface Compliance", test_interface_compliance),
        ("Configuration Handling", test_configuration_handling),
        ("Symbol Normalization", test_symbol_normalization),
        ("Order Types & Enums", test_order_types_and_enums)
    ]
    
    # Note: Connection test requires async, running separately
    
    results = {}
    for test_name, test_func in tests:
        print(f"\n" + "="*70)
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"Test {test_name} failed with exception: {e}")
            results[test_name] = False
    
    # Run async connection test
    print(f"\n" + "="*70)
    try:
        connection_result = asyncio.run(test_connection_methods())
        results["Connection Methods"] = connection_result
    except Exception as e:
        print(f"Connection test failed with exception: {e}")
        results["Connection Methods"] = False
    
    # Summary
    print(f"\n" + "="*70)
    print("📊 EXCHANGE VALIDATION SUMMARY")
    print("="*70)
    
    total_tests = len(results)
    passed_tests = sum(1 for result in results.values() if result)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name:25}: {status}")
    
    success_rate = (passed_tests / total_tests) * 100
    print(f"\nSuccess Rate: {success_rate:.1f}% ({passed_tests}/{total_tests})")
    
    if success_rate >= 85:
        print("🎉 EXCHANGE INTEGRATION VALIDATION PASSED")
        print("✅ All exchanges are properly implemented")
        print("✅ Multi-exchange trading is ready")
        return True
    else:
        print("🚨 EXCHANGE INTEGRATION ISSUES DETECTED")
        print("❌ Some exchanges need fixes before production use")
        return False

if __name__ == "__main__":
    success = run_exchange_validation()
    if not success:
        print("\n🚨 Fix exchange implementation issues before proceeding!")
        sys.exit(1)
    else:
        print("\n🎉 Exchange integration validation complete - system ready for multi-exchange trading")