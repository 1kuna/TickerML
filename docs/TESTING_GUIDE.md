# TickerML Paper Trading Bot Testing Guide

This guide will help you test the paper trading bot system step by step, including data collection, trading engine, and RL training components.

## Quick Start

### 1. **Setup Test Environment**
```bash
# Run the setup script first
python scripts/setup_test_env.py

# Or manually create directories
mkdir -p data/db data/dumps data/features logs models/checkpoints models/onnx
```

### 2. **Run All Tests**
```bash
# Run comprehensive test suite
python tests/test_data_collection.py
python tests/test_paper_trader.py
python tests/test_rl_training.py
python tests/test_sentiment.py
```

This will run comprehensive tests covering data collection, paper trading engine, RL training, and sentiment analysis.

## Individual Test Components

### **Test 1: API Connection** 🌐
Tests connectivity to Binance API without requiring authentication.

**Manual test:**
```bash
curl "https://api.binance.us/api/v3/ping"
curl "https://api.binance.us/api/v3/klines?symbol=BTCUSD&interval=1m&limit=1"
curl "https://api.coingecko.com/api/v3/simple/price?ids=bitcoin,ethereum&vs_currencies=usd"
```

**What it checks:**
- ✅ Internet connectivity
- ✅ Binance.US API availability (primary)
- ✅ CoinGecko API availability (fallback)
- ✅ Klines endpoint functionality
- ✅ Data format validation

### **Test 2: Database Setup** 🗄️
Tests SQLite database creation and basic operations.

**Manual test:**
```bash
python raspberry_pi/harvest.py  # This will create the database
sqlite3 data/db/crypto_ohlcv.db ".tables"  # Check if tables exist
```

**What it checks:**
- ✅ Database file creation
- ✅ Table schema creation
- ✅ Index creation
- ✅ Data insertion/retrieval

### **Test 3: Data Harvesting** 📊
Tests actual data collection from Binance API.

**Manual test:**
```bash
# Single harvest run
python raspberry_pi/harvest.py

# Check what was collected
sqlite3 data/db/crypto_ohlcv.db "SELECT COUNT(*) FROM ohlcv;"
```

**What it checks:**
- ✅ API data fetching for both BTC and ETH (Binance.US primary)
- ✅ Fallback to CoinGecko when needed
- ✅ Data parsing and validation
- ✅ Database storage
- ✅ Duplicate handling

### **Test 4: Data Export** 📁
Tests CSV export functionality for data analysis.

**Manual test:**
```bash
# Run export (requires existing data)
python raspberry_pi/export_etl.py

# Check generated files
ls -la data/dumps/
```

**What it checks:**
- ✅ CSV file generation
- ✅ Data formatting
- ✅ File naming conventions
- ✅ Data completeness

### **Test 5: Database Queries** 🔍
Tests query performance and data integrity.

**Manual test:**
```bash
sqlite3 data/db/crypto_ohlcv.db << EOF
SELECT symbol, COUNT(*) as records, 
       MIN(datetime(timestamp/1000, 'unixepoch')) as earliest,
       MAX(datetime(timestamp/1000, 'unixepoch')) as latest
FROM ohlcv 
GROUP BY symbol;
EOF
```

**What it checks:**
- ✅ Query performance
- ✅ Data distribution
- ✅ Time range coverage
- ✅ Symbol-specific data

### **Test 6: Dashboard Data** 📈
Tests data availability for the web dashboard.

**Manual test:**
```bash
# Start dashboard (requires data)
python raspberry_pi/dashboard.py

# Visit http://localhost:5000 in browser
```

**What it checks:**
- ✅ 24-hour data availability
- ✅ Price statistics calculation
- ✅ Data formatting for charts
- ✅ API endpoint functionality

## Continuous Testing

### **Run Live Collection Test**
```bash
# Test continuous collection for 5 minutes
python tests/test_data_collection.py
# When prompted, choose 'y' for continuous test
```

This simulates the actual cron job behavior by collecting data every minute.

## Paper Trading Bot Tests

### **Test 7: Paper Trading Engine** 💰
Tests the virtual trading engine with portfolio management.

**Manual test:**
```bash
# Test paper trading initialization
python raspberry_pi/paper_trader.py --test

# Check portfolio state
sqlite3 data/db/crypto_ohlcv.db "SELECT * FROM portfolio_state ORDER BY timestamp DESC LIMIT 1;"
```

**What it checks:**
- ✅ Virtual portfolio initialization ($10,000 starting balance)
- ✅ Order execution simulation with slippage
- ✅ Position sizing and risk management
- ✅ Stop-loss and take-profit mechanisms
- ✅ Transaction cost calculation
- ✅ Portfolio performance tracking

### **Test 8: RL Training Pipeline** 🧠
Tests the reinforcement learning training system.

**Manual test:**
```bash
# Test RL environment setup
python pc/rl_trainer.py --test --episodes 10

# Check experience replay buffer
python -c "
import pickle
with open('models/rl/experience_buffer.pkl', 'rb') as f:
    buffer = pickle.load(f)
print(f'Experience buffer size: {len(buffer)}')
"
```

**What it checks:**
- ✅ RL environment initialization
- ✅ State space construction (order book + technical + portfolio features)
- ✅ Action space validation (buy/sell/hold + position sizing)
- ✅ Reward function calculation (risk-adjusted returns)
- ✅ Experience replay buffer functionality
- ✅ Model training convergence

### **Test 9: Trading Decision Engine** ⚡
Tests real-time inference and trading decisions.

**Manual test:**
```bash
# Test trading inference
python raspberry_pi/infer.py --test

# Check decision logs
tail -f logs/trading_decisions.log
```

**What it checks:**
- ✅ Sub-second inference performance (<50ms)
- ✅ Feature calculation from order book data
- ✅ Multi-task model output (price + action + position + risk)
- ✅ Confidence thresholding
- ✅ Ensemble model voting
- ✅ Decision logging and audit trail

### **Test 10: Risk Management** 🛡️
Tests portfolio risk controls and circuit breakers.

**Manual test:**
```bash
# Test risk management system
python raspberry_pi/risk_monitor.py --test

# Simulate high correlation scenario
python tests/test_risk_management.py --scenario high_correlation
```

**What it checks:**
- ✅ Position correlation monitoring
- ✅ Drawdown calculation and limits
- ✅ Exposure limits by asset
- ✅ Circuit breaker activation
- ✅ Risk alerts and notifications
- ✅ VaR (Value at Risk) calculation

### **Test 11: Multi-Exchange Data** 🔄
Tests real-time data from multiple exchanges.

**Manual test:**
```bash
# Test multi-exchange WebSocket connections
python raspberry_pi/harvest.py --multi-exchange --test

# Check order book synchronization
sqlite3 data/db/crypto_ohlcv.db "
SELECT exchange, symbol, COUNT(*) as snapshots,
       MAX(timestamp) as latest_timestamp
FROM order_books 
GROUP BY exchange, symbol;
"
```

**What it checks:**
- ✅ WebSocket connection stability across exchanges
- ✅ Order book synchronization
- ✅ Cross-exchange arbitrage detection
- ✅ Data quality and completeness
- ✅ Latency monitoring
- ✅ Reconnection handling

### **Test 12: Performance Metrics** 📈
Tests trading performance calculation and reporting.

**Manual test:**
```bash
# Generate performance report
python raspberry_pi/performance_report.py

# Check key metrics
python -c "
import sqlite3
conn = sqlite3.connect('data/db/crypto_ohlcv.db')
cursor = conn.execute('SELECT * FROM performance_metrics ORDER BY timestamp DESC LIMIT 1')
metrics = cursor.fetchone()
print(f'Sharpe Ratio: {metrics[1]:.2f}')
print(f'Max Drawdown: {metrics[2]:.2f}%')
print(f'Win Rate: {metrics[3]:.2f}%')
conn.close()
"
```

**What it checks:**
- ✅ Sharpe ratio calculation
- ✅ Maximum drawdown tracking
- ✅ Win rate and profit factor
- ✅ Risk-adjusted returns
- ✅ Benchmark comparison (buy & hold)
- ✅ Performance attribution analysis

## Advanced Testing

### **Backtesting Validation**
```bash
# Run comprehensive backtesting
python pc/backtest.py --start-date 2024-01-01 --end-date 2024-12-31

# Walk-forward analysis
python pc/backtest.py --walk-forward --train-days 90 --test-days 30
```

### **Stress Testing**
```bash
# Test extreme market conditions
python tests/test_stress_scenarios.py --scenario black_swan
python tests/test_stress_scenarios.py --scenario flash_crash
python tests/test_stress_scenarios.py --scenario high_volatility
```

### **Model Validation**
```bash
# Test model attention mechanisms
python tests/test_attention_analysis.py

# Validate feature importance
python tests/test_feature_importance.py

# Check for overfitting
python tests/test_overfitting_detection.py
```

## Performance Benchmarks

### **Expected Results:**
- **Data Collection**: WebSocket latency < 10ms, order book processing < 20ms
- **Trading Decisions**: Inference time < 50ms, decision frequency = 1 Hz
- **Risk Management**: Risk calculation < 5ms, alert generation < 100ms
- **Paper Trading**: Order execution simulation < 1ms per trade

### **Trading Performance Targets:**
- **Sharpe Ratio**: > 1.5
- **Maximum Drawdown**: < 15%
- **Win Rate**: > 55%
- **Profit Factor**: > 1.5
- **System Uptime**: > 99.9%

## Troubleshooting Common Issues

### **❌ API Connection Failed**
```bash
# Check internet connection
ping google.com

# Test Binance API directly
curl -v "https://api.binance.com/api/v3/ping"

# Check if behind firewall/proxy
```

### **❌ Database Errors**
```bash
# Check file permissions
ls -la data/db/

# Manually create database
python -c "
import sqlite3
from pathlib import Path
Path('data/db').mkdir(parents=True, exist_ok=True)
conn = sqlite3.connect('data/db/crypto_ohlcv.db')
conn.close()
print('Database created')
"
```

### **❌ Import Errors**
```bash
# Install missing dependencies
pip install requests pandas

# For Raspberry Pi specific packages
pip install -r raspberry_pi/requirements.txt
```

### **❌ No Data Collected**
```bash
# Check if harvest script runs without errors
python raspberry_pi/harvest.py

# Verify data was stored
sqlite3 data/db/crypto_ohlcv.db "SELECT * FROM ohlcv ORDER BY timestamp DESC LIMIT 5;"

# Check logs
tail -f logs/harvest.log  # If logging to file
```

## Performance Benchmarks

### **Expected Results:**
- **API Response Time:** < 2 seconds
- **Database Insert:** < 100ms per record
- **CSV Export:** < 5 seconds for 24h data
- **Dashboard Load:** < 3 seconds

### **Data Collection Rates:**
- **Binance.US API Limit:** 1200 requests/minute
- **CoinGecko API Limit:** 10-50 requests/minute (free tier)
- **Our Usage:** ~2 requests/minute (well within limits)
- **Storage:** ~1MB per day for both symbols

## Setting Up Automated Collection

### **1. Test Manual Collection First**
```bash
# Ensure single run works
python raspberry_pi/harvest.py
```

### **2. Set Up Cron Job**
```bash
# Edit crontab
crontab -e

# Add this line (adjust path to your project):
* * * * * cd /path/to/TickerML && python raspberry_pi/harvest.py >> logs/harvest.log 2>&1
```

### **3. Monitor Collection**
```bash
# Watch logs in real-time
tail -f logs/harvest.log

# Check data growth
watch "sqlite3 data/db/crypto_ohlcv.db 'SELECT symbol, COUNT(*) FROM ohlcv GROUP BY symbol;'"
```

## Advanced Testing

### **Load Testing**
```bash
# Test multiple rapid collections
for i in {1..10}; do
    python raspberry_pi/harvest.py
    sleep 5
done
```

### **Data Validation**
```bash
# Check for data gaps
python -c "
import sqlite3
import pandas as pd
from datetime import datetime, timedelta

conn = sqlite3.connect('data/db/crypto_ohlcv.db')
df = pd.read_sql('SELECT * FROM ohlcv ORDER BY timestamp', conn)
df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')

# Check for gaps > 2 minutes
df['time_diff'] = df['datetime'].diff()
gaps = df[df['time_diff'] > pd.Timedelta(minutes=2)]
print(f'Found {len(gaps)} gaps > 2 minutes')
conn.close()
"
```

### **Memory Usage**
```bash
# Monitor memory during collection
python -c "
import psutil
import time
import subprocess

print('Starting memory monitoring...')
for i in range(5):
    # Run harvest
    subprocess.run(['python', 'raspberry_pi/harvest.py'])
    
    # Check memory
    process = psutil.Process()
    memory_mb = process.memory_info().rss / 1024 / 1024
    print(f'Cycle {i+1}: Memory usage: {memory_mb:.1f} MB')
    
    time.sleep(60)
"
```

## Success Criteria

Your data collection pipeline is ready when:

- ✅ All 6 tests pass
- ✅ Data is collected every minute for at least 1 hour
- ✅ No gaps in data collection
- ✅ CSV exports work correctly
- ✅ Dashboard displays data properly
- ✅ Memory usage remains stable

## Next Steps

Once data collection is working:

1. **Set up cron jobs** for automated collection
2. **Configure the dashboard** for monitoring
3. **Collect data for 24+ hours** before training models
4. **Set up the PC environment** for feature engineering and training

## Getting Help

If tests fail:
1. Check the error messages in the test output
2. Review the troubleshooting section above
3. Ensure all dependencies are installed
4. Verify internet connectivity and API access
5. Check file permissions and directory structure 