# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

TickerML is an intelligent crypto paper trading bot using reinforcement learning and market microstructure analysis. It operates across two platforms:
- **Raspberry Pi**: Real-time order book collection, paper trading engine, risk management, and dashboard
- **PC**: RL training pipeline, feature engineering, backtesting, and model optimization

**Supported exchanges**: Binance.US, Coinbase, Kraken, KuCoin
**Trading pairs**: BTC/USDT, ETH/USDT, BTC/USD, ETH/USD
**Trading capabilities**: Automated buy/sell/hold decisions, position sizing, risk management, multi-exchange arbitrage.

## Architecture

```
[Raspberry Pi] -> WebSocket streams -> Real-time order books -> TimescaleDB
[PC] -> RL training (PPO/A2C) -> Multi-task Transformer -> ONNX export
[Raspberry Pi] -> Paper trading engine -> Portfolio management -> Live P&L dashboard
```

## Common Development Commands

### Setup and Environment
```bash
# Setup test environment
python scripts/setup_test_env.py

# Platform-specific setup (auto-detects Pi vs PC)
chmod +x scripts/setup.sh && ./scripts/setup.sh
```

### Testing and Validation
```bash
# Run comprehensive test suite
python tests/test_data_collection.py

# Test individual components
python tests/test_news_harvest.py
python tests/test_sentiment.py
python tests/test_sentiment_no_fallback.py
python tests/test_features.py
python tests/test_pipeline.py
python tests/test_newsapi.py

# Check system status
python tests/test_summary.py

# Verify Qwen 3 configuration
python scripts/verify_qwen_config.py
```

### Data Collection & Paper Trading (Raspberry Pi)
```bash
# Real-time order book collection
python raspberry_pi/harvest.py

# Start paper trading engine
python raspberry_pi/paper_trader.py

# News and sentiment analysis
python raspberry_pi/news_harvest.py

# Check collected data
sqlite3 data/db/crypto_data.db "SELECT symbol, COUNT(*) FROM ohlcv GROUP BY symbol;"
sqlite3 data/db/crypto_data.db "SELECT * FROM ohlcv ORDER BY timestamp DESC LIMIT 5;"

# Monitor trading performance
tail -f logs/paper_trader.log
```

### RL Training & Model Development (PC)
```bash
# Market microstructure feature engineering
python pc/features.py

# Transformer model training (multi-task)
python pc/train.py --epochs 100 --batch_size 32 --learning_rate 0.001

# Export and quantize model
python pc/export_quantize.py
```

### Trading Dashboard & Monitoring (Raspberry Pi)
```bash
# Real-time trading decisions
python raspberry_pi/infer.py

# Start trading dashboard
python raspberry_pi/dashboard.py
# Access at http://localhost:5000 for live P&L tracking

# Live inference for trading decisions
python raspberry_pi/infer.py
```

### Monitoring
```bash
# View logs
tail -f logs/harvest.log
tail -f logs/paper_trader.log
tail -f logs/news_harvest.log
tail -f logs/infer.log

# Monitor data collection
watch "sqlite3 data/db/crypto_data.db 'SELECT symbol, COUNT(*) FROM ohlcv GROUP BY symbol;'"

# Check recent data
watch "sqlite3 data/db/crypto_data.db 'SELECT * FROM ohlcv ORDER BY timestamp DESC LIMIT 5;'"
```

## Code Architecture

### Core Components

**Real-time Data Collection** (`raspberry_pi/harvest.py`):
- Multi-exchange WebSocket streams (Binance.US, Coinbase, Kraken, KuCoin)
- Order book snapshots (5-10 second intervals, 20-50 levels deep)
- Trade flow analysis and market microstructure features
- Millisecond precision timestamps and graceful reconnection

**Paper Trading Engine** (`raspberry_pi/paper_trader.py`):
- Portfolio management with virtual $10,000 starting balance
- Real-time order execution simulation with slippage modeling
- Position sizing algorithms (0-25% of portfolio per trade)
- Risk management with stop-loss, take-profit, and correlation limits

**News & Sentiment** (`raspberry_pi/news_harvest.py`):
- NewsAPI integration with crypto keyword filtering
- Qwen 3 LLM sentiment analysis via Ollama
- Market regime detection and sentiment aggregation
- Fallback to keyword-based sentiment scoring

**Feature Engineering** (`pc/features.py`):
- Market microstructure features: order imbalance, bid-ask spread dynamics, order flow toxicity
- Technical indicators: SMA, EMA, MACD, RSI, Bollinger Bands, Stochastic, Williams %R, ATR, CCI, MFI, ROC
- Portfolio-aware features: current P&L, time in position, portfolio heat, correlation analysis
- Multi-exchange features: cross-exchange arbitrage signals, relative spread analysis
- Sentiment integration: hourly news sentiment scores and market regime indicators

**RL Training Pipeline** (`pc/rl_trainer.py`):
- PPO/A2C algorithms with experience replay buffer
- State space: order book features, technical indicators, portfolio state, sentiment
- Action space: buy/sell/hold with continuous position sizing
- Reward function: risk-adjusted returns with drawdown penalties
- Multi-environment training for robustness

**Multi-task Transformer Training** (`pc/train.py`):
- TradingTransformer architecture (d_model=128, n_heads=4, n_layers=6)
- Four output heads: price prediction, trading action, position sizing, risk assessment
- Attention mechanism analysis and visualization
- Mixed precision training with early stopping and checkpoint management

**Trading Decision Engine** (`raspberry_pi/infer.py`):
- Sub-second inference using quantized ONNX models
- Real-time feature calculation from order book data
- Multi-task output: trading signals, position sizing, risk scores
- Confidence thresholding and decision logging
- Ensemble model voting for robustness

**Trading Dashboard** (`raspberry_pi/dashboard.py`):
- Real-time P&L tracking and portfolio visualization
- Live order book depth charts and trade execution logs
- Performance metrics: Sharpe ratio, drawdown, win rate, profit factor
- Attention weight visualizations and feature importance analysis
- Risk monitoring: position correlation, exposure limits, circuit breakers

### Database Schema

**Trading Data** (`data/db/crypto_data.db`):
- `ohlcv` table: timestamp, symbol, open, high, low, close, volume
- `predictions` table: timestamp, symbol, prediction_5m, prediction_10m, prediction_30m, direction
- `sentiment_data` table: timestamp, news_count, sentiment_score, market_regime

**News Data** (integrated into main database):
- `news_articles` table: timestamp, title, content, url, sentiment_score
- `news_sentiment_hourly` table: hourly sentiment aggregates

### Configuration

All settings in `config/config.yaml`:
- API endpoints and symbols
- Model hyperparameters
- Feature engineering settings
- Dashboard preferences
- Cron schedules
- Ollama host configuration

### Dependencies

**Raspberry Pi**: requests, pandas, numpy, flask, onnxruntime, pyyaml, coloredlogs
**PC**: torch, torchvision, scikit-learn, ta, onnx, matplotlib, seaborn, transformers, ollama

## Development Patterns

### Error Handling
- Graceful API fallbacks (Binance.US -> CoinGecko)
- Database connection retry logic
- SIGINT/SIGTERM signal handling for clean shutdown

### Data Validation
- Price data sanity checks (positive values, reasonable ranges)
- Sentiment score normalization (-1 to 1)
- Feature scaling and outlier detection

### Performance Optimizations
- ONNX quantization for edge deployment
- Batch processing for feature engineering
- Efficient SQLite indexing and queries
- Memory-mapped file operations for large datasets

### Cron Job Setup
```bash
# Minute-level data collection
* * * * * cd /path/to/TickerML && python raspberry_pi/harvest.py >> logs/harvest.log 2>&1

# News collection every 15 minutes
*/15 * * * * cd /path/to/TickerML && python raspberry_pi/news_harvest.py >> logs/news_harvest.log 2>&1

# Daily ETL export at midnight
0 0 * * * cd /path/to/TickerML && python raspberry_pi/export_etl.py >> logs/etl.log 2>&1

# Inference every 5 minutes
*/5 * * * * cd /path/to/TickerML && python raspberry_pi/infer.py >> logs/infer.log 2>&1
```

## Key Technical Details

### Transformer Model
- Positional encoding for time-series data
- Multi-head attention with 60-minute sequence length
- Dual output heads: regression (price) + classification (direction)
- Dropout regularization and layer normalization

### Sentiment Analysis
- Qwen 3 4B model via Ollama for contextual understanding
- Real-time crypto news processing from multiple sources
- Keyword fallback system for reliability
- Hourly aggregation for reduced noise

### Edge Deployment
- ONNX Runtime optimization for Raspberry Pi
- INT8 quantization reducing model size by ~75%
- Efficient feature computation using vectorized operations
- Memory usage monitoring and garbage collection

## Current Implementation Status

**✅ Completed Components:**
- Real-time OHLCV data harvesting from Binance.US API
- Fallback to CoinGecko API for redundancy
- SQLite database with automatic backfill (up to 7 days)
- News collection with NewsAPI integration
- Qwen 3 sentiment analysis via Ollama
- Transformer model training with multi-task outputs
- ONNX model export and quantization
- Web dashboard with real-time price visualization
- Comprehensive test suite

**🚧 In Progress:**
- Paper trading engine implementation
- Reinforcement learning training pipeline
- Multi-exchange order book collection
- Real-time trading decision system
- Portfolio management and risk controls

## Troubleshooting

**API Issues**: Check internet connectivity, verify API endpoints, monitor rate limits
**Database Errors**: Verify file permissions, check disk space, validate schema
**Model Training**: Monitor GPU memory, adjust batch size, check data quality
**Inference Performance**: Profile ONNX model, optimize feature computation, monitor memory usage
**Sentiment Analysis**: Ensure Ollama is running (`ollama serve`), verify model availability (`ollama list`)
**Qwen Model**: If missing, install with `ollama pull qwen3:4b` (or `qwen3:1b` for lighter systems)

## File Locations

**Configuration**: `config/config.yaml`
**Database**: `data/db/crypto_data.db`
**Models**: `models/checkpoints/`, `models/onnx/`
**Data Exports**: `data/dumps/`
**Logs**: `logs/`