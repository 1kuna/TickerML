# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

TickerML is an intelligent crypto paper trading bot using reinforcement learning and market microstructure analysis. It operates across two platforms:
- **Raspberry Pi**: Real-time order book collection, paper trading engine, risk management, and dashboard
- **PC**: RL training pipeline, feature engineering, backtesting, and model optimization

**Supported exchanges**: Binance.US, Coinbase, Kraken, KuCoin
**Trading pairs**: BTC/USDT, ETH/USDT, BTC/USD, ETH/USD
**Trading capabilities**: Automated buy/sell/hold decisions, position sizing, risk management, multi-exchange arbitrage

## Architecture

```
[Raspberry Pi] -> WebSocket streams -> Order books -> Kafka -> Feature generation
[PC] -> Decision Transformer (frozen backbone) -> Offline RL -> ONNX export
[Raspberry Pi] -> Paper trading engine -> Risk management -> Live P&L dashboard
```

## Common Development Commands

### Setup and Environment
```bash
# Platform-specific setup (auto-detects Pi vs PC)
chmod +x scripts/setup.sh && ./scripts/setup.sh

# Setup test environment
python scripts/setup_test_env.py

# Setup Kafka infrastructure
python scripts/setup_kafka.py

# Verify Qwen 3 configuration
python scripts/verify_qwen_config.py
```

### Testing
```bash
# Run comprehensive test suite
python tests/test_new_components.py
python tests/test_enhanced_trading.py
python tests/test_decision_transformer.py
python tests/test_orderbook_collector.py

# Test individual components
python tests/test_data_collection.py
python tests/test_news_harvest.py
python tests/test_sentiment.py
python tests/test_features.py
python tests/test_pipeline.py

# Check system status
python tests/test_summary.py
```

### Data Collection & Paper Trading (Raspberry Pi)
```bash
# Real-time order book collection
python raspberry_pi/harvest.py

# Start paper trading engine
python raspberry_pi/paper_trader.py

# Kafka event streaming
python raspberry_pi/kafka_producers/orderbook_producer.py
python raspberry_pi/kafka_producers/trade_producer.py

# News and sentiment analysis
python raspberry_pi/news_harvest.py

# Check collected data
sqlite3 data/db/crypto_data.db "SELECT symbol, COUNT(*) FROM ohlcv GROUP BY symbol;"

# Monitor trading performance
tail -f logs/paper_trader.log
```

### RL Training & Model Development (PC)
```bash
# Market microstructure feature engineering
python pc/features.py
python pc/enhanced_features.py

# Decision Transformer training (frozen backbone)
python pc/train_decision_transformer.py

# Offline RL training (30-day quarantine)
python pc/offline_rl_trainer.py

# Multi-task Transformer training
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
# Access at http://localhost:5005

# Kafka consumers
python raspberry_pi/kafka_consumers/feature_consumer.py
python raspberry_pi/kafka_consumers/trading_consumer.py
```

### Database Migration
```bash
# Migrate SQLite to TimescaleDB
python scripts/migrate_to_timescale.py
```

### Monitoring
```bash
# View logs
tail -f logs/harvest.log
tail -f logs/paper_trader.log
tail -f logs/kafka_producer_*.log

# Monitor Kafka lag
kafka-consumer-groups --bootstrap-server localhost:9092 --describe --group feature-generator

# Check recent data
watch "sqlite3 data/db/crypto_data.db 'SELECT * FROM ohlcv ORDER BY timestamp DESC LIMIT 5;'"
```

## Code Architecture

### Core Components

**Event-Driven Data Collection**:
- `raspberry_pi/orderbook_collector.py`: WebSocket L2/L3 order book with incremental updates
- `raspberry_pi/trade_stream.py`: Real-time trade flow analysis
- `raspberry_pi/kafka_producers/`: Event streaming producers for order books, trades, news
- `raspberry_pi/kafka_consumers/`: Real-time feature generation and trading decisions

**Advanced Paper Trading**:
- `raspberry_pi/paper_trader.py`: Portfolio management with $10,000 starting balance
- `raspberry_pi/execution_simulator.py`: FIFO queue modeling, partial fills, toxic fill detection
- `raspberry_pi/risk_manager.py`: Correlation-based risk, volatility regimes, circuit breakers
- `raspberry_pi/funding_monitor.py`: Perpetuals funding rate tracking (critical for profitability)

**Model Architecture**:
- `pc/models/decision_transformer.py`: Frozen backbone with trainable action/value heads
- `pc/offline_rl_trainer.py`: 30-day quarantine rule, combinatorial purged CV
- `pc/enhanced_features.py`: Order book imbalance, microprice, VWAP deviations
- BF16 mixed precision (NOT FP16), Flash Attention optimization for RTX 4090

**Production Infrastructure**:
- Kafka event streaming with sub-second latency
- TimescaleDB with hypertables and retention policies
- Comprehensive monitoring and alerting
- GPU optimization with NCCL_P2P_DISABLE=1

### Database Schema

**TimescaleDB Tables** (production):
```sql
-- Order books with microsecond precision
CREATE TABLE order_books (
    timestamp TIMESTAMPTZ NOT NULL,
    exchange VARCHAR(20) NOT NULL,
    symbol VARCHAR(20) NOT NULL,
    bids JSONB NOT NULL,  -- [[price, quantity, queue_position], ...]
    asks JSONB NOT NULL,
    mid_price NUMERIC(20,8),
    spread NUMERIC(20,8),
    imbalance NUMERIC(10,4)
);

-- Portfolio state tracking
CREATE TABLE portfolio_state (
    timestamp TIMESTAMPTZ PRIMARY KEY,
    cash_balance NUMERIC(20,8),
    total_value NUMERIC(20,8),
    positions JSONB,  -- {symbol: {quantity, entry_price, pnl}}
    daily_pnl NUMERIC(20,8),
    max_drawdown NUMERIC(10,4),
    sharpe_ratio NUMERIC(10,4)
);
```

### Configuration

**Key Configuration Files**:
- `config/config.yaml`: Main configuration
- `config/kafka_config.yaml`: Event streaming topics and consumers
- `config/risk_limits.yaml`: Max position 25%, max drawdown 25%
- `config/model_config.yaml`: Decision Transformer, BF16, GPU settings
- `config/monitoring_config.yaml`: Alerts, dashboards, Kafka lag monitoring

**Critical Risk Parameters**:
```python
# Fractional Kelly criterion
kelly_fraction = 0.25  # NEVER use full Kelly
max_position_size = 0.25  # Maximum 25% portfolio per position
max_drawdown = 0.25  # Emergency stop at 25% loss
correlation_threshold = 0.7  # Reduce position size if correlated
```

## Critical Technical Details

### Decision Transformer
- Frozen pre-trained backbone (prevents catastrophic forgetting)
- Only last 2 transformer layers trainable
- Return-to-go conditioning for target-driven decisions
- Multi-task heads: action, position sizing, risk assessment
- Weekly refresh of outer layers, monthly full retrain

### Execution Realism
- FIFO queue position modeling (position >10 = toxic fill)
- Exchange-specific latency: Binance 50-100ms, Coinbase 100-200ms
- Partial fill simulation with progressive execution
- Market impact modeling based on order book depth
- Adverse selection always assumed

### Market Microstructure Features
1. **Order book imbalance** - Strongest <1min predictor
2. **Microprice** - Better than mid for execution
3. **Trade flow imbalance** - Momentum indicator
4. **Cross-exchange spreads** - Arbitrage opportunities
5. **VWAP deviations** - Mean reversion signals
6. **Queue position estimates** - Critical for limit orders

### Critical Rules
- **30-Day Quarantine**: NEVER train on data from last 30 days
- **Paper Trading ≠ Training**: Paper results validate only, never update weights
- **Event Synchronization**: Order books and trades must be replayed by timestamp
- **Funding Rates**: Can be up to 1% daily on perpetuals
- **BF16 > FP16**: Financial data can overflow FP16

## Current Implementation Status

**✅ Completed**:
- Event-driven Kafka infrastructure
- Decision Transformer with frozen backbone
- Advanced execution simulation (FIFO queues)
- Correlation-based risk management
- WebSocket order book collection
- TimescaleDB migration script
- Production monitoring configuration

**🚧 In Progress**:
- Offline RL trainer implementation
- Multi-exchange abstractions
- Cross-exchange arbitrage
- Integration testing

## Troubleshooting

**Kafka Issues**: Check `kafka-server-start` logs, ensure topics created
**WebSocket Disconnects**: Check exchange rate limits, implement exponential backoff
**Model Training**: Monitor GPU memory, use gradient accumulation if needed
**TimescaleDB**: Ensure proper indexes, monitor chunk size
**Sentiment Analysis**: Ensure Ollama running (`ollama serve`), model available (`ollama list`)

## Development Patterns

### Error Handling
- Graceful WebSocket reconnection with exponential backoff
- Kafka producer retries with dead letter queue
- Circuit breakers for anomalous market conditions
- Database connection pooling with retry logic

### Performance Optimization
- ONNX INT8 quantization for Raspberry Pi inference
- Batch Kafka message processing
- TimescaleDB chunk optimization
- Flash Attention for transformer models

### Testing Strategy
- Unit tests for each component
- Integration tests with mock Kafka
- End-to-end backtesting on historical data
- Paper trading validation before any capital deployment

## File Locations

**Models**: `models/checkpoints/`, `models/onnx/`
**Logs**: `logs/` (rotated daily)
**Data**: `data/db/` (SQLite), future: TimescaleDB
**Kafka**: `data/kafka/` (topics and offsets)
**Config**: `config/*.yaml`