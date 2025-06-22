# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

TickerML is an intelligent crypto paper trading bot using reinforcement learning and market microstructure analysis. It operates as a full-stack application with three main components:
- **Backend (FastAPI)**: REST API, WebSocket streaming, authentication, and business logic
- **Frontend (React)**: Modern dashboard for monitoring trading, portfolio, and system health  
- **Trading Engine (Raspberry Pi/PC)**: Real-time data collection, paper trading, RL training pipeline

**Supported exchanges**: Binance.US, Coinbase, Kraken, KuCoin
**Trading pairs**: BTC/USDT, ETH/USDT, BTC/USD, ETH/USD
**Trading capabilities**: Automated buy/sell/hold decisions, position sizing, risk management, multi-exchange arbitrage

## Architecture

```
[Frontend (React)] <--> [Backend (FastAPI)] <--> [Redis/PostgreSQL]
                              |
[Trading Engine] -> WebSocket streams -> Order books -> Kafka -> Feature generation
[PC] -> Decision Transformer (frozen backbone) -> Offline RL -> ONNX export
[Trading Engine] -> Paper trading engine -> Risk management -> Live P&L dashboard
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

# Setup production database (TimescaleDB)
python scripts/setup_production_db.py

# Verify Qwen 3 configuration
python scripts/verify_qwen_config.py
```

### Full-Stack Development

#### **Backend (FastAPI)**
```bash
# Install backend dependencies
cd backend && pip install -r requirements.txt

# Start development server
cd backend && uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# Run backend tests
cd backend && pytest

# Check API documentation
# http://localhost:8000/docs (Swagger UI)
# http://localhost:8000/redoc (ReDoc)
```

#### **Frontend (React)**
```bash
# Install frontend dependencies
cd frontend && npm install

# Start development server
cd frontend && npm start
# Access at http://localhost:3000

# Build for production
cd frontend && npm run build

# Type checking
cd frontend && npx tsc --noEmit
```

#### **Production Deployment**
```bash
# Traditional server deployment
cd deployment/scripts && chmod +x deploy.sh && sudo ./deploy.sh

# Docker deployment
cd deployment/docker && docker-compose --profile production up -d

# Check deployment status
sudo systemctl status tickerml-dashboard
sudo systemctl status tickerml-data-collector
sudo systemctl status tickerml-paper-trader
```

### Testing
```bash
# Run comprehensive test suite
python tests/test_new_components.py
python tests/test_enhanced_trading.py
python tests/test_decision_transformer.py
python tests/test_orderbook_collector.py

# Core system tests
python tests/test_data_collection.py
python tests/test_paper_trading.py
python tests/test_risk_manager.py
python tests/test_execution_sim.py
python tests/test_arbitrage.py

# Feature and ML tests
python tests/test_features.py
python tests/test_news_harvest.py
python tests/test_sentiment.py

# Integration and performance tests
python tests/test_integration.py
python tests/test_performance.py
python tests/test_system_health.py

# Check system status and summary
python tests/test_summary.py
```

### Data Collection & Trading Engine
```bash
# Real-time order book collection with microstructure analysis
python raspberry_pi/orderbook_collector.py

# Trade stream monitoring and analysis
python raspberry_pi/trade_stream.py

# Comprehensive data harvesting (legacy)
python raspberry_pi/harvest.py

# Paper trading engine with portfolio management
python raspberry_pi/paper_trader.py

# Advanced execution simulation with FIFO queues
python raspberry_pi/execution_simulator.py

# Risk management with correlation analysis
python raspberry_pi/risk_manager.py

# Multi-exchange arbitrage monitoring
python raspberry_pi/arbitrage_monitor.py

# Funding rate tracking for perpetuals
python raspberry_pi/funding_monitor.py

# Kafka event streaming
python raspberry_pi/kafka_producers/orderbook_producer.py
python raspberry_pi/kafka_producers/trade_producer.py
python raspberry_pi/kafka_producers/news_producer.py

# News and sentiment analysis
python raspberry_pi/news_harvest.py

# Event synchronization and data validation
python raspberry_pi/event_synchronizer.py
python raspberry_pi/data_validator.py

# Check collected data
sqlite3 data/db/crypto_data.db "SELECT symbol, COUNT(*) FROM order_books GROUP BY symbol;"
sqlite3 data/db/crypto_data.db "SELECT symbol, COUNT(*) FROM trades GROUP BY symbol;"

# Monitor trading performance
tail -f logs/paper_trader.log
tail -f logs/trade_stream.log
tail -f logs/funding_monitor.log
```

### RL Training & Model Development (PC)
```bash
# Market microstructure feature engineering
python pc/features.py
python pc/enhanced_features.py
python pc/microstructure_features.py

# Decision Transformer training (frozen backbone)
python pc/train_decision_transformer.py

# Offline RL training (30-day quarantine)
python pc/offline_rl_trainer.py

# Multi-task Transformer training
python pc/train.py --epochs 100 --batch_size 32 --learning_rate 0.001

# Export and quantize model for edge deployment
python pc/export_quantize.py

# Automated model refresh pipeline
python scripts/automated_model_refresh.py
```

### Trading Dashboard & Monitoring
```bash
# Real-time trading decisions
python raspberry_pi/infer.py

# Production monitoring dashboard
python raspberry_pi/monitoring_dashboard.py

# Kafka consumers for real-time processing
python raspberry_pi/kafka_consumers/feature_consumer.py
python raspberry_pi/kafka_consumers/trading_consumer.py

# FastAPI backend server (development)
cd backend && uvicorn app.main:app --reload --port 8000

# React frontend (development)
cd frontend && npm start
# Access at http://localhost:3000

# Full-stack development
# Backend: http://localhost:8000
# Frontend: http://localhost:3000
# API docs: http://localhost:8000/docs
```

### Database Operations
```bash
# Migrate SQLite to TimescaleDB
python scripts/migrate_to_timescale.py

# Setup production database with proper schemas
python scripts/setup_production_db.py

# Generate implementation summary and status
python scripts/implementation_summary.py
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

**Full-Stack Web Application**:
- `backend/app/main.py`: FastAPI application with REST API and WebSocket support
- `backend/app/routers/`: API endpoints for auth, trading, portfolio, data, system monitoring
- `frontend/src/`: React TypeScript application with Redux state management
- `frontend/src/components/`: Dashboard components for trading, portfolio, system monitoring

**Event-Driven Data Collection**:
- `raspberry_pi/orderbook_collector.py`: WebSocket L2/L3 order book with incremental updates
- `raspberry_pi/trade_stream.py`: Real-time trade flow analysis
- `raspberry_pi/exchanges/`: Multi-exchange abstraction layer (Binance, Coinbase, Kraken, KuCoin)
- `raspberry_pi/kafka_producers/`: Event streaming producers for order books, trades, news
- `raspberry_pi/kafka_consumers/`: Real-time feature generation and trading decisions

**Advanced Paper Trading & Risk Management**:
- `raspberry_pi/paper_trader.py`: Portfolio management with $10,000 starting balance
- `raspberry_pi/execution_simulator.py`: FIFO queue modeling, partial fills, toxic fill detection
- `raspberry_pi/risk_manager.py`: Correlation-based risk, volatility regimes, circuit breakers
- `raspberry_pi/funding_monitor.py`: Perpetuals funding rate tracking (critical for profitability)
- `raspberry_pi/arbitrage_monitor.py`: Multi-exchange arbitrage opportunity detection
- `raspberry_pi/data_validator.py`: Real-time data quality validation
- `raspberry_pi/event_synchronizer.py`: Cross-exchange event synchronization

**Model Architecture & Training**:
- `pc/models/decision_transformer.py`: Frozen backbone with trainable action/value heads
- `pc/offline_rl_trainer.py`: 30-day quarantine rule, combinatorial purged CV
- `pc/enhanced_features.py`: Order book imbalance, microprice, VWAP deviations
- `pc/microstructure_features.py`: Advanced market microstructure feature engineering
- BF16 mixed precision (NOT FP16), Flash Attention optimization for RTX 4090

**Production Infrastructure**:
- FastAPI backend with JWT authentication and role-based access
- React frontend with real-time WebSocket updates
- PostgreSQL with TimescaleDB for time-series data
- Redis for caching and session management
- Kafka event streaming with sub-second latency
- Docker deployment with Nginx reverse proxy
- Systemd services for production deployment
- Comprehensive monitoring and alerting

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
- `config/config.yaml`: Main application configuration
- `config/kafka_config.yaml`: Event streaming topics and consumers
- `config/risk_limits.yaml`: Max position 25%, max drawdown 25%
- `config/model_config.yaml`: Decision Transformer, BF16, GPU settings
- `config/monitoring_config.yaml`: Alerts, dashboards, Kafka lag monitoring
- `config/exchanges_config.yaml`: Multi-exchange connection parameters
- `config/timescale_config.yaml`: TimescaleDB configuration and schemas

**Deployment Configuration**:
- `deployment/docker/docker-compose.yml`: Full-stack Docker deployment
- `deployment/nginx/tickerml.conf`: Nginx reverse proxy configuration
- `deployment/systemd/`: Linux systemd service definitions
- `backend/requirements.txt`: FastAPI backend dependencies
- `frontend/package.json`: React frontend dependencies
- `raspberry_pi/requirements.txt`: Trading engine dependencies
- `pc/requirements.txt`: ML training dependencies

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

## Full-Stack Application Architecture

The system is now a complete full-stack application with modern web technologies:

**Frontend (React + TypeScript)**:
- Modern dashboard with real-time updates via WebSocket
- Redux state management for global application state
- Component-based architecture with reusable UI elements
- Responsive design for desktop and mobile access
- Real-time portfolio monitoring and trading controls

**Backend (FastAPI + Python)**:
- RESTful API with automatic OpenAPI documentation
- JWT-based authentication with role-based access control
- WebSocket endpoints for real-time data streaming
- Background task processing for trading operations
- Integration with Redis for caching and session management

**Database Layer**:
- PostgreSQL with TimescaleDB for time-series data
- SQLite for development and testing
- Redis for caching and real-time data
- Automated migration scripts for production deployment

**Deployment Infrastructure**:
- Docker containers for all services
- Nginx reverse proxy with SSL termination
- Systemd services for production deployment
- Automated deployment scripts
- Health monitoring and alerting

## Current Implementation Status

**✅ Completed**:
- Full-stack web application (FastAPI + React)
- Modern dashboard with real-time WebSocket updates
- Production deployment infrastructure (Docker + Nginx + Systemd)
- Event-driven Kafka infrastructure
- Decision Transformer with frozen backbone
- Advanced execution simulation (FIFO queues)
- Correlation-based risk management
- Multi-exchange order book collection
- Arbitrage monitoring and opportunity detection
- Comprehensive risk management system
- Data validation and event synchronization
- TimescaleDB migration and production database setup

**🚧 In Progress**:
- Offline RL trainer implementation
- Advanced microstructure feature engineering
- Cross-exchange arbitrage execution
- Enhanced monitoring and alerting
- Integration testing and performance optimization

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

**Application Code**:
- `backend/`: FastAPI backend application
- `frontend/`: React TypeScript frontend
- `raspberry_pi/`: Trading engine and data collection
- `pc/`: ML training and model development
- `scripts/`: Utility and setup scripts
- `tests/`: Comprehensive test suite

**Data Storage**:
- `data/db/`: SQLite databases (development)
- `data/dumps/`: CSV exports and ETL data
- `data/features/`: Preprocessed ML features
- `models/checkpoints/`: PyTorch model checkpoints
- `models/onnx/`: Quantized ONNX models for deployment

**Configuration & Deployment**:
- `config/`: YAML configuration files
- `deployment/`: Production deployment configurations
- `logs/`: Application logs (rotated daily)

**Documentation**:
- `docs/`: Project documentation
- `notebooks/`: Jupyter analysis notebooks