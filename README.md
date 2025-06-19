# TickerML: Crypto Paper Trading Bot with Reinforcement Learning

## Overview

An intelligent paper trading bot that uses reinforcement learning and market microstructure analysis to make automated crypto trading decisions. The system combines real-time order book analysis, multi-exchange monitoring, and transformer-based models for optimal trade execution.

- **Data Engine (Raspberry Pi):** Real-time order book collection, trade flow analysis, multi-exchange monitoring
- **RL Training & Model Ops (PC):** Reinforcement learning training, feature engineering, model optimization
- **Paper Trading Engine (Raspberry Pi):** Automated trading decisions, portfolio management, risk control
- **Dashboard & Analytics (Raspberry Pi):** Live P&L tracking, performance metrics, attention visualizations

**Supported Exchanges:** Binance.US, Coinbase, Kraken, KuCoin
**Trading Pairs:** BTC/USDT, ETH/USDT, BTC/USD, ETH/USD

**Trading Capabilities:**
- Automated buy/sell/hold decisions with position sizing
- Multi-exchange arbitrage detection
- Risk-adjusted portfolio management
- Real-time market microstructure analysis

## Architecture

```
[Raspberry Pi - Data Collection]
├─ WebSocket streams → Real-time order books → TimescaleDB
├─ Trade flow analysis → Market microstructure features
├─ Multi-exchange monitoring → Arbitrage opportunities
└─ News sentiment → Qwen 3 analysis → Hourly aggregates

[PC - RL Training]
├─ Experience replay buffer → PPO/A2C training
├─ Multi-task Transformer → Trading decisions + Risk assessment
├─ Backtesting engine → Strategy validation
└─ Model export → ONNX quantization → Edge deployment

[Raspberry Pi - Trading Engine]
├─ Real-time inference → Trading decisions (50ms)
├─ Paper trading engine → Portfolio management
├─ Risk management → Position sizing + Stop losses
└─ Performance dashboard → Live P&L + Attention maps
```

## Project Structure

```
TickerML/
├── raspberry_pi/          # Raspberry Pi components
│   ├── harvest.py         # Real-time data collection (order books + trades)
│   ├── paper_trader.py    # Paper trading engine & portfolio management
│   ├── infer.py          # RL-based trading decisions
│   ├── news_harvest.py   # News collection & Qwen 3 sentiment analysis
│   ├── dashboard.py      # Trading dashboard with P&L tracking
│   ├── export_etl.py      # Data export utilities
│   └── requirements.txt   # Pi dependencies
├── pc/                   # PC components
│   ├── rl_trainer.py     # Reinforcement learning training pipeline
│   ├── features.py       # Feature engineering + market microstructure
│   ├── train.py         # Multi-task transformer training
│   ├── export_quantize.py # ONNX export & quantization
│   └── requirements.txt  # PC dependencies
├── scripts/              # Utility scripts
│   ├── setup.sh          # Environment setup
│   ├── setup_test_env.py # Test environment setup
│   └── verify_gemma_config.py # Gemma configuration verifier
├── tests/                # Test scripts
│   ├── test_data_collection.py # Comprehensive data collection tests
│   ├── test_pipeline.py  # Full pipeline tests
│   ├── test_features.py  # Feature engineering tests
│   ├── test_news_harvest.py # News harvesting tests
│   ├── test_sentiment.py # Sentiment analysis tests
│   └── test_summary.py   # Status overview tests
├── docs/                 # Documentation
│   └── TESTING_GUIDE.md  # Detailed testing guide
├── notebooks/            # Jupyter notebooks
│   └── analysis.ipynb    # Data analysis notebook
├── models/               # Model artifacts
│   ├── checkpoints/     # PyTorch checkpoints
│   └── onnx/           # ONNX models
├── data/                # Data storage
│   ├── dumps/          # ETL CSV dumps
│   └── db/             # Database files
├── config/              # Configuration files
│   └── config.yaml     # Main configuration
└── logs/                # Log files
```

## Quick Start

### 🚀 Ready to Use Commands

#### **Setup Paper Trading Bot**
```bash
# Setup environment and dependencies
python scripts/setup_test_env.py

# Run comprehensive system tests
python tests/test_data_collection.py
python tests/test_paper_trader.py

# Check system status
python tests/test_summary.py
```

#### **Start Data Collection**
```bash
# Real-time order book collection (WebSocket)
python raspberry_pi/harvest.py

# News and sentiment analysis
python raspberry_pi/news_harvest.py

# Check collected data
sqlite3 data/db/crypto_ohlcv.db "SELECT symbol, COUNT(*) FROM order_books GROUP BY symbol;"
sqlite3 data/db/crypto_news.db "SELECT COUNT(*) FROM news_sentiment_hourly;"
```

#### **Train RL Trading Model**
```bash
# Feature engineering with market microstructure
python pc/features.py

# Reinforcement learning training
python pc/rl_trainer.py --episodes 1000 --learning_rate 0.001

# Export optimized model
python pc/export_quantize.py
```

#### **Start Paper Trading**
```bash
# Launch paper trading engine
python raspberry_pi/paper_trader.py

# Start trading dashboard
python raspberry_pi/dashboard.py
# Visit: http://localhost:5000 for live P&L tracking
```

#### **Monitor Performance**
```bash
# Check trading performance
python -c "
import sqlite3
conn = sqlite3.connect('data/db/crypto_ohlcv.db')
cursor = conn.execute('SELECT * FROM portfolio_state ORDER BY timestamp DESC LIMIT 1')
print('Current Portfolio:', cursor.fetchone())
conn.close()
"

# View recent trades
tail -f logs/paper_trader.log
```

### Automated Setup

Run the setup script to automatically configure your environment:

```bash
chmod +x scripts/setup.sh
./scripts/setup.sh
```

This script will:
- Detect your platform (Raspberry Pi or PC)
- Create virtual environment
- Install platform-specific dependencies
- Create necessary directories
- Set up logging
- Provide cron configuration instructions

### Manual Setup

#### Raspberry Pi Setup

1. **Install dependencies:**
```bash
cd raspberry_pi
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

2. **Configure the system:**
```bash
# Copy and edit configuration files
cp config/config.yaml.sample config/config.yaml
cp config/env.sample .env

# Edit config.yaml with your preferences
# Edit .env with your API keys and secrets
```

3. **Set up cron jobs:**
```bash
# Add to crontab -e
* * * * * cd /path/to/TickerML && ./venv/bin/python raspberry_pi/harvest.py >> logs/harvest.log 2>&1
*/15 * * * * cd /path/to/TickerML && ./venv/bin/python raspberry_pi/news_harvest.py >> logs/news_harvest.log 2>&1
0 0 * * * cd /path/to/TickerML && ./venv/bin/python raspberry_pi/export_etl.py >> logs/etl.log 2>&1
*/5 * * * * cd /path/to/TickerML && ./venv/bin/python raspberry_pi/infer.py >> logs/infer.log 2>&1
```

4. **Start the dashboard:**
```bash
python raspberry_pi/dashboard.py
# Access at http://localhost:5000
```

#### PC Setup

1. **Install dependencies:**
```bash
cd pc
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

2. **Install and setup Ollama for Gemma 3 sentiment analysis:**
```bash
# Install Ollama (visit https://ollama.ai for platform-specific instructions)
# Then pull the Gemma 3 4B model
ollama pull gemma3:4b
```

3. **Configure NewsAPI (optional):**
   - Get API key from https://newsapi.org/
   - Add to `config/config.yaml` under `features.sentiment.news_api_key`

4. **Run training pipeline:**
```bash
python pc/features.py      # Feature engineering
python pc/train.py         # Model training
python pc/export_quantize.py  # ONNX export & quantization
```

## Current Status

✅ **Current Implementation:**
- Real-time OHLCV data collection from Binance.US
- News harvesting and Qwen 3 sentiment analysis  
- Basic prediction model with Transformer architecture
- Web dashboard with price visualization
- Database operations and logging

🚧 **In Development (Paper Trading Bot):**
- Real-time order book collection
- Multi-exchange support (Coinbase, Kraken, KuCoin)
- Paper trading engine with portfolio management
- Reinforcement learning training pipeline
- Market microstructure feature engineering
- Live P&L tracking and risk management

✅ **APIs Supported:**
- Binance.US: `https://api.binance.us/api/v3` (WebSocket + REST)
- Coinbase: `https://api.exchange.coinbase.com` (planned)
- Kraken: `https://api.kraken.com` (planned)
- KuCoin: `https://api.kucoin.com` (planned)
- NewsAPI: `https://newsapi.org/v2`

❌ **Current Limitations:**
- Paper trading only (no real money)
- Limited to major crypto pairs
- Requires 24+ hours of data for RL training

## Key Files

| File | Purpose |
|------|---------|
| `raspberry_pi/harvest.py` | Real-time order book & trade data collection |
| `raspberry_pi/paper_trader.py` | Paper trading engine with portfolio management |
| `raspberry_pi/infer.py` | RL-based trading decision engine |
| `raspberry_pi/news_harvest.py` | News collection and Qwen 3 sentiment analysis |
| `raspberry_pi/dashboard.py` | Trading dashboard with live P&L tracking |
| `pc/rl_trainer.py` | Reinforcement learning training pipeline |
| `pc/features.py` | Market microstructure feature engineering |
| `tests/test_paper_trader.py` | Paper trading system validation |
| `tests/test_rl_training.py` | RL training pipeline tests |
| `tests/test_data_collection.py` | Multi-exchange data collection tests |
| `scripts/setup_test_env.py` | Test environment setup |

## Configuration

Edit `config/config.yaml` to customize:
- API endpoints and symbols
- Model hyperparameters
- Feature engineering settings
- Dashboard preferences
- Alert configurations

## Tech Stack

| Layer | Technology |
|-------|------------|
| **Language** | Python 3.10+ |
| **Database** | TimescaleDB / SQLite |
| **ML Framework** | PyTorch |
| **RL Framework** | Stable-Baselines3 / Ray RLlib |
| **Inference** | ONNX Runtime (INT8 quantized) |
| **Real-time Data** | WebSocket APIs + asyncio |
| **Dashboard** | Flask + Chart.js + Plotly |
| **Message Queue** | Redis Streams / Apache Kafka |
| **Feature Engineering** | pandas, numpy, ta-lib |
| **Sentiment Analysis** | Qwen 3 (via Ollama) |
| **Paper Trading** | Custom engine with slippage modeling |
| **Risk Management** | Portfolio optimization libraries |

## Components & Responsibilities

| Component | Platform | Primary Tasks | Schedule |
|-----------|----------|---------------|----------|
| **Real-time Data Collector** | Raspberry Pi | WebSocket streams from multiple exchanges<br>Order book collection (5-10 sec snapshots)<br>Trade flow analysis and market microstructure features | Continuous (WebSocket) |
| **News & Sentiment Engine** | Raspberry Pi | Fetch crypto news via NewsAPI<br>Analyze sentiment with Qwen 3 LLM<br>Generate hourly sentiment scores and market regime indicators | Every 15 minutes |
| **Paper Trading Engine** | Raspberry Pi | Execute trading decisions based on RL model<br>Portfolio management with risk controls<br>Position sizing and stop-loss management<br>Transaction cost simulation | Real-time (sub-second) |
| **Feature Engineering** | PC | Market microstructure features (order imbalance, spread dynamics)<br>Technical indicators and portfolio-aware features<br>Multi-exchange arbitrage signals | On training run |
| **RL Training Pipeline** | PC (GPU) | PPO/A2C training with experience replay<br>Multi-task learning (price + action + risk)<br>Backtesting and strategy validation<br>Attention mechanism analysis | Daily/Weekly retraining |
| **Model Deployment** | PC → Pi | ONNX export with INT8 quantization<br>Model validation and performance benchmarking<br>Edge deployment optimization | Post-training |
| **Trading Dashboard** | Raspberry Pi | Live P&L tracking and performance metrics<br>Attention weight visualizations<br>Risk monitoring and portfolio analytics<br>Trade execution logs and analysis | Continuous |
| **Risk Management** | Raspberry Pi | Real-time drawdown monitoring<br>Position correlation analysis<br>Circuit breakers for anomalous conditions<br>Performance degradation alerts | Continuous |

## Model Architecture

The system uses a multi-task Transformer for trading decisions and risk assessment:

```python
class TradingTransformer(nn.Module):
    def __init__(self, d_model=128, n_heads=4, n_layers=6, feature_dim=F):
        super().__init__()
        self.pos_enc = PositionalEncoding(d_model)
        self.input_proj = nn.Linear(feature_dim, d_model)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, n_heads), num_layers=n_layers
        )
        # Multi-task heads for trading bot
        self.price_head = nn.Linear(d_model, 3)      # price predictions
        self.action_head = nn.Linear(d_model, 3)     # buy/sell/hold
        self.position_head = nn.Linear(d_model, 1)   # position sizing
        self.risk_head = nn.Linear(d_model, 1)       # risk assessment

    def forward(self, x):
        x = self.input_proj(x) + self.pos_enc(x)
        z = self.transformer(x)  
        features = z[:, -1, :]
        
        return {
            'price_pred': self.price_head(features),
            'action': self.action_head(features),
            'position_size': torch.sigmoid(self.position_head(features)),
            'risk_score': torch.sigmoid(self.risk_head(features))
        }
```

## Reinforcement Learning Integration

The trading bot uses PPO (Proximal Policy Optimization) for continuous learning:

- **State Space**: Order book features, technical indicators, portfolio state, sentiment scores
- **Action Space**: Buy/Sell/Hold with position sizing (0-25% of portfolio)
- **Reward Function**: Risk-adjusted returns with drawdown penalties
- **Experience Replay**: Stores trading decisions and outcomes for training

## Usage Examples

### Testing the Pipeline

```bash
# Test data harvesting
python raspberry_pi/harvest.py

# Test news harvesting
python raspberry_pi/news_harvest.py

# Test news harvesting functionality
python tests/test_news_harvest.py

# Setup test environment
python scripts/setup_test_env.py

# Run comprehensive data collection tests
python tests/test_data_collection.py

# Check system status
python tests/test_summary.py

# Verify Gemma 3 configuration
python scripts/verify_gemma_config.py

# Test sentiment analysis with Gemma 3
python tests/test_sentiment.py

# Test feature engineering
python pc/features.py --test

# Test model training (small dataset)
python pc/train.py --epochs 5 --batch_size 16

# Test inference
python raspberry_pi/infer.py --test

# Run full pipeline test
python tests/test_pipeline.py
```

### Monitoring

```bash
# Check harvest logs
tail -f logs/harvest.log

# Check news harvest logs
tail -f logs/news_harvest.log

# Check inference logs
tail -f logs/infer.log

# View database status
sqlite3 data/db/crypto_ohlcv.db "SELECT COUNT(*) FROM ohlcv;"

# Check predictions
sqlite3 data/db/crypto_ohlcv.db "SELECT symbol, COUNT(*) FROM predictions GROUP BY symbol;"

# View news data status
sqlite3 data/db/crypto_news.db "SELECT symbol, COUNT(*) FROM news_articles GROUP BY symbol;"

# Check sentiment data
sqlite3 data/db/crypto_news.db "SELECT symbol, COUNT(*) FROM news_sentiment_hourly GROUP BY symbol;"

# Check model performance
python -c "
import pandas as pd
df = pd.read_csv('data/dumps/predictions.csv')
print(df.tail())
"
```

## Troubleshooting

### Common Issues

1. **Cron jobs not running:**
   - Check cron service: `sudo systemctl status cron`
   - Verify paths in crontab are absolute
   - Check log files for errors

2. **Database connection errors:**
   - Ensure data/db directory exists
   - Check file permissions
   - Verify SQLite installation

3. **API rate limits:**
   - Binance allows 1200 requests per minute
   - Current setup uses ~2 requests per minute
   - Monitor logs for 429 errors

4. **Model training issues:**
   - Check GPU availability: `nvidia-smi`
   - Verify CUDA installation for PyTorch
   - Reduce batch size if out of memory

5. **Dashboard not accessible:**
   - Check Flask is running: `ps aux | grep dashboard`
   - Verify port 5000 is not blocked
   - Check firewall settings

6. **Sentiment analysis issues:**
   - Ensure Ollama is installed and running: `ollama serve`
   - Check if Gemma 3 model is available: `ollama list`
   - Pull the model if missing: `ollama pull gemma3:4b`
   - For resource-constrained systems, consider using: `ollama pull gemma3:1b`
   - Verify Ollama host configuration in config.yaml

## Development Timeline

- **Day 1:** Pi harvester + ETL setup
- **Day 2-3:** PC feature engineering + sentiment
- **Day 4-6:** Transformer training
- **Day 7:** ONNX export + Pi inference
- **Day 8:** Dashboard + end-to-end testing

## Documentation

- **[Testing Guide](docs/TESTING_GUIDE.md)** - Comprehensive testing instructions
- **[Analysis Notebook](notebooks/analysis.ipynb)** - Data analysis and visualization tools

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details. 