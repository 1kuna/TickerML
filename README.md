# Crypto Time-Series Transformer Pipeline

## Overview

A two-stage system for minute-level crypto price forecasting using a lightweight Transformer.

- **Data Harvester (Raspberry Pi):** continuous minute-level fetch + daily/weekly ETL
- **Training & Model Ops (PC):** feature engineering, training, quantization
- **Inference & Dashboard (Raspberry Pi):** scheduled inference + local dashboard

**Symbols:** BTC/USDT, ETH/USDT

**Prediction Targets:**
- Next-5, 10, 30 minute price (regression)
- Up/down movement (classification + confidence)

## Architecture

```
[Raspberry Pi]
├─ Cron @ 1 min → Fetch tick data → TS DB
└─ Cron @ midnight → ETL dump → CSV / remote sync

[PC]
├─ Ingest ETL dumps
├─ Feature engineering (indicators + sentiment)
├─ Train Transformer → export ONNX → quantize
└─ Sync model → Raspberry Pi

[Raspberry Pi]
├─ Cron @ N min → Load features → ONNX inference → log
└─ Flask dashboard → visualize preds vs actuals
```

## Project Structure

```
TickerML/
├── raspberry_pi/          # Raspberry Pi components
│   ├── harvest.py         # Minute-level data harvester
│   ├── export_etl.py      # Daily ETL dump
│   ├── infer.py          # Inference service
│   ├── dashboard.py      # Flask dashboard
│   └── requirements.txt   # Pi dependencies
├── pc/                   # PC components
│   ├── features.py       # Feature engineering
│   ├── train.py         # Model training
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

#### **Test Everything**
```bash
# Setup environment
python scripts/setup_test_env.py

# Run comprehensive tests
python tests/test_data_collection.py

# Check status
python tests/test_summary.py
```

#### **Manual Data Collection**
```bash
# Single collection (Binance.US + CoinGecko fallback)
python raspberry_pi/harvest.py

# Export to CSV
python raspberry_pi/export_etl.py

# Check what was collected
sqlite3 data/db/crypto_ohlcv.db "SELECT symbol, COUNT(*) FROM ohlcv GROUP BY symbol;"
```

#### **Automated Collection Setup**
```bash
# Edit crontab
crontab -e

# Add this line (replace /path/to/TickerML with your actual path):
* * * * * cd /Users/zach/Documents/Git/TickerML && python raspberry_pi/harvest.py >> logs/harvest.log 2>&1

# Monitor collection
tail -f logs/harvest.log
```

#### **Dashboard (after collecting data)**
```bash
# Start web dashboard
python raspberry_pi/dashboard.py

# Visit: http://localhost:5000
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

✅ **Working Components:**
- Database setup and operations
- Data harvesting from Binance.US
- News harvesting and sentiment analysis (Raspberry Pi)
- CSV export functionality
- Error handling and logging
- Duplicate prevention

✅ **APIs Available:**
- Binance.US: `https://api.binance.us/api/v3`
- CoinGecko: `https://api.coingecko.com/api/v3`
- NewsAPI: `https://newsapi.org/v2` (optional)

❌ **Known Issues:**
- Binance.com blocked (geographic restriction)
- Need 24+ hours of data for meaningful analysis

## Key Files

| File | Purpose |
|------|---------|
| `raspberry_pi/harvest.py` | Data collection (Binance.US + CoinGecko fallback) |
| `raspberry_pi/news_harvest.py` | News collection and sentiment analysis |
| `raspberry_pi/export_etl.py` | CSV export |
| `raspberry_pi/dashboard.py` | Web dashboard |
| `tests/test_data_collection.py` | Comprehensive testing |
| `tests/test_news_harvest.py` | News harvesting tests |
| `tests/test_summary.py` | Status overview |
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
| **Database** | SQLite (or TimescaleDB) |
| **ML Framework** | PyTorch |
| **Inference** | ONNX Runtime |
| **Dashboard** | Flask + Chart.js |
| **Scheduling** | cron |
| **Feature Engineering** | pandas, numpy, ta |
| **Sentiment Analysis** | Gemma 3 4B (via Ollama) |
| **Quantization** | onnxruntime.quantization |

## Components & Responsibilities

| Component | Platform | Primary Tasks | Schedule |
|-----------|----------|---------------|----------|
| **Minute-level Harvester** | Raspberry Pi | Call Binance REST API for BTC/USDT & ETH/USDT every 1 min<br>Insert timestamped OHLCV into TS DB | Every minute (cron) |
| **News Harvester** | Raspberry Pi | Fetch crypto news via NewsAPI<br>Analyze sentiment with Gemma 3 or keyword fallback<br>Store articles and hourly sentiment aggregates in DB | Every 15 minutes (cron) |
| **ETL Dump & Sync** | Raspberry Pi | At 00:00 UTC, export last 24h of DB → CSV<br>(Optional) SCP to PC or cloud storage | Daily at midnight (cron) |
| **Feature Engineering** | PC | Read CSVs/DB dumps<br>Compute RSI, MACD, VWAP, Bollinger bands, moving avgs, volatility<br>Use stored sentiment data or fetch fresh via NewsAPI | On training run (ad-hoc) |
| **Model Training** | PC (GPU/CPU) | Prepare sliding windows (60 min history → targets at 5,10,30 min)<br>Define 4-6-layer Transformer<br>Train with mixed precision, track MSE & classification AUC | As needed (ad-hoc) |
| **Export & Quantization** | PC | Export best PyTorch checkpoint → ONNX<br>Run ONNX Runtime INT8 quantization | Post-training |
| **Inference Service** | Raspberry Pi | Load quantized ONNX model<br>Every 5 min: query last 60 min features from DB → predict next intervals<br>Log preds + confidences | Every 5 min (cron or daemon) |
| **Dashboard & Alerts** | Raspberry Pi | Flask/FastAPI app with time series charts<br>Confidence gauges<br>(Optional) Email or webhook alerts on high-confidence signals | Continuous |

## Model Architecture

The system uses a lightweight Transformer for time-series forecasting:

```python
class TimeSeriesTransformer(nn.Module):
    def __init__(self, d_model=128, n_heads=4, n_layers=6, feature_dim=F):
        super().__init__()
        self.pos_enc = PositionalEncoding(d_model)
        self.input_proj = nn.Linear(feature_dim, d_model)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, n_heads), num_layers=n_layers
        )
        self.reg_head = nn.Linear(d_model, 3)  # next-5/10/30-min price
        self.cls_head = nn.Linear(d_model, 2)  # up/down

    def forward(self, x):
        x = self.input_proj(x) + self.pos_enc(x)
        z = self.transformer(x)
        out_reg = self.reg_head(z[:, -1, :])
        out_cls = self.cls_head(z[:, -1, :])
        return out_reg, out_cls
```

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