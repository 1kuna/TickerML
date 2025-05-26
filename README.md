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
├── models/               # Model artifacts
│   ├── checkpoints/     # PyTorch checkpoints
│   └── onnx/           # ONNX models
├── data/                # Data storage
│   ├── dumps/          # ETL CSV dumps
│   └── db/             # Database files
├── config/              # Configuration files
│   └── config.yaml     # Main configuration
└── scripts/            # Utility scripts
    └── setup.sh        # Environment setup
```

## Quick Start

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

2. **Download NLTK data for sentiment analysis:**
```bash
python -c "import nltk; nltk.download('punkt'); nltk.download('vader_lexicon')"
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
| **Sentiment Analysis** | transformers (DistilBERT/FinBERT), vaderSentiment |
| **Quantization** | onnxruntime.quantization |

## Components & Responsibilities

| Component | Platform | Primary Tasks | Schedule |
|-----------|----------|---------------|----------|
| **Minute-level Harvester** | Raspberry Pi | Call Binance REST API for BTC/USDT & ETH/USDT every 1 min<br>Insert timestamped OHLCV into TS DB | Every minute (cron) |
| **ETL Dump & Sync** | Raspberry Pi | At 00:00 UTC, export last 24h of DB → CSV<br>(Optional) SCP to PC or cloud storage | Daily at midnight (cron) |
| **Feature Engineering** | PC | Read CSVs/DB dumps<br>Compute RSI, MACD, VWAP, Bollinger bands, moving avgs, volatility<br>Pull news headlines via NewsAPI → sentiment score | On training run (ad-hoc) |
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

# Test feature engineering
python pc/features.py --test

# Test model training (small dataset)
python pc/train.py --epochs 5 --batch_size 16

# Test inference
python raspberry_pi/infer.py --test

# Run full pipeline test
python scripts/test_pipeline.py
```

### Monitoring

```bash
# Check harvest logs
tail -f logs/harvest.log

# Check inference logs
tail -f logs/infer.log

# View database status
sqlite3 data/db/crypto_data.db "SELECT COUNT(*) FROM ohlcv;"

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

## Development Timeline

- **Day 1:** Pi harvester + ETL setup
- **Day 2-3:** PC feature engineering + sentiment
- **Day 4-6:** Transformer training
- **Day 7:** ONNX export + Pi inference
- **Day 8:** Dashboard + end-to-end testing

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details. 