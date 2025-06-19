# 🚀 TickerML Raspberry Pi Deployment Guide

## MAXIMUM DATA COLLECTION FOR OPTIMAL MODEL PERFORMANCE

This deployment guide sets up **ULTRA HIGH-FREQUENCY** data collection on your Raspberry Pi for the best possible model training.

## 📊 Data Collection Overview

**What you'll be collecting every 30 seconds:**
- 📈 **Multi-timeframe OHLCV**: 1m, 5m, 15m, 1h intervals
- 🔍 **Order book depth**: Top 20 bid/ask levels
- 💱 **Recent trades**: Last 100 trades per symbol
- 📋 **24hr market stats**: Complete trading statistics
- 🗞️ **News sentiment**: Every 15 minutes with Qwen 3 analysis

**Total data points per collection cycle:**
- ~400+ data points per symbol per cycle
- ~800+ total data points every 30 seconds
- ~1.2M+ data points per day for maximum model richness

## 🔧 Quick Setup

### 1. Clone and Setup
```bash
cd ~
git clone <your-repo-url> TickerML
cd TickerML

# Make setup script executable
chmod +x scripts/setup.sh
./scripts/setup.sh
```

### 2. Install Dependencies
```bash
# Install Python packages
pip install -r raspberry_pi/requirements.txt

# Install Ollama for sentiment analysis
curl -fsSL https://ollama.ai/install.sh | sh

# Install Qwen 3 model
ollama pull qwen3:4b  # Or qwen3:1b for lighter systems
```

### 3. Configure API Keys
```bash
# Copy config template
cp config/config.yaml.sample config/config.yaml

# Edit with your NewsAPI key
nano config/config.yaml
```

Add your NewsAPI key to the config:
```yaml
features:
  sentiment:
    news_api_key: "YOUR_NEWSAPI_KEY_HERE"
```

### 4. Test the System
```bash
# Test data collection
python raspberry_pi/harvest.py

# Test news harvesting
python raspberry_pi/news_harvest.py

# Check system status
python tests/test_summary.py
```

## ⚡ HIGH-FREQUENCY AUTOMATED COLLECTION

### Cron Job Configuration (MAXIMUM DATA)

Add these to your crontab (`crontab -e`):

```bash
# ULTRA HIGH-FREQUENCY data collection - Every 30 seconds
* * * * * cd ~/TickerML && python raspberry_pi/harvest.py >> logs/harvest.log 2>&1
* * * * * cd ~/TickerML && sleep 30 && python raspberry_pi/harvest.py >> logs/harvest.log 2>&1

# News and sentiment - Every 15 minutes for maximum coverage
*/15 * * * * cd ~/TickerML && python raspberry_pi/news_harvest.py >> logs/news_harvest.log 2>&1

# Daily data export at midnight
0 0 * * * cd ~/TickerML && python raspberry_pi/export_etl.py >> logs/etl.log 2>&1

# Paper trading inference (once trained)
*/5 * * * * cd ~/TickerML && python raspberry_pi/infer.py >> logs/infer.log 2>&1
```

**This gives you data collection every 30 seconds = 2,880 collection cycles per day!**

### Alternative: Even More Frequent Collection

For MAXIMUM data density, you can collect every 15 seconds:

```bash
# EXTREME frequency - Every 15 seconds (caution: high API usage)
* * * * * cd ~/TickerML && python raspberry_pi/harvest.py >> logs/harvest.log 2>&1
* * * * * cd ~/TickerML && sleep 15 && python raspberry_pi/harvest.py >> logs/harvest.log 2>&1
* * * * * cd ~/TickerML && sleep 30 && python raspberry_pi/harvest.py >> logs/harvest.log 2>&1
* * * * * cd ~/TickerML && sleep 45 && python raspberry_pi/harvest.py >> logs/harvest.log 2>&1
```

**This gives you 5,760 collection cycles per day = ~2.3M data points daily!**

## 📈 Data Volume Expectations

### Conservative (30-second collection):
- **Per Day**: ~1.2M data points
- **Per Week**: ~8.4M data points  
- **Per Month**: ~36M data points

### Aggressive (15-second collection):
- **Per Day**: ~2.3M data points
- **Per Week**: ~16.1M data points
- **Per Month**: ~69M data points

## 🗄️ Database Storage

### Disk Space Requirements:
- **Conservative**: ~2-3 GB per month
- **Aggressive**: ~4-6 GB per month

### Database Optimization:
```bash
# Weekly database optimization
0 2 * * 0 cd ~/TickerML && sqlite3 data/db/crypto_ohlcv.db "VACUUM; REINDEX;"
```

## 📊 Monitoring Your Data Collection

### Real-time Monitoring:
```bash
# Watch live collection
tail -f logs/harvest.log

# Check data volume
watch "sqlite3 data/db/crypto_ohlcv.db 'SELECT symbol, timeframe, COUNT(*) FROM ohlcv GROUP BY symbol, timeframe;'"

# Monitor order book data
watch "sqlite3 data/db/crypto_ohlcv.db 'SELECT symbol, COUNT(*) as trades FROM trades GROUP BY symbol;'"
```

### Data Quality Dashboard:
```bash
# Start monitoring dashboard
python raspberry_pi/dashboard.py
# Access at http://your-pi-ip:5000
```

## 🔧 Advanced Configuration

### Maximum Data Collection Settings

Edit `raspberry_pi/harvest.py` for even more data:

```python
# EXTREME DATA COLLECTION SETTINGS
ORDER_BOOK_DEPTH = 50  # Top 50 levels instead of 20
RECENT_TRADES_LIMIT = 500  # Last 500 trades instead of 100
KLINE_LIMIT = 200  # More historical bars per request
TIMEFRAMES = ["1m", "3m", "5m", "15m", "30m", "1h", "4h"]  # More timeframes
```

### News Collection Enhancement:

```python
# In raspberry_pi/news_harvest.py
COLLECTION_FREQUENCY_MINUTES = 10  # Every 10 minutes instead of 15
NEWS_SOURCES = ["newsapi", "reddit", "twitter"]  # Multiple sources
```

## 🎯 Performance Optimization

### For Raspberry Pi 4:
```bash
# Increase swap for heavy data processing
sudo dphys-swapfile swapoff
sudo nano /etc/dphys-swapfile
# Set CONF_SWAPSIZE=2048
sudo dphys-swapfile setup
sudo dphys-swapfile swapon
```

### Memory Management:
```bash
# Add to crontab for memory cleanup
0 3 * * * sudo sync && sudo sysctl vm.drop_caches=3
```

## 🚨 Rate Limiting & API Management

### Binance.US API Limits:
- **Weight limit**: 1200 per minute
- **Order limit**: 10 per second
- **Daily limit**: 100,000 requests

### Our Usage:
- **30-second collection**: ~8 requests per minute (safe)
- **15-second collection**: ~16 requests per minute (safe)

### Backup Data Sources:
The system automatically falls back to CoinGecko if Binance.US fails.

## 📋 Maintenance Tasks

### Daily:
```bash
# Check logs for errors
grep -i error logs/*.log | tail -20

# Verify data collection
python tests/test_summary.py
```

### Weekly:
```bash
# Database optimization
sqlite3 data/db/crypto_ohlcv.db "VACUUM; REINDEX;"

# Cleanup old logs
find logs/ -name "*.log" -mtime +7 -delete
```

### Monthly:
```bash
# Export data for PC training
python raspberry_pi/export_etl.py

# Archive old data
tar -czf backup-$(date +%Y%m).tar.gz data/
```

## 🎉 Success Metrics

**Your Pi is working optimally when you see:**
- ✅ Consistent data collection every 30 seconds
- ✅ Order book updates with 20+ levels
- ✅ Trade data flowing continuously  
- ✅ News sentiment analysis running
- ✅ Multi-timeframe data across 4+ intervals
- ✅ Database growing by ~1GB+ per month

## 🆘 Troubleshooting

### Common Issues:

**1. API Rate Limits:**
```bash
# Check logs
grep "429" logs/harvest.log
# Solution: Increase delays between requests
```

**2. Disk Space:**
```bash
# Check space
df -h
# Solution: Setup automatic cleanup or larger SD card
```

**3. Memory Issues:**
```bash
# Check memory
free -h
# Solution: Restart services or increase swap
```

**4. Ollama Issues:**
```bash
# Restart Ollama
sudo systemctl restart ollama
# Check status
ollama list
```

## 🔮 Next Steps

Once data collection is running smoothly:

1. **Wait 24-48 hours** for sufficient training data
2. **Transfer data to PC** for model training
3. **Train models** with the rich dataset
4. **Deploy trained models** back to Pi for inference
5. **Start paper trading** with live predictions

Your TickerML system will have one of the most comprehensive crypto datasets possible! 🚀