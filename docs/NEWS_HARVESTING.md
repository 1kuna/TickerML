# News Harvesting & Sentiment Analysis

## Overview

The news harvesting system continuously collects cryptocurrency-related news articles and performs sentiment analysis, storing the results in the database for use by the prediction models. This runs as an asynchronous process on the Raspberry Pi, similar to the price data harvester.

## Architecture

```
[Raspberry Pi] News Harvester
├─ Fetch news articles via NewsAPI
├─ Analyze sentiment using Qwen 3 LLM (REQUIRED)
├─ Store articles in database with deduplication
├─ Generate hourly sentiment aggregates
└─ Run every 15 minutes via cron

[PC] Feature Engineering
├─ Read stored sentiment data from database
├─ Fall back to live news fetch if needed
└─ Combine with price data for model training
```

## Database Schema

### `news_articles` Table
Stores individual news articles with metadata and sentiment scores.

| Column | Type | Description |
|--------|------|-------------|
| `id` | INTEGER | Primary key |
| `article_hash` | TEXT | Unique hash to prevent duplicates |
| `symbol` | TEXT | Associated crypto symbol (BTCUSD, ETHUSD) |
| `title` | TEXT | Article title |
| `description` | TEXT | Article description/summary |
| `content` | TEXT | Full article content (if available) |
| `source` | TEXT | News source name |
| `url` | TEXT | Article URL |
| `published_at` | INTEGER | Publication timestamp (milliseconds) |
| `fetched_at` | INTEGER | When the article was fetched |
| `sentiment_score` | REAL | Sentiment score (-1 to 1) |
| `sentiment_processed_at` | INTEGER | When sentiment was analyzed |

### `news_sentiment_hourly` Table
Stores hourly sentiment aggregates for fast lookups.

| Column | Type | Description |
|--------|------|-------------|
| `id` | INTEGER | Primary key |
| `symbol` | TEXT | Associated crypto symbol |
| `hour_timestamp` | INTEGER | Hour timestamp (rounded down) |
| `avg_sentiment` | REAL | Average sentiment for that hour |
| `article_count` | INTEGER | Number of articles in the average |

## Setup

### 1. Configure NewsAPI (Optional but Recommended)

1. Get a free API key from [NewsAPI.org](https://newsapi.org/)
2. Add to your configuration:

```yaml
# config/config.yaml
features:
  sentiment:
    news_api_key: "your_actual_api_key_here"
    update_interval_minutes: 15
    ollama_host: "http://localhost:11434"
    model: "qwen3:4b"
```

Or set environment variable:
```bash
export NEWS_API_KEY="your_actual_api_key_here"
```

### 2. Set Up Sentiment Analysis (REQUIRED)

#### Ollama + Qwen 3 LLM (REQUIRED)
The system requires Ollama with Qwen 3 model for sentiment analysis. There is no fallback method.

```bash
# Install Ollama
curl -fsSL https://ollama.ai/install.sh | sh

# Pull Qwen 3 model
ollama pull qwen3:4b

# Start Ollama service
ollama serve
```

**Important**: The news harvester will fail if Ollama is not available or if the Qwen 3 model cannot be accessed. This ensures consistent, high-quality sentiment analysis.

### 3. Initialize Database Tables
```bash
python raspberry_pi/news_harvest.py
```

This creates the necessary database tables on first run.

### 4. Set Up Automated Harvesting

Add to crontab (`crontab -e`):
```bash
# News harvesting every 15 minutes
*/15 * * * * cd /path/to/TickerML && python raspberry_pi/news_harvest.py >> logs/news_harvest.log 2>&1
```

## Usage

### Manual News Harvesting
```bash
# Run news harvester once
python raspberry_pi/news_harvest.py

# Test the news harvesting system
python scripts/test_news_harvest.py
```

### Check News Data
```bash
# View articles in database
sqlite3 data/db/crypto_news.db "SELECT symbol, COUNT(*) FROM news_articles GROUP BY symbol;"

# View hourly sentiment data
sqlite3 data/db/crypto_news.db "SELECT symbol, COUNT(*) FROM news_sentiment_hourly GROUP BY symbol;"

# View recent articles
sqlite3 data/db/crypto_news.db "SELECT title, sentiment_score FROM news_articles ORDER BY published_at DESC LIMIT 5;"
```

### Monitor Logs
```bash
# Monitor news harvesting logs
tail -f logs/news_harvest.log

# Check for errors
grep ERROR logs/news_harvest.log
```

## Features

### Sentiment Analysis Method

**Qwen 3 LLM** (Required)
- High-quality sentiment analysis with contextual understanding
- Returns scores from -1 (very negative) to +1 (very positive)
- Requires Ollama installation and running service
- **No fallback method** - system fails if unavailable

This ensures all sentiment scores are generated using the same high-quality method, maintaining consistency across all stored data.

### Deduplication
- Articles are deduplicated using MD5 hash of title + URL + timestamp
- Prevents storing the same article multiple times

### Rate Limiting
- Respects NewsAPI rate limits
- 1-second delays between symbol requests
- Configurable update intervals

### Symbol Mapping
The system maps crypto symbols to search terms:
- `BTCUSD`/`BTCUSDT` → "Bitcoin BTC cryptocurrency"
- `ETHUSD`/`ETHUSDT` → "Ethereum ETH cryptocurrency"

## Integration with Feature Engineering

The PC-based feature engineering automatically uses stored news data:

```python
# pc/features.py will first try stored data
sentiment_scores = fetch_news_sentiment(symbol, use_stored_data=True)

# Falls back to live fetch if no stored data
if not sentiment_scores:
    sentiment_scores = fetch_news_sentiment(symbol, use_stored_data=False)
```

## Configuration Options

### `config/config.yaml`
```yaml
features:
  sentiment:
    enabled: true
    model: "qwen3:4b"
    update_interval_minutes: 15
    sources: ["newsapi"]
    ollama_host: "http://localhost:11434"
    news_api_key: "your_key_here"
```

### Environment Variables
```bash
NEWS_API_KEY=your_newsapi_key_here
OLLAMA_HOST=http://localhost:11434
OLLAMA_MODEL=qwen3:4b
```

## Monitoring & Maintenance

### Health Checks
```bash
# Check if news harvester is running
ps aux | grep news_harvest

# Check database size
du -sh data/db/crypto_news.db

# Check latest fetch time
sqlite3 data/db/crypto_news.db "SELECT MAX(fetched_at) FROM news_articles;"
```

### Cleanup Old Data
```bash
# Clean up old articles (older than 30 days)
sqlite3 data/db/crypto_news.db "DELETE FROM news_articles WHERE fetched_at < $(date -d '30 days ago' +%s000);"

# Clean up old sentiment data
sqlite3 data/db/crypto_news.db "DELETE FROM news_sentiment_hourly WHERE hour_timestamp < $(date -d '30 days ago' +%s000);"
```