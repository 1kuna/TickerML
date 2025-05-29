#!/usr/bin/env python3
"""
News harvester for Raspberry Pi
Fetches crypto-related news and performs sentiment analysis
Stores articles and sentiment scores in database for later use
"""

import sqlite3
import requests
import logging
from datetime import datetime, timedelta
from pathlib import Path
import yaml
import os
import time
import hashlib
import signal # Added for graceful shutdown

# Global shutdown flag
shutdown_requested = False

# Signal handler
def handle_shutdown_signal(signum, frame):
    global shutdown_requested
    logger.info(f"Received signal {signum}. Requesting shutdown...")
    shutdown_requested = True

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Default Configuration
DEFAULT_DB_PATH = Path(__file__).parent.parent / "data" / "db" / "crypto_data.db"
DEFAULT_SYMBOLS = ["BTCUSDT", "ETHUSDT"]

def load_config():
    """Load configuration from YAML file with fallbacks"""
    config_path = Path(__file__).parent.parent / "config" / "config.yaml"
    
    # Initialize with defaults
    config_values = {
        "db_path": str(DEFAULT_DB_PATH),
        "symbols": DEFAULT_SYMBOLS,
        "news_api_key": None,
        "ollama_host": "http://localhost:11434",
        "ollama_model": "gemma3:4b",
        "update_interval_minutes": 15,
    }

    if not config_path.exists():
        logger.warning(f"Config file not found at {config_path}. Using default values.")
        return config_values

    try:
        with open(config_path, 'r') as f:
            yaml_config = yaml.safe_load(f)

        if yaml_config:
            # Database path
            config_values["db_path"] = yaml_config.get("database", {}).get("path", DEFAULT_DB_PATH)
            
            # Symbols
            config_values["symbols"] = yaml_config.get("data", {}).get("symbols", DEFAULT_SYMBOLS)
            
            # Sentiment analysis config
            sentiment_config = yaml_config.get("features", {}).get("sentiment", {})
            config_values["news_api_key"] = sentiment_config.get("news_api_key")
            config_values["ollama_host"] = sentiment_config.get("ollama_host", "http://localhost:11434")
            config_values["ollama_model"] = sentiment_config.get("model", "gemma3:4b")
            config_values["update_interval_minutes"] = sentiment_config.get("update_interval_minutes", 15)

    except Exception as e:
        logger.error(f"Error loading config file {config_path}: {e}. Using default values.")
        
    # Check for environment variables
    config_values["news_api_key"] = os.getenv("NEWS_API_KEY", config_values["news_api_key"])
    config_values["ollama_host"] = os.getenv("OLLAMA_HOST", config_values["ollama_host"])
    config_values["ollama_model"] = os.getenv("OLLAMA_MODEL", config_values["ollama_model"])
    
    return config_values

# Load configuration
config = load_config()
DB_PATH = Path(config['db_path'])
SYMBOLS = config['symbols']
NEWS_API_KEY = config['news_api_key']
OLLAMA_HOST = config['ollama_host']
OLLAMA_MODEL = config['ollama_model']
UPDATE_INTERVAL_MINUTES = config['update_interval_minutes']

def init_news_database():
    """Initialize SQLite database with news tables"""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = None
    try:
        conn = sqlite3.connect(str(DB_PATH))
        cursor = conn.cursor()
        
        # Create news articles table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS news_articles (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            article_hash TEXT UNIQUE NOT NULL,
            symbol TEXT NOT NULL,
            title TEXT NOT NULL,
            description TEXT,
            content TEXT,
            source TEXT,
            url TEXT,
            published_at INTEGER NOT NULL,
            fetched_at INTEGER NOT NULL,
            sentiment_score REAL,
            sentiment_processed_at INTEGER,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    # Create news sentiment index table for quick lookups
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS news_sentiment_hourly (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT NOT NULL,
            hour_timestamp INTEGER NOT NULL,
            avg_sentiment REAL NOT NULL,
            article_count INTEGER NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(symbol, hour_timestamp)
        )
    """)
    
    # Create indexes for faster queries
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_news_symbol_published 
        ON news_articles(symbol, published_at)
    """)
    
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_news_hash 
        ON news_articles(article_hash)
    """)
    
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_sentiment_symbol_hour 
        ON news_sentiment_hourly(symbol, hour_timestamp)
    """)
    
        conn.commit()
        logger.info(f"News database tables initialized at {DB_PATH}")
    except sqlite3.Error as e:
        logger.error(f"Database initialization error: {e}")
        raise # Re-raise to indicate critical failure
    finally:
        if conn:
            conn.close()

def generate_article_hash(title, url, published_at):
    """Generate a unique hash for an article to prevent duplicates"""
    content = f"{title}|{url}|{published_at}"
    return hashlib.md5(content.encode()).hexdigest()

def fetch_news_for_symbol(symbol, hours_back=2):
    """Fetch news articles for a specific symbol"""
    if not NEWS_API_KEY or NEWS_API_KEY == "your_newsapi_key_here":
        logger.warning("NewsAPI key not configured, skipping news fetch")
        return []
    
    # Map crypto symbols to search terms that actually appear in headlines
    search_terms = {
        "BTCUSDT": "Bitcoin",
        "BTCUSD": "Bitcoin", 
        "ETHUSDT": "Ethereum",
        "ETHUSD": "Ethereum"
    }
    
    query = search_terms.get(symbol, "cryptocurrency")
    
    # Calculate time range
    end_time = datetime.now()
    start_time = end_time - timedelta(hours=hours_back)
    
    try:
        # NewsAPI request
        url = "https://newsapi.org/v2/everything"
        params = {
            'q': query,
            'from': start_time.isoformat(),
            'to': end_time.isoformat(),
            'sortBy': 'publishedAt',
            'language': 'en',
            'apiKey': NEWS_API_KEY,
            'pageSize': 50  # Limit articles per request
        }
        
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        
        news_data = response.json()
        articles = news_data.get('articles', [])
        
        logger.info(f"Fetched {len(articles)} articles for {symbol}")
        return articles
        
    except Exception as e:
        logger.error(f"Error fetching news for {symbol}: {e}")
        return []

def analyze_sentiment_simple(text):
    """Sentiment analysis using Gemma 3 4B LLM via Ollama - NO FALLBACK"""
    try:
        # Import ollama - fail if not available
        import ollama
        
        # Prepare the prompt for sentiment analysis
        prompt = f"""Analyze the sentiment of the following financial news text and return only a numerical score between -1 and 1, where:
-1 = very negative sentiment
0 = neutral sentiment  
1 = very positive sentiment

Text: "{text[:500]}"

Return only the numerical score (e.g., 0.3, -0.7, 0.0):"""

        # Call Gemma 3 via Ollama
        client = ollama.Client(host=OLLAMA_HOST)
        response = client.chat(
            model=OLLAMA_MODEL,
            messages=[{
                'role': 'user',
                'content': prompt
            }],
            options={
                'temperature': 0.1,  # Low temperature for consistent results
                'num_predict': 10,   # Short response expected
            }
        )
        
        # Extract and parse the sentiment score
        sentiment_text = response['message']['content'].strip()
        
        # Try to extract a number from the response
        import re
        numbers = re.findall(r'-?\d+\.?\d*', sentiment_text)
        
        if numbers:
            sentiment_score = float(numbers[0])
            # Clamp to [-1, 1] range
            sentiment_score = max(-1.0, min(1.0, sentiment_score))
            return sentiment_score
        else:
            logger.error(f"Could not parse sentiment score from Gemma 3 response: {sentiment_text}")
            raise ValueError(f"Invalid sentiment response from Gemma 3: {sentiment_text}")
            
    except ImportError as e:
        logger.error("Ollama library not available - cannot perform sentiment analysis")
        raise ImportError("Ollama library required for sentiment analysis") from e
    except Exception as e:
        logger.error(f"Error with Gemma 3 sentiment analysis via Ollama: {e}")
        raise RuntimeError(f"Gemma 3 sentiment analysis failed: {e}") from e

def store_news_article(symbol, article, sentiment_score=None):
    """Store a news article in the database"""
    try:
        # Parse published date
        published_at = datetime.fromisoformat(
            article['publishedAt'].replace('Z', '+00:00')
        )
        published_timestamp = int(published_at.timestamp() * 1000)
        
        # Generate unique hash
        article_hash = generate_article_hash(
            article.get('title', ''),
            article.get('url', ''),
            published_timestamp
        )
        
        conn = None
        try:
            conn = sqlite3.connect(str(DB_PATH))
            cursor = conn.cursor()
            
            # Insert article (will be ignored if hash already exists)
            cursor.execute("""
            INSERT OR IGNORE INTO news_articles 
            (article_hash, symbol, title, description, content, source, url, 
             published_at, fetched_at, sentiment_score, sentiment_processed_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            article_hash,
            symbol,
            article.get('title', ''),
            article.get('description', ''),
            article.get('content', ''),
            article.get('source', {}).get('name', ''),
            article.get('url', ''),
            published_timestamp,
            int(time.time() * 1000),
            sentiment_score,
            int(time.time() * 1000) if sentiment_score is not None else None
        ))
        
            rows_affected = cursor.rowcount
            conn.commit()
            
            if rows_affected > 0:
                logger.debug(f"Stored new article: {article.get('title', '')[:50]}...")
            
            return rows_affected > 0
            
        except Exception as e:
            logger.error(f"Error storing article: {e}")
            return False
        finally:
            if conn:
                conn.close()

def update_hourly_sentiment(symbol):
    """Update hourly sentiment aggregates for a symbol"""
    conn = None
    try:
        conn = sqlite3.connect(str(DB_PATH))
        cursor = conn.cursor()
        
        # Get the latest hour we need to process
        end_time = datetime.now()
        start_time = end_time - timedelta(hours=24)  # Process last 24 hours
        
        start_timestamp = int(start_time.timestamp() * 1000)
        end_timestamp = int(end_time.timestamp() * 1000)
        
        # Query articles with sentiment scores
        cursor.execute("""
            SELECT published_at, sentiment_score
            FROM news_articles 
            WHERE symbol = ? 
            AND published_at >= ? 
            AND published_at <= ?
            AND sentiment_score IS NOT NULL
            ORDER BY published_at
        """, (symbol, start_timestamp, end_timestamp))
        
        articles = cursor.fetchall()
        
        if not articles:
            logger.info(f"No articles with sentiment found for {symbol} in the last 24 hours.")
            # conn.close() will be handled by finally
            return
        
        # Group by hour and calculate averages
        hourly_data = {}
        for published_at, sentiment_score in articles:
            # Round down to the nearest hour
            hour_timestamp = (published_at // (60 * 60 * 1000)) * (60 * 60 * 1000)
            
            if hour_timestamp not in hourly_data:
                hourly_data[hour_timestamp] = []
            hourly_data[hour_timestamp].append(sentiment_score)
        
        # Insert/update hourly sentiment data
        for hour_timestamp, sentiment_scores in hourly_data.items():
            avg_sentiment = sum(sentiment_scores) / len(sentiment_scores)
            article_count = len(sentiment_scores)
            
            cursor.execute("""
                INSERT OR REPLACE INTO news_sentiment_hourly
                (symbol, hour_timestamp, avg_sentiment, article_count)
                VALUES (?, ?, ?, ?)
            """, (symbol, hour_timestamp, avg_sentiment, article_count))
        
        conn.commit()
        logger.info(f"Updated hourly sentiment for {symbol}: {len(hourly_data)} hours processed")
        
    except Exception as e:
        logger.error(f"Error updating hourly sentiment for {symbol}: {e}")
    finally:
        if conn:
            conn.close()

def harvest_news_for_symbol(symbol):
    """Harvest news for a single symbol"""
    logger.info(f"Harvesting news for {symbol}")
    
    # Fetch articles - use longer time range for testing (24 hours)
    hours_back = 24  # Use 24 hours for testing instead of short interval
    
    # Check shutdown flag before making API call
    if shutdown_requested:
        logger.info(f"Shutdown requested before fetching news for {symbol}.")
        return

    articles = fetch_news_for_symbol(symbol, hours_back=hours_back)
    
    new_articles_count = 0
    for article in articles:
        if shutdown_requested:
            logger.info(f"Shutdown requested during article processing for {symbol}.")
            break
        try:
            # Analyze sentiment
            text = f"{article.get('title', '')} {article.get('description', '')}"
            if not text.strip():
                logger.warning(f"Skipping article with no text: {article.get('title', 'No title')}")
                continue
                
            sentiment_score = analyze_sentiment_simple(text)
            
            # Store article
            if store_news_article(symbol, article, sentiment_score): # This function now handles its own conn
                new_articles_count += 1
                
        except (ImportError, RuntimeError) as e: # Specific errors that should stop this symbol's harvest
            logger.error(f"Critical sentiment analysis failure for {symbol}: {e}. Stopping harvest for this symbol.")
            # No re-raise here to allow other symbols to be processed if desired,
            # or re-raise if one failure should stop all. For now, stop for this symbol.
            break # Stop processing articles for this symbol
        except Exception as e:
            logger.error(f"Error processing article '{article.get('title', 'No title')}': {e}")
            continue  # Skip this article but continue with others for this symbol
    
    # Update hourly aggregates if not shutting down and new articles were processed
    if not shutdown_requested and new_articles_count > 0:
        update_hourly_sentiment(symbol) # This function now handles its own conn
        logger.info(f"Stored {new_articles_count} new articles for {symbol}")
    elif new_articles_count == 0 and not shutdown_requested:
        logger.info(f"No new articles found or processed for {symbol}")
    elif shutdown_requested:
        logger.info(f"Shutdown requested, hourly sentiment update may be skipped for {symbol}.")


def main():
    """Main news harvesting function"""
    global shutdown_requested
    logger.info("Starting news harvesting...")

    try:
        # Initialize database
        try:
            init_news_database()
        except Exception as e:
            logger.error(f"Failed to initialize news database: {e}. Exiting.")
            return

        if shutdown_requested:
            logger.info("Shutdown requested before news harvesting loop.")
            return
            
        # Harvest news for each symbol
        for symbol in SYMBOLS:
            if shutdown_requested:
                logger.info("Shutdown requested, skipping further symbols.")
                break
            try:
                harvest_news_for_symbol(symbol) # This function now checks shutdown_requested
                if not shutdown_requested: # Avoid sleep if shutting down
                    time.sleep(1)  # Rate limiting
            except Exception as e: # Catch errors from harvest_news_for_symbol if it raises one
                logger.error(f"Error harvesting news for {symbol}: {e}")
        
        if not shutdown_requested:
            logger.info("News harvesting completed for all symbols.")

    except KeyboardInterrupt:
        logger.info("KeyboardInterrupt received. Requesting shutdown...")
        shutdown_requested = True
    except Exception as e:
        logger.error(f"An unexpected error occurred in main: {e}", exc_info=True)
    finally:
        if shutdown_requested:
            logger.info("News harvesting shutdown process initiated.")
        else:
            logger.info("News harvesting run completed normally.")

if __name__ == "__main__":
    # Register signal handlers
    signal.signal(signal.SIGINT, handle_shutdown_signal)
    signal.signal(signal.SIGTERM, handle_shutdown_signal)
    
    main() 