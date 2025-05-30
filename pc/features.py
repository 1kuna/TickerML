#!/usr/bin/env python3
"""
Feature engineering script for PC
Computes technical indicators and sentiment analysis
"""

import pandas as pd
import numpy as np
import ta
import requests
import logging
from datetime import datetime, timedelta
from pathlib import Path
import ollama
import sqlite3
import yaml
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration
DATA_PATH = Path(__file__).parent.parent / "data" / "dumps"
OUTPUT_PATH = Path(__file__).parent.parent / "data" / "features"
CONFIG_PATH = Path(__file__).parent.parent / "config" / "config.yaml"

# Load configuration
def load_config():
    """Load configuration from config.yaml"""
    try:
        if CONFIG_PATH.exists():
            with open(CONFIG_PATH, 'r') as f:
                return yaml.safe_load(f)
        else:
            logger.warning("Config file not found, using defaults")
            return {}
    except Exception as e:
        logger.error(f"Error loading config: {e}")
        return {}

config = load_config()
NEWS_API_KEY = os.getenv("NEWS_API_KEY", config.get("features", {}).get("sentiment", {}).get("news_api_key", "your_newsapi_key_here"))
OLLAMA_HOST = os.getenv("OLLAMA_HOST", config.get("features", {}).get("sentiment", {}).get("ollama_host", "http://localhost:11434"))
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", config.get("features", {}).get("sentiment", {}).get("model", "gemma3:4b"))
SYMBOLS = config.get("data", {}).get("symbols", ["BTCUSDT", "ETHUSDT"])

def load_price_data(symbol, days=30):
    """Load price data from CSV dumps"""
    try:
        # Get list of CSV files for the symbol
        csv_files = list(DATA_PATH.glob(f"{symbol}_*.csv"))
        
        if not csv_files:
            logger.warning(f"No CSV files found for {symbol}")
            return None
        
        # Sort by date and take the most recent files
        csv_files.sort(reverse=True)
        csv_files = csv_files[:days]  # Take last N days
        
        # Load and combine data
        dfs = []
        for csv_file in csv_files:
            try:
                df = pd.read_csv(csv_file)
                dfs.append(df)
            except Exception as e:
                logger.warning(f"Error loading {csv_file}: {e}")
                continue
        
        if not dfs:
            logger.error(f"No valid CSV files loaded for {symbol}")
            return None
        
        # Combine all dataframes
        combined_df = pd.concat(dfs, ignore_index=True)
        
        # Remove duplicates and sort by timestamp
        combined_df = combined_df.drop_duplicates(subset=['timestamp']).sort_values('timestamp')
        
        # Convert timestamp to datetime
        combined_df['datetime'] = pd.to_datetime(combined_df['timestamp'], unit='ms')
        
        logger.info(f"Loaded {len(combined_df)} records for {symbol}")
        return combined_df
        
    except Exception as e:
        logger.error(f"Error loading price data for {symbol}: {e}")
        return None

def compute_technical_indicators(df):
    """Compute technical indicators using ta library"""
    try:
        # Make a copy to avoid modifying original
        df = df.copy()

        df = _compute_basic_price_features(df)
        df = _compute_moving_averages(df)
        df = _compute_macd(df)
        df = _compute_rsi(df)
        df = _compute_bollinger_bands(df)
        df = _compute_stochastic_oscillator(df)
        df = _compute_williams_r(df)
        df = _compute_atr(df)
        df = _compute_volume_indicators(df)
        df = _compute_cci(df)
        df = _compute_mfi(df)
        df = _compute_roc(df)
        df = _compute_volatility_features(df)
        df = _compute_price_position_features(df)
        df = _compute_momentum_features(df)

        logger.info(f"Computed technical indicators, shape: {df.shape}")
        return df

    except Exception as e:
        logger.error(f"Error computing technical indicators: {e}")
        return df


def _compute_basic_price_features(df: pd.DataFrame) -> pd.DataFrame:
    """Computes basic price features."""
    df['price_change'] = df['close'].pct_change()
    df['volume_change'] = df['volume'].pct_change()
    df['high_low_pct'] = (df['high'] - df['low']) / df['close']
    df['open_close_pct'] = (df['close'] - df['open']) / df['open']
    return df


def _compute_moving_averages(df: pd.DataFrame) -> pd.DataFrame:
    """Computes moving averages (SMA, EMA)."""
    df['sma_5'] = ta.trend.sma_indicator(df['close'], window=5)
    df['sma_10'] = ta.trend.sma_indicator(df['close'], window=10)
    df['sma_20'] = ta.trend.sma_indicator(df['close'], window=20)
    df['sma_50'] = ta.trend.sma_indicator(df['close'], window=50)
    df['ema_12'] = ta.trend.ema_indicator(df['close'], window=12)
    df['ema_26'] = ta.trend.ema_indicator(df['close'], window=26)
    return df


def _compute_macd(df: pd.DataFrame) -> pd.DataFrame:
    """Computes MACD indicators."""
    df['macd'] = ta.trend.macd_diff(df['close'])
    df['macd_signal'] = ta.trend.macd_signal(df['close'])
    df['macd_histogram'] = ta.trend.macd(df['close'])
    return df


def _compute_rsi(df: pd.DataFrame) -> pd.DataFrame:
    """Computes RSI."""
    df['rsi'] = ta.momentum.rsi(df['close'], window=14)
    return df


def _compute_bollinger_bands(df: pd.DataFrame) -> pd.DataFrame:
    """Computes Bollinger Bands."""
    bb_indicator = ta.volatility.BollingerBands(df['close'], window=20, window_dev=2)
    df['bb_upper'] = bb_indicator.bollinger_hband()
    df['bb_middle'] = bb_indicator.bollinger_mavg()
    df['bb_lower'] = bb_indicator.bollinger_lband()
    df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
    df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
    return df


def _compute_stochastic_oscillator(df: pd.DataFrame) -> pd.DataFrame:
    """Computes Stochastic Oscillator."""
    df['stoch_k'] = ta.momentum.stoch(df['high'], df['low'], df['close'])
    df['stoch_d'] = ta.momentum.stoch_signal(df['high'], df['low'], df['close'])
    return df


def _compute_williams_r(df: pd.DataFrame) -> pd.DataFrame:
    """Computes Williams %R."""
    df['williams_r'] = ta.momentum.williams_r(df['high'], df['low'], df['close'])
    return df


def _compute_atr(df: pd.DataFrame) -> pd.DataFrame:
    """Computes Average True Range (ATR)."""
    df['atr'] = ta.volatility.average_true_range(df['high'], df['low'], df['close'])
    return df


def _compute_volume_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Computes volume indicators."""
    df['volume_sma'] = ta.volume.volume_sma(df['close'], df['volume'], window=20) # Original had df['close'] as first arg, but ta.volume.volume_sma expects series_close, series_volume. Corrected.
    df['volume_weighted_price'] = ta.volume.volume_weighted_average_price(
        df['high'], df['low'], df['close'], df['volume'], window=14
    )
    return df


def _compute_cci(df: pd.DataFrame) -> pd.DataFrame:
    """Computes Commodity Channel Index (CCI)."""
    df['cci'] = ta.trend.cci(df['high'], df['low'], df['close'], window=20)
    return df


def _compute_mfi(df: pd.DataFrame) -> pd.DataFrame:
    """Computes Money Flow Index (MFI)."""
    df['mfi'] = ta.volume.money_flow_index(
        df['high'], df['low'], df['close'], df['volume'], window=14
    )
    return df


def _compute_roc(df: pd.DataFrame) -> pd.DataFrame:
    """Computes Rate of Change (ROC)."""
    df['roc'] = ta.momentum.roc(df['close'], window=12)
    return df


def _compute_volatility_features(df: pd.DataFrame) -> pd.DataFrame:
    """Computes volatility features."""
    df['volatility_5'] = df['price_change'].rolling(5).std()
    df['volatility_10'] = df['price_change'].rolling(10).std()
    df['volatility_20'] = df['price_change'].rolling(20).std()
    return df


def _compute_price_position_features(df: pd.DataFrame) -> pd.DataFrame:
    """Computes price position features."""
    df['price_position_5'] = (df['close'] - df['close'].rolling(5).min()) / (
        df['close'].rolling(5).max() - df['close'].rolling(5).min()
    )
    df['price_position_20'] = (df['close'] - df['close'].rolling(20).min()) / (
        df['close'].rolling(20).max() - df['close'].rolling(20).min()
    )
    return df


def _compute_momentum_features(df: pd.DataFrame) -> pd.DataFrame:
    """Computes momentum features."""
    df['momentum_5'] = df['close'] / df['close'].shift(5) - 1
    df['momentum_10'] = df['close'] / df['close'].shift(10) - 1
    df['momentum_20'] = df['close'] / df['close'].shift(20) - 1
    return df


def analyze_sentiment_with_gemma3(text):
    """Analyze sentiment using Gemma 3 4B LLM via Ollama"""
    try:
        # Prepare the prompt for sentiment analysis
        prompt = f"""Analyze the sentiment of the following financial news text and return only a numerical score between -1 and 1, where:
-1 = very negative sentiment
0 = neutral sentiment  
1 = very positive sentiment

Text: "{text}"

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
            logger.warning(f"Could not parse sentiment score from Gemma 3 response: {sentiment_text}")
            return 0.0
            
    except Exception as e:
        logger.error(f"Error analyzing sentiment with Gemma 3: {e}")
        # Fallback to neutral sentiment
        return 0.0

def fetch_news_sentiment(symbol_name, hours=24, use_stored_data=True):
    """Fetch news and compute sentiment scores - now supports using stored data"""
    try:
        if use_stored_data:
            # Try to use stored news data first
            sentiment_scores = fetch_stored_sentiment(symbol_name, hours)
            if sentiment_scores:
                logger.info(f"Using stored sentiment data for {symbol_name}: {len(sentiment_scores)} records")
                return sentiment_scores
            else:
                logger.warning(f"No stored sentiment data found for {symbol_name}, falling back to live fetch")
        
        # Original live fetching code
        if NEWS_API_KEY == "your_newsapi_key_here":
            logger.warning("NewsAPI key not configured, using dummy sentiment")
            return generate_dummy_sentiment(hours)
        
        # Map crypto symbols to search terms
        search_terms = {
            "BTCUSDT": "Bitcoin BTC cryptocurrency",
            "ETHUSDT": "Ethereum ETH cryptocurrency"
        }
        
        query = search_terms.get(symbol_name, "cryptocurrency")
        
        # Calculate time range
        end_time = datetime.now()
        start_time = end_time - timedelta(hours=hours)
        
        # NewsAPI request
        url = "https://newsapi.org/v2/everything"
        params = {
            'q': query,
            'from': start_time.isoformat(),
            'to': end_time.isoformat(),
            'sortBy': 'publishedAt',
            'language': 'en',
            'apiKey': NEWS_API_KEY
        }
        
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        
        news_data = response.json()
        articles = news_data.get('articles', [])
        
        if not articles:
            logger.warning(f"No news articles found for {symbol_name}")
            return generate_dummy_sentiment(hours)
        
        # Process articles and compute sentiment
        sentiment_scores = []
        
        for article in articles:
            try:
                # Combine title and description
                text = f"{article.get('title', '')} {article.get('description', '')}"
                
                if not text.strip():
                    continue
                
                # Compute sentiment using Gemma 3
                sentiment = analyze_sentiment_with_gemma3(text)
                
                # Parse published date
                published_at = datetime.fromisoformat(
                    article['publishedAt'].replace('Z', '+00:00')
                )
                
                sentiment_scores.append({
                    'timestamp': int(published_at.timestamp() * 1000),
                    'sentiment': sentiment,
                    'title': article.get('title', '')
                })
                
            except Exception as e:
                logger.warning(f"Error processing article: {e}")
                continue
        
        logger.info(f"Processed {len(sentiment_scores)} news articles for {symbol_name}")
        return sentiment_scores
        
    except Exception as e:
        logger.error(f"Error fetching news sentiment: {e}")
        return generate_dummy_sentiment(hours)

def fetch_stored_sentiment(symbol_name, hours=24):
    """Fetch sentiment data from the database"""
    try:
        # Check if database exists and has news tables
        db_path = Path(__file__).parent.parent / "data" / "db" / "crypto_news.db"
        if not db_path.exists():
            return []
        
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()
        
        # Check if news_sentiment_hourly table exists
        cursor.execute("""
            SELECT name FROM sqlite_master 
            WHERE type='table' AND name='news_sentiment_hourly'
        """)
        if not cursor.fetchone():
            conn.close()
            return []
        
        # Calculate time range
        end_time = datetime.now()
        start_time = end_time - timedelta(hours=hours)
        
        start_timestamp = int(start_time.timestamp() * 1000)
        end_timestamp = int(end_time.timestamp() * 1000)
        
        # Fetch hourly sentiment data
        cursor.execute("""
            SELECT hour_timestamp, avg_sentiment, article_count
            FROM news_sentiment_hourly 
            WHERE symbol = ? 
            AND hour_timestamp >= ? 
            AND hour_timestamp <= ?
            ORDER BY hour_timestamp
        """, (symbol_name, start_timestamp, end_timestamp))
        
        rows = cursor.fetchall()
        conn.close()
        
        if not rows:
            return []
        
        # Convert to the format expected by the rest of the code
        sentiment_scores = []
        for hour_timestamp, avg_sentiment, article_count in rows:
            sentiment_scores.append({
                'timestamp': hour_timestamp,
                'sentiment': avg_sentiment,
                'title': f"Aggregated sentiment from {article_count} articles"
            })
        
        return sentiment_scores
        
    except Exception as e:
        logger.error(f"Error fetching stored sentiment: {e}")
        return []

def generate_dummy_sentiment(hours=24):
    """Generate dummy sentiment data for testing"""
    sentiment_scores = []
    
    # Generate hourly sentiment scores
    for i in range(hours):
        timestamp = int((datetime.now() - timedelta(hours=i)).timestamp() * 1000)
        # Random sentiment between -0.5 and 0.5
        sentiment = np.random.uniform(-0.5, 0.5)
        
        sentiment_scores.append({
            'timestamp': timestamp,
            'sentiment': sentiment,
            'title': f"Dummy news article {i}"
        })
    
    return sentiment_scores

def merge_sentiment_with_price(df, sentiment_scores):
    """Merge sentiment scores with price data"""
    try:
        # Convert sentiment to DataFrame
        sentiment_df = pd.DataFrame(sentiment_scores)
        
        if sentiment_df.empty:
            # Add dummy sentiment columns
            df['sentiment_1h'] = 0.0
            df['sentiment_4h'] = 0.0
            df['sentiment_24h'] = 0.0
            return df
        
        sentiment_df['datetime'] = pd.to_datetime(sentiment_df['timestamp'], unit='ms')
        
        # Resample sentiment to hourly averages
        sentiment_hourly = sentiment_df.set_index('datetime').resample('1H')['sentiment'].mean()
        
        # Merge with price data
        df = df.set_index('datetime')
        
        # Add sentiment features with different time windows
        df['sentiment_1h'] = sentiment_hourly.reindex(df.index, method='ffill')
        df['sentiment_4h'] = df['sentiment_1h'].rolling(4).mean()
        df['sentiment_24h'] = df['sentiment_1h'].rolling(24).mean()
        
        # Fill NaN values
        df['sentiment_1h'] = df['sentiment_1h'].fillna(0)
        df['sentiment_4h'] = df['sentiment_4h'].fillna(0)
        df['sentiment_24h'] = df['sentiment_24h'].fillna(0)
        
        df = df.reset_index()
        
        logger.info("Merged sentiment data with price data")
        return df
        
    except Exception as e:
        logger.error(f"Error merging sentiment data: {e}")
        # Add dummy sentiment columns
        df['sentiment_1h'] = 0.0
        df['sentiment_4h'] = 0.0
        df['sentiment_24h'] = 0.0
        return df

def create_target_variables(df):
    """Create target variables for prediction"""
    try:
        # Future price targets (5, 10, 30 minutes ahead)
        df['target_5min'] = df['close'].shift(-5)
        df['target_10min'] = df['close'].shift(-10)
        df['target_30min'] = df['close'].shift(-30)
        
        # Direction targets (up/down)
        df['target_direction_5min'] = (df['target_5min'] > df['close']).astype(int)
        df['target_direction_10min'] = (df['target_10min'] > df['close']).astype(int)
        df['target_direction_30min'] = (df['target_30min'] > df['close']).astype(int)
        
        # Price change percentage targets
        df['target_change_5min'] = (df['target_5min'] / df['close'] - 1) * 100
        df['target_change_10min'] = (df['target_10min'] / df['close'] - 1) * 100
        df['target_change_30min'] = (df['target_30min'] / df['close'] - 1) * 100
        
        logger.info("Created target variables")
        return df
        
    except Exception as e:
        logger.error(f"Error creating target variables: {e}")
        return df

def save_features(df, symbol):
    """Save processed features to file"""
    try:
        OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
        
        # Save as CSV
        csv_path = OUTPUT_PATH / f"{symbol}_features.csv"
        df.to_csv(csv_path, index=False)
        
        # Save as pickle for faster loading
        pickle_path = OUTPUT_PATH / f"{symbol}_features.pkl"
        df.to_pickle(pickle_path)
        
        logger.info(f"Saved features for {symbol} to {csv_path}")
        
        # Print feature summary
        logger.info(f"Feature summary for {symbol}:")
        logger.info(f"  Shape: {df.shape}")
        logger.info(f"  Date range: {df['datetime'].min()} to {df['datetime'].max()}")
        logger.info(f"  Missing values: {df.isnull().sum().sum()}")
        
        return csv_path
        
    except Exception as e:
        logger.error(f"Error saving features for {symbol}: {e}")
        return None

def main():
    """Main feature engineering pipeline"""
    logger.info("Starting feature engineering pipeline")
    
    for symbol in SYMBOLS:
        logger.info(f"Processing {symbol}")
        
        # Load price data
        df = load_price_data(symbol)
        if df is None:
            logger.warning(f"Skipping {symbol} due to missing data")
            continue
        
        # Compute technical indicators
        df = compute_technical_indicators(df)
        
        # Fetch and merge sentiment data
        sentiment_scores = fetch_news_sentiment(symbol)
        df = merge_sentiment_with_price(df, sentiment_scores)
        
        # Create target variables
        df = create_target_variables(df)
        
        # Save processed features
        save_features(df, symbol)
    
    logger.info("Feature engineering pipeline completed")

if __name__ == "__main__":
    main() 