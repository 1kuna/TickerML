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
from textblob import TextBlob
import sqlite3

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration
DATA_PATH = Path(__file__).parent.parent / "data" / "dumps"
OUTPUT_PATH = Path(__file__).parent.parent / "data" / "features"
NEWS_API_KEY = "your_newsapi_key_here"  # Replace with actual API key
SYMBOLS = ["BTCUSDT", "ETHUSDT"]

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
        
        # Basic price features
        df['price_change'] = df['close'].pct_change()
        df['volume_change'] = df['volume'].pct_change()
        df['high_low_pct'] = (df['high'] - df['low']) / df['close']
        df['open_close_pct'] = (df['close'] - df['open']) / df['open']
        
        # Moving averages
        df['sma_5'] = ta.trend.sma_indicator(df['close'], window=5)
        df['sma_10'] = ta.trend.sma_indicator(df['close'], window=10)
        df['sma_20'] = ta.trend.sma_indicator(df['close'], window=20)
        df['sma_50'] = ta.trend.sma_indicator(df['close'], window=50)
        
        # Exponential moving averages
        df['ema_12'] = ta.trend.ema_indicator(df['close'], window=12)
        df['ema_26'] = ta.trend.ema_indicator(df['close'], window=26)
        
        # MACD
        df['macd'] = ta.trend.macd_diff(df['close'])
        df['macd_signal'] = ta.trend.macd_signal(df['close'])
        df['macd_histogram'] = ta.trend.macd(df['close'])
        
        # RSI
        df['rsi'] = ta.momentum.rsi(df['close'], window=14)
        
        # Bollinger Bands
        bb_indicator = ta.volatility.BollingerBands(df['close'], window=20, window_dev=2)
        df['bb_upper'] = bb_indicator.bollinger_hband()
        df['bb_middle'] = bb_indicator.bollinger_mavg()
        df['bb_lower'] = bb_indicator.bollinger_lband()
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
        df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        
        # Stochastic Oscillator
        df['stoch_k'] = ta.momentum.stoch(df['high'], df['low'], df['close'])
        df['stoch_d'] = ta.momentum.stoch_signal(df['high'], df['low'], df['close'])
        
        # Williams %R
        df['williams_r'] = ta.momentum.williams_r(df['high'], df['low'], df['close'])
        
        # Average True Range (ATR)
        df['atr'] = ta.volatility.average_true_range(df['high'], df['low'], df['close'])
        
        # Volume indicators
        df['volume_sma'] = ta.volume.volume_sma(df['close'], df['volume'], window=20)
        df['volume_weighted_price'] = ta.volume.volume_weighted_average_price(
            df['high'], df['low'], df['close'], df['volume'], window=14
        )
        
        # Commodity Channel Index
        df['cci'] = ta.trend.cci(df['high'], df['low'], df['close'], window=20)
        
        # Money Flow Index
        df['mfi'] = ta.volume.money_flow_index(
            df['high'], df['low'], df['close'], df['volume'], window=14
        )
        
        # Rate of Change
        df['roc'] = ta.momentum.roc(df['close'], window=12)
        
        # Volatility features
        df['volatility_5'] = df['price_change'].rolling(5).std()
        df['volatility_10'] = df['price_change'].rolling(10).std()
        df['volatility_20'] = df['price_change'].rolling(20).std()
        
        # Price position features
        df['price_position_5'] = (df['close'] - df['close'].rolling(5).min()) / (
            df['close'].rolling(5).max() - df['close'].rolling(5).min()
        )
        df['price_position_20'] = (df['close'] - df['close'].rolling(20).min()) / (
            df['close'].rolling(20).max() - df['close'].rolling(20).min()
        )
        
        # Momentum features
        df['momentum_5'] = df['close'] / df['close'].shift(5) - 1
        df['momentum_10'] = df['close'] / df['close'].shift(10) - 1
        df['momentum_20'] = df['close'] / df['close'].shift(20) - 1
        
        logger.info(f"Computed technical indicators, shape: {df.shape}")
        return df
        
    except Exception as e:
        logger.error(f"Error computing technical indicators: {e}")
        return df

def fetch_news_sentiment(symbol_name, hours=24):
    """Fetch news and compute sentiment scores"""
    try:
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
                
                # Compute sentiment using TextBlob
                blob = TextBlob(text)
                sentiment = blob.sentiment.polarity  # -1 to 1
                
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