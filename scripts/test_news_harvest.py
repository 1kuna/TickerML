#!/usr/bin/env python3
"""
Test script for news harvesting functionality
"""

import sys
import os
import sqlite3
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root / "raspberry_pi"))

def test_news_database_init():
    """Test 1: Initialize news database tables"""
    logger.info("🔍 Test 1: Testing news database initialization...")
    
    try:
        from news_harvest import init_news_database, DB_PATH
        
        # Initialize database
        init_news_database()
        
        # Check if tables were created
        conn = sqlite3.connect(str(DB_PATH))
        cursor = conn.cursor()
        
        # Check for news_articles table
        cursor.execute("""
            SELECT name FROM sqlite_master 
            WHERE type='table' AND name='news_articles'
        """)
        articles_table = cursor.fetchone()
        
        # Check for news_sentiment_hourly table
        cursor.execute("""
            SELECT name FROM sqlite_master 
            WHERE type='table' AND name='news_sentiment_hourly'
        """)
        sentiment_table = cursor.fetchone()
        
        conn.close()
        
        if articles_table and sentiment_table:
            logger.info("✅ News database tables created successfully")
            return True
        else:
            logger.error("❌ News database tables not created")
            return False
            
    except Exception as e:
        logger.error(f"❌ News database initialization failed: {e}")
        return False

def test_sentiment_analysis():
    """Test 2: Test sentiment analysis functions"""
    logger.info("🔍 Test 2: Testing sentiment analysis...")
    
    try:
        from news_harvest import analyze_sentiment_basic, analyze_sentiment_simple
        
        test_texts = [
            "Bitcoin price surges to new all-time high",
            "Cryptocurrency market crashes amid panic",
            "Ethereum network operates normally today"
        ]
        
        all_passed = True
        
        for text in test_texts:
            # Test basic sentiment
            basic_score = analyze_sentiment_basic(text)
            logger.info(f"Basic sentiment for '{text[:30]}...': {basic_score:.3f}")
            
            # Test simple sentiment (may fall back to basic if Ollama not available)
            simple_score = analyze_sentiment_simple(text)
            logger.info(f"Simple sentiment for '{text[:30]}...': {simple_score:.3f}")
            
            # Check if scores are in valid range
            if not (-1.0 <= basic_score <= 1.0) or not (-1.0 <= simple_score <= 1.0):
                logger.error(f"❌ Sentiment scores out of range")
                all_passed = False
        
        if all_passed:
            logger.info("✅ Sentiment analysis functions working correctly")
            return True
        else:
            logger.error("❌ Sentiment analysis failed")
            return False
            
    except Exception as e:
        logger.error(f"❌ Sentiment analysis test failed: {e}")
        return False

def test_news_fetch_config():
    """Test 3: Test news fetching configuration"""
    logger.info("🔍 Test 3: Testing news fetching configuration...")
    
    try:
        from news_harvest import NEWS_API_KEY, SYMBOLS, fetch_news_for_symbol
        
        logger.info(f"NewsAPI Key configured: {'Yes' if NEWS_API_KEY and NEWS_API_KEY != 'your_newsapi_key_here' else 'No'}")
        logger.info(f"Symbols configured: {SYMBOLS}")
        
        # Test fetching news for first symbol
        if SYMBOLS:
            symbol = SYMBOLS[0]
            logger.info(f"Testing news fetch for {symbol}...")
            
            articles = fetch_news_for_symbol(symbol, hours_back=1)
            logger.info(f"Fetched {len(articles)} articles")
            
            if NEWS_API_KEY and NEWS_API_KEY != "your_newsapi_key_here":
                logger.info("✅ News fetching configuration looks good")
            else:
                logger.warning("⚠️  NewsAPI key not configured - will use fallback sentiment only")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ News fetch configuration test failed: {e}")
        return False

def test_news_storage():
    """Test 4: Test storing and retrieving news articles"""
    logger.info("🔍 Test 4: Testing news storage...")
    
    try:
        from news_harvest import store_news_article, DB_PATH
        
        # Create test article
        test_article = {
            'title': 'Test Bitcoin News Article',
            'description': 'This is a test article about Bitcoin price movements',
            'content': 'Full content of the test article...',
            'source': {'name': 'Test Source'},
            'url': 'https://example.com/test-article',
            'publishedAt': '2024-01-01T12:00:00Z'
        }
        
        # Store the article
        success = store_news_article("BTCUSD", test_article, sentiment_score=0.5)
        
        if success:
            # Check if article was stored
            conn = sqlite3.connect(str(DB_PATH))
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT COUNT(*) FROM news_articles 
                WHERE title = 'Test Bitcoin News Article'
            """)
            count = cursor.fetchone()[0]
            conn.close()
            
            if count > 0:
                logger.info("✅ News storage working correctly")
                return True
            else:
                logger.error("❌ Article not found in database")
                return False
        else:
            logger.error("❌ Failed to store article")
            return False
            
    except Exception as e:
        logger.error(f"❌ News storage test failed: {e}")
        return False

def test_full_harvest_process():
    """Test 5: Test full news harvesting process"""
    logger.info("🔍 Test 5: Testing full harvest process...")
    
    try:
        from news_harvest import harvest_news_for_symbol, SYMBOLS
        
        if not SYMBOLS:
            logger.warning("⚠️  No symbols configured for testing")
            return True
        
        symbol = SYMBOLS[0]
        logger.info(f"Testing full harvest for {symbol}...")
        
        # Run harvest for one symbol
        harvest_news_for_symbol(symbol)
        
        # Check if any data was processed
        conn = sqlite3.connect(str(project_root / "data" / "db" / "crypto_data.db"))
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT COUNT(*) FROM news_articles 
            WHERE symbol = ?
        """, (symbol,))
        article_count = cursor.fetchone()[0]
        
        cursor.execute("""
            SELECT COUNT(*) FROM news_sentiment_hourly 
            WHERE symbol = ?
        """, (symbol,))
        sentiment_count = cursor.fetchone()[0]
        
        conn.close()
        
        logger.info(f"Articles in database for {symbol}: {article_count}")
        logger.info(f"Sentiment records for {symbol}: {sentiment_count}")
        
        logger.info("✅ Full harvest process completed")
        return True
        
    except Exception as e:
        logger.error(f"❌ Full harvest test failed: {e}")
        return False

def main():
    """Run all news harvest tests"""
    logger.info("🚀 Starting News Harvest Tests")
    logger.info("=" * 50)
    
    tests = [
        test_news_database_init,
        test_sentiment_analysis,
        test_news_fetch_config,
        test_news_storage,
        test_full_harvest_process
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            if test():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            logger.error(f"Test {test.__name__} crashed: {e}")
            failed += 1
        
        logger.info("-" * 30)
    
    logger.info("=" * 50)
    logger.info(f"📊 Test Results: {passed} passed, {failed} failed")
    
    if failed == 0:
        logger.info("🎉 All tests passed!")
    else:
        logger.warning(f"⚠️  {failed} tests failed")
    
    logger.info("\n📋 Next Steps:")
    logger.info("1. Set up NewsAPI key in config/config.yaml for full functionality")
    logger.info("2. Add to crontab for automated harvesting:")
    logger.info(f"   */15 * * * * cd {project_root} && python raspberry_pi/news_harvest.py >> logs/news_harvest.log 2>&1")
    logger.info("3. Monitor logs: tail -f logs/news_harvest.log")

if __name__ == "__main__":
    main() 