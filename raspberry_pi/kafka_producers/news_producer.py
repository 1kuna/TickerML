#!/usr/bin/env python3
"""
Kafka Producer for News and Sentiment Data
Streams crypto news and sentiment scores to Kafka topics
"""

import json
import logging
import asyncio
from typing import Dict, Any, Optional
from datetime import datetime, timedelta
import yaml
from kafka import KafkaProducer
from kafka.errors import KafkaError
import coloredlogs
import schedule
import time

# Local imports
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from raspberry_pi.news_harvest import fetch_crypto_news, calculate_sentiment

class NewsProducer:
    """Produces news and sentiment data to Kafka topics"""
    
    def __init__(self, config_path: str = 'config/kafka_config.yaml'):
        """Initialize Kafka producer for news data"""
        self.logger = logging.getLogger(self.__class__.__name__)
        coloredlogs.install(level='INFO', logger=self.logger)
        
        # Load configuration
        with open(config_path, 'r') as f:
            kafka_config = yaml.safe_load(f)
            
        self.config = kafka_config['kafka']
        self.topic = self.config['topics']['news']
        
        # Initialize Kafka producer
        self.producer = KafkaProducer(
            bootstrap_servers=self.config['brokers'],
            value_serializer=lambda v: json.dumps(v).encode('utf-8'),
            key_serializer=lambda k: k.encode('utf-8') if k else None,
            compression_type='gzip',
            batch_size=16384,
            linger_ms=1000,  # Can wait longer for news
            acks='all',
            retries=3
        )
        
        # News collection interval (minutes)
        self.collection_interval = 15
        self.last_collection_time = None
        
        self.logger.info(f"Initialized NewsProducer with topic: {self.topic}")
    
    def _create_message(self, article: Dict[str, Any], sentiment_score: float) -> Dict[str, Any]:
        """Create Kafka message from news article"""
        return {
            'timestamp': datetime.utcnow().isoformat(),
            'published_at': article.get('publishedAt'),
            'source': article.get('source', {}).get('name'),
            'title': article.get('title'),
            'description': article.get('description'),
            'url': article.get('url'),
            'sentiment': {
                'score': sentiment_score,
                'category': self._categorize_sentiment(sentiment_score)
            },
            'metadata': {
                'author': article.get('author'),
                'content_preview': article.get('content', '')[:500] if article.get('content') else None,
                'collection_timestamp': datetime.utcnow().isoformat()
            }
        }
    
    def _categorize_sentiment(self, score: float) -> str:
        """Categorize sentiment score"""
        if score >= 0.6:
            return 'very_positive'
        elif score >= 0.2:
            return 'positive'
        elif score >= -0.2:
            return 'neutral'
        elif score >= -0.6:
            return 'negative'
        else:
            return 'very_negative'
    
    def _create_aggregate_message(self, articles: list, sentiment_scores: list) -> Dict[str, Any]:
        """Create aggregate sentiment message"""
        if not sentiment_scores:
            return None
            
        avg_sentiment = sum(sentiment_scores) / len(sentiment_scores)
        
        # Count sentiment categories
        categories = [self._categorize_sentiment(s) for s in sentiment_scores]
        category_counts = {cat: categories.count(cat) for cat in set(categories)}
        
        return {
            'timestamp': datetime.utcnow().isoformat(),
            'window_minutes': self.collection_interval,
            'article_count': len(articles),
            'sentiment': {
                'average': avg_sentiment,
                'min': min(sentiment_scores),
                'max': max(sentiment_scores),
                'std': self._calculate_std(sentiment_scores),
                'category_distribution': category_counts,
                'market_regime': self._determine_market_regime(avg_sentiment, self._calculate_std(sentiment_scores))
            },
            'metadata': {
                'sources': list(set(a.get('source', {}).get('name', 'Unknown') for a in articles)),
                'top_keywords': self._extract_keywords(articles)
            }
        }
    
    def _calculate_std(self, values: list) -> float:
        """Calculate standard deviation"""
        if len(values) < 2:
            return 0.0
        avg = sum(values) / len(values)
        variance = sum((x - avg) ** 2 for x in values) / (len(values) - 1)
        return variance ** 0.5
    
    def _determine_market_regime(self, avg_sentiment: float, std_sentiment: float) -> str:
        """Determine market regime based on sentiment"""
        if avg_sentiment > 0.4 and std_sentiment < 0.3:
            return 'euphoria'
        elif avg_sentiment > 0.2:
            return 'optimistic'
        elif avg_sentiment < -0.4 and std_sentiment < 0.3:
            return 'panic'
        elif avg_sentiment < -0.2:
            return 'pessimistic'
        elif std_sentiment > 0.5:
            return 'uncertain'
        else:
            return 'neutral'
    
    def _extract_keywords(self, articles: list, top_n: int = 10) -> list:
        """Extract top keywords from articles"""
        # Simple keyword extraction (could be enhanced with NLP)
        keywords = {}
        crypto_terms = ['bitcoin', 'btc', 'ethereum', 'eth', 'crypto', 'blockchain', 
                       'defi', 'nft', 'trading', 'price', 'market', 'bull', 'bear']
        
        for article in articles:
            text = f"{article.get('title', '')} {article.get('description', '')}".lower()
            for term in crypto_terms:
                if term in text:
                    keywords[term] = keywords.get(term, 0) + 1
        
        # Sort by frequency
        sorted_keywords = sorted(keywords.items(), key=lambda x: x[1], reverse=True)
        return [k[0] for k in sorted_keywords[:top_n]]
    
    async def collect_and_send_news(self):
        """Collect news and send to Kafka"""
        try:
            self.logger.info("Collecting crypto news...")
            
            # Fetch news
            articles = fetch_crypto_news()
            
            if not articles:
                self.logger.warning("No news articles fetched")
                return
            
            sentiment_scores = []
            
            # Process each article
            for article in articles:
                try:
                    # Calculate sentiment
                    sentiment_score = calculate_sentiment(
                        article.get('title', ''),
                        article.get('description', '')
                    )
                    sentiment_scores.append(sentiment_score)
                    
                    # Create and send message
                    message = self._create_message(article, sentiment_score)
                    
                    # Use source as key for partitioning
                    key = message['source'] or 'unknown'
                    
                    self.producer.send(
                        self.topic,
                        key=key,
                        value=message
                    )
                    
                except Exception as e:
                    self.logger.error(f"Error processing article: {e}")
            
            # Send aggregate sentiment
            aggregate = self._create_aggregate_message(articles, sentiment_scores)
            if aggregate:
                self.producer.send(
                    f"{self.topic}-sentiment",
                    key='aggregate',
                    value=aggregate
                )
                self.logger.info(f"Sent aggregate sentiment: {aggregate['sentiment']['average']:.3f} "
                               f"({aggregate['sentiment']['market_regime']})")
            
            # Flush to ensure delivery
            self.producer.flush()
            
            self.logger.info(f"Processed {len(articles)} news articles")
            self.last_collection_time = datetime.utcnow()
            
        except Exception as e:
            self.logger.error(f"Error collecting news: {e}")
    
    async def run(self):
        """Main loop to periodically collect and produce news data"""
        self.logger.info("Starting news producer...")
        
        try:
            # Initial collection
            await self.collect_and_send_news()
            
            # Schedule periodic collection
            while True:
                # Check if it's time to collect
                if self.last_collection_time:
                    time_since_last = datetime.utcnow() - self.last_collection_time
                    if time_since_last >= timedelta(minutes=self.collection_interval):
                        await self.collect_and_send_news()
                else:
                    await self.collect_and_send_news()
                
                # Sleep for a minute before checking again
                await asyncio.sleep(60)
                
        except KeyboardInterrupt:
            self.logger.info("Shutting down news producer...")
        except Exception as e:
            self.logger.error(f"Producer error: {e}")
        finally:
            # Clean shutdown
            self.producer.flush()
            self.producer.close()
            self.logger.info("News producer stopped")
    
    def health_check(self) -> Dict[str, Any]:
        """Check producer health"""
        metrics = self.producer.metrics()
        
        time_since_last = None
        if self.last_collection_time:
            time_since_last = (datetime.utcnow() - self.last_collection_time).total_seconds()
        
        return {
            'producer_active': True,
            'last_collection_seconds_ago': time_since_last,
            'collection_interval_minutes': self.collection_interval,
            'record_send_rate': metrics.get('record-send-rate', 0),
            'record_error_rate': metrics.get('record-error-rate', 0),
            'is_overdue': time_since_last > (self.collection_interval * 60 * 1.5) if time_since_last else False
        }


def main():
    """Main entry point"""
    producer = NewsProducer()
    asyncio.run(producer.run())


if __name__ == "__main__":
    main()