"""
Redis service for caching and real-time data management
"""

import json
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import aioredis
import asyncio
from pathlib import Path

logger = logging.getLogger(__name__)

class RedisService:
    """Redis service for caching and real-time data"""
    
    def __init__(self):
        self.redis: Optional[aioredis.Redis] = None
        self.connection_pool = None
        
    async def connect(self, redis_url: str = "redis://localhost:6379"):
        """Connect to Redis"""
        try:
            self.connection_pool = aioredis.ConnectionPool.from_url(
                redis_url, max_connections=20, retry_on_timeout=True
            )
            self.redis = aioredis.Redis(connection_pool=self.connection_pool)
            
            # Test connection
            await self.redis.ping()
            logger.info("Connected to Redis successfully")
            
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
            self.redis = None
            raise
    
    async def disconnect(self):
        """Disconnect from Redis"""
        if self.redis:
            await self.redis.aclose()
            logger.info("Disconnected from Redis")
    
    async def is_connected(self) -> bool:
        """Check if Redis is connected"""
        if not self.redis:
            return False
        try:
            await self.redis.ping()
            return True
        except Exception:
            return False
    
    # Cache operations
    async def set(self, key: str, value: Any, expire: Optional[int] = None) -> bool:
        """Set a value in Redis with optional expiration"""
        if not self.redis:
            return False
        
        try:
            # Serialize complex objects to JSON
            if isinstance(value, (dict, list)):
                value = json.dumps(value, default=str)
            
            await self.redis.set(key, value, ex=expire)
            return True
        except Exception as e:
            logger.error(f"Failed to set Redis key {key}: {e}")
            return False
    
    async def get(self, key: str) -> Optional[Any]:
        """Get a value from Redis"""
        if not self.redis:
            return None
        
        try:
            value = await self.redis.get(key)
            if value is None:
                return None
            
            # Try to deserialize JSON
            try:
                return json.loads(value)
            except (json.JSONDecodeError, TypeError):
                return value.decode() if isinstance(value, bytes) else value
                
        except Exception as e:
            logger.error(f"Failed to get Redis key {key}: {e}")
            return None
    
    async def delete(self, key: str) -> bool:
        """Delete a key from Redis"""
        if not self.redis:
            return False
        
        try:
            result = await self.redis.delete(key)
            return result > 0
        except Exception as e:
            logger.error(f"Failed to delete Redis key {key}: {e}")
            return False
    
    async def exists(self, key: str) -> bool:
        """Check if a key exists in Redis"""
        if not self.redis:
            return False
        
        try:
            return await self.redis.exists(key) > 0
        except Exception as e:
            logger.error(f"Failed to check Redis key {key}: {e}")
            return False
    
    # Market data caching
    async def cache_market_data(self, symbol: str, exchange: str, data: Dict, expire: int = 300):
        """Cache market data (5 minute default expiration)"""
        key = f"market:{exchange}:{symbol}"
        data["cached_at"] = datetime.now().isoformat()
        await self.set(key, data, expire)
    
    async def get_market_data(self, symbol: str, exchange: str) -> Optional[Dict]:
        """Get cached market data"""
        key = f"market:{exchange}:{symbol}"
        return await self.get(key)
    
    async def cache_order_book(self, symbol: str, exchange: str, order_book: Dict, expire: int = 30):
        """Cache order book data (30 second expiration)"""
        key = f"orderbook:{exchange}:{symbol}"
        order_book["cached_at"] = datetime.now().isoformat()
        await self.set(key, order_book, expire)
    
    async def get_order_book(self, symbol: str, exchange: str) -> Optional[Dict]:
        """Get cached order book"""
        key = f"orderbook:{exchange}:{symbol}"
        return await self.get(key)
    
    # Portfolio caching
    async def cache_portfolio_snapshot(self, portfolio_data: Dict, expire: int = 60):
        """Cache portfolio snapshot (1 minute expiration)"""
        key = "portfolio:snapshot"
        portfolio_data["cached_at"] = datetime.now().isoformat()
        await self.set(key, portfolio_data, expire)
    
    async def get_portfolio_snapshot(self) -> Optional[Dict]:
        """Get cached portfolio snapshot"""
        key = "portfolio:snapshot"
        return await self.get(key)
    
    # Session management
    async def store_user_session(self, user_id: str, session_data: Dict, expire: int = 3600):
        """Store user session data (1 hour default)"""
        key = f"session:{user_id}"
        session_data["created_at"] = datetime.now().isoformat()
        await self.set(key, session_data, expire)
    
    async def get_user_session(self, user_id: str) -> Optional[Dict]:
        """Get user session data"""
        key = f"session:{user_id}"
        return await self.get(key)
    
    async def invalidate_user_session(self, user_id: str):
        """Invalidate user session"""
        key = f"session:{user_id}"
        await self.delete(key)
    
    # Real-time updates
    async def publish_market_update(self, channel: str, data: Dict):
        """Publish market update to Redis pub/sub"""
        if not self.redis:
            return False
        
        try:
            message = json.dumps(data, default=str)
            await self.redis.publish(channel, message)
            return True
        except Exception as e:
            logger.error(f"Failed to publish to channel {channel}: {e}")
            return False
    
    async def subscribe_to_updates(self, channels: List[str]):
        """Subscribe to real-time updates"""
        if not self.redis:
            return None
        
        try:
            pubsub = self.redis.pubsub()
            await pubsub.subscribe(*channels)
            return pubsub
        except Exception as e:
            logger.error(f"Failed to subscribe to channels {channels}: {e}")
            return None
    
    # System metrics caching
    async def cache_system_metrics(self, metrics: Dict, expire: int = 60):
        """Cache system metrics (1 minute expiration)"""
        key = "system:metrics"
        metrics["cached_at"] = datetime.now().isoformat()
        await self.set(key, metrics, expire)
    
    async def get_system_metrics(self) -> Optional[Dict]:
        """Get cached system metrics"""
        key = "system:metrics"
        return await self.get(key)
    
    # Trading state management
    async def set_trading_state(self, state: str):
        """Set current trading state"""
        key = "trading:state"
        state_data = {
            "state": state,
            "timestamp": datetime.now().isoformat()
        }
        await self.set(key, state_data)
    
    async def get_trading_state(self) -> Optional[Dict]:
        """Get current trading state"""
        key = "trading:state"
        return await self.get(key)
    
    # Rate limiting
    async def check_rate_limit(self, key: str, limit: int, window: int) -> bool:
        """Check if rate limit is exceeded"""
        if not self.redis:
            return True  # Allow if Redis is unavailable
        
        try:
            current = await self.redis.get(key)
            if current is None:
                await self.redis.setex(key, window, 1)
                return True
            
            count = int(current)
            if count >= limit:
                return False
            
            await self.redis.incr(key)
            return True
            
        except Exception as e:
            logger.error(f"Rate limit check failed for {key}: {e}")
            return True  # Allow on error
    
    # Cache invalidation
    async def invalidate_pattern(self, pattern: str):
        """Invalidate all keys matching pattern"""
        if not self.redis:
            return
        
        try:
            keys = await self.redis.keys(pattern)
            if keys:
                await self.redis.delete(*keys)
                logger.info(f"Invalidated {len(keys)} keys matching pattern: {pattern}")
        except Exception as e:
            logger.error(f"Failed to invalidate pattern {pattern}: {e}")
    
    # Health check
    async def health_check(self) -> Dict:
        """Get Redis health information"""
        if not self.redis:
            return {"status": "disconnected", "error": "No Redis connection"}
        
        try:
            start_time = asyncio.get_event_loop().time()
            await self.redis.ping()
            ping_time = (asyncio.get_event_loop().time() - start_time) * 1000
            
            info = await self.redis.info()
            
            return {
                "status": "connected",
                "ping_ms": round(ping_time, 2),
                "version": info.get("redis_version"),
                "used_memory_human": info.get("used_memory_human"),
                "connected_clients": info.get("connected_clients"),
                "total_commands_processed": info.get("total_commands_processed"),
                "keyspace_hits": info.get("keyspace_hits", 0),
                "keyspace_misses": info.get("keyspace_misses", 0),
                "hit_rate": round(
                    info.get("keyspace_hits", 0) / 
                    max(info.get("keyspace_hits", 0) + info.get("keyspace_misses", 0), 1) * 100, 
                    2
                )
            }
            
        except Exception as e:
            return {"status": "error", "error": str(e)}

# Global Redis service instance
redis_service = RedisService()

async def get_redis() -> RedisService:
    """Dependency injection for Redis service"""
    return redis_service

async def init_redis(redis_url: str = "redis://localhost:6379"):
    """Initialize Redis connection"""
    await redis_service.connect(redis_url)

async def close_redis():
    """Close Redis connection"""
    await redis_service.disconnect()