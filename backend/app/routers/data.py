"""
Data collection management router
"""

from fastapi import APIRouter, Depends, HTTPException
from typing import Dict, List, Optional
from datetime import datetime, timedelta
from pydantic import BaseModel
import sqlite3
import json
import yaml
from pathlib import Path
import logging

from app.routers.auth import get_current_user, require_role, User

router = APIRouter()
logger = logging.getLogger(__name__)

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"
DB_PATH = PROJECT_ROOT / "data" / "db" / "crypto_data.db"
KAFKA_CONFIG_PATH = PROJECT_ROOT / "config" / "kafka_config.yaml"

# Pydantic models
class CollectorStatus(BaseModel):
    name: str
    type: str  # orderbook, trades, news, funding
    status: str  # running, stopped, error
    exchanges: List[str]
    symbols: List[str]
    messages_per_second: float
    last_update: Optional[datetime]
    error_count: int
    last_error: Optional[str]

class CollectorConfig(BaseModel):
    exchanges: Optional[List[str]] = None
    symbols: Optional[List[str]] = None
    interval_ms: Optional[int] = None
    order_book_depth: Optional[int] = None
    enable_trades: Optional[bool] = None
    enable_order_books: Optional[bool] = None

class DataStatistics(BaseModel):
    total_records: int
    unique_symbols: int
    unique_exchanges: int
    oldest_record: Optional[datetime]
    newest_record: Optional[datetime]
    data_gaps: List[Dict]
    quality_score: float

class DataQuality(BaseModel):
    symbol: str
    exchange: str
    missing_data_percent: float
    outliers_percent: float
    last_update: datetime
    update_frequency: float  # updates per minute
    quality_score: float

# Helper functions
def load_config() -> Dict:
    """Load configuration from YAML"""
    try:
        with open(CONFIG_PATH, 'r') as f:
            return yaml.safe_load(f)
    except Exception as e:
        logger.error(f"Failed to load config: {e}")
        return {}

def save_config(config: Dict):
    """Save configuration to YAML"""
    try:
        with open(CONFIG_PATH, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
    except Exception as e:
        logger.error(f"Failed to save config: {e}")
        raise HTTPException(status_code=500, detail="Failed to save configuration")

def get_db_connection():
    """Get database connection"""
    return sqlite3.connect(DB_PATH, check_same_thread=False)

def get_kafka_metrics() -> Dict:
    """Get Kafka metrics for data collection"""
    # In production, this would query Kafka metrics
    # For now, return mock data
    return {
        "orderbooks": {
            "messages_per_second": 150.5,
            "lag": 0,
            "errors": 0
        },
        "trades": {
            "messages_per_second": 89.3,
            "lag": 0,
            "errors": 0
        },
        "news": {
            "messages_per_second": 0.1,
            "lag": 0,
            "errors": 0
        }
    }

# API Endpoints
@router.get("/collectors")
async def get_collectors(current_user: User = Depends(get_current_user)) -> Dict[str, CollectorStatus]:
    """List all data collectors and their status"""
    config = load_config()
    kafka_metrics = get_kafka_metrics()
    
    collectors = {}
    
    # Order book collector
    collectors["orderbook"] = CollectorStatus(
        name="Order Book Collector",
        type="orderbook",
        status="running",  # Would check actual process status
        exchanges=["binance", "coinbase", "kraken", "kucoin"],
        symbols=config.get("data", {}).get("symbols", []),
        messages_per_second=kafka_metrics["orderbooks"]["messages_per_second"],
        last_update=datetime.now() - timedelta(seconds=5),
        error_count=0,
        last_error=None
    )
    
    # Trade stream collector
    collectors["trades"] = CollectorStatus(
        name="Trade Stream",
        type="trades",
        status="running",
        exchanges=["binance", "coinbase", "kraken", "kucoin"],
        symbols=config.get("data", {}).get("symbols", []),
        messages_per_second=kafka_metrics["trades"]["messages_per_second"],
        last_update=datetime.now() - timedelta(seconds=3),
        error_count=0,
        last_error=None
    )
    
    # News collector
    collectors["news"] = CollectorStatus(
        name="News Harvester",
        type="news",
        status="running",
        exchanges=[],  # Not exchange-specific
        symbols=config.get("data", {}).get("symbols", []),
        messages_per_second=kafka_metrics["news"]["messages_per_second"],
        last_update=datetime.now() - timedelta(minutes=5),
        error_count=0,
        last_error=None
    )
    
    # Funding rate monitor
    collectors["funding"] = CollectorStatus(
        name="Funding Monitor",
        type="funding",
        status="stopped",
        exchanges=["binance", "bybit"],
        symbols=["BTCUSDT", "ETHUSDT"],
        messages_per_second=0,
        last_update=None,
        error_count=0,
        last_error=None
    )
    
    return collectors

@router.post("/collectors/{collector_type}/configure")
async def configure_collector(
    collector_type: str,
    config: CollectorConfig,
    current_user: User = Depends(require_role("trader"))
) -> Dict[str, str]:
    """Update collector configuration"""
    valid_types = ["orderbook", "trades", "news", "funding"]
    if collector_type not in valid_types:
        raise HTTPException(status_code=404, detail=f"Collector type {collector_type} not found")
    
    # Load current config
    current_config = load_config()
    
    # Update relevant sections
    if config.exchanges is not None:
        current_config.setdefault("exchanges", {})[collector_type] = config.exchanges
    
    if config.symbols is not None:
        current_config["data"]["symbols"] = config.symbols
    
    if config.order_book_depth is not None:
        current_config["data"]["order_book_depth"] = config.order_book_depth
    
    if config.enable_trades is not None:
        current_config["data"]["collect_trades"] = config.enable_trades
    
    if config.enable_order_books is not None:
        current_config["data"]["collect_order_books"] = config.enable_order_books
    
    # Save config
    save_config(current_config)
    
    return {
        "status": "updated",
        "collector": collector_type,
        "message": "Configuration updated. Restart collector to apply changes."
    }

@router.get("/statistics")
async def get_data_statistics(
    hours: int = 24,
    current_user: User = Depends(get_current_user)
) -> DataStatistics:
    """Get data collection statistics"""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    try:
        # Total records
        cursor.execute("SELECT COUNT(*) FROM ohlcv")
        total_records = cursor.fetchone()[0]
        
        # Unique symbols and exchanges
        cursor.execute("SELECT COUNT(DISTINCT symbol) FROM ohlcv")
        unique_symbols = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(DISTINCT exchange) FROM ohlcv")
        unique_exchanges = cursor.fetchone()[0]
        
        # Time range
        cursor.execute("SELECT MIN(timestamp), MAX(timestamp) FROM ohlcv")
        time_range = cursor.fetchone()
        oldest_timestamp, newest_timestamp = time_range
        
        oldest_record = datetime.fromtimestamp(oldest_timestamp / 1000) if oldest_timestamp else None
        newest_record = datetime.fromtimestamp(newest_timestamp / 1000) if newest_timestamp else None
        
        # Check for data gaps (simplified)
        cutoff = datetime.now().timestamp() * 1000 - (hours * 3600 * 1000)
        cursor.execute("""
            SELECT symbol, exchange, MIN(timestamp) as start_gap, MAX(timestamp) as end_gap
            FROM (
                SELECT symbol, exchange, timestamp,
                       LAG(timestamp) OVER (PARTITION BY symbol, exchange ORDER BY timestamp) as prev_timestamp
                FROM ohlcv
                WHERE timestamp > ?
            )
            WHERE timestamp - prev_timestamp > 120000  -- 2 minute gap
            GROUP BY symbol, exchange
            LIMIT 10
        """, (cutoff,))
        
        gaps = []
        for row in cursor.fetchall():
            gaps.append({
                "symbol": row[0],
                "exchange": row[1],
                "start": datetime.fromtimestamp(row[2] / 1000).isoformat(),
                "end": datetime.fromtimestamp(row[3] / 1000).isoformat()
            })
        
        # Quality score (simplified - ratio of expected vs actual records)
        expected_records = hours * 60 * unique_symbols * unique_exchanges  # 1 per minute
        quality_score = min(1.0, total_records / max(expected_records, 1))
        
        conn.close()
        
        return DataStatistics(
            total_records=total_records,
            unique_symbols=unique_symbols,
            unique_exchanges=unique_exchanges,
            oldest_record=oldest_record,
            newest_record=newest_record,
            data_gaps=gaps,
            quality_score=quality_score
        )
        
    except Exception as e:
        conn.close()
        logger.error(f"Failed to get statistics: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve statistics")

@router.get("/quality")
async def get_data_quality(
    current_user: User = Depends(get_current_user)
) -> List[DataQuality]:
    """Get data quality metrics by symbol and exchange"""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    try:
        # Get quality metrics for each symbol/exchange pair
        cursor.execute("""
            SELECT 
                symbol,
                exchange,
                COUNT(*) as record_count,
                MAX(timestamp) as last_update,
                MIN(timestamp) as first_update,
                AVG(CASE WHEN volume = 0 THEN 1 ELSE 0 END) as zero_volume_ratio
            FROM ohlcv
            WHERE timestamp > ?
            GROUP BY symbol, exchange
        """, (datetime.now().timestamp() * 1000 - 86400000,))  # Last 24 hours
        
        quality_metrics = []
        
        for row in cursor.fetchall():
            symbol, exchange, record_count, last_update, first_update, zero_volume_ratio = row
            
            # Calculate update frequency
            time_span_minutes = (last_update - first_update) / 60000
            update_frequency = record_count / max(time_span_minutes, 1)
            
            # Calculate missing data percent (assuming 1 update per minute)
            expected_records = time_span_minutes
            missing_percent = max(0, (expected_records - record_count) / expected_records) * 100
            
            # Quality score (combination of factors)
            quality_score = max(0, 1 - (missing_percent / 100) - zero_volume_ratio)
            
            quality_metrics.append(DataQuality(
                symbol=symbol,
                exchange=exchange,
                missing_data_percent=missing_percent,
                outliers_percent=zero_volume_ratio * 100,
                last_update=datetime.fromtimestamp(last_update / 1000),
                update_frequency=update_frequency,
                quality_score=quality_score
            ))
        
        conn.close()
        return quality_metrics
        
    except Exception as e:
        conn.close()
        logger.error(f"Failed to get quality metrics: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve quality metrics")

@router.get("/streams")
async def get_active_streams(current_user: User = Depends(get_current_user)) -> Dict:
    """Get active data streams information"""
    # Check Kafka topics and consumer groups
    kafka_config = {}
    try:
        with open(KAFKA_CONFIG_PATH, 'r') as f:
            kafka_config = yaml.safe_load(f)
    except Exception:
        pass
    
    topics = kafka_config.get("topics", {})
    
    return {
        "kafka_topics": list(topics.keys()),
        "active_streams": {
            "orderbooks": {
                "topic": "crypto-orderbooks",
                "partitions": topics.get("crypto-orderbooks", {}).get("partitions", 3),
                "status": "active"
            },
            "trades": {
                "topic": "crypto-trades",
                "partitions": topics.get("crypto-trades", {}).get("partitions", 3),
                "status": "active"
            },
            "features": {
                "topic": "crypto-features",
                "partitions": topics.get("crypto-features", {}).get("partitions", 2),
                "status": "active"
            },
            "signals": {
                "topic": "trading-signals",
                "partitions": topics.get("trading-signals", {}).get("partitions", 1),
                "status": "active"
            }
        }
    }

@router.post("/collectors/{collector_type}/reset")
async def reset_collector_errors(
    collector_type: str,
    current_user: User = Depends(require_role("trader"))
) -> Dict[str, str]:
    """Reset error counters for a collector"""
    valid_types = ["orderbook", "trades", "news", "funding"]
    if collector_type not in valid_types:
        raise HTTPException(status_code=404, detail=f"Collector type {collector_type} not found")
    
    # In production, this would reset actual error counters
    return {
        "status": "reset",
        "collector": collector_type,
        "message": "Error counters reset successfully"
    }

@router.get("/exchanges")
async def get_supported_exchanges(current_user: User = Depends(get_current_user)) -> List[Dict]:
    """Get list of supported exchanges"""
    exchanges = [
        {
            "id": "binance",
            "name": "Binance.US",
            "supported_features": ["spot", "orderbook", "trades"],
            "status": "active"
        },
        {
            "id": "coinbase",
            "name": "Coinbase",
            "supported_features": ["spot", "orderbook", "trades"],
            "status": "active"
        },
        {
            "id": "kraken",
            "name": "Kraken",
            "supported_features": ["spot", "orderbook", "trades"],
            "status": "active"
        },
        {
            "id": "kucoin",
            "name": "KuCoin",
            "supported_features": ["spot", "orderbook", "trades"],
            "status": "active"
        }
    ]
    return exchanges