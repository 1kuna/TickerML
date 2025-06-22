"""
Configuration management router for runtime system configuration
"""

from fastapi import APIRouter, Depends, HTTPException
from typing import Dict, List, Optional
from pydantic import BaseModel
import yaml
import json
from pathlib import Path
import logging
import os
from datetime import datetime

from app.routers.auth import get_current_user, require_role, User

router = APIRouter()
logger = logging.getLogger(__name__)

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
CONFIG_DIR = PROJECT_ROOT / "config"
CONFIG_FILES = {
    "main": CONFIG_DIR / "config.yaml",
    "risk": CONFIG_DIR / "risk_limits.yaml",
    "model": CONFIG_DIR / "model_config.yaml",
    "kafka": CONFIG_DIR / "kafka_config.yaml",
    "exchanges": CONFIG_DIR / "exchanges_config.yaml",
    "monitoring": CONFIG_DIR / "monitoring_config.yaml",
    "timescale": CONFIG_DIR / "timescale_config.yaml"
}

# Pydantic models
class ConfigFile(BaseModel):
    name: str
    path: str
    last_modified: datetime
    size_bytes: int
    description: str
    editable: bool

class ConfigUpdate(BaseModel):
    file_name: str
    content: Dict
    backup: bool = True

class EnvironmentVariable(BaseModel):
    key: str
    value: str
    masked: bool = False
    description: Optional[str] = None

# Helper functions
def load_config_file(file_path: Path) -> Dict:
    """Load configuration from YAML file"""
    try:
        if not file_path.exists():
            return {}
        
        with open(file_path, 'r') as f:
            return yaml.safe_load(f) or {}
    except Exception as e:
        logger.error(f"Failed to load config {file_path}: {e}")
        return {}

def save_config_file(file_path: Path, config: Dict, backup: bool = True):
    """Save configuration to YAML file with optional backup"""
    try:
        # Create backup if requested
        if backup and file_path.exists():
            backup_path = file_path.with_suffix(f".backup.{int(datetime.now().timestamp())}.yaml")
            file_path.rename(backup_path)
            logger.info(f"Created backup: {backup_path}")
        
        # Save new configuration
        with open(file_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, indent=2)
            
    except Exception as e:
        logger.error(f"Failed to save config {file_path}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to save configuration: {e}")

def validate_config(file_name: str, config: Dict) -> bool:
    """Validate configuration based on file type"""
    if file_name == "risk":
        required_fields = ["max_position_size_percent", "max_drawdown_percent"]
        for field in required_fields:
            if field not in config:
                raise HTTPException(status_code=400, detail=f"Missing required field: {field}")
    
    elif file_name == "model":
        if "decision_transformer" in config:
            dt_config = config["decision_transformer"]
            if "hidden_size" not in dt_config or dt_config["hidden_size"] < 64:
                raise HTTPException(status_code=400, detail="Invalid model hidden_size")
    
    elif file_name == "kafka":
        if "topics" not in config:
            raise HTTPException(status_code=400, detail="Kafka config must include topics")
    
    return True

# API Endpoints
@router.get("/files")
async def list_config_files(
    current_user: User = Depends(get_current_user)
) -> List[ConfigFile]:
    """List all configuration files"""
    files = []
    
    descriptions = {
        "main": "Main system configuration",
        "risk": "Risk management limits and controls",
        "model": "Model architecture and training parameters",
        "kafka": "Event streaming configuration",
        "exchanges": "Exchange API settings and credentials",
        "monitoring": "Monitoring and alerting configuration",
        "timescale": "TimescaleDB connection and settings"
    }
    
    for name, path in CONFIG_FILES.items():
        if path.exists():
            stats = path.stat()
            files.append(ConfigFile(
                name=name,
                path=str(path),
                last_modified=datetime.fromtimestamp(stats.st_mtime),
                size_bytes=stats.st_size,
                description=descriptions.get(name, "Configuration file"),
                editable=name != "exchanges"  # Exchanges config contains secrets
            ))
    
    return files

@router.get("/{file_name}")
async def get_config(
    file_name: str,
    current_user: User = Depends(get_current_user)
) -> Dict:
    """Get specific configuration file"""
    if file_name not in CONFIG_FILES:
        raise HTTPException(status_code=404, detail="Configuration file not found")
    
    # Restrict access to sensitive configs
    if file_name == "exchanges" and current_user.role != "admin":
        raise HTTPException(status_code=403, detail="Insufficient permissions")
    
    config = load_config_file(CONFIG_FILES[file_name])
    
    # Mask sensitive fields
    if file_name == "exchanges":
        for exchange, settings in config.items():
            if isinstance(settings, dict) and "api_secret" in settings:
                settings["api_secret"] = "***MASKED***"
    
    return config

@router.put("/{file_name}")
async def update_config(
    file_name: str,
    update: ConfigUpdate,
    current_user: User = Depends(require_role("admin"))
) -> Dict:
    """Update configuration file"""
    if file_name not in CONFIG_FILES:
        raise HTTPException(status_code=404, detail="Configuration file not found")
    
    # Validate configuration
    validate_config(file_name, update.content)
    
    # Save configuration
    file_path = CONFIG_FILES[file_name]
    save_config_file(file_path, update.content, update.backup)
    
    return {
        "status": "updated",
        "file": file_name,
        "timestamp": datetime.now().isoformat(),
        "backup_created": update.backup
    }

@router.get("/{file_name}/history")
async def get_config_history(
    file_name: str,
    current_user: User = Depends(get_current_user)
) -> List[Dict]:
    """Get configuration change history (backups)"""
    if file_name not in CONFIG_FILES:
        raise HTTPException(status_code=404, detail="Configuration file not found")
    
    config_path = CONFIG_FILES[file_name]
    backup_pattern = f"{config_path.stem}.backup.*.yaml"
    
    backups = []
    for backup_file in CONFIG_DIR.glob(backup_pattern):
        try:
            # Extract timestamp from filename
            timestamp_str = backup_file.stem.split('.')[-1]
            timestamp = datetime.fromtimestamp(int(timestamp_str))
            
            stats = backup_file.stat()
            backups.append({
                "timestamp": timestamp.isoformat(),
                "file_name": backup_file.name,
                "size_bytes": stats.st_size
            })
        except (ValueError, OSError) as e:
            logger.warning(f"Failed to process backup file {backup_file}: {e}")
    
    # Sort by timestamp descending
    backups.sort(key=lambda x: x["timestamp"], reverse=True)
    return backups

@router.post("/{file_name}/restore")
async def restore_config(
    file_name: str,
    backup_timestamp: str,
    current_user: User = Depends(require_role("admin"))
) -> Dict:
    """Restore configuration from backup"""
    if file_name not in CONFIG_FILES:
        raise HTTPException(status_code=404, detail="Configuration file not found")
    
    # Find backup file
    config_path = CONFIG_FILES[file_name]
    backup_file = CONFIG_DIR / f"{config_path.stem}.backup.{backup_timestamp}.yaml"
    
    if not backup_file.exists():
        raise HTTPException(status_code=404, detail="Backup file not found")
    
    try:
        # Load backup content
        backup_content = load_config_file(backup_file)
        
        # Save current as backup before restoring
        save_config_file(config_path, backup_content, backup=True)
        
        return {
            "status": "restored",
            "file": file_name,
            "restored_from": backup_timestamp,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to restore config: {e}")
        raise HTTPException(status_code=500, detail="Failed to restore configuration")

@router.get("/environment/variables")
async def get_environment_variables(
    current_user: User = Depends(require_role("admin"))
) -> List[EnvironmentVariable]:
    """Get environment variables (masked)"""
    sensitive_keys = {
        "API_KEY", "SECRET", "PASSWORD", "TOKEN", "PRIVATE_KEY", 
        "DATABASE_URL", "REDIS_URL"
    }
    
    variables = []
    for key, value in os.environ.items():
        if key.startswith(("TICKERML_", "DASHBOARD_", "BINANCE_", "COINBASE_", "KRAKEN_", "KUCOIN_")):
            is_sensitive = any(sensitive in key.upper() for sensitive in sensitive_keys)
            
            variables.append(EnvironmentVariable(
                key=key,
                value="***MASKED***" if is_sensitive else value,
                masked=is_sensitive,
                description=get_env_description(key)
            ))
    
    return sorted(variables, key=lambda x: x.key)

@router.put("/environment/{key}")
async def update_environment_variable(
    key: str,
    value: str,
    current_user: User = Depends(require_role("admin"))
) -> Dict:
    """Update environment variable (runtime only)"""
    if not key.startswith(("TICKERML_", "DASHBOARD_")):
        raise HTTPException(status_code=400, detail="Can only update TICKERML_ or DASHBOARD_ variables")
    
    # Update environment variable
    os.environ[key] = value
    
    return {
        "status": "updated",
        "key": key,
        "timestamp": datetime.now().isoformat(),
        "note": "Runtime update only. Restart required for persistence."
    }

@router.post("/validate/{file_name}")
async def validate_config_file(
    file_name: str,
    config: Dict,
    current_user: User = Depends(require_role("admin"))
) -> Dict:
    """Validate configuration without saving"""
    if file_name not in CONFIG_FILES:
        raise HTTPException(status_code=404, detail="Configuration file not found")
    
    try:
        validate_config(file_name, config)
        return {
            "valid": True,
            "message": "Configuration is valid"
        }
    except HTTPException as e:
        return {
            "valid": False,
            "message": e.detail
        }

@router.get("/schema/{file_name}")
async def get_config_schema(
    file_name: str,
    current_user: User = Depends(get_current_user)
) -> Dict:
    """Get configuration schema/template"""
    schemas = {
        "risk": {
            "max_position_size_percent": 25,
            "max_drawdown_percent": 25,
            "daily_loss_limit_percent": 5,
            "max_positions": 5,
            "position_sizing": {
                "kelly_fraction": 0.25,
                "min_position_percent": 1,
                "max_position_percent": 25
            },
            "circuit_breaker": {
                "max_loss_per_minute_percent": 2,
                "volatility_halt_threshold": 0.1
            }
        },
        "model": {
            "decision_transformer": {
                "hidden_size": 256,
                "num_layers": 6,
                "num_heads": 8,
                "context_length": 100,
                "dropout": 0.1,
                "use_mixed_precision": True
            },
            "training": {
                "batch_size": 32,
                "learning_rate": 0.001,
                "epochs": 100,
                "early_stopping_patience": 10
            }
        },
        "kafka": {
            "bootstrap_servers": ["localhost:9092"],
            "topics": {
                "crypto-orderbooks": {"partitions": 3, "replication_factor": 1},
                "crypto-trades": {"partitions": 3, "replication_factor": 1},
                "crypto-features": {"partitions": 2, "replication_factor": 1}
            }
        }
    }
    
    if file_name not in schemas:
        raise HTTPException(status_code=404, detail="Schema not available")
    
    return schemas[file_name]

def get_env_description(key: str) -> Optional[str]:
    """Get description for environment variable"""
    descriptions = {
        "DASHBOARD_SECRET_KEY": "JWT secret key for authentication",
        "DASHBOARD_HOST": "Dashboard server host",
        "DASHBOARD_PORT": "Dashboard server port",
        "BINANCE_API_KEY": "Binance API key",
        "COINBASE_API_KEY": "Coinbase API key",
        "KRAKEN_API_KEY": "Kraken API key",
        "KUCOIN_API_KEY": "KuCoin API key",
        "TICKERML_LOG_LEVEL": "Application log level"
    }
    return descriptions.get(key)