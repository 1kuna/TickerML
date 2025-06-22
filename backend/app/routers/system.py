"""
System control router for managing TickerML services
"""

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from typing import Dict, List, Optional
import subprocess
import psutil
import logging
import os
import signal
import json
from pathlib import Path
from pydantic import BaseModel
from datetime import datetime, timedelta
import asyncio
from collections import deque
import threading
import time

from app.routers.auth import get_current_user, require_role, User
from app.services.redis_service import redis_service

router = APIRouter()
logger = logging.getLogger(__name__)

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
RASPBERRY_PI_DIR = PROJECT_ROOT / "raspberry_pi"
PC_DIR = PROJECT_ROOT / "pc"
SCRIPTS_DIR = PROJECT_ROOT / "scripts"

# Service definitions
SERVICES = {
    "data_collector": {
        "name": "Order Book Collector",
        "command": f"python {RASPBERRY_PI_DIR}/orderbook_collector.py",
        "description": "Collects real-time order book data from exchanges",
        "type": "data"
    },
    "trade_stream": {
        "name": "Trade Stream",
        "command": f"python {RASPBERRY_PI_DIR}/trade_stream.py",
        "description": "Streams real-time trade data",
        "type": "data"
    },
    "paper_trader": {
        "name": "Paper Trader",
        "command": f"python {RASPBERRY_PI_DIR}/paper_trader.py",
        "description": "Paper trading engine with risk management",
        "type": "trading"
    },
    "kafka_orderbook": {
        "name": "Kafka Order Book Producer",
        "command": f"python {RASPBERRY_PI_DIR}/kafka_producers/orderbook_producer.py",
        "description": "Streams order book data to Kafka",
        "type": "kafka"
    },
    "kafka_trades": {
        "name": "Kafka Trade Producer",
        "command": f"python {RASPBERRY_PI_DIR}/kafka_producers/trade_producer.py",
        "description": "Streams trade data to Kafka",
        "type": "kafka"
    },
    "feature_consumer": {
        "name": "Feature Consumer",
        "command": f"python {RASPBERRY_PI_DIR}/kafka_consumers/feature_consumer.py",
        "description": "Consumes and processes features from Kafka",
        "type": "kafka"
    },
    "news_harvester": {
        "name": "News Harvester",
        "command": f"python {RASPBERRY_PI_DIR}/news_harvest.py",
        "description": "Collects and analyzes crypto news",
        "type": "data"
    },
    "arbitrage_monitor": {
        "name": "Arbitrage Monitor",
        "command": f"python {RASPBERRY_PI_DIR}/arbitrage_monitor.py",
        "description": "Monitors cross-exchange arbitrage opportunities",
        "type": "trading"
    }
}

# Pydantic models
class ServiceStatus(BaseModel):
    name: str
    status: str  # running, stopped, error
    pid: Optional[int] = None
    cpu_percent: Optional[float] = None
    memory_mb: Optional[float] = None
    uptime_seconds: Optional[int] = None
    last_error: Optional[str] = None

class SystemMetrics(BaseModel):
    cpu_percent: float
    memory_percent: float
    memory_available_gb: float
    memory_total_gb: float
    disk_percent: float
    disk_free_gb: float
    gpu_usage: Optional[float] = None
    gpu_memory_mb: Optional[float] = None
    network_bytes_sent: int
    network_bytes_recv: int
    timestamp: datetime

class ServiceCommand(BaseModel):
    service_name: str
    action: str  # start, stop, restart

# Service management functions
def get_service_pid(service_key: str) -> Optional[int]:
    """Get PID of a running service"""
    try:
        # Look for process by command pattern
        service_info = SERVICES.get(service_key)
        if not service_info:
            return None
            
        for proc in psutil.process_iter(['pid', 'cmdline']):
            try:
                cmdline = proc.info['cmdline']
                if cmdline and service_info['command'] in ' '.join(cmdline):
                    return proc.info['pid']
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
    except Exception as e:
        logger.error(f"Error getting PID for {service_key}: {e}")
    return None

def get_process_info(pid: int) -> Dict:
    """Get process information"""
    try:
        proc = psutil.Process(pid)
        return {
            "cpu_percent": proc.cpu_percent(interval=0.1),
            "memory_mb": proc.memory_info().rss / 1024 / 1024,
            "create_time": proc.create_time()
        }
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return {}

async def start_service(service_key: str) -> Dict[str, str]:
    """Start a service"""
    service_info = SERVICES.get(service_key)
    if not service_info:
        raise HTTPException(status_code=404, detail=f"Service {service_key} not found")
    
    # Check if already running
    pid = get_service_pid(service_key)
    if pid:
        return {"status": "already_running", "pid": pid}
    
    try:
        # Start the service
        process = subprocess.Popen(
            service_info['command'].split(),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=True
        )
        
        # Wait a moment to check if it started successfully
        await asyncio.sleep(2)
        
        if process.poll() is None:
            return {"status": "started", "pid": process.pid}
        else:
            stderr = process.stderr.read().decode() if process.stderr else "Unknown error"
            return {"status": "failed", "error": stderr}
            
    except Exception as e:
        logger.error(f"Failed to start {service_key}: {e}")
        return {"status": "error", "error": str(e)}

async def stop_service(service_key: str) -> Dict[str, str]:
    """Stop a service"""
    pid = get_service_pid(service_key)
    if not pid:
        return {"status": "not_running"}
    
    try:
        os.kill(pid, signal.SIGTERM)
        
        # Wait for graceful shutdown
        for _ in range(10):
            if not psutil.pid_exists(pid):
                return {"status": "stopped"}
            await asyncio.sleep(0.5)
        
        # Force kill if still running
        os.kill(pid, signal.SIGKILL)
        return {"status": "force_stopped"}
        
    except Exception as e:
        logger.error(f"Failed to stop {service_key}: {e}")
        return {"status": "error", "error": str(e)}

# Historical metrics storage
metrics_history = deque(maxlen=60)  # Keep last 60 minutes of data
metrics_lock = threading.Lock()

def get_gpu_usage() -> Optional[Dict[str, float]]:
    """Get GPU usage if available"""
    try:
        import GPUtil
        gpus = GPUtil.getGPUs()
        if gpus:
            gpu = gpus[0]
            return {
                "usage": gpu.load * 100,
                "memory_used_mb": gpu.memoryUsed,
                "memory_total_mb": gpu.memoryTotal,
                "memory_percent": (gpu.memoryUsed / gpu.memoryTotal) * 100 if gpu.memoryTotal > 0 else 0,
                "temperature": gpu.temperature,
                "name": gpu.name,
                "uuid": gpu.uuid
            }
    except ImportError:
        logger.warning("GPUtil not available - GPU monitoring disabled")
    except Exception as e:
        logger.error(f"Error getting GPU usage: {e}")
    return None

def get_network_usage() -> Dict[str, int]:
    """Get network usage statistics"""
    try:
        net_io = psutil.net_io_counters()
        return {
            "bytes_sent": net_io.bytes_sent,
            "bytes_recv": net_io.bytes_recv,
            "packets_sent": net_io.packets_sent,
            "packets_recv": net_io.packets_recv,
            "errin": net_io.errin,
            "errout": net_io.errout,
            "dropin": net_io.dropin,
            "dropout": net_io.dropout
        }
    except Exception as e:
        logger.error(f"Error getting network usage: {e}")
        return {}

def collect_metrics():
    """Collect system metrics and store in history"""
    try:
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        network = get_network_usage()
        gpu_info = get_gpu_usage()
        
        # Get CPU per-core usage
        cpu_per_core = psutil.cpu_percent(percpu=True)
        
        # Get process count
        process_count = len(psutil.pids())
        
        # Get load average (Linux/Mac)
        try:
            load_avg = os.getloadavg()
        except AttributeError:
            load_avg = [0, 0, 0]  # Windows doesn't have load average
        
        metrics = {
            "timestamp": datetime.now().isoformat(),
            "cpu": {
                "total_percent": cpu_percent,
                "per_core": cpu_per_core,
                "load_avg_1m": load_avg[0],
                "load_avg_5m": load_avg[1],
                "load_avg_15m": load_avg[2],
                "logical_cores": psutil.cpu_count(logical=True),
                "physical_cores": psutil.cpu_count(logical=False)
            },
            "memory": {
                "percent": memory.percent,
                "available_gb": memory.available / 1024**3,
                "total_gb": memory.total / 1024**3,
                "used_gb": memory.used / 1024**3,
                "free_gb": memory.free / 1024**3,
                "buffers_gb": getattr(memory, 'buffers', 0) / 1024**3,
                "cached_gb": getattr(memory, 'cached', 0) / 1024**3
            },
            "disk": {
                "percent": disk.percent,
                "free_gb": disk.free / 1024**3,
                "used_gb": disk.used / 1024**3,
                "total_gb": disk.total / 1024**3
            },
            "network": network,
            "gpu": gpu_info,
            "system": {
                "process_count": process_count,
                "boot_time": psutil.boot_time()
            }
        }
        
        with metrics_lock:
            metrics_history.append(metrics)
            
        # Store in Redis for real-time access
        if redis_service and redis_service.client:
            try:
                redis_service.client.setex(
                    "system:metrics:latest",
                    300,  # 5 minute expiry
                    json.dumps(metrics, default=str)
                )
            except Exception as e:
                logger.error(f"Failed to store metrics in Redis: {e}")
                
    except Exception as e:
        logger.error(f"Error collecting metrics: {e}")

# Background task to collect metrics
def metrics_collector_task():
    """Background task that collects metrics every minute"""
    while True:
        collect_metrics()
        time.sleep(60)  # Collect every minute

# Start metrics collection thread
metrics_thread = threading.Thread(target=metrics_collector_task, daemon=True)
metrics_thread.start()

# API Endpoints
@router.get("/status")
async def get_system_status(current_user: User = Depends(get_current_user)) -> Dict:
    """Get status of all system components"""
    # System metrics
    cpu_percent = psutil.cpu_percent(interval=1)
    memory = psutil.virtual_memory()
    disk = psutil.disk_usage('/')
    network = psutil.net_io_counters()
    
    gpu_info = get_gpu_usage()
    
    system_metrics = SystemMetrics(
        cpu_percent=cpu_percent,
        memory_percent=memory.percent,
        memory_available_gb=memory.available / 1024**3,
        memory_total_gb=memory.total / 1024**3,
        disk_percent=disk.percent,
        disk_free_gb=disk.free / 1024**3,
        gpu_usage=gpu_info['usage'] if gpu_info else None,
        gpu_memory_mb=gpu_info['memory_mb'] if gpu_info else None,
        network_bytes_sent=network.bytes_sent,
        network_bytes_recv=network.bytes_recv,
        timestamp=datetime.now()
    )
    
    # Service statuses
    service_statuses = {}
    for service_key, service_info in SERVICES.items():
        pid = get_service_pid(service_key)
        
        if pid:
            proc_info = get_process_info(pid)
            status = ServiceStatus(
                name=service_info['name'],
                status="running",
                pid=pid,
                cpu_percent=proc_info.get('cpu_percent'),
                memory_mb=proc_info.get('memory_mb'),
                uptime_seconds=int(datetime.now().timestamp() - proc_info.get('create_time', 0))
            )
        else:
            status = ServiceStatus(
                name=service_info['name'],
                status="stopped"
            )
        
        service_statuses[service_key] = status.dict()
    
    # Check Kafka status
    kafka_running = check_kafka_status()
    
    return {
        "system_metrics": system_metrics.dict(),
        "services": service_statuses,
        "kafka": {
            "running": kafka_running,
            "status": "running" if kafka_running else "stopped"
        }
    }

@router.post("/services/{service_name}/start")
async def start_service_endpoint(
    service_name: str,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(require_role("trader"))
) -> Dict[str, str]:
    """Start a system service"""
    if service_name not in SERVICES:
        raise HTTPException(status_code=404, detail=f"Service {service_name} not found")
    
    result = await start_service(service_name)
    return {
        "service": service_name,
        "action": "start",
        **result
    }

@router.post("/services/{service_name}/stop")
async def stop_service_endpoint(
    service_name: str,
    current_user: User = Depends(require_role("trader"))
) -> Dict[str, str]:
    """Stop a system service"""
    if service_name not in SERVICES:
        raise HTTPException(status_code=404, detail=f"Service {service_name} not found")
    
    result = await stop_service(service_name)
    return {
        "service": service_name,
        "action": "stop",
        **result
    }

@router.post("/services/{service_name}/restart")
async def restart_service_endpoint(
    service_name: str,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(require_role("trader"))
) -> Dict[str, str]:
    """Restart a system service"""
    if service_name not in SERVICES:
        raise HTTPException(status_code=404, detail=f"Service {service_name} not found")
    
    # Stop first
    await stop_service(service_name)
    await asyncio.sleep(2)  # Wait for clean shutdown
    
    # Then start
    result = await start_service(service_name)
    return {
        "service": service_name,
        "action": "restart",
        **result
    }

@router.post("/services/all/start")
async def start_all_services(
    background_tasks: BackgroundTasks,
    current_user: User = Depends(require_role("admin"))
) -> Dict[str, Dict]:
    """Start all services"""
    results = {}
    
    # Start Kafka first if needed
    if not check_kafka_status():
        kafka_result = start_kafka()
        results["kafka"] = kafka_result
        await asyncio.sleep(5)  # Wait for Kafka to be ready
    
    # Start services in order
    service_order = [
        "data_collector", "trade_stream", "news_harvester",
        "kafka_orderbook", "kafka_trades",
        "feature_consumer", "paper_trader", "arbitrage_monitor"
    ]
    
    for service_key in service_order:
        if service_key in SERVICES:
            results[service_key] = await start_service(service_key)
            await asyncio.sleep(1)  # Stagger starts
    
    return results

@router.post("/services/all/stop")
async def stop_all_services(
    current_user: User = Depends(require_role("admin"))
) -> Dict[str, Dict]:
    """Stop all services"""
    results = {}
    
    # Stop services in reverse order
    service_order = [
        "arbitrage_monitor", "paper_trader", "feature_consumer",
        "kafka_trades", "kafka_orderbook",
        "news_harvester", "trade_stream", "data_collector"
    ]
    
    for service_key in service_order:
        if service_key in SERVICES:
            results[service_key] = await stop_service(service_key)
    
    return results

@router.get("/services")
async def list_services(current_user: User = Depends(get_current_user)) -> List[Dict]:
    """List all available services"""
    services = []
    for key, info in SERVICES.items():
        pid = get_service_pid(key)
        services.append({
            "key": key,
            "name": info["name"],
            "description": info["description"],
            "type": info["type"],
            "running": pid is not None,
            "pid": pid
        })
    return services

# Kafka management
def check_kafka_status() -> bool:
    """Check if Kafka is running"""
    try:
        for proc in psutil.process_iter(['pid', 'name']):
            if 'kafka' in proc.info['name'].lower():
                return True
    except Exception:
        pass
    return False

def start_kafka() -> Dict[str, str]:
    """Start Kafka services"""
    try:
        script_path = SCRIPTS_DIR / "start_kafka.sh"
        if script_path.exists():
            subprocess.Popen([str(script_path)], start_new_session=True)
            return {"status": "started"}
        else:
            return {"status": "error", "error": "Kafka start script not found"}
    except Exception as e:
        return {"status": "error", "error": str(e)}

@router.post("/kafka/start")
async def start_kafka_endpoint(
    current_user: User = Depends(require_role("admin"))
) -> Dict[str, str]:
    """Start Kafka services"""
    return start_kafka()

@router.post("/kafka/stop")
async def stop_kafka_endpoint(
    current_user: User = Depends(require_role("admin"))
) -> Dict[str, str]:
    """Stop Kafka services"""
    try:
        # Find and stop Kafka processes
        for proc in psutil.process_iter(['pid', 'name']):
            if 'kafka' in proc.info['name'].lower():
                os.kill(proc.info['pid'], signal.SIGTERM)
        return {"status": "stopped"}
    except Exception as e:
        return {"status": "error", "error": str(e)}

@router.get("/metrics/history")
async def get_metrics_history(
    hours: int = 1,
    current_user: User = Depends(get_current_user)
) -> List[Dict]:
    """Get historical system metrics"""
    with metrics_lock:
        # Filter metrics based on time range
        now = datetime.now()
        cutoff = now - timedelta(hours=hours)
        
        filtered_metrics = []
        for metric in metrics_history:
            try:
                metric_time = datetime.fromisoformat(metric["timestamp"])
                if metric_time >= cutoff:
                    filtered_metrics.append(metric)
            except (ValueError, KeyError):
                continue
                
        return filtered_metrics

@router.get("/metrics/current")
async def get_current_metrics(
    current_user: User = Depends(get_current_user)
) -> Dict:
    """Get current system metrics"""
    try:
        # Try to get from Redis first
        if redis_service and redis_service.client:
            cached_metrics = redis_service.client.get("system:metrics:latest")
            if cached_metrics:
                return json.loads(cached_metrics)
        
        # Fall back to collecting fresh metrics
        with metrics_lock:
            if metrics_history:
                return metrics_history[-1]
        
        # If no history, collect now
        collect_metrics()
        with metrics_lock:
            if metrics_history:
                return metrics_history[-1]
                
        return {"error": "No metrics available"}
        
    except Exception as e:
        logger.error(f"Error getting current metrics: {e}")
        return {"error": str(e)}

@router.get("/metrics/summary")
async def get_metrics_summary(
    current_user: User = Depends(get_current_user)
) -> Dict:
    """Get summarized metrics for dashboard"""
    try:
        current = await get_current_metrics(current_user)
        
        if "error" in current:
            return current
            
        # Calculate some derived metrics
        summary = {
            "cpu_usage": current["cpu"]["total_percent"],
            "memory_usage": current["memory"]["percent"],
            "disk_usage": current["disk"]["percent"],
            "gpu_usage": current["gpu"]["usage"] if current.get("gpu") else None,
            "gpu_memory": current["gpu"]["memory_percent"] if current.get("gpu") else None,
            "load_average": current["cpu"]["load_avg_1m"],
            "process_count": current["system"]["process_count"],
            "memory_available_gb": current["memory"]["available_gb"],
            "disk_free_gb": current["disk"]["free_gb"],
            "uptime_hours": (time.time() - current["system"]["boot_time"]) / 3600,
            "network_throughput": {
                "bytes_sent": current["network"]["bytes_sent"],
                "bytes_recv": current["network"]["bytes_recv"]
            }
        }
        
        # Add health status based on thresholds
        summary["health_status"] = "healthy"
        if summary["cpu_usage"] > 90 or summary["memory_usage"] > 90:
            summary["health_status"] = "critical"
        elif summary["cpu_usage"] > 75 or summary["memory_usage"] > 75:
            summary["health_status"] = "warning"
            
        return summary
        
    except Exception as e:
        logger.error(f"Error getting metrics summary: {e}")
        return {"error": str(e)}

@router.get("/health")
async def health_check() -> Dict:
    """Comprehensive health check endpoint"""
    try:
        health = {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "checks": {}
        }
        
        # System resources check
        metrics = await get_metrics_summary(None)  # Skip auth for health check
        if "error" not in metrics:
            cpu_healthy = metrics["cpu_usage"] < 90
            memory_healthy = metrics["memory_usage"] < 90
            disk_healthy = metrics["disk_usage"] < 95
            
            health["checks"]["system_resources"] = {
                "status": "healthy" if all([cpu_healthy, memory_healthy, disk_healthy]) else "unhealthy",
                "cpu_usage": metrics["cpu_usage"],
                "memory_usage": metrics["memory_usage"],
                "disk_usage": metrics["disk_usage"]
            }
        
        # Services check
        service_statuses = {}
        for service_key in SERVICES:
            pid = get_service_pid(service_key)
            service_statuses[service_key] = pid is not None
            
        critical_services = ["data_collector", "paper_trader"]
        critical_running = all(service_statuses.get(service, False) for service in critical_services)
        
        health["checks"]["critical_services"] = {
            "status": "healthy" if critical_running else "unhealthy",
            "services": service_statuses
        }
        
        # Kafka check
        kafka_running = check_kafka_status()
        health["checks"]["kafka"] = {
            "status": "healthy" if kafka_running else "unhealthy",
            "running": kafka_running
        }
        
        # Overall status
        all_checks_healthy = all(
            check["status"] == "healthy" 
            for check in health["checks"].values()
        )
        health["status"] = "healthy" if all_checks_healthy else "unhealthy"
        
        return health
        
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }