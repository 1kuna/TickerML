"""
Logs and monitoring router for system log management and analysis
"""

from fastapi import APIRouter, Depends, Query, HTTPException
from typing import Dict, List, Optional
from pydantic import BaseModel
from pathlib import Path
from datetime import datetime, timedelta
import logging
import re
import json
from enum import Enum

from app.routers.auth import get_current_user, User

router = APIRouter()
logger = logging.getLogger(__name__)

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
LOGS_DIR = PROJECT_ROOT / "logs"

# Enums
class LogLevel(str, Enum):
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"

# Pydantic models
class LogEntry(BaseModel):
    timestamp: datetime
    level: LogLevel
    service: str
    message: str
    file: Optional[str] = None
    line_number: Optional[int] = None
    exception: Optional[str] = None

class LogFile(BaseModel):
    name: str
    path: str
    size_bytes: int
    last_modified: datetime
    service: str
    entries_count: Optional[int] = None

class LogStatistics(BaseModel):
    total_entries: int
    entries_by_level: Dict[str, int]
    entries_by_service: Dict[str, int]
    error_rate: float
    warning_rate: float
    most_common_errors: List[Dict[str, any]]
    time_range: Dict[str, datetime]

class AlertRule(BaseModel):
    id: str
    name: str
    pattern: str
    level: LogLevel
    threshold: int
    time_window_minutes: int
    enabled: bool
    action: str  # email, webhook, etc.

# Helper functions
def get_log_files() -> List[LogFile]:
    """Get list of available log files"""
    if not LOGS_DIR.exists():
        return []
    
    log_files = []
    for log_file in LOGS_DIR.glob("*.log"):
        try:
            stats = log_file.stat()
            
            # Determine service name from filename
            service_name = log_file.stem.replace("_", " ").title()
            
            log_files.append(LogFile(
                name=log_file.name,
                path=str(log_file),
                size_bytes=stats.st_size,
                last_modified=datetime.fromtimestamp(stats.st_mtime),
                service=service_name
            ))
        except OSError as e:
            logger.warning(f"Failed to process log file {log_file}: {e}")
    
    return sorted(log_files, key=lambda x: x.last_modified, reverse=True)

def parse_log_line(line: str, service: str) -> Optional[LogEntry]:
    """Parse a single log line into LogEntry"""
    # Common log format: YYYY-MM-DD HH:MM:SS - LEVEL - MESSAGE
    log_pattern = r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}(?:\.\d{3})?)\s*-\s*(\w+)\s*-\s*(.*)"
    
    match = re.match(log_pattern, line.strip())
    if not match:
        return None
    
    timestamp_str, level_str, message = match.groups()
    
    try:
        # Parse timestamp
        try:
            timestamp = datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S.%f")
        except ValueError:
            timestamp = datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S")
        
        # Validate log level
        try:
            level = LogLevel(level_str.upper())
        except ValueError:
            level = LogLevel.INFO
        
        # Extract file and line info if present
        file_match = re.search(r"(\w+\.py):(\d+)", message)
        file_name = file_match.group(1) if file_match else None
        line_number = int(file_match.group(2)) if file_match else None
        
        # Extract exception info if present
        exception = None
        if "Traceback" in message or "Exception" in message:
            exception = message
        
        return LogEntry(
            timestamp=timestamp,
            level=level,
            service=service,
            message=message,
            file=file_name,
            line_number=line_number,
            exception=exception
        )
        
    except Exception as e:
        logger.warning(f"Failed to parse log line: {e}")
        return None

def tail_log_file(file_path: Path, lines: int) -> List[str]:
    """Get last N lines from log file"""
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            # Read all lines and get the last N
            all_lines = f.readlines()
            return [line.rstrip() for line in all_lines[-lines:]]
    except Exception as e:
        logger.error(f"Failed to read log file {file_path}: {e}")
        return []

def search_logs(file_path: Path, pattern: str, max_results: int = 100) -> List[str]:
    """Search for pattern in log file"""
    try:
        results = []
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            for line_num, line in enumerate(f, 1):
                if re.search(pattern, line, re.IGNORECASE):
                    results.append(f"Line {line_num}: {line.rstrip()}")
                    if len(results) >= max_results:
                        break
        return results
    except Exception as e:
        logger.error(f"Failed to search log file {file_path}: {e}")
        return []

# API Endpoints
@router.get("/files")
async def list_log_files(
    current_user: User = Depends(get_current_user)
) -> List[LogFile]:
    """List available log files"""
    return get_log_files()

@router.get("/")
async def get_logs(
    service: Optional[str] = Query(None, description="Filter by service name"),
    level: Optional[LogLevel] = Query(None, description="Filter by log level"),
    lines: int = Query(100, ge=1, le=1000),
    search: Optional[str] = Query(None, description="Search pattern"),
    since: Optional[datetime] = Query(None, description="Show logs since timestamp"),
    current_user: User = Depends(get_current_user)
) -> List[LogEntry]:
    """Get system logs with filtering"""
    
    log_files = get_log_files()
    
    # Filter by service if specified
    if service:
        service_lower = service.lower()
        log_files = [f for f in log_files if service_lower in f.service.lower()]
    
    if not log_files:
        return []
    
    # Read and parse logs
    all_entries = []
    
    for log_file in log_files:
        try:
            file_path = Path(log_file.path)
            
            if search:
                # Search for pattern
                matching_lines = search_logs(file_path, search, lines)
                for line in matching_lines:
                    # Remove line number prefix for parsing
                    clean_line = re.sub(r'^Line \d+:\s*', '', line)
                    entry = parse_log_line(clean_line, log_file.service)
                    if entry:
                        all_entries.append(entry)
            else:
                # Get recent lines
                recent_lines = tail_log_file(file_path, lines)
                for line in recent_lines:
                    entry = parse_log_line(line, log_file.service)
                    if entry:
                        all_entries.append(entry)
                        
        except Exception as e:
            logger.error(f"Failed to process log file {log_file.path}: {e}")
    
    # Apply filters
    filtered_entries = all_entries
    
    if level:
        filtered_entries = [e for e in filtered_entries if e.level == level]
    
    if since:
        filtered_entries = [e for e in filtered_entries if e.timestamp >= since]
    
    # Sort by timestamp descending and limit
    filtered_entries.sort(key=lambda x: x.timestamp, reverse=True)
    
    return filtered_entries[:lines]

@router.get("/statistics")
async def get_log_statistics(
    hours: int = Query(24, ge=1, le=168),
    current_user: User = Depends(get_current_user)
) -> LogStatistics:
    """Get log statistics for the specified time period"""
    
    since = datetime.now() - timedelta(hours=hours)
    
    # Get all log entries
    log_entries = await get_logs(since=since, lines=10000, current_user=current_user)
    
    if not log_entries:
        return LogStatistics(
            total_entries=0,
            entries_by_level={},
            entries_by_service={},
            error_rate=0,
            warning_rate=0,
            most_common_errors=[],
            time_range={"start": since, "end": datetime.now()}
        )
    
    # Calculate statistics
    total_entries = len(log_entries)
    
    # Count by level
    entries_by_level = {}
    for entry in log_entries:
        level = entry.level.value
        entries_by_level[level] = entries_by_level.get(level, 0) + 1
    
    # Count by service
    entries_by_service = {}
    for entry in log_entries:
        service = entry.service
        entries_by_service[service] = entries_by_service.get(service, 0) + 1
    
    # Calculate rates
    error_count = entries_by_level.get("ERROR", 0) + entries_by_level.get("CRITICAL", 0)
    warning_count = entries_by_level.get("WARNING", 0)
    
    error_rate = (error_count / total_entries) * 100 if total_entries > 0 else 0
    warning_rate = (warning_count / total_entries) * 100 if total_entries > 0 else 0
    
    # Find most common errors
    error_messages = {}
    for entry in log_entries:
        if entry.level in [LogLevel.ERROR, LogLevel.CRITICAL]:
            # Simplify error message for grouping
            simplified_msg = re.sub(r'\d+', 'N', entry.message)[:100]
            error_messages[simplified_msg] = error_messages.get(simplified_msg, 0) + 1
    
    most_common_errors = [
        {"message": msg, "count": count}
        for msg, count in sorted(error_messages.items(), key=lambda x: x[1], reverse=True)[:10]
    ]
    
    return LogStatistics(
        total_entries=total_entries,
        entries_by_level=entries_by_level,
        entries_by_service=entries_by_service,
        error_rate=error_rate,
        warning_rate=warning_rate,
        most_common_errors=most_common_errors,
        time_range={"start": since, "end": datetime.now()}
    )

@router.get("/search")
async def search_logs_endpoint(
    pattern: str = Query(..., description="Search pattern (regex supported)"),
    service: Optional[str] = Query(None, description="Filter by service"),
    level: Optional[LogLevel] = Query(None, description="Filter by log level"),
    max_results: int = Query(100, ge=1, le=1000),
    current_user: User = Depends(get_current_user)
) -> List[str]:
    """Search logs for specific patterns"""
    
    log_files = get_log_files()
    
    # Filter by service if specified
    if service:
        service_lower = service.lower()
        log_files = [f for f in log_files if service_lower in f.service.lower()]
    
    all_results = []
    
    for log_file in log_files:
        try:
            file_path = Path(log_file.path)
            results = search_logs(file_path, pattern, max_results - len(all_results))
            
            # Add service prefix to results
            for result in results:
                all_results.append(f"[{log_file.service}] {result}")
            
            if len(all_results) >= max_results:
                break
                
        except Exception as e:
            logger.error(f"Failed to search log file {log_file.path}: {e}")
    
    return all_results

@router.get("/tail/{service}")
async def tail_service_logs(
    service: str,
    lines: int = Query(50, ge=1, le=500),
    current_user: User = Depends(get_current_user)
) -> List[str]:
    """Get real-time tail of service logs"""
    
    # Find log file for service
    log_files = get_log_files()
    service_file = None
    
    for log_file in log_files:
        if service.lower() in log_file.service.lower():
            service_file = log_file
            break
    
    if not service_file:
        raise HTTPException(status_code=404, detail=f"Log file for service '{service}' not found")
    
    file_path = Path(service_file.path)
    return tail_log_file(file_path, lines)

@router.get("/download/{service}")
async def download_log_file(
    service: str,
    current_user: User = Depends(get_current_user)
):
    """Download complete log file for a service"""
    from fastapi.responses import FileResponse
    
    # Find log file for service
    log_files = get_log_files()
    service_file = None
    
    for log_file in log_files:
        if service.lower() in log_file.service.lower():
            service_file = log_file
            break
    
    if not service_file:
        raise HTTPException(status_code=404, detail=f"Log file for service '{service}' not found")
    
    file_path = Path(service_file.path)
    
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Log file not found")
    
    return FileResponse(
        path=str(file_path),
        media_type='text/plain',
        filename=file_path.name
    )

@router.get("/alerts/rules")
async def get_alert_rules(
    current_user: User = Depends(get_current_user)
) -> List[AlertRule]:
    """Get log alert rules"""
    # Mock alert rules
    return [
        AlertRule(
            id="error-threshold",
            name="High Error Rate",
            pattern="ERROR|CRITICAL",
            level=LogLevel.ERROR,
            threshold=10,
            time_window_minutes=5,
            enabled=True,
            action="email"
        ),
        AlertRule(
            id="connection-failure",
            name="Connection Failures",
            pattern="connection.*failed|timeout",
            level=LogLevel.WARNING,
            threshold=5,
            time_window_minutes=10,
            enabled=True,
            action="webhook"
        )
    ]

@router.post("/alerts/rules")
async def create_alert_rule(
    rule: AlertRule,
    current_user: User = Depends(get_current_user)
) -> Dict:
    """Create new log alert rule"""
    # In production, save to configuration
    return {
        "status": "created",
        "rule_id": rule.id,
        "message": "Alert rule created successfully"
    }

@router.get("/health")
async def get_log_health(
    current_user: User = Depends(get_current_user)
) -> Dict:
    """Get logging system health status"""
    
    log_files = get_log_files()
    
    # Check if logs are recent
    now = datetime.now()
    recent_threshold = now - timedelta(minutes=5)
    
    healthy_services = []
    stale_services = []
    
    for log_file in log_files:
        if log_file.last_modified >= recent_threshold:
            healthy_services.append(log_file.service)
        else:
            stale_services.append({
                "service": log_file.service,
                "last_update": log_file.last_modified.isoformat(),
                "minutes_ago": int((now - log_file.last_modified).total_seconds() / 60)
            })
    
    return {
        "status": "healthy" if not stale_services else "warning",
        "total_log_files": len(log_files),
        "healthy_services": healthy_services,
        "stale_services": stale_services,
        "logs_directory_size_mb": sum(f.size_bytes for f in log_files) / (1024 * 1024),
        "oldest_log": min(log_files, key=lambda x: x.last_modified).last_modified.isoformat() if log_files else None,
        "newest_log": max(log_files, key=lambda x: x.last_modified).last_modified.isoformat() if log_files else None
    }