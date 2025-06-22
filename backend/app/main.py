#!/usr/bin/env python3
"""
TickerML Dashboard API - Main FastAPI Application
Unified control interface for the entire trading system
"""

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
import uvicorn
import os
import logging
from pathlib import Path
import time
from typing import Dict

# Import routers
from app.routers import (
    auth, system, data, trading, portfolio,
    models, market, config, logs, websocket
)

# Import services
from app.services.redis_service import init_redis, close_redis, redis_service

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="TickerML Dashboard API",
    description="Unified control interface for TickerML crypto trading bot",
    version="1.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc"
)

# CORS configuration for development
origins = [
    "http://localhost:3000",  # React dev server
    "http://localhost:5005",  # Production frontend
    "http://127.0.0.1:3000",
    "http://127.0.0.1:5005",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request timing middleware
@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    return response

# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Global exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error", "type": str(type(exc).__name__)}
    )

# Include routers
app.include_router(auth.router, prefix="/api/v1/auth", tags=["Authentication"])
app.include_router(system.router, prefix="/api/v1/system", tags=["System Control"])
app.include_router(data.router, prefix="/api/v1/data", tags=["Data Collection"])
app.include_router(trading.router, prefix="/api/v1/trading", tags=["Trading"])
app.include_router(portfolio.router, prefix="/api/v1/portfolio", tags=["Portfolio"])
app.include_router(models.router, prefix="/api/v1/models", tags=["Models"])
app.include_router(market.router, prefix="/api/v1/market", tags=["Market Data"])
app.include_router(config.router, prefix="/api/v1/config", tags=["Configuration"])
app.include_router(logs.router, prefix="/api/v1/logs", tags=["Logs"])
app.include_router(websocket.router, prefix="/api/v1", tags=["WebSocket"])

# Mount static files for React frontend (in production)
frontend_build_path = Path(__file__).parent.parent.parent / "frontend" / "build"
if frontend_build_path.exists():
    app.mount("/", StaticFiles(directory=str(frontend_build_path), html=True), name="static")
    logger.info(f"Mounted frontend static files from {frontend_build_path}")

# Health check endpoint
@app.get("/api/health")
async def health_check() -> Dict:
    """Health check endpoint"""
    health_data = {
        "status": "healthy",
        "service": "tickerml-dashboard-api",
        "version": "1.0.0",
        "timestamp": time.time()
    }
    
    # Add Redis health check
    try:
        redis_health = await redis_service.health_check()
        health_data["redis"] = redis_health
        
        # Update overall status based on Redis
        if redis_health.get("status") == "error":
            health_data["status"] = "degraded"
            
    except Exception as e:
        health_data["redis"] = {"status": "error", "error": str(e)}
        health_data["status"] = "degraded"
    
    return health_data

# Startup and shutdown events
@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    try:
        # Initialize Redis connection
        redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")
        await init_redis(redis_url)
        logger.info("Redis connection initialized")
    except Exception as e:
        logger.error(f"Failed to initialize Redis: {e}")
        # Continue without Redis if it's not available

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    try:
        await close_redis()
        logger.info("Redis connection closed")
    except Exception as e:
        logger.error(f"Error during shutdown: {e}")

# Root endpoint
@app.get("/api")
async def root() -> Dict[str, str]:
    """API root endpoint"""
    return {
        "message": "TickerML Dashboard API",
        "documentation": "/api/docs",
        "health": "/api/health"
    }

def main():
    """Main entry point"""
    host = os.getenv("DASHBOARD_HOST", "0.0.0.0")
    port = int(os.getenv("DASHBOARD_PORT", "8000"))
    
    logger.info(f"Starting TickerML Dashboard API on {host}:{port}")
    
    uvicorn.run(
        "app.main:app",
        host=host,
        port=port,
        reload=True if os.getenv("ENVIRONMENT", "development") == "development" else False,
        log_level="info"
    )

if __name__ == "__main__":
    main()