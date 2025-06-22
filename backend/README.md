# TickerML Dashboard API

Unified control interface for the TickerML crypto trading bot system.

## Features

- **Authentication**: JWT-based authentication with role-based access control
- **System Control**: Start/stop services, monitor system health
- **Data Management**: Configure and monitor data collectors
- **Trading Control**: Manage paper trading, set risk limits
- **Portfolio Tracking**: Real-time portfolio analytics and performance metrics
- **WebSocket Streaming**: Real-time market data, portfolio updates, and alerts
- **RESTful API**: Comprehensive endpoints for all system functions

## Quick Start

### Prerequisites

- Python 3.8+
- Redis (for caching and Celery)
- Running TickerML services (data collectors, paper trader, etc.)

### Installation

1. Install dependencies:
```bash
cd backend
pip install -r requirements.txt
```

2. Set environment variables:
```bash
export DASHBOARD_SECRET_KEY="your-secure-secret-key"
export DASHBOARD_HOST="0.0.0.0"
export DASHBOARD_PORT="8000"
```

3. Start Redis (if not running):
```bash
redis-server
```

4. Run the API server:
```bash
python -m app.main
```

The API will be available at `http://localhost:8000`

## API Documentation

Interactive API documentation is available at:
- Swagger UI: `http://localhost:8000/api/docs`
- ReDoc: `http://localhost:8000/api/redoc`

## Authentication

Default users for testing:
- **Admin**: username: `admin`, password: `admin123`
- **Trader**: username: `trader`, password: `trader123`  
- **Viewer**: username: `viewer`, password: `viewer123`

To authenticate:
1. POST to `/api/v1/auth/login` with credentials
2. Use the returned `access_token` in the `Authorization: Bearer <token>` header

## Main Endpoints

### System Control
- `GET /api/v1/system/status` - Get system status
- `POST /api/v1/system/services/{service_name}/start` - Start a service
- `POST /api/v1/system/services/{service_name}/stop` - Stop a service
- `POST /api/v1/system/services/all/start` - Start all services

### Data Collection
- `GET /api/v1/data/collectors` - List data collectors
- `POST /api/v1/data/collectors/{type}/configure` - Configure collector
- `GET /api/v1/data/statistics` - Get data statistics
- `GET /api/v1/data/quality` - Get data quality metrics

### Trading
- `GET /api/v1/trading/status` - Get trading status
- `POST /api/v1/trading/start` - Start paper trading
- `POST /api/v1/trading/stop` - Stop paper trading
- `GET /api/v1/trading/positions` - Get open positions
- `GET /api/v1/trading/settings` - Get risk settings

### Portfolio
- `GET /api/v1/portfolio/snapshot` - Current portfolio snapshot
- `GET /api/v1/portfolio/history` - Portfolio value history
- `GET /api/v1/portfolio/performance` - Performance metrics
- `GET /api/v1/portfolio/risk` - Risk metrics

### WebSocket Endpoints
- `ws://localhost:8000/api/v1/ws/market` - Real-time market data
- `ws://localhost:8000/api/v1/ws/portfolio` - Portfolio updates
- `ws://localhost:8000/api/v1/ws/trades` - Trade updates
- `ws://localhost:8000/api/v1/ws/orderbook` - Order book updates
- `ws://localhost:8000/api/v1/ws/alerts` - System alerts

## Development

### Running Tests
```bash
pytest tests/
```

### Code Structure
```
backend/
├── app/
│   ├── main.py          # FastAPI application
│   ├── routers/         # API route handlers
│   │   ├── auth.py      # Authentication
│   │   ├── system.py    # System control
│   │   ├── data.py      # Data management
│   │   ├── trading.py   # Trading control
│   │   ├── portfolio.py # Portfolio tracking
│   │   └── websocket.py # WebSocket endpoints
│   ├── models/          # Pydantic models
│   ├── services/        # Business logic
│   └── utils/           # Utilities
├── requirements.txt
└── README.md
```

## Security Considerations

1. **Change the secret key** in production
2. Use **HTTPS** in production
3. Configure **CORS** appropriately
4. Implement **rate limiting**
5. Use **environment variables** for sensitive data
6. Regular **security audits**

## Next Steps

1. Connect the React frontend
2. Implement Redis caching
3. Setup Celery for background tasks
4. Add comprehensive error handling
5. Implement proper logging
6. Add monitoring and metrics