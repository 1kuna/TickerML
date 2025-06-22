# TickerML Unified Dashboard - Complete Implementation Guide

*A comprehensive guide to building a production-grade web dashboard for complete trading bot control*

## 🎯 Dashboard Overview

### Core Requirements
- **Single Interface**: Control everything from one dashboard
- **Real-Time Updates**: Live data streams, portfolio values, system metrics
- **Remote Access**: Secure access from anywhere
- **Beautiful UI**: Modern, professional trading interface
- **High Performance**: Handle real-time data without lag
- **Desktop Optimized**: Designed for multi-monitor trading setups

### Technology Stack
```yaml
Backend:
  Framework: FastAPI (async, WebSocket support, auto-docs)
  Real-time: WebSockets + Server-Sent Events
  Task Queue: Celery with Redis
  Cache: Redis for real-time data
  
Frontend:
  Framework: React with TypeScript
  UI Library: Ant Design or Material-UI
  Charts: TradingView Lightweight Charts + Recharts
  State: Redux Toolkit with RTK Query
  Real-time: Socket.io-client
  
Deployment:
  Reverse Proxy: Nginx
  Process Manager: PM2 or Supervisor
  SSL: Let's Encrypt
  Auth: JWT with refresh tokens
```

---

## 📐 Architecture Design

### System Architecture
```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│                 │     │                  │     │                 │
│  React Frontend ├────►│  FastAPI Backend ├────►│  Trading System │
│                 │     │                  │     │                 │
└─────────────────┘     └────────┬─────────┘     └─────────────────┘
                                 │
                    ┌────────────┼────────────┐
                    │            │            │
              ┌─────▼─────┐ ┌───▼────┐ ┌─────▼─────┐
              │   Redis   │ │ Kafka  │ │TimescaleDB│
              │  (Cache)  │ │(Events)│ │  (Data)   │
              └───────────┘ └────────┘ └───────────┘
```

### API Structure
```
/api/v1/
├── /auth           # Authentication endpoints
├── /system         # System control (start/stop services)
├── /data           # Data collection management
├── /trading        # Paper trading control
├── /portfolio      # Portfolio and positions
├── /models         # Model training and management
├── /market         # Real-time market data
├── /config         # Configuration management
├── /logs           # System logs and alerts
└── /ws            # WebSocket endpoints
```

---

## 🛠️ Backend Implementation

### 1. FastAPI Application Structure
```python
# app/main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import uvicorn

app = FastAPI(title="TickerML Dashboard API")

# CORS for development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount React build
app.mount("/", StaticFiles(directory="frontend/build", html=True))

# Include routers
from app.routers import (
    auth, system, data, trading, portfolio, 
    models, market, config, logs, websocket
)

app.include_router(auth.router, prefix="/api/v1/auth")
app.include_router(system.router, prefix="/api/v1/system")
# ... include all routers
```

### 2. Authentication System
```python
# app/routers/auth.py
from fastapi import APIRouter, Depends, HTTPException
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt
from datetime import datetime, timedelta

router = APIRouter()
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# Secure configuration
SECRET_KEY = os.getenv("DASHBOARD_SECRET_KEY")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

@router.post("/login")
async def login(credentials: LoginCredentials):
    # Verify credentials
    # Generate JWT token
    # Return access & refresh tokens
    
@router.post("/refresh")
async def refresh_token(refresh_token: str):
    # Validate refresh token
    # Generate new access token
```

### 3. System Control Endpoints
```python
# app/routers/system.py
from fastapi import APIRouter, BackgroundTasks
import subprocess
import psutil

router = APIRouter()

@router.get("/status")
async def get_system_status():
    """Get status of all system components"""
    return {
        "kafka": check_kafka_status(),
        "data_collectors": get_collector_status(),
        "paper_trader": get_trader_status(),
        "model_trainer": get_trainer_status(),
        "system_resources": {
            "cpu_percent": psutil.cpu_percent(),
            "memory_percent": psutil.virtual_memory().percent,
            "disk_usage": psutil.disk_usage('/').percent,
            "gpu_usage": get_gpu_usage()  # nvidia-ml-py
        }
    }

@router.post("/services/{service_name}/start")
async def start_service(service_name: str, background_tasks: BackgroundTasks):
    """Start a system service"""
    services = {
        "data_collector": "python raspberry_pi/orderbook_collector.py",
        "paper_trader": "python raspberry_pi/paper_trader.py",
        "kafka": "scripts/start_kafka.sh",
        # ... more services
    }
    
    if service_name in services:
        background_tasks.add_task(run_service, services[service_name])
        return {"status": "starting", "service": service_name}
        
@router.post("/services/{service_name}/stop")
async def stop_service(service_name: str):
    """Stop a system service"""
    # Find and terminate process
    # Update status in Redis
```

### 4. Data Collection Management
```python
# app/routers/data.py
@router.get("/collectors")
async def get_collectors():
    """List all data collectors and their status"""
    return {
        "orderbook": {
            "status": "running",
            "exchanges": ["binance", "coinbase", "kraken"],
            "symbols": ["BTC/USDT", "ETH/USDT"],
            "messages_per_second": get_kafka_rate("orderbooks"),
            "last_update": get_last_update("orderbook")
        },
        "trades": {...},
        "funding": {...}
    }

@router.post("/collectors/{collector_type}/configure")
async def configure_collector(collector_type: str, config: CollectorConfig):
    """Update collector configuration"""
    # Update config file
    # Restart collector if running
    
@router.get("/data/statistics")
async def get_data_statistics():
    """Get data collection statistics"""
    # Query TimescaleDB for counts, gaps, quality metrics
```

### 5. Trading Control
```python
# app/routers/trading.py
@router.post("/trading/start")
async def start_trading(config: TradingConfig):
    """Start paper trading with configuration"""
    # Validate config
    # Start paper trader
    # Return session ID
    
@router.post("/trading/stop")
async def stop_trading():
    """Stop paper trading"""
    # Gracefully close positions
    # Stop trader
    
@router.post("/trading/orders")
async def place_manual_order(order: OrderRequest):
    """Place manual order (for testing)"""
    # Validate order
    # Send to paper trader
    
@router.get("/trading/settings")
async def get_trading_settings():
    """Get current risk limits and settings"""
    # Return from config/risk_limits.yaml
```

### 6. Real-Time WebSocket Endpoints
```python
# app/routers/websocket.py
from fastapi import WebSocket, WebSocketDisconnect
import asyncio
import json

class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []
        
    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        
    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)
        
    async def broadcast(self, message: dict):
        for connection in self.active_connections:
            await connection.send_json(message)

manager = ConnectionManager()

@router.websocket("/ws/market")
async def websocket_market_data(websocket: WebSocket):
    """Stream real-time market data"""
    await manager.connect(websocket)
    try:
        # Subscribe to Kafka topics
        consumer = create_kafka_consumer(['orderbooks', 'trades'])
        
        while True:
            # Get data from Kafka
            data = await get_next_market_update(consumer)
            await websocket.send_json(data)
            
    except WebSocketDisconnect:
        manager.disconnect(websocket)
        
@router.websocket("/ws/portfolio")
async def websocket_portfolio(websocket: WebSocket):
    """Stream portfolio updates"""
    # Similar pattern for portfolio updates
```

### 7. Model Management
```python
# app/routers/models.py
@router.get("/models")
async def list_models():
    """List all trained models"""
    # Scan model directory
    # Return model info with metrics
    
@router.post("/models/train")
async def start_training(config: TrainingConfig):
    """Start model training"""
    # Validate data availability
    # Start training job with Celery
    # Return job ID
    
@router.get("/models/training/{job_id}")
async def get_training_status(job_id: str):
    """Get training job status"""
    # Query Celery for job status
    # Return progress, metrics, ETA
    
@router.post("/models/{model_id}/deploy")
async def deploy_model(model_id: str):
    """Deploy model for paper trading"""
    # Validate model
    # Update active model
    # Restart paper trader
```

---

## 🎨 Frontend Implementation

### 1. React Application Structure
```
frontend/
├── public/
├── src/
│   ├── components/
│   │   ├── Layout/
│   │   │   ├── Dashboard.tsx
│   │   │   ├── Sidebar.tsx
│   │   │   └── Header.tsx
│   │   ├── Trading/
│   │   │   ├── OrderBook.tsx
│   │   │   ├── TradingChart.tsx
│   │   │   ├── PositionManager.tsx
│   │   │   └── OrderForm.tsx
│   │   ├── Portfolio/
│   │   │   ├── PortfolioOverview.tsx
│   │   │   ├── PerformanceChart.tsx
│   │   │   └── RiskMetrics.tsx
│   │   ├── DataCollection/
│   │   │   ├── CollectorStatus.tsx
│   │   │   ├── DataQuality.tsx
│   │   │   └── StreamMonitor.tsx
│   │   ├── Models/
│   │   │   ├── ModelList.tsx
│   │   │   ├── TrainingProgress.tsx
│   │   │   └── ModelMetrics.tsx
│   │   └── System/
│   │       ├── SystemHealth.tsx
│   │       ├── LogViewer.tsx
│   │       └── ConfigEditor.tsx
│   ├── hooks/
│   │   ├── useWebSocket.ts
│   │   ├── useApi.ts
│   │   └── useAuth.ts
│   ├── store/
│   │   ├── index.ts
│   │   └── slices/
│   ├── services/
│   │   └── api.ts
│   └── App.tsx
```

### 2. Main Dashboard Layout
```tsx
// src/components/Layout/Dashboard.tsx
import React from 'react';
import { Layout, Menu, Card, Row, Col } from 'antd';
import {
  DashboardOutlined,
  LineChartOutlined,
  DatabaseOutlined,
  RobotOutlined,
  SettingOutlined
} from '@ant-design/icons';

const { Header, Sider, Content } = Layout;

export const Dashboard: React.FC = () => {
  return (
    <Layout style={{ minHeight: '100vh' }}>
      <Sider width={200} theme="dark">
        <div className="logo">TickerML</div>
        <Menu theme="dark" mode="inline" defaultSelectedKeys={['1']}>
          <Menu.Item key="1" icon={<DashboardOutlined />}>
            Overview
          </Menu.Item>
          <Menu.Item key="2" icon={<LineChartOutlined />}>
            Trading
          </Menu.Item>
          <Menu.Item key="3" icon={<DatabaseOutlined />}>
            Data Collection
          </Menu.Item>
          <Menu.Item key="4" icon={<RobotOutlined />}>
            Models
          </Menu.Item>
          <Menu.Item key="5" icon={<SettingOutlined />}>
            Settings
          </Menu.Item>
        </Menu>
      </Sider>
      
      <Layout>
        <Header style={{ background: '#fff', padding: 0 }}>
          <SystemStatus />
        </Header>
        
        <Content style={{ margin: '24px 16px 0' }}>
          <RouterOutlet />
        </Content>
      </Layout>
    </Layout>
  );
};
```

### 3. Real-Time Trading View
```tsx
// src/components/Trading/TradingChart.tsx
import React, { useEffect, useRef } from 'react';
import { createChart, IChartApi } from 'lightweight-charts';
import { useWebSocket } from '../../hooks/useWebSocket';

export const TradingChart: React.FC = () => {
  const chartContainerRef = useRef<HTMLDivElement>(null);
  const chart = useRef<IChartApi>();
  const { data } = useWebSocket('/ws/market');
  
  useEffect(() => {
    if (chartContainerRef.current) {
      chart.current = createChart(chartContainerRef.current, {
        width: chartContainerRef.current.clientWidth,
        height: 400,
        layout: {
          backgroundColor: '#ffffff',
          textColor: '#333',
        },
        grid: {
          vertLines: { color: '#e1e1e1' },
          horzLines: { color: '#e1e1e1' },
        },
      });
      
      const candlestickSeries = chart.current.addCandlestickSeries();
      const volumeSeries = chart.current.addHistogramSeries({
        priceFormat: { type: 'volume' },
        priceScaleId: '',
      });
      
      // Add order book visualization
      const orderBookSeries = chart.current.addAreaSeries({
        topColor: 'rgba(38, 166, 154, 0.28)',
        bottomColor: 'rgba(38, 166, 154, 0.05)',
        lineColor: 'rgba(38, 166, 154, 1)',
      });
    }
  }, []);
  
  // Update chart with WebSocket data
  useEffect(() => {
    if (data && chart.current) {
      // Update candlesticks, volume, order book
    }
  }, [data]);
  
  return <div ref={chartContainerRef} />;
};
```

### 4. Portfolio Performance Dashboard
```tsx
// src/components/Portfolio/PortfolioOverview.tsx
import React from 'react';
import { Card, Row, Col, Statistic, Progress } from 'antd';
import { ArrowUpOutlined, ArrowDownOutlined } from '@ant-design/icons';
import { usePortfolio } from '../../hooks/usePortfolio';

export const PortfolioOverview: React.FC = () => {
  const { portfolio, performance } = usePortfolio();
  
  return (
    <div>
      <Row gutter={16}>
        <Col span={6}>
          <Card>
            <Statistic
              title="Portfolio Value"
              value={portfolio.totalValue}
              precision={2}
              prefix="$"
              valueStyle={{ color: '#3f8600' }}
            />
          </Card>
        </Col>
        
        <Col span={6}>
          <Card>
            <Statistic
              title="Today's P&L"
              value={performance.dailyPnL}
              precision={2}
              prefix="$"
              valueStyle={{
                color: performance.dailyPnL >= 0 ? '#3f8600' : '#cf1322'
              }}
              suffix={
                performance.dailyPnL >= 0 ? 
                <ArrowUpOutlined /> : 
                <ArrowDownOutlined />
              }
            />
          </Card>
        </Col>
        
        <Col span={6}>
          <Card>
            <Statistic
              title="Win Rate"
              value={performance.winRate}
              precision={1}
              suffix="%"
            />
          </Card>
        </Col>
        
        <Col span={6}>
          <Card>
            <Statistic title="Max Drawdown" value={performance.maxDrawdown}>
              <Progress 
                percent={performance.maxDrawdown} 
                status={performance.maxDrawdown > 20 ? 'exception' : 'normal'}
              />
            </Statistic>
          </Card>
        </Col>
      </Row>
      
      <Row gutter={16} style={{ marginTop: 16 }}>
        <Col span={12}>
          <Card title="Positions">
            <PositionList positions={portfolio.positions} />
          </Card>
        </Col>
        
        <Col span={12}>
          <Card title="Risk Metrics">
            <RiskMetrics metrics={performance.riskMetrics} />
          </Card>
        </Col>
      </Row>
    </div>
  );
};
```

### 5. Model Training Interface
```tsx
// src/components/Models/TrainingProgress.tsx
import React, { useState } from 'react';
import { Card, Button, Progress, Timeline, Form, Select } from 'antd';
import { PlayCircleOutlined, PauseCircleOutlined } from '@ant-design/icons';

export const TrainingProgress: React.FC = () => {
  const [training, setTraining] = useState(false);
  const [progress, setProgress] = useState(0);
  
  const startTraining = async (values: any) => {
    const response = await api.post('/models/train', values);
    setTraining(true);
    // Poll for progress
  };
  
  return (
    <Card title="Model Training">
      <Form onFinish={startTraining}>
        <Form.Item name="modelType" label="Model Type">
          <Select>
            <Select.Option value="decision_transformer">
              Decision Transformer
            </Select.Option>
            <Select.Option value="price_prediction">
              Price Prediction
            </Select.Option>
          </Select>
        </Form.Item>
        
        <Form.Item name="dataRange" label="Data Range">
          <RangePicker />
        </Form.Item>
        
        <Form.Item>
          <Button 
            type="primary" 
            icon={<PlayCircleOutlined />}
            htmlType="submit"
            disabled={training}
          >
            Start Training
          </Button>
        </Form.Item>
      </Form>
      
      {training && (
        <div>
          <Progress percent={progress} status="active" />
          <Timeline>
            <Timeline.Item color="green">
              Data preprocessing complete
            </Timeline.Item>
            <Timeline.Item color="blue">
              Training epoch 10/100
            </Timeline.Item>
            <Timeline.Item color="gray">
              Validation pending
            </Timeline.Item>
          </Timeline>
        </div>
      )}
    </Card>
  );
};
```

### 6. System Control Panel
```tsx
// src/components/System/SystemControl.tsx
import React from 'react';
import { Card, Switch, Button, Tag, Space } from 'antd';
import { useSystemStatus } from '../../hooks/useSystemStatus';

export const SystemControl: React.FC = () => {
  const { services, toggleService } = useSystemStatus();
  
  return (
    <Card title="System Services">
      <Space direction="vertical" style={{ width: '100%' }}>
        {Object.entries(services).map(([name, status]) => (
          <Card key={name} size="small">
            <Space style={{ width: '100%', justifyContent: 'space-between' }}>
              <span>{name}</span>
              <Tag color={status.running ? 'green' : 'red'}>
                {status.running ? 'Running' : 'Stopped'}
              </Tag>
              <Switch
                checked={status.running}
                onChange={() => toggleService(name)}
                checkedChildren="ON"
                unCheckedChildren="OFF"
              />
            </Space>
          </Card>
        ))}
      </Space>
      
      <Space style={{ marginTop: 16 }}>
        <Button type="primary">Start All</Button>
        <Button danger>Stop All</Button>
        <Button>Restart All</Button>
      </Space>
    </Card>
  );
};
```

---

## 🚀 Deployment Guide

### 1. Production Setup
```bash
# Install dependencies
cd frontend && npm install && npm run build
cd ../backend && pip install -r requirements.txt

# Setup Nginx
sudo nano /etc/nginx/sites-available/tickerml
```

### 2. Nginx Configuration
```nginx
server {
    listen 80;
    server_name your-domain.com;
    
    # Redirect to HTTPS
    return 301 https://$server_name$request_uri;
}

server {
    listen 443 ssl http2;
    server_name your-domain.com;
    
    ssl_certificate /etc/letsencrypt/live/your-domain.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/your-domain.com/privkey.pem;
    
    # WebSocket support
    location /ws {
        proxy_pass http://localhost:8000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
    }
    
    # API routes
    location /api {
        proxy_pass http://localhost:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
    
    # Static files
    location / {
        root /var/www/tickerml/frontend/build;
        try_files $uri $uri/ /index.html;
    }
}
```

### 3. Process Management
```bash
# Create systemd service
sudo nano /etc/systemd/system/tickerml-dashboard.service

[Unit]
Description=TickerML Dashboard
After=network.target

[Service]
Type=simple
User=tickerml
WorkingDirectory=/home/tickerml/TickerML
Environment="PATH=/home/tickerml/.local/bin"
ExecStart=/usr/bin/python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
Restart=always

[Install]
WantedBy=multi-user.target

# Enable and start
sudo systemctl enable tickerml-dashboard
sudo systemctl start tickerml-dashboard
```

---

## 🎨 UI/UX Best Practices

### 1. Dashboard Design Principles
- **Dark Mode Support**: Essential for long trading sessions
- **Information Hierarchy**: Most important metrics prominent
- **Color Coding**: Green = profit, Red = loss, Yellow = warning
- **Responsive Grids**: Adapt to screen size
- **Real-Time Updates**: Smooth animations, no jarring refreshes

### 2. Key Dashboard Views

#### Main Overview
```
┌─────────────────────────────────────────────────────┐
│ Portfolio Value: $12,453  |  Daily P&L: +$234 (1.9%)│
├──────────────┬──────────────┬──────────────────────┤
│              │              │                      │
│  Trading     │  Order Book  │   System Health      │
│   Chart      │   Depth      │   CPU: 45%           │
│              │              │   RAM: 62%           │
│              │              │   GPU: 78%           │
├──────────────┴──────────────┼──────────────────────┤
│  Open Positions             │  Recent Trades       │
│  BTC/USDT  +2.3%           │  09:32 Buy BTC       │
│  ETH/USDT  -0.8%           │  09:28 Sell ETH     │
└─────────────────────────────┴──────────────────────┘
```

#### Data Collection Monitor
```
┌─────────────────────────────────────────────────────┐
│              Data Collection Status                  │
├─────────────┬───────────┬──────────┬───────────────┤
│  Exchange   │  Status   │  Msg/sec │  Last Update  │
├─────────────┼───────────┼──────────┼───────────────┤
│  Binance    │  🟢 Live  │   145    │  2 sec ago    │
│  Coinbase   │  🟢 Live  │   132    │  1 sec ago    │
│  Kraken     │  🟡 Slow  │   45     │  5 sec ago    │
│  KuCoin     │  🔴 Down  │   0      │  2 min ago    │
└─────────────┴───────────┴──────────┴───────────────┘
```

### 3. Interactive Features
- **Drag & Drop**: Rearrange dashboard widgets
- **Detachable Panels**: Pop out charts to separate windows
- **Zoom Controls**: On all charts with mouse wheel support
- **Quick Actions**: One-click stop trading, pause data
- **Notifications**: Browser notifications for alerts
- **Keyboard Shortcuts**: For power users (Ctrl+T for trade, Ctrl+S for stop, etc.)
- **Multi-Chart Layouts**: Compare multiple timeframes/symbols
- **Workspace Saving**: Save and load custom layouts

---

## 🔒 Security Considerations

### 1. Authentication & Authorization
```python
# Implement role-based access
class UserRole(Enum):
    VIEWER = "viewer"      # Read-only access
    TRADER = "trader"      # Can execute trades
    ADMIN = "admin"        # Full system control

# Protect sensitive endpoints
@router.post("/trading/start", dependencies=[Depends(require_role(UserRole.TRADER))])
```

### 2. Security Checklist
- [ ] HTTPS only (no HTTP in production)
- [ ] JWT tokens with short expiry
- [ ] Rate limiting on all endpoints
- [ ] Input validation and sanitization
- [ ] SQL injection protection
- [ ] XSS prevention
- [ ] CORS properly configured
- [ ] Secrets in environment variables
- [ ] Regular security audits

## 💻 Desktop UI/UX Best Practices

### 1. Dashboard Design Principles
- **Dark Mode Support**: Essential for long trading sessions
- **Information Hierarchy**: Most important metrics prominent
- **Color Coding**: Green = profit, Red = loss, Yellow = warning
- **Multi-Monitor Support**: Detachable panels for multiple screens
- **Real-Time Updates**: Smooth animations, no jarring refreshes
- **High Information Density**: Maximize use of screen real estate

### 2. Key Dashboard Views

#### Main Overview (Optimized for 1920x1080+)
```
┌─────────────────────────────────────────────────────────────────┐
│ Portfolio Value: $12,453  |  Daily P&L: +$234 (1.9%) | Sharpe: 1.8│
├──────────────┬──────────────┬─────────────────┬─────────────────┤
│              │              │                 │                 │
│  Trading     │  Order Book  │  Risk Metrics   │  System Health  │
│   Chart      │   Depth      │  Drawdown: 12%  │  CPU: 45%       │
│  (50%)       │  (25%)       │  VaR: $1,234    │  RAM: 62%       │
│              │              │  Correlation    │  GPU: 78%       │
│              │              │  Matrix         │  Disk: 41%      │
├──────────────┴──────────────┼─────────────────┴─────────────────┤
│  Open Positions (25%)       │  Recent Trades & Alerts (25%)     │
│  BTC/USDT  +2.3%  $5,234   │  09:32 Buy BTC @ 43,567          │
│  ETH/USDT  -0.8%  $3,421   │  09:28 Sell ETH @ 3,234          │
│  Total: 2 positions         │  09:25 ALERT: High correlation    │
└─────────────────────────────┴───────────────────────────────────┘
```

---

## 🧪 Testing Strategy

### 1. Backend Tests
```python
# tests/test_api.py
import pytest
from fastapi.testclient import TestClient

def test_system_status(client: TestClient):
    response = client.get("/api/v1/system/status")
    assert response.status_code == 200
    assert "kafka" in response.json()

def test_start_trading(client: TestClient, auth_headers):
    response = client.post(
        "/api/v1/trading/start",
        headers=auth_headers,
        json={"initial_balance": 10000}
    )
    assert response.status_code == 200
```

### 2. Frontend Tests
```tsx
// src/components/Portfolio/PortfolioOverview.test.tsx
import { render, screen } from '@testing-library/react';
import { PortfolioOverview } from './PortfolioOverview';

test('displays portfolio value', () => {
  render(<PortfolioOverview />);
  expect(screen.getByText(/Portfolio Value/i)).toBeInTheDocument();
});
```

### 3. E2E Tests
```javascript
// cypress/integration/trading.spec.js
describe('Trading Flow', () => {
  it('can start and stop paper trading', () => {
    cy.login();
    cy.visit('/trading');
    cy.contains('Start Trading').click();
    cy.contains('Trading Active').should('be.visible');
    cy.contains('Stop Trading').click();
    cy.contains('Trading Stopped').should('be.visible');
  });
});
```

---

## 📋 Implementation Checklist

### Phase 1: Backend Foundation (Week 1)
- [x] Setup FastAPI project structure
- [x] Implement authentication system
- [x] Create system control endpoints
- [x] Setup WebSocket connections
- [x] Integrate with existing services
- [x] Create Redis caching layer

### Phase 2: Core Frontend (Week 2)
- [x] Setup React with TypeScript
- [x] Implement routing and layout
- [x] Create authentication flow
- [x] Build system status component
- [x] Implement WebSocket hooks
- [x] Add state management

### Phase 3: Trading Features (Week 3)
- [x] Trading chart component
- [x] Order book visualization
- [x] Portfolio management UI
- [x] Position tracking
- [x] P&L calculations
- [x] Risk metrics display

### Phase 4: Data & Models (Week 4)
- [x] Data collection monitor
- [x] Model training interface
- [x] Performance metrics
- [x] Configuration editor
- [x] Log viewer
- [x] Alert management

### Phase 5: Polish & Deploy (Week 5)
- [ ] Responsive design
- [x] Dark mode
- [x] Performance optimization
- [ ] Security hardening
- [x] Documentation
- [x] Production deployment

---

## 🎯 Success Criteria

1. **Performance**: Updates render in <100ms
2. **Reliability**: 99.9% uptime
3. **Usability**: New users productive in <10 minutes
4. **Desktop Experience**: Optimized for 1080p+ displays
5. **Security**: Pass security audit
6. **Scalability**: Handle 1000+ updates/second

---

*This guide provides a complete roadmap for building a professional-grade dashboard that gives you full control over your TickerML trading system. Follow each section systematically for best results.*