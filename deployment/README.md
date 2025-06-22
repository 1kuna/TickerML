# TickerML Production Deployment Guide

This directory contains all the necessary configuration files and scripts for deploying TickerML in a production environment.

## 🚀 Deployment Options

### Option 1: Traditional Server Deployment

Perfect for VPS, dedicated servers, or cloud instances.

**Requirements:**
- Ubuntu 20.04+ or similar Linux distribution
- 4GB+ RAM
- 2+ CPU cores
- 50GB+ storage
- Domain name with DNS pointing to server

**Quick Deploy:**
```bash
# Clone repository
git clone https://github.com/yourusername/TickerML.git
cd TickerML/deployment/scripts

# Set environment variables
export DOMAIN=your-domain.com
export EMAIL=admin@your-domain.com

# Run deployment script
chmod +x deploy.sh
sudo ./deploy.sh
```

### Option 2: Docker Deployment

Ideal for development, testing, or containerized production environments.

**Requirements:**
- Docker 20.10+
- Docker Compose 2.0+
- 8GB+ RAM
- 4+ CPU cores

**Quick Deploy:**
```bash
# Clone repository
git clone https://github.com/yourusername/TickerML.git
cd TickerML/deployment/docker

# Configure environment
cp .env.example .env
# Edit .env with your API keys and passwords

# Deploy with Docker Compose
docker-compose up -d

# For production with monitoring
docker-compose --profile production --profile monitoring up -d
```

## 📋 Pre-Deployment Checklist

### 1. Server Requirements
- [ ] Linux server with root/sudo access
- [ ] Domain name configured with DNS
- [ ] SSL certificate (automated via Let's Encrypt)
- [ ] Required ports open (80, 443, 8000)

### 2. API Keys Setup
- [ ] Binance US API keys (read-only recommended)
- [ ] Coinbase Pro API keys (read-only recommended)
- [ ] Kraken API keys (read-only recommended)
- [ ] KuCoin API keys (read-only recommended)

### 3. Security Configuration
- [ ] Strong passwords for database and services
- [ ] JWT secret key generated
- [ ] Firewall configured
- [ ] Regular backup strategy planned

## 🔧 Configuration Files

### Nginx Configuration (`nginx/tickerml.conf`)
- SSL/TLS termination with Let's Encrypt
- Rate limiting for API endpoints
- Static file serving with caching
- Security headers and CORS configuration
- WebSocket support for real-time features

### Systemd Services (`systemd/`)
- `tickerml-dashboard.service` - Main dashboard API
- `tickerml-data-collector.service` - Real-time data collection
- `tickerml-paper-trader.service` - Paper trading engine

### Docker Configuration (`docker/`)
- Multi-stage build for optimized images
- PostgreSQL with TimescaleDB for time-series data
- Redis for caching and session storage
- Kafka for event streaming
- Optional monitoring with Prometheus and Grafana

## 🔐 Security Features

### Application Security
- JWT-based authentication with refresh tokens
- Rate limiting on all API endpoints
- Input validation and sanitization
- CORS properly configured for production
- Secure session management

### Infrastructure Security
- HTTPS-only with HSTS headers
- Security headers (CSP, X-Frame-Options, etc.)
- Database connections encrypted
- Non-root user execution
- Resource limits and quotas

### Monitoring & Alerting
- Health check endpoints
- Automated service restart on failure
- Log rotation and archiving
- Performance metrics collection
- Error rate monitoring

## 🏗️ Architecture Overview

```
Internet
    ↓
Nginx (SSL, Rate Limiting)
    ↓
TickerML Dashboard API (FastAPI)
    ↓
┌─────────────────┬─────────────────┬─────────────────┐
│   PostgreSQL    │     Redis       │     Kafka       │
│  (TimescaleDB)  │   (Caching)     │  (Streaming)    │
└─────────────────┴─────────────────┴─────────────────┘
    ↓
Background Services
├── Data Collector (Real-time market data)
├── Paper Trader (Trading simulation)
├── Feature Generator (ML feature engineering)
└── Risk Manager (Risk monitoring)
```

## 📊 Monitoring & Maintenance

### Service Management
```bash
# Check service status
sudo systemctl status tickerml-dashboard
sudo systemctl status tickerml-data-collector
sudo systemctl status tickerml-paper-trader

# View logs
sudo journalctl -u tickerml-dashboard -f
sudo journalctl -u tickerml-data-collector -f

# Restart services
sudo systemctl restart tickerml-*
```

### Health Monitoring
```bash
# Manual health check
sudo /usr/local/bin/tickerml-health-check

# Check API health
curl https://your-domain.com/api/health

# Monitor system resources
htop
df -h
free -h
```

### Log Management
```bash
# View application logs
tail -f /opt/tickerml/logs/*.log

# View nginx logs
sudo tail -f /var/log/nginx/tickerml_*.log

# View system logs
sudo tail -f /var/log/syslog
```

## 🔄 Updates & Maintenance

### Application Updates
```bash
# Stop services
sudo systemctl stop tickerml-*

# Backup data
sudo -u tickerml pg_dump tickerml > /opt/tickerml/backups/db_$(date +%Y%m%d).sql

# Update code
cd /opt/tickerml
sudo -u tickerml git pull
sudo -u tickerml venv/bin/pip install -r backend/requirements.txt

# Rebuild frontend
cd frontend
sudo -u tickerml npm install
sudo -u tickerml npm run build

# Restart services
sudo systemctl start tickerml-*
```

### Database Maintenance
```bash
# Backup database
sudo -u tickerml pg_dump tickerml > backup.sql

# Optimize database
sudo -u postgres psql -d tickerml -c "VACUUM ANALYZE;"

# Check database size
sudo -u postgres psql -d tickerml -c "SELECT pg_size_pretty(pg_database_size('tickerml'));"
```

## 🚨 Troubleshooting

### Common Issues

**Service won't start:**
```bash
# Check logs for errors
sudo journalctl -u tickerml-dashboard --no-pager

# Check configuration
sudo nginx -t
sudo -u tickerml python -m backend.app.main --check
```

**Database connection issues:**
```bash
# Test database connection
sudo -u tickerml psql -d tickerml -c "SELECT 1;"

# Check PostgreSQL status
sudo systemctl status postgresql
```

**High resource usage:**
```bash
# Monitor resource usage
htop
iotop
nethogs

# Check service resource usage
sudo systemctl show tickerml-dashboard --property=MemoryCurrent
```

**SSL certificate issues:**
```bash
# Check certificate status
sudo certbot certificates

# Renew certificate manually
sudo certbot renew --dry-run
```

### Log Locations
- Application logs: `/opt/tickerml/logs/`
- Nginx logs: `/var/log/nginx/`
- System logs: `/var/log/syslog`
- Service logs: `sudo journalctl -u service-name`

## 📞 Support

For deployment issues or questions:
1. Check the logs for specific error messages
2. Review the troubleshooting section above
3. Consult the main project documentation
4. Open an issue on the GitHub repository

## 🔒 Security Considerations

### Production Checklist
- [ ] Change all default passwords
- [ ] Configure firewall (UFW or iptables)
- [ ] Enable automatic security updates
- [ ] Set up log monitoring and alerting
- [ ] Configure backup retention policy
- [ ] Review and audit access logs regularly
- [ ] Keep API keys secure and rotate regularly
- [ ] Monitor for unusual trading patterns
- [ ] Set up database access restrictions
- [ ] Enable audit logging for sensitive operations