#!/bin/bash
set -e

# TickerML Production Deployment Script
# This script deploys the TickerML dashboard to a production server

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
DEPLOY_USER="tickerml"
DEPLOY_PATH="/opt/tickerml"
DOMAIN="${DOMAIN:-your-domain.com}"
EMAIL="${EMAIL:-admin@your-domain.com}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

log() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')] $1${NC}"
}

warn() {
    echo -e "${YELLOW}[WARNING] $1${NC}"
}

error() {
    echo -e "${RED}[ERROR] $1${NC}"
    exit 1
}

success() {
    echo -e "${GREEN}[SUCCESS] $1${NC}"
}

# Check if running as root
if [[ $EUID -eq 0 ]]; then
    error "This script should not be run as root. Run as a user with sudo privileges."
fi

# Check if required environment variables are set
if [[ -z "$DOMAIN" ]]; then
    warn "DOMAIN not set, using default: your-domain.com"
fi

log "Starting TickerML deployment..."
log "Domain: $DOMAIN"
log "Deploy path: $DEPLOY_PATH"
log "Deploy user: $DEPLOY_USER"

# 1. System dependencies
log "Installing system dependencies..."
sudo apt-get update
sudo apt-get install -y \
    nginx \
    postgresql \
    redis-server \
    python3 \
    python3-pip \
    python3-venv \
    certbot \
    python3-certbot-nginx \
    nodejs \
    npm \
    git \
    htop \
    curl \
    wget \
    unzip

# 2. Create deploy user if it doesn't exist
if ! id "$DEPLOY_USER" &>/dev/null; then
    log "Creating deploy user: $DEPLOY_USER"
    sudo useradd -r -s /bin/bash -d "$DEPLOY_PATH" -m "$DEPLOY_USER"
    sudo usermod -aG sudo "$DEPLOY_USER"
else
    log "Deploy user $DEPLOY_USER already exists"
fi

# 3. Create directory structure
log "Creating directory structure..."
sudo mkdir -p "$DEPLOY_PATH"/{logs,data,config,backups}
sudo chown -R "$DEPLOY_USER:$DEPLOY_USER" "$DEPLOY_PATH"

# 4. Copy application files
log "Copying application files..."
sudo -u "$DEPLOY_USER" cp -r "$PROJECT_ROOT"/* "$DEPLOY_PATH/"

# 5. Setup Python environment
log "Setting up Python virtual environment..."
cd "$DEPLOY_PATH"
sudo -u "$DEPLOY_USER" python3 -m venv venv
sudo -u "$DEPLOY_USER" venv/bin/pip install --upgrade pip
sudo -u "$DEPLOY_USER" venv/bin/pip install -r backend/requirements.txt
sudo -u "$DEPLOY_USER" venv/bin/pip install -r raspberry_pi/requirements.txt

# 6. Build frontend
log "Building frontend..."
cd "$DEPLOY_PATH/frontend"
sudo -u "$DEPLOY_USER" npm install
sudo -u "$DEPLOY_USER" npm run build

# 7. Setup database
log "Setting up PostgreSQL database..."
sudo -u postgres createdb tickerml 2>/dev/null || log "Database already exists"
sudo -u postgres createuser "$DEPLOY_USER" 2>/dev/null || log "Database user already exists"
sudo -u postgres psql -c "ALTER USER $DEPLOY_USER WITH PASSWORD 'tickerml_password';" 
sudo -u postgres psql -c "GRANT ALL PRIVILEGES ON DATABASE tickerml TO $DEPLOY_USER;"

# Enable TimescaleDB extension
sudo -u postgres psql -d tickerml -c "CREATE EXTENSION IF NOT EXISTS timescaledb;"

# 8. Configure environment
log "Setting up environment configuration..."
if [[ ! -f "$DEPLOY_PATH/.env" ]]; then
    sudo -u "$DEPLOY_USER" tee "$DEPLOY_PATH/.env" > /dev/null <<EOF
# TickerML Production Environment
ENVIRONMENT=production
DEBUG=false

# Database
DATABASE_URL=postgresql://$DEPLOY_USER:tickerml_password@localhost:5432/tickerml

# Redis
REDIS_URL=redis://localhost:6379

# Dashboard
DASHBOARD_SECRET_KEY=$(openssl rand -hex 32)
DASHBOARD_HOST=0.0.0.0
DASHBOARD_PORT=8000

# Exchanges API keys (configure these manually)
BINANCE_API_KEY=
BINANCE_SECRET_KEY=
COINBASE_API_KEY=
COINBASE_SECRET_KEY=
KRAKEN_API_KEY=
KRAKEN_SECRET_KEY=
KUCOIN_API_KEY=
KUCOIN_SECRET_KEY=

# Logging
LOG_LEVEL=INFO
LOG_FILE=/opt/tickerml/logs/app.log
EOF
    success "Environment file created. Please configure API keys manually."
else
    log "Environment file already exists"
fi

# 9. Install systemd services
log "Installing systemd services..."
sudo cp "$DEPLOY_PATH/deployment/systemd/"*.service /etc/systemd/system/
sudo systemctl daemon-reload

# Enable services
sudo systemctl enable tickerml-dashboard
sudo systemctl enable tickerml-data-collector
sudo systemctl enable tickerml-paper-trader

# 10. Configure Nginx
log "Configuring Nginx..."
sudo cp "$DEPLOY_PATH/deployment/nginx/tickerml.conf" /etc/nginx/sites-available/
sudo sed -i "s/your-domain.com/$DOMAIN/g" /etc/nginx/sites-available/tickerml.conf
sudo ln -sf /etc/nginx/sites-available/tickerml.conf /etc/nginx/sites-enabled/
sudo rm -f /etc/nginx/sites-enabled/default

# Test Nginx configuration
sudo nginx -t

# 11. Setup SSL certificate
log "Setting up SSL certificate..."
sudo mkdir -p /var/www/letsencrypt
if ! sudo certbot certificates | grep -q "$DOMAIN"; then
    log "Obtaining SSL certificate for $DOMAIN..."
    sudo certbot certonly --webroot -w /var/www/letsencrypt -d "$DOMAIN" -d "www.$DOMAIN" --email "$EMAIL" --agree-tos --non-interactive
else
    log "SSL certificate for $DOMAIN already exists"
fi

# 12. Setup automatic certificate renewal
log "Setting up automatic SSL renewal..."
sudo tee /etc/cron.d/certbot > /dev/null <<EOF
0 12 * * * root certbot renew --quiet && systemctl reload nginx
EOF

# 13. Configure log rotation
log "Setting up log rotation..."
sudo tee /etc/logrotate.d/tickerml > /dev/null <<EOF
/opt/tickerml/logs/*.log {
    daily
    missingok
    rotate 52
    compress
    delaycompress
    notifempty
    create 644 $DEPLOY_USER $DEPLOY_USER
    postrotate
        systemctl reload tickerml-dashboard
        systemctl reload tickerml-data-collector
        systemctl reload tickerml-paper-trader
    endscript
}
EOF

# 14. Setup monitoring scripts
log "Setting up monitoring scripts..."
sudo tee /usr/local/bin/tickerml-health-check > /dev/null <<'EOF'
#!/bin/bash
# TickerML Health Check Script

DEPLOY_PATH="/opt/tickerml"
LOG_FILE="/var/log/tickerml-health.log"

log() {
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

# Check if services are running
services=("tickerml-dashboard" "tickerml-data-collector" "tickerml-paper-trader")
for service in "${services[@]}"; do
    if ! systemctl is-active --quiet "$service"; then
        log "ERROR: $service is not running"
        systemctl restart "$service"
        log "Attempted to restart $service"
    else
        log "OK: $service is running"
    fi
done

# Check disk space
DISK_USAGE=$(df "$DEPLOY_PATH" | awk 'NR==2 {print $5}' | sed 's/%//')
if [[ $DISK_USAGE -gt 80 ]]; then
    log "WARNING: Disk usage is ${DISK_USAGE}%"
fi

# Check database connectivity
if ! sudo -u tickerml psql -d tickerml -c "SELECT 1;" >/dev/null 2>&1; then
    log "ERROR: Cannot connect to database"
fi

# Check Redis connectivity
if ! redis-cli ping >/dev/null 2>&1; then
    log "ERROR: Cannot connect to Redis"
fi

log "Health check completed"
EOF

sudo chmod +x /usr/local/bin/tickerml-health-check

# Setup health check cron job
sudo tee /etc/cron.d/tickerml-health > /dev/null <<EOF
*/5 * * * * root /usr/local/bin/tickerml-health-check
EOF

# 15. Start services
log "Starting services..."
sudo systemctl start redis
sudo systemctl start postgresql
sudo systemctl start nginx

# Wait a moment for services to start
sleep 5

sudo systemctl start tickerml-dashboard
sudo systemctl start tickerml-data-collector
sudo systemctl start tickerml-paper-trader

# 16. Verify deployment
log "Verifying deployment..."
sleep 10

# Check service status
for service in tickerml-dashboard tickerml-data-collector tickerml-paper-trader; do
    if systemctl is-active --quiet "$service"; then
        success "$service is running"
    else
        error "$service failed to start"
    fi
done

# Check API health
if curl -f -s "http://localhost:8000/api/health" >/dev/null; then
    success "API health check passed"
else
    warn "API health check failed"
fi

# Check web interface
if curl -f -s "https://$DOMAIN" >/dev/null; then
    success "Web interface is accessible"
else
    warn "Web interface check failed"
fi

# 17. Display final information
log "Deployment completed successfully!"
success "TickerML Dashboard is now running at: https://$DOMAIN"
success "API documentation available at: https://$DOMAIN/api/docs"

echo ""
echo "Next steps:"
echo "1. Configure exchange API keys in $DEPLOY_PATH/.env"
echo "2. Review logs: sudo journalctl -u tickerml-dashboard -f"
echo "3. Monitor health: sudo tail -f /var/log/tickerml-health.log"
echo "4. Access dashboard: https://$DOMAIN"
echo ""
echo "Useful commands:"
echo "- Restart all services: sudo systemctl restart tickerml-*"
echo "- Check service status: sudo systemctl status tickerml-dashboard"
echo "- View logs: sudo journalctl -u tickerml-dashboard"
echo "- Run health check: sudo /usr/local/bin/tickerml-health-check"

log "Deployment script completed successfully!"