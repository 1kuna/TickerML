# TickerML Dashboard - Production Docker Image
# Multi-stage build for optimized production image

# Stage 1: Frontend build
FROM node:18-alpine AS frontend-builder

WORKDIR /app/frontend

# Copy package files
COPY frontend/package*.json ./
RUN npm ci --only=production

# Copy frontend source
COPY frontend/ ./

# Build frontend
RUN npm run build

# Stage 2: Python backend
FROM python:3.11-slim AS backend

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV ENVIRONMENT=production

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Create application user
RUN groupadd -r tickerml && useradd -r -g tickerml tickerml

# Create application directory
WORKDIR /app

# Copy requirements and install Python dependencies
COPY backend/requirements.txt raspberry_pi/requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY backend/ ./backend/
COPY raspberry_pi/ ./raspberry_pi/
COPY pc/ ./pc/
COPY config/ ./config/
COPY scripts/ ./scripts/

# Copy frontend build from previous stage
COPY --from=frontend-builder /app/frontend/build ./frontend/build

# Create necessary directories
RUN mkdir -p /app/logs /app/data /app/models \
    && chown -R tickerml:tickerml /app

# Switch to non-root user
USER tickerml

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/api/health || exit 1

# Start command
CMD ["python", "-m", "uvicorn", "backend.app.main:app", "--host", "0.0.0.0", "--port", "8000"]