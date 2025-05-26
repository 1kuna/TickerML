#!/bin/bash

# Crypto Time-Series Transformer Pipeline Setup Script

set -e  # Exit on any error

echo "🚀 Setting up Crypto Time-Series Transformer Pipeline..."

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Detect platform
PLATFORM=""
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    if grep -q "Raspberry Pi" /proc/cpuinfo 2>/dev/null; then
        PLATFORM="raspberry_pi"
    else
        PLATFORM="pc"
    fi
elif [[ "$OSTYPE" == "darwin"* ]]; then
    PLATFORM="pc"
else
    print_error "Unsupported platform: $OSTYPE"
    exit 1
fi

print_status "Detected platform: $PLATFORM"

# Check Python version
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}' | cut -d. -f1,2)
REQUIRED_VERSION="3.8"

if [ "$(printf '%s\n' "$REQUIRED_VERSION" "$PYTHON_VERSION" | sort -V | head -n1)" != "$REQUIRED_VERSION" ]; then
    print_error "Python 3.8+ required. Found: $PYTHON_VERSION"
    exit 1
fi

print_success "Python version check passed: $PYTHON_VERSION"

# Create virtual environment
print_status "Creating virtual environment..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
    print_success "Virtual environment created"
else
    print_warning "Virtual environment already exists"
fi

# Activate virtual environment
print_status "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
print_status "Upgrading pip..."
pip install --upgrade pip

# Install platform-specific requirements
if [ "$PLATFORM" == "raspberry_pi" ]; then
    print_status "Installing Raspberry Pi requirements..."
    pip install -r raspberry_pi/requirements.txt
    
    # Install additional Pi-specific packages
    print_status "Installing additional Raspberry Pi packages..."
    sudo apt-get update
    sudo apt-get install -y sqlite3 cron
    
elif [ "$PLATFORM" == "pc" ]; then
    print_status "Installing PC requirements..."
    pip install -r pc/requirements.txt
    
    # Check for CUDA availability
    if command -v nvidia-smi &> /dev/null; then
        print_status "NVIDIA GPU detected, installing CUDA-enabled PyTorch..."
        pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
    else
        print_warning "No NVIDIA GPU detected, using CPU-only PyTorch"
    fi
fi

# Create necessary directories
print_status "Creating directory structure..."
mkdir -p data/dumps
mkdir -p data/db
mkdir -p data/features
mkdir -p models/checkpoints
mkdir -p models/onnx
mkdir -p logs
mkdir -p raspberry_pi/templates

print_success "Directory structure created"

# Set up logging
print_status "Setting up logging..."
touch logs/crypto_pipeline.log
chmod 644 logs/crypto_pipeline.log

# Make scripts executable
print_status "Making scripts executable..."
chmod +x raspberry_pi/*.py
chmod +x pc/*.py
chmod +x scripts/*.sh

# Platform-specific setup
if [ "$PLATFORM" == "raspberry_pi" ]; then
    print_status "Setting up Raspberry Pi specific configurations..."
    
    # Create systemd service for dashboard (optional)
    print_status "Creating systemd service template..."
    cat > /tmp/crypto-dashboard.service << EOF
[Unit]
Description=Crypto Price Prediction Dashboard
After=network.target

[Service]
Type=simple
User=pi
WorkingDirectory=$(pwd)
Environment=PATH=$(pwd)/venv/bin
ExecStart=$(pwd)/venv/bin/python $(pwd)/raspberry_pi/dashboard.py
Restart=always

[Install]
WantedBy=multi-user.target
EOF
    
    print_status "Systemd service template created at /tmp/crypto-dashboard.service"
    print_warning "To install: sudo cp /tmp/crypto-dashboard.service /etc/systemd/system/"
    print_warning "Then run: sudo systemctl enable crypto-dashboard && sudo systemctl start crypto-dashboard"
    
    # Cron setup instructions
    print_status "Cron setup instructions:"
    echo "Add these lines to your crontab (crontab -e):"
    echo "# Crypto data harvester - every minute"
    echo "* * * * * cd $(pwd) && ./venv/bin/python raspberry_pi/harvest.py >> logs/harvest.log 2>&1"
    echo ""
    echo "# Daily ETL export - midnight"
    echo "0 0 * * * cd $(pwd) && ./venv/bin/python raspberry_pi/export_etl.py >> logs/etl.log 2>&1"
    echo ""
    echo "# Inference service - every 5 minutes"
    echo "*/5 * * * * cd $(pwd) && ./venv/bin/python raspberry_pi/infer.py >> logs/infer.log 2>&1"
    
elif [ "$PLATFORM" == "pc" ]; then
    print_status "Setting up PC specific configurations..."
    
    # Setup Ollama for Gemma 3 sentiment analysis
    print_status "Setting up Ollama for sentiment analysis..."
    print_warning "Please ensure Ollama is installed on your system:"
    print_warning "Visit https://ollama.ai for installation instructions"
    print_warning "After installation, run: ollama pull gemma3:4b"
    
    # Create Jupyter notebook for analysis
    print_status "Creating analysis notebook..."
    cat > analysis.ipynb << 'EOF'
{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Crypto Time-Series Analysis\n",
    "\n",
    "This notebook provides tools for analyzing the crypto prediction pipeline."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "source": [
    "import pandas as pd\n",
    "import numpy as np\n",
    "import matplotlib.pyplot as plt\n",
    "import seaborn as sns\n",
    "from pathlib import Path\n",
    "\n",
    "# Load data\n",
    "data_path = Path('data/features')\n",
    "print(f\"Available feature files: {list(data_path.glob('*.csv'))}\")"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
EOF
fi

# Test installations
print_status "Testing installations..."

# Test basic imports
python3 -c "
import pandas as pd
import numpy as np
import requests
print('✓ Basic packages imported successfully')
"

if [ "$PLATFORM" == "pc" ]; then
    python3 -c "
import torch
import onnx
import onnxruntime
print('✓ ML packages imported successfully')
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
"
fi

# Create sample configuration
print_status "Creating sample configuration..."
cp config/config.yaml config/config.yaml.sample

print_success "Setup completed successfully!"

echo ""
echo "📋 Next Steps:"
echo "1. Update config/config.yaml with your API keys"
echo "2. For NewsAPI: Get a free key from https://newsapi.org/"

if [ "$PLATFORM" == "raspberry_pi" ]; then
    echo "3. Set up cron jobs (see instructions above)"
    echo "4. Start data collection: python raspberry_pi/harvest.py"
    echo "5. Start dashboard: python raspberry_pi/dashboard.py"
elif [ "$PLATFORM" == "pc" ]; then
    echo "3. Run feature engineering: python pc/features.py"
    echo "4. Train model: python pc/train.py"
    echo "5. Export to ONNX: python pc/export_quantize.py"
fi

echo ""
echo "🔧 Useful Commands:"
echo "- Activate environment: source venv/bin/activate"
echo "- View logs: tail -f logs/crypto_pipeline.log"
echo "- Check database: sqlite3 data/db/crypto_data.db"

if [ "$PLATFORM" == "pc" ]; then
    echo "- Start Jupyter: jupyter notebook"
fi

echo ""
print_success "Happy trading! 📈" 