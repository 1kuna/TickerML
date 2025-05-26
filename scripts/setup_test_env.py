#!/usr/bin/env python3
"""
Quick setup script for testing the data collection pipeline
Creates necessary directories and installs basic dependencies
"""

import subprocess
import sys
from pathlib import Path

def run_command(command, description):
    """Run a command and handle errors"""
    print(f"🔧 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed: {e.stderr}")
        return False

def main():
    """Setup test environment"""
    print("🚀 Setting up TickerML Data Collection Test Environment")
    print("=" * 60)
    
    project_root = Path(__file__).parent
    
    # Create directory structure
    directories = [
        "data/db",
        "data/dumps", 
        "data/features",
        "logs",
        "models/checkpoints",
        "models/onnx",
        "raspberry_pi/templates"
    ]
    
    print("📁 Creating directory structure...")
    for directory in directories:
        dir_path = project_root / directory
        dir_path.mkdir(parents=True, exist_ok=True)
        print(f"   ✅ {directory}")
    
    # Check Python version
    print(f"\n🐍 Python version: {sys.version}")
    if sys.version_info < (3, 8):
        print("⚠️  Warning: Python 3.8+ recommended")
    
    # Install basic dependencies
    basic_deps = [
        "requests",
        "pandas", 
        "sqlite3"  # Usually built-in
    ]
    
    print("\n📦 Checking basic dependencies...")
    missing_deps = []
    
    for dep in basic_deps:
        try:
            if dep == "sqlite3":
                import sqlite3
            else:
                __import__(dep)
            print(f"   ✅ {dep}")
        except ImportError:
            print(f"   ❌ {dep} - missing")
            missing_deps.append(dep)
    
    if missing_deps:
        print(f"\n🔧 Installing missing dependencies: {', '.join(missing_deps)}")
        deps_to_install = [dep for dep in missing_deps if dep != "sqlite3"]
        if deps_to_install:
            install_cmd = f"{sys.executable} -m pip install {' '.join(deps_to_install)}"
            run_command(install_cmd, "Installing dependencies")
    
    # Create basic config if it doesn't exist
    config_path = project_root / "config" / "config.yaml"
    if not config_path.exists():
        print("\n⚙️  Creating basic config...")
        config_path.parent.mkdir(exist_ok=True)
        
        basic_config = """# Basic configuration for testing
data:
  symbols: ["BTCUSDT", "ETHUSDT"]
  binance_api_base: "https://api.binance.com/api/v3"
  interval: "1m"

database:
  type: "sqlite"
  path: "data/db/crypto_data.db"

logging:
  level: "INFO"
  format: "%(asctime)s - %(levelname)s - %(message)s"
"""
        
        with open(config_path, 'w') as f:
            f.write(basic_config)
        print("   ✅ config/config.yaml created")
    
    # Make test script executable
    test_script = project_root / "test_data_collection.py"
    if test_script.exists():
        run_command(f"chmod +x {test_script}", "Making test script executable")
    
    print("\n🎉 Setup completed!")
    print("\n📋 Next steps:")
    print("1. Run the test script: python test_data_collection.py")
    print("2. Or test individual components:")
    print("   - python raspberry_pi/harvest.py")
    print("   - python raspberry_pi/export_etl.py")
    print("   - python raspberry_pi/dashboard.py")
    print("\n💡 For full setup, run: bash scripts/setup.sh")

if __name__ == "__main__":
    main() 