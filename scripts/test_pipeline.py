#!/usr/bin/env python3
"""
Test script for the crypto pipeline
Verifies that all components work correctly
"""

import sys
import subprocess
import sqlite3
import pandas as pd
import numpy as np
from pathlib import Path
import logging
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

def test_imports():
    """Test that all required packages can be imported"""
    logger.info("Testing imports...")
    
    try:
        import pandas as pd
        import numpy as np
        import requests
        import sqlite3
        logger.info("✓ Basic packages imported successfully")
        
        # Test platform-specific imports
        try:
            import torch
            import onnx
            import onnxruntime
            logger.info("✓ ML packages imported successfully (PC mode)")
        except ImportError:
            logger.info("ℹ ML packages not available (Raspberry Pi mode)")
        
        try:
            import flask
            logger.info("✓ Flask imported successfully")
        except ImportError:
            logger.warning("⚠ Flask not available")
        
        return True
        
    except ImportError as e:
        logger.error(f"✗ Import failed: {e}")
        return False

def test_database_creation():
    """Test database creation and basic operations"""
    logger.info("Testing database operations...")
    
    try:
        # Import harvest module
        sys.path.append(str(project_root / "raspberry_pi"))
        from harvest import init_database, store_data
        
        # Initialize database
        init_database()
        
        # Test data insertion
        test_data = {
            'timestamp': int(time.time() * 1000),
            'open': 50000.0,
            'high': 50100.0,
            'low': 49900.0,
            'close': 50050.0,
            'volume': 1000.0
        }
        
        store_data("TESTUSDT", test_data)
        logger.info("✓ Database operations successful")
        return True
        
    except Exception as e:
        logger.error(f"✗ Database test failed: {e}")
        return False

def test_api_connection():
    """Test connection to Binance API"""
    logger.info("Testing API connection...")
    
    try:
        import requests
        
        url = "https://api.binance.com/api/v3/ping"
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        
        logger.info("✓ Binance API connection successful")
        return True
        
    except Exception as e:
        logger.error(f"✗ API connection failed: {e}")
        return False

def test_feature_engineering():
    """Test feature engineering pipeline"""
    logger.info("Testing feature engineering...")
    
    try:
        # Create sample data
        dates = pd.date_range('2023-01-01', periods=1000, freq='1min')
        np.random.seed(42)
        
        sample_data = pd.DataFrame({
            'timestamp': [int(d.timestamp() * 1000) for d in dates],
            'datetime': dates,
            'open': 50000 + np.random.randn(1000) * 100,
            'high': 50000 + np.random.randn(1000) * 100 + 50,
            'low': 50000 + np.random.randn(1000) * 100 - 50,
            'close': 50000 + np.random.randn(1000) * 100,
            'volume': np.random.uniform(100, 1000, 1000)
        })
        
        # Test technical indicators
        sys.path.append(str(project_root / "pc"))
        from features import compute_technical_indicators
        
        result = compute_technical_indicators(sample_data)
        
        if len(result.columns) > len(sample_data.columns):
            logger.info("✓ Feature engineering successful")
            return True
        else:
            logger.error("✗ No features were added")
            return False
        
    except Exception as e:
        logger.error(f"✗ Feature engineering test failed: {e}")
        return False

def test_model_architecture():
    """Test model architecture (PC only)"""
    logger.info("Testing model architecture...")
    
    try:
        import torch
        sys.path.append(str(project_root / "pc"))
        from train import TimeSeriesTransformer
        
        # Create model
        model = TimeSeriesTransformer(feature_dim=10)
        
        # Test forward pass
        dummy_input = torch.randn(1, 60, 10)
        reg_output, cls_output = model(dummy_input)
        
        if reg_output.shape == (1, 3) and cls_output.shape == (1, 2):
            logger.info("✓ Model architecture test successful")
            return True
        else:
            logger.error(f"✗ Unexpected output shapes: {reg_output.shape}, {cls_output.shape}")
            return False
        
    except ImportError:
        logger.info("ℹ Skipping model test (PyTorch not available)")
        return True
    except Exception as e:
        logger.error(f"✗ Model test failed: {e}")
        return False

def test_dashboard_components():
    """Test dashboard components"""
    logger.info("Testing dashboard components...")
    
    try:
        sys.path.append(str(project_root / "raspberry_pi"))
        from dashboard import app
        
        # Test app creation
        with app.test_client() as client:
            # Test main route
            response = client.get('/')
            if response.status_code == 200:
                logger.info("✓ Dashboard test successful")
                return True
            else:
                logger.error(f"✗ Dashboard returned status {response.status_code}")
                return False
        
    except Exception as e:
        logger.error(f"✗ Dashboard test failed: {e}")
        return False

def test_file_structure():
    """Test that all required files exist"""
    logger.info("Testing file structure...")
    
    required_files = [
        "raspberry_pi/harvest.py",
        "raspberry_pi/export_etl.py",
        "raspberry_pi/infer.py",
        "raspberry_pi/dashboard.py",
        "raspberry_pi/requirements.txt",
        "pc/features.py",
        "pc/train.py",
        "pc/export_quantize.py",
        "pc/requirements.txt",
        "config/config.yaml",
        "scripts/setup.sh",
        "README.md"
    ]
    
    missing_files = []
    for file_path in required_files:
        if not (project_root / file_path).exists():
            missing_files.append(file_path)
    
    if missing_files:
        logger.error(f"✗ Missing files: {missing_files}")
        return False
    else:
        logger.info("✓ All required files present")
        return True

def run_all_tests():
    """Run all tests"""
    logger.info("🧪 Starting pipeline tests...")
    
    tests = [
        ("File Structure", test_file_structure),
        ("Imports", test_imports),
        ("Database", test_database_creation),
        ("API Connection", test_api_connection),
        ("Feature Engineering", test_feature_engineering),
        ("Model Architecture", test_model_architecture),
        ("Dashboard", test_dashboard_components)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        logger.info(f"\n--- Running {test_name} Test ---")
        try:
            results[test_name] = test_func()
        except Exception as e:
            logger.error(f"✗ {test_name} test crashed: {e}")
            results[test_name] = False
    
    # Summary
    logger.info("\n" + "="*50)
    logger.info("TEST SUMMARY")
    logger.info("="*50)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results.items():
        status = "✓ PASS" if result else "✗ FAIL"
        logger.info(f"{test_name:20} {status}")
        if result:
            passed += 1
    
    logger.info(f"\nPassed: {passed}/{total}")
    
    if passed == total:
        logger.info("🎉 All tests passed! Pipeline is ready.")
        return True
    else:
        logger.warning(f"⚠ {total - passed} tests failed. Check the logs above.")
        return False

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1) 