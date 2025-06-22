#!/usr/bin/env python3
"""
Direct test of 30-day quarantine rule - CRITICAL SAFETY FEATURE
"""

import pandas as pd
from datetime import datetime, timedelta
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from pc.offline_rl_trainer import OfflineRLTrainer, TrainingConfig
from pc.models.decision_transformer import DecisionTransformer, DecisionTransformerConfig

def test_quarantine_rule():
    """Test that 30-day quarantine is properly enforced"""
    
    print("🔒 Testing 30-Day Quarantine Rule (CRITICAL SAFETY)")
    print("=" * 60)
    
    # Create test data spanning recent dates
    now = datetime.now()
    
    # Create data that includes recent dates (should be filtered)
    test_data = pd.DataFrame({
        'timestamp': [
            (now - timedelta(days=45)).timestamp(),  # Should be included (> 30 days ago)
            (now - timedelta(days=35)).timestamp(),  # Should be included (> 30 days ago)
            (now - timedelta(days=25)).timestamp(),  # Should be EXCLUDED (< 30 days ago)
            (now - timedelta(days=15)).timestamp(),  # Should be EXCLUDED (< 30 days ago)
            (now - timedelta(days=5)).timestamp(),   # Should be EXCLUDED (< 30 days ago)
            (now - timedelta(days=1)).timestamp(),   # Should be EXCLUDED (< 30 days ago)
        ],
        'close': [100, 101, 102, 103, 104, 105],
        'symbol': ['BTCUSD'] * 6
    })
    
    print(f"Original data: {len(test_data)} records")
    print(f"Date range: {datetime.fromtimestamp(test_data['timestamp'].min())} to {datetime.fromtimestamp(test_data['timestamp'].max())}")
    
    # Initialize trainer
    config = TrainingConfig(quarantine_days=30)
    model_config = DecisionTransformerConfig()
    model = DecisionTransformer(model_config)
    trainer = OfflineRLTrainer(config, model)
    
    # Apply quarantine
    quarantined_data = trainer.apply_quarantine(test_data)
    
    print(f"After quarantine: {len(quarantined_data)} records")
    print(f"Date range: {datetime.fromtimestamp(quarantined_data['timestamp'].min())} to {datetime.fromtimestamp(quarantined_data['timestamp'].max())}")
    
    # Validation
    cutoff_date = now - timedelta(days=30)
    cutoff_timestamp = cutoff_date.timestamp()
    
    recent_data_count = len(test_data[test_data['timestamp'] >= cutoff_timestamp])
    quarantined_recent_count = len(quarantined_data[quarantined_data['timestamp'] >= cutoff_timestamp])
    
    print(f"\nValidation:")
    print(f"Recent data in original: {recent_data_count}")
    print(f"Recent data after quarantine: {quarantined_recent_count}")
    
    if quarantined_recent_count == 0:
        print("✅ QUARANTINE WORKING: No recent data in quarantined dataset")
        print("✅ CRITICAL SAFETY FEATURE CONFIRMED")
        return True
    else:
        print("❌ QUARANTINE FAILED: Recent data found in quarantined dataset")
        print("❌ CRITICAL SAFETY VIOLATION")
        return False

if __name__ == "__main__":
    success = test_quarantine_rule()
    if not success:
        print("\n🚨 CRITICAL ERROR: Quarantine rule is not working properly!")
        print("🚨 This could lead to forward-looking bias and overfitting!")
        sys.exit(1)
    else:
        print("\n🎉 Quarantine rule is working correctly - system is safe to use")