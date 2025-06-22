#!/usr/bin/env python3
"""
Direct test of frozen backbone functionality - CRITICAL SAFETY FEATURE
"""

import torch
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from pc.models.decision_transformer import DecisionTransformer, DecisionTransformerConfig

def test_frozen_backbone():
    """Test that backbone freezing is properly implemented"""
    
    print("🧊 Testing Frozen Backbone Functionality (CRITICAL SAFETY)")
    print("=" * 65)
    
    # Create model with frozen backbone enabled
    config = DecisionTransformerConfig(
        d_model=128,
        n_heads=4,
        n_layers=6,
        freeze_backbone=True,
        trainable_layers=2
    )
    
    model = DecisionTransformer(config)
    
    print(f"Model created with {config.n_layers} layers")
    print(f"Trainable layers: {config.trainable_layers}")
    print(f"Frozen layers: {config.n_layers - config.trainable_layers}")
    
    # Check which parameters are trainable
    trainable_params = []
    frozen_params = []
    
    for name, param in model.named_parameters():
        if param.requires_grad:
            trainable_params.append(name)
        else:
            frozen_params.append(name)
    
    print(f"\n📊 Parameter Analysis:")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters())}")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    print(f"Frozen parameters: {sum(p.numel() for p in model.parameters() if not p.requires_grad)}")
    
    print(f"\n🔒 Frozen Parameters ({len(frozen_params)}):")
    for param_name in frozen_params[:10]:  # Show first 10
        print(f"  - {param_name}")
    if len(frozen_params) > 10:
        print(f"  ... and {len(frozen_params) - 10} more")
    
    print(f"\n🔓 Trainable Parameters ({len(trainable_params)}):")
    for param_name in trainable_params:
        print(f"  - {param_name}")
    
    # Validation checks
    print(f"\n✅ Validation Checks:")
    
    # Check 1: Embeddings should be frozen
    embedding_frozen = all(
        not param.requires_grad for name, param in model.named_parameters() 
        if 'embedding' in name
    )
    print(f"Embeddings frozen: {'✅' if embedding_frozen else '❌'}")
    
    # Check 2: First transformer blocks should be frozen
    early_blocks_frozen = all(
        not param.requires_grad for name, param in model.named_parameters()
        if 'transformer_blocks.0.' in name or 'transformer_blocks.1.' in name or 
           'transformer_blocks.2.' in name or 'transformer_blocks.3.' in name
    )
    print(f"Early transformer blocks frozen: {'✅' if early_blocks_frozen else '❌'}")
    
    # Check 3: Last transformer blocks should be trainable
    late_blocks_trainable = any(
        param.requires_grad for name, param in model.named_parameters()
        if f'transformer_blocks.{config.n_layers-1}.' in name or 
           f'transformer_blocks.{config.n_layers-2}.' in name
    )
    print(f"Late transformer blocks trainable: {'✅' if late_blocks_trainable else '❌'}")
    
    # Check 4: Prediction heads should be trainable
    heads_trainable = all(
        param.requires_grad for name, param in model.named_parameters()
        if '_head.' in name
    )
    print(f"Prediction heads trainable: {'✅' if heads_trainable else '❌'}")
    
    # Overall validation
    all_checks_passed = (embedding_frozen and early_blocks_frozen and 
                        late_blocks_trainable and heads_trainable)
    
    if all_checks_passed:
        print("\n🎉 FROZEN BACKBONE WORKING CORRECTLY")
        print("✅ This prevents catastrophic forgetting")
        print("✅ Only task-specific heads and last layers are trainable")
        return True
    else:
        print("\n🚨 FROZEN BACKBONE NOT WORKING PROPERLY")
        print("❌ Risk of catastrophic forgetting")
        return False

if __name__ == "__main__":
    success = test_frozen_backbone()
    if not success:
        print("\n🚨 CRITICAL ERROR: Frozen backbone is not working properly!")
        print("🚨 This could lead to catastrophic forgetting!")
        sys.exit(1)
    else:
        print("\n🎉 Frozen backbone is working correctly - system is safe to use")