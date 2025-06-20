#!/usr/bin/env python3
"""
Enhanced training script that supports Decision Transformer
Integrates with offline RL trainer while maintaining compatibility
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import logging
import argparse
import os
import sys
from pathlib import Path
from datetime import datetime

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from pc.models.decision_transformer import DecisionTransformer, DecisionTransformerConfig
from pc.offline_rl_trainer import OfflineRLTrainer, TrainingConfig
from pc.enhanced_features import EnhancedFeatureEngineering

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DecisionTransformerTrainingPipeline:
    """Complete training pipeline for Decision Transformer"""
    
    def __init__(self, args):
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        # Check for BF16 support
        self.use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        if self.use_bf16:
            logger.info("BF16 support detected - will use bfloat16 for training")
        else:
            logger.warning("BF16 not supported - using FP32")
    
    def prepare_features(self):
        """Prepare enhanced features for Decision Transformer"""
        logger.info("Preparing enhanced features...")
        
        # Initialize feature engineering
        feature_eng = EnhancedFeatureEngineering()
        
        # Load and process data
        db_path = Path(self.args.data_path)
        if not db_path.exists():
            raise FileNotFoundError(f"Database not found: {db_path}")
        
        # Process features for each symbol
        all_features = []
        symbols = ['BTCUSD', 'ETHUSD']  # Add more symbols as needed
        
        for symbol in symbols:
            logger.info(f"Processing features for {symbol}")
            try:
                features = feature_eng.process_symbol(symbol, str(db_path))
                if features is not None:
                    all_features.append(features)
            except Exception as e:
                logger.error(f"Error processing {symbol}: {e}")
        
        if not all_features:
            raise ValueError("No features could be processed")
        
        # Combine features
        combined_features = np.vstack(all_features)
        logger.info(f"Combined features shape: {combined_features.shape}")
        
        # Save features
        features_path = Path(self.args.output_dir) / 'decision_transformer_features.npy'
        features_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(features_path, combined_features)
        logger.info(f"Features saved to {features_path}")
        
        return str(features_path), combined_features.shape[1]
    
    def create_model(self, feature_dim: int):
        """Create Decision Transformer model"""
        config = DecisionTransformerConfig(
            hidden_size=self.args.hidden_size,
            num_attention_heads=self.args.num_heads,
            num_hidden_layers=self.args.num_layers,
            intermediate_size=self.args.hidden_size * 4,
            max_position_embeddings=1024,
            context_length=self.args.context_length,
            dropout_prob=self.args.dropout,
            use_flash_attention=self.args.use_flash_attention,
            use_bf16=self.use_bf16,
            max_position_size=self.args.max_position_size
        )
        
        model = DecisionTransformer(config)
        
        # Load pre-trained checkpoint if specified
        if self.args.pretrained_checkpoint:
            logger.info(f"Loading pre-trained checkpoint from {self.args.pretrained_checkpoint}")
            checkpoint = torch.load(self.args.pretrained_checkpoint, map_location=self.device)
            model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        
        # Freeze backbone
        model.freeze_backbone()
        
        # Unfreeze last n layers for fine-tuning
        if self.args.unfreeze_layers > 0:
            model.unfreeze_last_n_layers(self.args.unfreeze_layers)
        
        model.to(self.device)
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"Total parameters: {total_params:,}")
        logger.info(f"Trainable parameters: {trainable_params:,}")
        
        return model
    
    def train(self):
        """Main training function"""
        logger.info("Starting Decision Transformer training pipeline")
        
        # Prepare features
        if self.args.features_path:
            features_path = self.args.features_path
            # Load features to get dimension
            features = np.load(features_path)
            feature_dim = features.shape[1]
        else:
            features_path, feature_dim = self.prepare_features()
        
        # Create model
        model = self.create_model(feature_dim)
        
        # Create training config
        training_config = TrainingConfig(
            quarantine_days=self.args.quarantine_days,
            min_history_days=self.args.min_history_days,
            sequence_length=self.args.context_length,
            batch_size=self.args.batch_size,
            learning_rate=self.args.learning_rate,
            weight_decay=self.args.weight_decay,
            num_epochs=self.args.epochs,
            warmup_steps=self.args.warmup_steps,
            gradient_clip=self.args.gradient_clip,
            use_bf16=self.use_bf16,
            checkpoint_dir=self.args.output_dir,
            max_position_size=self.args.max_position_size,
            target_sharpe=self.args.target_sharpe
        )
        
        # Create trainer
        trainer = OfflineRLTrainer(training_config, model)
        
        # Train
        db_path = str(Path(self.args.data_path))
        train_metrics, val_metrics = trainer.train(db_path, features_path)
        
        # Save final model
        final_model_path = Path(self.args.output_dir) / 'decision_transformer_final.pt'
        model.save_checkpoint(str(final_model_path))
        logger.info(f"Final model saved to {final_model_path}")
        
        # Save training history
        history = {
            'train_metrics': train_metrics,
            'val_metrics': val_metrics,
            'config': vars(self.args),
            'feature_dim': feature_dim
        }
        
        history_path = Path(self.args.output_dir) / 'training_history.json'
        with open(history_path, 'w') as f:
            import json
            json.dump(history, f, indent=2, default=str)
        logger.info(f"Training history saved to {history_path}")
        
        logger.info("Training completed successfully!")

def main():
    parser = argparse.ArgumentParser(description='Train Decision Transformer for crypto trading')
    
    # Data arguments
    parser.add_argument('--data_path', type=str, default='data/db/crypto_data.db',
                        help='Path to database')
    parser.add_argument('--features_path', type=str, default=None,
                        help='Path to pre-computed features (optional)')
    parser.add_argument('--output_dir', type=str, default='models/checkpoints/decision_transformer',
                        help='Output directory for checkpoints')
    
    # Model arguments
    parser.add_argument('--hidden_size', type=int, default=512,
                        help='Hidden size of transformer')
    parser.add_argument('--num_heads', type=int, default=8,
                        help='Number of attention heads')
    parser.add_argument('--num_layers', type=int, default=6,
                        help='Number of transformer layers')
    parser.add_argument('--context_length', type=int, default=60,
                        help='Context length for sequences')
    parser.add_argument('--dropout', type=float, default=0.1,
                        help='Dropout probability')
    parser.add_argument('--use_flash_attention', action='store_true',
                        help='Use Flash Attention for RTX 4090')
    parser.add_argument('--pretrained_checkpoint', type=str, default=None,
                        help='Path to pre-trained checkpoint')
    parser.add_argument('--unfreeze_layers', type=int, default=2,
                        help='Number of layers to unfreeze for fine-tuning')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                        help='Weight decay')
    parser.add_argument('--warmup_steps', type=int, default=1000,
                        help='Warmup steps')
    parser.add_argument('--gradient_clip', type=float, default=1.0,
                        help='Gradient clipping value')
    
    # Risk management arguments
    parser.add_argument('--quarantine_days', type=int, default=30,
                        help='Days to quarantine recent data')
    parser.add_argument('--min_history_days', type=int, default=90,
                        help='Minimum days of historical data required')
    parser.add_argument('--max_position_size', type=float, default=0.25,
                        help='Maximum position size as fraction of portfolio')
    parser.add_argument('--target_sharpe', type=float, default=2.0,
                        help='Target Sharpe ratio for reward shaping')
    
    args = parser.parse_args()
    
    # Create output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    # Initialize pipeline
    pipeline = DecisionTransformerTrainingPipeline(args)
    
    # Train
    pipeline.train()

if __name__ == "__main__":
    main()