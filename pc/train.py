#!/usr/bin/env python3
"""
Model training script for PC
Trains the TimeSeriesTransformer model
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import logging
from datetime import datetime
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, accuracy_score, roc_auc_score
import json
import math

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration
FEATURES_PATH = Path(__file__).parent.parent / "data" / "features"
MODELS_PATH = Path(__file__).parent.parent / "models" / "checkpoints"
SEQUENCE_LENGTH = 60  # 60 minutes of history
BATCH_SIZE = 32
EPOCHS = 100
LEARNING_RATE = 0.001
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class PositionalEncoding(nn.Module):
    """Positional encoding for transformer"""
    
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        return x + self.pe[:x.size(0), :]

class TimeSeriesTransformer(nn.Module):
    """Time Series Transformer for crypto price prediction"""
    
    def __init__(self, feature_dim, d_model=128, n_heads=4, n_layers=6, dropout=0.1):
        super(TimeSeriesTransformer, self).__init__()
        
        self.d_model = d_model
        self.feature_dim = feature_dim
        
        # Input projection
        self.input_proj = nn.Linear(feature_dim, d_model)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation='relu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        # Output heads
        self.regression_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 3)  # 5min, 10min, 30min predictions
        )
        
        self.classification_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 2)  # up/down classification
        )
        
        # Initialize weights
        self.init_weights()
    
    def init_weights(self):
        """Initialize model weights"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, x):
        # x shape: (batch_size, sequence_length, feature_dim)
        
        # Project input to model dimension
        x = self.input_proj(x)  # (batch_size, seq_len, d_model)
        
        # Add positional encoding
        x = x.transpose(0, 1)  # (seq_len, batch_size, d_model)
        x = self.pos_encoder(x)
        x = x.transpose(0, 1)  # (batch_size, seq_len, d_model)
        
        # Apply transformer
        transformer_output = self.transformer(x)  # (batch_size, seq_len, d_model)
        
        # Use the last time step for prediction
        last_output = transformer_output[:, -1, :]  # (batch_size, d_model)
        
        # Generate predictions
        regression_output = self.regression_head(last_output)  # (batch_size, 3)
        classification_output = self.classification_head(last_output)  # (batch_size, 2)
        
        return regression_output, classification_output

class CryptoDataset(Dataset):
    """Dataset for crypto time series data"""
    
    def __init__(self, features, targets_reg, targets_cls, sequence_length=60):
        self.features = features
        self.targets_reg = targets_reg
        self.targets_cls = targets_cls
        self.sequence_length = sequence_length
        
        # Create sequences
        self.sequences = []
        self.reg_targets = []
        self.cls_targets = []
        
        for i in range(len(features) - sequence_length):
            # Feature sequence
            seq = features[i:i + sequence_length]
            self.sequences.append(seq)
            
            # Targets at the end of sequence
            end_idx = i + sequence_length - 1
            self.reg_targets.append(targets_reg[end_idx])
            self.cls_targets.append(targets_cls[end_idx])
        
        self.sequences = np.array(self.sequences, dtype=np.float32)
        self.reg_targets = np.array(self.reg_targets, dtype=np.float32)
        self.cls_targets = np.array(self.cls_targets, dtype=np.float32)
        
        logger.info(f"Created dataset with {len(self.sequences)} sequences")
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return (
            torch.FloatTensor(self.sequences[idx]),
            torch.FloatTensor(self.reg_targets[idx]),
            torch.LongTensor(self.cls_targets[idx])
        )

def load_and_prepare_data():
    """Load and prepare training data"""
    try:
        # Load feature files
        all_features = []
        all_targets_reg = []
        all_targets_cls = []
        
        for symbol in ["BTCUSDT", "ETHUSDT"]:
            feature_file = FEATURES_PATH / f"{symbol}_features.pkl"
            
            if not feature_file.exists():
                logger.warning(f"Feature file not found: {feature_file}")
                continue
            
            df = pd.read_pickle(feature_file)
            logger.info(f"Loaded {len(df)} records for {symbol}")
            
            # Select feature columns (exclude metadata and targets)
            feature_columns = [col for col in df.columns if not col.startswith('target_') 
                             and col not in ['timestamp', 'symbol', 'datetime', 'created_at']]
            
            # Remove columns with too many NaN values
            feature_columns = [col for col in feature_columns 
                             if df[col].isnull().sum() / len(df) < 0.5]
            
            logger.info(f"Selected {len(feature_columns)} features for {symbol}")
            
            # Prepare features
            features = df[feature_columns].fillna(0).values
            
            # Prepare regression targets (price predictions)
            reg_targets = df[['target_5min', 'target_10min', 'target_30min']].fillna(0).values
            
            # Prepare classification targets (direction predictions)
            cls_targets = df[['target_direction_5min', 'target_direction_10min', 'target_direction_30min']].fillna(0).values
            
            # Remove rows with invalid targets
            valid_mask = ~np.isnan(reg_targets).any(axis=1) & ~np.isnan(cls_targets).any(axis=1)
            features = features[valid_mask]
            reg_targets = reg_targets[valid_mask]
            cls_targets = cls_targets[valid_mask]
            
            all_features.append(features)
            all_targets_reg.append(reg_targets)
            all_targets_cls.append(cls_targets)
        
        if not all_features:
            raise ValueError("No valid feature data found")
        
        # Combine all symbols
        combined_features = np.vstack(all_features)
        combined_reg_targets = np.vstack(all_targets_reg)
        combined_cls_targets = np.vstack(all_targets_cls)
        
        logger.info(f"Combined dataset shape: {combined_features.shape}")
        
        # Normalize features
        scaler = StandardScaler()
        combined_features = scaler.fit_transform(combined_features)
        
        # Save scaler for inference
        scaler_path = MODELS_PATH / "feature_scaler.pkl"
        MODELS_PATH.mkdir(parents=True, exist_ok=True)
        pd.to_pickle(scaler, scaler_path)
        
        # Split data
        train_features, val_features, train_reg, val_reg, train_cls, val_cls = train_test_split(
            combined_features, combined_reg_targets, combined_cls_targets,
            test_size=0.2, random_state=42, shuffle=False  # Don't shuffle time series
        )
        
        # Create datasets
        train_dataset = CryptoDataset(train_features, train_reg, train_cls, SEQUENCE_LENGTH)
        val_dataset = CryptoDataset(val_features, val_reg, val_cls, SEQUENCE_LENGTH)
        
        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
        
        return train_loader, val_loader, combined_features.shape[1], feature_columns
        
    except Exception as e:
        logger.error(f"Error preparing data: {e}")
        raise

def train_epoch(model, train_loader, optimizer, criterion_reg, criterion_cls):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    reg_loss_sum = 0
    cls_loss_sum = 0
    
    for batch_idx, (features, reg_targets, cls_targets) in enumerate(train_loader):
        features = features.to(DEVICE)
        reg_targets = reg_targets.to(DEVICE)
        cls_targets = cls_targets.to(DEVICE)
        
        optimizer.zero_grad()
        
        # Forward pass
        reg_pred, cls_pred = model(features)
        
        # Calculate losses
        reg_loss = criterion_reg(reg_pred, reg_targets)
        cls_loss = criterion_cls(cls_pred, cls_targets[:, 0])  # Use 5min direction for classification
        
        # Combined loss
        total_batch_loss = reg_loss + cls_loss
        
        # Backward pass
        total_batch_loss.backward()
        optimizer.step()
        
        total_loss += total_batch_loss.item()
        reg_loss_sum += reg_loss.item()
        cls_loss_sum += cls_loss.item()
        
        if batch_idx % 100 == 0:
            logger.info(f'Batch {batch_idx}, Loss: {total_batch_loss.item():.4f}')
    
    avg_loss = total_loss / len(train_loader)
    avg_reg_loss = reg_loss_sum / len(train_loader)
    avg_cls_loss = cls_loss_sum / len(train_loader)
    
    return avg_loss, avg_reg_loss, avg_cls_loss

def validate_epoch(model, val_loader, criterion_reg, criterion_cls):
    """Validate for one epoch"""
    model.eval()
    total_loss = 0
    reg_loss_sum = 0
    cls_loss_sum = 0
    
    all_reg_preds = []
    all_reg_targets = []
    all_cls_preds = []
    all_cls_targets = []
    
    with torch.no_grad():
        for features, reg_targets, cls_targets in val_loader:
            features = features.to(DEVICE)
            reg_targets = reg_targets.to(DEVICE)
            cls_targets = cls_targets.to(DEVICE)
            
            # Forward pass
            reg_pred, cls_pred = model(features)
            
            # Calculate losses
            reg_loss = criterion_reg(reg_pred, reg_targets)
            cls_loss = criterion_cls(cls_pred, cls_targets[:, 0])
            
            total_loss += (reg_loss + cls_loss).item()
            reg_loss_sum += reg_loss.item()
            cls_loss_sum += cls_loss.item()
            
            # Store predictions for metrics
            all_reg_preds.append(reg_pred.cpu().numpy())
            all_reg_targets.append(reg_targets.cpu().numpy())
            all_cls_preds.append(torch.softmax(cls_pred, dim=1).cpu().numpy())
            all_cls_targets.append(cls_targets[:, 0].cpu().numpy())
    
    # Calculate metrics
    all_reg_preds = np.vstack(all_reg_preds)
    all_reg_targets = np.vstack(all_reg_targets)
    all_cls_preds = np.vstack(all_cls_preds)
    all_cls_targets = np.concatenate(all_cls_targets)
    
    # Regression metrics
    mse = mean_squared_error(all_reg_targets, all_reg_preds)
    mae = mean_absolute_error(all_reg_targets, all_reg_preds)
    
    # Classification metrics
    cls_pred_labels = np.argmax(all_cls_preds, axis=1)
    accuracy = accuracy_score(all_cls_targets, cls_pred_labels)
    auc = roc_auc_score(all_cls_targets, all_cls_preds[:, 1])
    
    avg_loss = total_loss / len(val_loader)
    avg_reg_loss = reg_loss_sum / len(val_loader)
    avg_cls_loss = cls_loss_sum / len(val_loader)
    
    return avg_loss, avg_reg_loss, avg_cls_loss, mse, mae, accuracy, auc

def save_checkpoint(model, optimizer, epoch, loss, metrics, feature_columns):
    """Save model checkpoint"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'metrics': metrics,
        'feature_columns': feature_columns,
        'model_config': {
            'feature_dim': model.feature_dim,
            'd_model': model.d_model,
            'sequence_length': SEQUENCE_LENGTH
        }
    }
    
    checkpoint_path = MODELS_PATH / f"crypto_transformer_epoch_{epoch}.pt"
    torch.save(checkpoint, checkpoint_path)
    logger.info(f"Saved checkpoint: {checkpoint_path}")
    
    return checkpoint_path

def main():
    """Main training function"""
    logger.info(f"Starting training on device: {DEVICE}")
    
    # Load and prepare data
    train_loader, val_loader, feature_dim, feature_columns = load_and_prepare_data()
    
    # Initialize model
    model = TimeSeriesTransformer(feature_dim=feature_dim).to(DEVICE)
    logger.info(f"Model initialized with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Loss functions and optimizer
    criterion_reg = nn.MSELoss()
    criterion_cls = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
    
    # Training loop
    best_val_loss = float('inf')
    best_checkpoint_path = None
    
    for epoch in range(EPOCHS):
        logger.info(f"Epoch {epoch + 1}/{EPOCHS}")
        
        # Train
        train_loss, train_reg_loss, train_cls_loss = train_epoch(
            model, train_loader, optimizer, criterion_reg, criterion_cls
        )
        
        # Validate
        val_loss, val_reg_loss, val_cls_loss, mse, mae, accuracy, auc = validate_epoch(
            model, val_loader, criterion_reg, criterion_cls
        )
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        # Log metrics
        logger.info(f"Train Loss: {train_loss:.4f} (Reg: {train_reg_loss:.4f}, Cls: {train_cls_loss:.4f})")
        logger.info(f"Val Loss: {val_loss:.4f} (Reg: {val_reg_loss:.4f}, Cls: {val_cls_loss:.4f})")
        logger.info(f"Val Metrics - MSE: {mse:.4f}, MAE: {mae:.4f}, Acc: {accuracy:.4f}, AUC: {auc:.4f}")
        
        # Save checkpoint if best
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            metrics = {
                'mse': mse,
                'mae': mae,
                'accuracy': accuracy,
                'auc': auc
            }
            best_checkpoint_path = save_checkpoint(
                model, optimizer, epoch, val_loss, metrics, feature_columns
            )
            logger.info(f"New best model saved with validation loss: {val_loss:.4f}")
        
        # Early stopping
        if optimizer.param_groups[0]['lr'] < 1e-6:
            logger.info("Learning rate too small, stopping training")
            break
    
    logger.info(f"Training completed. Best model: {best_checkpoint_path}")
    return best_checkpoint_path

if __name__ == "__main__":
    main() 