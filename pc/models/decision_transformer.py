#!/usr/bin/env python3
"""
Decision Transformer for Trading
Transforms from price prediction to action prediction with frozen backbone
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

@dataclass
class DecisionTransformerConfig:
    """Configuration for Decision Transformer"""
    # Model architecture
    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 6
    d_ff: int = 512
    dropout: float = 0.1
    
    # Sequence parameters
    max_seq_length: int = 60  # 60 minutes of data
    state_dim: int = 32       # Feature dimensions
    action_dim: int = 3       # buy, sell, hold
    
    # Training parameters
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    use_bf16: bool = True     # BF16 mixed precision for financial data
    
    # Decision Transformer specific
    return_conditioning: bool = True
    max_return: float = 1.0   # Maximum expected return
    return_scale: float = 100.0  # Scale returns for numerical stability
    
    # Frozen backbone
    freeze_backbone: bool = True
    trainable_layers: int = 2  # Only train last 2 layers


class PositionalEncoding(nn.Module):
    """Positional encoding for time series data"""
    
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to input"""
        return x + self.pe[:x.size(0), :]


class MultiHeadAttention(nn.Module):
    """Multi-head attention with Flash Attention optimization"""
    
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.d_k)
        
        # Flash Attention support (if available)
        self.use_flash_attention = hasattr(F, 'scaled_dot_product_attention')
    
    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, seq_len = query.size(0), query.size(1)
        
        # Linear transformations and reshape
        Q = self.w_q(query).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = self.w_k(key).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        V = self.w_v(value).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        
        if self.use_flash_attention and mask is None:
            # Use Flash Attention for RTX 4090 optimization
            attn_output = F.scaled_dot_product_attention(
                Q, K, V, 
                dropout_p=self.dropout.p if self.training else 0.0,
                is_causal=True  # Causal mask for autoregressive generation
            )
        else:
            # Standard attention
            scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
            
            if mask is not None:
                scores.masked_fill_(mask == 0, -1e9)
            
            attn_weights = F.softmax(scores, dim=-1)
            attn_weights = self.dropout(attn_weights)
            
            attn_output = torch.matmul(attn_weights, V)
        
        # Reshape and apply output projection
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.d_model
        )
        
        return self.w_o(attn_output)


class TransformerBlock(nn.Module):
    """Transformer block with pre-layer normalization"""
    
    def __init__(self, config: DecisionTransformerConfig):
        super().__init__()
        
        self.attention = MultiHeadAttention(
            config.d_model, config.n_heads, config.dropout
        )
        self.feed_forward = nn.Sequential(
            nn.Linear(config.d_model, config.d_ff),
            nn.GELU(),
            nn.Linear(config.d_ff, config.d_model),
            nn.Dropout(config.dropout)
        )
        
        self.norm1 = nn.LayerNorm(config.d_model)
        self.norm2 = nn.LayerNorm(config.d_model)
        self.dropout = nn.Dropout(config.dropout)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Pre-norm attention
        attn_out = self.attention(
            self.norm1(x), self.norm1(x), self.norm1(x), mask
        )
        x = x + self.dropout(attn_out)
        
        # Pre-norm feed-forward
        ff_out = self.feed_forward(self.norm2(x))
        x = x + self.dropout(ff_out)
        
        return x


class DecisionTransformer(nn.Module):
    """
    Decision Transformer for trading actions
    
    Key features:
    - Frozen backbone to prevent catastrophic forgetting
    - BF16 mixed precision for financial data stability
    - Return-to-go conditioning
    - Multi-task heads for action, position sizing, and risk assessment
    """
    
    def __init__(self, config: DecisionTransformerConfig):
        super().__init__()
        self.config = config
        
        # Input embeddings
        self.state_embedding = nn.Linear(config.state_dim, config.d_model)
        self.action_embedding = nn.Embedding(config.action_dim, config.d_model)
        self.return_embedding = nn.Linear(1, config.d_model)
        self.timestep_embedding = nn.Embedding(config.max_seq_length, config.d_model)
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(config.d_model, config.max_seq_length * 3)
        
        # Transformer backbone
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(config) for _ in range(config.n_layers)
        ])
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(config.d_model)
        
        # Multi-task prediction heads
        self.action_head = nn.Sequential(
            nn.Linear(config.d_model, config.d_ff // 2),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_ff // 2, config.action_dim)
        )
        
        self.position_size_head = nn.Sequential(
            nn.Linear(config.d_model, config.d_ff // 2),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_ff // 2, 1),
            nn.Sigmoid()  # Output between 0 and 1
        )
        
        self.risk_head = nn.Sequential(
            nn.Linear(config.d_model, config.d_ff // 2),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_ff // 2, 1),
            nn.Sigmoid()  # Risk score between 0 and 1
        )
        
        # Value head for RL training
        self.value_head = nn.Sequential(
            nn.Linear(config.d_model, config.d_ff // 2),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_ff // 2, 1)
        )
        
        # Initialize weights
        self.apply(self._init_weights)
        
        # Freeze backbone layers if specified
        if config.freeze_backbone:
            self._freeze_backbone()
    
    def _init_weights(self, module):
        """Initialize weights with proper scaling"""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.ones_(module.weight)
            torch.nn.init.zeros_(module.bias)
    
    def _freeze_backbone(self):
        """Freeze backbone layers except the last few"""
        logger.info(f"Freezing backbone, keeping last {self.config.trainable_layers} layers trainable")
        
        # Freeze embeddings
        for param in self.state_embedding.parameters():
            param.requires_grad = False
        for param in self.action_embedding.parameters():
            param.requires_grad = False
        for param in self.return_embedding.parameters():
            param.requires_grad = False
        for param in self.timestep_embedding.parameters():
            param.requires_grad = False
        
        # Freeze all but last N transformer blocks
        trainable_start = len(self.transformer_blocks) - self.config.trainable_layers
        for i, block in enumerate(self.transformer_blocks):
            if i < trainable_start:
                for param in block.parameters():
                    param.requires_grad = False
        
        # Keep prediction heads trainable (they are new)
        # Keep layer norm trainable
    
    def forward(self, 
                states: torch.Tensor,
                actions: torch.Tensor,
                returns_to_go: torch.Tensor,
                timesteps: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Forward pass
        
        Args:
            states: (batch_size, seq_len, state_dim)
            actions: (batch_size, seq_len) - int64 action indices
            returns_to_go: (batch_size, seq_len, 1) - target returns
            timesteps: (batch_size, seq_len) - timestep indices
            attention_mask: (batch_size, seq_len) - mask for padding
        
        Returns:
            Dict with action_logits, position_sizes, risk_scores, values
        """
        batch_size, seq_len = states.shape[0], states.shape[1]
        
        # Embed inputs
        state_embeddings = self.state_embedding(states)
        action_embeddings = self.action_embedding(actions)
        return_embeddings = self.return_embedding(returns_to_go)
        timestep_embeddings = self.timestep_embedding(timesteps)
        
        # Combine embeddings: [return, state, action] sequence
        # This creates a sequence of length seq_len * 3
        embeddings = torch.stack([
            return_embeddings, state_embeddings, action_embeddings
        ], dim=2).view(batch_size, seq_len * 3, self.config.d_model)
        
        # Add positional encoding
        embeddings = self.pos_encoding(embeddings.transpose(0, 1)).transpose(0, 1)
        
        # Create causal attention mask
        if attention_mask is not None:
            # Expand mask for [return, state, action] sequence
            expanded_mask = attention_mask.unsqueeze(2).repeat(1, 1, 3).view(batch_size, seq_len * 3)
        else:
            expanded_mask = None
        
        # Pass through transformer
        x = embeddings
        for block in self.transformer_blocks:
            x = block(x, expanded_mask)
        
        x = self.layer_norm(x)
        
        # Extract state representations (every 3rd token starting from index 1)
        state_representations = x[:, 1::3, :]  # [return, state, action] -> state
        
        # Multi-task predictions
        action_logits = self.action_head(state_representations)
        position_sizes = self.position_size_head(state_representations)
        risk_scores = self.risk_head(state_representations)
        values = self.value_head(state_representations)
        
        return {
            'action_logits': action_logits,
            'position_sizes': position_sizes.squeeze(-1),
            'risk_scores': risk_scores.squeeze(-1),
            'values': values.squeeze(-1),
            'state_representations': state_representations
        }
    
    def get_action(self, 
                   states: torch.Tensor,
                   returns_to_go: torch.Tensor,
                   timesteps: torch.Tensor) -> Dict[str, Any]:
        """
        Get action for current state (inference mode)
        
        Args:
            states: (1, 1, state_dim) - current state
            returns_to_go: (1, 1, 1) - target return
            timesteps: (1, 1) - current timestep
        
        Returns:
            Dict with action, position_size, risk_score, confidence
        """
        self.eval()
        with torch.no_grad():
            # Dummy action for the sequence (will be ignored)
            dummy_actions = torch.zeros(1, 1, dtype=torch.long, device=states.device)
            
            outputs = self.forward(states, dummy_actions, returns_to_go, timesteps)
            
            # Get predictions
            action_probs = F.softmax(outputs['action_logits'][:, -1], dim=-1)
            action = torch.argmax(action_probs, dim=-1)
            confidence = torch.max(action_probs, dim=-1)[0]
            
            position_size = outputs['position_sizes'][:, -1]
            risk_score = outputs['risk_scores'][:, -1]
            
            return {
                'action': action.item(),
                'action_probs': action_probs.cpu().numpy()[0],
                'position_size': position_size.item(),
                'risk_score': risk_score.item(),
                'confidence': confidence.item()
            }
    
    def get_trainable_parameters(self) -> List[torch.nn.Parameter]:
        """Get only trainable parameters for optimization"""
        return [p for p in self.parameters() if p.requires_grad]
    
    def unfreeze_all(self):
        """Unfreeze all parameters (for fine-tuning)"""
        for param in self.parameters():
            param.requires_grad = True
        logger.info("Unfroze all parameters")
    
    def freeze_backbone_only(self):
        """Re-freeze only the backbone"""
        self._freeze_backbone()


class DecisionTransformerLoss(nn.Module):
    """Multi-task loss for Decision Transformer"""
    
    def __init__(self, 
                 action_weight: float = 1.0,
                 position_weight: float = 0.5,
                 risk_weight: float = 0.3,
                 value_weight: float = 0.2):
        super().__init__()
        self.action_weight = action_weight
        self.position_weight = position_weight
        self.risk_weight = risk_weight
        self.value_weight = value_weight
        
        self.action_loss = nn.CrossEntropyLoss()
        self.regression_loss = nn.MSELoss()
    
    def forward(self, 
                outputs: Dict[str, torch.Tensor],
                targets: Dict[str, torch.Tensor],
                mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Calculate multi-task loss
        
        Args:
            outputs: Model outputs
            targets: Target values
            mask: Sequence mask for variable length sequences
        
        Returns:
            Dict with individual losses and total loss
        """
        losses = {}
        
        # Action classification loss
        if 'actions' in targets:
            action_loss = self.action_loss(
                outputs['action_logits'].view(-1, outputs['action_logits'].size(-1)),
                targets['actions'].view(-1)
            )
            losses['action_loss'] = action_loss * self.action_weight
        
        # Position sizing loss
        if 'position_sizes' in targets:
            pos_loss = self.regression_loss(
                outputs['position_sizes'],
                targets['position_sizes']
            )
            losses['position_loss'] = pos_loss * self.position_weight
        
        # Risk prediction loss
        if 'risk_scores' in targets:
            risk_loss = self.regression_loss(
                outputs['risk_scores'],
                targets['risk_scores']
            )
            losses['risk_loss'] = risk_loss * self.risk_weight
        
        # Value function loss (for RL)
        if 'values' in targets:
            value_loss = self.regression_loss(
                outputs['values'],
                targets['values']
            )
            losses['value_loss'] = value_loss * self.value_weight
        
        # Apply mask if provided
        if mask is not None:
            for key in losses:
                if key != 'action_loss':  # Action loss already reshaped
                    losses[key] = (losses[key] * mask).sum() / mask.sum()
        
        # Total loss
        losses['total_loss'] = sum(losses.values())
        
        return losses


def create_decision_transformer(config: DecisionTransformerConfig) -> DecisionTransformer:
    """Factory function to create Decision Transformer"""
    model = DecisionTransformer(config)
    
    logger.info(f"Created Decision Transformer with {sum(p.numel() for p in model.parameters())} parameters")
    logger.info(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    
    return model


def load_pretrained_backbone(model: DecisionTransformer, checkpoint_path: str) -> DecisionTransformer:
    """Load pretrained weights for the backbone"""
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # Load only backbone weights
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in checkpoint.items() 
                          if k in model_dict and 'head' not in k}
        
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        
        logger.info(f"Loaded pretrained backbone from {checkpoint_path}")
        return model
        
    except Exception as e:
        logger.error(f"Error loading pretrained backbone: {e}")
        return model