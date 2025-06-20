#!/usr/bin/env python3
"""
Test suite for Decision Transformer implementation
Tests model architecture, training, and inference capabilities
"""

import unittest
import torch
import numpy as np
import tempfile
import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from pc.models.decision_transformer import (
    DecisionTransformer, 
    DecisionTransformerConfig,
    FlashMultiheadAttention,
    TransformerBlock
)
from pc.offline_rl_trainer import (
    OfflineRLTrainer,
    TrainingConfig,
    ExperienceReplayBuffer,
    Experience,
    CombinatorialPurgedCV
)

class TestDecisionTransformer(unittest.TestCase):
    """Test Decision Transformer model"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.config = DecisionTransformerConfig(
            hidden_size=128,
            num_attention_heads=4,
            num_hidden_layers=2,
            context_length=30,
            use_flash_attention=False,  # Disable for testing
            use_bf16=False  # Disable for testing
        )
        self.model = DecisionTransformer(self.config)
        self.model.to(self.device)
    
    def test_model_creation(self):
        """Test model initialization"""
        self.assertIsInstance(self.model, DecisionTransformer)
        self.assertEqual(len(self.model.transformer_blocks), 2)
        
        # Check task heads exist
        self.assertTrue(hasattr(self.model, 'action_head'))
        self.assertTrue(hasattr(self.model, 'position_size_head'))
        self.assertTrue(hasattr(self.model, 'risk_score_head'))
        self.assertTrue(hasattr(self.model, 'value_head'))
    
    def test_forward_pass(self):
        """Test forward pass through model"""
        batch_size = 4
        seq_len = 30
        feature_dim = 256
        
        # Create dummy inputs
        states = torch.randn(batch_size, seq_len, feature_dim).to(self.device)
        actions = torch.randint(0, 3, (batch_size, seq_len)).to(self.device)
        returns_to_go = torch.randn(batch_size, seq_len, 1).to(self.device)
        timesteps = torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1).to(self.device)
        
        # Forward pass
        outputs = self.model(states, actions, returns_to_go, timesteps)
        
        # Check outputs
        self.assertIn('action_logits', outputs)
        self.assertIn('action_probs', outputs)
        self.assertIn('position_size', outputs)
        self.assertIn('risk_score', outputs)
        self.assertIn('value', outputs)
        
        # Check shapes
        self.assertEqual(outputs['action_logits'].shape, (batch_size, seq_len, 3))
        self.assertEqual(outputs['position_size'].shape, (batch_size, seq_len, 1))
        self.assertEqual(outputs['risk_score'].shape, (batch_size, seq_len, 1))
        self.assertEqual(outputs['value'].shape, (batch_size, seq_len, 1))
        
        # Check value ranges
        self.assertTrue(torch.all(outputs['position_size'] >= 0))
        self.assertTrue(torch.all(outputs['position_size'] <= self.config.max_position_size))
        self.assertTrue(torch.all(outputs['risk_score'] >= 0))
        self.assertTrue(torch.all(outputs['risk_score'] <= 1))
    
    def test_backbone_freezing(self):
        """Test backbone freezing functionality"""
        # Count trainable parameters before freezing
        trainable_before = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        # Freeze backbone
        self.model.freeze_backbone()
        
        # Count trainable parameters after freezing
        trainable_after = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        # Should have fewer trainable parameters
        self.assertLess(trainable_after, trainable_before)
        
        # Check that transformer blocks are frozen
        for block in self.model.transformer_blocks:
            for param in block.parameters():
                self.assertFalse(param.requires_grad)
        
        # Check that task heads are still trainable
        for param in self.model.action_head.parameters():
            self.assertTrue(param.requires_grad)
    
    def test_unfreeze_layers(self):
        """Test selective unfreezing of layers"""
        self.model.freeze_backbone()
        self.model.unfreeze_last_n_layers(1)
        
        # Last layer should be trainable
        last_block = self.model.transformer_blocks[-1]
        for param in last_block.parameters():
            self.assertTrue(param.requires_grad)
        
        # Earlier layers should still be frozen
        if len(self.model.transformer_blocks) > 1:
            first_block = self.model.transformer_blocks[0]
            for param in first_block.parameters():
                self.assertFalse(param.requires_grad)
    
    def test_inference(self):
        """Test single-step inference"""
        self.model.eval()
        
        # Single sequence
        states = torch.randn(1, 30, 256).to(self.device)
        actions = torch.randint(0, 3, (1, 30)).to(self.device)
        returns_to_go = torch.randn(1, 30, 1).to(self.device)
        
        # Get action
        action, position_size, risk_score = self.model.get_action(
            states[0], actions[0], returns_to_go[0], deterministic=True
        )
        
        # Check outputs
        self.assertIn(action, [0, 1, 2])  # Valid action
        self.assertGreaterEqual(position_size, 0)
        self.assertLessEqual(position_size, self.config.max_position_size)
        self.assertGreaterEqual(risk_score, 0)
        self.assertLessEqual(risk_score, 1)
    
    def test_checkpoint_save_load(self):
        """Test model checkpoint saving and loading"""
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = os.path.join(tmpdir, 'test_checkpoint.pt')
            
            # Save checkpoint
            optimizer = torch.optim.Adam(self.model.parameters())
            self.model.save_checkpoint(checkpoint_path, optimizer, epoch=5)
            
            # Check file exists
            self.assertTrue(os.path.exists(checkpoint_path))
            
            # Load checkpoint
            loaded_model, checkpoint = DecisionTransformer.load_checkpoint(
                checkpoint_path, device=self.device
            )
            
            # Check loaded correctly
            self.assertEqual(checkpoint['epoch'], 5)
            self.assertIsInstance(loaded_model, DecisionTransformer)
            
            # Compare model outputs
            self.model.eval()
            loaded_model.eval()
            
            with torch.no_grad():
                test_input = torch.randn(1, 30, 256).to(self.device)
                original_output = self.model.state_encoder(test_input)
                loaded_output = loaded_model.state_encoder(test_input)
                
                torch.testing.assert_close(original_output, loaded_output)

class TestFlashAttention(unittest.TestCase):
    """Test Flash Attention implementation"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.config = DecisionTransformerConfig(
            hidden_size=128,
            num_attention_heads=4,
            use_flash_attention=False  # Test standard attention path
        )
        self.attention = FlashMultiheadAttention(self.config)
    
    def test_attention_forward(self):
        """Test attention forward pass"""
        batch_size = 2
        seq_len = 16
        hidden_size = 128
        
        hidden_states = torch.randn(batch_size, seq_len, hidden_size)
        
        # Forward pass
        output = self.attention(hidden_states, causal_mask=True)
        
        # Check output shape
        self.assertEqual(output.shape, (batch_size, seq_len, hidden_size))
        
        # Check that output is not NaN
        self.assertFalse(torch.isnan(output).any())

class TestExperienceReplayBuffer(unittest.TestCase):
    """Test experience replay buffer"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.buffer = ExperienceReplayBuffer(capacity=100)
    
    def test_add_and_sample(self):
        """Test adding and sampling experiences"""
        # Create dummy experiences
        for i in range(10):
            exp = Experience(
                states=np.random.randn(30, 256),
                actions=np.random.randint(0, 3, 30),
                rewards=np.random.randn(30),
                returns_to_go=np.random.randn(30),
                timesteps=np.arange(30),
                dones=np.zeros(30)
            )
            self.buffer.add(exp, priority=1.0 + i * 0.1)
        
        # Check buffer size
        self.assertEqual(len(self.buffer), 10)
        
        # Sample batch
        batch = self.buffer.sample(batch_size=5)
        self.assertEqual(len(batch), 5)
        
        # Check sampled experiences
        for exp in batch:
            self.assertEqual(exp.states.shape, (30, 256))
            self.assertEqual(exp.actions.shape, (30,))

class TestCombinatorialPurgedCV(unittest.TestCase):
    """Test combinatorial purged cross-validation"""
    
    def test_split_generation(self):
        """Test CV split generation with embargo"""
        import pandas as pd
        
        # Create dummy data
        data = pd.DataFrame({
            'timestamp': pd.date_range('2023-01-01', periods=1000, freq='H'),
            'value': np.random.randn(1000)
        })
        
        # Create CV splitter
        cv = CombinatorialPurgedCV(n_splits=5, embargo_pct=0.01)
        splits = cv.split(data)
        
        # Check number of splits
        self.assertEqual(len(splits), 5)
        
        # Check each split
        for train_idx, test_idx in splits:
            # Check no overlap
            overlap = set(train_idx) & set(test_idx)
            self.assertEqual(len(overlap), 0)
            
            # Check embargo gap exists
            if len(train_idx) > 0 and len(test_idx) > 0:
                # Find closest train index to test set
                test_start = test_idx.min()
                test_end = test_idx.max()
                
                train_before_test = train_idx[train_idx < test_start]
                train_after_test = train_idx[train_idx > test_end]
                
                # Check embargo gap
                if len(train_before_test) > 0:
                    gap_before = test_start - train_before_test.max()
                    self.assertGreater(gap_before, 5)  # At least 5 samples gap
                
                if len(train_after_test) > 0:
                    gap_after = train_after_test.min() - test_end
                    self.assertGreater(gap_after, 5)

class TestOfflineRLTrainer(unittest.TestCase):
    """Test offline RL trainer"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.config = TrainingConfig(
            quarantine_days=30,
            batch_size=4,
            num_epochs=2,
            learning_rate=1e-4
        )
        
        model_config = DecisionTransformerConfig(
            hidden_size=64,
            num_attention_heads=2,
            num_hidden_layers=2,
            context_length=30
        )
        
        self.model = DecisionTransformer(model_config)
        self.trainer = OfflineRLTrainer(self.config, self.model)
    
    def test_quarantine_application(self):
        """Test 30-day quarantine rule"""
        import pandas as pd
        from datetime import datetime, timedelta
        
        # Create data with recent timestamps
        current_time = datetime.now()
        data = pd.DataFrame({
            'timestamp': [
                (current_time - timedelta(days=40)).timestamp(),
                (current_time - timedelta(days=20)).timestamp(),
                (current_time - timedelta(days=10)).timestamp(),
                (current_time - timedelta(days=5)).timestamp(),
            ],
            'value': [1, 2, 3, 4]
        })
        
        # Apply quarantine
        quarantined = self.trainer.apply_quarantine(data)
        
        # Check that recent data is removed
        self.assertEqual(len(quarantined), 2)  # Only first 2 rows should remain
        self.assertTrue(all(quarantined['value'] == [1, 2]))
    
    def test_returns_to_go_calculation(self):
        """Test returns-to-go calculation"""
        rewards = np.array([1, 2, 3, 4, 5])
        gamma = 0.9
        
        returns_to_go = self.trainer.calculate_returns_to_go(rewards, gamma)
        
        # Check last value
        self.assertEqual(returns_to_go[-1], 5)
        
        # Check second to last
        expected = 4 + gamma * 5
        self.assertAlmostEqual(returns_to_go[-2], expected, places=5)
    
    def test_risk_adjusted_rewards(self):
        """Test risk-adjusted reward calculation"""
        returns = np.array([0.01, -0.02, 0.015, -0.005, 0.01])
        positions = np.array([0.1, 0.1, 0.2, 0.2, 0.1])
        
        rewards = self.trainer.calculate_risk_adjusted_rewards(returns, positions)
        
        # Check that rewards are computed
        self.assertEqual(len(rewards), len(returns))
        
        # Check that drawdowns are penalized (negative returns should have more negative rewards)
        self.assertLess(rewards[1], returns[1])  # Drawdown penalty applied

def run_tests():
    """Run all tests"""
    unittest.main(argv=[''], exit=False)

if __name__ == '__main__':
    print("Running Decision Transformer tests...")
    run_tests()