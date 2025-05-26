#!/usr/bin/env python3
"""
ONNX export and quantization script for PC
Converts PyTorch model to optimized ONNX format for Raspberry Pi inference
"""

import torch
import torch.onnx
import onnxruntime as ort
from onnxruntime.quantization import quantize_dynamic, QuantType
import numpy as np
import logging
from pathlib import Path
import json

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration
CHECKPOINTS_PATH = Path(__file__).parent.parent / "models" / "checkpoints"
ONNX_PATH = Path(__file__).parent.parent / "models" / "onnx"
SEQUENCE_LENGTH = 60

# Import model class from train.py
import sys
sys.path.append(str(Path(__file__).parent))
from train import TimeSeriesTransformer

def find_best_checkpoint():
    """Find the best checkpoint file"""
    try:
        checkpoint_files = list(CHECKPOINTS_PATH.glob("crypto_transformer_epoch_*.pt"))
        
        if not checkpoint_files:
            raise FileNotFoundError("No checkpoint files found")
        
        # Load each checkpoint and find the one with lowest validation loss
        best_checkpoint = None
        best_loss = float('inf')
        
        for checkpoint_file in checkpoint_files:
            try:
                checkpoint = torch.load(checkpoint_file, map_location='cpu')
                val_loss = checkpoint.get('loss', float('inf'))
                
                if val_loss < best_loss:
                    best_loss = val_loss
                    best_checkpoint = checkpoint_file
                    
            except Exception as e:
                logger.warning(f"Error loading checkpoint {checkpoint_file}: {e}")
                continue
        
        if best_checkpoint is None:
            raise ValueError("No valid checkpoint found")
        
        logger.info(f"Best checkpoint: {best_checkpoint} (loss: {best_loss:.4f})")
        return best_checkpoint
        
    except Exception as e:
        logger.error(f"Error finding best checkpoint: {e}")
        raise

def load_model_from_checkpoint(checkpoint_path):
    """Load model from checkpoint"""
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # Get model configuration
        model_config = checkpoint['model_config']
        feature_dim = model_config['feature_dim']
        d_model = model_config.get('d_model', 128)
        
        # Initialize model
        model = TimeSeriesTransformer(
            feature_dim=feature_dim,
            d_model=d_model,
            n_heads=4,
            n_layers=6
        )
        
        # Load state dict
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        logger.info(f"Loaded model with {sum(p.numel() for p in model.parameters())} parameters")
        
        # Return model and metadata
        return model, {
            'feature_dim': feature_dim,
            'd_model': d_model,
            'sequence_length': SEQUENCE_LENGTH,
            'feature_columns': checkpoint.get('feature_columns', []),
            'metrics': checkpoint.get('metrics', {}),
            'epoch': checkpoint.get('epoch', 0)
        }
        
    except Exception as e:
        logger.error(f"Error loading model from checkpoint: {e}")
        raise

def export_to_onnx(model, feature_dim, output_path):
    """Export PyTorch model to ONNX format"""
    try:
        # Create dummy input
        dummy_input = torch.randn(1, SEQUENCE_LENGTH, feature_dim)
        
        # Export to ONNX
        torch.onnx.export(
            model,
            dummy_input,
            output_path,
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['regression_output', 'classification_output'],
            dynamic_axes={
                'input': {0: 'batch_size'},
                'regression_output': {0: 'batch_size'},
                'classification_output': {0: 'batch_size'}
            }
        )
        
        logger.info(f"Exported ONNX model to {output_path}")
        
        # Verify the exported model
        verify_onnx_model(output_path, dummy_input)
        
        return output_path
        
    except Exception as e:
        logger.error(f"Error exporting to ONNX: {e}")
        raise

def verify_onnx_model(onnx_path, dummy_input):
    """Verify ONNX model by running inference"""
    try:
        # Load ONNX model
        session = ort.InferenceSession(str(onnx_path))
        
        # Get input/output info
        input_name = session.get_inputs()[0].name
        output_names = [output.name for output in session.get_outputs()]
        
        logger.info(f"ONNX model inputs: {[inp.name for inp in session.get_inputs()]}")
        logger.info(f"ONNX model outputs: {output_names}")
        
        # Run inference
        onnx_outputs = session.run(output_names, {input_name: dummy_input.numpy()})
        
        logger.info(f"ONNX inference successful:")
        logger.info(f"  Regression output shape: {onnx_outputs[0].shape}")
        logger.info(f"  Classification output shape: {onnx_outputs[1].shape}")
        
        return True
        
    except Exception as e:
        logger.error(f"Error verifying ONNX model: {e}")
        return False

def quantize_onnx_model(onnx_path, quantized_path):
    """Quantize ONNX model for faster inference"""
    try:
        # Quantize the model
        quantize_dynamic(
            model_input=str(onnx_path),
            model_output=str(quantized_path),
            weight_type=QuantType.QInt8
        )
        
        logger.info(f"Quantized model saved to {quantized_path}")
        
        # Compare model sizes
        original_size = onnx_path.stat().st_size / (1024 * 1024)  # MB
        quantized_size = quantized_path.stat().st_size / (1024 * 1024)  # MB
        compression_ratio = original_size / quantized_size
        
        logger.info(f"Model size comparison:")
        logger.info(f"  Original: {original_size:.2f} MB")
        logger.info(f"  Quantized: {quantized_size:.2f} MB")
        logger.info(f"  Compression ratio: {compression_ratio:.2f}x")
        
        # Verify quantized model
        verify_quantized_model(quantized_path)
        
        return quantized_path
        
    except Exception as e:
        logger.error(f"Error quantizing ONNX model: {e}")
        raise

def verify_quantized_model(quantized_path):
    """Verify quantized model performance"""
    try:
        # Load quantized model
        session = ort.InferenceSession(str(quantized_path))
        
        # Create test input
        test_input = np.random.randn(1, SEQUENCE_LENGTH, session.get_inputs()[0].shape[2]).astype(np.float32)
        input_name = session.get_inputs()[0].name
        
        # Run inference multiple times to measure performance
        import time
        
        num_runs = 100
        start_time = time.time()
        
        for _ in range(num_runs):
            outputs = session.run(None, {input_name: test_input})
        
        end_time = time.time()
        avg_inference_time = (end_time - start_time) / num_runs * 1000  # ms
        
        logger.info(f"Quantized model performance:")
        logger.info(f"  Average inference time: {avg_inference_time:.2f} ms")
        logger.info(f"  Regression output range: [{outputs[0].min():.4f}, {outputs[0].max():.4f}]")
        logger.info(f"  Classification output range: [{outputs[1].min():.4f}, {outputs[1].max():.4f}]")
        
        return True
        
    except Exception as e:
        logger.error(f"Error verifying quantized model: {e}")
        return False

def save_model_metadata(metadata, output_path):
    """Save model metadata for inference"""
    try:
        metadata_path = output_path.parent / "model_metadata.json"
        
        # Convert numpy types to native Python types for JSON serialization
        def convert_numpy_types(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {key: convert_numpy_types(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(item) for item in obj]
            else:
                return obj
        
        metadata_clean = convert_numpy_types(metadata)
        
        with open(metadata_path, 'w') as f:
            json.dump(metadata_clean, f, indent=2)
        
        logger.info(f"Saved model metadata to {metadata_path}")
        
    except Exception as e:
        logger.error(f"Error saving model metadata: {e}")

def compare_pytorch_onnx_outputs(model, onnx_path, feature_dim):
    """Compare PyTorch and ONNX model outputs"""
    try:
        # Create test input
        test_input = torch.randn(1, SEQUENCE_LENGTH, feature_dim)
        
        # PyTorch inference
        model.eval()
        with torch.no_grad():
            pytorch_reg, pytorch_cls = model(test_input)
        
        # ONNX inference
        session = ort.InferenceSession(str(onnx_path))
        input_name = session.get_inputs()[0].name
        onnx_outputs = session.run(None, {input_name: test_input.numpy()})
        
        # Compare outputs
        reg_diff = np.abs(pytorch_reg.numpy() - onnx_outputs[0]).max()
        cls_diff = np.abs(pytorch_cls.numpy() - onnx_outputs[1]).max()
        
        logger.info(f"PyTorch vs ONNX comparison:")
        logger.info(f"  Max regression difference: {reg_diff:.6f}")
        logger.info(f"  Max classification difference: {cls_diff:.6f}")
        
        # Check if differences are acceptable
        tolerance = 1e-5
        if reg_diff < tolerance and cls_diff < tolerance:
            logger.info("✓ PyTorch and ONNX outputs match within tolerance")
            return True
        else:
            logger.warning("⚠ PyTorch and ONNX outputs differ significantly")
            return False
        
    except Exception as e:
        logger.error(f"Error comparing outputs: {e}")
        return False

def main():
    """Main export and quantization function"""
    logger.info("Starting ONNX export and quantization")
    
    # Create output directory
    ONNX_PATH.mkdir(parents=True, exist_ok=True)
    
    # Find best checkpoint
    best_checkpoint = find_best_checkpoint()
    
    # Load model
    model, metadata = load_model_from_checkpoint(best_checkpoint)
    
    # Export to ONNX
    onnx_output_path = ONNX_PATH / "crypto_transformer.onnx"
    export_to_onnx(model, metadata['feature_dim'], onnx_output_path)
    
    # Compare PyTorch and ONNX outputs
    compare_pytorch_onnx_outputs(model, onnx_output_path, metadata['feature_dim'])
    
    # Quantize ONNX model
    quantized_output_path = ONNX_PATH / "crypto_transformer_quantized.onnx"
    quantize_onnx_model(onnx_output_path, quantized_output_path)
    
    # Save metadata
    save_model_metadata(metadata, quantized_output_path)
    
    logger.info("Export and quantization completed successfully")
    logger.info(f"Final model: {quantized_output_path}")
    
    return quantized_output_path

if __name__ == "__main__":
    main() 