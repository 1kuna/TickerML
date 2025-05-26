#!/usr/bin/env python3
"""
Verify Gemma 3 4B model configuration
"""

import yaml
import os
from pathlib import Path

def verify_config():
    """Verify that all configuration files use Gemma 3 4B model"""
    
    project_root = Path(__file__).parent.parent
    
    print("Verifying Gemma 3 4B Model Configuration")
    print("=" * 50)
    
    # Check config.yaml
    config_path = project_root / "config" / "config.yaml"
    if config_path.exists():
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        model = config.get("features", {}).get("sentiment", {}).get("model", "")
        print(f"✓ config.yaml model: {model}")
        
        if model == "gemma3:4b":
            print("✓ config.yaml correctly configured for Gemma 3 4B")
        else:
            print(f"✗ config.yaml has incorrect model: {model}")
    else:
        print("✗ config.yaml not found")
    
    # Check config.yaml.sample
    config_sample_path = project_root / "config" / "config.yaml.sample"
    if config_sample_path.exists():
        with open(config_sample_path, 'r') as f:
            config_sample = yaml.safe_load(f)
        
        model_sample = config_sample.get("features", {}).get("sentiment", {}).get("model", "")
        print(f"✓ config.yaml.sample model: {model_sample}")
        
        if model_sample == "gemma3:4b":
            print("✓ config.yaml.sample correctly configured for Gemma 3 4B")
        else:
            print(f"✗ config.yaml.sample has incorrect model: {model_sample}")
    else:
        print("✗ config.yaml.sample not found")
    
    # Check env.sample
    env_sample_path = project_root / "config" / "env.sample"
    if env_sample_path.exists():
        with open(env_sample_path, 'r') as f:
            env_content = f.read()
        
        if "OLLAMA_MODEL=gemma3:4b" in env_content:
            print("✓ env.sample correctly configured for Gemma 3 4B")
        else:
            print("✗ env.sample has incorrect model configuration")
    else:
        print("✗ env.sample not found")
    
    # Check README.md
    readme_path = project_root / "README.md"
    if readme_path.exists():
        with open(readme_path, 'r') as f:
            readme_content = f.read()
        
        if "ollama pull gemma3:4b" in readme_content:
            print("✓ README.md correctly references Gemma 3 4B")
        else:
            print("✗ README.md has incorrect model references")
    else:
        print("✗ README.md not found")
    
    # Check setup.sh
    setup_path = project_root / "scripts" / "setup.sh"
    if setup_path.exists():
        with open(setup_path, 'r') as f:
            setup_content = f.read()
        
        if "ollama pull gemma3:4b" in setup_content:
            print("✓ setup.sh correctly references Gemma 3 4B")
        else:
            print("✗ setup.sh has incorrect model references")
    else:
        print("✗ setup.sh not found")
    
    print("\n" + "=" * 50)
    print("Configuration verification completed!")
    print("\nTo use Gemma 3 4B model:")
    print("1. Install Ollama: https://ollama.ai")
    print("2. Pull the model: ollama pull gemma3:4b")
    print("3. Start Ollama: ollama serve")
    print("4. Run feature engineering: python pc/features.py")

if __name__ == "__main__":
    verify_config() 