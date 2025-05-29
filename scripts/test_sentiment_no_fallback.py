#!/usr/bin/env python3
"""
Test script to verify sentiment analysis fails without Ollama
"""

import sys
from pathlib import Path

# Add parent directory to path to import our modules
sys.path.append(str(Path(__file__).parent.parent))

from raspberry_pi.news_harvest import analyze_sentiment_simple

def test_sentiment_no_fallback():
    """Test that sentiment analysis fails when Ollama is not available"""
    
    test_text = "Bitcoin surges to new all-time high amid positive market sentiment"
    
    print("Testing sentiment analysis without fallback...")
    print(f"Test text: {test_text}")
    print()
    
    try:
        sentiment_score = analyze_sentiment_simple(test_text)
        print(f"✅ Sentiment analysis succeeded: {sentiment_score}")
        print("This means Ollama with Gemma 3 is available and working")
    except ImportError as e:
        print(f"❌ ImportError: {e}")
        print("This is expected if Ollama library is not installed")
    except RuntimeError as e:
        print(f"❌ RuntimeError: {e}")
        print("This is expected if Ollama service is not running or Gemma 3 model is not available")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        print("This indicates an unexpected problem")

if __name__ == "__main__":
    test_sentiment_no_fallback() 