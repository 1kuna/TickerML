#!/usr/bin/env python3
"""
Test script for Gemma 3 sentiment analysis
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'pc'))

from features import analyze_sentiment_with_gemma3

def test_sentiment_analysis():
    """Test the Gemma 3 sentiment analysis function"""
    
    test_cases = [
        "Bitcoin price surges to new all-time high as institutional adoption increases",
        "Cryptocurrency market crashes amid regulatory concerns and investor panic",
        "Ethereum network upgrade completed successfully with minimal disruption",
        "Mixed signals in crypto market as some coins rise while others fall"
    ]
    
    print("Testing Gemma 3 Sentiment Analysis")
    print("=" * 50)
    
    for i, text in enumerate(test_cases, 1):
        print(f"\nTest Case {i}:")
        print(f"Text: {text}")
        
        try:
            sentiment_score = analyze_sentiment_with_gemma3(text)
            print(f"Sentiment Score: {sentiment_score:.3f}")
            
            if sentiment_score > 0.1:
                sentiment_label = "Positive"
            elif sentiment_score < -0.1:
                sentiment_label = "Negative"
            else:
                sentiment_label = "Neutral"
                
            print(f"Sentiment Label: {sentiment_label}")
            
        except Exception as e:
            print(f"Error: {e}")
    
    print("\n" + "=" * 50)
    print("Test completed!")

if __name__ == "__main__":
    test_sentiment_analysis() 