#!/usr/bin/env python3
"""
Quick test script to debug NewsAPI connection
"""

import requests
from datetime import datetime, timedelta
import json

# Your API key
NEWS_API_KEY = "57307964fd22448b88b7c2def01a90dd"

def test_newsapi():
    print("🔍 Testing NewsAPI connection...")
    print(f"API Key: {NEWS_API_KEY[:10]}...")
    
    # Test 1: Simple search for "Bitcoin"
    print("\n📰 Test 1: Searching for 'Bitcoin' in last 24 hours...")
    
    end_time = datetime.now()
    start_time = end_time - timedelta(hours=24)
    
    url = "https://newsapi.org/v2/everything"
    params = {
        'q': 'Bitcoin',
        'from': start_time.isoformat(),
        'to': end_time.isoformat(),
        'sortBy': 'publishedAt',
        'language': 'en',
        'apiKey': NEWS_API_KEY,
        'pageSize': 5
    }
    
    try:
        response = requests.get(url, params=params, timeout=30)
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"Total Results: {data.get('totalResults', 0)}")
            print(f"Articles Returned: {len(data.get('articles', []))}")
            
            if data.get('articles'):
                print("\n🔥 Sample Articles:")
                for i, article in enumerate(data['articles'][:3]):
                    print(f"  {i+1}. {article.get('title', 'No title')}")
                    print(f"     Source: {article.get('source', {}).get('name', 'Unknown')}")
                    print(f"     Published: {article.get('publishedAt', 'Unknown')}")
            else:
                print("❌ No articles found")
                
        else:
            print(f"❌ Error: {response.status_code}")
            print(f"Response: {response.text}")
            
    except Exception as e:
        print(f"❌ Exception: {e}")
    
    # Test 2: Check API status
    print("\n🔧 Test 2: Checking API status...")
    
    try:
        # Try a simple sources endpoint
        sources_url = "https://newsapi.org/v2/top-headlines/sources"
        sources_params = {'apiKey': NEWS_API_KEY}
        
        response = requests.get(sources_url, params=sources_params, timeout=10)
        print(f"Sources API Status: {response.status_code}")
        
        if response.status_code == 200:
            sources_data = response.json()
            print(f"Available sources: {len(sources_data.get('sources', []))}")
            print("✅ API key is working!")
        else:
            print(f"❌ API Error: {response.text}")
            
    except Exception as e:
        print(f"❌ Sources test failed: {e}")

if __name__ == "__main__":
    test_newsapi() 