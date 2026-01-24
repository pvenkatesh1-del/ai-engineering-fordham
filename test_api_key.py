#!/usr/bin/env python3
"""
Test script to verify your OpenAI API key is set up correctly.
Run this after setting up your .env file.
"""

import os
from dotenv import load_dotenv
import litellm

# Load environment variables
load_dotenv()

def test_openai_key():
    """Test if OpenAI API key is configured and working."""
    api_key = os.getenv("OPENAI_API_KEY")
    
    if not api_key or api_key == "your-openai-api-key-here":
        print("❌ OPENAI_API_KEY not set or still has placeholder value!")
        print("   Please update your .env file with your actual API key.")
        return False
    
    print("✅ OPENAI_API_KEY found in environment")
    print(f"   Key starts with: {api_key[:7]}...")
    
    # Test the API key with a simple call
    try:
        print("\n🧪 Testing API key with a simple call...")
        response = litellm.completion(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "Say 'API working!' and nothing else."}],
            max_tokens=20
        )
        result = response.choices[0].message.content
        print(f"✅ API key works! Response: {result}")
        return True
    except Exception as e:
        print(f"❌ API call failed: {e}")
        print("   Please check that your API key is valid and has credits.")
        return False

if __name__ == "__main__":
    print("=" * 50)
    print("OpenAI API Key Test")
    print("=" * 50)
    test_openai_key()
