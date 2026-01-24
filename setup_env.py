#!/usr/bin/env python3
"""
Setup script to create .env file for API keys.
Run this script to set up your environment variables.
"""

import os
from pathlib import Path

def setup_env():
    """Create .env file with API key placeholders."""
    env_path = Path(".env")
    
    if env_path.exists():
        print("⚠️  .env file already exists!")
        response = input("Do you want to overwrite it? (yes/no): ")
        if response.lower() != "yes":
            print("Cancelled. Your existing .env file is unchanged.")
            return
    
    print("\n🔑 Setting up API keys for your environment...")
    print("\nYou'll need API keys from:")
    print("  1. OpenAI: https://platform.openai.com/api-keys")
    print("  2. Google AI Studio: https://aistudio.google.com/app/apikey")
    print()
    
    openai_key = input("Enter your OpenAI API key (or press Enter to skip): ").strip()
    google_key = input("Enter your Google API key (or press Enter to skip): ").strip()
    
    # Create .env content
    env_content = []
    env_content.append("# OpenAI API Key")
    env_content.append("# Get your API key from: https://platform.openai.com/api-keys")
    if openai_key:
        env_content.append(f"OPENAI_API_KEY={openai_key}")
    else:
        env_content.append("OPENAI_API_KEY=your-openai-api-key-here")
    
    env_content.append("")
    env_content.append("# Google GenAI API Key (for image generation)")
    env_content.append("# Get your API key from: https://aistudio.google.com/app/apikey")
    if google_key:
        env_content.append(f"GOOGLE_API_KEY={google_key}")
    else:
        env_content.append("GOOGLE_API_KEY=your-google-api-key-here")
    
    # Write .env file
    try:
        with open(env_path, "w") as f:
            f.write("\n".join(env_content))
        print(f"\n✅ Created .env file at {env_path.absolute()}")
        if not openai_key or not google_key:
            print("⚠️  Remember to update the placeholder values with your actual API keys!")
    except Exception as e:
        print(f"\n❌ Error creating .env file: {e}")
        print("\nYou can manually create a .env file with:")
        print("OPENAI_API_KEY=your-key-here")
        print("GOOGLE_API_KEY=your-key-here")

if __name__ == "__main__":
    setup_env()
