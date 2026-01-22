import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

def main():
    print("Hello from ai-engineering-fordham!")
    
    # Get OpenAI API key from environment variable
    openai_api_key = os.getenv("OPENAI_API_KEY")
    
    if openai_api_key:
        print("OpenAI API key loaded successfully!")
        # Use the API key here for your OpenAI API calls
    else:
        print("Warning: OPENAI_API_KEY not found in environment variables")


if __name__ == "__main__":
    main()
