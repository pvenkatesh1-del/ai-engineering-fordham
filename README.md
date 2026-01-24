# ai-engineering-fordham
My coursework for Introduction to AI Engineering at Fordham

## Setup

### Environment Variables (API Keys)

This project requires API keys for OpenAI and Google GenAI. Set them up using one of the following methods:

#### Option 1: Use the setup script (Recommended)
```bash
python setup_env.py
```

#### Option 2: Create .env file manually
Create a `.env` file in the project root with:
```bash
OPENAI_API_KEY=your-openai-api-key-here
GOOGLE_API_KEY=your-google-api-key-here
```

**Get your API keys:**
- OpenAI: https://platform.openai.com/api-keys
- Google AI Studio: https://aistudio.google.com/app/apikey

The `.env` file is already in `.gitignore`, so your keys won't be committed to git.
