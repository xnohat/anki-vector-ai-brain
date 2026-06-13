#!/bin/bash
# Launcher script for Vector Advanced AI
# Automatically activates virtual environment and runs the app

cd "$(dirname "$0")"

echo "=========================================="
echo "   Vector Advanced AI - Launcher"
echo "=========================================="
echo

# Check if virtual environment exists
if [ ! -d ".venv" ]; then
    echo "Virtual environment not found. Creating one..."
    python3 -m venv .venv
    source .venv/bin/activate
    echo "Installing dependencies..."
    pip install -r requirements.txt
    pip install -e vector-python-sdk/
else
    source .venv/bin/activate
fi

# Load .env if present (so OPENAI_API_KEY / VECTOR_GPT_MODEL can live in a file)
if [ -f ".env" ]; then
    set -a
    # shellcheck disable=SC1091
    source .env
    set +a
fi

# Check for OpenAI API key
if [ -z "$OPENAI_API_KEY" ]; then
    echo "⚠️  WARNING: OPENAI_API_KEY not set!"
    echo
    echo "The app requires an OpenAI API key for ChatGPT and Whisper."
    echo "Set it with:"
    echo "  export OPENAI_API_KEY='your-api-key-here'"
    echo
    echo "Or add to ~/.bashrc for permanent setup:"
    echo "  echo \"export OPENAI_API_KEY='your-key'\" >> ~/.bashrc"
    echo
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Run the app
echo
echo "Starting Vector Advanced AI..."
echo "- Object Detection: YOLOv8 (fast mode)"
echo "- Speech Recognition: Whisper"
echo "- AI Brain: ${VECTOR_GPT_MODEL:-gpt-5.5} (vision-enabled)"
echo
python app.py

