#!/bin/bash
set -e

echo "🔧 Installing system dependencies..."

# Update package list
apt-get update -qq

# Install ffmpeg (REQUIRED by Whisper for audio processing)
echo "📦 Installing ffmpeg (this is REQUIRED)..."
if apt-get install -y ffmpeg; then
    echo "✅ ffmpeg installed successfully"
else
    echo "❌ FAILED to install ffmpeg - this is critical!"
    exit 1
fi

# Double-check ffmpeg is available
if ! command -v ffmpeg &> /dev/null; then
    echo "❌ ERROR: ffmpeg is not in PATH after installation!"
    exit 1
fi

echo "ffmpeg version:"
ffmpeg -version | head -1

echo ""
echo "🐍 Installing Python dependencies..."

# Install spacy model for tense detection
python -m spacy download en_core_web_sm

echo ""
echo "✅ Setup complete! All dependencies installed."



