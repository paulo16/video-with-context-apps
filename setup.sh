#!/bin/bash
set -e

echo "🔧 Installing system dependencies..."

# Update package list
apt-get update -qq

# Install ffmpeg (REQUIRED by Whisper for audio processing)
echo "📦 Installing ffmpeg (this is REQUIRED)..."
if apt-get install -y ffmpeg unzip ca-certificates curl; then
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
echo "📦 Installing Deno (JavaScript runtime for yt-dlp / YouTube EJS)..."
# https://github.com/yt-dlp/yt-dlp/wiki/EJS — PyPI yt-dlp needs a JS runtime + yt-dlp-ejs (see requirements).
ARCH=$(uname -m)
case "$ARCH" in
  x86_64|amd64) DENO_ZIP="deno-x86_64-unknown-linux-gnu.zip" ;;
  aarch64|arm64) DENO_ZIP="deno-aarch64-unknown-linux-gnu.zip" ;;
  *)
    echo "⚠️  Unknown arch '$ARCH': skip Deno (YouTube URLs may return HTTP 403)."
    DENO_ZIP=""
    ;;
esac
if [ -n "$DENO_ZIP" ]; then
  if curl -fsSL "https://github.com/denoland/deno/releases/latest/download/${DENO_ZIP}" -o /tmp/deno.zip; then
    unzip -o /tmp/deno.zip -d /usr/local/bin
    chmod +x /usr/local/bin/deno
    /usr/local/bin/deno --version || true
    echo "✅ Deno installed to /usr/local/bin/deno"
  else
    echo "⚠️  Deno download failed — YouTube from URL may still hit HTTP 403."
  fi
fi

echo ""
echo "🐍 Installing Python dependencies..."

# Install spacy model for tense detection
python -m spacy download en_core_web_sm

echo ""
echo "✅ Setup complete! All dependencies installed."



