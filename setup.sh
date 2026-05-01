#!/bin/bash
# Install spacy model for tense detection
python -m spacy download en_core_web_sm

# Install ffmpeg for video processing (optional, not strictly required for mp4)
apt-get update && apt-get install -y ffmpeg 2>/dev/null || echo "ffmpeg installation skipped or unavailable"

