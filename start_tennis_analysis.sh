#!/bin/bash

# Tennis Analysis System Startup Script
# This script launches the dual-viewer tennis analysis system

echo "🎾 Tennis Analysis System Startup"
echo "=================================="

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "❌ Python3 is not installed or not in PATH"
    exit 1
fi

# Check if required files exist
required_files=("src/core/tennis_master.py" "src/core/tennis_CV.py" "src/core/tennis_analytics.py" "config.yaml")
for file in "${required_files[@]}"; do
    if [ ! -f "$file" ]; then
        echo "❌ Required file not found: $file"
        exit 1
    fi
done

# Find available video files in data/raw/
video_files=($(ls data/raw/*.mp4 2>/dev/null | xargs -n 1 basename 2>/dev/null))
if [ ${#video_files[@]} -eq 0 ]; then
    echo "❌ No MP4 video files found in data/raw/ directory"
    exit 1
fi

# If multiple videos, let user choose
if [ ${#video_files[@]} -gt 1 ]; then
    echo "📹 Available video files:"
    for i in "${!video_files[@]}"; do
        echo "  $((i+1)). ${video_files[$i]}"
    done
    
    read -p "Select video file (1-${#video_files[@]}): " choice
    if [[ ! "$choice" =~ ^[0-9]+$ ]] || [ "$choice" -lt 1 ] || [ "$choice" -gt ${#video_files[@]} ]; then
        echo "❌ Invalid selection"
        exit 1
    fi
    
    selected_video="${video_files[$((choice-1))]}"
else
    selected_video="${video_files[0]}"
fi

echo "✅ Starting tennis analysis with video: $selected_video"
echo "🚀 Launching dual-viewer system..."
echo ""

# Run the master controller
python3 src/core/tennis_master.py --video "data/raw/$selected_video"

echo ""
echo "🎾 Tennis analysis completed"
