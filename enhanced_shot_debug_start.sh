#!/bin/bash

# Enhanced Shot Classification Debug Tool Startup Script

echo "🎾 Enhanced Shot Classification Debug Tool"
echo "========================================="

# Activate virtual environment
echo "Activating virtual environment..."
source tennis_env/bin/activate

# Check if CSV file exists
if [ ! -f "tennis_analysis_data.csv" ]; then
    echo "❌ Error: CSV file 'tennis_analysis_data.csv' not found!"
    echo "Please run tennis_master.py first to generate the CSV data."
    exit 1
fi

# Check if video file is provided
if [ $# -eq 0 ]; then
    echo "Usage: $0 <video_file> [start_frame] [num_frames]"
    echo "Example: $0 tennis_test5.mp4 0 20"
    exit 1
fi

VIDEO_FILE=$1
START_FRAME=${2:-0}
NUM_FRAMES=${3:-20}

# Check if video file exists
if [ ! -f "$VIDEO_FILE" ]; then
    echo "❌ Error: Video file '$VIDEO_FILE' not found!"
    exit 1
fi

echo "📊 CSV data: tennis_analysis_data.csv"
echo "📹 Input video: $VIDEO_FILE"
echo "🎯 Start frame: $START_FRAME"
echo "📈 Number of frames: $NUM_FRAMES"
echo ""

echo "🚀 Starting Enhanced Shot Classification Debug Tool..."
echo "Controls:"
echo "  - Press 'q' to quit"
echo "  - Press 'n' to go to next frame"
echo "  - Press 'space' to pause"
echo "  - Keypoints are color-coded by confidence"
echo "  - Arm lines: Green = extended, Red = not extended"
echo ""

# Run the enhanced shot classification debug tool
python3 enhanced_shot_debug.py \
    --csv "tennis_analysis_data.csv" \
    --video "$VIDEO_FILE" \
    --start "$START_FRAME" \
    --frames "$NUM_FRAMES"

echo ""
echo "✅ Debug session completed!"
echo "Use the insights to tune the classification logic."
