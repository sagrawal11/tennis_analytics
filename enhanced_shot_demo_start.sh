#!/bin/bash

# Enhanced Shot Classification Demo Startup Script

echo "🎾 Enhanced Shot Classification Demo"
echo "=================================="

# Activate virtual environment
echo "Activating virtual environment..."
source tennis_env/bin/activate

# Check if video file is provided
if [ $# -eq 0 ]; then
    echo "Usage: $0 <video_file> [output_file]"
    echo "Example: $0 tennis_test5.mp4 enhanced_shots_output.mp4"
    exit 1
fi

VIDEO_FILE=$1
OUTPUT_FILE=${2:-"enhanced_shots_output.mp4"}

# Check if video file exists
if [ ! -f "$VIDEO_FILE" ]; then
    echo "❌ Error: Video file '$VIDEO_FILE' not found!"
    exit 1
fi

echo "📹 Input video: $VIDEO_FILE"
echo "🎬 Output video: $OUTPUT_FILE"
echo ""

# Run the enhanced shot classification demo
echo "🚀 Starting Enhanced Shot Classification Demo..."
echo "Controls:"
echo "  - Press 'q' to quit"
echo "  - Press 'space' to pause/unpause"
echo "  - Shot types will be displayed below each player"
echo ""

python3 enhanced_shot_demo.py --video "$VIDEO_FILE" --output "$OUTPUT_FILE"

echo ""
echo "✅ Demo completed!"
echo "📊 Check the output video: $OUTPUT_FILE"
