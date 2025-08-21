#!/bin/bash

echo "🎾 Enhanced Shot Classification Debug Video"
echo "=========================================="

# Check if virtual environment exists
if [ ! -d "tennis_env" ]; then
    echo "❌ Virtual environment not found. Please run setup first."
    exit 1
fi

# Check if CSV file exists
if [ ! -f "tennis_analysis_data.csv" ]; then
    echo "❌ CSV file not found: tennis_analysis_data.csv"
    echo "Please run tennis_master.py first to generate the analysis data."
    exit 1
fi

# Check if video file exists
if [ ! -f "$1" ]; then
    echo "❌ Video file not found: $1"
    echo "Usage: ./enhanced_shot_debug_video_start.sh <video_file>"
    echo "Example: ./enhanced_shot_debug_video_start.sh tennis_test5.mp4"
    exit 1
fi

echo "📊 CSV data: tennis_analysis_data.csv"
echo "📹 Input video: $1"
echo "🎬 Output debug video: shot_debugging.mp4"
echo ""

echo "🚀 Creating Enhanced Shot Classification Debug Video..."
echo "This will create a detailed debug video for frame-by-frame analysis."
echo ""

# Activate virtual environment and run debug video
source tennis_env/bin/activate && python3 enhanced_shot_debug_video.py tennis_analysis_data.csv "$1" shot_debugging.mp4

echo ""
echo "✅ Debug video creation completed!"
echo "📊 Check the output video: shot_debugging.mp4"
echo "🎯 You can now scrub through frame-by-frame to analyze classifications"
