#!/bin/bash

# Enhanced Shot Classification from CSV Startup Script

echo "🎾 Enhanced Shot Classification from CSV"
echo "======================================="

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
    echo "Usage: $0 <video_file> [output_file]"
    echo "Example: $0 tennis_test5.mp4 enhanced_shots_from_csv.mp4"
    exit 1
fi

VIDEO_FILE=$1
OUTPUT_FILE=${2:-"enhanced_shots_from_csv.mp4"}

# Check if video file exists
if [ ! -f "$VIDEO_FILE" ]; then
    echo "❌ Error: Video file '$VIDEO_FILE' not found!"
    exit 1
fi

echo "📊 CSV data: tennis_analysis_data.csv"
echo "📹 Input video: $VIDEO_FILE"
echo "🎬 Output video: $OUTPUT_FILE"
echo ""

# Run the enhanced shot classification from CSV
echo "🚀 Starting Enhanced Shot Classification from CSV..."
echo "Controls:"
echo "  - Press 'q' to quit"
echo "  - Press 'space' to pause/unpause"
echo "  - Enhanced shot types will be displayed below each player"
echo ""

python3 enhanced_shot_from_csv.py --csv tennis_analysis_data.csv --video "$VIDEO_FILE" --output "$OUTPUT_FILE"

echo ""
echo "✅ Enhanced shot classification completed!"
echo "📊 Check the output video: $OUTPUT_FILE"
