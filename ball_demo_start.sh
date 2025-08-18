#!/bin/bash

echo "🏀 Ball Tracking Demo Quick Start"
echo "================================="

# Check if virtual environment exists
if [ ! -d "tennis_env" ]; then
    echo "🔧 Virtual environment not found. Running setup first..."
    ./setup_and_run.sh
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source tennis_env/bin/activate

# Check if tennis_test.mp4 exists
if [ -f "tennis_test.mp4" ]; then
    echo "🏀 Found tennis_test.mp4 - starting ball tracking..."
    echo ""
    echo "🎮 Controls:"
    echo "   - Press 'q' to quit"
    echo "   - Press 'p' to pause/resume"
    echo "   - Press 's' to save current frame"
    echo ""
    echo "🎯 Visualization:"
    echo "   - 🟡 Yellow circles: Combined predictions"
    echo "   - 🔴 Red circles: TrackNet predictions"
    echo "   - 🔵 Blue circles: YOLO predictions"
    echo "   - 🟢 Green arrows: Velocity vectors"
    echo "   - 🟡 Yellow lines: Ball trajectory"
    echo ""
    
    # Run the ball tracking demo
    python3 ball_demo.py --video tennis_test.mp4
else
    echo "❌ tennis_test.mp4 not found!"
    echo "Available video files:"
    for video_file in *.mp4; do
        if [ -f "$video_file" ]; then
            echo "  - $video_file"
        fi
    done
    echo ""
    echo "Please specify a video file:"
    echo "  python3 ball_demo.py --video your_video.mp4"
fi
