#!/bin/bash

# Bounce Detection Demo Startup Script
# This script launches the bounce detection demo with tennis_test5.mp4

echo "🎾 Bounce Detection Demo Startup"
echo "================================="

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "❌ Python3 is not installed or not in PATH"
    exit 1
fi

# Check if bounce_demo.py exists
if [ ! -f "bounce_demo.py" ]; then
    echo "❌ bounce_demo.py not found"
    exit 1
fi

# Check if tennis_test5.mp4 exists
if [ ! -f "tennis_test5.mp4" ]; then
    echo "❌ tennis_test5.mp4 not found"
    echo "Available video files:"
    ls *.mp4 2>/dev/null || echo "No MP4 files found"
    exit 1
fi

# Check if bounce_detector.cbm exists
if [ ! -f "models/bounce_detector.cbm" ]; then
    echo "❌ models/bounce_detector.cbm not found"
    echo "Available model files:"
    ls models/*.cbm 2>/dev/null || echo "No .cbm files found in models/"
    exit 1
fi

echo "✅ All required files found"
echo "🚀 Launching bounce detection demo..."
echo ""
echo "Controls:"
echo "  't' - Toggle trajectory display"
echo "  'f' - Toggle feature display"
echo "  'c' - Toggle confidence display"
echo "  SPACE - Pause/Resume"
echo "  'q' - Quit"
echo ""

# Run the bounce detection demo
python3 bounce_demo.py --video tennis_test5.mp4 --model models/bounce_detector.cbm

echo ""
echo "🎾 Bounce detection demo completed"
