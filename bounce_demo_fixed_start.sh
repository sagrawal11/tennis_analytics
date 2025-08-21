#!/bin/bash

# Fixed Bounce Detection Demo Startup Script
# This version uses the working ball tracking from tennis_CV.py

echo "🎾 Fixed Bounce Detection Demo Startup"
echo "======================================"

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "❌ Python3 is not installed or not in PATH"
    exit 1
fi

# Check if bounce_demo_fixed.py exists
if [ ! -f "bounce_demo_fixed.py" ]; then
    echo "❌ bounce_demo_fixed.py not found"
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
echo "🚀 Launching fixed bounce detection demo..."
echo ""
echo "This version uses the WORKING ball tracking from tennis_CV.py"
echo "So it should detect actual ball movement and provide proper trajectory data!"
echo ""
echo "Controls:"
echo "  't' - Toggle trajectory display"
echo "  'f' - Toggle feature display"
echo "  'c' - Toggle confidence display"
echo "  'd' - Toggle debug info (features)"
echo "  '+' - Increase confidence threshold (0.01)"
echo "  '-' - Decrease confidence threshold (0.01)"
echo "  '0' - Reset threshold to 0.0 (most sensitive)"
echo "  '1' - Set threshold to 0.1"
echo "  '2' - Set threshold to 0.2"
echo "  'a' - Toggle show all predictions"
echo "  SPACE - Pause/Resume"
echo "  'q' - Quit"
echo ""
echo "💡 Key Improvements:"
echo "  - Uses working ball tracking from tennis_CV.py"
echo "  - RF-DETR + YOLO + TrackNet integration"
echo "  - Proper ball trajectory data for bounce detection"
echo "  - Red dots show potential bounces with confidence"
echo ""

# Run the fixed bounce detection demo
python3 bounce_demo_fixed.py --video tennis_test5.mp4 --model models/bounce_detector.cbm

echo ""
echo "🎾 Fixed bounce detection demo completed"
