#!/bin/bash

# Bounce Detection Debug Demo Startup Script
# This script launches the enhanced bounce detection demo with debugging capabilities

echo "🎾 Bounce Detection Debug Demo Startup"
echo "======================================"

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "❌ Python3 is not installed or not in PATH"
    exit 1
fi

# Check if bounce_demo_debug.py exists
if [ ! -f "bounce_demo_debug.py" ]; then
    echo "❌ bounce_demo_debug.py not found"
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
echo "🚀 Launching bounce detection debug demo..."
echo ""
echo "Enhanced Controls:"
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
echo "💡 Debug Features:"
echo "  - Real-time confidence threshold adjustment (0.01 increments)"
echo "  - Quick threshold presets (0, 1, 2 keys)"
echo "  - Red dots show ALL potential bounces (confidence > 0.05)"
echo "  - Feature value display for debugging"
echo "  - Confidence distribution analysis"
echo "  - Enhanced ball detection with multiple color ranges"
echo ""

# Run the bounce detection debug demo
python3 bounce_demo_debug.py --video tennis_test5.mp4 --model models/bounce_detector.cbm

echo ""
echo "🎾 Bounce detection debug demo completed"
