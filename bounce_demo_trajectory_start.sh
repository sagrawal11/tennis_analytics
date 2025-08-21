#!/bin/bash

echo "🎾 Starting Trajectory-Based Bounce Detection Demo"
echo "=================================================="
echo ""
echo "This version focuses on SHARP TRAJECTORY ANGLE CHANGES"
echo "when the ball is AWAY FROM PLAYERS to detect ground bounces."
echo ""
echo "Key Features:"
echo "• Analyzes ball trajectory over multiple frames"
echo "• Detects sharp angle changes (>45°) in trajectory"
echo "• Only considers bounces when ball is away from players"
echo "• Filters out serves, volleys, and player interactions"
echo "• Uses your working ball tracking system"
echo ""
echo "Controls:"
echo "• 't' - Toggle trajectory display"
echo "• 'p' - Toggle physics info display" 
echo "• 'c' - Toggle confidence display"
echo "• 'q' - Quit"
echo "• SPACE - Pause/Resume"
echo ""
echo "Red dots = Detected bounces"
echo "Green lines = Ball trajectory"
echo "Blue dots = Ball positions over time"
echo ""

# Check if video file is provided
if [ $# -eq 0 ]; then
    echo "❌ Error: No video file specified"
    echo "Usage: $0 <video_file>"
    echo "Example: $0 tennis_test5.mp4"
    exit 1
fi

VIDEO_FILE="$1"

# Check if video file exists
if [ ! -f "$VIDEO_FILE" ]; then
    echo "❌ Error: Video file '$VIDEO_FILE' not found"
    exit 1
fi

echo "📹 Video file: $VIDEO_FILE"
echo "🚀 Starting demo..."
echo ""

# Activate virtual environment and run
source tennis_env/bin/activate
python3 bounce_demo_trajectory.py --video "$VIDEO_FILE"

echo ""
echo "✅ Demo completed!"
