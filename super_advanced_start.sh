#!/bin/bash

echo "🎾🏟️ SUPER ADVANCED TENNIS ANALYSIS ENGINE"
echo "=========================================="
echo "🚀 The Future of Tennis Analytics is Here!"
echo ""

# Check if virtual environment exists
if [ ! -d "tennis_env" ]; then
    echo "🔧 Virtual environment not found. Running setup first..."
    ./setup_and_run.sh
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source tennis_env/bin/activate

# Check for required models
echo "🔍 Checking for required AI models..."
models_found=0
total_models=6

# Check each required model
if [ -f "models/playersnball4.pt" ]; then
    echo "  ✅ Player Detection Model (YOLOv8)"
    ((models_found++))
else
    echo "  ❌ Player Detection Model - MISSING"
fi

if [ -f "models/yolov8n-pose.pt" ]; then
    echo "  ✅ Pose Estimation Model (YOLOv8-pose)"
    ((models_found++))
else
    echo "  ❌ Pose Estimation Model - MISSING"
fi

if [ -f "models/bounce_detector.cbm" ]; then
    echo "  ✅ Bounce Detection Model (CatBoost)"
    ((models_found++))
else
    echo "  ❌ Bounce Detection Model - MISSING"
fi

if [ -f "pretrained_ball_detection.pt" ]; then
    echo "  ✅ Ball Tracking Model (TrackNet)"
    ((models_found++))
else
    echo "  ❌ Ball Tracking Model - MISSING"
fi

if [ -f "model_tennis_court_det.pt" ]; then
    echo "  ✅ Court Detection Model (Deep Learning)"
    ((models_found++))
else
    echo "  ❌ Court Detection Model - MISSING"
fi

if [ -d "TennisCourtDetector" ]; then
    echo "  ✅ TennisCourtDetector Library"
    ((models_found++))
else
    echo "  ❌ TennisCourtDetector Library - MISSING"
fi

echo ""
echo "📊 Model Status: ${models_found}/${total_models} models found"

if [ $models_found -lt $total_models ]; then
    echo ""
    echo "⚠️  WARNING: Some models are missing!"
    echo "   The system will run with reduced functionality."
    echo "   For full functionality, ensure all models are available."
    echo ""
fi

# Check if tennis_test.mp4 exists
if [ -f "tennis_test.mp4" ]; then
    echo "🎾 Found tennis_test.mp4 - ready to launch super advanced analysis!"
    echo ""
    echo "🚀 LAUNCHING SUPER ADVANCED TENNIS ANALYSIS ENGINE..."
    echo ""
    echo "🎮 Controls:"
    echo "   - Press 'q' to quit"
    echo "   - Press 'p' to pause/resume"
    echo "   - Press 's' to save current frame"
    echo ""
    echo "🔬 INTEGRATED SYSTEMS:"
    echo "   ✅ Player Detection (YOLOv8)"
    echo "   ✅ Pose Estimation (YOLOv8-pose)"
    echo "   ✅ Ball Tracking (TrackNet + YOLO)"
    echo "   ✅ Bounce Detection (CatBoost)"
    echo "   ✅ Court Detection (Deep Learning + Geometric Validation)"
    echo ""
    
    # Run the super advanced demo
    python3 tennis_analysis_demo.py --video tennis_test.mp4
else
    echo "❌ tennis_test.mp4 not found!"
    echo ""
    echo "Available video files:"
    for video_file in *.mp4; do
        if [ -f "$video_file" ]; then
            echo "  - $video_file"
        fi
    done
    echo ""
    echo "🎯 To run with a custom video:"
    echo "   python3 tennis_analysis_demo.py --video your_video.mp4"
    echo ""
    echo "💾 To save output video:"
    echo "   python3 tennis_analysis_demo.py --video your_video.mp4 --output analyzed_output.mp4"
    echo ""
    echo "⚙️  To use custom configuration:"
    echo "   python3 tennis_analysis_demo.py --config your_config.yaml"
fi

echo ""
echo "🎾🏟️ SUPER ADVANCED TENNIS ANALYSIS ENGINE - READY FOR THE FUTURE! 🚀"
