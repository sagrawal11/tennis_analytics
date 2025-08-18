#!/bin/bash

echo "🎾 Tennis Analysis Demo Setup"
echo "=============================="

# Check if Python 3 is available
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed. Please install Python 3.8+ first."
    exit 1
fi

echo "✅ Python 3 found: $(python3 --version)"

# Check if pip is available
if ! command -v pip3 &> /dev/null; then
    echo "❌ pip3 is not installed. Please install pip first."
    exit 1
fi

echo "✅ pip3 found: $(pip3 --version)"

# Create virtual environment if it doesn't exist
if [ ! -d "tennis_env" ]; then
    echo "🔧 Creating virtual environment..."
    python3 -m venv tennis_env
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source tennis_env/bin/activate

# Upgrade pip
echo "🔧 Upgrading pip..."
pip install --upgrade pip

# Install requirements
echo "🔧 Installing dependencies..."
pip install -r requirements.txt

echo ""
echo "✅ Setup complete! Virtual environment 'tennis_env' is ready."
echo ""
echo "🎯 To run the demo:"
echo "   1. Activate the environment: source tennis_env/bin/activate"
echo "   2. Run the demo: python3 tennis_analysis_demo.py"
echo ""
echo "🎯 Or use this shortcut:"
echo "   ./tennis_env/bin/python tennis_analysis_demo.py"
echo ""
echo "🎯 Available options:"
echo "   - Default: python3 tennis_analysis_demo.py"
echo "   - Custom video: python3 tennis_analysis_demo.py --video your_video.mp4"
echo "   - Save output: python3 tennis_analysis_demo.py --output analyzed_video.mp4"
echo "   - Custom config: python3 tennis_analysis_demo.py --config your_config.yaml"
echo ""
echo "🎮 Controls during playback:"
echo "   - Press 'q' to quit"
echo "   - Press 'p' to pause/resume"
echo "   - Press 's' to save current frame"
echo ""
