# 🎾 Tennis Analytics System

A comprehensive computer vision system for analyzing tennis matches using state-of-the-art AI models for player detection, pose estimation, and ball tracking.

## 🚀 Features

- **Player Detection**: YOLOv8-based detection of tennis players
- **Pose Estimation**: YOLOv11-pose analysis of player body positions and swing mechanics
- **Ball Tracking**: TrackNet-based precise ball trajectory analysis
- **Swing Analysis**: Automatic classification of swing phases (backswing, contact, follow-through)
- **Real-time Processing**: Live video analysis with optional recording
- **Comprehensive Analytics**: Detailed statistics and insights

## 🏗️ Architecture

The system is built with a modular architecture:

```
src/
├── player_detector.py      # YOLOv8 player detection
├── pose_estimator.py       # YOLOv11-pose estimation
├── ball_tracker.py         # TrackNet ball tracking
└── tennis_analyzer.py      # Main orchestrator
```

## 📋 Requirements

- Python 3.8+
- PyTorch 2.0+
- TensorFlow 2.13+
- OpenCV 4.8+
- CUDA-compatible GPU (recommended)

## 🛠️ Installation

1. **Clone the repository**:
```bash
git clone <repository-url>
cd yolo_tennis_training
```

2. **Create virtual environment**:
```bash
python -m venv tennis_env
source tennis_env/bin/activate  # On Windows: tennis_env\Scripts\activate
```

3. **Install dependencies**:
```bash
pip install -r requirements.txt
```

4. **Setup project structure**:
```bash
python main.py --setup
```

5. **Download required models**:
   - **YOLOv8n.pt**: [Download](https://github.com/ultralytics/assets/releases)
   - **YOLOv11-pose.pt**: [Download](https://github.com/ultralytics/assets/releases)  
   - **TrackNet.h5**: [Download](https://github.com/yu4u/tracknet)

   Place all models in the `models/` directory.

## 🎯 Usage

### Basic Video Analysis

```bash
python main.py --video path/to/tennis_video.mp4
```

### Save Annotated Video

```bash
python main.py --video path/to/tennis_video.mp4 --save-video
```

### Custom Configuration

```bash
python main.py --video path/to/tennis_video.mp4 --config custom_config.yaml
```

### Command Line Options

- `--video, -v`: Input video file path
- `--config, -c`: Configuration file (default: config.yaml)
- `--output, -o`: Output results file (default: data/output/analysis_results.json)
- `--save-video`: Save annotated video output
- `--no-display`: Disable real-time display
- `--setup`: Initialize project directories
- `--verbose`: Enable detailed logging

## ⚙️ Configuration

The system is configured via `config.yaml`:

```yaml
# Model paths and settings
models:
  yolo_player: "models/yolov8n.pt"
  yolo_pose: "models/yolov11-pose.pt"
  tracknet: "models/tracknet.h5"

# Analysis parameters
analysis:
  swing_detection:
    min_confidence: 0.7
  ball_tracking:
    min_trajectory_length: 10
```

## 📊 Output

The system generates comprehensive analytics:

### Frame-level Analysis
- Player detections with bounding boxes
- Pose keypoints and skeleton visualization
- Ball positions and trajectories
- Swing phase classification

### Session Summary
- Total frames processed
- Detection rates for players, poses, and ball
- Swing phase distribution
- Ball trajectory statistics
- Processing performance metrics

### Data Export
- JSON format analysis results
- Optional annotated video output
- Detailed trajectory data
- Swing mechanics analysis

## 🔧 Customization

### Adding New Models
1. Implement model interface in appropriate module
2. Update configuration file
3. Modify `TennisAnalyzer._initialize_components()`

### Custom Analytics
1. Extend analysis methods in respective modules
2. Add new metrics to summary statistics
3. Update visualization functions

### Performance Tuning
- Adjust confidence thresholds in config
- Modify frame processing parameters
- Optimize model inference settings

## 🧪 Testing

### Test with Sample Video
```bash
# Download sample tennis video
wget <sample_video_url> -O data/raw_videos/sample.mp4

# Run analysis
python main.py --video data/raw_videos/sample.mp4 --save-video
```

### Validate Components
```bash
# Test individual modules
python -c "from src.player_detector import PlayerDetector; print('Player detector OK')"
python -c "from src.pose_estimator import PoseEstimator; print('Pose estimator OK')"
python -c "from src.ball_tracker import TrackNet; print('Ball tracker OK')"
```

## 📈 Performance

### Hardware Requirements
- **Minimum**: CPU-only processing (slower)
- **Recommended**: GPU with CUDA support
- **Optimal**: RTX 3080+ or equivalent

### Expected Performance
- **Player Detection**: 30+ FPS
- **Pose Estimation**: 15-25 FPS  
- **Ball Tracking**: 20-30 FPS
- **Overall System**: 10-20 FPS (bottlenecked by slowest component)

## 🐛 Troubleshooting

### Common Issues

1. **Model not found errors**:
   - Ensure models are downloaded to `models/` directory
   - Check file permissions and paths

2. **CUDA/GPU errors**:
   - Verify PyTorch/TensorFlow GPU installation
   - Check CUDA compatibility

3. **Memory issues**:
   - Reduce batch sizes in config
   - Process lower resolution videos
   - Use CPU-only mode if needed

### Debug Mode
```bash
python main.py --video video.mp4 --verbose
```

## 🤝 Contributing

1. Fork the repository
2. Create feature branch
3. Implement changes with tests
4. Submit pull request

## 📄 License

This project is licensed under the MIT License - see LICENSE file for details.

## 🙏 Acknowledgments

- [Ultralytics](https://github.com/ultralytics) for YOLOv8 and YOLOv11
- [TrackNet](https://github.com/yu4u/tracknet) for ball tracking
- OpenCV and PyTorch communities

## 📞 Support

For questions and support:
- Open an issue on GitHub
- Check the troubleshooting section
- Review configuration examples

---

**Happy Tennis Analysis! 🎾📊**
