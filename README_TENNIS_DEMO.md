# 🎾 Tennis Analysis Demo

A comprehensive tennis analytics engine that integrates **player detection**, **pose estimation**, and **ball bounce detection** for real-time tennis video analysis.

## ✨ Features

- **🎯 Player Detection**: YOLOv8-based detection of tennis players and balls
- **🧍 Pose Estimation**: Full-body pose analysis using YOLOv8-pose
- **🏀 Ball Bounce Detection**: CatBoost-based bounce detection with confidence scoring
- **📊 Real-time Visualization**: Live overlay of all detections and analysis
- **💾 Video Recording**: Save analyzed videos with annotations
- **📸 Frame Capture**: Save individual frames for detailed analysis

## 🚀 Quick Start

### Option 1: Automated Setup & Run
```bash
./quick_start.sh
```

### Option 2: Manual Setup
```bash
# 1. Run setup script
./setup_and_run.sh

# 2. Activate virtual environment
source tennis_env/bin/activate

# 3. Run the demo
python3 tennis_analysis_demo.py
```

## 📋 Requirements

- **Python 3.8+**
- **OpenCV 4.8+**
- **PyTorch 1.13+**
- **Ultralytics 8.0+**
- **CatBoost 1.2+**

## 🎮 Controls

During video playback:
- **`q`** - Quit the application
- **`p`** - Pause/Resume playback
- **`s`** - Save current frame as image

## 🔧 Configuration

The system uses `config.yaml` for model paths and parameters:

```yaml
models:
  yolo_player: "models/playersnball4.pt"      # Player detection model
  yolo_pose: "models/yolov8n-pose.pt"         # Pose estimation model
  bounce_detector: "models/bounce_detector.cbm" # Bounce detection model

yolo_player:
  conf_threshold: 0.5    # Detection confidence threshold
  iou_threshold: 0.45    # IoU threshold for NMS
  max_det: 10            # Maximum detections per frame

yolo_pose:
  conf_threshold: 0.3    # Pose confidence threshold
  keypoints: 17          # Number of COCO keypoints
```

## 📁 Project Structure

```
tennis_analytics/
├── tennis_analysis_demo.py    # Main demo script
├── requirements.txt            # Python dependencies
├── setup_and_run.sh           # Setup script
├── quick_start.sh             # Quick start script
├── config.yaml                # Configuration file
├── models/                    # Trained models
│   ├── playersnball4.pt      # Player detection model
│   ├── yolov8n-pose.pt       # Pose estimation model
│   └── bounce_detector.cbm   # Bounce detection model
├── tennis_test.mp4            # Test video
└── README_TENNIS_DEMO.md     # This file
```

## 🎯 Usage Examples

### Basic Analysis
```bash
python3 tennis_analysis_demo.py
```

### Custom Video
```bash
python3 tennis_analysis_demo.py --video your_video.mp4
```

### Save Output Video
```bash
python3 tennis_analysis_demo.py --video tennis_test.mp4 --output analyzed_output.mp4
```

### Custom Configuration
```bash
python3 tennis_analysis_demo.py --config my_config.yaml
```

## 🔍 What You'll See

### Player Detection
- **Green boxes** around detected players
- **Blue boxes** around detected balls
- Confidence scores for each detection

### Pose Estimation
- **Yellow skeleton** connecting body keypoints
- **Yellow dots** for individual keypoints
- Pose confidence scores

### Ball Bounce Detection
- **Red "BOUNCE!" indicator** when bounce detected
- **Red circle** in top-left corner
- Bounce probability score

### Statistics Overlay
- Frame counter
- Total players detected
- Total poses estimated
- Total bounces detected

## 🛠️ Troubleshooting

### Common Issues

1. **"Ultralytics not installed"**
   ```bash
   pip install ultralytics
   ```

2. **"CatBoost not installed"**
   ```bash
   pip install catboost
   ```

3. **"OpenCV not found"**
   ```bash
   pip install opencv-python
   ```

4. **Model loading errors**
   - Check if model files exist in `models/` directory
   - Verify file permissions
   - Ensure models are compatible with current versions

### Performance Tips

- **Frame skipping**: Adjust `frame_skip` in config for faster processing
- **Confidence thresholds**: Lower thresholds for more detections, higher for accuracy
- **Model selection**: Use smaller models (e.g., `yolov8n`) for faster inference

## 🔬 Technical Details

### Player Detection
- **Model**: YOLOv8 custom-trained on tennis data
- **Classes**: Players (class 1), Balls (class 0)
- **Output**: Bounding boxes, confidence scores, class labels

### Pose Estimation
- **Model**: YOLOv8-pose (17 COCO keypoints)
- **Keypoints**: Nose, eyes, ears, shoulders, elbows, wrists, hips, knees, ankles
- **Output**: 2D keypoint coordinates with confidence scores

### Bounce Detection
- **Model**: CatBoost classifier
- **Features**: Histogram, edge density, gradient magnitude
- **Output**: Bounce probability (0.0 - 1.0)

## 📈 Future Enhancements

- **Court detection** and homography
- **Ball trajectory tracking**
- **Swing phase analysis**
- **Performance metrics** (serve speed, ball spin)
- **Multi-camera support**
- **Real-time streaming** support

## 🤝 Contributing

This is a demo system showcasing the integration of multiple AI models for tennis analysis. Feel free to:

- Improve feature extraction for bounce detection
- Add new analysis components
- Optimize performance
- Enhance visualization

## 📄 License

This project is for educational and research purposes. Please respect the licenses of the underlying models and libraries used.

---

**Happy Tennis Analysis! 🎾📊**
