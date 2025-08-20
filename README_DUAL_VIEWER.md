# 🎾 Tennis Analysis Dual-Viewer System

This is a sophisticated tennis analysis system that provides **two synchronized viewers** for comprehensive tennis video analysis:

1. **CV Viewer** (`tennis_CV.py`) - Shows the original video with all computer vision overlays
2. **Analytics Viewer** (`tennis_analytics.py`) - Shows only the overlays moving on a clean court background

## 🚀 Quick Start

### Option 1: Easy Startup Script
```bash
./start_tennis_analysis.sh
```

### Option 2: Manual Launch
```bash
python3 tennis_master.py --video your_tennis_video.mp4
```

## 📺 What You'll See

When you run the system, **two windows will open simultaneously**:

### 1. Super Advanced Tennis Analysis Engine (CV Viewer)
- **Full video playback** with all overlays
- **Real-time computer vision** processing
- **Player detection** with bounding boxes
- **Pose estimation** with skeleton overlays
- **Ball tracking** with trajectory visualization
- **Court detection** with keypoint overlays
- **Bounce detection** indicators

### 2. Tennis Analytics Viewer (Analytics Viewer)
- **Clean court background** (no video)
- **Mirrored overlays** from the CV system
- **Ball trajectory** with fading trails
- **Player positions** and movements
- **Bounce events** with visual indicators
- **Real-time analytics panel** with statistics
- **Performance metrics** (FPS, detection counts)

## 🎮 Controls

### CV Viewer Controls
- **`q`** - Quit the viewer
- **`p`** - Pause/Resume video
- **`s`** - Save current frame

### Analytics Viewer Controls
- **`q`** - Quit the viewer
- **`t`** - Toggle ball trajectories
- **`a`** - Toggle analytics panel
- **`h`** - Toggle heatmap (future feature)

### Master Controller
- **`Ctrl+C`** - Gracefully shutdown both viewers

## 🔧 System Architecture

```
tennis_master.py (Master Controller)
├── tennis_CV.py (CV Viewer)
│   ├── Player Detection
│   ├── Pose Estimation
│   ├── Ball Tracking
│   ├── Court Detection
│   └── Bounce Detection
└── tennis_analytics.py (Analytics Viewer)
    ├── Overlay Mirroring
    ├── Trajectory Visualization
    ├── Analytics Dashboard
    └── Performance Metrics
```

## 📊 Analytics Features

The Analytics Viewer provides:

### Real-time Statistics
- Frame counter
- Player detection count
- Pose estimation count
- Ball detection count
- Court detection count
- Bounce detection count
- Average processing FPS

### Visual Analytics
- **Ball Trajectories**: Fading trails showing ball movement
- **Player Movements**: Real-time player position tracking
- **Court Overlay**: Clean tennis court visualization
- **Bounce Events**: Visual indicators for ball bounces
- **Performance Metrics**: Real-time FPS and processing stats

## 🛠️ Technical Details

### Data Flow
1. **CV Viewer** processes video frames and extracts data
2. **Master Controller** coordinates data flow between viewers
3. **Analytics Viewer** receives data and renders clean overlays

### Coordinate Scaling
The Analytics Viewer automatically scales video coordinates to fit the court visualization, providing a clean, abstract view of the tennis action.

### Performance Optimization
- **Frame skipping** for real-time processing
- **Efficient data structures** for smooth visualization
- **Background processing** to maintain UI responsiveness

## 📁 File Structure

```
tennis_analytics/
├── tennis_master.py          # Master controller
├── tennis_CV.py             # CV viewer (renamed from tennis_analysis_demo.py)
├── tennis_analytics.py      # Analytics viewer
├── start_tennis_analysis.sh # Easy startup script
├── config.yaml              # Configuration file
├── README_DUAL_VIEWER.md    # This file
└── [video files].mp4        # Your tennis videos
```

## 🎯 Use Cases

### For Coaches
- **Clean visualization** of player movements without video distraction
- **Ball trajectory analysis** for technique improvement
- **Real-time statistics** during training sessions

### For Players
- **Movement analysis** with pose estimation overlays
- **Shot placement** visualization
- **Performance tracking** with analytics

### For Analysts
- **Data extraction** for further analysis
- **Pattern recognition** in player movements
- **Statistical analysis** of game performance

## 🔮 Future Enhancements

### Planned Features
- **Heatmap visualization** of player movements
- **Shot type classification** (forehand, backhand, serve, volley)
- **Advanced analytics** (serve speed, ball spin, etc.)
- **Export capabilities** for data analysis
- **Multi-camera support** for 3D analysis

### Analytics Improvements
- **Machine learning** shot prediction
- **Player fatigue** analysis
- **Tactical pattern** recognition
- **Performance benchmarking** against historical data

## 🐛 Troubleshooting

### Common Issues

**"Required file not found"**
- Ensure all Python files are in the same directory
- Check that `config.yaml` exists

**"No MP4 video files found"**
- Place your tennis video files in the project directory
- Ensure files have `.mp4` extension

**Viewers don't start**
- Check Python installation: `python3 --version`
- Verify OpenCV installation: `python3 -c "import cv2; print(cv2.__version__)"`
- Check system requirements in main README

**Poor performance**
- Reduce video resolution
- Increase frame skip in `config.yaml`
- Close other applications to free up resources

## 📞 Support

For issues or questions:
1. Check the main project README for system requirements
2. Verify all dependencies are installed
3. Check the troubleshooting section above
4. Review the configuration options in `config.yaml`

---

**🎾 Enjoy your enhanced tennis analysis experience!**
