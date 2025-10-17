# Core System Components

Main system components that orchestrate the tennis analysis pipeline.

## Modules

### tennis_master.py
**Master controller for the tennis analysis system**

Orchestrates the entire analysis pipeline by:
- Launching `tennis_CV.py` for video processing
- Launching `tennis_analytics.py` for analytics visualization
- Managing inter-process communication
- Coordinating data flow between components

**Usage:**
```bash
python src/core/tennis_master.py --video data/raw/tennis_test5.mp4
```

**Features:**
- Parallel process management
- Socket-based communication
- Error handling and recovery
- Graceful shutdown

### tennis_CV.py
**Main computer vision processing engine** (3000+ lines)

The core analysis engine that integrates:
- **Player detection** - YOLOv8 and RF-DETR for player tracking
- **Pose estimation** - YOLOv8-Pose for player keypoints
- **Ball tracking** - TrackNet and YOLO for ball detection
- **Bounce detection** - ML-based bounce identification
- **Court detection** - Court keypoint detection and homography

**Features:**
- Multi-model ensemble detection
- Real-time video processing
- Frame-by-frame analysis
- CSV output generation
- Video annotation and visualization

**Usage:**
```bash
python src/core/tennis_CV.py \
    --video data/raw/tennis_test5.mp4 \
    --config config.yaml \
    --output outputs/videos/analysis.mp4
```

### tennis_analytics.py
**Analytics viewer and visualization**

Provides interactive visualization of analysis results:
- Reads processed CSV data
- Displays player positions
- Shows ball trajectories
- Highlights bounces
- Shot classification overlay
- Statistics and metrics

**Usage:**
```bash
python src/core/tennis_analytics.py \
    --csv data/processed/csv/tennis_analysis_data.csv \
    --video data/raw/tennis_test5.mp4 \
    --output outputs/videos/analytics_viz.mp4
```

## System Architecture

```
tennis_master.py (orchestrator)
    ├── tennis_CV.py (processing)
    │   ├── Player Detection
    │   ├── Pose Estimation  
    │   ├── Ball Tracking
    │   ├── Bounce Detection
    │   └── Court Detection
    │       └── Outputs: CSV data
    │
    └── tennis_analytics.py (visualization)
        └── Reads: CSV data
        └── Outputs: Annotated video
```

## Configuration

All core modules use `config.yaml` for configuration:
- Model paths
- Detection thresholds
- Video processing settings
- Output settings

## Data Flow

1. **Input**: Raw video from `data/raw/`
2. **Processing**: `tennis_CV.py` analyzes each frame
3. **Output**: CSV data to `data/processed/csv/`
4. **Visualization**: `tennis_analytics.py` creates annotated video
5. **Final Output**: Videos to `outputs/videos/`

## Dependencies

External libraries integrated:
- `external/rf-detr/` - Enhanced detection
- `external/TennisCourtDetector/` - Court detection
- `external/TrackNet/` - Ball tracking

## Related

- **Detection modules**: `src/detection/`
- **Analysis modules**: `src/analysis/`
- **Configuration**: `config.yaml`
- **Models**: `models/`

