# Tennis Analytics - Project Structure

**Last Updated:** October 17, 2025

## Overview

This document describes the reorganized structure of the Tennis Analytics project, a production-grade multi-AI computer vision system for analyzing tennis match videos.

---

## 📁 Directory Structure

```
tennis_analytics/
├── config.yaml                    # Main configuration file
├── start_tennis_analysis.sh       # Quick start script
├── requirements.txt               # Python dependencies
├── .gitignore                     # Git ignore rules
│
├── src/                          # Source code
│   ├── __init__.py
│   ├── core/                     # Core system components
│   │   ├── __init__.py
│   │   ├── tennis_master.py      # Master controller (orchestrates system)
│   │   ├── tennis_CV.py          # CV processing engine (3000+ lines)
│   │   └── tennis_analytics.py   # Analytics viewer
│   │
│   ├── detection/                # Detection modules
│   │   ├── __init__.py
│   │   ├── players/              # Player detection
│   │   │   ├── __init__.py
│   │   │   └── player_detector.py
│   │   └── bounce/               # Bounce detection
│   │       ├── __init__.py
│   │       ├── base_detector.py
│   │       ├── ensemble_detector.py
│   │       ├── hybrid_detector.py
│   │       ├── sequence_detector.py
│   │       ├── simple_ultra_detector.py
│   │       └── optimized_ultra_detector.py
│   │
│   ├── analysis/                 # Analysis modules
│   │   ├── __init__.py
│   │   ├── tennis_shot_classifier.py    # Shot classification
│   │   └── tennis_data_aggregator.py    # Data aggregation
│   │
│   ├── training/                 # Training scripts
│   │   ├── __init__.py
│   │   ├── bounce/               # Bounce detection training
│   │   │   ├── __init__.py
│   │   │   ├── improved_bounce_trainer.py
│   │   │   ├── tennis_bounce_ml_trainer.py
│   │   │   ├── train_bounce_model.py
│   │   │   ├── ultra_advanced_trainer.py
│   │   │   ├── lstm_bounce_trainer.py
│   │   │   ├── sequence_bounce_trainer.py
│   │   │   ├── improved_annotation_strategy.py
│   │   │   └── generate_annotation_template.py
│   │   │
│   │   └── shot/                 # Shot classification training
│   │       └── __init__.py
│   │
│   ├── evaluation/               # Evaluation scripts
│   │   ├── __init__.py
│   │   ├── evaluate_simple_detector.py
│   │   ├── evaluate_sequence_detector.py
│   │   ├── evaluate_optimized_detector.py
│   │   └── debug_feature_extraction.py
│   │
│   └── utils/                    # Utility functions
│       └── __init__.py
│
├── models/                       # AI/ML Models
│   ├── player/                   # Player detection models
│   │   ├── playersnball4.pt      # YOLOv8 player+ball model
│   │   └── playersnball5.pt      # Enhanced YOLOv8 model
│   │
│   ├── pose/                     # Pose estimation models
│   │   ├── yolov8n-pose.pt       # Nano pose model
│   │   └── yolov8x-pose.pt       # Extra-large pose model
│   │
│   ├── ball/                     # Ball tracking models
│   │   ├── pretrained_ball_detection.pt  # TrackNet ball model
│   │   └── tracknet.h5           # TrackNet H5 format
│   │
│   ├── court/                    # Court detection models
│   │   └── model_tennis_court_det.pt
│   │
│   ├── bounce/                   # Bounce detection models
│   │   └── bounce_detector.cbm   # CatBoost bounce classifier
│   │
│   ├── rf-detr/                  # RF-DETR models
│   │   └── rf-detr-base.pth
│   │
│   ├── advanced/                 # Advanced/trained models
│   │   ├── feature_names.json
│   │   ├── logistic_regression_model.joblib
│   │   ├── random_forest_model.joblib
│   │   ├── svm_model.joblib
│   │   └── training_info.json
│   │
│   ├── sequence/                 # Sequence models
│   │   └── (sequence model files)
│   │
│   └── ultra/                    # Ultra models
│       └── (ultra model files)
│
├── data/                         # Data files
│   ├── raw/                      # Raw input videos
│   │   ├── tennis_test1.mp4
│   │   ├── tennis_test2.mp4
│   │   ├── tennis_test3.mp4
│   │   ├── tennis_test4.mp4
│   │   ├── tennis_test5.mp4
│   │   ├── tennis_ball.mp4
│   │   └── tennis_ball2.mp4
│   │
│   ├── processed/                # Processed data
│   │   └── csv/                  # CSV outputs
│   │       ├── tennis_analysis_data.csv
│   │       ├── tennis_ball_results.csv
│   │       └── tennis_ball_test.csv
│   │
│   ├── annotations/              # Annotation files
│   │   ├── bounce/               # Bounce annotations
│   │   │   ├── tennis_test1_annotations.csv
│   │   │   ├── tennis_test2_annotations.csv
│   │   │   ├── tennis_test3_annotations.csv
│   │   │   ├── tennis_test4_annotations.csv
│   │   │   ├── tennis_test5_annotations.csv
│   │   │   ├── all_bounce_annotations.csv
│   │   │   └── improved_bounce_annotations.csv
│   │   │
│   │   └── shot/                 # Shot annotations
│   │       └── tennis_shot_features.csv
│   │
│   ├── training/                 # Training data
│   │   └── bounce/
│   │       ├── advanced_bounce_training_data.csv
│   │       ├── advanced_bounce_train.csv
│   │       ├── advanced_bounce_test.csv
│   │       ├── bounce_training_data.csv
│   │       ├── bounce_training_data_high_quality.csv
│   │       ├── bounce_training_data_high_quality_train.csv
│   │       ├── bounce_training_data_high_quality_test.csv
│   │       └── feature_names/
│   │           └── (feature name JSON files)
│   │
│   ├── ball_coordinates/         # Ball coordinate files
│   │   ├── ball_coords_test1.csv
│   │   ├── ball_coords_test2.csv
│   │   ├── ball_coords_test3.csv
│   │   ├── ball_coords_test4.csv
│   │   ├── ball_coords_test5.csv
│   │   └── all_ball_coordinates.csv
│   │
│   └── court/                    # Court-specific data
│       └── tennis_test3_court_keypoints.csv
│
├── outputs/                      # Generated outputs
│   ├── videos/                   # Output videos
│   │   ├── tennis_analysis_output.mp4
│   │   ├── tennis_analytics_output.mp4
│   │   └── (other generated videos)
│   │
│   └── results/                  # Analysis results
│       └── optimized_detector_results.json
│
├── external/                     # External libraries
│   ├── TrackNet/                 # Ball tracking library
│   ├── TennisCourtDetector/      # Court detection library
│   ├── rf-detr/                  # RF-DETR detection transformer
│   ├── TRACE/                    # TRACE library
│   └── AI-Tennis-Ball-Bounce-Detection/  # Research notebooks
│
├── scripts/                      # Additional scripts
│   └── legacy/                   # Legacy scripts
│       └── (old script_modules content)
│
├── docs/                         # Documentation
│   ├── PROJECT_STRUCTURE.md      # This file
│   ├── README.md                 # Main README
│   └── 1907.03698v1.pdf          # Research paper
│
└── tests/                        # Test files (future)
    └── (test files)

```

---

## 🎯 Key Components

### **Core System** (`src/core/`)

1. **tennis_master.py**
   - Master controller that orchestrates the entire system
   - Launches CV processing engine first (blocking)
   - Then launches analytics viewer after CV completes
   - Manages inter-process communication
   - Handles graceful shutdown

2. **tennis_CV.py** (3000+ lines)
   - Main CV processing engine
   - Integrates 5 AI models:
     - YOLOv8 for player detection
     - YOLOv8-pose for pose estimation
     - TrackNet + YOLO for ball tracking (intelligent fusion)
     - Court detection with geometric validation
     - CatBoost for bounce detection
   - Outputs CSV data to `data/processed/csv/`
   - Generates annotated video to `outputs/videos/`

3. **tennis_analytics.py**
   - Clean visualization on black background
   - Reads CSV from `data/processed/csv/`
   - Interactive controls (pause, speed, step)
   - Real-time statistics dashboard
   - Outputs video to `outputs/videos/`

### **Detection Modules** (`src/detection/`)

- **players/player_detector.py**: YOLO and RF-DETR player detection
- **bounce/base_detector.py**: Physics-based bounce detection
- **bounce/**: Various bounce detector variants (ensemble, hybrid, sequence, optimized)

### **Analysis Modules** (`src/analysis/`)

- **tennis_shot_classifier.py**: ML-based shot classification (forehand, backhand, serve, etc.)
- **tennis_data_aggregator.py**: Combines multiple analysis outputs

### **Training Infrastructure** (`src/training/`)

- **bounce/**: Advanced bounce detection training with 100+ engineered features
- **shot/**: Shot classification training (future)

### **Evaluation** (`src/evaluation/`)

- Scripts to evaluate different detector variants
- Debug and performance analysis tools

---

## 🔧 Configuration

### **config.yaml**

All model paths now use the organized structure:

```yaml
models:
  yolo_player: "models/player/playersnball4.pt"
  yolo_pose: "models/pose/yolov8n-pose.pt"
  tracknet: "models/ball/pretrained_ball_detection.pt"
  court_detector: "models/court/model_tennis_court_det.pt"
  bounce_detector: "models/bounce/bounce_detector.cbm"
  rfdetr_model: "models/player/playersnball5.pt"

data:
  raw_videos: "data/raw/"
  processed_frames: "data/processed/"
  annotations: "data/annotations/"
  output: "outputs/"
```

---

## 🚀 Usage

### **Quick Start**

```bash
./start_tennis_analysis.sh
```

This script:
1. Checks for required files in new locations
2. Lists available videos in `data/raw/`
3. Launches `src/core/tennis_master.py`

### **Manual Launch**

```bash
python3 src/core/tennis_master.py --video data/raw/tennis_test5.mp4
```

### **Analytics Viewer Only**

```bash
python3 src/core/tennis_analytics.py --csv data/processed/csv/tennis_analysis_data.csv --output outputs/videos/my_analysis.mp4
```

### **Training Bounce Detector**

```bash
python3 src/training/bounce/improved_bounce_trainer.py \
    --annotations data/annotations/bounce/all_bounce_annotations.csv \
    --ball-data data/ball_coordinates/all_ball_coordinates.csv \
    --output models/bounce/bounce_detector.cbm
```

---

## 📊 Data Flow

```
1. INPUT: data/raw/tennis_video.mp4
   ↓
2. src/core/tennis_master.py (orchestrator)
   ↓
3. src/core/tennis_CV.py (processes video)
   - Loads models from models/
   - Uses external libs from external/
   - Writes CSV to data/processed/csv/
   - Writes video to outputs/videos/
   ↓
4. src/core/tennis_analytics.py (visualization)
   - Reads CSV from data/processed/csv/
   - Writes video to outputs/videos/
```

---

## 🔑 Important Path Changes

### **Before → After**

| Old Path | New Path |
|----------|----------|
| `tennis_master.py` | `src/core/tennis_master.py` |
| `tennis_CV.py` | `src/core/tennis_CV.py` |
| `tennis_analytics.py` | `src/core/tennis_analytics.py` |
| `models/playersnball4.pt` | `models/player/playersnball4.pt` |
| `pretrained_ball_detection.pt` | `models/ball/pretrained_ball_detection.pt` |
| `model_tennis_court_det.pt` | `models/court/model_tennis_court_det.pt` |
| `models/bounce_detector.cbm` | `models/bounce/bounce_detector.cbm` |
| `tennis_test5.mp4` | `data/raw/tennis_test5.mp4` |
| `tennis_analysis_data.csv` | `data/processed/csv/tennis_analysis_data.csv` |
| `tennis_analysis_output.mp4` | `outputs/videos/tennis_analysis_output.mp4` |
| `TrackNet/` | `external/TrackNet/` |
| `TennisCourtDetector/` | `external/TennisCourtDetector/` |
| `rf-detr/` | `external/rf-detr/` |

---

## 📝 Import Path Updates

All Python files have been updated with proper import paths:

```python
# In src/core/tennis_CV.py and tennis_master.py
PROJECT_ROOT = Path(__file__).parent.parent.parent

# External libraries
sys.path.insert(0, str(PROJECT_ROOT / "external" / "TennisCourtDetector"))
sys.path.insert(0, str(PROJECT_ROOT / "external" / "rf-detr"))

# Data paths
csv_path = PROJECT_ROOT / "data" / "processed" / "csv" / "tennis_analysis_data.csv"
output_path = PROJECT_ROOT / "outputs" / "videos" / "analysis.mp4"
```

---

## 🎯 Benefits of New Structure

1. **Clear Separation of Concerns**: Core, detection, analysis, training, evaluation
2. **Organized Models**: Models grouped by type (player, pose, ball, court, bounce)
3. **Structured Data**: Raw vs processed, annotations by type, training data separate
4. **External Libraries Isolated**: All 3rd-party code in `external/`
5. **Professional Layout**: Follows Python project best practices
6. **Easier Navigation**: Find what you need quickly
7. **Better Version Control**: .gitignore can target specific directories
8. **Scalable**: Easy to add new modules, models, or data types

---

## 🔮 Future Enhancements

- `tests/`: Unit and integration tests
- `notebooks/`: Jupyter notebooks for analysis
- `api/`: REST API for system integration
- `frontend/`: Web interface
- `configs/`: Multiple configuration profiles
- `scripts/deployment/`: Deployment scripts
- `docker/`: Containerization files

---

## 📚 Additional Documentation

- **README.md**: System overview and getting started
- **config.yaml**: Configuration reference
- **requirements.txt**: Python dependencies
- **1907.03698v1.pdf**: Research paper on tennis ball bounce detection

---

## ⚠️ Migration Notes

If you have existing scripts or workflows that reference old paths:

1. Update all file references to use new paths
2. Update model paths in any custom configs
3. Update data input/output paths
4. Use `config.yaml` for centralized path management
5. All core scripts now use `PROJECT_ROOT` for path resolution

---

**Questions or issues?** Check the main README.md or open an issue.

