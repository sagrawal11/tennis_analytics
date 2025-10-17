# Analysis Modules

Runtime analysis and classification components for tennis video analysis.

## Modules

### tennis_shot_classifier.py
**Shot classification using ML and rule-based methods**

Classifies tennis shots into categories:
- Forehand
- Backhand  
- Serve
- Overhead smash
- Ready stance
- Moving

**Features:**
- Movement classification (ready vs moving)
- Shot type classification
- Multiple ML models (Random Forest, SVM, Logistic Regression, XGBoost)
- Pose-based feature extraction
- Temporal analysis

**Usage:**
```python
from src.analysis.tennis_shot_classifier import ShotClassifier

classifier = ShotClassifier()
classifier.load_models()
shot_type = classifier.classify_shot(pose_keypoints, ball_position)
```

### tennis_data_aggregator.py  
**Aggregates data from multiple analysis components**

Combines outputs from:
- Player detection
- Ball tracking
- Bounce detection
- Pose estimation
- Shot classification

Creates comprehensive CSV datasets for further analysis and visualization.

**Usage:**
```python
from src.analysis.tennis_data_aggregator import TennisDataAggregator

aggregator = TennisDataAggregator(config_path)
aggregator.process_video(video_path, output_csv)
```

## Data Flow

1. **Detection** (`src/detection/`) → Raw detections (players, ball, bounces)
2. **Analysis** (this module) → Higher-level insights (shot types, patterns)
3. **Output** → CSV files in `data/processed/csv/`

## Training

ML models used by these modules are trained using:
- `src/training/shot/` - Shot classification training
- Trained models stored in `models/advanced/`

## Related

- **Detection**: `src/detection/`
- **Core system**: `src/core/`
- **Models**: `models/advanced/`
- **Training**: `src/training/`

