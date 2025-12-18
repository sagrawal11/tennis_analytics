# Detection Modules

This directory contains all detection-related components for tennis video analysis.

## Current Structure

### Core Detection Modules
- **tennis_player.py** - Player detection and tracking using YOLOv8 and RF-DETR
- **tennis_bounce.py** - Basic trajectory-based bounce detection

### Bounce Detection Variants

Multiple bounce detection implementations with different approaches:

- **ensemble_bounce_detector.py** - Combines multiple detection methods for robust results
- **hybrid_bounce_detector.py** - Hybrid physics + ML approach
- **sequence_bounce_detector.py** - Sequence-based temporal analysis using LSTM/RNN
- **simple_ultra_detector.py** - Lightweight fast detector for real-time use
- **optimized_ultra_detector.py** - Performance-optimized version with enhanced features

## Future Organization

**⚠️ Planned Reorganization:**

This directory will be reorganized into subdirectories:
```
detection/
├── players/
│   └── player_detector.py
├── ball/
│   └── ball_detector.py
└── bounce/
    ├── base_detector.py
    ├── ensemble_detector.py
    ├── hybrid_detector.py
    ├── sequence_detector.py
    ├── simple_ultra_detector.py
    └── optimized_ultra_detector.py
```

## Usage

Each detector can be imported and used independently:

```python
# Player detection
from src.detection.players.player_detector import PlayerDetector
player_detector = PlayerDetector(model_path, config)
players = player_detector.detect_players(frame)

# Bounce detection
from src.detection.bounce.base_detector import BounceDetector
bounce_detector = BounceDetector()
bounce_detector.add_ball_position(x, y, frame_num)
is_bounce, confidence = bounce_detector.detect_bounce(frame_num)
```

## Integration

These detectors are integrated into the main analysis pipeline in `src/core/tennis_CV.py`.

## Related

- **Core system**: `src/core/`
- **Training**: `src/training/bounce/`
- **Evaluation**: `src/evaluation/`
- **Models**: `models/bounce/`, `models/player/`

