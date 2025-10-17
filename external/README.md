# External Libraries

Third-party libraries and research code used in the tennis analytics system.

---

## Production Dependencies

### rf-detr/
**RF-DETR Object Detection Framework**

Enhanced object detection model used for player and ball detection.

- **Purpose**: High-performance player and ball detection
- **Used by**: `src/core/tennis_CV.py`, `src/detection/players/player_detector.py`
- **Model**: `models/rf-detr/rf-detr-base.pth`
- **Documentation**: See `rf-detr/README.md`

### TennisCourtDetector/
**Tennis Court Detection and Homography**

Court keypoint detection and homography transformation for geometric analysis.

- **Purpose**: Detect court lines and perform court-to-world coordinate mapping
- **Used by**: `src/core/tennis_CV.py`
- **Model**: `models/court/model_tennis_court_det.pt`
- **Features**: 
  - Court keypoint detection (14 points)
  - Homography matrix computation
  - Court line refinement
- **Documentation**: See `TennisCourtDetector/README.md`

### TrackNet/
**Ball Tracking Neural Network**

Deep learning model for tennis ball detection and tracking.

- **Purpose**: High-precision ball detection and trajectory tracking
- **Used by**: `src/core/tennis_CV.py`
- **Model**: `models/ball/pretrained_ball_detection.pt`
- **Architecture**: TrackNet architecture for ball tracking
- **Documentation**: See `TrackNet/README.md`

### TRACE/
**Tennis Research Analysis and Computer-vision Engine**

Comprehensive tennis analysis library.

- **Purpose**: Reference implementation and research code
- **Status**: Reference only (not directly used in production)
- **Features**: Ball detection, mapping, court detection, body tracking
- **Documentation**: See `TRACE/README.md`

---

## Research & Reference

### AI-Tennis-Ball-Bounce-Detection/
**Research Notebooks and Datasets**

Collection of Jupyter notebooks and research materials for bounce detection.

- **Purpose**: Research reference and training methodology
- **Status**: **Reference material only** (not used in production)
- **Contents**:
  - Jupyter notebooks for bounce detection research
  - Training data examples (8,462 labeled frames)
  - YOLO training configurations
  - Performance benchmarks
- **Usage**: Reference for understanding bounce detection approaches

**Key Notebooks:**
- `Module1_Step1_Downloading_from_youtube.ipynb` - Data collection
- `Module1_Step2_Image_for_Annotation.ipynb` - Annotation workflow
- `Module2_Step2*_Custom_training_YOLOv4-tiny_*.ipynb` - Training examples
- `Module3_Step1_Video_Frame_Object_Detection.ipynb` - Detection pipeline
- `Module3_Step2_Bounce_detector_preprocessing.ipynb` - Preprocessing
- `Module3_Step3_Bounce_detector_final.ipynb` - Final bounce detection

---

## Integration

These libraries are dynamically added to Python's `sys.path` in core modules:

```python
# Example from src/core/tennis_CV.py
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "external" / "TennisCourtDetector"))
sys.path.insert(0, str(PROJECT_ROOT / "external" / "rf-detr"))
```

This allows importing from external libraries:

```python
# TennisCourtDetector imports
from tracknet import BallTrackerNet
from postprocess import postprocess, refine_kps
from homography import get_trans_matrix

# RF-DETR imports
from rfdetr import RFDETRNano
```

---

## Installation Requirements

Each library has its own requirements. See individual `requirements.txt` files:

- `TennisCourtDetector/requirements.txt`
- `TrackNet/requirements.txt`
- `rf-detr/pyproject.toml`
- `AI-Tennis-Ball-Bounce-Detection/requirements.txt`

Main project requirements are in the root `requirements.txt`.

---

## Maintenance

### Updating External Libraries

To update an external library:

1. Pull latest changes into the library directory
2. Test integration with main system
3. Update requirements if needed
4. Document any API changes

### Adding New External Libraries

To add a new external library:

1. Clone/download to `external/`
2. Add to `.gitignore` if appropriate (or use git submodules)
3. Update `sys.path` insertion in relevant modules
4. Document in this README
5. Add requirements to main `requirements.txt`

---

## License Information

Each external library has its own license:

- **rf-detr**: See `rf-detr/LICENSE`
- **TennisCourtDetector**: Check repository for license
- **TrackNet**: Check repository for license
- **TRACE**: Check repository for license
- **AI-Tennis-Ball-Bounce-Detection**: Research/educational use

Please review individual licenses before distribution.

---

## References

### Papers & Research

- **RF-DETR**: See `rf-detr/CITATION.cff`
- **TrackNet**: Ball tracking neural network architecture
- **Tennis Court Detection**: Court keypoint detection methods
- **Bounce Detection**: AI-based bounce detection research

### Related

- **Main system**: `src/core/`
- **Detection modules**: `src/detection/`
- **Models**: `models/`
- **Configuration**: `config.yaml`

---

## Troubleshooting

### Import Errors

If you encounter import errors:

1. Verify the library exists in `external/`
2. Check `sys.path` insertion in your module
3. Install library requirements
4. Verify Python version compatibility

### Model Not Found

If models can't be loaded:

1. Check model path in `config.yaml`
2. Verify model exists in `models/` directory
3. Check file permissions
4. Ensure correct model format (.pt, .pth, .h5)

### Version Conflicts

If there are version conflicts:

1. Check individual library requirements
2. Use virtual environment for isolation
3. Update conflicting packages
4. Document version constraints in `requirements.txt`

---

**For system-wide documentation**, see:
- `docs/PROJECT_STRUCTURE.md` - Complete project structure
- `docs/README.md` - System documentation
- `MIGRATION_GUIDE.md` - Path migration guide

