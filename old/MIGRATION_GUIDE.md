# Migration Guide - Tennis Analytics Reorganization

**Date:** October 17, 2025

## Overview

The tennis_analytics repository has been completely reorganized for better maintainability and professional structure.

## Quick Reference

### Starting the System

**Before:**
```bash
python3 tennis_master.py --video tennis_test5.mp4
```

**After:**
```bash
./start_tennis_analysis.sh
# or manually:
python3 src/core/tennis_master.py --video data/raw/tennis_test5.mp4
```

### Key File Locations

| Component | Old Location | New Location |
|-----------|-------------|--------------|
| Main scripts | Root directory | `src/core/` |
| Detection modules | Root directory | `src/detection/` |
| Training scripts | Root directory | `src/training/bounce/` |
| Models | `models/` (mixed) | `models/{player,pose,ball,court,bounce}/` |
| Videos | Root directory | `data/raw/` |
| Output videos | Root/`video_outputs/` | `outputs/videos/` |
| CSV data | Root directory | `data/processed/csv/` |
| Annotations | Root directory | `data/annotations/{bounce,shot}/` |
| External libs | Root directory | `external/` |
| Documentation | Root directory | `docs/` |

## Updated Paths

### Models
- `models/playersnball4.pt` → `models/player/playersnball4.pt`
- `models/yolov8n-pose.pt` → `models/pose/yolov8n-pose.pt`
- `pretrained_ball_detection.pt` → `models/ball/pretrained_ball_detection.pt`
- `model_tennis_court_det.pt` → `models/court/model_tennis_court_det.pt`
- `models/bounce_detector.cbm` → `models/bounce/bounce_detector.cbm`
- `rf-detr-base.pth` → `models/rf-detr/rf-detr-base.pth`

### Data Files
- `tennis_test*.mp4` → `data/raw/tennis_test*.mp4`
- `tennis_analysis_data.csv` → `data/processed/csv/tennis_analysis_data.csv`
- `*_annotations.csv` → `data/annotations/bounce/*_annotations.csv`
- `ball_coords_*.csv` → `data/ball_coordinates/ball_coords_*.csv`
- `*training_data*.csv` → `data/training/bounce/`

### Scripts
- `tennis_master.py` → `src/core/tennis_master.py`
- `tennis_CV.py` → `src/core/tennis_CV.py`
- `tennis_analytics.py` → `src/core/tennis_analytics.py`
- `tennis_player.py` → `src/detection/players/player_detector.py`
- `tennis_bounce.py` → `src/detection/bounce/base_detector.py`
- `tennis_shot_classifier.py` → `src/analysis/tennis_shot_classifier.py`
- `improved_bounce_trainer.py` → `src/training/bounce/improved_bounce_trainer.py`

### External Libraries
- `TrackNet/` → `external/TrackNet/`
- `TennisCourtDetector/` → `external/TennisCourtDetector/`
- `rf-detr/` → `external/rf-detr/`

## What Changed

### 1. File Organization
All files have been moved to appropriate directories based on their function.

### 2. Configuration Updated
`config.yaml` now uses the new organized paths.

### 3. Import Paths Fixed
All Python files use `PROJECT_ROOT` for path resolution:
```python
PROJECT_ROOT = Path(__file__).parent.parent.parent
```

### 4. External Libraries
Third-party libraries moved to `external/` and added to sys.path dynamically.

### 5. Data Separation
- **Raw data**: `data/raw/` (input videos)
- **Processed data**: `data/processed/csv/` (analysis outputs)
- **Annotations**: `data/annotations/` (ground truth labels)
- **Training data**: `data/training/` (ML training datasets)

### 6. Output Organization
- **Videos**: `outputs/videos/`
- **Results**: `outputs/results/` (JSON, analysis files)

## Compatibility

### ✅ What Works Automatically
- Running `./start_tennis_analysis.sh`
- Running core system via `python3 src/core/tennis_master.py`
- All internal imports and path references
- Config.yaml references

### ⚠️ What Needs Updates
If you have:
- **Custom scripts**: Update file paths
- **Jupyter notebooks**: Update data/model paths
- **External workflows**: Update video input/output paths
- **CI/CD pipelines**: Update paths in automation scripts

## Testing the Migration

1. **Test the main system:**
   ```bash
   ./start_tennis_analysis.sh
   ```

2. **Verify model loading:**
   ```bash
   python3 -c "from pathlib import Path; print([str(p) for p in Path('models').rglob('*.pt')])"
   ```

3. **Check data structure:**
   ```bash
   ls -R data/
   ```

4. **Verify external libraries:**
   ```bash
   ls external/
   ```

## Need Help?

- Check `docs/PROJECT_STRUCTURE.md` for complete directory layout
- Check `docs/README.md` for system documentation
- All core functionality remains unchanged - only locations have changed

## Rollback (If Needed)

A backup was created: `config.yaml.backup`

To rollback:
1. Review git history: `git log --oneline`
2. Reset if needed: `git reset --hard <commit-hash>`
3. Or manually restore files from backup

## Future Maintenance

Going forward:
- Place new models in appropriate `models/` subdirectories
- Put new videos in `data/raw/`
- Training scripts go in `src/training/`
- Analysis outputs go in `outputs/`
- Keep external libraries in `external/`

---

**The reorganization is complete! The system is now production-ready and professionally structured.** 🎾

