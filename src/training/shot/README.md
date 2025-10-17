# Shot Classification Training

This directory is reserved for shot classification model training scripts.

## Purpose

Training scripts for tennis shot classification models including:
- Forehand detection training
- Backhand detection training
- Serve detection training
- Overhead smash detection training
- Movement classification training

## Status

**📋 Planned** - Training infrastructure to be implemented.

## Current Implementation

Shot classification is currently handled by `src/analysis/tennis_shot_classifier.py` using:
- Rule-based classification
- ML models (Random Forest, SVM, Logistic Regression, XGBoost)
- Trained models stored in `models/advanced/`

## Future Training Scripts

Planned scripts:
- `shot_classifier_trainer.py` - Main shot classification training
- `movement_classifier_trainer.py` - Movement detection training
- `data_augmentation.py` - Data augmentation for shot classification

## Related

- Analysis: `src/analysis/tennis_shot_classifier.py`
- Models: `models/advanced/`
- Evaluation: `src/evaluation/`

