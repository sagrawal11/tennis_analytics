# Sequence Models Directory

This directory is reserved for sequence-based bounce detection models.

## Planned Models

- LSTM-based sequence detectors
- Temporal CNN models
- Transformer-based detectors
- RNN variants for temporal analysis

## Status

**📋 Pending Training** - Models will be added as they are trained.

## Related Training Scripts

See `src/training/bounce/` for training scripts:
- `lstm_bounce_trainer.py` - LSTM model training
- `sequence_bounce_trainer.py` - Sequence model training

## Usage

Once trained, models will be saved here and loaded by sequence bounce detectors in `src/detection/bounce/`.

