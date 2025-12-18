# Evaluation Modules

Tools for evaluating detector performance and debugging the tennis analysis pipeline.

## Modules

### evaluate_simple_detector.py
**Evaluate basic bounce detectors**

Tests simple trajectory-based bounce detection against ground truth annotations.

**Metrics:**
- Precision, Recall, F1-Score
- True Positives, False Positives, False Negatives
- Temporal accuracy (frame-level)

### evaluate_sequence_detector.py
**Evaluate sequence-based bounce detectors**

Tests LSTM/RNN-based sequence detectors that analyze temporal patterns.

**Features:**
- Sequence model evaluation
- Temporal window analysis
- Confusion matrices
- Performance over different sequence lengths

### evaluate_optimized_detector.py
**Evaluate optimized and ultra bounce detectors**

Tests performance-optimized bounce detectors including:
- Simple ultra detector
- Optimized ultra detector
- Ensemble methods

**Benchmarks:**
- Accuracy metrics
- Inference speed
- Memory usage
- Real-time performance

### debug_feature_extraction.py
**Debug feature extraction pipeline**

Debugging tool for:
- Feature extraction from pose keypoints
- Temporal feature computation
- Data preprocessing
- Model input validation

**Usage:**
Helps identify issues in the feature engineering pipeline before training.

## Usage

### Running Evaluations

```python
# Basic evaluation
python src/evaluation/evaluate_simple_detector.py \
    --annotations data/annotations/bounce/all_bounce_annotations.csv \
    --ball-data data/ball_coordinates/all_ball_coordinates.csv

# Sequence detector evaluation  
python src/evaluation/evaluate_sequence_detector.py \
    --model models/sequence/sequence_model.pkl \
    --test-data data/training/bounce/test_data.csv

# Debug features
python src/evaluation/debug_feature_extraction.py \
    --csv data/processed/csv/tennis_analysis_data.csv
```

## Ground Truth Data

Evaluations require annotated ground truth data from:
- `data/annotations/bounce/` - Bounce annotations
- `data/ball_coordinates/` - Ball tracking data
- `data/training/bounce/` - Training/test datasets

## Metrics

Standard evaluation metrics used:
- **Precision** = TP / (TP + FP)
- **Recall** = TP / (TP + FN)
- **F1-Score** = 2 × (Precision × Recall) / (Precision + Recall)
- **Accuracy** = (TP + TN) / (TP + TN + FP + FN)

Temporal tolerance typically: ±3 frames (±100ms at 30fps)

## Related

- **Detection**: `src/detection/`
- **Training**: `src/training/bounce/`
- **Annotations**: `data/annotations/`
- **Models**: `models/bounce/`

## Adding New Evaluations

To evaluate a new detector:

1. Create evaluation script following existing patterns
2. Load ground truth annotations
3. Run detector on test videos/data
4. Compare predictions vs ground truth
5. Report standard metrics
6. Generate visualizations (confusion matrices, PR curves)

