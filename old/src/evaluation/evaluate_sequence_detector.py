#!/usr/bin/env python3
"""
Evaluate Sequence Bounce Detector

Quick evaluation of the sequence detector with different thresholds.
"""

import pandas as pd
import numpy as np
import logging
from typing import List, Tuple, Dict
from pathlib import Path
from sequence_bounce_detector import SequenceBounceDetector

logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')
logger = logging.getLogger(__name__)


def evaluate_sequence_detector():
    """Evaluate sequence detector with different thresholds"""
    
    # Load data
    annotations_df = pd.read_csv('improved_bounce_annotations.csv')
    ball_df = pd.read_csv('all_ball_coordinates.csv')
    
    # Handle column names
    if 'ball_x' in ball_df.columns:
        x_col, y_col = 'ball_x', 'ball_y'
    else:
        x_col, y_col = 'x', 'y'
    
    logger.info(f"Evaluating on {len(annotations_df)} annotations")
    
    # Test different thresholds
    thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]
    
    for threshold in thresholds:
        logger.info(f"\n=== Testing threshold: {threshold} ===")
        
        detector = SequenceBounceDetector(threshold=threshold)
        
        all_predictions = []
        all_labels = []
        all_confidences = []
        
        for video_name, video_annotations in annotations_df.groupby('video_name'):
            # Get ball data for this video
            video_ball_data = ball_df[ball_df.get('video_name', '') == video_name] if 'video_name' in ball_df.columns else ball_df
            
            # Create trajectory for this video
            trajectory = []
            for _, row in video_ball_data.iterrows():
                frame = row['frame']
                x = row[x_col] if pd.notna(row[x_col]) else None
                y = row[y_col] if pd.notna(row[y_col]) else None
                trajectory.append((x, y, frame))
            
            # Test detector on each annotation
            for _, annotation in video_annotations.iterrows():
                frame_number = annotation['frame_number']
                true_label = annotation['is_bounce']
                
                # Get detector prediction
                is_bounce, confidence = detector.detect_bounce(trajectory, frame_number)
                
                all_predictions.append(1 if is_bounce else 0)
                all_labels.append(true_label)
                all_confidences.append(confidence)
        
        # Calculate metrics
        predictions = np.array(all_predictions)
        labels = np.array(all_labels)
        confidences = np.array(all_confidences)
        
        total_samples = len(predictions)
        true_positives = np.sum((predictions == 1) & (labels == 1))
        false_positives = np.sum((predictions == 1) & (labels == 0))
        true_negatives = np.sum((predictions == 0) & (labels == 0))
        false_negatives = np.sum((predictions == 0) & (labels == 1))
        
        accuracy = (true_positives + true_negatives) / total_samples
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        # AUC calculation
        from sklearn.metrics import roc_auc_score
        try:
            auc_score = roc_auc_score(labels, confidences)
        except ValueError:
            auc_score = 0.5
        
        print(f"Threshold: {threshold}")
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  AUC: {auc_score:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall: {recall:.4f}")
        print(f"  F1-Score: {f1_score:.4f}")
        print(f"  Detected bounces: {true_positives + false_positives}")
        print(f"  True bounces: {true_positives + false_negatives}")
        print(f"  Bounce detection rate: {(true_positives + false_positives) / total_samples:.4f}")
        
        # Performance assessment
        if auc_score >= 0.9:
            print(f"  🎉 EXCELLENT! AUC >= 90%")
        elif auc_score >= 0.8:
            print(f"  🚀 VERY GOOD! AUC >= 80%")
        elif auc_score >= 0.7:
            print(f"  ✅ GOOD! AUC >= 70%")
        elif auc_score >= 0.6:
            print(f"  📈 DECENT! AUC >= 60%")
        else:
            print(f"  ⚠️  Needs improvement")


if __name__ == "__main__":
    evaluate_sequence_detector()
