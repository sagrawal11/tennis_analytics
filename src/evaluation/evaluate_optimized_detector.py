#!/usr/bin/env python3
"""
Evaluate Optimized Ultra-Advanced Bounce Detector

Test the optimized detector on real data to measure performance improvements.
"""

import pandas as pd
import numpy as np
import logging
from typing import List, Tuple, Dict
from pathlib import Path
from optimized_ultra_detector import OptimizedUltraDetector

logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')
logger = logging.getLogger(__name__)


class OptimizedDetectorEvaluator:
    """Evaluator for the optimized bounce detector"""
    
    def __init__(self):
        self.detector = OptimizedUltraDetector()
        self.results = []
        
    def evaluate_on_annotations(self, annotations_file: str, ball_data_file: str) -> Dict:
        """Evaluate detector performance on annotated data"""
        
        # Load data
        annotations_df = pd.read_csv(annotations_file)
        ball_df = pd.read_csv(ball_data_file)
        
        # Handle column names
        if 'ball_x' in ball_df.columns:
            x_col, y_col = 'ball_x', 'ball_y'
        else:
            x_col, y_col = 'x', 'y'
        
        logger.info(f"Evaluating on {len(annotations_df)} annotations")
        
        # Group by video
        all_predictions = []
        all_labels = []
        all_confidences = []
        
        for video_name, video_annotations in annotations_df.groupby('video_name'):
            logger.info(f"Evaluating {video_name}...")
            
            # Get ball data for this video
            video_ball_data = ball_df[ball_df.get('video_name', '') == video_name] if 'video_name' in ball_df.columns else ball_df
            
            # Create trajectory for this video
            trajectory = self._create_trajectory(video_ball_data, x_col, y_col)
            
            # Test detector on each annotation
            video_predictions = []
            video_labels = []
            video_confidences = []
            
            for _, annotation in video_annotations.iterrows():
                frame_number = annotation['frame_number']
                true_label = annotation['is_bounce']
                
                # Get detector prediction
                is_bounce, confidence, details = self.detector.detect_bounce_optimized(
                    trajectory, frame_number
                )
                
                video_predictions.append(1 if is_bounce else 0)
                video_labels.append(true_label)
                video_confidences.append(confidence)
            
            all_predictions.extend(video_predictions)
            all_labels.extend(video_labels)
            all_confidences.extend(video_confidences)
            
            logger.info(f"  {video_name}: {np.sum(video_predictions)} predictions, {np.sum(video_labels)} true bounces")
        
        # Calculate metrics
        metrics = self._calculate_metrics(all_predictions, all_labels, all_confidences)
        
        return metrics
    
    def _create_trajectory(self, ball_df: pd.DataFrame, x_col: str, y_col: str) -> List[Tuple[float, float, int]]:
        """Create trajectory from ball tracking data"""
        trajectory = []
        
        for _, row in ball_df.iterrows():
            frame = row['frame']
            x = row[x_col] if pd.notna(row[x_col]) else None
            y = row[y_col] if pd.notna(row[y_col]) else None
            
            trajectory.append((x, y, frame))
        
        return trajectory
    
    def _calculate_metrics(self, predictions: List[int], labels: List[int], 
                          confidences: List[float]) -> Dict:
        """Calculate performance metrics"""
        
        predictions = np.array(predictions)
        labels = np.array(labels)
        confidences = np.array(confidences)
        
        # Basic metrics
        total_samples = len(predictions)
        true_positives = np.sum((predictions == 1) & (labels == 1))
        false_positives = np.sum((predictions == 1) & (labels == 0))
        true_negatives = np.sum((predictions == 0) & (labels == 0))
        false_negatives = np.sum((predictions == 0) & (labels == 1))
        
        # Calculate metrics
        accuracy = (true_positives + true_negatives) / total_samples
        
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        # AUC calculation
        from sklearn.metrics import roc_auc_score
        try:
            auc_score = roc_auc_score(labels, confidences)
        except ValueError:
            auc_score = 0.5  # Default for edge cases
        
        # Bounce-specific metrics
        bounce_precision = precision
        bounce_recall = recall
        bounce_f1 = f1_score
        
        # Non-bounce metrics
        non_bounce_precision = true_negatives / (true_negatives + false_negatives) if (true_negatives + false_negatives) > 0 else 0
        non_bounce_recall = true_negatives / (true_negatives + false_positives) if (true_negatives + false_positives) > 0 else 0
        non_bounce_f1 = 2 * (non_bounce_precision * non_bounce_recall) / (non_bounce_precision + non_bounce_recall) if (non_bounce_precision + non_bounce_recall) > 0 else 0
        
        metrics = {
            'total_samples': total_samples,
            'true_positives': true_positives,
            'false_positives': false_positives,
            'true_negatives': true_negatives,
            'false_negatives': false_negatives,
            'accuracy': accuracy,
            'auc_score': auc_score,
            'bounce_precision': bounce_precision,
            'bounce_recall': bounce_recall,
            'bounce_f1': bounce_f1,
            'non_bounce_precision': non_bounce_precision,
            'non_bounce_recall': non_bounce_recall,
            'non_bounce_f1': non_bounce_f1,
            'bounce_rate': np.mean(labels),
            'prediction_rate': np.mean(predictions)
        }
        
        return metrics
    
    def print_results(self, metrics: Dict):
        """Print evaluation results"""
        print("\n" + "="*60)
        print("OPTIMIZED ULTRA-ADVANCED DETECTOR EVALUATION")
        print("="*60)
        
        print(f"Total samples: {metrics['total_samples']}")
        print(f"True bounces: {metrics['true_positives'] + metrics['false_negatives']}")
        print(f"Detected bounces: {metrics['true_positives'] + metrics['false_positives']}")
        print(f"Bounce rate: {metrics['bounce_rate']:.3f}")
        print(f"Prediction rate: {metrics['prediction_rate']:.3f}")
        
        print(f"\nOverall Performance:")
        print(f"  Accuracy: {metrics['accuracy']:.4f}")
        print(f"  AUC Score: {metrics['auc_score']:.4f}")
        
        print(f"\nBounce Detection:")
        print(f"  Precision: {metrics['bounce_precision']:.4f}")
        print(f"  Recall: {metrics['bounce_recall']:.4f}")
        print(f"  F1-Score: {metrics['bounce_f1']:.4f}")
        
        print(f"\nNon-Bounce Detection:")
        print(f"  Precision: {metrics['non_bounce_precision']:.4f}")
        print(f"  Recall: {metrics['non_bounce_recall']:.4f}")
        print(f"  F1-Score: {metrics['non_bounce_f1']:.4f}")
        
        print(f"\nConfusion Matrix:")
        print(f"  True Positives:  {metrics['true_positives']}")
        print(f"  False Positives: {metrics['false_positives']}")
        print(f"  True Negatives:  {metrics['true_negatives']}")
        print(f"  False Negatives: {metrics['false_negatives']}")
        
        # Performance assessment
        print(f"\nPerformance Assessment:")
        if metrics['auc_score'] >= 0.9:
            print("  🎉 EXCELLENT! AUC >= 90% - Target achieved!")
        elif metrics['auc_score'] >= 0.8:
            print("  🚀 VERY GOOD! AUC >= 80% - Close to target!")
        elif metrics['auc_score'] >= 0.7:
            print("  ✅ GOOD! AUC >= 70% - Significant improvement!")
        else:
            print("  ⚠️  Needs improvement - AUC < 70%")
        
        if metrics['bounce_precision'] >= 0.5:
            print("  ✅ High bounce precision - Good false positive control!")
        else:
            print("  ⚠️  Low bounce precision - Too many false positives")
        
        if metrics['bounce_recall'] >= 0.5:
            print("  ✅ High bounce recall - Good at finding bounces!")
        else:
            print("  ⚠️  Low bounce recall - Missing many bounces")


def main():
    """Main evaluation function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate optimized bounce detector')
    parser.add_argument('--annotations', default='improved_bounce_annotations.csv',
                       help='Annotations file')
    parser.add_argument('--ball-data', default='all_ball_coordinates.csv',
                       help='Ball tracking data file')
    
    args = parser.parse_args()
    
    try:
        # Initialize evaluator
        evaluator = OptimizedDetectorEvaluator()
        
        # Run evaluation
        logger.info("Starting evaluation...")
        metrics = evaluator.evaluate_on_annotations(args.annotations, args.ball_data)
        
        # Print results
        evaluator.print_results(metrics)
        
        # Save results
        results_file = 'optimized_detector_results.json'
        import json
        with open(results_file, 'w') as f:
            json.dump(metrics, f, indent=2)
        logger.info(f"Results saved to {results_file}")
        
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
