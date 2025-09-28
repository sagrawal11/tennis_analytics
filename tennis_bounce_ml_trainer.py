#!/usr/bin/env python3
"""
Tennis Ball Bounce Detection - ML Training Data Preparation

This script prepares training data for machine learning-based bounce detection.
It takes annotated bounce data and creates features for training a model.

Usage:
    python tennis_bounce_ml_trainer.py --annotations bounce_annotations.csv --ball-data ball_tracking_data.csv
"""

import pandas as pd
import numpy as np
import argparse
import logging
from typing import List, Tuple, Dict, Optional
from pathlib import Path
import json

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')
logger = logging.getLogger(__name__)


class BounceMLTrainer:
    """Machine Learning trainer for tennis ball bounce detection"""
    
    def __init__(self, window_size: int = 10):
        """Initialize ML trainer with window size for trajectory analysis"""
        self.window_size = window_size
        self.feature_names = []
        
        logger.info(f"Initialized BounceMLTrainer with window size: {window_size}")
    
    def prepare_training_data(self, annotations_file: str, ball_data_file: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare training data from annotations and ball tracking data
        
        Args:
            annotations_file: CSV with columns: video_name, frame_number, is_bounce
            ball_data_file: CSV with ball tracking data (ball_x, ball_y, frame)
            
        Returns:
            Tuple of (features, labels)
        """
        # Load annotations
        annotations_df = self._load_annotations(annotations_file)
        
        # Load ball tracking data
        ball_df = self._load_ball_data(ball_data_file)
        
        # Create training samples
        features, labels = self._create_training_samples(annotations_df, ball_df)
        
        logger.info(f"Created {len(features)} training samples")
        logger.info(f"Bounce samples: {np.sum(labels)} ({np.sum(labels)/len(labels)*100:.1f}%)")
        
        return features, labels
    
    def _load_annotations(self, annotations_file: str) -> pd.DataFrame:
        """Load bounce annotations from CSV file"""
        try:
            df = pd.read_csv(annotations_file)
            
            # Validate required columns
            required_cols = ['video_name', 'frame_number', 'is_bounce']
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                raise ValueError(f"Missing required columns: {missing_cols}")
            
            logger.info(f"Loaded {len(df)} annotations from {annotations_file}")
            return df
            
        except Exception as e:
            logger.error(f"Error loading annotations: {e}")
            raise
    
    def _load_ball_data(self, ball_data_file: str) -> pd.DataFrame:
        """Load ball tracking data from CSV file"""
        try:
            df = pd.read_csv(ball_data_file)
            
            # Validate required columns
            required_cols = ['ball_x', 'ball_y', 'frame']
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                raise ValueError(f"Missing required columns: {missing_cols}")
            
            logger.info(f"Loaded {len(df)} ball tracking records from {ball_data_file}")
            return df
            
        except Exception as e:
            logger.error(f"Error loading ball data: {e}")
            raise
    
    def _create_training_samples(self, annotations_df: pd.DataFrame, ball_df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Create training samples from frame-by-frame annotations and ball data"""
        features_list = []
        labels_list = []
        
        # Group annotations by video
        for video_name, video_annotations in annotations_df.groupby('video_name'):
            logger.info(f"Processing video: {video_name}")
            
            # Get ball data for this video
            video_ball_data = ball_df[ball_df.get('video_name', '') == video_name] if 'video_name' in ball_df.columns else ball_df
            
            # Sort annotations by frame number
            video_annotations = video_annotations.sort_values('frame_number')
            
            # Create samples for each frame annotation
            for _, annotation in video_annotations.iterrows():
                frame_number = annotation['frame_number']
                is_bounce = annotation['is_bounce']
                
                # Extract features around this frame
                features = self._extract_features_at_frame(video_ball_data, frame_number)
                
                if features is not None:
                    features_list.append(features)
                    labels_list.append(is_bounce)
                else:
                    logger.warning(f"Could not extract features for video {video_name}, frame {frame_number}")
        
        return np.array(features_list), np.array(labels_list)
    
    def _extract_features_at_frame(self, ball_df: pd.DataFrame, frame_number: int) -> Optional[np.ndarray]:
        """Extract trajectory features at a specific frame"""
        try:
            # Get window of frames around the target frame
            window_start = frame_number - self.window_size // 2
            window_end = frame_number + self.window_size // 2
            
            # Extract ball positions in the window
            window_data = ball_df[
                (ball_df['frame'] >= window_start) & 
                (ball_df['frame'] <= window_end)
            ].sort_values('frame')
            
            if len(window_data) < self.window_size * 0.7:  # Need at least 70% of window
                return None
            
            # Extract trajectory features
            features = self._compute_trajectory_features(window_data)
            
            return features
            
        except Exception as e:
            logger.warning(f"Error extracting features at frame {frame_number}: {e}")
            return None
    
    def _compute_trajectory_features(self, window_data: pd.DataFrame) -> np.ndarray:
        """Compute trajectory features from window data"""
        features = []
        
        # Get x, y coordinates
        x_coords = window_data['ball_x'].values
        y_coords = window_data['ball_y'].values
        
        # Ensure we have enough data
        if len(x_coords) < 3:
            # Pad with zeros if insufficient data
            x_coords = np.pad(x_coords, (0, max(0, 3 - len(x_coords))), 'constant')
            y_coords = np.pad(y_coords, (0, max(0, 3 - len(y_coords))), 'constant')
        
        # 1. Position features (normalized coordinates)
        features.extend([
            np.mean(x_coords),
            np.mean(y_coords),
            np.std(x_coords),
            np.std(y_coords)
        ])
        
        # 2. Velocity features
        if len(x_coords) >= 2:
            x_velocities = np.diff(x_coords)
            y_velocities = np.diff(y_coords)
            
            features.extend([
                np.mean(x_velocities),
                np.mean(y_velocities),
                np.std(x_velocities),
                np.std(y_velocities)
            ])
        else:
            features.extend([0, 0, 0, 0])
        
        # 3. Acceleration features
        if len(x_coords) >= 3:
            x_accelerations = np.diff(x_velocities)
            y_accelerations = np.diff(y_velocities)
            
            features.extend([
                np.mean(x_accelerations),
                np.mean(y_accelerations),
                np.std(x_accelerations),
                np.std(y_accelerations)
            ])
        else:
            features.extend([0, 0, 0, 0])
        
        # 4. Trajectory shape features
        features.extend(self._compute_trajectory_shape_features(x_coords, y_coords))
        
        # 5. Bounce-specific features
        features.extend(self._compute_bounce_specific_features(x_coords, y_coords))
        
        return np.array(features)
    
    def _compute_trajectory_shape_features(self, x_coords: np.ndarray, y_coords: np.ndarray) -> List[float]:
        """Compute trajectory shape features"""
        features = []
        
        if len(x_coords) >= 3:
            # Total trajectory length
            total_length = np.sum(np.sqrt(np.diff(x_coords)**2 + np.diff(y_coords)**2))
            features.append(total_length)
            
            # Straight-line distance
            straight_distance = np.sqrt((x_coords[-1] - x_coords[0])**2 + (y_coords[-1] - y_coords[0])**2)
            features.append(straight_distance)
            
            # Curvature (how much the trajectory curves)
            if total_length > 0:
                curvature = straight_distance / total_length
                features.append(curvature)
            else:
                features.append(0)
            
            # Direction change (total angle change)
            total_angle_change = 0
            for i in range(1, len(x_coords)-1):
                dx1 = x_coords[i] - x_coords[i-1]
                dy1 = y_coords[i] - y_coords[i-1]
                dx2 = x_coords[i+1] - x_coords[i]
                dy2 = y_coords[i+1] - y_coords[i]
                
                if abs(dx1) > 0.1 or abs(dy1) > 0.1:
                    angle1 = np.arctan2(dy1, dx1)
                    angle2 = np.arctan2(dy2, dx2)
                    angle_change = abs(angle2 - angle1)
                    if angle_change > np.pi:
                        angle_change = 2*np.pi - angle_change
                    total_angle_change += angle_change
            
            features.append(total_angle_change)
        else:
            features.extend([0, 0, 0, 0])
        
        return features
    
    def _compute_bounce_specific_features(self, x_coords: np.ndarray, y_coords: np.ndarray) -> List[float]:
        """Compute features specific to bounce detection"""
        features = []
        
        if len(y_coords) >= 3:
            # Y-velocity reversal detection
            y_velocities = np.diff(y_coords)
            
            # Count velocity reversals
            reversals = 0
            for i in range(1, len(y_velocities)):
                if (y_velocities[i-1] < 0 and y_velocities[i] > 0) or \
                   (y_velocities[i-1] > 0 and y_velocities[i] < 0):
                    reversals += 1
            
            features.append(reversals)
            
            # Maximum velocity magnitude
            max_velocity = np.max(np.abs(y_velocities))
            features.append(max_velocity)
            
            # Y-position range (how much vertical movement)
            y_range = np.max(y_coords) - np.min(y_coords)
            features.append(y_range)
            
            # Y-velocity variance (how much velocity changes)
            y_vel_variance = np.var(y_velocities)
            features.append(y_vel_variance)
        else:
            features.extend([0, 0, 0, 0])
        
        return features
    
    def save_training_data(self, features: np.ndarray, labels: np.ndarray, output_file: str):
        """Save training data to file"""
        try:
            # Create feature names
            feature_names = [
                'mean_x', 'mean_y', 'std_x', 'std_y',
                'mean_vel_x', 'mean_vel_y', 'std_vel_x', 'std_vel_y',
                'mean_acc_x', 'mean_acc_y', 'std_acc_x', 'std_acc_y',
                'trajectory_length', 'straight_distance', 'curvature', 'total_angle_change',
                'velocity_reversals', 'max_velocity', 'y_range', 'y_vel_variance'
            ]
            
            # Create DataFrame
            df = pd.DataFrame(features, columns=feature_names)
            df['is_bounce'] = labels
            
            # Save to CSV
            df.to_csv(output_file, index=False)
            logger.info(f"Saved training data to {output_file}")
            
            # Save feature names for later use
            feature_names_file = output_file.replace('.csv', '_feature_names.json')
            with open(feature_names_file, 'w') as f:
                json.dump(feature_names, f, indent=2)
            logger.info(f"Saved feature names to {feature_names_file}")
            
        except Exception as e:
            logger.error(f"Error saving training data: {e}")
            raise


def main():
    """Main function for standalone usage"""
    parser = argparse.ArgumentParser(description='Tennis Ball Bounce Detection - ML Training Data Preparation')
    parser.add_argument('--annotations', required=True, help='CSV file with bounce annotations')
    parser.add_argument('--ball-data', required=True, help='CSV file with ball tracking data')
    parser.add_argument('--output', default='bounce_training_data.csv', help='Output CSV file for training data')
    parser.add_argument('--window-size', type=int, default=10, help='Window size for trajectory analysis')
    
    args = parser.parse_args()
    
    try:
        # Initialize trainer
        trainer = BounceMLTrainer(window_size=args.window_size)
        
        # Prepare training data
        features, labels = trainer.prepare_training_data(args.annotations, args.ball_data)
        
        # Save training data
        trainer.save_training_data(features, labels, args.output)
        
        logger.info("Training data preparation completed successfully!")
        
    except Exception as e:
        logger.error(f"Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
