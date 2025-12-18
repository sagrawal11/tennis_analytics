#!/usr/bin/env python3
"""
Sequence-Based Tennis Ball Bounce Detector

Uses the sequence model that achieved 68% AUC with proper feature extraction.
"""

import pandas as pd
import numpy as np
import logging
import joblib
from typing import List, Tuple, Optional
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')
logger = logging.getLogger(__name__)


class SequenceBounceDetector:
    """Sequence-based bounce detector using the working 68% AUC model"""
    
    def __init__(self, model_path: str = "sequence_models/sequence_random_forest_model.joblib",
                 scaler_path: str = "sequence_models/sequence_scaler.joblib",
                 threshold: float = 0.5):
        """
        Initialize sequence detector
        
        Args:
            model_path: Path to sequence model
            scaler_path: Path to feature scaler
            threshold: Bounce detection threshold
        """
        self.model = None
        self.scaler = None
        self.threshold = threshold
        self.last_bounce_frame = -10  # Minimum gap between bounces
        self.window_size = 15  # Same as training
        
        # Load model and scaler
        self._load_model_and_scaler(model_path, scaler_path)
        
    def _load_model_and_scaler(self, model_path: str, scaler_path: str):
        """Load the sequence model and scaler"""
        try:
            if Path(model_path).exists():
                self.model = joblib.load(model_path)
                logger.info(f"Loaded sequence model from {model_path}")
            else:
                logger.error(f"Model not found: {model_path}")
                return
            
            if Path(scaler_path).exists():
                self.scaler = joblib.load(scaler_path)
                logger.info(f"Loaded scaler from {scaler_path}")
            else:
                logger.error(f"Scaler not found: {scaler_path}")
                
        except Exception as e:
            logger.error(f"Error loading model/scaler: {e}")
    
    def detect_bounce(self, ball_trajectory: List[Tuple[float, float, int]], 
                     current_frame: int) -> Tuple[bool, float]:
        """
        Sequence-based bounce detection
        
        Returns:
            (is_bounce, confidence)
        """
        # Skip if too soon after last bounce
        if current_frame - self.last_bounce_frame < 5:
            return False, 0.0
        
        # Extract sequence features
        features = self._extract_sequence_features(ball_trajectory, current_frame)
        
        if features is None:
            return False, 0.0
        
        # Get model prediction
        confidence = self._get_model_prediction(features)
        
        # Simple threshold decision
        is_bounce = confidence >= self.threshold
        
        if is_bounce:
            self.last_bounce_frame = current_frame
        
        return is_bounce, confidence
    
    def _extract_sequence_features(self, ball_trajectory: List[Tuple[float, float, int]], 
                                  current_frame: int) -> Optional[np.ndarray]:
        """Extract sequence features using the same logic as training"""
        try:
            # Get window around current frame
            window_start = current_frame - self.window_size // 2
            window_end = current_frame + self.window_size // 2
            
            # Extract ball positions in window
            window_data = [(x, y, frame) for x, y, frame in ball_trajectory 
                          if window_start <= frame <= window_end and x is not None and y is not None]
            
            if len(window_data) < self.window_size * 0.8:  # Need 80% coverage
                return None
            
            # Sort by frame and extract coordinates
            window_data.sort(key=lambda x: x[2])
            x_coords = np.array([pos[0] for pos in window_data])
            y_coords = np.array([pos[1] for pos in window_data])
            
            # Quality check
            if len(x_coords) < 5 or np.std(x_coords) < 2 or np.std(y_coords) < 2:
                return None
            
            # Interpolate missing data (same as training)
            x_coords, y_coords = self._interpolate_missing_data(x_coords, y_coords)
            
            # Extract the same 34 features as the sequence trainer
            features = self._compute_sequence_features(x_coords, y_coords)
            
            if features is None or len(features) != 34:
                return None
            
            return np.array(features)
            
        except Exception as e:
            logger.warning(f"Error extracting sequence features: {e}")
            return None
    
    def _interpolate_missing_data(self, x_coords: np.ndarray, y_coords: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Interpolate missing data points"""
        try:
            # Create full frame range
            full_frames = np.arange(len(x_coords))
            
            # Find valid data points
            valid_mask = ~(np.isnan(x_coords) | np.isnan(y_coords))
            
            if np.sum(valid_mask) < 3:
                return x_coords, y_coords
            
            valid_frames = full_frames[valid_mask]
            valid_x = x_coords[valid_mask]
            valid_y = y_coords[valid_mask]
            
            # Interpolate missing values
            if len(valid_frames) > 1:
                x_interpolated = np.interp(full_frames, valid_frames, valid_x)
                y_interpolated = np.interp(full_frames, valid_frames, valid_y)
                return x_interpolated, y_interpolated
            else:
                return x_coords, y_coords
                
        except Exception as e:
            logger.warning(f"Interpolation failed: {e}")
            return x_coords, y_coords
    
    def _compute_sequence_features(self, x_coords: np.ndarray, y_coords: np.ndarray) -> Optional[List[float]]:
        """Compute sequence features - same as sequence trainer"""
        try:
            features = []
            
            # 1. Position statistics (4 features)
            features.extend([np.mean(x_coords), np.mean(y_coords), np.std(x_coords), np.std(y_coords)])
            
            # 2. Velocity features (8 features)
            if len(x_coords) >= 2:
                x_vel = np.diff(x_coords)
                y_vel = np.diff(y_coords)
                features.extend([np.mean(x_vel), np.mean(y_vel), np.std(x_vel), np.std(y_vel)])
                
                # 3. Acceleration features (8 features)
                if len(x_vel) >= 2:
                    x_accel = np.diff(x_vel)
                    y_accel = np.diff(y_vel)
                    features.extend([np.mean(x_accel), np.mean(y_accel), np.std(x_accel), np.std(y_accel)])
                else:
                    features.extend([0.0, 0.0, 0.0, 0.0])
            else:
                features.extend([0.0] * 8)
            
            # 4. Trajectory shape features (4 features)
            if len(x_coords) >= 3:
                # Trajectory length
                total_length = np.sum(np.sqrt(np.diff(x_coords)**2 + np.diff(y_coords)**2))
                features.append(total_length)
                
                # Straight distance
                straight_distance = np.sqrt((x_coords[-1] - x_coords[0])**2 + (y_coords[-1] - y_coords[0])**2)
                features.append(straight_distance)
                
                # Curvature (simplified)
                if len(x_coords) >= 5:
                    curvature = np.std(np.diff(np.diff(y_coords)))
                    features.append(curvature)
                else:
                    features.append(0.0)
                
                # Total angle change
                if len(x_coords) >= 3:
                    angles = np.arctan2(np.diff(y_coords), np.diff(x_coords))
                    angle_changes = np.diff(angles)
                    total_angle_change = np.sum(np.abs(angle_changes))
                    features.append(total_angle_change)
                else:
                    features.append(0.0)
            else:
                features.extend([0.0] * 4)
            
            # 5. Bounce-specific features (4 features)
            if len(y_coords) >= 3:
                y_vel = np.diff(y_coords)
                
                # Velocity reversals
                reversals = 0
                for i in range(1, len(y_vel)):
                    if (y_vel[i-1] < 0 and y_vel[i] > 0) or (y_vel[i-1] > 0 and y_vel[i] < 0):
                        reversals += 1
                features.append(reversals)
                
                # Max velocity
                max_velocity = np.max(np.abs(y_vel))
                features.append(max_velocity)
                
                # Y range
                y_range = np.max(y_coords) - np.min(y_coords)
                features.append(y_range)
                
                # Y velocity variance
                y_vel_variance = np.var(y_vel)
                features.append(y_vel_variance)
            else:
                features.extend([0.0] * 4)
            
            # 6. Min/Max position features (8 features)
            if len(y_coords) >= 3:
                y_min_pos = np.min(y_coords)
                y_max_pos = np.max(y_coords)
                x_min_pos = np.min(x_coords)
                x_max_pos = np.max(x_coords)
                
                features.extend([y_min_pos, y_max_pos, x_min_pos, x_max_pos])
                
                # Velocity at extreme points
                y_vel = np.diff(y_coords)
                min_y_idx = np.argmin(y_coords)
                max_y_idx = np.argmax(y_coords)
                
                if min_y_idx < len(y_vel):
                    y_vel_at_min_y = y_vel[min_y_idx]
                else:
                    y_vel_at_min_y = 0.0
                
                if max_y_idx < len(y_vel):
                    y_vel_at_max_y = y_vel[max_y_idx]
                else:
                    y_vel_at_max_y = 0.0
                
                features.extend([y_vel_at_min_y, y_vel_at_max_y])
                
                # Time to extreme points
                time_to_min_y = min_y_idx / len(y_coords)
                time_to_max_y = max_y_idx / len(y_coords)
                features.extend([time_to_min_y, time_to_max_y])
            else:
                features.extend([0.0] * 8)
            
            # 7. Speed features (2 features)
            if len(x_coords) >= 2:
                speeds = np.sqrt(np.diff(x_coords)**2 + np.diff(y_coords)**2)
                avg_speed = np.mean(speeds)
                speed_std = np.std(speeds)
                features.extend([avg_speed, speed_std])
            else:
                features.extend([0.0, 0.0])
            
            # 8. Jerk features (2 features)
            if len(x_coords) >= 4:
                x_vel = np.diff(x_coords)
                y_vel = np.diff(y_coords)
                x_accel = np.diff(x_vel)
                y_accel = np.diff(y_vel)
                x_jerk = np.diff(x_accel)
                y_jerk = np.diff(y_accel)
                
                jerk_magnitude = np.sqrt(x_jerk**2 + y_jerk**2)
                features.extend([np.mean(jerk_magnitude), np.std(jerk_magnitude)])
            else:
                features.extend([0.0, 0.0])
            
            # 9. Prominence features (2 features)
            if len(y_coords) >= 5:
                # Simple prominence calculation
                y_peak_prominence = np.max(y_coords) - np.mean(y_coords)
                y_trough_prominence = np.mean(y_coords) - np.min(y_coords)
                features.extend([y_peak_prominence, y_trough_prominence])
            else:
                features.extend([0.0, 0.0])
            
            return features if len(features) == 34 else None
            
        except Exception as e:
            logger.warning(f"Sequence feature computation error: {e}")
            return None
    
    def _get_model_prediction(self, features: np.ndarray) -> float:
        """Get prediction from the sequence model"""
        try:
            if self.model is None:
                return 0.0
            
            # Scale features
            if self.scaler is not None:
                features_scaled = self.scaler.transform([features])[0]
            else:
                features_scaled = features
            
            # Predict probability
            proba = self.model.predict_proba([features_scaled])[0]
            return proba[1] if len(proba) > 1 else 0.0
            
        except Exception as e:
            logger.warning(f"Error in model prediction: {e}")
            return 0.0


def test_sequence_detector():
    """Test the sequence detector"""
    detector = SequenceBounceDetector(threshold=0.3)  # Lower threshold
    
    # Sample trajectory with bounce
    sample_trajectory = [
        (100, 200, 1), (105, 190, 2), (110, 180, 3), (115, 170, 4),
        (120, 160, 5), (125, 150, 6), (130, 140, 7), (135, 130, 8),
        (140, 120, 9), (145, 110, 10), (150, 100, 11), (155, 110, 12),
        (160, 120, 13), (165, 130, 14), (170, 140, 15), (175, 150, 16)
    ]
    
    # Test detection
    for frame in range(8, 16):
        is_bounce, confidence = detector.detect_bounce(sample_trajectory, frame)
        if is_bounce:
            print(f"Frame {frame}: Sequence bounce detected! Confidence: {confidence:.3f}")
            break


if __name__ == "__main__":
    test_sequence_detector()
