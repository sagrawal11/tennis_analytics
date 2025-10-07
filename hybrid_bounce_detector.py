#!/usr/bin/env python3
"""
Hybrid Tennis Ball Bounce Detection System

Combines heuristic physics-based detection with machine learning validation
for robust bounce detection.
"""

import pandas as pd
import numpy as np
import logging
from typing import List, Tuple, Dict, Optional
import joblib
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')
logger = logging.getLogger(__name__)


class HybridBounceDetector:
    """Hybrid bounce detector combining heuristics and ML"""
    
    def __init__(self, ml_model_path: str = "advanced_models/random_forest_model.joblib", 
                 ml_scaler_path: str = None):
        """
        Initialize hybrid detector
        
        Args:
            ml_model_path: Path to trained ML model
            ml_scaler_path: Path to feature scaler (if needed)
        """
        self.ml_model = None
        self.ml_scaler = None
        self.last_bounce_frame = -1
        self.min_bounce_gap = 5  # Minimum frames between bounces
        
        # Load ML model if available
        if Path(ml_model_path).exists():
            try:
                self.ml_model = joblib.load(ml_model_path)
                logger.info(f"Loaded ML model from {ml_model_path}")
                
                # Load scaler if provided
                if ml_scaler_path and Path(ml_scaler_path).exists():
                    self.ml_scaler = joblib.load(ml_scaler_path)
                    logger.info(f"Loaded ML scaler from {ml_scaler_path}")
                    
            except Exception as e:
                logger.warning(f"Could not load ML model: {e}")
        else:
            logger.warning(f"ML model not found at {ml_model_path}, using heuristics only")
    
    def detect_bounce(self, ball_trajectory: List[Tuple[float, float, int]], 
                     current_frame: int) -> Tuple[bool, float, Dict]:
        """
        Detect bounce using hybrid approach
        
        Args:
            ball_trajectory: List of (x, y, frame) tuples
            current_frame: Current frame number
            
        Returns:
            (is_bounce, confidence, details)
        """
        # Skip if too soon after last bounce
        if current_frame - self.last_bounce_frame < self.min_bounce_gap:
            return False, 0.0, {"reason": "too_soon_after_last_bounce"}
        
        # Get heuristic confidence
        heuristic_confidence = self._heuristic_bounce_detection(ball_trajectory, current_frame)
        
        # Get ML confidence if model available
        ml_confidence = 0.0
        if self.ml_model is not None:
            ml_confidence = self._ml_bounce_detection(ball_trajectory, current_frame)
        
        # Combine confidences using weighted average
        if self.ml_model is not None:
            # Hybrid: 60% heuristics, 40% ML
            combined_confidence = 0.6 * heuristic_confidence + 0.4 * ml_confidence
            method = "hybrid"
        else:
            # Heuristics only
            combined_confidence = heuristic_confidence
            method = "heuristics_only"
        
        # Decision threshold
        bounce_threshold = 0.7
        
        is_bounce = combined_confidence >= bounce_threshold
        
        if is_bounce:
            self.last_bounce_frame = current_frame
        
        details = {
            "method": method,
            "heuristic_confidence": heuristic_confidence,
            "ml_confidence": ml_confidence,
            "combined_confidence": combined_confidence,
            "threshold": bounce_threshold
        }
        
        return is_bounce, combined_confidence, details
    
    def _heuristic_bounce_detection(self, ball_trajectory: List[Tuple[float, float, int]], 
                                   current_frame: int) -> float:
        """Heuristic bounce detection based on physics"""
        try:
            # Get recent trajectory data (last 10 frames)
            recent_trajectory = [(x, y, frame) for x, y, frame in ball_trajectory 
                               if frame >= current_frame - 10 and x is not None and y is not None]
            
            if len(recent_trajectory) < 5:
                return 0.0
            
            # Sort by frame
            recent_trajectory.sort(key=lambda x: x[2])
            
            x_coords = [pos[0] for pos in recent_trajectory]
            y_coords = [pos[1] for pos in recent_trajectory]
            
            confidence = 0.0
            
            # 1. Y-velocity reversal (most important)
            y_velocity_score = self._analyze_y_velocity_reversal(y_coords)
            confidence += y_velocity_score * 0.4
            
            # 2. Trajectory curvature
            curvature_score = self._analyze_trajectory_curvature(x_coords, y_coords)
            confidence += curvature_score * 0.3
            
            # 3. Speed changes
            speed_score = self._analyze_speed_changes(x_coords, y_coords)
            confidence += speed_score * 0.2
            
            # 4. Acceleration patterns
            acceleration_score = self._analyze_acceleration_patterns(x_coords, y_coords)
            confidence += acceleration_score * 0.1
            
            return min(1.0, confidence)
            
        except Exception as e:
            logger.warning(f"Error in heuristic detection: {e}")
            return 0.0
    
    def _ml_bounce_detection(self, ball_trajectory: List[Tuple[float, float, int]], 
                           current_frame: int) -> float:
        """ML-based bounce detection"""
        try:
            # Get window around current frame
            window_data = [(x, y, frame) for x, y, frame in ball_trajectory 
                          if abs(frame - current_frame) <= 10 and x is not None and y is not None]
            
            if len(window_data) < 5:
                return 0.0
            
            # Sort by frame
            window_data.sort(key=lambda x: x[2])
            
            # Extract features (simplified version of advanced features)
            features = self._extract_ml_features(window_data)
            
            if features is None:
                return 0.0
            
            # Scale features if scaler available
            if self.ml_scaler is not None:
                features = self.ml_scaler.transform([features])[0]
            
            # Predict probability
            proba = self.ml_model.predict_proba([features])[0]
            bounce_prob = proba[1] if len(proba) > 1 else 0.0
            
            return bounce_prob
            
        except Exception as e:
            logger.warning(f"Error in ML detection: {e}")
            return 0.0
    
    def _extract_ml_features(self, window_data: List[Tuple[float, float, int]]) -> Optional[np.ndarray]:
        """Extract simplified ML features"""
        try:
            if len(window_data) < 5:
                return None
            
            x_coords = np.array([pos[0] for pos in window_data])
            y_coords = np.array([pos[1] for pos in window_data])
            
            features = []
            
            # Basic physics features
            y_velocities = np.diff(y_coords)
            
            # Velocity reversal
            reversals = 0
            for i in range(1, len(y_velocities)):
                if (y_velocities[i-1] < 0 and y_velocities[i] > 0) or \
                   (y_velocities[i-1] > 0 and y_velocities[i] < 0):
                    reversals += 1
            
            features.extend([
                len(y_velocities),  # trajectory_length
                np.std(y_coords),   # y_std
                reversals,          # velocity_reversals
                np.max(np.abs(y_velocities)),  # max_velocity
                np.max(y_coords) - np.min(y_coords),  # y_range
                np.var(y_velocities)  # y_vel_variance
            ])
            
            # Pad to expected feature count (24)
            while len(features) < 24:
                features.append(0.0)
            
            return np.array(features[:24])
            
        except Exception as e:
            logger.warning(f"Error extracting ML features: {e}")
            return None
    
    def _analyze_y_velocity_reversal(self, y_coords: List[float]) -> float:
        """Analyze Y-velocity reversal patterns"""
        if len(y_coords) < 3:
            return 0.0
        
        y_velocities = np.diff(y_coords)
        
        # Count reversals
        reversals = 0
        for i in range(1, len(y_velocities)):
            if (y_velocities[i-1] < 0 and y_velocities[i] > 0) or \
               (y_velocities[i-1] > 0 and y_velocities[i] < 0):
                reversals += 1
        
        # Score based on reversal frequency
        max_possible_reversals = len(y_velocities) - 1
        if max_possible_reversals > 0:
            reversal_ratio = reversals / max_possible_reversals
            return min(1.0, reversal_ratio * 2)  # Scale up
        
        return 0.0
    
    def _analyze_trajectory_curvature(self, x_coords: List[float], y_coords: List[float]) -> float:
        """Analyze trajectory curvature"""
        if len(x_coords) < 3:
            return 0.0
        
        # Calculate total curvature
        total_curvature = 0.0
        for i in range(1, len(x_coords)-1):
            # Simple curvature approximation
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
                total_curvature += angle_change
        
        # Normalize by trajectory length
        trajectory_length = len(x_coords)
        if trajectory_length > 0:
            normalized_curvature = total_curvature / trajectory_length
            return min(1.0, normalized_curvature / np.pi)  # Scale to [0,1]
        
        return 0.0
    
    def _analyze_speed_changes(self, x_coords: List[float], y_coords: List[float]) -> float:
        """Analyze speed changes"""
        if len(x_coords) < 3:
            return 0.0
        
        # Calculate speeds
        speeds = []
        for i in range(1, len(x_coords)):
            dx = x_coords[i] - x_coords[i-1]
            dy = y_coords[i] - y_coords[i-1]
            speed = np.sqrt(dx*dx + dy*dy)
            speeds.append(speed)
        
        if len(speeds) < 2:
            return 0.0
        
        # Calculate speed changes
        speed_changes = np.abs(np.diff(speeds))
        
        # Score based on speed change magnitude
        avg_speed = np.mean(speeds)
        if avg_speed > 0:
            normalized_changes = speed_changes / avg_speed
            return min(1.0, np.mean(normalized_changes))
        
        return 0.0
    
    def _analyze_acceleration_patterns(self, x_coords: List[float], y_coords: List[float]) -> float:
        """Analyze acceleration patterns"""
        if len(x_coords) < 4:
            return 0.0
        
        # Calculate accelerations
        x_velocities = np.diff(x_coords)
        y_velocities = np.diff(y_coords)
        
        if len(x_velocities) < 2:
            return 0.0
        
        x_accelerations = np.diff(x_velocities)
        y_accelerations = np.diff(y_velocities)
        
        # Calculate acceleration magnitude
        accel_magnitudes = np.sqrt(x_accelerations*x_accelerations + y_accelerations*y_accelerations)
        
        # Score based on acceleration variance
        if len(accel_magnitudes) > 0:
            accel_variance = np.var(accel_magnitudes)
            return min(1.0, accel_variance / 100.0)  # Scale down
        
        return 0.0


def test_hybrid_detector():
    """Test the hybrid detector with sample data"""
    detector = HybridBounceDetector()
    
    # Sample trajectory with a bounce
    sample_trajectory = [
        (100, 200, 1), (105, 190, 2), (110, 180, 3), (115, 170, 4),
        (120, 160, 5), (125, 150, 6), (130, 140, 7), (135, 130, 8),
        (140, 120, 9), (145, 110, 10), (150, 100, 11), (155, 110, 12),  # Bounce here
        (160, 120, 13), (165, 130, 14), (170, 140, 15)
    ]
    
    # Test detection
    for frame in range(5, 15):
        is_bounce, confidence, details = detector.detect_bounce(sample_trajectory, frame)
        if is_bounce:
            print(f"Frame {frame}: Bounce detected! Confidence: {confidence:.3f}")
            print(f"  Details: {details}")
            break


if __name__ == "__main__":
    test_hybrid_detector()
