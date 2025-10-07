#!/usr/bin/env python3
"""
Simple Ultra-Advanced Bounce Detector

Simplified version of the ultra-advanced model without complex temporal smoothing.
Focuses on the 72% AUC performance with straightforward thresholding.
"""

import pandas as pd
import numpy as np
import logging
import joblib
from typing import List, Tuple, Optional
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')
logger = logging.getLogger(__name__)


class SimpleUltraDetector:
    """Simple ultra-advanced bounce detector"""
    
    def __init__(self, model_path: str = "ultra_models/ultra_random_forest_model.joblib",
                 scaler_path: str = "ultra_models/ultra_scaler.joblib",
                 threshold: float = 0.5):
        """
        Initialize simple detector
        
        Args:
            model_path: Path to ultra-advanced model
            scaler_path: Path to feature scaler
            threshold: Bounce detection threshold
        """
        self.model = None
        self.scaler = None
        self.threshold = threshold
        self.last_bounce_frame = -10  # Minimum gap between bounces
        
        # Load model and scaler
        self._load_model_and_scaler(model_path, scaler_path)
        
    def _load_model_and_scaler(self, model_path: str, scaler_path: str):
        """Load the ultra-advanced model and scaler"""
        try:
            if Path(model_path).exists():
                self.model = joblib.load(model_path)
                logger.info(f"Loaded ultra model from {model_path}")
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
        Simple bounce detection
        
        Returns:
            (is_bounce, confidence)
        """
        # Skip if too soon after last bounce
        if current_frame - self.last_bounce_frame < 5:
            return False, 0.0
        
        # Extract features using the same logic as ultra-advanced trainer
        features = self._extract_ultra_features(ball_trajectory, current_frame)
        
        if features is None:
            return False, 0.0
        
        # Get model prediction
        confidence = self._get_model_prediction(features)
        
        # Simple threshold decision
        is_bounce = confidence >= self.threshold
        
        if is_bounce:
            self.last_bounce_frame = current_frame
        
        return is_bounce, confidence
    
    def _extract_ultra_features(self, ball_trajectory: List[Tuple[float, float, int]], 
                               current_frame: int) -> Optional[np.ndarray]:
        """Extract ultra-advanced features using the exact same logic as training"""
        try:
            # Get window around current frame
            window_size = 20
            window_start = current_frame - window_size // 2
            window_end = current_frame + window_size // 2
            
            # Extract ball positions in window
            window_data = [(x, y, frame) for x, y, frame in ball_trajectory 
                          if window_start <= frame <= window_end and x is not None and y is not None]
            
            if len(window_data) < window_size * 0.7:  # Need 70% coverage
                return None
            
            # Sort by frame and extract coordinates
            window_data.sort(key=lambda x: x[2])
            x_coords = np.array([pos[0] for pos in window_data])
            y_coords = np.array([pos[1] for pos in window_data])
            
            # Quality check
            if len(x_coords) < 5 or np.std(x_coords) < 2 or np.std(y_coords) < 2:
                return None
            
            # Extract the same 58 features as the ultra-advanced trainer
            features = self._compute_ultra_features(x_coords, y_coords)
            
            if features is None or len(features) != 58:
                return None
            
            return np.array(features)
            
        except Exception as e:
            logger.warning(f"Error extracting features: {e}")
            return None
    
    def _compute_ultra_features(self, x_coords: np.ndarray, y_coords: np.ndarray) -> Optional[List[float]]:
        """Compute ultra-advanced features - simplified version of the training logic"""
        try:
            features = []
            
            # 1. Basic physics features (15 features)
            features.extend(self._compute_basic_physics_features(x_coords, y_coords))
            
            # 2. Trajectory features (12 features)  
            features.extend(self._compute_trajectory_features(x_coords, y_coords))
            
            # 3. Temporal features (10 features)
            features.extend(self._compute_temporal_features(x_coords, y_coords))
            
            # 4. Multi-scale features (8 features)
            features.extend(self._compute_multiscale_features(x_coords, y_coords))
            
            # 5. Statistical features (10 features)
            features.extend(self._compute_statistical_features(x_coords, y_coords))
            
            # 6. Additional features (3 features)
            features.extend(self._compute_additional_features(x_coords, y_coords))
            
            return features if len(features) == 58 else None
            
        except Exception as e:
            logger.warning(f"Error computing features: {e}")
            return None
    
    def _compute_basic_physics_features(self, x_coords: np.ndarray, y_coords: np.ndarray) -> List[float]:
        """Compute basic physics features"""
        features = []
        
        if len(y_coords) < 5:
            return [0.0] * 15
        
        # Y-velocity analysis
        y_vel = np.diff(y_coords)
        if len(y_vel) < 3:
            return [0.0] * 15
        
        # Find minimum y (potential bounce point)
        min_y_idx = np.argmin(y_coords)
        
        # Impact and rebound analysis
        if min_y_idx > 0 and min_y_idx < len(y_vel):
            impact_velocity = abs(y_vel[min_y_idx-1])
            rebound_velocity = y_vel[min_y_idx]
            velocity_ratio = rebound_velocity / (impact_velocity + 1e-6)
            energy_loss = 1.0 - velocity_ratio**2
        else:
            impact_velocity = rebound_velocity = velocity_ratio = energy_loss = 0.0
        
        features.extend([impact_velocity, rebound_velocity, velocity_ratio, energy_loss])
        
        # Bounce angle analysis
        if len(x_coords) >= 3:
            x_vel = np.diff(x_coords)
            if len(x_vel) >= 2:
                angle_before = np.arctan2(y_vel[len(y_vel)//2-1], x_vel[len(x_vel)//2-1])
                angle_after = np.arctan2(y_vel[len(y_vel)//2], x_vel[len(x_vel)//2])
                bounce_angle = abs(angle_after - angle_before)
                trajectory_curvature = abs(angle_after - angle_before) / (len(y_coords) + 1e-6)
            else:
                bounce_angle = trajectory_curvature = 0.0
        else:
            bounce_angle = trajectory_curvature = 0.0
        
        features.extend([bounce_angle, trajectory_curvature])
        
        # Acceleration analysis
        if len(y_vel) >= 2:
            y_accel = np.diff(y_vel)
            acceleration_spike = np.max(np.abs(y_accel))
            deceleration_spike = np.min(y_accel)
        else:
            acceleration_spike = deceleration_spike = 0.0
        
        features.extend([acceleration_spike, deceleration_spike])
        
        # Ground contact analysis
        min_y = np.min(y_coords)
        max_y = np.max(y_coords)
        y_range = max_y - min_y
        ground_threshold = min_y + y_range * 0.1
        ground_frames = np.sum(y_coords <= ground_threshold)
        ground_contact_duration = ground_frames / len(y_coords)
        air_time_ratio = 1.0 - ground_contact_duration
        
        features.extend([ground_contact_duration, air_time_ratio])
        
        # Velocity reversal strength
        reversals = 0
        reversal_strength = 0
        for i in range(1, len(y_vel)):
            if (y_vel[i-1] < 0 and y_vel[i] > 0) or (y_vel[i-1] > 0 and y_vel[i] < 0):
                reversals += 1
                reversal_strength += abs(y_vel[i-1] - y_vel[i])
        
        vertical_velocity_reversal_strength = reversal_strength / (reversals + 1e-6)
        
        # Additional physics features
        if len(x_coords) >= 3:
            x_vel = np.diff(x_coords)
            horizontal_velocity_change = np.std(x_vel)
            impact_force_estimate = acceleration_spike * 0.01  # Rough estimate
            bounce_efficiency = velocity_ratio
            collision_elasticity = min(1.0, velocity_ratio)
        else:
            horizontal_velocity_change = impact_force_estimate = bounce_efficiency = collision_elasticity = 0.0
        
        features.extend([vertical_velocity_reversal_strength, horizontal_velocity_change, 
                        impact_force_estimate, bounce_efficiency, collision_elasticity])
        
        return features
    
    def _compute_trajectory_features(self, x_coords: np.ndarray, y_coords: np.ndarray) -> List[float]:
        """Compute trajectory analysis features"""
        features = []
        
        if len(x_coords) < 5:
            return [0.0] * 12
        
        # Trajectory smoothness
        if len(x_coords) >= 3:
            x_vel = np.diff(x_coords)
            y_vel = np.diff(y_coords)
            x_accel = np.diff(x_vel) if len(x_vel) > 1 else np.array([0])
            y_accel = np.diff(y_vel) if len(y_vel) > 1 else np.array([0])
            
            trajectory_smoothness = 1.0 / (np.var(x_accel) + np.var(y_accel) + 1e-6)
            direction_change_frequency = len(np.where(np.diff(np.sign(y_vel)))[0]) / len(y_vel)
            curvature_variance = np.var(x_accel) + np.var(y_accel)
            path_deviation = np.std(y_coords)
        else:
            trajectory_smoothness = direction_change_frequency = curvature_variance = path_deviation = 0.0
        
        features.extend([trajectory_smoothness, direction_change_frequency, curvature_variance, path_deviation])
        
        # Motion analysis
        if len(y_coords) >= 3:
            trajectory_complexity = len(np.where(np.diff(np.sign(np.diff(y_coords))))[0]) / len(y_coords)
            motion_consistency = 1.0 / (np.std(y_coords) + 1e-6)
            velocity_variance = np.var(y_vel)
            acceleration_variance = np.var(y_accel) if len(y_accel) > 0 else 0
        else:
            trajectory_complexity = motion_consistency = velocity_variance = acceleration_variance = 0.0
        
        features.extend([trajectory_complexity, motion_consistency, velocity_variance, acceleration_variance])
        
        # Additional trajectory features
        jerk_magnitude = np.mean(np.abs(y_accel)) if len(y_accel) > 0 else 0
        trajectory_symmetry = 1.0 - abs(np.mean(y_coords[:len(y_coords)//2]) - np.mean(y_coords[len(y_coords)//2:])) / (np.std(y_coords) + 1e-6)
        motion_regularity = 1.0 / (np.var(y_vel) + 1e-6)
        path_optimality = 1.0 / (np.std(y_coords) + 1e-6)
        
        features.extend([jerk_magnitude, trajectory_symmetry, motion_regularity, path_optimality])
        
        return features
    
    def _compute_temporal_features(self, x_coords: np.ndarray, y_coords: np.ndarray) -> List[float]:
        """Compute temporal pattern features"""
        features = []
        
        if len(y_coords) < 5:
            return [0.0] * 10
        
        # Oscillation analysis
        y_vel = np.diff(y_coords)
        zero_crossings = len(np.where(np.diff(np.sign(y_vel)))[0])
        oscillation_frequency = zero_crossings / len(y_vel)
        bounce_periodicity = oscillation_frequency
        
        # Temporal consistency
        temporal_consistency = 1.0 / (np.std(y_coords) + 1e-6)
        
        # Pattern analysis
        pattern_regularity = 1.0 / (np.var(y_vel) + 1e-6)
        sequence_complexity = len(np.unique(np.round(y_coords, 2))) / len(y_coords)
        
        # Time domain features
        time_domain_features = np.mean(np.abs(y_vel))
        frequency_domain_features = np.std(y_vel)
        spectral_energy = np.sum(y_vel**2) / len(y_vel)
        
        # Correlation analysis
        if len(y_coords) >= 10:
            correlation = np.corrcoef(y_coords[:-5], y_coords[5:])[0, 1]
            temporal_correlation = abs(correlation) if not np.isnan(correlation) else 0.0
        else:
            temporal_correlation = 0.0
        
        # Pattern matching
        pattern_matching_score = 1.0 / (np.var(y_coords) + 1e-6)
        
        features.extend([oscillation_frequency, bounce_periodicity, temporal_consistency, 
                        pattern_regularity, sequence_complexity, time_domain_features,
                        frequency_domain_features, spectral_energy, temporal_correlation,
                        pattern_matching_score])
        
        return features
    
    def _compute_multiscale_features(self, x_coords: np.ndarray, y_coords: np.ndarray) -> List[float]:
        """Compute multi-scale analysis features"""
        features = []
        
        # Simple multi-scale analysis
        scales = [2, 4, 8, 16]
        
        for scale in scales:
            if len(y_coords) >= scale * 2:
                downsampled = y_coords[::scale]
                micro_features = np.std(downsampled)
                meso_features = np.mean(downsampled)
            else:
                micro_features = meso_features = 0.0
            
            features.extend([micro_features, meso_features])
        
        return features
    
    def _compute_statistical_features(self, x_coords: np.ndarray, y_coords: np.ndarray) -> List[float]:
        """Compute statistical analysis features"""
        features = []
        
        # Statistical moments
        mean_y = np.mean(y_coords)
        var_y = np.var(y_coords)
        skewness = np.mean((y_coords - mean_y)**3) / (var_y**1.5 + 1e-6)
        kurtosis = np.mean((y_coords - mean_y)**4) / (var_y**2 + 1e-6)
        
        features.extend([skewness, kurtosis, var_y])
        
        # Distribution analysis
        statistical_moments = np.mean(y_coords)
        distribution_shape = skewness + kurtosis
        outlier_detection = np.sum(np.abs(y_coords - mean_y) > 2 * np.std(y_coords)) / len(y_coords)
        anomaly_score = outlier_detection
        
        features.extend([statistical_moments, distribution_shape, outlier_detection, anomaly_score])
        
        # Statistical significance
        statistical_significance = 1.0 / (var_y + 1e-6)
        confidence_interval = np.std(y_coords) / np.sqrt(len(y_coords))
        hypothesis_test_pvalue = 1.0 / (np.var(y_coords) + 1e-6)
        effect_size = np.std(y_coords)
        
        features.extend([statistical_significance, confidence_interval, hypothesis_test_pvalue, effect_size])
        
        return features
    
    def _compute_additional_features(self, x_coords: np.ndarray, y_coords: np.ndarray) -> List[float]:
        """Compute additional features"""
        features = []
        
        # Correlation strength
        if len(y_coords) >= 3:
            correlation_strength = np.corrcoef(x_coords, y_coords)[0, 1]
            correlation_strength = abs(correlation_strength) if not np.isnan(correlation_strength) else 0.0
        else:
            correlation_strength = 0.0
        
        # Mutual information (simplified)
        mutual_information = 1.0 / (np.std(y_coords) + 1e-6)
        
        # Additional physics feature
        if len(y_coords) >= 5:
            y_vel = np.diff(y_coords)
            mid_point = len(y_vel) // 2
            momentum_before = np.mean(y_vel[:mid_point])
            momentum_after = np.mean(y_vel[mid_point:])
            momentum_ratio = momentum_after / (momentum_before + 1e-6)
            momentum_ratio = min(1.0, max(-1.0, momentum_ratio))
        else:
            momentum_ratio = 0.0
        
        features.extend([correlation_strength, mutual_information, momentum_ratio])
        
        return features
    
    def _get_model_prediction(self, features: np.ndarray) -> float:
        """Get prediction from the ultra-advanced model"""
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


def test_simple_detector():
    """Test the simple detector"""
    detector = SimpleUltraDetector(threshold=0.3)  # Lower threshold
    
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
            print(f"Frame {frame}: Simple bounce detected! Confidence: {confidence:.3f}")
            break


if __name__ == "__main__":
    test_simple_detector()
