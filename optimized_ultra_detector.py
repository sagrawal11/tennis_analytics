#!/usr/bin/env python3
"""
Optimized Ultra-Advanced Tennis Ball Bounce Detection

Optimized version of the best performing model (72% AUC) with temporal smoothing
and confidence calibration to push toward 90% performance.
"""

import pandas as pd
import numpy as np
import logging
from typing import List, Tuple, Dict, Optional
import joblib
from pathlib import Path
from collections import deque

logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')
logger = logging.getLogger(__name__)


class OptimizedUltraDetector:
    """Optimized ultra-advanced bounce detector with temporal smoothing"""
    
    def __init__(self, model_path: str = "ultra_models/ultra_random_forest_model.joblib",
                 scaler_path: str = "ultra_models/ultra_scaler.joblib"):
        """
        Initialize optimized detector
        
        Args:
            model_path: Path to ultra-advanced model
            scaler_path: Path to feature scaler
        """
        self.model = None
        self.scaler = None
        self.last_bounce_frame = -1
        self.min_bounce_gap = 2
        
        # Temporal smoothing buffers
        self.confidence_history = deque(maxlen=10)  # Last 10 confidence scores
        self.feature_history = deque(maxlen=5)      # Last 5 feature vectors
        self.bounce_candidates = []                 # Potential bounce frames
        
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
    
    def detect_bounce_optimized(self, ball_trajectory: List[Tuple[float, float, int]], 
                               current_frame: int) -> Tuple[bool, float, Dict]:
        """
        Optimized bounce detection with temporal smoothing and confidence calibration
        """
        # Skip if too soon after last bounce
        if current_frame - self.last_bounce_frame < self.min_bounce_gap:
            return False, 0.0, {"reason": "too_soon_after_last_bounce"}
        
        # Extract ultra-advanced features
        features = self._extract_optimized_features(ball_trajectory, current_frame)
        
        if features is None:
            return False, 0.0, {"reason": "insufficient_trajectory_data"}
        
        # Get raw model prediction
        raw_confidence = self._get_model_prediction(features)
        
        # Apply temporal smoothing
        smoothed_confidence = self._apply_temporal_smoothing(raw_confidence, features)
        
        # Apply confidence calibration
        calibrated_confidence = self._calibrate_confidence(smoothed_confidence, features)
        
        # Make final decision with adaptive threshold
        threshold = self._get_adaptive_threshold()
        is_bounce = calibrated_confidence >= threshold
        
        # Update bounce candidates for temporal analysis
        if calibrated_confidence > 0.4:  # Lower threshold for candidates
            self.bounce_candidates.append((current_frame, calibrated_confidence))
        
        # Clean old candidates
        self.bounce_candidates = [(f, c) for f, c in self.bounce_candidates 
                                 if current_frame - f <= 10]
        
        # Final bounce decision with temporal validation
        final_is_bounce, final_confidence = self._validate_temporal_bounce(
            is_bounce, calibrated_confidence, current_frame
        )
        
        if final_is_bounce:
            self.last_bounce_frame = current_frame
            # Clear nearby candidates
            self.bounce_candidates = [(f, c) for f, c in self.bounce_candidates 
                                     if abs(f - current_frame) > 3]
        
        details = {
            "raw_confidence": raw_confidence,
            "smoothed_confidence": smoothed_confidence,
            "calibrated_confidence": calibrated_confidence,
            "final_confidence": final_confidence,
            "adaptive_threshold": threshold,
            "bounce_candidates": len(self.bounce_candidates),
            "temporal_validation": "applied"
        }
        
        return final_is_bounce, final_confidence, details
    
    def _extract_optimized_features(self, ball_trajectory: List[Tuple[float, float, int]], 
                                   current_frame: int) -> Optional[np.ndarray]:
        """Extract optimized ultra-advanced features with better quality control"""
        try:
            # Get extended window for better context
            window_size = 25
            window_start = current_frame - window_size // 2
            window_end = current_frame + window_size // 2
            
            # Extract ball positions in window
            window_data = [(x, y, frame) for x, y, frame in ball_trajectory 
                          if window_start <= frame <= window_end and x is not None and y is not None]
            
            if len(window_data) < window_size * 0.8:  # Need 80% coverage
                return None
            
            # Sort by frame and extract coordinates
            window_data.sort(key=lambda x: x[2])
            x_coords = np.array([pos[0] for pos in window_data])
            y_coords = np.array([pos[1] for pos in window_data])
            
            # Quality check: ensure reasonable coordinates
            if np.std(x_coords) < 5 or np.std(y_coords) < 5:
                return None  # Too little movement
            
            # Extract enhanced features
            features = []
            
            # 1. Enhanced Physics Features (20 features)
            features.extend(self._extract_enhanced_physics_features(x_coords, y_coords))
            
            # 2. Advanced Trajectory Features (15 features)
            features.extend(self._extract_advanced_trajectory_features(x_coords, y_coords))
            
            # 3. Temporal Pattern Features (12 features)
            features.extend(self._extract_temporal_pattern_features(x_coords, y_coords))
            
            # 4. Multi-scale Analysis Features (8 features)
            features.extend(self._extract_multiscale_features(x_coords, y_coords))
            
            # 5. Statistical Features (3 features)
            features.extend(self._extract_statistical_features(x_coords, y_coords))
            
            # 6. Additional Physics Feature (1 feature) - to match 58 total
            features.append(self._extract_additional_physics_feature(x_coords, y_coords))
            
            # Validate features
            features_array = np.array(features)
            if np.any(np.isnan(features_array)) or np.any(np.isinf(features_array)):
                return None
            
            return features_array
            
        except Exception as e:
            logger.warning(f"Error extracting optimized features: {e}")
            return None
    
    def _extract_enhanced_physics_features(self, x_coords: np.ndarray, y_coords: np.ndarray) -> List[float]:
        """Extract enhanced physics-based features"""
        features = []
        
        if len(y_coords) < 5:
            return [0.0] * 20
        
        # Calculate velocities and accelerations
        x_vel = np.diff(x_coords)
        y_vel = np.diff(y_coords)
        
        if len(y_vel) < 3:
            return [0.0] * 20
        
        # 1. Impact and rebound analysis (5 features)
        min_y_idx = np.argmin(y_coords)
        if min_y_idx > 0 and min_y_idx < len(y_vel):
            impact_velocity = abs(y_vel[min_y_idx-1])
            rebound_velocity = y_vel[min_y_idx]
            velocity_ratio = rebound_velocity / (impact_velocity + 1e-6)
            
            # Enhanced energy analysis
            energy_before = np.mean(y_vel[:min_y_idx]**2) if min_y_idx > 0 else 0
            energy_after = np.mean(y_vel[min_y_idx:]**2) if min_y_idx < len(y_vel) else 0
            energy_efficiency = energy_after / (energy_before + 1e-6)
        else:
            impact_velocity = rebound_velocity = velocity_ratio = energy_efficiency = 0.0
        
        features.extend([impact_velocity, rebound_velocity, velocity_ratio, energy_efficiency])
        
        # 2. Bounce angle analysis (3 features)
        if len(x_vel) >= 2:
            # Trajectory angles before and after bounce
            angle_before = np.arctan2(y_vel[len(y_vel)//2-1], x_vel[len(x_vel)//2-1])
            angle_after = np.arctan2(y_vel[len(y_vel)//2], x_vel[len(x_vel)//2])
            angle_change = abs(angle_after - angle_before)
            if angle_change > np.pi:
                angle_change = 2*np.pi - angle_change
            
            # Bounce symmetry
            bounce_symmetry = 1.0 - angle_change / np.pi
            features.extend([angle_change, bounce_symmetry, np.sin(angle_change)])
        else:
            features.extend([0.0, 0.0, 0.0])
        
        # 3. Velocity reversal analysis (4 features)
        reversals = 0
        reversal_strength = 0
        max_reversal = 0
        reversal_consistency = 0
        
        for i in range(1, len(y_vel)):
            if (y_vel[i-1] < 0 and y_vel[i] > 0) or (y_vel[i-1] > 0 and y_vel[i] < 0):
                reversals += 1
                strength = abs(y_vel[i-1] - y_vel[i])
                reversal_strength += strength
                max_reversal = max(max_reversal, strength)
        
        if reversals > 0:
            reversal_consistency = reversal_strength / reversals
        
        features.extend([reversals, reversal_strength, max_reversal, reversal_consistency])
        
        # 4. Acceleration analysis (4 features)
        if len(y_vel) >= 2:
            y_accel = np.diff(y_vel)
            max_accel = np.max(np.abs(y_accel))
            min_accel = np.min(y_accel)
            accel_variance = np.var(y_accel)
            accel_consistency = 1.0 / (accel_variance + 1e-6)
        else:
            max_accel = min_accel = accel_variance = accel_consistency = 0.0
        
        features.extend([max_accel, min_accel, accel_variance, accel_consistency])
        
        # 5. Ground contact analysis (4 features)
        min_y = np.min(y_coords)
        max_y = np.max(y_coords)
        y_range = max_y - min_y
        
        # Ground contact duration
        ground_threshold = min_y + y_range * 0.15
        ground_frames = np.sum(y_coords <= ground_threshold)
        ground_ratio = ground_frames / len(y_coords)
        
        # Ground contact consistency
        ground_consistency = 1.0 / (np.std(y_coords[y_coords <= ground_threshold]) + 1e-6)
        
        # Bounce height recovery
        if len(y_coords) >= 3:
            min_idx = np.argmin(y_coords)
            max_after_min = np.max(y_coords[min_idx:])
            bounce_height = max_after_min - min_y
            height_recovery = bounce_height / (y_range + 1e-6)
        else:
            height_recovery = 0.0
        
        features.extend([ground_ratio, ground_consistency, height_recovery, y_range])
        
        return features
    
    def _extract_advanced_trajectory_features(self, x_coords: np.ndarray, y_coords: np.ndarray) -> List[float]:
        """Extract advanced trajectory analysis features"""
        features = []
        
        if len(x_coords) < 5:
            return [0.0] * 15
        
        # 1. Trajectory smoothness (3 features)
        if len(x_coords) >= 3:
            x_vel = np.diff(x_coords)
            y_vel = np.diff(y_coords)
            x_accel = np.diff(x_vel)
            y_accel = np.diff(y_vel)
            
            smoothness = 1.0 / (np.var(x_accel) + np.var(y_accel) + 1e-6)
            jerk_magnitude = np.sqrt(np.mean(x_accel**2) + np.mean(y_accel**2))
            motion_consistency = 1.0 / (np.std(x_vel) + np.std(y_vel) + 1e-6)
        else:
            smoothness = jerk_magnitude = motion_consistency = 0.0
        
        features.extend([smoothness, jerk_magnitude, motion_consistency])
        
        # 2. Curvature analysis (4 features)
        curvatures = []
        for i in range(2, len(x_coords)-1):
            curv = self._compute_curvature_at_point(x_coords, y_coords, i)
            curvatures.append(curv)
        
        if curvatures:
            max_curvature = np.max(curvatures)
            mean_curvature = np.mean(curvatures)
            curvature_variance = np.var(curvatures)
            curvature_consistency = 1.0 / (curvature_variance + 1e-6)
        else:
            max_curvature = mean_curvature = curvature_variance = curvature_consistency = 0.0
        
        features.extend([max_curvature, mean_curvature, curvature_variance, curvature_consistency])
        
        # 3. Direction analysis (4 features)
        if len(x_coords) >= 3:
            directions = np.arctan2(np.diff(y_coords), np.diff(x_coords))
            direction_changes = 0
            total_angle_change = 0
            
            for i in range(1, len(directions)):
                angle_diff = abs(directions[i] - directions[i-1])
                if angle_diff > np.pi:
                    angle_diff = 2*np.pi - angle_diff
                
                if angle_diff > np.pi/6:  # 30 degrees
                    direction_changes += 1
                
                total_angle_change += angle_diff
            
            direction_frequency = direction_changes / len(directions)
            mean_angle_change = total_angle_change / len(directions)
        else:
            direction_frequency = mean_angle_change = 0.0
            direction_changes = 0
        
        features.extend([direction_frequency, mean_angle_change, direction_changes, len(directions)])
        
        # 4. Path optimality (4 features)
        if len(x_coords) >= 3:
            # Path deviation from straight line
            x_range = np.max(x_coords) - np.min(x_coords)
            y_range = np.max(y_coords) - np.min(y_coords)
            
            if x_range > 0:
                slope = y_range / x_range
                deviations = []
                for i in range(len(x_coords)):
                    expected_y = y_coords[0] + slope * (x_coords[i] - x_coords[0])
                    deviations.append(abs(y_coords[i] - expected_y))
                path_deviation = np.mean(deviations)
            else:
                path_deviation = 0.0
            
            # Path efficiency
            total_length = np.sum(np.sqrt(np.diff(x_coords)**2 + np.diff(y_coords)**2))
            straight_distance = np.sqrt((x_coords[-1] - x_coords[0])**2 + (y_coords[-1] - y_coords[0])**2)
            path_efficiency = straight_distance / (total_length + 1e-6)
            
            # Path symmetry
            mid = len(x_coords) // 2
            first_half = y_coords[:mid]
            second_half = y_coords[mid:]
            
            if len(first_half) == len(second_half):
                path_symmetry = 1.0 - np.mean(np.abs(first_half - second_half[::-1])) / (np.std(y_coords) + 1e-6)
            else:
                path_symmetry = 0.0
            
            # Motion regularity
            if len(y_coords) >= 3:
                y_vel = np.diff(y_coords)
                y_accel = np.diff(y_vel)
                motion_regularity = 1.0 / (np.var(y_accel) + 1e-6)
            else:
                motion_regularity = 0.0
        else:
            path_deviation = path_efficiency = path_symmetry = motion_regularity = 0.0
        
        features.extend([path_deviation, path_efficiency, path_symmetry, motion_regularity])
        
        return features
    
    def _extract_temporal_pattern_features(self, x_coords: np.ndarray, y_coords: np.ndarray) -> List[float]:
        """Extract temporal pattern analysis features"""
        features = []
        
        if len(y_coords) < 5:
            return [0.0] * 12
        
        # 1. Oscillation analysis (4 features)
        y_vel = np.diff(y_coords)
        zero_crossings = 0
        oscillation_amplitude = np.max(y_coords) - np.min(y_coords)
        
        for i in range(1, len(y_vel)):
            if (y_vel[i-1] < 0 and y_vel[i] > 0) or (y_vel[i-1] > 0 and y_vel[i] < 0):
                zero_crossings += 1
        
        oscillation_frequency = zero_crossings / len(y_vel)
        
        # Local extrema analysis
        minima = 0
        maxima = 0
        for i in range(1, len(y_coords)-1):
            if y_coords[i] < y_coords[i-1] and y_coords[i] < y_coords[i+1]:
                minima += 1
            elif y_coords[i] > y_coords[i-1] and y_coords[i] > y_coords[i+1]:
                maxima += 1
        
        features.extend([oscillation_frequency, oscillation_amplitude, minima, maxima])
        
        # 2. Temporal consistency (4 features)
        temporal_consistency = 1.0 / (np.std(y_coords) + 1e-6)
        
        # Autocorrelation
        if len(y_coords) >= 10:
            correlation = np.corrcoef(y_coords[:-5], y_coords[5:])[0, 1]
            temporal_correlation = abs(correlation) if not np.isnan(correlation) else 0.0
        else:
            temporal_correlation = 0.0
        
        # Pattern regularity
        if len(y_coords) >= 8:
            # Simple pattern matching
            pattern_score = 0
            for i in range(2, len(y_coords)-2):
                if (y_coords[i] < y_coords[i-1] and y_coords[i] < y_coords[i+1] and
                    y_coords[i+1] > y_coords[i] and y_coords[i+2] < y_coords[i+1]):
                    pattern_score += 1
            pattern_regularity = pattern_score / (len(y_coords) - 4)
        else:
            pattern_regularity = 0.0
        
        # Sequence complexity
        if len(y_coords) >= 4:
            # Simplified Lempel-Ziv complexity
            y_vel = np.diff(y_coords)
            binary_seq = (y_vel > 0).astype(int)
            patterns = set()
            for i in range(len(binary_seq)):
                for j in range(i+1, len(binary_seq)+1):
                    patterns.add(tuple(binary_seq[i:j]))
            sequence_complexity = len(patterns) / len(binary_seq)
        else:
            sequence_complexity = 0.0
        
        features.extend([temporal_consistency, temporal_correlation, pattern_regularity, sequence_complexity])
        
        # 3. Frequency domain features (4 features)
        if len(y_coords) >= 8:
            # Simplified FFT analysis
            fft = np.fft.fft(y_coords)
            magnitude = np.abs(fft)
            
            # Dominant frequency
            dominant_freq = np.argmax(magnitude[1:len(magnitude)//2]) + 1
            dominant_freq_norm = dominant_freq / len(y_coords)
            
            # Spectral energy
            spectral_energy = np.sum(magnitude**2) / len(y_coords)
            
            # Spectral centroid
            if np.sum(magnitude) > 0:
                spectral_centroid = np.sum(np.arange(len(magnitude)) * magnitude) / np.sum(magnitude)
            else:
                spectral_centroid = 0.0
            
            # Spectral rolloff
            cumulative_energy = np.cumsum(magnitude**2)
            total_energy = cumulative_energy[-1]
            rolloff_idx = np.where(cumulative_energy >= 0.85 * total_energy)[0]
            spectral_rolloff = rolloff_idx[0] / len(magnitude) if len(rolloff_idx) > 0 else 0.0
        else:
            dominant_freq_norm = spectral_energy = spectral_centroid = spectral_rolloff = 0.0
        
        features.extend([dominant_freq_norm, spectral_energy, spectral_centroid, spectral_rolloff])
        
        return features
    
    def _extract_multiscale_features(self, x_coords: np.ndarray, y_coords: np.ndarray) -> List[float]:
        """Extract multi-scale analysis features"""
        features = []
        
        scales = [2, 4, 8, 16]
        
        for scale in scales:
            if len(y_coords) >= scale * 2:
                # Downsample at this scale
                downsampled = y_coords[::scale]
                
                # Compute features at this scale
                scale_std = np.std(downsampled)
                scale_mean = np.mean(downsampled)
                scale_range = np.max(downsampled) - np.min(downsampled)
                
                features.extend([scale_std, scale_mean])
            else:
                features.extend([0.0, 0.0])
        
        return features
    
    def _extract_statistical_features(self, x_coords: np.ndarray, y_coords: np.ndarray) -> List[float]:
        """Extract statistical analysis features"""
        features = []
        
        # 1. Distribution moments
        mean_y = np.mean(y_coords)
        var_y = np.var(y_coords)
        skewness = np.mean((y_coords - mean_y)**3) / (var_y**1.5 + 1e-6)
        kurtosis = np.mean((y_coords - mean_y)**4) / (var_y**2 + 1e-6)
        
        features.extend([skewness, kurtosis, var_y])
        
        return features
    
    def _extract_additional_physics_feature(self, x_coords: np.ndarray, y_coords: np.ndarray) -> float:
        """Extract additional physics feature to match 58 total features"""
        if len(y_coords) < 5:
            return 0.0
        
        # Bounce momentum conservation analysis
        y_vel = np.diff(y_coords)
        if len(y_vel) < 3:
            return 0.0
        
        # Calculate momentum before and after potential bounce
        mid_point = len(y_vel) // 2
        momentum_before = np.mean(y_vel[:mid_point])
        momentum_after = np.mean(y_vel[mid_point:])
        
        # Momentum conservation ratio (should be close to 1 for elastic bounce)
        if abs(momentum_before) > 1e-6:
            momentum_ratio = momentum_after / momentum_before
            return min(1.0, max(-1.0, momentum_ratio))  # Clamp to [-1, 1]
        
        return 0.0
    
    def _compute_curvature_at_point(self, x_coords: np.ndarray, y_coords: np.ndarray, idx: int) -> float:
        """Compute curvature at a specific point"""
        if idx < 1 or idx >= len(y_coords) - 1:
            return 0.0
        
        x1, x2, x3 = x_coords[idx-1], x_coords[idx], x_coords[idx+1]
        y1, y2, y3 = y_coords[idx-1], y_coords[idx], y_coords[idx+1]
        
        # Curvature formula for discrete points
        numerator = abs((x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1))
        denominator = ((x2 - x1)**2 + (y2 - y1)**2)**1.5
        
        if denominator > 0:
            return numerator / denominator
        return 0.0
    
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
    
    def _apply_temporal_smoothing(self, raw_confidence: float, features: np.ndarray) -> float:
        """Apply temporal smoothing to reduce noise"""
        # Add to history
        self.confidence_history.append(raw_confidence)
        self.feature_history.append(features.copy())
        
        if len(self.confidence_history) < 3:
            return raw_confidence
        
        # Weighted moving average with exponential decay
        weights = np.exp(-np.arange(len(self.confidence_history)) * 0.2)
        weights = weights / np.sum(weights)
        
        smoothed_confidence = np.average(list(self.confidence_history), weights=weights)
        
        # Boost confidence if features are consistent
        if len(self.feature_history) >= 3:
            feature_consistency = self._compute_feature_consistency()
            consistency_boost = feature_consistency * 0.1
            smoothed_confidence = min(1.0, smoothed_confidence + consistency_boost)
        
        return smoothed_confidence
    
    def _compute_feature_consistency(self) -> float:
        """Compute consistency of recent features"""
        if len(self.feature_history) < 2:
            return 0.0
        
        # Compute variance of recent features
        recent_features = np.array(list(self.feature_history))
        feature_variance = np.mean(np.var(recent_features, axis=0))
        
        # Convert variance to consistency (inverse relationship)
        consistency = 1.0 / (feature_variance + 1e-6)
        return min(1.0, consistency / 1000.0)  # Normalize
    
    def _calibrate_confidence(self, smoothed_confidence: float, features: np.ndarray) -> float:
        """Apply confidence calibration based on feature quality"""
        # Feature quality indicators
        quality_score = 1.0
        
        # Check for reasonable feature values
        if np.any(np.isnan(features)) or np.any(np.isinf(features)):
            quality_score *= 0.5
        
        # Check feature magnitude (very large values might indicate noise)
        if np.any(np.abs(features) > 1e6):
            quality_score *= 0.8
        
        # Boost confidence for high-quality physics features
        if len(features) > 10:
            physics_features = features[:10]  # First 10 are physics features
            if np.std(physics_features) > 0.1:  # Good variation in physics features
                quality_score *= 1.1
        
        calibrated_confidence = smoothed_confidence * quality_score
        return min(1.0, max(0.0, calibrated_confidence))
    
    def _get_adaptive_threshold(self) -> float:
        """Get adaptive threshold based on recent performance"""
        base_threshold = 0.6
        
        # Adjust threshold based on recent confidence history
        if len(self.confidence_history) >= 5:
            recent_avg = np.mean(list(self.confidence_history)[-5:])
            
            if recent_avg < 0.3:
                # Low confidence recently - lower threshold
                return base_threshold * 0.8
            elif recent_avg > 0.7:
                # High confidence recently - raise threshold
                return base_threshold * 1.2
        
        return base_threshold
    
    def _validate_temporal_bounce(self, is_bounce: bool, confidence: float, current_frame: int) -> Tuple[bool, float]:
        """Validate bounce decision using temporal context"""
        if not is_bounce:
            return False, confidence
        
        # Check if there are nearby bounce candidates
        nearby_candidates = [c for f, c in self.bounce_candidates if abs(f - current_frame) <= 5]
        
        if len(nearby_candidates) >= 2:
            # Multiple candidates nearby - boost confidence
            candidate_boost = np.mean(nearby_candidates) * 0.2
            final_confidence = min(1.0, confidence + candidate_boost)
            return True, final_confidence
        elif len(nearby_candidates) == 1:
            # Single candidate - moderate boost
            final_confidence = min(1.0, confidence + 0.1)
            return True, final_confidence
        else:
            # No nearby candidates - might be false positive
            return confidence > 0.8, confidence


def test_optimized_detector():
    """Test the optimized detector"""
    detector = OptimizedUltraDetector()
    
    # Sample trajectory with bounce
    sample_trajectory = [
        (100, 200, 1), (105, 190, 2), (110, 180, 3), (115, 170, 4),
        (120, 160, 5), (125, 150, 6), (130, 140, 7), (135, 130, 8),
        (140, 120, 9), (145, 110, 10), (150, 100, 11), (155, 110, 12),
        (160, 120, 13), (165, 130, 14), (170, 140, 15), (175, 150, 16)
    ]
    
    # Test detection
    for frame in range(8, 16):
        is_bounce, confidence, details = detector.detect_bounce_optimized(sample_trajectory, frame)
        if is_bounce:
            print(f"Frame {frame}: Optimized bounce detected!")
            print(f"  Raw confidence: {details['raw_confidence']:.3f}")
            print(f"  Smoothed: {details['smoothed_confidence']:.3f}")
            print(f"  Calibrated: {details['calibrated_confidence']:.3f}")
            print(f"  Final: {details['final_confidence']:.3f}")
            break


if __name__ == "__main__":
    test_optimized_detector()
