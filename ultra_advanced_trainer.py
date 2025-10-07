#!/usr/bin/env python3
"""
Ultra-Advanced Tennis Ball Bounce Detection

Implements cutting-edge features specifically designed to capture bounce physics.
"""

import pandas as pd
import numpy as np
import argparse
import logging
from typing import List, Tuple, Dict, Optional
from pathlib import Path
import json
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import joblib

logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')
logger = logging.getLogger(__name__)


class UltraAdvancedTrainer:
    """Ultra-advanced trainer with physics-based bounce detection"""
    
    def __init__(self, window_size: int = 20):
        self.window_size = window_size
        self.feature_names = []
        
    def prepare_ultra_data(self, annotations_file: str, ball_data_file: str) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare ultra-advanced training data"""
        
        # Load data
        annotations_df = self._load_annotations(annotations_file)
        ball_df = self._load_ball_data(ball_data_file)
        
        # Create ultra-advanced features
        features, labels = self._create_ultra_features(annotations_df, ball_df)
        
        logger.info(f"Created {len(features)} samples with {features.shape[1]} ultra-advanced features")
        logger.info(f"Bounce samples: {np.sum(labels)} ({np.sum(labels)/len(labels)*100:.1f}%)")
        
        return features, labels
    
    def _load_annotations(self, annotations_file: str) -> pd.DataFrame:
        """Load bounce annotations"""
        df = pd.read_csv(annotations_file)
        required_cols = ['video_name', 'frame_number', 'is_bounce']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        logger.info(f"Loaded {len(df)} annotations")
        return df
    
    def _load_ball_data(self, ball_data_file: str) -> pd.DataFrame:
        """Load ball tracking data"""
        df = pd.read_csv(ball_data_file)
        if 'ball_x' in df.columns and 'ball_y' in df.columns:
            required_cols = ['ball_x', 'ball_y', 'frame']
        elif 'x' in df.columns and 'y' in df.columns:
            df = df.rename(columns={'x': 'ball_x', 'y': 'ball_y'})
            required_cols = ['ball_x', 'ball_y', 'frame']
        else:
            raise ValueError("Missing required columns")
        
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        logger.info(f"Loaded {len(df)} ball tracking records")
        return df
    
    def _create_ultra_features(self, annotations_df: pd.DataFrame, ball_df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Create ultra-advanced features"""
        features_list = []
        labels_list = []
        
        # Initialize feature names
        self._initialize_ultra_feature_names()
        
        for video_name, video_annotations in annotations_df.groupby('video_name'):
            logger.info(f"Processing {video_name}...")
            
            video_ball_data = ball_df[ball_df.get('video_name', '') == video_name] if 'video_name' in ball_df.columns else ball_df
            video_annotations = video_annotations.sort_values('frame_number')
            
            for _, annotation in video_annotations.iterrows():
                frame_number = annotation['frame_number']
                is_bounce = annotation['is_bounce']
                
                # Extract ultra-advanced features
                features = self._extract_ultra_features(video_ball_data, frame_number)
                
                if features is not None:
                    features_list.append(features)
                    labels_list.append(is_bounce)
        
        return np.array(features_list), np.array(labels_list)
    
    def _initialize_ultra_feature_names(self):
        """Initialize ultra-advanced feature names"""
        self.feature_names = [
            # Physics-based bounce features (15)
            'impact_velocity', 'rebound_velocity', 'velocity_ratio', 'energy_loss',
            'bounce_angle', 'trajectory_curvature', 'acceleration_spike', 'deceleration_spike',
            'ground_contact_duration', 'air_time_ratio', 'vertical_velocity_reversal_strength',
            'horizontal_velocity_change', 'impact_force_estimate', 'bounce_efficiency', 'collision_elasticity',
            
            # Advanced trajectory features (12)
            'trajectory_smoothness', 'direction_change_frequency', 'curvature_variance', 'path_deviation',
            'trajectory_complexity', 'motion_consistency', 'velocity_variance', 'acceleration_variance',
            'jerk_magnitude', 'trajectory_symmetry', 'motion_regularity', 'path_optimality',
            
            # Temporal pattern features (10)
            'oscillation_frequency', 'bounce_periodicity', 'temporal_consistency', 'pattern_regularity',
            'sequence_complexity', 'time_domain_features', 'frequency_domain_features', 'spectral_energy',
            'temporal_correlation', 'pattern_matching_score',
            
            # Multi-scale analysis features (8)
            'micro_scale_features', 'meso_scale_features', 'macro_scale_features', 'scale_invariance',
            'fractal_dimension', 'multi_resolution_analysis', 'wavelet_coefficients', 'scale_interaction',
            
            # Statistical features (10)
            'statistical_moments', 'distribution_shape', 'outlier_detection', 'anomaly_score',
            'statistical_significance', 'confidence_interval', 'hypothesis_test_pvalue', 'effect_size',
            'correlation_strength', 'mutual_information'
        ]
    
    def _extract_ultra_features(self, ball_df: pd.DataFrame, frame_number: int) -> Optional[np.ndarray]:
        """Extract ultra-advanced features"""
        try:
            # Get extended window
            window_start = frame_number - self.window_size // 2
            window_end = frame_number + self.window_size // 2
            
            window_data = ball_df[
                (ball_df['frame'] >= window_start) & 
                (ball_df['frame'] <= window_end)
            ].sort_values('frame')
            
            if len(window_data) < self.window_size * 0.7:
                return None
            
            # Extract coordinates
            x_coords = window_data['ball_x'].values
            y_coords = window_data['ball_y'].values
            
            # Remove NaN values
            valid_mask = ~(np.isnan(x_coords) | np.isnan(y_coords))
            x_coords = x_coords[valid_mask]
            y_coords = y_coords[valid_mask]
            
            if len(x_coords) < 10:
                return None
            
            features = []
            
            # 1. Physics-based bounce features
            features.extend(self._compute_physics_bounce_features(x_coords, y_coords))
            
            # 2. Advanced trajectory features
            features.extend(self._compute_advanced_trajectory_features(x_coords, y_coords))
            
            # 3. Temporal pattern features
            features.extend(self._compute_temporal_pattern_features(x_coords, y_coords))
            
            # 4. Multi-scale analysis features
            features.extend(self._compute_multiscale_features(x_coords, y_coords))
            
            # 5. Statistical features
            features.extend(self._compute_statistical_features(x_coords, y_coords))
            
            # Validate features
            features_array = np.array(features)
            if np.any(np.isnan(features_array)) or np.any(np.isinf(features_array)):
                return None
            
            return features_array
            
        except Exception as e:
            logger.warning(f"Error extracting ultra features: {e}")
            return None
    
    def _compute_physics_bounce_features(self, x_coords: np.ndarray, y_coords: np.ndarray) -> List[float]:
        """Compute physics-based bounce detection features"""
        features = []
        
        if len(y_coords) < 5:
            return [0.0] * 15
        
        # Calculate velocities and accelerations
        x_vel = np.diff(x_coords)
        y_vel = np.diff(y_coords)
        
        if len(y_vel) < 3:
            return [0.0] * 15
        
        # 1. Impact and rebound velocities
        min_y_idx = np.argmin(y_coords)
        if min_y_idx > 0 and min_y_idx < len(y_vel):
            impact_velocity = abs(y_vel[min_y_idx-1]) if min_y_idx > 0 else 0
            rebound_velocity = y_vel[min_y_idx] if min_y_idx < len(y_vel) else 0
            velocity_ratio = rebound_velocity / (impact_velocity + 1e-6)
        else:
            impact_velocity = rebound_velocity = velocity_ratio = 0.0
        
        features.extend([impact_velocity, rebound_velocity, velocity_ratio])
        
        # 2. Energy analysis
        kinetic_energy_before = np.mean(y_vel[:len(y_vel)//2] ** 2)
        kinetic_energy_after = np.mean(y_vel[len(y_vel)//2:] ** 2)
        energy_loss = (kinetic_energy_before - kinetic_energy_after) / (kinetic_energy_before + 1e-6)
        features.append(energy_loss)
        
        # 3. Bounce angle (trajectory change)
        if len(x_vel) >= 2:
            angle_before = np.arctan2(y_vel[len(y_vel)//2-1], x_vel[len(x_vel)//2-1])
            angle_after = np.arctan2(y_vel[len(y_vel)//2], x_vel[len(x_vel)//2])
            bounce_angle = abs(angle_after - angle_before)
            features.append(bounce_angle)
        else:
            features.append(0.0)
        
        # 4. Trajectory curvature at bounce point
        if len(y_coords) >= 3:
            curvature = self._compute_curvature_at_point(y_coords, len(y_coords)//2)
            features.append(curvature)
        else:
            features.append(0.0)
        
        # 5. Acceleration spikes
        if len(y_vel) >= 2:
            y_accel = np.diff(y_vel)
            acceleration_spike = np.max(np.abs(y_accel))
            deceleration_spike = np.min(y_accel)
            features.extend([acceleration_spike, deceleration_spike])
        else:
            features.extend([0.0, 0.0])
        
        # 6. Ground contact duration (frames with minimal Y movement)
        y_std = np.std(y_coords)
        ground_frames = np.sum(np.abs(y_coords - np.mean(y_coords)) < y_std * 0.1)
        ground_contact_duration = ground_frames / len(y_coords)
        features.append(ground_contact_duration)
        
        # 7. Air time ratio
        air_time = len(y_coords) - ground_frames
        air_time_ratio = air_time / len(y_coords)
        features.append(air_time_ratio)
        
        # 8. Vertical velocity reversal strength
        reversal_strength = self._compute_reversal_strength(y_vel)
        features.append(reversal_strength)
        
        # 9. Horizontal velocity change
        if len(x_vel) >= 2:
            x_vel_change = np.std(x_vel)
            features.append(x_vel_change)
        else:
            features.append(0.0)
        
        # 10. Impact force estimate (acceleration * mass estimate)
        if len(y_vel) >= 2:
            max_accel = np.max(np.abs(np.diff(y_vel)))
            impact_force_estimate = max_accel  # Simplified
            features.append(impact_force_estimate)
        else:
            features.append(0.0)
        
        # 11. Bounce efficiency (energy retention)
        bounce_efficiency = velocity_ratio ** 2  # Simplified
        features.append(bounce_efficiency)
        
        # 12. Collision elasticity
        collision_elasticity = min(1.0, velocity_ratio) if velocity_ratio > 0 else 0.0
        features.append(collision_elasticity)
        
        return features
    
    def _compute_advanced_trajectory_features(self, x_coords: np.ndarray, y_coords: np.ndarray) -> List[float]:
        """Compute advanced trajectory analysis features"""
        features = []
        
        if len(x_coords) < 5:
            return [0.0] * 12
        
        # 1. Trajectory smoothness (inverse of acceleration variance)
        if len(x_coords) >= 3:
            x_vel = np.diff(x_coords)
            y_vel = np.diff(y_coords)
            x_accel = np.diff(x_vel)
            y_accel = np.diff(y_vel)
            smoothness = 1.0 / (np.var(x_accel) + np.var(y_accel) + 1e-6)
            features.append(smoothness)
        else:
            features.append(0.0)
        
        # 2. Direction change frequency
        if len(x_coords) >= 3:
            directions = np.arctan2(np.diff(y_coords), np.diff(x_coords))
            direction_changes = 0
            for i in range(1, len(directions)):
                angle_diff = abs(directions[i] - directions[i-1])
                if angle_diff > np.pi:
                    angle_diff = 2*np.pi - angle_diff
                if angle_diff > np.pi/6:  # 30 degrees
                    direction_changes += 1
            direction_freq = direction_changes / len(directions)
            features.append(direction_freq)
        else:
            features.append(0.0)
        
        # 3. Curvature variance
        curvatures = []
        for i in range(2, len(x_coords)-1):
            curv = self._compute_curvature_at_point(y_coords, i)
            curvatures.append(curv)
        features.append(np.var(curvatures) if curvatures else 0.0)
        
        # 4. Path deviation from straight line
        if len(x_coords) >= 3:
            # Fit line and compute deviations
            x_range = np.max(x_coords) - np.min(x_coords)
            y_range = np.max(y_coords) - np.min(y_coords)
            if x_range > 0:
                slope = y_range / x_range
                deviations = []
                for i in range(len(x_coords)):
                    expected_y = y_coords[0] + slope * (x_coords[i] - x_coords[0])
                    deviations.append(abs(y_coords[i] - expected_y))
                path_deviation = np.mean(deviations)
                features.append(path_deviation)
            else:
                features.append(0.0)
        else:
            features.append(0.0)
        
        # 5. Trajectory complexity (fractal dimension approximation)
        complexity = self._compute_trajectory_complexity(x_coords, y_coords)
        features.append(complexity)
        
        # 6. Motion consistency
        if len(x_coords) >= 3:
            x_vel = np.diff(x_coords)
            y_vel = np.diff(y_coords)
            motion_consistency = 1.0 / (np.std(x_vel) + np.std(y_vel) + 1e-6)
            features.append(motion_consistency)
        else:
            features.append(0.0)
        
        # 7. Velocity variance
        if len(x_coords) >= 2:
            x_vel = np.diff(x_coords)
            y_vel = np.diff(y_coords)
            velocity_variance = np.var(x_vel) + np.var(y_vel)
            features.append(velocity_variance)
        else:
            features.append(0.0)
        
        # 8. Acceleration variance
        if len(x_coords) >= 3:
            x_vel = np.diff(x_coords)
            y_vel = np.diff(y_coords)
            x_accel = np.diff(x_vel)
            y_accel = np.diff(y_vel)
            acceleration_variance = np.var(x_accel) + np.var(y_accel)
            features.append(acceleration_variance)
        else:
            features.append(0.0)
        
        # 9. Jerk magnitude (third derivative)
        if len(x_coords) >= 4:
            x_vel = np.diff(x_coords)
            y_vel = np.diff(y_coords)
            x_accel = np.diff(x_vel)
            y_accel = np.diff(y_vel)
            x_jerk = np.diff(x_accel)
            y_jerk = np.diff(y_accel)
            jerk_magnitude = np.sqrt(np.mean(x_jerk**2) + np.mean(y_jerk**2))
            features.append(jerk_magnitude)
        else:
            features.append(0.0)
        
        # 10. Trajectory symmetry
        symmetry = self._compute_trajectory_symmetry(x_coords, y_coords)
        features.append(symmetry)
        
        # 11. Motion regularity
        regularity = self._compute_motion_regularity(x_coords, y_coords)
        features.append(regularity)
        
        # 12. Path optimality (how close to minimum energy path)
        optimality = self._compute_path_optimality(x_coords, y_coords)
        features.append(optimality)
        
        return features
    
    def _compute_temporal_pattern_features(self, x_coords: np.ndarray, y_coords: np.ndarray) -> List[float]:
        """Compute temporal pattern analysis features"""
        features = []
        
        if len(y_coords) < 5:
            return [0.0] * 10
        
        # 1. Oscillation frequency
        y_vel = np.diff(y_coords)
        zero_crossings = 0
        for i in range(1, len(y_vel)):
            if (y_vel[i-1] < 0 and y_vel[i] > 0) or (y_vel[i-1] > 0 and y_vel[i] < 0):
                zero_crossings += 1
        oscillation_frequency = zero_crossings / len(y_vel)
        features.append(oscillation_frequency)
        
        # 2. Bounce periodicity
        periodicity = self._compute_bounce_periodicity(y_coords)
        features.append(periodicity)
        
        # 3. Temporal consistency
        temporal_consistency = 1.0 / (np.std(y_coords) + 1e-6)
        features.append(temporal_consistency)
        
        # 4. Pattern regularity
        pattern_regularity = self._compute_pattern_regularity(y_coords)
        features.append(pattern_regularity)
        
        # 5. Sequence complexity
        sequence_complexity = self._compute_sequence_complexity(y_coords)
        features.append(sequence_complexity)
        
        # 6. Time domain features
        time_features = self._compute_time_domain_features(y_coords)
        features.append(time_features)
        
        # 7. Frequency domain features
        freq_features = self._compute_frequency_domain_features(y_coords)
        features.append(freq_features)
        
        # 8. Spectral energy
        spectral_energy = self._compute_spectral_energy(y_coords)
        features.append(spectral_energy)
        
        # 9. Temporal correlation
        temporal_correlation = self._compute_temporal_correlation(y_coords)
        features.append(temporal_correlation)
        
        # 10. Pattern matching score
        pattern_score = self._compute_pattern_matching_score(y_coords)
        features.append(pattern_score)
        
        return features
    
    def _compute_multiscale_features(self, x_coords: np.ndarray, y_coords: np.ndarray) -> List[float]:
        """Compute multi-scale analysis features"""
        features = []
        
        # Simplified multi-scale features
        scales = [2, 4, 8]
        
        for scale in scales:
            if len(y_coords) >= scale * 2:
                # Downsample
                downsampled = y_coords[::scale]
                # Compute features at this scale
                micro_feature = np.std(downsampled)
                features.append(micro_feature)
            else:
                features.append(0.0)
        
        # Add more scale features
        features.extend([0.0] * 5)  # Placeholder for more complex multi-scale features
        
        return features
    
    def _compute_statistical_features(self, x_coords: np.ndarray, y_coords: np.ndarray) -> List[float]:
        """Compute statistical analysis features"""
        features = []
        
        # 1. Statistical moments
        moments = [np.mean(y_coords), np.var(y_coords), 
                  np.mean((y_coords - np.mean(y_coords))**3),  # Skewness
                  np.mean((y_coords - np.mean(y_coords))**4)]  # Kurtosis
        features.extend(moments)
        
        # 2. Distribution shape
        distribution_shape = moments[2] / (moments[1]**1.5 + 1e-6)  # Normalized skewness
        features.append(distribution_shape)
        
        # 3. Outlier detection
        q75, q25 = np.percentile(y_coords, [75, 25])
        iqr = q75 - q25
        outliers = np.sum((y_coords < q25 - 1.5*iqr) | (y_coords > q75 + 1.5*iqr))
        outlier_ratio = outliers / len(y_coords)
        features.append(outlier_ratio)
        
        # 4. Anomaly score
        anomaly_score = self._compute_anomaly_score(y_coords)
        features.append(anomaly_score)
        
        # 5. Statistical significance (simplified)
        stat_significance = 1.0 / (np.std(y_coords) + 1e-6)
        features.append(stat_significance)
        
        # 6. Confidence interval
        confidence_interval = 1.96 * np.std(y_coords) / np.sqrt(len(y_coords))
        features.append(confidence_interval)
        
        # 7. Hypothesis test p-value (simplified)
        p_value = 0.05 if np.std(y_coords) > np.mean(np.abs(y_coords)) else 0.95
        features.append(p_value)
        
        # 8. Effect size
        effect_size = np.std(y_coords) / (np.mean(np.abs(y_coords)) + 1e-6)
        features.append(effect_size)
        
        # 9. Correlation strength
        if len(x_coords) == len(y_coords):
            correlation = np.corrcoef(x_coords, y_coords)[0, 1]
            features.append(abs(correlation) if not np.isnan(correlation) else 0.0)
        else:
            features.append(0.0)
        
        # 10. Mutual information (simplified)
        mutual_info = 0.5  # Placeholder
        features.append(mutual_info)
        
        return features
    
    # Helper methods for complex computations
    def _compute_curvature_at_point(self, y_coords: np.ndarray, idx: int) -> float:
        """Compute curvature at a specific point"""
        if idx < 1 or idx >= len(y_coords) - 1:
            return 0.0
        
        y1, y2, y3 = y_coords[idx-1], y_coords[idx], y_coords[idx+1]
        curvature = abs(y1 - 2*y2 + y3) / (1 + (y3 - y1)**2)**1.5
        return curvature
    
    def _compute_reversal_strength(self, y_vel: np.ndarray) -> float:
        """Compute strength of velocity reversals"""
        reversals = 0
        total_strength = 0
        
        for i in range(1, len(y_vel)):
            if (y_vel[i-1] < 0 and y_vel[i] > 0) or (y_vel[i-1] > 0 and y_vel[i] < 0):
                reversals += 1
                total_strength += abs(y_vel[i-1] - y_vel[i])
        
        return total_strength / max(reversals, 1)
    
    def _compute_trajectory_complexity(self, x_coords: np.ndarray, y_coords: np.ndarray) -> float:
        """Compute trajectory complexity measure"""
        if len(x_coords) < 3:
            return 0.0
        
        total_length = np.sum(np.sqrt(np.diff(x_coords)**2 + np.diff(y_coords)**2))
        straight_distance = np.sqrt((x_coords[-1] - x_coords[0])**2 + (y_coords[-1] - y_coords[0])**2)
        
        if straight_distance > 0:
            return total_length / straight_distance
        return 0.0
    
    def _compute_trajectory_symmetry(self, x_coords: np.ndarray, y_coords: np.ndarray) -> float:
        """Compute trajectory symmetry measure"""
        if len(y_coords) < 2:
            return 0.0
        
        mid = len(y_coords) // 2
        first_half = y_coords[:mid]
        second_half = y_coords[mid:]
        
        if len(first_half) != len(second_half):
            return 0.0
        
        # Reverse second half and compare
        second_half_reversed = second_half[::-1]
        symmetry = 1.0 - np.mean(np.abs(first_half - second_half_reversed)) / (np.std(y_coords) + 1e-6)
        return max(0.0, symmetry)
    
    def _compute_motion_regularity(self, x_coords: np.ndarray, y_coords: np.ndarray) -> float:
        """Compute motion regularity"""
        if len(y_coords) < 3:
            return 0.0
        
        y_vel = np.diff(y_coords)
        y_accel = np.diff(y_vel)
        
        # Regular motion has consistent acceleration patterns
        regularity = 1.0 / (np.std(y_accel) + 1e-6)
        return min(1.0, regularity / 100.0)
    
    def _compute_path_optimality(self, x_coords: np.ndarray, y_coords: np.ndarray) -> float:
        """Compute path optimality (energy efficiency)"""
        if len(y_coords) < 2:
            return 0.0
        
        # Optimal path minimizes energy (smooth trajectory)
        y_vel = np.diff(y_coords)
        y_accel = np.diff(y_vel)
        
        energy = np.sum(y_accel**2)
        optimality = 1.0 / (energy + 1e-6)
        return min(1.0, optimality)
    
    def _compute_bounce_periodicity(self, y_coords: np.ndarray) -> float:
        """Compute bounce periodicity"""
        # Find local minima (potential bounce points)
        minima = []
        for i in range(1, len(y_coords)-1):
            if y_coords[i] < y_coords[i-1] and y_coords[i] < y_coords[i+1]:
                minima.append(i)
        
        if len(minima) < 2:
            return 0.0
        
        # Compute periodicity
        periods = np.diff(minima)
        if len(periods) > 0:
            periodicity = 1.0 / (np.std(periods) + 1e-6)
            return min(1.0, periodicity / 10.0)
        return 0.0
    
    def _compute_pattern_regularity(self, y_coords: np.ndarray) -> float:
        """Compute pattern regularity"""
        # Autocorrelation-based regularity
        if len(y_coords) < 10:
            return 0.0
        
        # Simplified autocorrelation
        correlation = np.corrcoef(y_coords[:-5], y_coords[5:])[0, 1]
        return abs(correlation) if not np.isnan(correlation) else 0.0
    
    def _compute_sequence_complexity(self, y_coords: np.ndarray) -> float:
        """Compute sequence complexity"""
        if len(y_coords) < 3:
            return 0.0
        
        # Lempel-Ziv complexity approximation
        y_vel = np.diff(y_coords)
        # Convert to binary sequence
        binary_seq = (y_vel > 0).astype(int)
        
        # Count unique patterns
        patterns = set()
        for i in range(len(binary_seq)):
            for j in range(i+1, len(binary_seq)+1):
                patterns.add(tuple(binary_seq[i:j]))
        
        complexity = len(patterns) / len(binary_seq)
        return min(1.0, complexity)
    
    def _compute_time_domain_features(self, y_coords: np.ndarray) -> float:
        """Compute time domain features"""
        if len(y_coords) < 3:
            return 0.0
        
        # Zero crossing rate
        y_vel = np.diff(y_coords)
        zero_crossings = np.sum(np.diff(np.sign(y_vel)) != 0)
        zcr = zero_crossings / len(y_vel)
        return zcr
    
    def _compute_frequency_domain_features(self, y_coords: np.ndarray) -> float:
        """Compute frequency domain features"""
        if len(y_coords) < 8:
            return 0.0
        
        # Simplified FFT-based features
        fft = np.fft.fft(y_coords)
        magnitude = np.abs(fft)
        
        # Dominant frequency
        dominant_freq = np.argmax(magnitude[1:len(magnitude)//2]) + 1
        return dominant_freq / len(y_coords)
    
    def _compute_spectral_energy(self, y_coords: np.ndarray) -> float:
        """Compute spectral energy"""
        if len(y_coords) < 8:
            return 0.0
        
        fft = np.fft.fft(y_coords)
        energy = np.sum(np.abs(fft)**2)
        return energy / len(y_coords)
    
    def _compute_temporal_correlation(self, y_coords: np.ndarray) -> float:
        """Compute temporal correlation"""
        if len(y_coords) < 10:
            return 0.0
        
        # Lag-1 autocorrelation
        correlation = np.corrcoef(y_coords[:-1], y_coords[1:])[0, 1]
        return abs(correlation) if not np.isnan(correlation) else 0.0
    
    def _compute_pattern_matching_score(self, y_coords: np.ndarray) -> float:
        """Compute pattern matching score"""
        # Simplified pattern matching
        if len(y_coords) < 6:
            return 0.0
        
        # Look for bounce-like pattern (down-up-down)
        pattern_score = 0.0
        for i in range(2, len(y_coords)-2):
            if (y_coords[i] < y_coords[i-1] and y_coords[i] < y_coords[i+1] and
                y_coords[i+1] > y_coords[i] and y_coords[i+2] < y_coords[i+1]):
                pattern_score += 1
        
        return pattern_score / (len(y_coords) - 4)
    
    def _compute_anomaly_score(self, y_coords: np.ndarray) -> float:
        """Compute anomaly score"""
        if len(y_coords) < 3:
            return 0.0
        
        # Z-score based anomaly detection
        mean_y = np.mean(y_coords)
        std_y = np.std(y_coords)
        
        if std_y == 0:
            return 0.0
        
        z_scores = np.abs((y_coords - mean_y) / std_y)
        anomaly_score = np.max(z_scores)
        return min(1.0, anomaly_score / 3.0)  # Normalize to [0,1]
    
    def train_ultra_models(self, features: np.ndarray, labels: np.ndarray, test_size: float = 0.2) -> Dict:
        """Train ultra-advanced models"""
        logger.info("Training ultra-advanced models...")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            features, labels, test_size=test_size, random_state=42, stratify=labels
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Define advanced models
        models_to_train = {
            'ultra_random_forest': RandomForestClassifier(
                n_estimators=500,
                max_depth=20,
                min_samples_split=2,
                min_samples_leaf=1,
                class_weight='balanced',
                random_state=42,
                n_jobs=-1
            ),
            'gradient_boosting': GradientBoostingClassifier(
                n_estimators=200,
                learning_rate=0.1,
                max_depth=10,
                random_state=42
            ),
            'ultra_svm': SVC(
                class_weight='balanced',
                probability=True,
                random_state=42,
                kernel='rbf',
                C=10,
                gamma='scale'
            )
        }
        
        results = {}
        
        for name, model in models_to_train.items():
            logger.info(f"Training {name}...")
            
            try:
                if name == 'ultra_svm':
                    model.fit(X_train_scaled, y_train)
                    y_pred = model.predict(X_test_scaled)
                    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
                else:
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                    y_pred_proba = model.predict_proba(X_test)[:, 1]
                
                auc_score = roc_auc_score(y_test, y_pred_proba)
                
                results[name] = {
                    'auc_score': auc_score,
                    'predictions': y_pred,
                    'probabilities': y_pred_proba,
                    'test_labels': y_test,
                    'model': model
                }
                
                logger.info(f"{name} - AUC: {auc_score:.4f}")
                
            except Exception as e:
                logger.error(f"Error training {name}: {e}")
                continue
        
        # Save scaler
        results['scaler'] = scaler
        
        return results


def main():
    parser = argparse.ArgumentParser(description='Train ultra-advanced bounce detection models')
    parser.add_argument('--annotations', default='all_bounce_annotations.csv', help='Annotations CSV file')
    parser.add_argument('--ball-data', default='all_ball_coordinates.csv', help='Ball tracking CSV file')
    parser.add_argument('--output-dir', default='ultra_models', help='Output directory')
    parser.add_argument('--window-size', type=int, default=20, help='Window size')
    
    args = parser.parse_args()
    
    try:
        # Initialize trainer
        trainer = UltraAdvancedTrainer(window_size=args.window_size)
        
        # Prepare ultra-advanced data
        features, labels = trainer.prepare_ultra_data(args.annotations, args.ball_data)
        
        if len(features) == 0:
            logger.error("No valid features created!")
            return 1
        
        # Train models
        results = trainer.train_ultra_models(features, labels)
        
        # Save models
        output_path = Path(args.output_dir)
        output_path.mkdir(exist_ok=True)
        
        for name, result in results.items():
            if name == 'scaler':
                continue
            
            model_file = output_path / f"{name}_model.joblib"
            joblib.dump(result['model'], model_file)
            logger.info(f"Saved {name} to {model_file}")
        
        # Save scaler
        scaler_file = output_path / "ultra_scaler.joblib"
        joblib.dump(results['scaler'], scaler_file)
        
        # Print results
        print(f"\n🎉 Ultra-Advanced Training completed!")
        
        best_model = None
        best_auc = 0
        
        for name, result in results.items():
            if name == 'scaler':
                continue
            
            auc = result['auc_score']
            print(f"{name.upper()}: AUC = {auc:.4f}")
            
            if auc > best_auc:
                best_auc = auc
                best_model = name
        
        if best_model:
            print(f"\n🏆 BEST MODEL: {best_model.upper()} (AUC: {best_auc:.4f})")
            
            # Classification report
            report = classification_report(results[best_model]['test_labels'], 
                                        results[best_model]['predictions'], 
                                        target_names=['No Bounce', 'Bounce'])
            print(f"\nClassification Report:\n{report}")
        
    except Exception as e:
        logger.error(f"Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
