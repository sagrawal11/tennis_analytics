#!/usr/bin/env python3
"""
Improved Tennis Ball Bounce Detection - Advanced Feature Engineering

This script creates much more sophisticated features specifically designed for bounce detection.
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


class ImprovedBounceTrainer:
    """Advanced trainer with sophisticated bounce-specific features"""
    
    def __init__(self, window_size: int = 10):
        self.window_size = window_size
        self.feature_names = []
        
    def prepare_training_data(self, annotations_file: str, ball_data_file: str) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare training data with advanced features"""
        # Load annotations
        annotations_df = self._load_annotations(annotations_file)
        
        # Load ball tracking data
        ball_df = self._load_ball_data(ball_data_file)
        
        # Create training samples with advanced features
        features, labels = self._create_advanced_training_samples(annotations_df, ball_df)
        
        logger.info(f"Created {len(features)} training samples with {features.shape[1] if len(features) > 0 else 0} features")
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
            raise ValueError("Missing required columns: need either ['ball_x', 'ball_y', 'frame'] or ['x', 'y', 'frame']")
        
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        logger.info(f"Loaded {len(df)} ball tracking records")
        return df
    
    def _create_advanced_training_samples(self, annotations_df: pd.DataFrame, ball_df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Create training samples with advanced bounce-specific features"""
        features_list = []
        labels_list = []
        
        # Initialize feature names
        self._initialize_feature_names()
        
        for video_name, video_annotations in annotations_df.groupby('video_name'):
            logger.info(f"Processing video: {video_name}")
            
            video_ball_data = ball_df[ball_df.get('video_name', '') == video_name] if 'video_name' in ball_df.columns else ball_df
            video_annotations = video_annotations.sort_values('frame_number')
            
            for _, annotation in video_annotations.iterrows():
                frame_number = annotation['frame_number']
                is_bounce = annotation['is_bounce']
                
                # Extract advanced features around this frame
                features = self._extract_advanced_features(video_ball_data, frame_number)
                
                if features is not None:
                    features_list.append(features)
                    labels_list.append(is_bounce)
        
        return np.array(features_list), np.array(labels_list)
    
    def _initialize_feature_names(self):
        """Initialize the feature names list"""
        self.feature_names = [
            # Bounce physics features (8)
            'reversal_strength', 'reversal_count', 'max_reversal', 'normalized_reversal',
            'y_vel_consistency', 'accel_before_bounce', 'accel_after_bounce', 'bounce_height_ratio',
            
            # Trajectory analysis features (6)
            'max_curvature', 'mean_curvature', 'curvature_std', 'direction_change_freq',
            'trajectory_smoothness', 'line_deviation',
            
            # Temporal pattern features (4)
            'local_minima_count', 'local_maxima_count', 'oscillation_amplitude', 'oscillation_events',
            
            # Collision features (3)
            'max_velocity_change', 'mean_velocity_change', 'significant_changes',
            
            # Energy features (3)
            'avg_kinetic_energy', 'kinetic_energy_std', 'kinetic_energy_range'
        ]
    
    def _extract_advanced_features(self, ball_df: pd.DataFrame, frame_number: int) -> Optional[np.ndarray]:
        """Extract advanced bounce-specific features"""
        try:
            # Get extended window for better context
            extended_window = self.window_size * 2  # 20 frames total
            window_start = frame_number - extended_window // 2
            window_end = frame_number + extended_window // 2
            
            # Get ball positions in the window
            window_data = ball_df[
                (ball_df['frame'] >= window_start) & 
                (ball_df['frame'] <= window_end)
            ].sort_values('frame')
            
            # Quality checks
            valid_data = window_data.dropna(subset=['ball_x', 'ball_y'])
            if len(valid_data) < extended_window * 0.6:  # Need at least 60% of extended window
                return None
            
            # Extract coordinates
            x_coords = valid_data['ball_x'].values
            y_coords = valid_data['ball_y'].values
            
            # Advanced feature extraction
            features = []
            
            # 1. BOUNCE-SPECIFIC PHYSICS FEATURES
            features.extend(self._extract_bounce_physics_features(x_coords, y_coords))
            
            # 2. TRAJECTORY ANALYSIS FEATURES  
            features.extend(self._extract_trajectory_analysis_features(x_coords, y_coords))
            
            # 3. TEMPORAL PATTERN FEATURES
            features.extend(self._extract_temporal_pattern_features(x_coords, y_coords))
            
            # 4. COLLISION DETECTION FEATURES
            features.extend(self._extract_collision_features(x_coords, y_coords))
            
            # 5. ENERGY AND MOMENTUM FEATURES
            features.extend(self._extract_energy_features(x_coords, y_coords))
            
            # Validate features
            features_array = np.array(features)
            if np.any(np.isnan(features_array)) or np.any(np.isinf(features_array)):
                return None
            
            return features_array
            
        except Exception as e:
            logger.warning(f"Error extracting advanced features at frame {frame_number}: {e}")
            return None
    
    def _extract_bounce_physics_features(self, x_coords: np.ndarray, y_coords: np.ndarray) -> List[float]:
        """Extract physics-based bounce detection features"""
        features = []
        
        if len(y_coords) < 5:
            return [0.0] * 8  # Return zeros if insufficient data
        
        # 1. Y-velocity reversal strength (most important for bounces)
        y_velocities = np.diff(y_coords)
        
        # Find velocity reversal points
        reversal_strength = 0
        reversal_count = 0
        max_reversal = 0
        
        for i in range(1, len(y_velocities)):
            if (y_velocities[i-1] < 0 and y_velocities[i] > 0) or \
               (y_velocities[i-1] > 0 and y_velocities[i] < 0):
                reversal_strength += abs(y_velocities[i-1] - y_velocities[i])
                reversal_count += 1
                max_reversal = max(max_reversal, abs(y_velocities[i-1] - y_velocities[i]))
        
        features.extend([
            reversal_strength,
            reversal_count,
            max_reversal,
            reversal_strength / max(len(y_velocities), 1)  # Normalized reversal strength
        ])
        
        # 2. Y-velocity consistency (bounces have more erratic Y-velocity)
        y_vel_consistency = np.std(y_velocities) / (np.mean(np.abs(y_velocities)) + 1e-6)
        features.append(y_vel_consistency)
        
        # 3. Downward acceleration before bounce
        if len(y_velocities) >= 3:
            accel_before_bounce = np.mean(np.diff(y_velocities[:-2]))  # Acceleration in first half
            features.append(accel_before_bounce)
        else:
            features.append(0.0)
        
        # 4. Upward acceleration after bounce  
        if len(y_velocities) >= 3:
            accel_after_bounce = np.mean(np.diff(y_velocities[-2:]))  # Acceleration in last half
            features.append(accel_after_bounce)
        else:
            features.append(0.0)
        
        # 5. Bounce height ratio (how much the ball bounces back up)
        if len(y_coords) >= 3:
            min_y = np.min(y_coords)
            max_y_after_min = np.max(y_coords[np.argmin(y_coords):])
            bounce_height_ratio = (max_y_after_min - min_y) / (np.max(y_coords) - np.min(y_coords) + 1e-6)
            features.append(bounce_height_ratio)
        else:
            features.append(0.0)
        
        return features
    
    def _extract_trajectory_analysis_features(self, x_coords: np.ndarray, y_coords: np.ndarray) -> List[float]:
        """Extract trajectory analysis features"""
        features = []
        
        if len(x_coords) < 5:
            return [0.0] * 6
        
        # 1. Trajectory curvature changes (bounces create sharp direction changes)
        curvatures = []
        for i in range(2, len(x_coords)-2):
            # Calculate curvature using three points
            p1 = np.array([x_coords[i-2], y_coords[i-2]])
            p2 = np.array([x_coords[i], y_coords[i]])
            p3 = np.array([x_coords[i+2], y_coords[i+2]])
            
            # Curvature formula
            cross_product = np.cross(p2 - p1, p3 - p2)
            norm1 = np.linalg.norm(p2 - p1)
            norm2 = np.linalg.norm(p3 - p2)
            
            if norm1 > 0 and norm2 > 0:
                curvature = abs(cross_product) / (norm1 * norm2 * norm1)
                curvatures.append(curvature)
        
        features.extend([
            np.max(curvatures) if curvatures else 0.0,
            np.mean(curvatures) if curvatures else 0.0,
            np.std(curvatures) if curvatures else 0.0
        ])
        
        # 2. Direction change frequency
        directions = []
        for i in range(1, len(x_coords)):
            dx = x_coords[i] - x_coords[i-1]
            dy = y_coords[i] - y_coords[i-1]
            direction = np.arctan2(dy, dx)
            directions.append(direction)
        
        direction_changes = 0
        for i in range(1, len(directions)):
            angle_diff = abs(directions[i] - directions[i-1])
            if angle_diff > np.pi:
                angle_diff = 2*np.pi - angle_diff
            if angle_diff > np.pi/4:  # Significant direction change
                direction_changes += 1
        
        features.append(direction_changes / max(len(directions), 1))
        
        # 3. Trajectory smoothness (bounces make trajectories less smooth)
        if len(x_coords) >= 3:
            # Calculate second derivatives (acceleration)
            x_accel = np.diff(np.diff(x_coords))
            y_accel = np.diff(np.diff(y_coords))
            smoothness = 1.0 / (np.std(x_accel) + np.std(y_accel) + 1e-6)
            features.append(smoothness)
        else:
            features.append(0.0)
        
        # 4. Trajectory deviation from straight line
        if len(x_coords) >= 3:
            # Fit line and calculate deviation
            x_range = np.max(x_coords) - np.min(x_coords)
            y_range = np.max(y_coords) - np.min(y_coords)
            if x_range > 0:
                slope = y_range / x_range
                deviations = []
                for i in range(len(x_coords)):
                    expected_y = y_coords[0] + slope * (x_coords[i] - x_coords[0])
                    deviations.append(abs(y_coords[i] - expected_y))
                features.append(np.mean(deviations))
            else:
                features.append(0.0)
        else:
            features.append(0.0)
        
        return features
    
    def _extract_temporal_pattern_features(self, x_coords: np.ndarray, y_coords: np.ndarray) -> List[float]:
        """Extract temporal pattern features"""
        features = []
        
        if len(y_coords) < 5:
            return [0.0] * 4
        
        # 1. Y-position oscillation pattern
        # Find local minima and maxima
        minima = []
        maxima = []
        
        for i in range(1, len(y_coords)-1):
            if y_coords[i] < y_coords[i-1] and y_coords[i] < y_coords[i+1]:
                minima.append(y_coords[i])
            elif y_coords[i] > y_coords[i-1] and y_coords[i] > y_coords[i+1]:
                maxima.append(y_coords[i])
        
        features.extend([
            len(minima),  # Number of local minima
            len(maxima),  # Number of local maxima
            np.mean(maxima) - np.mean(minima) if minima and maxima else 0.0,  # Average oscillation amplitude
            len(minima) + len(maxima)  # Total oscillation events
        ])
        
        return features
    
    def _extract_collision_features(self, x_coords: np.ndarray, y_coords: np.ndarray) -> List[float]:
        """Extract collision detection features"""
        features = []
        
        if len(y_coords) < 3:
            return [0.0] * 3
        
        # 1. Impact detection (sudden velocity change)
        y_velocities = np.diff(y_coords)
        velocity_changes = np.abs(np.diff(y_velocities))
        
        features.extend([
            np.max(velocity_changes),  # Maximum velocity change
            np.mean(velocity_changes),  # Average velocity change
            len(velocity_changes[velocity_changes > np.std(velocity_changes) * 2])  # Number of significant velocity changes
        ])
        
        return features
    
    def _extract_energy_features(self, x_coords: np.ndarray, y_coords: np.ndarray) -> List[float]:
        """Extract energy and momentum features"""
        features = []
        
        if len(y_coords) < 3:
            return [0.0] * 3
        
        # 1. Kinetic energy changes (velocity squared)
        y_velocities = np.diff(y_coords)
        kinetic_energies = y_velocities ** 2
        
        features.extend([
            np.mean(kinetic_energies),  # Average kinetic energy
            np.std(kinetic_energies),   # Kinetic energy variance
            np.max(kinetic_energies) - np.min(kinetic_energies)  # Kinetic energy range
        ])
        
        return features
    
    def save_training_data(self, features: np.ndarray, labels: np.ndarray, output_file: str):
        """Save advanced training data"""
        try:
            
            # Create DataFrame
            df = pd.DataFrame(features, columns=self.feature_names)
            df['is_bounce'] = labels
            
            # Save to CSV
            df.to_csv(output_file, index=False)
            logger.info(f"Saved advanced training data to {output_file}")
            
            # Save feature names
            feature_names_file = output_file.replace('.csv', '_feature_names.json')
            with open(feature_names_file, 'w') as f:
                json.dump(self.feature_names, f, indent=2)
            logger.info(f"Saved feature names to {feature_names_file}")
            
        except Exception as e:
            logger.error(f"Error saving training data: {e}")
            raise


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Create advanced bounce detection training data')
    parser.add_argument('--annotations', default='all_bounce_annotations.csv', help='Annotations CSV file')
    parser.add_argument('--ball-data', default='all_ball_coordinates.csv', help='Ball tracking CSV file')
    parser.add_argument('--output', default='advanced_bounce_training_data.csv', help='Output CSV file')
    parser.add_argument('--window-size', type=int, default=10, help='Window size for analysis')
    
    args = parser.parse_args()
    
    try:
        # Initialize trainer
        trainer = ImprovedBounceTrainer(window_size=args.window_size)
        
        # Prepare training data
        features, labels = trainer.prepare_training_data(args.annotations, args.ball_data)
        
        # Save training data
        trainer.save_training_data(features, labels, args.output)
        
        logger.info("Advanced training data preparation completed successfully!")
        
    except Exception as e:
        logger.error(f"Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
