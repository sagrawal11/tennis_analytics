#!/usr/bin/env python3
"""
Sequence-based Tennis Ball Bounce Detection

Alternative to LSTM using sequence features with traditional ML models.
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
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import joblib

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')
logger = logging.getLogger(__name__)


class SequenceBounceTrainer:
    """Sequence-based bounce detection trainer"""
    
    def __init__(self, sequence_length: int = 15):
        """
        Initialize sequence trainer
        
        Args:
            sequence_length: Number of frames in each sequence
        """
        self.sequence_length = sequence_length
        self.scaler = StandardScaler()
        self.models = {}
        
    def prepare_sequence_features(self, annotations_file: str, ball_data_file: str) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare sequence-based features for training"""
        # Load data
        annotations_df = self._load_annotations(annotations_file)
        ball_df = self._load_ball_data(ball_data_file)
        
        # Create sequence features
        features, labels = self._create_sequence_features(annotations_df, ball_df)
        
        logger.info(f"Created {len(features)} sequence samples with {features.shape[1]} features")
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
    
    def _create_sequence_features(self, annotations_df: pd.DataFrame, ball_df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Create sequence-based features"""
        features_list = []
        labels_list = []
        
        for video_name, video_annotations in annotations_df.groupby('video_name'):
            logger.info(f"Processing video: {video_name}")
            
            # Get ball data for this video
            video_ball_data = ball_df[ball_df.get('video_name', '') == video_name] if 'video_name' in ball_df.columns else ball_df
            video_ball_data = video_ball_data.sort_values('frame')
            
            # Create sequences centered around annotated frames
            for _, annotation in video_annotations.iterrows():
                center_frame = annotation['frame_number']
                is_bounce = annotation['is_bounce']
                
                # Extract sequence features around this frame
                sequence_features = self._extract_sequence_features(video_ball_data, center_frame)
                
                if sequence_features is not None:
                    features_list.append(sequence_features)
                    labels_list.append(is_bounce)
        
        return np.array(features_list), np.array(labels_list)
    
    def _extract_sequence_features(self, ball_df: pd.DataFrame, center_frame: int) -> Optional[np.ndarray]:
        """Extract sequence-based features around a center frame"""
        try:
            # Get window around center frame
            half_length = self.sequence_length // 2
            start_frame = center_frame - half_length
            end_frame = center_frame + half_length
            
            # Get ball positions in window
            window_data = ball_df[
                (ball_df['frame'] >= start_frame) & 
                (ball_df['frame'] <= end_frame)
            ].sort_values('frame')
            
            if len(window_data) < self.sequence_length * 0.6:
                return None
            
            # Fill missing frames
            sequence_data = self._fill_sequence(window_data, start_frame, end_frame)
            
            if sequence_data is None:
                return None
            
            # Extract features from the sequence
            features = self._compute_sequence_features(sequence_data)
            
            return features
            
        except Exception as e:
            logger.warning(f"Error extracting sequence features at frame {center_frame}: {e}")
            return None
    
    def _fill_sequence(self, window_data: pd.DataFrame, start_frame: int, end_frame: int) -> Optional[np.ndarray]:
        """Fill missing frames in sequence with interpolation"""
        try:
            # Create full frame range
            frame_range = list(range(start_frame, end_frame + 1))
            sequence = np.full((len(frame_range), 2), np.nan)  # x, y coordinates
            
            # Fill in available data
            for _, row in window_data.iterrows():
                frame_idx = row['frame'] - start_frame
                if 0 <= frame_idx < len(sequence):
                    sequence[frame_idx, 0] = row['ball_x']
                    sequence[frame_idx, 1] = row['ball_y']
            
            # Interpolate missing values
            for i in range(2):  # x and y coordinates
                valid_mask = ~np.isnan(sequence[:, i])
                if np.sum(valid_mask) >= 3:
                    valid_indices = np.where(valid_mask)[0]
                    valid_values = sequence[valid_indices, i]
                    sequence[:, i] = np.interp(frame_range, valid_indices + start_frame, valid_values)
                else:
                    return None
            
            return sequence
            
        except Exception as e:
            logger.warning(f"Error filling sequence: {e}")
            return None
    
    def _compute_sequence_features(self, sequence: np.ndarray) -> np.ndarray:
        """Compute features from a sequence of ball positions"""
        features = []
        
        x_coords = sequence[:, 0]
        y_coords = sequence[:, 1]
        
        # 1. POSITION FEATURES
        features.extend([
            np.mean(x_coords),
            np.mean(y_coords),
            np.std(x_coords),
            np.std(y_coords),
            np.max(x_coords) - np.min(x_coords),  # x_range
            np.max(y_coords) - np.min(y_coords),  # y_range
        ])
        
        # 2. VELOCITY FEATURES
        if len(x_coords) >= 2:
            x_velocities = np.diff(x_coords)
            y_velocities = np.diff(y_coords)
            
            features.extend([
                np.mean(x_velocities),
                np.mean(y_velocities),
                np.std(x_velocities),
                np.std(y_velocities),
                np.max(np.abs(x_velocities)),
                np.max(np.abs(y_velocities)),
                np.min(y_velocities),  # Most negative Y velocity (downward)
                np.max(y_velocities),  # Most positive Y velocity (upward)
            ])
        else:
            features.extend([0.0] * 8)
        
        # 3. ACCELERATION FEATURES
        if len(x_coords) >= 3:
            x_velocities = np.diff(x_coords)
            y_velocities = np.diff(y_coords)
            x_accelerations = np.diff(x_velocities)
            y_accelerations = np.diff(y_velocities)
            
            features.extend([
                np.mean(x_accelerations),
                np.mean(y_accelerations),
                np.std(x_accelerations),
                np.std(y_accelerations),
                np.max(np.abs(x_accelerations)),
                np.max(np.abs(y_accelerations)),
            ])
        else:
            features.extend([0.0] * 6)
        
        # 4. BOUNCE-SPECIFIC FEATURES
        bounce_features = self._compute_bounce_sequence_features(x_coords, y_coords)
        features.extend(bounce_features)
        
        # 5. TEMPORAL PATTERN FEATURES
        temporal_features = self._compute_temporal_features(x_coords, y_coords)
        features.extend(temporal_features)
        
        return np.array(features)
    
    def _compute_bounce_sequence_features(self, x_coords: np.ndarray, y_coords: np.ndarray) -> List[float]:
        """Compute bounce-specific features from sequence"""
        features = []
        
        if len(y_coords) < 3:
            return [0.0] * 8
        
        # Y-velocity analysis
        y_velocities = np.diff(y_coords)
        
        # 1. Velocity reversal analysis
        reversals = 0
        reversal_strength = 0
        max_reversal = 0
        
        for i in range(1, len(y_velocities)):
            if (y_velocities[i-1] < 0 and y_velocities[i] > 0) or \
               (y_velocities[i-1] > 0 and y_velocities[i] < 0):
                reversals += 1
                reversal_magnitude = abs(y_velocities[i-1] - y_velocities[i])
                reversal_strength += reversal_magnitude
                max_reversal = max(max_reversal, reversal_magnitude)
        
        features.extend([
            reversals,
            reversal_strength,
            max_reversal,
            reversal_strength / max(len(y_velocities), 1),
        ])
        
        # 2. Y-position analysis (finding bounce points)
        min_y_idx = np.argmin(y_coords)
        min_y_value = y_coords[min_y_idx]
        
        # Height recovery after minimum
        if min_y_idx < len(y_coords) - 1:
            max_y_after_min = np.max(y_coords[min_y_idx:])
            bounce_height = max_y_after_min - min_y_value
            features.append(bounce_height)
        else:
            features.append(0.0)
        
        # 3. Velocity consistency
        y_vel_consistency = np.std(y_velocities) / (np.mean(np.abs(y_velocities)) + 1e-6)
        features.append(y_vel_consistency)
        
        # 4. Downward vs upward velocity ratio
        downward_vel = np.sum(y_velocities[y_velocities < 0])
        upward_vel = np.sum(y_velocities[y_velocities > 0])
        vel_ratio = abs(downward_vel) / (abs(upward_vel) + 1e-6)
        features.append(vel_ratio)
        
        # 5. Energy analysis
        kinetic_energies = y_velocities ** 2
        energy_variance = np.var(kinetic_energies)
        features.append(energy_variance)
        
        return features
    
    def _compute_temporal_features(self, x_coords: np.ndarray, y_coords: np.ndarray) -> List[float]:
        """Compute temporal pattern features"""
        features = []
        
        if len(y_coords) < 3:
            return [0.0] * 6
        
        # 1. Local extrema analysis
        minima = 0
        maxima = 0
        
        for i in range(1, len(y_coords)-1):
            if y_coords[i] < y_coords[i-1] and y_coords[i] < y_coords[i+1]:
                minima += 1
            elif y_coords[i] > y_coords[i-1] and y_coords[i] > y_coords[i+1]:
                maxima += 1
        
        features.extend([minima, maxima])
        
        # 2. Oscillation analysis
        oscillation_amplitude = np.max(y_coords) - np.min(y_coords)
        features.append(oscillation_amplitude)
        
        # 3. Direction change frequency
        y_velocities = np.diff(y_coords)
        direction_changes = 0
        for i in range(1, len(y_velocities)):
            if (y_velocities[i-1] < 0 and y_velocities[i] > 0) or \
               (y_velocities[i-1] > 0 and y_velocities[i] < 0):
                direction_changes += 1
        
        features.append(direction_changes)
        
        # 4. Smoothness (inverse of acceleration variance)
        if len(y_coords) >= 3:
            y_accelerations = np.diff(y_velocities)
            smoothness = 1.0 / (np.var(y_accelerations) + 1e-6)
            features.append(smoothness)
        else:
            features.append(0.0)
        
        # 5. Trajectory deviation from straight line
        if len(x_coords) >= 3:
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
    
    def train_models(self, features: np.ndarray, labels: np.ndarray, test_size: float = 0.2) -> Dict:
        """Train multiple models and compare performance"""
        logger.info("Starting sequence-based model training...")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            features, labels, test_size=test_size, random_state=42, stratify=labels
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        logger.info(f"Training sequences: {len(X_train)} ({np.sum(y_train)} bounces)")
        logger.info(f"Test sequences: {len(X_test)} ({np.sum(y_test)} bounces)")
        
        # Define models
        models_to_train = {
            'random_forest': RandomForestClassifier(
                n_estimators=200,
                max_depth=15,
                min_samples_split=3,
                min_samples_leaf=1,
                class_weight='balanced',
                random_state=42
            ),
            'logistic_regression': LogisticRegression(
                class_weight='balanced',
                random_state=42,
                max_iter=1000
            ),
            'svm': SVC(
                class_weight='balanced',
                probability=True,
                random_state=42
            )
        }
        
        results = {}
        
        for name, model in models_to_train.items():
            logger.info(f"Training {name}...")
            
            try:
                # Use scaled features for logistic regression and SVM
                if name in ['logistic_regression', 'svm']:
                    model.fit(X_train_scaled, y_train)
                    y_pred = model.predict(X_test_scaled)
                    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
                else:
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                    y_pred_proba = model.predict_proba(X_test)[:, 1]
                
                # Calculate metrics
                auc_score = roc_auc_score(y_test, y_pred_proba)
                
                # Store model
                self.models[name] = model
                
                results[name] = {
                    'auc_score': auc_score,
                    'predictions': y_pred,
                    'probabilities': y_pred_proba,
                    'test_labels': y_test
                }
                
                logger.info(f"{name} - AUC: {auc_score:.4f}")
                
            except Exception as e:
                logger.error(f"Error training {name}: {e}")
                continue
        
        return results
    
    def save_models(self, output_dir: str):
        """Save trained models and scaler"""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        for name, model in self.models.items():
            model_file = output_path / f"sequence_{name}_model.joblib"
            joblib.dump(model, model_file)
            logger.info(f"Saved {name} model to {model_file}")
        
        # Save scaler
        scaler_file = output_path / "sequence_scaler.joblib"
        joblib.dump(self.scaler, scaler_file)
        logger.info(f"Saved scaler to {scaler_file}")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Train sequence-based bounce detection models')
    parser.add_argument('--annotations', default='all_bounce_annotations.csv', help='Annotations CSV file')
    parser.add_argument('--ball-data', default='all_ball_coordinates.csv', help='Ball tracking CSV file')
    parser.add_argument('--output-dir', default='sequence_models', help='Output directory for trained models')
    parser.add_argument('--sequence-length', type=int, default=15, help='Sequence length for analysis')
    
    args = parser.parse_args()
    
    try:
        # Initialize trainer
        trainer = SequenceBounceTrainer(sequence_length=args.sequence_length)
        
        # Prepare sequence features
        features, labels = trainer.prepare_sequence_features(args.annotations, args.ball_data)
        
        if len(features) == 0:
            logger.error("No valid sequences created!")
            return 1
        
        # Train models
        results = trainer.train_models(features, labels)
        
        # Save models
        trainer.save_models(args.output_dir)
        
        # Print results
        print(f"\n🎉 Sequence-based Training completed!")
        
        best_model = None
        best_auc = 0
        
        for name, result in results.items():
            auc = result['auc_score']
            print(f"{name.upper()}: AUC = {auc:.4f}")
            
            if auc > best_auc:
                best_auc = auc
                best_model = name
        
        print(f"\n🏆 BEST MODEL: {best_model.upper()} (AUC: {best_auc:.4f})")
        
        # Classification report for best model
        if best_model:
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
