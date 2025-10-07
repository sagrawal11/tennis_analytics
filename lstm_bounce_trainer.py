#!/usr/bin/env python3
"""
LSTM-based Tennis Ball Bounce Detection

Deep learning approach using LSTM networks for sequence-based bounce detection.
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
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

# Try to import TensorFlow/Keras
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    logger.warning("TensorFlow not available. Install with: pip install tensorflow")

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')
logger = logging.getLogger(__name__)


class LSTMBounceTrainer:
    """LSTM-based bounce detection trainer"""
    
    def __init__(self, sequence_length: int = 20, feature_count: int = 4):
        """
        Initialize LSTM trainer
        
        Args:
            sequence_length: Number of frames in each sequence
            feature_count: Number of features per frame
        """
        self.sequence_length = sequence_length
        self.feature_count = feature_count
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = ['x', 'y', 'dx', 'dy']  # Basic features
        
        if not TENSORFLOW_AVAILABLE:
            raise ImportError("TensorFlow is required for LSTM training")
    
    def prepare_sequence_data(self, annotations_file: str, ball_data_file: str) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare sequence data for LSTM training"""
        # Load data
        annotations_df = self._load_annotations(annotations_file)
        ball_df = self._load_ball_data(ball_data_file)
        
        # Create sequences
        sequences, labels = self._create_sequences(annotations_df, ball_df)
        
        logger.info(f"Created {len(sequences)} sequences of length {self.sequence_length}")
        logger.info(f"Bounce sequences: {np.sum(labels)} ({np.sum(labels)/len(labels)*100:.1f}%)")
        
        return sequences, labels
    
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
    
    def _create_sequences(self, annotations_df: pd.DataFrame, ball_df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Create sequences from ball trajectory data"""
        sequences_list = []
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
                
                # Extract sequence around this frame
                sequence = self._extract_sequence(video_ball_data, center_frame)
                
                if sequence is not None:
                    sequences_list.append(sequence)
                    labels_list.append(is_bounce)
        
        return np.array(sequences_list), np.array(labels_list)
    
    def _extract_sequence(self, ball_df: pd.DataFrame, center_frame: int) -> Optional[np.ndarray]:
        """Extract a sequence of features around a center frame"""
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
            
            if len(window_data) < self.sequence_length * 0.7:
                return None
            
            # Fill missing frames with interpolation
            sequence = self._interpolate_sequence(window_data, start_frame, end_frame)
            
            if sequence is None:
                return None
            
            return sequence
            
        except Exception as e:
            logger.warning(f"Error extracting sequence at frame {center_frame}: {e}")
            return None
    
    def _interpolate_sequence(self, window_data: pd.DataFrame, start_frame: int, end_frame: int) -> Optional[np.ndarray]:
        """Interpolate missing frames in sequence"""
        try:
            # Create full frame range
            frame_range = list(range(start_frame, end_frame + 1))
            
            # Initialize sequence with NaN
            sequence = np.full((len(frame_range), self.feature_count), np.nan)
            
            # Fill in available data
            for _, row in window_data.iterrows():
                frame_idx = row['frame'] - start_frame
                if 0 <= frame_idx < len(sequence):
                    sequence[frame_idx, 0] = row['ball_x']
                    sequence[frame_idx, 1] = row['ball_y']
            
            # Interpolate missing values
            for i in range(self.feature_count):
                valid_mask = ~np.isnan(sequence[:, i])
                if np.sum(valid_mask) >= 3:  # Need at least 3 points for interpolation
                    valid_indices = np.where(valid_mask)[0]
                    valid_values = sequence[valid_indices, i]
                    sequence[:, i] = np.interp(frame_range, valid_indices + start_frame, valid_values)
                else:
                    return None
            
            # Calculate derivatives (velocity)
            if len(sequence) >= 2:
                dx = np.diff(sequence[:, 0])
                dy = np.diff(sequence[:, 1])
                
                # Pad derivatives to match sequence length
                dx_padded = np.concatenate([[dx[0]], dx])  # Repeat first value
                dy_padded = np.concatenate([[dy[0]], dy])
                
                sequence[:, 2] = dx_padded
                sequence[:, 3] = dy_padded
            
            return sequence
            
        except Exception as e:
            logger.warning(f"Error interpolating sequence: {e}")
            return None
    
    def train_model(self, sequences: np.ndarray, labels: np.ndarray, 
                   test_size: float = 0.2, epochs: int = 100) -> Dict:
        """Train LSTM model"""
        logger.info("Starting LSTM model training...")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            sequences, labels, test_size=test_size, random_state=42, stratify=labels
        )
        
        logger.info(f"Training sequences: {len(X_train)} ({np.sum(y_train)} bounces)")
        logger.info(f"Test sequences: {len(X_test)} ({np.sum(y_test)} bounces)")
        
        # Normalize features
        # Reshape for scaling: (samples * timesteps, features)
        original_shape = X_train.shape
        X_train_reshaped = X_train.reshape(-1, self.feature_count)
        X_train_scaled = self.scaler.fit_transform(X_train_reshaped)
        X_train_scaled = X_train_scaled.reshape(original_shape)
        
        # Scale test data
        X_test_reshaped = X_test.reshape(-1, self.feature_count)
        X_test_scaled = self.scaler.transform(X_test_reshaped)
        X_test_scaled = X_test_scaled.reshape(X_test.shape)
        
        # Build LSTM model
        self.model = self._build_lstm_model()
        
        # Train model
        callbacks = [
            EarlyStopping(patience=10, restore_best_weights=True),
            ReduceLROnPlateau(patience=5, factor=0.5)
        ]
        
        history = self.model.fit(
            X_train_scaled, y_train,
            validation_data=(X_test_scaled, y_test),
            epochs=epochs,
            batch_size=32,
            callbacks=callbacks,
            verbose=1
        )
        
        # Evaluate model
        y_pred_proba = self.model.predict(X_test_scaled)
        y_pred = (y_pred_proba > 0.5).astype(int)
        
        auc_score = roc_auc_score(y_test, y_pred_proba)
        
        results = {
            'auc_score': auc_score,
            'history': history.history,
            'predictions': y_pred,
            'probabilities': y_pred_proba,
            'test_labels': y_test
        }
        
        logger.info(f"LSTM Model - AUC: {auc_score:.4f}")
        
        return results
    
    def _build_lstm_model(self):
        """Build LSTM model architecture"""
        model = Sequential([
            # First LSTM layer
            LSTM(64, return_sequences=True, input_shape=(self.sequence_length, self.feature_count)),
            BatchNormalization(),
            Dropout(0.3),
            
            # Second LSTM layer
            LSTM(32, return_sequences=False),
            BatchNormalization(),
            Dropout(0.3),
            
            # Dense layers
            Dense(16, activation='relu'),
            Dropout(0.2),
            Dense(8, activation='relu'),
            Dropout(0.1),
            
            # Output layer
            Dense(1, activation='sigmoid')
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        logger.info("Built LSTM model:")
        model.summary()
        
        return model
    
    def save_model(self, output_dir: str):
        """Save trained model and scaler"""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        if self.model is not None:
            model_file = output_path / "lstm_model.h5"
            self.model.save(str(model_file))
            logger.info(f"Saved LSTM model to {model_file}")
        
        # Save scaler
        scaler_file = output_path / "lstm_scaler.joblib"
        import joblib
        joblib.dump(self.scaler, scaler_file)
        logger.info(f"Saved scaler to {scaler_file}")
        
        # Save model info
        info = {
            'sequence_length': self.sequence_length,
            'feature_count': self.feature_count,
            'feature_names': self.feature_names
        }
        
        info_file = output_path / "lstm_info.json"
        with open(info_file, 'w') as f:
            json.dump(info, f, indent=2)
        logger.info(f"Saved model info to {info_file}")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Train LSTM model for bounce detection')
    parser.add_argument('--annotations', default='all_bounce_annotations.csv', help='Annotations CSV file')
    parser.add_argument('--ball-data', default='all_ball_coordinates.csv', help='Ball tracking CSV file')
    parser.add_argument('--output-dir', default='lstm_models', help='Output directory for trained model')
    parser.add_argument('--sequence-length', type=int, default=20, help='Sequence length for LSTM')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    
    args = parser.parse_args()
    
    if not TENSORFLOW_AVAILABLE:
        logger.error("TensorFlow not available. Install with: pip install tensorflow")
        return 1
    
    try:
        # Initialize trainer
        trainer = LSTMBounceTrainer(sequence_length=args.sequence_length)
        
        # Prepare sequence data
        sequences, labels = trainer.prepare_sequence_data(args.annotations, args.ball_data)
        
        if len(sequences) == 0:
            logger.error("No valid sequences created!")
            return 1
        
        # Train model
        results = trainer.train_model(sequences, labels, epochs=args.epochs)
        
        # Save model
        trainer.save_model(args.output_dir)
        
        # Print results
        print(f"\n🎉 LSTM Training completed!")
        print(f"AUC Score: {results['auc_score']:.4f}")
        
        # Classification report
        report = classification_report(results['test_labels'], results['predictions'], 
                                    target_names=['No Bounce', 'Bounce'])
        print(f"\nClassification Report:\n{report}")
        
    except Exception as e:
        logger.error(f"Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
