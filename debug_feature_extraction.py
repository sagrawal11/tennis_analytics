#!/usr/bin/env python3
"""
Debug Feature Extraction

Debug what's happening with feature extraction and model predictions.
"""

import pandas as pd
import numpy as np
import logging
import joblib
from typing import List, Tuple, Optional
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')
logger = logging.getLogger(__name__)


def debug_feature_extraction():
    """Debug feature extraction and model predictions"""
    
    # Load model and scaler
    model_path = "ultra_models/ultra_random_forest_model.joblib"
    scaler_path = "ultra_models/ultra_scaler.joblib"
    
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    
    logger.info(f"Model loaded: {type(model)}")
    logger.info(f"Scaler loaded: {type(scaler)}")
    
    # Load some data
    annotations_df = pd.read_csv('improved_bounce_annotations.csv')
    ball_df = pd.read_csv('all_ball_coordinates.csv')
    
    # Handle column names
    if 'ball_x' in ball_df.columns:
        x_col, y_col = 'ball_x', 'ball_y'
    else:
        x_col, y_col = 'x', 'y'
    
    # Get a sample bounce frame
    bounce_annotations = annotations_df[annotations_df['is_bounce'] == 1]
    if len(bounce_annotations) == 0:
        logger.error("No bounce annotations found!")
        return
    
    sample_annotation = bounce_annotations.iloc[0]
    video_name = sample_annotation['video_name']
    frame_number = sample_annotation['frame_number']
    
    logger.info(f"Testing video: {video_name}, frame: {frame_number}")
    
    # Get ball data for this video
    video_ball_data = ball_df[ball_df.get('video_name', '') == video_name] if 'video_name' in ball_df.columns else ball_df
    
    # Create trajectory
    trajectory = []
    for _, row in video_ball_data.iterrows():
        frame = row['frame']
        x = row[x_col] if pd.notna(row[x_col]) else None
        y = row[y_col] if pd.notna(row[y_col]) else None
        trajectory.append((x, y, frame))
    
    logger.info(f"Trajectory length: {len(trajectory)}")
    
    # Test feature extraction around the bounce frame
    window_size = 20
    window_start = frame_number - window_size // 2
    window_end = frame_number + window_size // 2
    
    logger.info(f"Window: {window_start} to {window_end}")
    
    # Extract window data
    window_data = [(x, y, frame) for x, y, frame in trajectory 
                  if window_start <= frame <= window_end and x is not None and y is not None]
    
    logger.info(f"Window data points: {len(window_data)}")
    
    if len(window_data) < window_size * 0.7:
        logger.warning(f"Insufficient data in window: {len(window_data)} < {window_size * 0.7}")
        return
    
    # Sort and extract coordinates
    window_data.sort(key=lambda x: x[2])
    x_coords = np.array([pos[0] for pos in window_data])
    y_coords = np.array([pos[1] for pos in window_data])
    
    logger.info(f"X coordinates: min={np.min(x_coords):.1f}, max={np.max(x_coords):.1f}, std={np.std(x_coords):.1f}")
    logger.info(f"Y coordinates: min={np.min(y_coords):.1f}, max={np.max(y_coords):.1f}, std={np.std(y_coords):.1f}")
    
    # Test simple feature extraction
    features = extract_simple_features(x_coords, y_coords)
    
    if features is None:
        logger.error("Feature extraction failed!")
        return
    
    logger.info(f"Extracted {len(features)} features")
    logger.info(f"Feature range: min={np.min(features):.6f}, max={np.max(features):.6f}")
    logger.info(f"Feature mean: {np.mean(features):.6f}")
    logger.info(f"Feature std: {np.std(features):.6f}")
    
    # Check for NaN or infinite values
    nan_count = np.sum(np.isnan(features))
    inf_count = np.sum(np.isinf(features))
    logger.info(f"NaN values: {nan_count}, Infinite values: {inf_count}")
    
    if nan_count > 0 or inf_count > 0:
        logger.error("Features contain NaN or infinite values!")
        return
    
    # Test model prediction
    try:
        features_scaled = scaler.transform([features])
        logger.info(f"Scaled features shape: {features_scaled.shape}")
        logger.info(f"Scaled features range: min={np.min(features_scaled):.6f}, max={np.max(features_scaled):.6f}")
        
        proba = model.predict_proba(features_scaled)[0]
        logger.info(f"Model probabilities: {proba}")
        
        confidence = proba[1] if len(proba) > 1 else 0.0
        logger.info(f"Bounce confidence: {confidence:.6f}")
        
        prediction = model.predict(features_scaled)[0]
        logger.info(f"Model prediction: {prediction}")
        
    except Exception as e:
        logger.error(f"Model prediction failed: {e}")
        return
    
    # Test on a few more samples
    logger.info("\n=== Testing more samples ===")
    test_samples = bounce_annotations.head(5)
    
    for _, annotation in test_samples.iterrows():
        test_frame = annotation['frame_number']
        test_video = annotation['video_name']
        
        # Get trajectory for this video
        test_ball_data = ball_df[ball_df.get('video_name', '') == test_video] if 'video_name' in ball_df.columns else ball_df
        test_trajectory = []
        for _, row in test_ball_data.iterrows():
            frame = row['frame']
            x = row[x_col] if pd.notna(row[x_col]) else None
            y = row[y_col] if pd.notna(row[y_col]) else None
            test_trajectory.append((x, y, frame))
        
        # Extract features
        test_features = extract_simple_features_from_trajectory(test_trajectory, test_frame)
        
        if test_features is not None:
            test_features_scaled = scaler.transform([test_features])
            test_proba = model.predict_proba(test_features_scaled)[0]
            test_confidence = test_proba[1] if len(test_proba) > 1 else 0.0
            test_prediction = model.predict(test_features_scaled)[0]
            
            logger.info(f"Video: {test_video}, Frame: {test_frame}")
            logger.info(f"  Confidence: {test_confidence:.6f}, Prediction: {test_prediction}")


def extract_simple_features(x_coords: np.ndarray, y_coords: np.ndarray) -> Optional[np.ndarray]:
    """Extract simple features for debugging"""
    try:
        features = []
        
        # Basic features
        features.extend([np.mean(x_coords), np.mean(y_coords), np.std(x_coords), np.std(y_coords)])
        
        if len(x_coords) < 3:
            return None
        
        # Velocity features
        x_vel = np.diff(x_coords)
        y_vel = np.diff(y_coords)
        features.extend([np.mean(x_vel), np.mean(y_vel), np.std(x_vel), np.std(y_vel)])
        
        if len(x_vel) < 2:
            return None
        
        # Acceleration features
        x_accel = np.diff(x_vel)
        y_accel = np.diff(y_vel)
        features.extend([np.mean(x_accel), np.mean(y_accel), np.std(x_accel), np.std(y_accel)])
        
        # Fill remaining features with zeros to reach 58
        while len(features) < 58:
            features.append(0.0)
        
        # Truncate if too many
        features = features[:58]
        
        return np.array(features)
        
    except Exception as e:
        logger.error(f"Feature extraction error: {e}")
        return None


def extract_simple_features_from_trajectory(trajectory: List[Tuple], frame_number: int) -> Optional[np.ndarray]:
    """Extract features from trajectory for debugging"""
    try:
        window_size = 20
        window_start = frame_number - window_size // 2
        window_end = frame_number + window_size // 2
        
        window_data = [(x, y, frame) for x, y, frame in trajectory 
                      if window_start <= frame <= window_end and x is not None and y is not None]
        
        if len(window_data) < window_size * 0.7:
            return None
        
        window_data.sort(key=lambda x: x[2])
        x_coords = np.array([pos[0] for pos in window_data])
        y_coords = np.array([pos[1] for pos in window_data])
        
        return extract_simple_features(x_coords, y_coords)
        
    except Exception as e:
        logger.error(f"Trajectory feature extraction error: {e}")
        return None


if __name__ == "__main__":
    debug_feature_extraction()
