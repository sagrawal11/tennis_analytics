#!/usr/bin/env python3
"""
Ensemble Tennis Ball Bounce Detection

Combines multiple models and approaches for maximum performance.
"""

import pandas as pd
import numpy as np
import logging
from typing import List, Tuple, Dict, Optional
import joblib
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')
logger = logging.getLogger(__name__)


class EnsembleBounceDetector:
    """Ensemble bounce detector combining multiple approaches"""
    
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.last_bounce_frame = -1
        self.min_bounce_gap = 3
        
    def load_models(self, model_dirs: List[str]):
        """Load multiple trained models"""
        for model_dir in model_dirs:
            model_path = Path(model_dir)
            if not model_path.exists():
                logger.warning(f"Model directory {model_dir} not found")
                continue
            
            # Load all models in directory
            for model_file in model_path.glob("*_model.joblib"):
                model_name = model_file.stem.replace("_model", "")
                try:
                    model = joblib.load(model_file)
                    self.models[model_name] = model
                    logger.info(f"Loaded {model_name} from {model_file}")
                except Exception as e:
                    logger.warning(f"Could not load {model_file}: {e}")
            
            # Load scalers
            for scaler_file in model_path.glob("*scaler.joblib"):
                scaler_name = scaler_file.stem.replace("_scaler", "")
                try:
                    scaler = joblib.load(scaler_file)
                    self.scalers[scaler_name] = scaler
                    logger.info(f"Loaded {scaler_name} scaler")
                except Exception as e:
                    logger.warning(f"Could not load scaler {scaler_file}: {e}")
    
    def detect_bounce_ensemble(self, ball_trajectory: List[Tuple[float, float, int]], 
                              current_frame: int) -> Tuple[bool, float, Dict]:
        """
        Ensemble bounce detection using multiple models and heuristics
        """
        # Skip if too soon after last bounce
        if current_frame - self.last_bounce_frame < self.min_bounce_gap:
            return False, 0.0, {"reason": "too_soon_after_last_bounce"}
        
        # Get predictions from all approaches
        predictions = {}
        confidences = {}
        
        # 1. Heuristic detection
        heuristic_conf = self._heuristic_detection(ball_trajectory, current_frame)
        predictions['heuristic'] = heuristic_conf > 0.7
        confidences['heuristic'] = heuristic_conf
        
        # 2. ML model predictions
        for model_name, model in self.models.items():
            try:
                ml_conf = self._ml_prediction(model, ball_trajectory, current_frame, model_name)
                predictions[model_name] = ml_conf > 0.5
                confidences[model_name] = ml_conf
            except Exception as e:
                logger.warning(f"Error with {model_name}: {e}")
                continue
        
        # 3. Ensemble decision
        ensemble_confidence = self._ensemble_decision(predictions, confidences)
        is_bounce = ensemble_confidence >= 0.6  # Lower threshold for ensemble
        
        if is_bounce:
            self.last_bounce_frame = current_frame
        
        details = {
            "ensemble_confidence": ensemble_confidence,
            "individual_predictions": predictions,
            "individual_confidences": confidences,
            "model_count": len(self.models)
        }
        
        return is_bounce, ensemble_confidence, details
    
    def _heuristic_detection(self, ball_trajectory: List[Tuple[float, float, int]], 
                           current_frame: int) -> float:
        """Enhanced heuristic bounce detection"""
        try:
            # Get recent trajectory
            recent_trajectory = [(x, y, frame) for x, y, frame in ball_trajectory 
                               if frame >= current_frame - 15 and x is not None and y is not None]
            
            if len(recent_trajectory) < 8:
                return 0.0
            
            recent_trajectory.sort(key=lambda x: x[2])
            
            x_coords = np.array([pos[0] for pos in recent_trajectory])
            y_coords = np.array([pos[1] for pos in recent_trajectory])
            
            confidence = 0.0
            
            # 1. Y-velocity reversal (40% weight)
            y_vel_score = self._analyze_y_velocity_reversal(y_coords)
            confidence += y_vel_score * 0.4
            
            # 2. Ground contact detection (30% weight)
            ground_score = self._detect_ground_contact(y_coords)
            confidence += ground_score * 0.3
            
            # 3. Trajectory curvature (20% weight)
            curve_score = self._analyze_trajectory_curvature(x_coords, y_coords)
            confidence += curve_score * 0.2
            
            # 4. Speed changes (10% weight)
            speed_score = self._analyze_speed_changes(x_coords, y_coords)
            confidence += speed_score * 0.1
            
            return min(1.0, confidence)
            
        except Exception as e:
            logger.warning(f"Error in heuristic detection: {e}")
            return 0.0
    
    def _ml_prediction(self, model, ball_trajectory: List[Tuple[float, float, int]], 
                      current_frame: int, model_name: str) -> float:
        """Get ML model prediction"""
        try:
            # Extract features based on model type
            if 'ultra' in model_name:
                features = self._extract_ultra_features(ball_trajectory, current_frame)
            elif 'sequence' in model_name:
                features = self._extract_sequence_features(ball_trajectory, current_frame)
            else:
                features = self._extract_basic_features(ball_trajectory, current_frame)
            
            if features is None:
                return 0.0
            
            # Scale features if scaler available
            scaler_key = model_name.replace('random_forest', '').replace('logistic_regression', '').replace('svm', '').strip('_')
            if scaler_key in self.scalers:
                features = self.scalers[scaler_key].transform([features])[0]
            
            # Predict probability
            proba = model.predict_proba([features])[0]
            return proba[1] if len(proba) > 1 else 0.0
            
        except Exception as e:
            logger.warning(f"Error in ML prediction for {model_name}: {e}")
            return 0.0
    
    def _ensemble_decision(self, predictions: Dict[str, bool], 
                          confidences: Dict[str, float]) -> float:
        """Make ensemble decision using weighted voting"""
        
        if not predictions:
            return 0.0
        
        # Weights for different approaches
        weights = {
            'heuristic': 0.3,
            'ultra_random_forest': 0.4,
            'sequence_random_forest': 0.2,
            'random_forest': 0.1
        }
        
        # Calculate weighted confidence
        total_weight = 0
        weighted_confidence = 0
        
        for approach, confidence in confidences.items():
            weight = weights.get(approach, 0.1)  # Default weight
            weighted_confidence += confidence * weight
            total_weight += weight
        
        if total_weight > 0:
            ensemble_confidence = weighted_confidence / total_weight
        else:
            ensemble_confidence = np.mean(list(confidences.values()))
        
        # Boost confidence if multiple approaches agree
        positive_votes = sum(predictions.values())
        total_votes = len(predictions)
        
        if positive_votes > total_votes / 2:
            # Majority agreement - boost confidence
            agreement_boost = (positive_votes / total_votes) * 0.2
            ensemble_confidence = min(1.0, ensemble_confidence + agreement_boost)
        
        return ensemble_confidence
    
    def _extract_basic_features(self, ball_trajectory: List[Tuple[float, float, int]], 
                               current_frame: int) -> Optional[np.ndarray]:
        """Extract basic features for traditional models"""
        try:
            window_data = [(x, y, frame) for x, y, frame in ball_trajectory 
                          if abs(frame - current_frame) <= 10 and x is not None and y is not None]
            
            if len(window_data) < 5:
                return None
            
            window_data.sort(key=lambda x: x[2])
            
            x_coords = np.array([pos[0] for pos in window_data])
            y_coords = np.array([pos[1] for pos in window_data])
            
            # Basic features
            features = [
                np.mean(x_coords), np.mean(y_coords),
                np.std(x_coords), np.std(y_coords),
                np.max(y_coords) - np.min(y_coords),  # y_range
            ]
            
            if len(y_coords) >= 2:
                y_vel = np.diff(y_coords)
                features.extend([
                    np.mean(y_vel), np.std(y_vel), np.max(np.abs(y_vel))
                ])
                
                # Velocity reversals
                reversals = 0
                for i in range(1, len(y_vel)):
                    if (y_vel[i-1] < 0 and y_vel[i] > 0) or (y_vel[i-1] > 0 and y_vel[i] < 0):
                        reversals += 1
                features.append(reversals)
            else:
                features.extend([0, 0, 0, 0])
            
            # Pad to expected length
            while len(features) < 20:
                features.append(0.0)
            
            return np.array(features[:20])
            
        except Exception as e:
            logger.warning(f"Error extracting basic features: {e}")
            return None
    
    def _extract_sequence_features(self, ball_trajectory: List[Tuple[float, float, int]], 
                                  current_frame: int) -> Optional[np.ndarray]:
        """Extract sequence features"""
        try:
            window_data = [(x, y, frame) for x, y, frame in ball_trajectory 
                          if abs(frame - current_frame) <= 15 and x is not None and y is not None]
            
            if len(window_data) < 8:
                return None
            
            window_data.sort(key=lambda x: x[2])
            
            x_coords = np.array([pos[0] for pos in window_data])
            y_coords = np.array([pos[1] for pos in window_data])
            
            # Sequence-based features (simplified)
            features = []
            
            # Position features
            features.extend([np.mean(x_coords), np.mean(y_coords), np.std(x_coords), np.std(y_coords)])
            
            # Velocity features
            if len(x_coords) >= 2:
                x_vel = np.diff(x_coords)
                y_vel = np.diff(y_coords)
                features.extend([np.mean(x_vel), np.mean(y_vel), np.std(x_vel), np.std(y_vel)])
            else:
                features.extend([0, 0, 0, 0])
            
            # Bounce-specific features
            if len(y_coords) >= 3:
                y_vel = np.diff(y_coords)
                reversals = 0
                for i in range(1, len(y_vel)):
                    if (y_vel[i-1] < 0 and y_vel[i] > 0) or (y_vel[i-1] > 0 and y_vel[i] < 0):
                        reversals += 1
                features.append(reversals)
                features.append(np.max(np.abs(y_vel)))
            else:
                features.extend([0, 0])
            
            # Pad to expected length
            while len(features) < 34:
                features.append(0.0)
            
            return np.array(features[:34])
            
        except Exception as e:
            logger.warning(f"Error extracting sequence features: {e}")
            return None
    
    def _extract_ultra_features(self, ball_trajectory: List[Tuple[float, float, int]], 
                               current_frame: int) -> Optional[np.ndarray]:
        """Extract ultra-advanced features"""
        try:
            window_data = [(x, y, frame) for x, y, frame in ball_trajectory 
                          if abs(frame - current_frame) <= 20 and x is not None and y is not None]
            
            if len(window_data) < 10:
                return None
            
            window_data.sort(key=lambda x: x[2])
            
            x_coords = np.array([pos[0] for pos in window_data])
            y_coords = np.array([pos[1] for pos in window_data])
            
            # Ultra-advanced features (simplified version)
            features = []
            
            # Physics features
            if len(y_coords) >= 3:
                y_vel = np.diff(y_coords)
                min_y_idx = np.argmin(y_coords)
                
                # Impact/rebound velocities
                if min_y_idx > 0 and min_y_idx < len(y_vel):
                    impact_vel = abs(y_vel[min_y_idx-1])
                    rebound_vel = y_vel[min_y_idx]
                    velocity_ratio = rebound_vel / (impact_vel + 1e-6)
                else:
                    impact_vel = rebound_vel = velocity_ratio = 0.0
                
                features.extend([impact_vel, rebound_vel, velocity_ratio])
                
                # Energy analysis
                energy_before = np.mean(y_vel[:len(y_vel)//2] ** 2)
                energy_after = np.mean(y_vel[len(y_vel)//2:] ** 2)
                energy_loss = (energy_before - energy_after) / (energy_before + 1e-6)
                features.append(energy_loss)
            else:
                features.extend([0, 0, 0, 0])
            
            # Pad to expected length
            while len(features) < 58:
                features.append(0.0)
            
            return np.array(features[:58])
            
        except Exception as e:
            logger.warning(f"Error extracting ultra features: {e}")
            return None
    
    def _analyze_y_velocity_reversal(self, y_coords: np.ndarray) -> float:
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
            return min(1.0, reversal_ratio * 3)  # Scale up
        
        return 0.0
    
    def _detect_ground_contact(self, y_coords: np.ndarray) -> float:
        """Detect ground contact based on Y-position patterns"""
        if len(y_coords) < 3:
            return 0.0
        
        # Look for minimum Y position (ball on ground)
        min_y = np.min(y_coords)
        max_y = np.max(y_coords)
        y_range = max_y - min_y
        
        if y_range < 10:  # Ball not moving much vertically
            return 0.0
        
        # Count frames near minimum Y (ground contact)
        ground_threshold = min_y + y_range * 0.1  # Within 10% of minimum
        ground_frames = np.sum(y_coords <= ground_threshold)
        ground_ratio = ground_frames / len(y_coords)
        
        # Score based on ground contact duration
        if ground_ratio > 0.1 and ground_ratio < 0.4:  # Reasonable ground contact
            return min(1.0, ground_ratio * 2)
        
        return 0.0
    
    def _analyze_trajectory_curvature(self, x_coords: np.ndarray, y_coords: np.ndarray) -> float:
        """Analyze trajectory curvature"""
        if len(x_coords) < 3:
            return 0.0
        
        # Calculate total curvature
        total_curvature = 0.0
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
                total_curvature += angle_change
        
        # Normalize by trajectory length
        trajectory_length = len(x_coords)
        if trajectory_length > 0:
            normalized_curvature = total_curvature / trajectory_length
            return min(1.0, normalized_curvature / np.pi)
        
        return 0.0
    
    def _analyze_speed_changes(self, x_coords: np.ndarray, y_coords: np.ndarray) -> float:
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


def test_ensemble_detector():
    """Test the ensemble detector"""
    detector = EnsembleBounceDetector()
    
    # Load models
    model_dirs = ['ultra_models', 'sequence_models', 'advanced_models']
    detector.load_models(model_dirs)
    
    # Sample trajectory
    sample_trajectory = [
        (100, 200, 1), (105, 190, 2), (110, 180, 3), (115, 170, 4),
        (120, 160, 5), (125, 150, 6), (130, 140, 7), (135, 130, 8),
        (140, 120, 9), (145, 110, 10), (150, 100, 11), (155, 110, 12),
        (160, 120, 13), (165, 130, 14), (170, 140, 15)
    ]
    
    # Test detection
    for frame in range(5, 15):
        is_bounce, confidence, details = detector.detect_bounce_ensemble(sample_trajectory, frame)
        if is_bounce:
            print(f"Frame {frame}: Ensemble bounce detected! Confidence: {confidence:.3f}")
            print(f"  Individual confidences: {details['individual_confidences']}")
            break


if __name__ == "__main__":
    test_ensemble_detector()
