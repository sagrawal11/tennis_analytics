#!/usr/bin/env python3
"""
Tennis Shot Classifier
Unified script for tennis shot classification with ML training and inference capabilities
"""

import cv2
import numpy as np
import pandas as pd
import argparse
import logging
import sys
import os
from pathlib import Path
from typing import List, Optional, Dict, Tuple, Any
from dataclasses import dataclass
from enum import Enum
from collections import deque
import joblib
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MovementType(Enum):
    """Enumeration of movement states"""
    READY_STANCE = "ready_stance"
    MOVING = "moving"
    UNKNOWN = "unknown"

class ShotType(Enum):
    """Enumeration of shot types"""
    FOREHAND = "forehand"
    BACKHAND = "backhand"
    OVERHEAD_SMASH = "overhead_smash"
    SERVE = "serve"
    READY_STANCE = "ready_stance"
    MOVING = "moving"
    UNKNOWN = "unknown"

@dataclass
class PlayerData:
    """Container for player-specific data"""
    bbox: List[int]  # [x1, y1, x2, y2]
    center: Tuple[float, float]  # (x, y)
    feet_position: Tuple[float, float]  # Bottom 10% of bbox (x, y)
    pose_keypoints: Optional[List[Tuple[float, float, float]]] = None  # [(x, y, confidence), ...]
    movement_type: MovementType = MovementType.UNKNOWN
    shot_type: ShotType = ShotType.UNKNOWN
    confidence: float = 0.0

@dataclass
class FrameData:
    """Container for complete frame data"""
    frame_number: int
    players: List[PlayerData]
    ball_position: Optional[Tuple[float, float]] = None
    ball_confidence: float = 0.0

class MovementClassifier:
    """Classifier for distinguishing between ready stance and moving"""
    
    def __init__(self):
        self.name = "Movement Classifier"
        
        # Movement detection parameters
        self.movement_history_length = 15  # Frames to track for movement analysis
        self.movement_threshold = 40.0     # Pixels of movement to classify as "moving"
        self.ready_stance_threshold = 20.0 # Pixels below which is definitely ready stance
        self.min_frames_for_analysis = 5   # Minimum frames needed for reliable analysis
        
        # Player movement history: {player_id: deque of feet_positions}
        self.movement_history = {}
        
        logger.info(f"Movement classifier initialized - Moving: >{self.movement_threshold}px, Ready: <{self.ready_stance_threshold}px")
    
    def classify_movement(self, player_data: PlayerData, frame_data: FrameData) -> Tuple[MovementType, float]:
        """Classify player movement based on feet position over time"""
        player_id = self._get_player_id(player_data, frame_data)
        
        # Initialize history for new players
        if player_id not in self.movement_history:
            self.movement_history[player_id] = deque(maxlen=self.movement_history_length)
        
        # Add current feet position to history
        self.movement_history[player_id].append(player_data.feet_position)
        
        # Need minimum frames for reliable analysis
        if len(self.movement_history[player_id]) < self.min_frames_for_analysis:
            return MovementType.UNKNOWN, 0.0
        
        # Calculate movement metrics
        movement_distance = self._calculate_movement_distance(player_id)
        
        # Classify based on movement distance
        if movement_distance < self.ready_stance_threshold:
            confidence = min(0.9, 0.7 + (self.ready_stance_threshold - movement_distance) / 10.0)
            return MovementType.READY_STANCE, confidence
        elif movement_distance > self.movement_threshold:
            confidence = min(0.9, 0.7 + (movement_distance - self.movement_threshold) / 20.0)
            return MovementType.MOVING, confidence
        else:
            # Buffer zone - use previous classification or default to ready stance
            return MovementType.READY_STANCE, 0.5
    
    def _get_player_id(self, player_data: PlayerData, frame_data: FrameData) -> int:
        """Get player ID based on position in frame"""
        for i, p in enumerate(frame_data.players):
            if p == player_data:
                return i
        return 0  # Fallback
    
    def _calculate_movement_distance(self, player_id: int) -> float:
        """Calculate total movement distance over recent frames"""
        history = self.movement_history[player_id]
        if len(history) < 2:
            return 0.0
        
        total_distance = 0.0
        for i in range(1, len(history)):
            prev_pos = history[i-1]
            curr_pos = history[i]
            distance = np.sqrt((curr_pos[0] - prev_pos[0])**2 + (curr_pos[1] - prev_pos[1])**2)
            total_distance += distance
        
        return total_distance

class ShotClassifier:
    """ML-based classifier for tennis shot detection using learned patterns"""
    
    def __init__(self, model_path: Optional[str] = None):
        self.name = "ML-Based Shot Classifier"
        
        # ML-based parameters (learned from data analysis) - now relative to player size
        self.ball_proximity_ratio = 0.8  # Ball within 80% of player width
        self.ball_distance_ratio = 1.2   # Ball within 120% of player width
        self.arm_extension_ratio = 0.15  # Arm extension at least 15% of player width
        self.min_confidence = 0.5
        
        # ML-based decision thresholds - now relative to player size
        self.near_player_forehand_wrist_ratio = 0.0  # Positive = forehand (relative to player width)
        self.near_player_backhand_wrist_ratio = -0.1  # Negative = backhand (10% of player width)
        self.far_player_arm_angle_threshold = 45.0  # Different angle patterns
        
        # Overhead/serve detection ratios
        self.overhead_wrist_y_ratio = -0.3  # Wrist 30% above body center
        self.overhead_arm_extension_ratio = 0.25  # Arm extension 25% of player width
        self.serve_wrist_y_ratio = -0.2  # Wrist 20% above body center
        self.serve_arm_extension_ratio = 0.2  # Arm extension 20% of player width
        
        # Shot persistence
        self.shot_identification_frames = 30
        self.current_shot_state = {}  # {player_id: (shot_type, frames_remaining)}
        
        # Ball proximity smoothing
        self.ball_proximity_history = {}
        self.proximity_smoothing_frames = 5
        self.max_proximity_gap = 3
        
        # ML model components
        self.model = None
        self.scaler = None
        self.label_encoder = None
        self.feature_names = None
        
        # Load model if provided
        if model_path and Path(model_path).exists():
            self.load_model(model_path)
        
        logger.info(f"ML-based shot classifier initialized with relative measurements:")
        logger.info(f"  - Ball proximity: {self.ball_proximity_ratio:.1f} player widths")
        logger.info(f"  - Ball distance: {self.ball_distance_ratio:.1f} player widths") 
        logger.info(f"  - Arm extension: {self.arm_extension_ratio:.1f} player widths")
    
    def load_model(self, model_path: str):
        """Load trained ML model"""
        try:
            model_data = joblib.load(model_path)
            self.model = model_data['model']
            self.scaler = model_data['scaler']
            self.label_encoder = model_data['label_encoder']
            self.feature_names = model_data['feature_names']
            logger.info(f"✓ Loaded trained model from {model_path}")
        except Exception as e:
            logger.warning(f"Failed to load model: {e}")
            self.model = None
    
    def classify_shot(self, player_data: PlayerData, frame_data: FrameData) -> Tuple[ShotType, float]:
        """Classify shot using ML-based insights"""
        player_id = self._get_player_id(player_data, frame_data)
        ball_distance = self._get_ball_distance_to_player(player_data, frame_data)
        
        # Check if we're continuing a current shot
        if player_id in self.current_shot_state:
            shot_type, frames_remaining = self.current_shot_state[player_id]
            player_width = self._get_player_width(player_data)
            ball_distance_ratio = ball_distance / player_width if player_width > 0 else float('inf')
            if ball_distance_ratio <= self.ball_distance_ratio and frames_remaining > 0:
                self.current_shot_state[player_id] = (shot_type, frames_remaining - 1)
                return shot_type, 0.9  # High confidence for continuing shots
        
        # Check ball proximity using relative distance
        player_width = self._get_player_width(player_data)
        ball_distance_ratio = ball_distance / player_width if player_width > 0 else float('inf')
        
        ball_is_near = ball_distance_ratio <= self.ball_proximity_ratio
        
        if not ball_is_near:
            # Ball is far - clear any current shot
            if player_id in self.current_shot_state:
                del self.current_shot_state[player_id]
            return ShotType.UNKNOWN, 0.0
        
        # Ball is near - try to detect shot
        if self.model:
            shot_type, confidence = self._detect_shot_ml_model(player_data, player_id)
        else:
            shot_type, confidence = self._detect_shot_rule_based(player_data, player_id)
        
        if shot_type != ShotType.UNKNOWN and confidence > 0.6:
            # Start tracking this shot
            self.current_shot_state[player_id] = (shot_type, self.shot_identification_frames)
            logger.info(f"Frame {frame_data.frame_number}, Player {player_id}: Started {shot_type.value} "
                       f"(ball distance: {ball_distance_ratio:.2f} player widths, confidence: {confidence:.2f})")
        
        return shot_type, confidence
    
    def _detect_shot_ml_model(self, player_data: PlayerData, player_id: int) -> Tuple[ShotType, float]:
        """Detect shot using trained ML model"""
        try:
            # Extract features
            features = self._extract_ml_features(player_data)
            if not features:
                return ShotType.UNKNOWN, 0.0
            
            # Prepare feature vector
            feature_vector = []
            for feature_name in self.feature_names:
                feature_vector.append(features.get(feature_name, 0.0))
            
            # Scale features
            feature_vector = self.scaler.transform([feature_vector])
            
            # Predict
            prediction = self.model.predict(feature_vector)[0]
            confidence = max(self.model.predict_proba(feature_vector)[0])
            
            # Convert prediction back to ShotType
            shot_type_str = self.label_encoder.inverse_transform([prediction])[0]
            shot_type = ShotType(shot_type_str)
            
            return shot_type, confidence
            
        except Exception as e:
            logger.warning(f"ML model prediction failed: {e}")
            return self._detect_shot_rule_based(player_data, player_id)
    
    def _detect_shot_rule_based(self, player_data: PlayerData, player_id: int) -> Tuple[ShotType, float]:
        """Detect shot using rule-based logic (fallback)"""
        try:
            # Extract key features
            features = self._extract_ml_features(player_data)
            
            if not features:
                return ShotType.UNKNOWN, 0.0
            
            # Use ML-based decision tree
            if player_id == 0:  # Near player
                return self._classify_near_player_shot(features)
            else:  # Far player
                return self._classify_far_player_shot(features)
                
        except Exception as e:
            logger.warning(f"Error in rule-based shot detection: {e}")
            return ShotType.UNKNOWN, 0.0
    
    def _get_player_id(self, player_data: PlayerData, frame_data: FrameData) -> int:
        """Get player ID based on position in frame"""
        for i, p in enumerate(frame_data.players):
            if p == player_data:
                return i
        return 0  # Fallback
    
    def _get_player_width(self, player_data: PlayerData) -> float:
        """Get player width from bounding box"""
        if not player_data.bbox or len(player_data.bbox) < 4:
            return 100.0  # Fallback width
        return float(player_data.bbox[2] - player_data.bbox[0])  # x2 - x1
    
    def _extract_ml_features(self, player_data: PlayerData) -> Optional[Dict]:
        """Extract ML features from player data"""
        try:
            if not player_data.pose_keypoints or len(player_data.pose_keypoints) < 17:
                return None
                
            # Key points: 5=left_shoulder, 6=right_shoulder, 8=right_elbow, 10=right_wrist
            keypoints = {
                'left_shoulder': player_data.pose_keypoints[5] if len(player_data.pose_keypoints) > 5 else [0, 0, 0],
                'right_shoulder': player_data.pose_keypoints[6] if len(player_data.pose_keypoints) > 6 else [0, 0, 0],
                'right_elbow': player_data.pose_keypoints[8] if len(player_data.pose_keypoints) > 8 else [0, 0, 0],
                'right_wrist': player_data.pose_keypoints[10] if len(player_data.pose_keypoints) > 10 else [0, 0, 0],
            }
            
            # Check confidence
            if (keypoints['right_elbow'][2] < self.min_confidence or 
                keypoints['right_wrist'][2] < self.min_confidence):
                return None
            
            # Calculate features
            features = {}
            
            # Body center
            if (keypoints['left_shoulder'][2] > 0.5 and keypoints['right_shoulder'][2] > 0.5):
                features['body_center_x'] = (keypoints['left_shoulder'][0] + keypoints['right_shoulder'][0]) / 2
                features['body_center_y'] = (keypoints['left_shoulder'][1] + keypoints['right_shoulder'][1]) / 2
            else:
                # Use bbox center as fallback
                features['body_center_x'] = player_data.center[0]
                features['body_center_y'] = player_data.center[1]
            
            # Get player width for relative measurements
            player_width = self._get_player_width(player_data)
            
            # Wrist position relative to body (in pixels)
            wrist_relative_x_px = keypoints['right_wrist'][0] - features['body_center_x']
            wrist_relative_y_px = keypoints['right_wrist'][1] - features['body_center_y']
            
            # Convert to relative measurements (as ratio of player width)
            features['wrist_relative_x'] = wrist_relative_x_px / player_width if player_width > 0 else 0
            features['wrist_relative_y'] = wrist_relative_y_px / player_width if player_width > 0 else 0
            
            # Arm extension (in pixels)
            arm_extension_px = np.sqrt(
                (keypoints['right_wrist'][0] - keypoints['right_elbow'][0])**2 +
                (keypoints['right_wrist'][1] - keypoints['right_elbow'][1])**2
            )
            
            # Convert to relative measurement
            features['arm_extension'] = arm_extension_px / player_width if player_width > 0 else 0
            
            # Arm angle
            dx = keypoints['right_wrist'][0] - keypoints['right_elbow'][0]
            dy = keypoints['right_wrist'][1] - keypoints['right_elbow'][1]
            features['arm_angle'] = np.arctan2(dy, dx) * 180 / np.pi
            
            # Player position (feet)
            features['feet_x'] = player_data.feet_position[0]
            features['feet_y'] = player_data.feet_position[1]
            
            return features
            
        except Exception as e:
            logger.warning(f"Error extracting ML features: {e}")
            return None
    
    def _classify_near_player_shot(self, features: Dict) -> Tuple[ShotType, float]:
        """Classify near player shot using ML insights"""
        # Near player: wrist X position is the key differentiator
        wrist_x = features['wrist_relative_x']
        arm_extension = features['arm_extension']
        wrist_y = features['wrist_relative_y']
        arm_angle = features['arm_angle']
        
        if arm_extension < self.arm_extension_ratio:
            return ShotType.UNKNOWN, 0.3
        
        # Check for overhead smash (wrist high above body, arm extended upward)
        if (wrist_y < self.overhead_wrist_y_ratio and 
            arm_extension > self.overhead_arm_extension_ratio and 
            (arm_angle > 60 or arm_angle < -60)):
            confidence = min(0.9, 0.7 + (arm_extension / 0.4))  # Scale by ratio
            return ShotType.OVERHEAD_SMASH, confidence
        
        # Check for serve (wrist high, arm extended, specific angle range)
        if (wrist_y < self.serve_wrist_y_ratio and 
            arm_extension > self.serve_arm_extension_ratio and 
            (arm_angle > 45 or arm_angle < -45)):
            confidence = min(0.9, 0.6 + (arm_extension / 0.35))  # Scale by ratio
            return ShotType.SERVE, confidence
        
        # ML insight: Near player forehand has positive wrist X, backhand has negative
        if wrist_x > self.near_player_forehand_wrist_ratio:
            confidence = min(0.9, 0.6 + (wrist_x / 0.2))  # Scale confidence by ratio
            return ShotType.FOREHAND, confidence
        elif wrist_x < self.near_player_backhand_wrist_ratio:
            confidence = min(0.9, 0.6 + (abs(wrist_x) / 0.2))
            return ShotType.BACKHAND, confidence
        else:
            return ShotType.UNKNOWN, 0.4
    
    def _classify_far_player_shot(self, features: Dict) -> Tuple[ShotType, float]:
        """Classify far player shot using ML insights"""
        # Far player: wrist X position is not reliable, use other features
        arm_extension = features['arm_extension']
        arm_angle = features['arm_angle']
        feet_x = features['feet_x']
        wrist_y = features['wrist_relative_y']
        
        if arm_extension < self.arm_extension_ratio:
            return ShotType.UNKNOWN, 0.3
        
        # Check for overhead smash (wrist high above body, arm extended upward)
        if (wrist_y < self.overhead_wrist_y_ratio and 
            arm_extension > self.overhead_arm_extension_ratio and 
            (arm_angle > 50 or arm_angle < -50)):
            confidence = min(0.9, 0.7 + (arm_extension / 0.35))  # Scale by ratio
            return ShotType.OVERHEAD_SMASH, confidence
        
        # Check for serve (wrist high, arm extended, specific angle range)
        if (wrist_y < self.serve_wrist_y_ratio and 
            arm_extension > self.serve_arm_extension_ratio and 
            (arm_angle > 40 or arm_angle < -40)):
            confidence = min(0.9, 0.6 + (arm_extension / 0.3))  # Scale by ratio
            return ShotType.SERVE, confidence
        
        # ML insight: Far player needs different approach
        # Use arm angle and position patterns
        if arm_angle > 45 and arm_angle < 135:  # Arm pointing right
            confidence = min(0.9, 0.6 + (arm_extension / 0.3))  # Scale by ratio
            return ShotType.FOREHAND, confidence
        elif arm_angle < -45 and arm_angle > -135:  # Arm pointing left
            confidence = min(0.9, 0.6 + (arm_extension / 0.3))  # Scale by ratio
            return ShotType.BACKHAND, confidence
        else:
            # Fallback to position-based classification
            if feet_x > 1000:  # Right side of court
                return ShotType.FOREHAND, 0.6
            else:  # Left side of court
                return ShotType.BACKHAND, 0.6
    
    def _get_ball_distance_to_player(self, player_data: PlayerData, frame_data: FrameData) -> float:
        """Calculate distance between ball and player center"""
        if not frame_data.ball_position:
            return float('inf')  # No ball detected
        
        ball_x, ball_y = frame_data.ball_position
        player_center = player_data.center
        
        distance = np.sqrt((ball_x - player_center[0])**2 + (ball_y - player_center[1])**2)
        return distance

class TennisShotMLTrainer:
    """ML trainer for tennis shot classification"""
    
    def __init__(self, annotations_csv: str):
        self.annotations_csv = annotations_csv
        self.annotations = []
        self.training_samples = []
        
    def load_annotations(self):
        """Load annotations from CSV file"""
        logger.info(f"Loading annotations from {self.annotations_csv}")
        df = pd.read_csv(self.annotations_csv)
        
        for _, row in df.iterrows():
            annotation = {
                'video_file': row['video_file'],
                'start_frame': int(row['start_frame']),
                'end_frame': int(row['end_frame']),
                'player_id': int(row['player_id']),
                'shot_type': row['shot_type'].lower(),
                'notes': row.get('notes', '')
            }
            self.annotations.append(annotation)
        
        logger.info(f"Loaded {len(self.annotations)} annotations")
        return self.annotations
    
    def generate_training_data(self, analysis_csv: str):
        """Generate training data from analysis CSV and annotations"""
        logger.info(f"Generating training data from {analysis_csv}")
        
        if not Path(analysis_csv).exists():
            logger.error(f"Analysis CSV file not found: {analysis_csv}")
            return None
        
        df = pd.read_csv(analysis_csv)
        logger.info(f"Loaded {len(df)} frames from analysis CSV")
        
        # Group annotations by video
        video_annotations = {}
        for ann in self.annotations:
            if ann['video_file'] not in video_annotations:
                video_annotations[ann['video_file']] = []
            video_annotations[ann['video_file']].append(ann)
        
        training_data = []
        
        # Process each annotation
        for annotation in self.annotations:
            logger.info(f"Processing annotation: {annotation['shot_type']} for player {annotation['player_id']} "
                       f"frames {annotation['start_frame']}-{annotation['end_frame']}")
            
            for frame_idx in range(annotation['start_frame'], annotation['end_frame'] + 1):
                if frame_idx >= len(df):
                    continue
                
                row = df.iloc[frame_idx]
                sample = self._extract_training_sample(annotation, row, frame_idx)
                
                if sample:
                    training_data.append(sample)
        
        self.training_samples = training_data
        logger.info(f"Generated {len(training_data)} training samples")
        return training_data
    
    def _extract_training_sample(self, annotation: Dict, row: pd.Series, frame_idx: int) -> Optional[Dict]:
        """Extract training sample from annotation and CSV row"""
        try:
            # Parse player data
            player_bboxes = self._parse_player_bboxes(row.get('player_bboxes', ''))
            player_poses = self._parse_player_poses(row.get('pose_keypoints', ''))
            
            if len(player_bboxes) <= annotation['player_id'] or len(player_poses) <= annotation['player_id']:
                return None
            
            # Extract features for the specific player
            bbox = player_bboxes[annotation['player_id']]
            pose = player_poses[annotation['player_id']]
            
            features = self._extract_ml_features(bbox, pose, row.get('ball_x', 0), row.get('ball_y', 0))
            
            if not features:
                return None
            
            return {
                'video_file': annotation['video_file'],
                'frame_number': frame_idx,
                'player_id': annotation['player_id'],
                'true_shot_type': annotation['shot_type'],
                **features
            }
            
        except Exception as e:
            logger.warning(f"Error extracting training sample: {e}")
            return None
    
    def _parse_player_bboxes(self, bbox_str: str) -> List[List[float]]:
        """Parse player bounding boxes from CSV string"""
        if not bbox_str or bbox_str == 'nan':
            return []
        try:
            bbox_parts = bbox_str.split(';')
            bboxes = []
            for part in bbox_parts:
                coords = [float(x) for x in part.split(',')]
                if len(coords) == 4:
                    bboxes.append(coords)
            return bboxes
        except:
            return []
    
    def _parse_player_poses(self, pose_str: str) -> List[List[float]]:
        """Parse player pose keypoints from CSV string"""
        if not pose_str or pose_str == 'nan':
            return []
        try:
            player_poses = []
            players = pose_str.split(';')
            for player in players:
                keypoints = []
                kp_parts = player.split('|')
                for kp in kp_parts:
                    coords = [float(x) for x in kp.split(',')]
                    if len(coords) == 3:
                        keypoints.append(coords)
                player_poses.append(keypoints)
            return player_poses
        except:
            return []
    
    def _extract_ml_features(self, bbox: List[float], pose: List[List[float]], 
                           ball_x: float, ball_y: float) -> Optional[Dict]:
        """Extract ML features from player data"""
        try:
            if len(bbox) < 4 or len(pose) < 17:
                return None
                
            x1, y1, x2, y2 = bbox
            features = {}
            
            # Basic bounding box features
            features['bbox_center_x'] = (x1 + x2) / 2
            features['bbox_center_y'] = (y1 + y2) / 2
            features['bbox_width'] = x2 - x1
            features['bbox_height'] = y2 - y1
            features['bbox_area'] = (x2 - x1) * (y2 - y1)
            
            # Feet position (bottom 10% of bbox)
            features['feet_y'] = y2
            features['feet_x'] = (x1 + x2) / 2
            
            # Ball distance (relative to player width)
            if ball_x > 0 and ball_y > 0:
                ball_distance_px = np.sqrt((features['bbox_center_x'] - ball_x)**2 + 
                                         (features['bbox_center_y'] - ball_y)**2)
                features['ball_distance'] = ball_distance_px / features['bbox_width'] if features['bbox_width'] > 0 else float('inf')
            else:
                features['ball_distance'] = float('inf')
            
            # Pose keypoint features
            if len(pose) >= 17:
                # Key points: 5=left_shoulder, 6=right_shoulder, 8=right_elbow, 10=right_wrist
                keypoints = {
                    'left_shoulder': pose[5] if len(pose) > 5 else [0, 0, 0],
                    'right_shoulder': pose[6] if len(pose) > 6 else [0, 0, 0],
                    'right_elbow': pose[8] if len(pose) > 8 else [0, 0, 0],
                    'right_wrist': pose[10] if len(pose) > 10 else [0, 0, 0],
                }
                
                # Extract keypoint coordinates and confidence
                for name, kp in keypoints.items():
                    if len(kp) >= 3:
                        features[f'{name}_x'] = kp[0]
                        features[f'{name}_y'] = kp[1] 
                        features[f'{name}_conf'] = kp[2]
                    else:
                        features[f'{name}_x'] = 0
                        features[f'{name}_y'] = 0
                        features[f'{name}_conf'] = 0
                
                # Calculate body center (midpoint of shoulders)
                if (features['left_shoulder_conf'] > 0.5 and 
                    features['right_shoulder_conf'] > 0.5):
                    features['body_center_x'] = (features['left_shoulder_x'] + 
                                               features['right_shoulder_x']) / 2
                    features['body_center_y'] = (features['left_shoulder_y'] + 
                                               features['right_shoulder_y']) / 2
                else:
                    features['body_center_x'] = features['bbox_center_x']
                    features['body_center_y'] = features['bbox_center_y']
                
                # Arm extension (distance from elbow to wrist, relative to player width)
                if (features['right_elbow_conf'] > 0.5 and 
                    features['right_wrist_conf'] > 0.5):
                    arm_extension_px = np.sqrt(
                        (features['right_wrist_x'] - features['right_elbow_x'])**2 +
                        (features['right_wrist_y'] - features['right_elbow_y'])**2
                    )
                    features['arm_extension'] = arm_extension_px / features['bbox_width'] if features['bbox_width'] > 0 else 0
                else:
                    features['arm_extension'] = 0
                
                # Wrist position relative to body center (in pixels first)
                if (features['right_wrist_conf'] > 0.5 and 
                    features['body_center_x'] > 0):
                    wrist_relative_x_px = features['right_wrist_x'] - features['body_center_x']
                    wrist_relative_y_px = features['right_wrist_y'] - features['body_center_y']
                    
                    # Convert to relative measurements
                    features['wrist_relative_x'] = wrist_relative_x_px / features['bbox_width'] if features['bbox_width'] > 0 else 0
                    features['wrist_relative_y'] = wrist_relative_y_px / features['bbox_width'] if features['bbox_width'] > 0 else 0
                    features['wrist_distance_from_center'] = np.sqrt(
                        features['wrist_relative_x']**2 + features['wrist_relative_y']**2
                    )
                else:
                    features['wrist_relative_x'] = 0
                    features['wrist_relative_y'] = 0
                    features['wrist_distance_from_center'] = 0
                
                # Arm angle (angle of arm from elbow to wrist)
                if (features['right_elbow_conf'] > 0.5 and 
                    features['right_wrist_conf'] > 0.5):
                    dx = features['right_wrist_x'] - features['right_elbow_x']
                    dy = features['right_wrist_y'] - features['right_elbow_y']
                    features['arm_angle'] = np.arctan2(dy, dx) * 180 / np.pi
                else:
                    features['arm_angle'] = 0
                    
            return features
            
        except Exception as e:
            logger.warning(f"Error extracting ML features: {e}")
            return None
    
    def train_models(self, output_path: str = "tennis_shot_model.pkl"):
        """Train ML models on the training data"""
        if not self.training_samples:
            logger.error("No training samples available")
            return None
        
        # Convert to DataFrame
        df = pd.DataFrame(self.training_samples)
        
        # Replace infinite values
        df = df.replace([np.inf, -np.inf], 999999)
        
        # Select feature columns (exclude metadata)
        exclude_cols = ['video_file', 'frame_number', 'player_id', 'true_shot_type']
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        # Remove columns with all NaN values
        valid_cols = []
        for col in feature_cols:
            if not df[col].isna().all():
                valid_cols.append(col)
        
        X = df[valid_cols].fillna(0)
        y = df['true_shot_type']
        
        logger.info(f"Training on {len(X)} samples with {len(valid_cols)} features")
        logger.info(f"Features: {valid_cols}")
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train Random Forest (usually performs well)
        logger.info("Training Random Forest...")
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X_train, y_train)
        
        # Evaluate
        y_pred = rf.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        logger.info(f"Random Forest accuracy: {accuracy:.4f}")
        
        # Save model
        model_data = {
            'model': rf,
            'scaler': scaler,
            'label_encoder': LabelEncoder().fit(y),
            'feature_names': valid_cols
        }
        
        joblib.dump(model_data, output_path)
        logger.info(f"✓ Model saved to {output_path}")
        
        # Show feature importance
        importances = rf.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        logger.info("Top 10 most important features:")
        for i in range(min(10, len(indices))):
            logger.info(f"{i+1:2d}. {valid_cols[indices[i]]:30s} {importances[indices[i]]:.4f}")
        
        return model_data

def main():
    parser = argparse.ArgumentParser(description='Tennis Shot Classifier - Unified Training and Inference')
    parser.add_argument('--csv', help='Input CSV file with analysis data')
    parser.add_argument('--video', help='Input video file')
    parser.add_argument('--annotations', help='Annotations CSV file for training')
    parser.add_argument('--model', help='Trained model file for inference')
    parser.add_argument('--output', help='Output video file')
    parser.add_argument('--train', action='store_true', help='Train the model')
    parser.add_argument('--viewer', action='store_true', help='Show viewer window')
    
    args = parser.parse_args()
    
    if args.train:
        # Training mode
        if not args.annotations:
            logger.error("Annotations CSV required for training (--annotations)")
            return 1
        
        if not args.csv:
            logger.error("Analysis CSV required for training (--csv)")
            return 1
        
        trainer = TennisShotMLTrainer(args.annotations)
        trainer.load_annotations()
        trainer.generate_training_data(args.csv)
        trainer.train_models()
        
        logger.info("✓ Training completed!")
        return 0
    
    else:
        # Inference mode
        if not args.csv or not args.video:
            logger.error("CSV and video required for inference")
            return 1
        
        # Load shot classifier
        shot_classifier = ShotClassifier(args.model)
        movement_classifier = MovementClassifier()
        
        # Process video (simplified for now)
        logger.info("Inference mode - processing video...")
        logger.info("Note: Full video processing not implemented yet")
        logger.info("Use --train to train the model first")
        
        return 0

if __name__ == "__main__":
    sys.exit(main())
