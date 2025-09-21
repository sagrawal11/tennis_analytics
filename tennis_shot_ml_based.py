#!/usr/bin/env python3
"""
Tennis Shot ML-Based Classifier
Uses machine learning insights to classify shots more accurately
"""

import pandas as pd
import numpy as np
import cv2
import argparse
import logging
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import ast
from enum import Enum
from dataclasses import dataclass

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ShotType(Enum):
    UNKNOWN = "UNKNOWN"
    FOREHAND = "FOREHAND"
    BACKHAND = "BACKHAND"
    MOVING = "MOVING"
    READY_STANCE = "READY_STANCE"

@dataclass
class PlayerData:
    player_id: int
    bbox: List[float]
    pose_keypoints: List[List[float]]
    feet_position: Tuple[float, float]
    ball_distance: float

@dataclass
class FrameData:
    frame_number: int
    ball_x: float
    ball_y: float
    ball_confidence: float
    players: List[PlayerData]

class MLBasedShotClassifier:
    def __init__(self):
        # Based on ML analysis insights
        self.ball_proximity_threshold = 250.0
        self.ball_distance_threshold = 350.0
        self.arm_extension_threshold = 15.0  # Lowered based on data
        self.min_confidence = 0.5
        
        # ML-based decision thresholds
        self.near_player_forehand_wrist_x_threshold = 0.0  # Positive = forehand
        self.near_player_backhand_wrist_x_threshold = -10.0  # Negative = backhand
        self.far_player_arm_angle_threshold = 45.0  # Different angle patterns
        
        # Shot persistence
        self.shot_identification_frames = 30
        self.current_shot_state = {}  # {player_id: (shot_type, frames_remaining)}
        
    def classify_shot(self, player_data: PlayerData, frame_data: FrameData) -> Tuple[ShotType, float]:
        """Classify shot using ML-based insights"""
        player_id = player_data.player_id
        ball_distance = player_data.ball_distance
        
        # Check if we're continuing a current shot
        if player_id in self.current_shot_state:
            shot_type, frames_remaining = self.current_shot_state[player_id]
            if ball_distance <= self.ball_distance_threshold and frames_remaining > 0:
                self.current_shot_state[player_id] = (shot_type, frames_remaining - 1)
                return shot_type, 0.9  # High confidence for continuing shots
        
        # Check ball proximity
        ball_is_near = ball_distance <= self.ball_proximity_threshold
        
        if not ball_is_near:
            # Ball is far - clear any current shot
            if player_id in self.current_shot_state:
                del self.current_shot_state[player_id]
            return ShotType.UNKNOWN, 0.0
        
        # Ball is near - try to detect shot
        shot_type, confidence = self._detect_shot_ml_based(player_data)
        
        if shot_type != ShotType.UNKNOWN and confidence > 0.6:
            # Start tracking this shot
            self.current_shot_state[player_id] = (shot_type, self.shot_identification_frames)
            logger.info(f"Frame {frame_data.frame_number}, Player {player_id}: Started {shot_type.value} "
                       f"(ball distance: {ball_distance:.1f}px, confidence: {confidence:.2f})")
        
        return shot_type, confidence
    
    def _detect_shot_ml_based(self, player_data: PlayerData) -> Tuple[ShotType, float]:
        """Detect shot using ML-based features"""
        try:
            # Extract key features
            features = self._extract_ml_features(player_data)
            
            if not features:
                return ShotType.UNKNOWN, 0.0
            
            # Use ML-based decision tree
            if player_data.player_id == 0:  # Near player
                return self._classify_near_player_shot(features)
            else:  # Far player
                return self._classify_far_player_shot(features)
                
        except Exception as e:
            logger.warning(f"Error in ML-based shot detection: {e}")
            return ShotType.UNKNOWN, 0.0
    
    def _extract_ml_features(self, player_data: PlayerData) -> Optional[Dict]:
        """Extract ML features from player data"""
        try:
            if len(player_data.pose_keypoints) < 17:
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
                x1, y1, x2, y2 = player_data.bbox
                features['body_center_x'] = (x1 + x2) / 2
                features['body_center_y'] = (y1 + y2) / 2
            
            # Wrist position relative to body
            features['wrist_relative_x'] = keypoints['right_wrist'][0] - features['body_center_x']
            features['wrist_relative_y'] = keypoints['right_wrist'][1] - features['body_center_y']
            
            # Arm extension
            features['arm_extension'] = np.sqrt(
                (keypoints['right_wrist'][0] - keypoints['right_elbow'][0])**2 +
                (keypoints['right_wrist'][1] - keypoints['right_elbow'][1])**2
            )
            
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
        
        if arm_extension < self.arm_extension_threshold:
            return ShotType.UNKNOWN, 0.3
        
        # ML insight: Near player forehand has positive wrist X, backhand has negative
        if wrist_x > self.near_player_forehand_wrist_x_threshold:
            confidence = min(0.9, 0.6 + (wrist_x / 20.0))  # Scale confidence
            return ShotType.FOREHAND, confidence
        elif wrist_x < self.near_player_backhand_wrist_x_threshold:
            confidence = min(0.9, 0.6 + (abs(wrist_x) / 20.0))
            return ShotType.BACKHAND, confidence
        else:
            return ShotType.UNKNOWN, 0.4
    
    def _classify_far_player_shot(self, features: Dict) -> Tuple[ShotType, float]:
        """Classify far player shot using ML insights"""
        # Far player: wrist X position is not reliable, use other features
        arm_extension = features['arm_extension']
        arm_angle = features['arm_angle']
        feet_x = features['feet_x']
        
        if arm_extension < self.arm_extension_threshold:
            return ShotType.UNKNOWN, 0.3
        
        # ML insight: Far player needs different approach
        # Use arm angle and position patterns
        if arm_angle > 45 and arm_angle < 135:  # Arm pointing right
            confidence = min(0.9, 0.6 + (arm_extension / 30.0))
            return ShotType.FOREHAND, confidence
        elif arm_angle < -45 and arm_angle > -135:  # Arm pointing left
            confidence = min(0.9, 0.6 + (arm_extension / 30.0))
            return ShotType.BACKHAND, confidence
        else:
            # Fallback to position-based classification
            if feet_x > 1000:  # Right side of court
                return ShotType.FOREHAND, 0.6
            else:  # Left side of court
                return ShotType.BACKHAND, 0.6

class MovementClassifier:
    def __init__(self):
        self.movement_history_length = 15
        self.movement_threshold = 40.0
        self.ready_stance_threshold = 20.0
        self.movement_history = {}  # {player_id: deque of positions}
        
    def classify_movement(self, player_data: PlayerData, frame_data: FrameData) -> Tuple[ShotType, float]:
        """Classify movement type"""
        player_id = player_data.player_id
        feet_pos = player_data.feet_position
        
        # Initialize history for new player
        if player_id not in self.movement_history:
            from collections import deque
            self.movement_history[player_id] = deque(maxlen=self.movement_history_length)
        
        # Add current position
        self.movement_history[player_id].append(feet_pos)
        
        # Need at least 5 frames for analysis
        if len(self.movement_history[player_id]) < 5:
            return ShotType.UNKNOWN, 0.3
        
        # Calculate total movement
        positions = list(self.movement_history[player_id])
        total_movement = 0
        for i in range(1, len(positions)):
            dx = positions[i][0] - positions[i-1][0]
            dy = positions[i][1] - positions[i-1][1]
            total_movement += np.sqrt(dx*dx + dy*dy)
        
        # Classify based on movement
        if total_movement > self.movement_threshold:
            return ShotType.MOVING, 0.9
        elif total_movement < self.ready_stance_threshold:
            return ShotType.READY_STANCE, 0.9
        else:
            # Buffer zone - slight movement
            return ShotType.READY_STANCE, 0.6

class TennisShotMLProcessor:
    def __init__(self):
        self.shot_classifier = MLBasedShotClassifier()
        self.movement_classifier = MovementClassifier()
        
    def process_csv_data(self, csv_path: str, video_path: str, viewer: bool = False):
        """Process CSV data and classify shots"""
        logger.info(f"Loading data from {csv_path}")
        df = pd.read_csv(csv_path)
        
        logger.info(f"Processing {len(df)} frames")
        
        # Video setup for viewer
        if viewer:
            cap = cv2.VideoCapture(video_path)
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            logger.info(f"Video: {width}x{height} @ {fps}fps")
        
        for idx, row in df.iterrows():
            frame_data = self._parse_frame_data(row, idx)
            if frame_data:
                # Process each player
                for player in frame_data.players:
                    # Classify movement
                    movement_type, movement_conf = self.movement_classifier.classify_movement(
                        player, frame_data
                    )
                    
                    # Classify shot
                    shot_type, shot_conf = self.shot_classifier.classify_shot(
                        player, frame_data
                    )
                    
                    # Log results
                    if shot_type != ShotType.UNKNOWN:
                        logger.info(f"Frame {frame_data.frame_number}, Player {player.player_id}: "
                                  f"{shot_type.value} (conf: {shot_conf:.2f})")
                    elif movement_type != ShotType.UNKNOWN:
                        logger.info(f"Frame {frame_data.frame_number}, Player {player.player_id}: "
                                  f"{movement_type.value} (conf: {movement_conf:.2f})")
                
                # Show viewer
                if viewer:
                    self._show_viewer(cap, frame_data, idx)
        
        if viewer:
            cap.release()
            cv2.destroyAllWindows()
        
        logger.info("Processing completed!")
    
    def _parse_frame_data(self, row: pd.Series, frame_idx: int) -> Optional[FrameData]:
        """Parse a single frame of data"""
        try:
            # Basic frame info
            frame_data = FrameData(
                frame_number=frame_idx,
                ball_x=row.get('ball_x', 0),
                ball_y=row.get('ball_y', 0),
                ball_confidence=row.get('ball_confidence', 0),
                players=[]
            )
            
            # Parse player data
            player_bboxes = self._parse_player_bboxes(row.get('player_bboxes', ''))
            player_poses = self._parse_player_poses(row.get('pose_keypoints', ''))
            
            if len(player_bboxes) >= 2 and len(player_poses) >= 2:
                for i in range(2):
                    bbox = player_bboxes[i]
                    pose = player_poses[i]
                    
                    # Calculate feet position (bottom 10% of bbox)
                    x1, y1, x2, y2 = bbox
                    feet_pos = ((x1 + x2) / 2, y2)
                    
                    # Calculate ball distance
                    ball_distance = float('inf')
                    if frame_data.ball_x > 0 and frame_data.ball_y > 0:
                        ball_distance = np.sqrt(
                            (frame_data.ball_x - (x1 + x2)/2)**2 + 
                            (frame_data.ball_y - (y1 + y2)/2)**2
                        )
                    
                    player_data = PlayerData(
                        player_id=i,
                        bbox=bbox,
                        pose_keypoints=pose,
                        feet_position=feet_pos,
                        ball_distance=ball_distance
                    )
                    frame_data.players.append(player_data)
            
            return frame_data
            
        except Exception as e:
            logger.warning(f"Error parsing frame {frame_idx}: {e}")
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
    
    def _show_viewer(self, cap, frame_data: FrameData, frame_idx: int):
        """Show video viewer with overlays"""
        ret, frame = cap.read()
        if not ret:
            return
        
        # Draw frame info
        cv2.putText(frame, f"Frame: {frame_data.frame_number}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        
        # Draw players and classifications
        for player in frame_data.players:
            x1, y1, x2, y2 = [int(x) for x in player.bbox]
            
            # Draw bounding box
            color = (0, 255, 0) if player.player_id == 0 else (0, 0, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Draw feet position
            feet_x, feet_y = int(player.feet_position[0]), int(player.feet_position[1])
            cv2.circle(frame, (feet_x, feet_y), 5, (0, 255, 255), -1)
            cv2.putText(frame, "FEET", (feet_x + 10, feet_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            
            # Draw ball distance
            if player.ball_distance != float('inf'):
                cv2.putText(frame, f"Ball: {player.ball_distance:.1f}px", 
                           (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Draw ball
        if frame_data.ball_x > 0 and frame_data.ball_y > 0:
            cv2.circle(frame, (int(frame_data.ball_x), int(frame_data.ball_y)), 
                      8, (0, 255, 255), -1)
            cv2.putText(frame, "BALL", (int(frame_data.ball_x) + 10, int(frame_data.ball_y)), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        
        cv2.imshow('Tennis Shot ML Analysis', frame)
        cv2.waitKey(1)

def main():
    parser = argparse.ArgumentParser(description='ML-based tennis shot classification')
    parser.add_argument('--csv', required=True, help='Input CSV file path')
    parser.add_argument('--video', required=True, help='Input video file path')
    parser.add_argument('--viewer', action='store_true', help='Show video viewer')
    
    args = parser.parse_args()
    
    if not Path(args.csv).exists():
        logger.error(f"CSV file not found: {args.csv}")
        return
        
    if not Path(args.video).exists():
        logger.error(f"Video file not found: {args.video}")
        return
    
    # Process the data
    processor = TennisShotMLProcessor()
    processor.process_csv_data(args.csv, args.video, args.viewer)

if __name__ == "__main__":
    main()
