#!/usr/bin/env python3
"""
Tennis Shot Classification System v2

A ground-up rebuild focusing on accurate movement detection using player feet position.
Starting with ready stance vs. moving classification.

Usage:
    python tennis_shot2.py --csv data.csv --video video.mp4 --viewer
"""

import cv2
import numpy as np
import pandas as pd
import argparse
import logging
from typing import List, Optional, Dict, Tuple, Any
from dataclasses import dataclass
from enum import Enum
from collections import deque

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')
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
    """Container for all data in a single frame"""
    frame_number: int
    timestamp: float
    ball_position: Optional[Tuple[float, float]] = None
    players: List[PlayerData] = None
    
    def __post_init__(self):
        if self.players is None:
            self.players = []


class ShotClassifier:
    """ML-based classifier for forehand/backhand detection using learned patterns"""
    
    def __init__(self):
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
        
        logger.info(f"ML-based shot classifier initialized with relative measurements:")
        logger.info(f"  - Ball proximity: {self.ball_proximity_ratio:.1f} player widths")
        logger.info(f"  - Ball distance: {self.ball_distance_ratio:.1f} player widths") 
        logger.info(f"  - Arm extension: {self.arm_extension_ratio:.1f} player widths")
    
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
        shot_type, confidence = self._detect_shot_ml_based(player_data, player_id)
        
        if shot_type != ShotType.UNKNOWN and confidence > 0.6:
            # Start tracking this shot
            self.current_shot_state[player_id] = (shot_type, self.shot_identification_frames)
            logger.info(f"Frame {frame_data.frame_number}, Player {player_id}: Started {shot_type.value} "
                       f"(ball distance: {ball_distance_ratio:.2f} player widths, confidence: {confidence:.2f})")
        
        return shot_type, confidence
    
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
    
    def _detect_shot_ml_based(self, player_data: PlayerData, player_id: int) -> Tuple[ShotType, float]:
        """Detect shot using ML-based features"""
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
            logger.warning(f"Error in ML-based shot detection: {e}")
            return ShotType.UNKNOWN, 0.0
    
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
    
    


class MovementClassifier:
    """Classifier for distinguishing between ready stance and moving"""
    
    def __init__(self):
        self.name = "Movement Classifier"
        
        # Movement detection parameters
        self.movement_history_length = 15  # Frames to track for movement analysis
        self.movement_threshold = 40.0     # Pixels of movement to classify as "moving" (increased)
        self.ready_stance_threshold = 20.0 # Pixels below which is definitely ready stance
        self.min_frames_for_analysis = 5   # Minimum frames needed for reliable analysis
        
        # Player movement history: {player_id: deque of feet_positions}
        self.movement_history = {}
        
        logger.info(f"Movement classifier initialized - Moving: >{self.movement_threshold}px, Ready: <{self.ready_stance_threshold}px")
    
    def classify_movement(self, player_data: PlayerData, frame_data: FrameData) -> Tuple[MovementType, float]:
        """
        Classify player movement based on feet position over time
        
        Args:
            player_data: Player-specific data with feet position
            frame_data: Complete frame data
            
        Returns:
            Tuple of (movement_type, confidence)
        """
        player_id = self._get_player_id(player_data, frame_data)
        
        # Initialize movement history for new players
        if player_id not in self.movement_history:
            self.movement_history[player_id] = deque(maxlen=self.movement_history_length)
        
        # Add current feet position to history
        self.movement_history[player_id].append(player_data.feet_position)
        
        # Need minimum frames for reliable analysis
        if len(self.movement_history[player_id]) < self.min_frames_for_analysis:
            return MovementType.UNKNOWN, 0.0
        
        # Calculate total movement over recent frames
        total_movement = self._calculate_total_movement(player_id)
        
        # Classify based on movement with buffer zones
        if total_movement > self.movement_threshold:
            # Clear movement - high confidence
            confidence = min(0.9, 0.7 + (total_movement - self.movement_threshold) / 40.0)
            return MovementType.MOVING, confidence
        elif total_movement < self.ready_stance_threshold:
            # Clear ready stance - high confidence
            confidence = min(0.9, 0.7 + (self.ready_stance_threshold - total_movement) / 20.0)
            return MovementType.READY_STANCE, confidence
        else:
            # Buffer zone - moderate movement, default to ready stance with lower confidence
            # This handles small movements like weight shifting, slight adjustments
            confidence = 0.6
            return MovementType.READY_STANCE, confidence
    
    def _get_player_id(self, player_data: PlayerData, frame_data: FrameData) -> int:
        """Get player ID based on position in frame"""
        for i, p in enumerate(frame_data.players):
            if p == player_data:
                return i
        return 0  # Fallback
    
    def _calculate_total_movement(self, player_id: int) -> float:
        """Calculate total movement of feet over recent frames"""
        history = list(self.movement_history[player_id])
        if len(history) < 2:
            return 0.0
        
        total_movement = 0.0
        
        # Calculate movement between consecutive frames
        for i in range(1, len(history)):
            prev_feet = history[i-1]
            curr_feet = history[i]
            
            dx = curr_feet[0] - prev_feet[0]
            dy = curr_feet[1] - prev_feet[1]
            distance = np.sqrt(dx*dx + dy*dy)
            total_movement += distance
        
        return total_movement


class TennisShotProcessor:
    """Main processor for tennis shot classification v2"""
    
    def __init__(self):
        """Initialize processor with both classifiers"""
        self.movement_classifier = MovementClassifier()
        self.shot_classifier = ShotClassifier()
        logger.info("Tennis shot processor v2 initialized with movement and shot classifiers")
    
    def process_csv_data(self, csv_file: str, video_file: str, output_file: str = None, show_viewer: bool = False):
        """Process CSV data and create analysis video"""
        # Load data
        df = pd.read_csv(csv_file)
        cap = cv2.VideoCapture(video_file)
        
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_file}")
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        logger.info(f"Processing {len(df)} frames from {video_file}")
        logger.info(f"Video: {width}x{height} @ {fps}fps")
        
        # Setup video writer if output specified
        out = None
        if output_file:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_file, fourcc, fps, (width, height))
        
        # Setup viewer
        if show_viewer:
            cv2.namedWindow('Tennis Shot Classification v2', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('Tennis Shot Classification v2', 1200, 800)
        
        try:
            for idx, row in df.iterrows():
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Parse frame data
                frame_data = self._parse_frame_data(row, idx)
                
                # Classify movement and shots for each player
                for player in frame_data.players:
                    # First classify movement
                    movement_type, movement_conf = self.movement_classifier.classify_movement(player, frame_data)
                    player.movement_type = movement_type
                    
                    # Then classify shot type
                    shot_type, shot_conf = self.shot_classifier.classify_shot(player, frame_data)
                    
                    # If shot classifier returns UNKNOWN, fall back to movement classification
                    if shot_type == ShotType.UNKNOWN:
                        # Use movement classification as the primary classification
                        player.shot_type = ShotType.UNKNOWN  # Keep as unknown for shot type
                        player.confidence = movement_conf
                    else:
                        # Shot was detected - use shot classification
                        player.shot_type = shot_type
                        player.confidence = shot_conf
                
                # Add overlays
                frame_with_overlays = self._add_overlays(frame, frame_data)
                
                # Write frame
                if out:
                    out.write(frame_with_overlays)
                
                # Show in viewer
                if show_viewer:
                    cv2.imshow('Tennis Shot Classification v2', frame_with_overlays)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        break
                    elif key == ord(' '):  # Space to pause
                        cv2.waitKey(0)
                
                # Progress update
                if idx % 30 == 0:
                    logger.info(f"Processed {idx}/{len(df)} frames ({idx/len(df)*100:.1f}%)")
        
        finally:
            # Cleanup
            cap.release()
            if out:
                out.release()
            if show_viewer:
                cv2.destroyAllWindows()
        
        logger.info("Processing completed!")
    
    def _parse_frame_data(self, row: pd.Series, frame_number: int) -> FrameData:
        """Parse CSV row into FrameData object"""
        # Parse ball position
        ball_x = self._parse_float(row.get('ball_x', ''))
        ball_y = self._parse_float(row.get('ball_y', ''))
        ball_position = (ball_x, ball_y) if ball_x is not None and ball_y is not None else None
        
        # Parse players
        players = []
        player_bboxes = self._parse_player_bboxes(row.get('player_bboxes', ''))
        player_poses = self._parse_player_poses(row.get('pose_keypoints', ''))
        
        for i, bbox in enumerate(player_bboxes):
            if bbox:
                center = ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)
                
                # Calculate feet position (bottom 10% of bounding box)
                feet_y = bbox[3] - (bbox[3] - bbox[1]) * 0.1  # Bottom 10%
                feet_position = (center[0], feet_y)  # Use center X, bottom 10% Y
                
                # Get pose data for this player
                pose = player_poses[i] if i < len(player_poses) else None
                
                # Log pose data for debugging
                if frame_number < 5:  # First few frames
                    logger.info(f"Frame {frame_number}, Player {i}: pose_keypoints={len(pose) if pose else 0} points")
                    if pose and len(pose) >= 11:
                        logger.info(f"  Keypoint 8 (right_elbow): {pose[8] if len(pose) > 8 else 'N/A'}")
                        logger.info(f"  Keypoint 10 (right_wrist): {pose[10] if len(pose) > 10 else 'N/A'}")
                        logger.info(f"  Keypoint 5 (left_shoulder): {pose[5] if len(pose) > 5 else 'N/A'}")
                        logger.info(f"  Keypoint 6 (right_shoulder): {pose[6] if len(pose) > 6 else 'N/A'}")
                
                player = PlayerData(
                    bbox=bbox, 
                    center=center, 
                    feet_position=feet_position,
                    pose_keypoints=pose
                )
                players.append(player)
        
        return FrameData(
            frame_number=frame_number,
            timestamp=row.get('timestamp', 0.0),
            ball_position=ball_position,
            players=players
        )
    
    def _parse_float(self, value: str) -> Optional[float]:
        """Parse float value from CSV string"""
        try:
            if pd.isna(value) or value == '':
                return None
            return float(value)
        except (ValueError, TypeError):
            return None
    
    def _parse_player_bboxes(self, bbox_str: str) -> List[Optional[List[int]]]:
        """Parse player bounding boxes from CSV string"""
        if not bbox_str or pd.isna(bbox_str):
            return []
        
        try:
            bboxes = []
            for bbox_str_item in bbox_str.split(';'):
                if bbox_str_item.strip():
                    x1, y1, x2, y2 = map(int, bbox_str_item.split(','))
                    bboxes.append([x1, y1, x2, y2])
                else:
                    bboxes.append(None)
            return bboxes
        except (ValueError, AttributeError):
            return []
    
    def _parse_player_poses(self, pose_str: str) -> List[Optional[List[Tuple[float, float, float]]]]:
        """Parse player poses from CSV string"""
        if not pose_str or pd.isna(pose_str):
            return []
        
        try:
            poses = []
            # Split by semicolon to get each player's pose data
            for pose_str_item in pose_str.split(';'):
                if pose_str_item.strip():
                    # Parse format: "x1,y1,conf1|x2,y2,conf2|..."
                    keypoints = []
                    for kp_str in pose_str_item.split('|'):
                        if kp_str.strip():
                            x, y, conf = map(float, kp_str.split(','))
                            keypoints.append((x, y, conf))
                    poses.append(keypoints if keypoints else None)
                else:
                    poses.append(None)
            return poses
        except (ValueError, AttributeError) as e:
            logger.warning(f"Error parsing poses: {e}")
            return []
    
    def _add_overlays(self, frame: np.ndarray, frame_data: FrameData) -> np.ndarray:
        """Add classification overlays to frame"""
        frame = frame.copy()
        
        # Add frame info
        # Draw frame info with larger, more visible counter
        cv2.putText(frame, f"Frame: {frame_data.frame_number}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 4)  # Black text with thick outline
        cv2.putText(frame, f"Frame: {frame_data.frame_number}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2)  # White text
        cv2.putText(frame, f"Movement: {self.movement_classifier.name}", (10, 70), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, f"Shot: {self.shot_classifier.name}", (10, 100), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Add player overlays
        for i, player in enumerate(frame_data.players):
            if player.bbox:
                # Draw bounding box
                x1, y1, x2, y2 = player.bbox
                color = (0, 255, 0) if i == 0 else (255, 0, 0)  # Green for player 0, red for player 1
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                
                # Draw feet position (bottom 10% marker)
                feet_x, feet_y = int(player.feet_position[0]), int(player.feet_position[1])
                cv2.circle(frame, (feet_x, feet_y), 5, (255, 255, 0), -1)  # Yellow circle for feet
                cv2.putText(frame, "FEET", (feet_x + 10, feet_y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
                
                # Draw primary classification (shot if detected, movement if not)
                if player.shot_type != ShotType.UNKNOWN:
                    # Shot detected - show shot classification
                    shot_text = f"P{i}: {player.shot_type.value} ({player.confidence:.2f})"
                    if player.shot_type == ShotType.FOREHAND:
                        text_color = (0, 255, 0)  # Green for forehand
                    elif player.shot_type == ShotType.BACKHAND:
                        text_color = (0, 0, 255)  # Red for backhand
                    elif player.shot_type == ShotType.OVERHEAD_SMASH:
                        text_color = (255, 0, 255)  # Magenta for overhead smash
                    elif player.shot_type == ShotType.SERVE:
                        text_color = (255, 165, 0)  # Orange for serve
                    else:
                        text_color = (255, 255, 255)  # White for other shots
                    cv2.putText(frame, shot_text, (x1, y1 - 10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2)
                    
                    # Show movement as secondary
                    movement_text = f"Movement: {player.movement_type.value}"
                    movement_color = (0, 255, 255) if player.movement_type == MovementType.MOVING else (255, 255, 0)
                    cv2.putText(frame, movement_text, (x1, y1 - 35), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, movement_color, 1)
                else:
                    # No shot detected - show movement as primary
                    movement_text = f"P{i}: {player.movement_type.value} ({player.confidence:.2f})"
                    movement_color = (0, 255, 255) if player.movement_type == MovementType.MOVING else (255, 255, 0)
                    cv2.putText(frame, movement_text, (x1, y1 - 10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, movement_color, 2)
                
                # Draw movement history trail (for debugging)
                if hasattr(self.movement_classifier, 'movement_history'):
                    history = self.movement_classifier.movement_history.get(i, deque())
                    if len(history) > 1:
                        points = [(int(pos[0]), int(pos[1])) for pos in history]
                        for j in range(1, len(points)):
                            alpha = j / len(points)  # Fade from old to new
                            trail_color = (int(255 * alpha), int(255 * alpha), 0)
                            cv2.line(frame, points[j-1], points[j], trail_color, 2)
        
        # Add ball position
        if frame_data.ball_position:
            bx, by = map(int, frame_data.ball_position)
            cv2.circle(frame, (bx, by), 8, (0, 255, 255), -1)
            cv2.putText(frame, "Ball", (bx + 10, by), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        
        return frame


def main():
    """Main function for standalone usage"""
    parser = argparse.ArgumentParser(description='Tennis Shot Classification System v2')
    parser.add_argument('--csv', required=True, help='Input CSV file with analysis data')
    parser.add_argument('--video', required=True, help='Input video file')
    parser.add_argument('--output', help='Output video file (optional)')
    parser.add_argument('--viewer', action='store_true', 
                       help='Show real-time viewer')
    parser.add_argument('--threshold', type=float, default=40.0,
                       help='Movement threshold in pixels (default: 40.0)')
    parser.add_argument('--ready-threshold', type=float, default=20.0,
                       help='Ready stance threshold in pixels (default: 20.0)')
    
    args = parser.parse_args()
    
    # Create processor and run analysis
    processor = TennisShotProcessor()
    
    # Update thresholds if specified
    if args.threshold != 40.0:
        processor.movement_classifier.movement_threshold = args.threshold
        logger.info(f"Updated movement threshold to {args.threshold}px")
    
    if args.ready_threshold != 20.0:
        processor.movement_classifier.ready_stance_threshold = args.ready_threshold
        logger.info(f"Updated ready stance threshold to {args.ready_threshold}px")
    
    processor.process_csv_data(args.csv, args.video, args.output, show_viewer=args.viewer)


if __name__ == "__main__":
    main()
