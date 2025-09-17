#!/usr/bin/env python3
"""
Tennis Shot Classification System

A comprehensive framework for classifying tennis shots from video analysis data.
Supports both motion-based and pose-based classification methods.

Usage:
    python tennis_shot.py --csv data.csv --video video.mp4 --method motion --viewer
"""

import cv2
import numpy as np
import pandas as pd
import argparse
import logging
from typing import List, Optional, Dict, Tuple, Any
from dataclasses import dataclass
from enum import Enum

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')
logger = logging.getLogger(__name__)


class ShotType(Enum):
    """Enumeration of possible tennis shot types"""
    FOREHAND = "forehand"
    BACKHAND = "backhand"
    SERVE = "serve"
    OVERHEAD_SMASH = "overhead_smash"
    READY_STANCE = "ready_stance"
    MOVING = "moving"
    UNKNOWN = "unknown"


@dataclass
class PlayerData:
    """Container for player-specific data"""
    bbox: List[int]  # [x1, y1, x2, y2]
    center: Tuple[float, float]  # (x, y)
    pose_keypoints: Optional[List[Tuple[float, float, float]]] = None  # [(x, y, confidence), ...]
    shot_type: ShotType = ShotType.UNKNOWN
    confidence: float = 0.0


@dataclass
class FrameData:
    """Container for all data in a single frame"""
    frame_number: int
    timestamp: float
    ball_position: Optional[Tuple[float, float]] = None
    players: List[PlayerData] = None
    court_keypoints: Optional[List[Tuple[float, float]]] = None
    
    def __post_init__(self):
        if self.players is None:
            self.players = []


class ShotClassifier:
    """Base class for shot classification algorithms"""
    
    def __init__(self):
        self.name = "Base Classifier"
    
    def classify_shot(self, player_data: PlayerData, frame_data: FrameData) -> Tuple[ShotType, float]:
        """
        Classify a shot for a given player
        
        Args:
            player_data: Player-specific data
            frame_data: Complete frame data including ball position, court, etc.
            
        Returns:
            Tuple of (shot_type, confidence)
        """
        raise NotImplementedError("Subclasses must implement classify_shot")


class MotionBasedClassifier(ShotClassifier):
    """Motion-based shot classification using arm movement analysis"""
    
    def __init__(self):
        super().__init__()
        self.name = "Motion-Based Classifier"
        
        # Motion analysis parameters
        self.motion_history_length = 30  # Longer window for full stroke analysis
        self.stroke_analysis_window = 20  # Analyze 20 frames (2/3 second) for stroke detection
        self.movement_threshold = 20.0   # Pixels of movement to detect motion
        self.swing_velocity_threshold = 8.0  # Minimum velocity for swing detection (lowered)
        
        # Shot persistence - once detected, maintain for full stroke duration
        self.shot_persistence_frames = 30  # Keep shot classification for 1 second
        
        # Player motion history: {player_id: deque of (right_wrist, body_center)}
        self.motion_history = {}
        self.current_shot = {}  # {player_id: (shot_type, start_frame)}
    
    def classify_shot(self, player_data: PlayerData, frame_data: FrameData) -> Tuple[ShotType, float]:
        """Classify shot based on arm movement relative to body center"""
        player_id = self._get_player_id(player_data, frame_data)
        
        # Initialize motion history for new players
        if player_id not in self.motion_history:
            from collections import deque
            self.motion_history[player_id] = deque(maxlen=self.motion_history_length)
            self.current_shot[player_id] = (None, 0)
        
        # Get right wrist and body center from pose keypoints
        right_wrist, body_center = self._get_arm_and_body_positions(player_data.pose_keypoints)
        if right_wrist is None or body_center is None:
            return ShotType.UNKNOWN, 0.0
        
        # Debug: Log first few frames
        if frame_data.frame_number < 10:
            logger.info(f"Frame {frame_data.frame_number}, Player {player_id}: wrist={right_wrist}, body={body_center}")
        
        # Update motion history with (right_wrist, body_center)
        self.motion_history[player_id].append((right_wrist, body_center))
        
        # Check for shot persistence
        if self._is_shot_persisting(player_id, frame_data.frame_number):
            shot_type, _ = self.current_shot[player_id]
            return shot_type, 0.8
        
        # Analyze motion for stroke detection over 20-frame window
        if len(self.motion_history[player_id]) >= self.stroke_analysis_window:
            stroke_type = self._analyze_stroke_sequence(player_id, frame_data.frame_number)
            if stroke_type != ShotType.UNKNOWN:
                self.current_shot[player_id] = (stroke_type, frame_data.frame_number)
                return stroke_type, 0.9
        
        # Check for movement vs ready stance
        movement_type = self._classify_movement(player_id)
        return movement_type, 0.6
    
    def _get_player_id(self, player_data: PlayerData, frame_data: FrameData) -> int:
        """Get player ID based on position in frame"""
        for i, p in enumerate(frame_data.players):
            if p == player_data:
                return i
        return 0  # Fallback
    
    def _get_arm_and_body_positions(self, pose_keypoints: Optional[List[Tuple[float, float, float]]]) -> Tuple[Optional[Tuple[float, float]], Optional[Tuple[float, float]]]:
        """Get right wrist position and body center for forehand/backhand detection"""
        if not pose_keypoints or len(pose_keypoints) < 11:
            return None, None
        
        try:
            # Pose keypoints: [nose, left_eye, right_eye, left_ear, right_ear, left_shoulder, right_shoulder, ...]
            # We need right wrist (index 10) and body center (average of shoulders)
            right_shoulder = pose_keypoints[6]  # Index 6
            left_shoulder = pose_keypoints[5]   # Index 5
            right_wrist = pose_keypoints[10]    # Index 10
            
            # Filter by confidence - only use high-confidence keypoints
            min_confidence = 0.6
            if (right_wrist[2] < min_confidence or 
                right_shoulder[2] < min_confidence or 
                left_shoulder[2] < min_confidence):
                return None, None
            
            # Calculate body center (midpoint between shoulders)
            body_center = ((left_shoulder[0] + right_shoulder[0]) / 2, (left_shoulder[1] + right_shoulder[1]) / 2)
            
            return (right_wrist[0], right_wrist[1]), body_center
        except (IndexError, TypeError):
            return None, None
    
    def _analyze_stroke_sequence(self, player_id: int, current_frame: int) -> ShotType:
        """Analyze 20-frame window to detect complete stroke sequence"""
        history = list(self.motion_history[player_id])
        if len(history) < self.stroke_analysis_window:
            return ShotType.UNKNOWN
        
        # Get the last 20 frames for analysis
        recent_window = history[-self.stroke_analysis_window:]
        
        # Check if there's a stroke sequence: movement -> swing -> movement
        stroke_phase = self._identify_stroke_phase(recent_window)
        
        if stroke_phase == "swing":
            # During swing phase, determine forehand vs backhand
            return self._classify_swing_direction(recent_window)
        else:
            return ShotType.UNKNOWN
    
    def _identify_stroke_phase(self, window: List[Tuple[Tuple[float, float], Tuple[float, float]]]) -> str:
        """Identify if we're in a swing phase of a stroke sequence"""
        if len(window) < 10:
            return "unknown"
        
        # Calculate arm velocities over the window
        velocities = []
        for i in range(1, len(window)):
            prev_wrist, _ = window[i-1]
            curr_wrist, _ = window[i]
            dx = curr_wrist[0] - prev_wrist[0]
            dy = curr_wrist[1] - prev_wrist[1]
            velocity = np.sqrt(dx*dx + dy*dy)
            velocities.append(velocity)
        
        # Look for velocity peaks that indicate swing
        if len(velocities) >= 10:
            # Check if there's a sustained high-velocity period (swing)
            high_velocity_frames = sum(1 for v in velocities if v > self.swing_velocity_threshold)
            if high_velocity_frames >= 5:  # At least 5 frames of high velocity
                return "swing"
        
        return "unknown"
    
    def _classify_swing_direction(self, window: List[Tuple[Tuple[float, float], Tuple[float, float]]]) -> ShotType:
        """Classify forehand vs backhand based on arm crossing body centerline over full window"""
        if len(window) < 10:
            return ShotType.UNKNOWN
        
        # Analyze the entire 20-frame window for consistent pattern
        left_side_count = 0
        right_side_count = 0
        
        for wrist_pos, body_center in window:
            if wrist_pos[0] < body_center[0]:  # Wrist left of body center
                left_side_count += 1
            else:  # Wrist right of body center
                right_side_count += 1
        
        # Determine swing type based on dominant side over entire window
        if left_side_count > right_side_count:
            return ShotType.BACKHAND  # Arm crossed to left side = backhand
        else:
            return ShotType.FOREHAND  # Arm stayed on right side = forehand
    
    def _classify_movement(self, player_id: int) -> ShotType:
        """Classify movement vs ready stance based on body center movement"""
        history = list(self.motion_history[player_id])
        if len(history) < 3:
            return ShotType.READY_STANCE
        
        # Calculate total movement of body center over recent frames
        total_movement = 0
        for i in range(1, min(len(history), 5)):
            _, prev_body = history[i-1]
            _, curr_body = history[i]
            dx = curr_body[0] - prev_body[0]
            dy = curr_body[1] - prev_body[1]
            total_movement += np.sqrt(dx*dx + dy*dy)
        
        if total_movement > self.movement_threshold:
            return ShotType.MOVING
        else:
            return ShotType.READY_STANCE
    
    def _is_shot_persisting(self, player_id: int, current_frame: int) -> bool:
        """Check if a shot is currently persisting"""
        if player_id not in self.current_shot:
            return False
        
        shot_type, start_frame = self.current_shot[player_id]
        if shot_type is None:
            return False
        
        return (current_frame - start_frame) < self.shot_persistence_frames


class PoseBasedClassifier(ShotClassifier):
    """Pose-based shot classification using body keypoints"""
    
    def __init__(self):
        super().__init__()
        self.name = "Pose-Based Classifier"
    
    def classify_shot(self, player_data: PlayerData, frame_data: FrameData) -> Tuple[ShotType, float]:
        """Classify shot based on pose analysis"""
        if not player_data.pose_keypoints:
            return ShotType.UNKNOWN, 0.0
        
        # Check for serve
        if self._is_serve(player_data, frame_data):
            return ShotType.SERVE, 0.9
        
        # Check for overhead smash
        if self._is_overhead_smash(player_data):
            return ShotType.OVERHEAD_SMASH, 0.8
        
        # Check for groundstrokes
        groundstroke = self._classify_groundstroke(player_data)
        if groundstroke != ShotType.UNKNOWN:
            return groundstroke, 0.7
        
        # Check for movement
        if self._is_moving(player_data):
            return ShotType.MOVING, 0.6
        
        return ShotType.READY_STANCE, 0.5
    
    def _is_serve(self, player_data: PlayerData, frame_data: FrameData) -> bool:
        """Check if player is serving"""
        # Simple heuristic: both players near baseline
        if frame_data.court_keypoints and len(frame_data.court_keypoints) >= 4:
            court_baseline_y = min(kp[1] for kp in frame_data.court_keypoints)
            player_y = player_data.center[1]
            return abs(player_y - court_baseline_y) < 100
        return False
    
    def _is_overhead_smash(self, player_data: PlayerData) -> bool:
        """Check if player is performing overhead smash"""
        if not player_data.pose_keypoints or len(player_data.pose_keypoints) < 11:
            return False
        
        # Check if both arms are raised high
        left_shoulder = player_data.pose_keypoints[5]
        right_shoulder = player_data.pose_keypoints[6]
        left_wrist = player_data.pose_keypoints[9]
        right_wrist = player_data.pose_keypoints[10]
        
        # Arms should be above shoulders
        return (left_wrist[1] < left_shoulder[1] and 
                right_wrist[1] < right_shoulder[1])
    
    def _classify_groundstroke(self, player_data: PlayerData) -> ShotType:
        """Classify forehand vs backhand"""
        if not player_data.pose_keypoints or len(player_data.pose_keypoints) < 11:
            return ShotType.UNKNOWN
        
        # Get key body points
        left_shoulder = player_data.pose_keypoints[5]
        right_shoulder = player_data.pose_keypoints[6]
        left_wrist = player_data.pose_keypoints[9]
        right_wrist = player_data.pose_keypoints[10]
        
        # Calculate body center
        body_center_x = (left_shoulder[0] + right_shoulder[0]) / 2
        
        # Check which hand is more extended
        left_extension = abs(left_wrist[0] - body_center_x)
        right_extension = abs(right_wrist[0] - body_center_x)
        
        # Simple heuristic: more extended hand indicates shot type
        if right_extension > left_extension:
            # Right hand more extended - check if it's on left side (backhand)
            if right_wrist[0] < body_center_x:
                return ShotType.BACKHAND
            else:
                return ShotType.FOREHAND
        else:
            # Left hand more extended - likely backhand
            return ShotType.BACKHAND
    
    def _is_moving(self, player_data: PlayerData) -> bool:
        """Check if player is moving (simplified)"""
        # This would need motion history in a real implementation
        return False


class TennisShotProcessor:
    """Main processor for tennis shot classification"""
    
    def __init__(self, classifier_type: str = "motion"):
        """Initialize processor with specified classifier"""
        self.classifier_type = classifier_type
        
        if classifier_type == "motion":
            self.classifier = MotionBasedClassifier()
        elif classifier_type == "pose":
            self.classifier = PoseBasedClassifier()
        else:
            raise ValueError(f"Unknown classifier type: {classifier_type}")
        
        logger.info(f"Tennis shot processor initialized with {self.classifier.name}")
    
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
            cv2.namedWindow('Tennis Shot Classification', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('Tennis Shot Classification', 1200, 800)
        
        try:
            for idx, row in df.iterrows():
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Parse frame data
                frame_data = self._parse_frame_data(row, idx)
                
                # Classify shots for each player
                for player in frame_data.players:
                    shot_type, confidence = self.classifier.classify_shot(player, frame_data)
                    player.shot_type = shot_type
                    player.confidence = confidence
                
                # Add overlays
                frame_with_overlays = self._add_overlays(frame, frame_data)
                
                # Write frame
                if out:
                    out.write(frame_with_overlays)
                
                # Show in viewer
                if show_viewer:
                    cv2.imshow('Tennis Shot Classification', frame_with_overlays)
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
        
        # Parse court keypoints
        court_keypoints = self._parse_court_keypoints(row.get('court_keypoints', ''))
        
        # Parse players
        players = []
        player_bboxes = self._parse_player_bboxes(row.get('player_bboxes', ''))
        player_poses = self._parse_player_poses(row.get('pose_keypoints', ''))
        
        for i, bbox in enumerate(player_bboxes):
            if bbox:
                center = ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)
                pose = player_poses[i] if i < len(player_poses) else None
                if frame_number < 5:  # Debug first few frames
                    logger.info(f"Frame {frame_number}, Player {i}: pose_keypoints={len(pose) if pose else 0} points")
                player = PlayerData(bbox=bbox, center=center, pose_keypoints=pose)
                players.append(player)
        
        return FrameData(
            frame_number=frame_number,
            timestamp=row.get('timestamp', 0.0),
            ball_position=ball_position,
            players=players,
            court_keypoints=court_keypoints
        )
    
    def _parse_float(self, value: str) -> Optional[float]:
        """Parse float value from CSV string"""
        try:
            if pd.isna(value) or value == '':
                return None
            return float(value)
        except (ValueError, TypeError):
            return None
    
    def _parse_court_keypoints(self, court_str: str) -> Optional[List[Tuple[float, float]]]:
        """Parse court keypoints from CSV string"""
        if not court_str or pd.isna(court_str):
            return None
        
        try:
            # Parse format: "x1,y1;x2,y2;..."
            points = []
            for point_str in court_str.split(';'):
                if point_str.strip():
                    x, y = map(float, point_str.split(','))
                    points.append((x, y))
            return points if points else None
        except (ValueError, AttributeError):
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
        cv2.putText(frame, f"Frame: {frame_data.frame_number}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(frame, f"Method: {self.classifier.name}", (10, 70), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Add player overlays
        for i, player in enumerate(frame_data.players):
            if player.bbox:
                # Draw bounding box
                x1, y1, x2, y2 = player.bbox
                color = (0, 255, 0) if i == 0 else (255, 0, 0)  # Green for player 0, red for player 1
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                
                # Draw shot type
                shot_text = f"P{i}: {player.shot_type.value} ({player.confidence:.2f})"
                cv2.putText(frame, shot_text, (x1, y1 - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # Add ball position
        if frame_data.ball_position:
            bx, by = map(int, frame_data.ball_position)
            cv2.circle(frame, (bx, by), 8, (0, 255, 255), -1)
            cv2.putText(frame, "Ball", (bx + 10, by), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        
        return frame


def main():
    """Main function for standalone usage"""
    parser = argparse.ArgumentParser(description='Tennis Shot Classification System')
    parser.add_argument('--csv', required=True, help='Input CSV file with analysis data')
    parser.add_argument('--video', required=True, help='Input video file')
    parser.add_argument('--output', help='Output video file (optional)')
    parser.add_argument('--method', choices=['motion', 'pose'], default='motion',
                       help='Classification method: motion-based or pose-based')
    parser.add_argument('--viewer', action='store_true', 
                       help='Show real-time viewer')
    
    args = parser.parse_args()
    
    # Create processor and run analysis
    processor = TennisShotProcessor(classifier_type=args.method)
    processor.process_csv_data(args.csv, args.video, args.output, show_viewer=args.viewer)


if __name__ == "__main__":
    main()
