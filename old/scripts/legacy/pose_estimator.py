"""
Pose Estimation Module using YOLOv8-pose
Analyzes player body positions and swing mechanics
"""

import cv2
import numpy as np
from ultralytics import YOLO
from typing import List, Tuple, Dict, Any, Optional
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class Keypoint:
    """Represents a single keypoint with position and confidence"""
    x: float
    y: float
    confidence: float
    visible: bool

@dataclass
class Pose:
    """Represents a complete pose with all keypoints"""
    keypoints: List[Keypoint]
    bbox: List[int]  # [x1, y1, x2, y2]
    confidence: float
    player_id: Optional[int] = None

class PoseEstimator:
    """YOLOv8-pose-based pose estimation for tennis swing analysis"""
    
    # COCO keypoint names for tennis analysis
    KEYPOINT_NAMES = [
        'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
        'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
        'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
        'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
    ]
    
    # Key keypoints for tennis swing analysis
    SWING_KEYPOINTS = ['left_wrist', 'right_wrist', 'left_elbow', 'right_elbow', 
                       'left_shoulder', 'right_shoulder', 'left_hip', 'right_hip']
    
    def __init__(self, model_path: str, config: Dict[str, Any]):
        """
        Initialize pose estimator
        
        Args:
            model_path: Path to YOLOv11-pose model weights
            config: Configuration dictionary
        """
        self.config = config
        self.model = YOLO(model_path)
        self.conf_threshold = config.get('conf_threshold', 0.3)
        self.iou_threshold = config.get('iou_threshold', 0.45)
        self.max_det = config.get('max_det', 4)
        self.keypoints = config.get('keypoints', 17)
        
        logger.info(f"Pose estimator initialized with model: {model_path}")
    
    def estimate_poses(self, frame: np.ndarray, player_detections: List[Dict[str, Any]]) -> List[Pose]:
        """
        Estimate poses for detected players using their bounding boxes
        
        Args:
            frame: Input frame
            player_detections: List of player detections from player detector
            
        Returns:
            List of estimated poses
        """
        poses = []
        
        try:
            for i, player_detection in enumerate(player_detections):
                # Get player bounding box
                x1, y1, x2, y2 = player_detection['bbox']
                
                # Extract player region with some padding
                padding = 20
                x1_pad = max(0, x1 - padding)
                y1_pad = max(0, y1 - padding)
                x2_pad = min(frame.shape[1], x2 + padding)
                y2_pad = min(frame.shape[0], y2 + padding)
                
                player_roi = frame[y1_pad:y2_pad, x1_pad:x2_pad]
                
                if player_roi.size == 0:
                    continue
                
                # Run pose estimation on the player ROI
                results = self.model(
                    player_roi,
                    conf=self.conf_threshold,
                    iou=self.iou_threshold,
                    max_det=1,  # Only expect 1 person in the ROI
                    verbose=False
                )
                
                for result in results:
                    if result.keypoints is not None and len(result.keypoints.data) > 0:
                        keypoints = result.keypoints.data[0].cpu().numpy()
                        
                        # Convert keypoints back to original frame coordinates
                        pose_keypoints = []
                        for j, kp in enumerate(keypoints):
                            if j < len(self.KEYPOINT_NAMES):
                                # Adjust coordinates back to original frame
                                x_orig = kp[0] + x1_pad
                                y_orig = kp[1] + y1_pad
                                
                                keypoint = Keypoint(
                                    x=float(x_orig),
                                    y=float(y_orig),
                                    confidence=float(kp[2]),
                                    visible=float(kp[2]) > 0.1
                                )
                                pose_keypoints.append(keypoint)
                        
                        # Create pose object with original bounding box
                        pose = Pose(
                            keypoints=pose_keypoints,
                            bbox=[x1, y1, x2, y2],
                            confidence=player_detection['confidence'],
                            player_id=i
                        )
                        poses.append(pose)
            
            logger.debug(f"Estimated poses for {len(poses)} players")
            return poses
            
        except Exception as e:
            logger.error(f"Error in pose estimation: {e}")
            return []
    
    def analyze_swing_mechanics(self, pose: Pose) -> Dict[str, Any]:
        """
        Analyze tennis swing mechanics from pose data
        
        Args:
            pose: Estimated pose
            
        Returns:
            Dictionary with swing analysis results
        """
        analysis = {}
        
        try:
            # Get key keypoints for swing analysis
            wrist_left = self._get_keypoint(pose, 'left_wrist')
            wrist_right = self._get_keypoint(pose, 'right_wrist')
            elbow_left = self._get_keypoint(pose, 'left_elbow')
            elbow_right = self._get_keypoint(pose, 'right_elbow')
            shoulder_left = self._get_keypoint(pose, 'left_shoulder')
            shoulder_right = self._get_keypoint(pose, 'right_shoulder')
            
            if all([wrist_left, wrist_right, elbow_left, elbow_right, shoulder_left, shoulder_right]):
                # Calculate arm angles
                left_arm_angle = self._calculate_arm_angle(
                    shoulder_left, elbow_left, wrist_left
                )
                right_arm_angle = self._calculate_arm_angle(
                    shoulder_right, elbow_right, wrist_right
                )
                
                # Determine dominant hand (assuming right-handed for now)
                dominant_arm_angle = right_arm_angle
                non_dominant_arm_angle = left_arm_angle
                
                # Analyze swing phase
                swing_phase = self._classify_swing_phase(dominant_arm_angle)
                
                analysis = {
                    'left_arm_angle': left_arm_angle,
                    'right_arm_angle': right_arm_angle,
                    'dominant_arm_angle': dominant_arm_angle,
                    'swing_phase': swing_phase,
                    'arm_extension': self._calculate_arm_extension(pose),
                    'stance_width': self._calculate_stance_width(pose),
                    'posture_angle': self._calculate_posture_angle(pose)
                }
        
        except Exception as e:
            logger.error(f"Error in swing analysis: {e}")
        
        return analysis
    
    def _get_keypoint(self, pose: Pose, keypoint_name: str) -> Optional[Keypoint]:
        """Get specific keypoint by name"""
        try:
            idx = self.KEYPOINT_NAMES.index(keypoint_name)
            if idx < len(pose.keypoints):
                return pose.keypoints[idx]
        except (ValueError, IndexError):
            pass
        return None
    
    def _calculate_arm_angle(self, shoulder: Keypoint, elbow: Keypoint, wrist: Keypoint) -> float:
        """Calculate angle between three keypoints (shoulder-elbow-wrist)"""
        try:
            # Vector from shoulder to elbow
            v1 = np.array([elbow.x - shoulder.x, elbow.y - shoulder.y])
            # Vector from elbow to wrist
            v2 = np.array([wrist.x - elbow.x, wrist.y - elbow.y])
            
            # Calculate angle
            cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
            cos_angle = np.clip(cos_angle, -1.0, 1.0)
            angle = np.arccos(cos_angle)
            
            return np.degrees(angle)
        except:
            return 0.0
    
    def _classify_swing_phase(self, arm_angle: float) -> str:
        """Classify the current swing phase based on arm angle"""
        if arm_angle < 45:
            return "backswing"
        elif arm_angle < 90:
            return "contact"
        elif arm_angle < 135:
            return "follow_through"
        else:
            return "ready_position"
    
    def _calculate_arm_extension(self, pose: Pose) -> float:
        """Calculate how extended the arms are"""
        try:
            left_shoulder = self._get_keypoint(pose, 'left_shoulder')
            right_shoulder = self._get_keypoint(pose, 'right_shoulder')
            left_wrist = self._get_keypoint(pose, 'left_wrist')
            right_wrist = self._get_keypoint(pose, 'right_wrist')
            
            if all([left_shoulder, right_shoulder, left_wrist, right_wrist]):
                # Calculate distance from shoulders to wrists
                left_dist = np.sqrt((left_wrist.x - left_shoulder.x)**2 + 
                                  (left_wrist.y - left_shoulder.y)**2)
                right_dist = np.sqrt((right_wrist.x - right_shoulder.x)**2 + 
                                   (right_wrist.y - right_shoulder.y)**2)
                
                return (left_dist + right_dist) / 2
        except:
            pass
        return 0.0
    
    def _calculate_stance_width(self, pose: Pose) -> float:
        """Calculate the width of the player's stance"""
        try:
            left_hip = self._get_keypoint(pose, 'left_hip')
            right_hip = self._get_keypoint(pose, 'right_hip')
            
            if left_hip and right_hip:
                return abs(right_hip.x - left_hip.x)
        except:
            pass
        return 0.0
    
    def _calculate_posture_angle(self, pose: Pose) -> float:
        """Calculate the player's posture angle (how upright they are)"""
        try:
            left_shoulder = self._get_keypoint(pose, 'left_shoulder')
            left_hip = self._get_keypoint(pose, 'left_hip')
            
            if left_shoulder and left_hip:
                # Calculate angle from vertical
                dx = left_hip.x - left_shoulder.x
                dy = left_hip.y - left_shoulder.y
                angle = np.arctan2(dx, dy)
                return np.degrees(angle)
        except:
            pass
        return 0.0
    
    def draw_poses(self, frame: np.ndarray, poses: List[Pose]) -> np.ndarray:
        """
        Draw pose keypoints and connections on frame
        
        Args:
            frame: Input frame
            poses: List of estimated poses
            
        Returns:
            Frame with drawn poses
        """
        frame_copy = frame.copy()
        
        # Define connections between keypoints (skeleton)
        connections = [
            (0, 1), (0, 2), (1, 3), (2, 4),  # Head
            (5, 6), (5, 7), (6, 8), (7, 9), (8, 10), (9, 11),  # Arms
            (5, 11), (6, 12), (11, 12), (11, 13), (12, 14), (13, 15), (14, 16)  # Body and legs
        ]
        
        for pose in poses:
            # Draw keypoints
            for i, keypoint in enumerate(pose.keypoints):
                if keypoint.visible:
                    cv2.circle(frame_copy, (int(keypoint.x), int(keypoint.y)), 
                              3, (0, 255, 255), -1)
            
            # Draw connections
            for connection in connections:
                if (connection[0] < len(pose.keypoints) and 
                    connection[1] < len(pose.keypoints)):
                    kp1 = pose.keypoints[connection[0]]
                    kp2 = pose.keypoints[connection[1]]
                    
                    if kp1.visible and kp2.visible:
                        cv2.line(frame_copy, 
                                (int(kp1.x), int(kp1.y)), 
                                (int(kp2.x), int(kp2.y)), 
                                (255, 0, 255), 2)
            
            # Draw bounding box
            x1, y1, x2, y2 = pose.bbox
            cv2.rectangle(frame_copy, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw confidence
            cv2.putText(frame_copy, f"Pose: {pose.confidence:.2f}", 
                       (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        return frame_copy
