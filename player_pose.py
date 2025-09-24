#!/usr/bin/env python3
"""
Tennis Player Pose Detection System
Extracts pose keypoints from tennis videos using YOLO pose estimation
"""

import cv2
import numpy as np
import pandas as pd
import argparse
import logging
from typing import List, Tuple, Optional, Dict, Any
import os
import sys
import yaml
from collections import deque

# Import YOLO for pose detection
try:
    from ultralytics import YOLO
except ImportError:
    print("Warning: ultralytics not found. Pose detection will be disabled.")
    YOLO = None

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')
logger = logging.getLogger(__name__)


class PoseEstimator:
    """YOLO-based pose estimation for tennis players"""
    
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
        """Initialize pose estimator"""
        self.config = config
        self.model = YOLO(model_path) if YOLO else None
        self.conf_threshold = config.get('conf_threshold', 0.3)
        self.iou_threshold = config.get('iou_threshold', 0.45)
        self.max_det = config.get('max_det', 4)
        self.keypoints = config.get('keypoints', 17)
        
        # Temporal smoothing parameters
        self.pose_history = {}  # player_id -> deque of recent poses
        self.history_length = 5  # Number of frames to remember
        self.max_movement_threshold = 100  # Max pixels a keypoint can move between frames
        self.temporal_weight = 0.7  # How much to weight temporal consistency vs current detection
        
        logger.info(f"Pose estimator initialized with model: {model_path}")
    
    def estimate_poses(self, frame: np.ndarray, player_detections: List[Dict]) -> List[Dict]:
        """Estimate poses for detected players"""
        if not self.model or not player_detections:
            return []
        
        poses = []
        
        try:
            # Process each player individually
            for player_idx, player in enumerate(player_detections):
                if 'bbox' not in player:
                    continue
                
                x1, y1, x2, y2 = player['bbox']
                
                # Extract player region with padding for limb extension
                padding = 60
                x1_pad = max(0, x1 - padding)
                y1_pad = max(0, y1 - padding)
                x2_pad = min(frame.shape[1], x2 + padding)
                y2_pad = min(frame.shape[0], y2 + padding)
                
                player_roi = frame[y1_pad:y2_pad, x1_pad:x2_pad]
                
                if player_roi.size == 0:
                    continue
                
                # Run pose detection on player ROI
                results = self.model(player_roi, verbose=False, max_det=1)
                
                for result in results:
                    if result.keypoints is not None:
                        keypoints = result.keypoints.data[0].cpu().numpy()
                        confidence = result.keypoints.conf[0].cpu().numpy() if result.keypoints.conf is not None else np.ones(len(keypoints))
                        
                        # Adjust keypoints back to full frame coordinates
                        keypoints[:, 0] += x1_pad  # Add x offset
                        keypoints[:, 1] += y1_pad  # Add y offset
                        
                        # Apply temporal smoothing
                        smoothed_keypoints, smoothed_confidence = self._apply_temporal_smoothing(
                            player_idx, keypoints, confidence
                        )
                        
                        # Create pose data
                        pose_data = {
                            'player_id': player_idx,
                            'keypoints': smoothed_keypoints.tolist(),
                            'confidence': smoothed_confidence.tolist(),
                            'bbox': [x1, y1, x2, y2],
                            'frame_keypoints': self._format_keypoints_for_csv(smoothed_keypoints, smoothed_confidence)
                        }
                        
                        poses.append(pose_data)
            
            return poses
            
        except Exception as e:
            logger.warning(f"Error in pose estimation: {e}")
            return []
    
    def _apply_temporal_smoothing(self, player_id: int, keypoints: np.ndarray, confidence: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Apply temporal smoothing to keypoints"""
        if player_id not in self.pose_history:
            self.pose_history[player_id] = deque(maxlen=self.history_length)
        
        # Add current pose to history
        self.pose_history[player_id].append({
            'keypoints': keypoints.copy(),
            'confidence': confidence.copy()
        })
        
        # If we don't have enough history, return current values
        if len(self.pose_history[player_id]) < 2:
            return keypoints, confidence
        
        # Calculate temporal consistency scores
        temporal_scores = self._calculate_temporal_consistency(player_id, keypoints)
        
        # Combine current detection with temporal consistency
        final_confidence = (
            confidence * (1 - self.temporal_weight) +
            temporal_scores * self.temporal_weight
        )
        
        # Simple temporal smoothing - average with recent poses
        if len(self.pose_history[player_id]) >= 3:
            recent_keypoints = np.array([pose['keypoints'] for pose in self.pose_history[player_id]])
            smoothed_keypoints = np.mean(recent_keypoints, axis=0)
        else:
            smoothed_keypoints = keypoints
        
        return smoothed_keypoints, final_confidence
    
    def _calculate_temporal_consistency(self, player_id: int, current_keypoints: np.ndarray) -> np.ndarray:
        """Calculate temporal consistency scores for keypoints"""
        if player_id not in self.pose_history or len(self.pose_history[player_id]) < 2:
            return np.ones(len(current_keypoints))
        
        # Get previous pose
        previous_pose = self.pose_history[player_id][-1]
        previous_keypoints = previous_pose['keypoints']
        
        # Calculate movement for each keypoint
        movement = np.sqrt(np.sum((current_keypoints - previous_keypoints) ** 2, axis=1))
        
        # Convert movement to consistency score (less movement = higher consistency)
        consistency_scores = np.exp(-movement / self.max_movement_threshold)
        
        return consistency_scores
    
    def _format_keypoints_for_csv(self, keypoints: np.ndarray, confidence: np.ndarray) -> str:
        """Format keypoints for CSV storage"""
        keypoint_strings = []
        for i, (kp, conf) in enumerate(zip(keypoints, confidence)):
            x, y = kp[0], kp[1]
            keypoint_strings.append(f"{x:.1f},{y:.1f},{conf:.3f}")
        
        return "|".join(keypoint_strings)
    
    def draw_poses(self, frame: np.ndarray, poses: List[Dict]) -> np.ndarray:
        """Draw poses on frame"""
        for pose in poses:
            keypoints = np.array(pose['keypoints'])
            confidence = np.array(pose['confidence'])
            
            # Draw keypoints
            for i, (kp, conf) in enumerate(zip(keypoints, confidence)):
                if conf > 0.3:  # Only draw high confidence keypoints
                    x, y = int(kp[0]), int(kp[1])
                    cv2.circle(frame, (x, y), 3, (0, 255, 0), -1)
                    cv2.putText(frame, str(i), (x+5, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
            
            # Draw skeleton connections
            self._draw_skeleton(frame, keypoints, confidence)
        
        return frame
    
    def _draw_skeleton(self, frame: np.ndarray, keypoints: np.ndarray, confidence: np.ndarray):
        """Draw skeleton connections between keypoints"""
        # Define skeleton connections (COCO format)
        skeleton = [
            (0, 1), (0, 2), (1, 3), (2, 4),  # Head
            (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),  # Arms
            (5, 11), (6, 12), (11, 12),  # Torso
            (11, 13), (13, 15), (12, 14), (14, 16)  # Legs
        ]
        
        for connection in skeleton:
            i, j = connection
            if (i < len(keypoints) and j < len(keypoints) and 
                confidence[i] > 0.3 and confidence[j] > 0.3):
                pt1 = (int(keypoints[i][0]), int(keypoints[i][1]))
                pt2 = (int(keypoints[j][0]), int(keypoints[j][1]))
                cv2.line(frame, pt1, pt2, (0, 255, 255), 2)


class PlayerPoseProcessor:
    """Processes video for pose detection and generates CSV output"""
    
    def __init__(self):
        """Initialize processor"""
        self.config = self._load_config()
        self.pose_estimator = None
        self.yolo_model = None
        
        # Initialize models
        self._initialize_models()
        
        # Pose data storage
        self.pose_data = []
    
    def _load_config(self) -> dict:
        """Load configuration from config.yaml"""
        config_path = 'config.yaml'
        if os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    config = yaml.safe_load(f)
                logger.info(f"Loaded configuration from {config_path}")
                return config
            except Exception as e:
                logger.warning(f"Could not load config file: {e}")
        
        # Return default configuration
        return {
            'models': {
                'yolo_pose': 'yolov8n-pose.pt',
                'yolo_player': 'yolov8n.pt'
            },
            'yolo_pose': {
                'conf_threshold': 0.3,
                'iou_threshold': 0.45,
                'max_det': 4,
                'keypoints': 17
            },
            'yolo_player': {
                'conf_threshold': 0.5,
                'iou_threshold': 0.45,
                'max_det': 10
            }
        }
    
    def _initialize_models(self):
        """Initialize YOLO models for player detection and pose estimation"""
        if YOLO is None:
            logger.error("YOLO not available. Cannot initialize models.")
            return
        
        try:
            # Initialize pose estimation model
            pose_model_path = self.config.get('models', {}).get('yolo_pose', 'yolov8n-pose.pt')
            self.pose_estimator = PoseEstimator(pose_model_path, self.config.get('yolo_pose', {}))
            logger.info(f"Pose estimator initialized: {pose_model_path}")
            
            # Initialize player detection model
            player_model_path = self.config.get('models', {}).get('yolo_player', 'yolov8n.pt')
            self.yolo_model = YOLO(player_model_path)
            logger.info(f"Player detection model initialized: {player_model_path}")
            
        except Exception as e:
            logger.error(f"Error initializing models: {e}")
            self.pose_estimator = None
            self.yolo_model = None
    
    def process_video(self, video_file: str, output_file: str = None, csv_output: str = None, show_viewer: bool = False):
        """Process video for pose detection"""
        cap = cv2.VideoCapture(video_file)
        
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_file}")
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        logger.info(f"Processing video: {video_file}")
        logger.info(f"Video: {width}x{height} @ {fps}fps, {total_frames} frames")
        
        # Setup video writer if output specified
        out = None
        if output_file:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_file, fourcc, fps, (width, height))
        
        # Setup viewer
        if show_viewer:
            cv2.namedWindow('Tennis Player Pose Detection', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('Tennis Player Pose Detection', 1200, 800)
        
        # Collect pose data for CSV output
        pose_data = []
        frame_number = 0
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Process frame for pose detection
                frame_with_poses, frame_pose_data = self._process_frame(frame, frame_number)
                
                # Store pose data
                if frame_pose_data:
                    pose_data.extend(frame_pose_data)
                
                # Write frame
                if out:
                    out.write(frame_with_poses)
                
                # Show in viewer
                if show_viewer:
                    cv2.imshow('Tennis Player Pose Detection', frame_with_poses)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        break
                    elif key == ord(' '):  # Space to pause
                        cv2.waitKey(0)
                
                # Progress update
                if frame_number % 30 == 0:
                    logger.info(f"Processed {frame_number}/{total_frames} frames ({frame_number/total_frames*100:.1f}%)")
                
                frame_number += 1
        
        finally:
            # Cleanup
            cap.release()
            if out:
                out.release()
            if show_viewer:
                cv2.destroyAllWindows()
        
        # Save pose data to CSV if specified
        if csv_output and pose_data:
            df_poses = pd.DataFrame(pose_data)
            df_poses.to_csv(csv_output, index=False)
            logger.info(f"Pose data saved to {csv_output}")
        
        # Print pose detection summary
        self._print_summary(pose_data)
    
    def _process_frame(self, frame: np.ndarray, frame_number: int) -> Tuple[np.ndarray, List[Dict]]:
        """Process a single frame for pose detection"""
        frame = frame.copy()
        
        # Add frame info
        cv2.putText(frame, f"Frame: {frame_number}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Detect players using YOLO
        player_detections = self._detect_players(frame)
        
        # Estimate poses for detected players
        poses = []
        if self.pose_estimator and player_detections:
            poses = self.pose_estimator.estimate_poses(frame, player_detections)
            
            # Draw poses on frame
            frame = self.pose_estimator.draw_poses(frame, poses)
        
        # Create pose data for CSV output
        frame_pose_data = []
        for pose in poses:
            frame_pose_data.append({
                'frame': frame_number,
                'player_id': pose['player_id'],
                'keypoints': pose['frame_keypoints'],
                'bbox_x1': pose['bbox'][0],
                'bbox_y1': pose['bbox'][1],
                'bbox_x2': pose['bbox'][2],
                'bbox_y2': pose['bbox'][3]
            })
        
        return frame, frame_pose_data
    
    def _detect_players(self, frame: np.ndarray) -> List[Dict]:
        """Detect players in frame using YOLO"""
        if self.yolo_model is None:
            return []
        
        try:
            # Get detection parameters from config
            yolo_config = self.config.get('yolo_player', {})
            conf_threshold = yolo_config.get('conf_threshold', 0.5)
            iou_threshold = yolo_config.get('iou_threshold', 0.45)
            max_det = yolo_config.get('max_det', 10)
            
            # Run YOLO detection
            results = self.yolo_model(
                frame, 
                conf=conf_threshold,
                iou=iou_threshold,
                max_det=max_det,
                verbose=False
            )
            
            player_detections = []
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        # Check if it's a person (class 0 in COCO dataset)
                        class_id = int(box.cls)
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        conf = float(box.conf[0].cpu().numpy())
                        
                        # For custom tennis models, accept any detection above threshold
                        if conf > conf_threshold:
                            player_detections.append({
                                'bbox': [x1, y1, x2, y2],
                                'confidence': conf,
                                'class_id': class_id
                            })
            
            # Sort by confidence and limit to top 2 players
            player_detections.sort(key=lambda x: x['confidence'], reverse=True)
            return player_detections[:2]  # Max 2 players
            
        except Exception as e:
            logger.warning(f"Error in player detection: {e}")
            return []
    
    def _print_summary(self, pose_data: List[Dict]):
        """Print pose detection summary"""
        logger.info("=== POSE DETECTION SUMMARY ===")
        
        if not pose_data:
            logger.info("No pose data available")
            return
        
        # Count poses per player
        player_stats = {}
        for pose in pose_data:
            player_id = pose['player_id']
            if player_id not in player_stats:
                player_stats[player_id] = 0
            player_stats[player_id] += 1
        
        # Print statistics
        for player_id, count in player_stats.items():
            logger.info(f"Player {player_id}: {count} poses detected")
        
        total_poses = len(pose_data)
        total_frames = len(set(pose['frame'] for pose in pose_data))
        logger.info(f"Total poses detected: {total_poses}")
        logger.info(f"Frames with poses: {total_frames}")


def main():
    """Main function for standalone usage"""
    parser = argparse.ArgumentParser(description='Tennis Player Pose Detection System')
    parser.add_argument('--video', required=True, help='Input video file')
    parser.add_argument('--output', default='tennis_pose_analysis.mp4', help='Output video file')
    parser.add_argument('--csv-output', default='player_poses.csv', help='Output CSV file for pose data')
    parser.add_argument('--viewer', action='store_true', help='Show real-time viewer')
    
    args = parser.parse_args()
    
    try:
        processor = PlayerPoseProcessor()
        processor.process_video(args.video, args.output, args.csv_output, show_viewer=args.viewer)
        
    except Exception as e:
        logger.error(f"Error running pose detection: {e}")


if __name__ == "__main__":
    main()
