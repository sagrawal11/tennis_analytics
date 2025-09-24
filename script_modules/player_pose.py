#!/usr/bin/env python3
"""
Simple Tennis Player Pose Detection System
Extracts pose keypoints from tennis videos using YOLO pose estimation
"""

import cv2
import numpy as np
import pandas as pd
import argparse
import logging
from typing import List, Tuple, Optional, Dict, Any
import os
import yaml

# Import YOLO for pose detection
try:
    from ultralytics import YOLO
except ImportError:
    print("Warning: ultralytics not found. Pose detection will be disabled.")
    YOLO = None

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')
logger = logging.getLogger(__name__)


class SimplePoseProcessor:
    """Simple pose detection processor"""
    
    def __init__(self):
        """Initialize processor"""
        self.config = self._load_config()
        self.pose_model = None
        self.player_model = None
        
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
        """Initialize YOLO models"""
        if YOLO is None:
            logger.error("YOLO not available. Cannot initialize models.")
            return
        
        try:
            # Initialize pose estimation model
            pose_model_path = self.config.get('models', {}).get('yolo_pose', 'yolov8n-pose.pt')
            self.pose_model = YOLO(pose_model_path)
            logger.info(f"Pose model initialized: {pose_model_path}")
            
            # Initialize player detection model
            player_model_path = self.config.get('models', {}).get('yolo_player', 'yolov8n.pt')
            self.player_model = YOLO(player_model_path)
            logger.info(f"Player model initialized: {player_model_path}")
            
        except Exception as e:
            logger.error(f"Error initializing models: {e}")
            self.pose_model = None
            self.player_model = None
    
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
        if self.pose_model and player_detections:
            poses = self._estimate_poses_simple(frame, player_detections)
            
            # Draw poses on frame
            frame = self._draw_poses_simple(frame, poses)
        
        # Create pose data for CSV output
        frame_pose_data = []
        for pose in poses:
            frame_pose_data.append({
                'frame': frame_number,
                'player_id': pose['player_id'],
                'keypoints': pose['keypoints_string'],
                'bbox_x1': pose['bbox'][0],
                'bbox_y1': pose['bbox'][1],
                'bbox_x2': pose['bbox'][2],
                'bbox_y2': pose['bbox'][3]
            })
        
        return frame, frame_pose_data
    
    def _detect_players(self, frame: np.ndarray) -> List[Dict]:
        """Detect players in frame using YOLO"""
        if self.player_model is None:
            return []
        
        try:
            # Get detection parameters from config
            yolo_config = self.config.get('yolo_player', {})
            conf_threshold = yolo_config.get('conf_threshold', 0.5)
            iou_threshold = yolo_config.get('iou_threshold', 0.45)
            max_det = yolo_config.get('max_det', 10)
            
            # Run YOLO detection
            results = self.player_model(
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
    
    def _estimate_poses_simple(self, frame: np.ndarray, player_detections: List[Dict]) -> List[Dict]:
        """Simple pose estimation without complex processing"""
        if not self.pose_model or not player_detections:
            return []
        
        poses = []
        
        try:
            # Process each player individually
            for player_idx, player in enumerate(player_detections):
                if 'bbox' not in player:
                    continue
                
                x1, y1, x2, y2 = player['bbox']
                
                # Extract player region with padding
                padding = 60
                x1_pad = max(0, int(x1 - padding))
                y1_pad = max(0, int(y1 - padding))
                x2_pad = min(frame.shape[1], int(x2 + padding))
                y2_pad = min(frame.shape[0], int(y2 + padding))
                
                player_roi = frame[y1_pad:y2_pad, x1_pad:x2_pad]
                
                if player_roi.size == 0:
                    continue
                
                # Run pose detection on player ROI
                results = self.pose_model(player_roi, verbose=False, max_det=1)
                
                for result in results:
                    if result.keypoints is not None and len(result.keypoints.data) > 0:
                        keypoints = result.keypoints.data[0].cpu().numpy()
                        
                        # Adjust keypoints back to full frame coordinates
                        keypoints[:, 0] += x1_pad  # Add x offset
                        keypoints[:, 1] += y1_pad  # Add y offset
                        
                        # Format keypoints as string
                        keypoint_strings = []
                        for kp in keypoints:
                            x, y = kp[0], kp[1]
                            keypoint_strings.append(f"{x:.1f},{y:.1f}")
                        keypoints_string = "|".join(keypoint_strings)
                        
                        # Create pose data
                        pose_data = {
                            'player_id': player_idx,
                            'keypoints': keypoints,
                            'keypoints_string': keypoints_string,
                            'bbox': [x1, y1, x2, y2]
                        }
                        
                        poses.append(pose_data)
            
            return poses
            
        except Exception as e:
            logger.warning(f"Error in pose estimation: {e}")
            return []
    
    def _draw_poses_simple(self, frame: np.ndarray, poses: List[Dict]) -> np.ndarray:
        """Draw poses on frame"""
        for pose in poses:
            keypoints = pose['keypoints']
            
            # Draw keypoints
            for i, kp in enumerate(keypoints):
                x, y = int(kp[0]), int(kp[1])
                cv2.circle(frame, (x, y), 3, (0, 255, 0), -1)
                cv2.putText(frame, str(i), (x+5, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
        
        return frame
    
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
    parser = argparse.ArgumentParser(description='Simple Tennis Player Pose Detection System')
    parser.add_argument('--video', required=True, help='Input video file')
    parser.add_argument('--output', default='tennis_pose_analysis.mp4', help='Output video file')
    parser.add_argument('--csv-output', default='player_poses.csv', help='Output CSV file for pose data')
    parser.add_argument('--viewer', action='store_true', help='Show real-time viewer')
    
    args = parser.parse_args()
    
    try:
        processor = SimplePoseProcessor()
        processor.process_video(args.video, args.output, args.csv_output, show_viewer=args.viewer)
        
    except Exception as e:
        logger.error(f"Error running pose detection: {e}")


if __name__ == "__main__":
    main()
