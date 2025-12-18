#!/usr/bin/env python3
"""
Tennis Player Detection and Pose Estimation
Focused script for player detection and pose estimation using the same setup as tennis_CV.py
"""

import cv2
import numpy as np
import yaml
import time
import torch
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import logging
from collections import deque
import sys
import argparse

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import RF-DETR for enhanced player detection
try:
    from rfdetr import RFDETRNano
    RFDETR_AVAILABLE = True
    logger.info("RF-DETR imports successful - Enhanced detection enabled")
except ImportError as e:
    RFDETR_AVAILABLE = False
    logger.warning(f"RF-DETR imports failed: {e} - Using YOLO fallback")

class PlayerDetector:
    """YOLOv8-based player detection"""
    
    def __init__(self, model_path: str, config: Dict[str, Any]):
        self.config = config
        try:
            from ultralytics import YOLO
            self.model = YOLO(model_path)
            logger.info(f"✓ YOLO player detector loaded: {model_path}")
        except Exception as e:
            logger.error(f"Failed to load YOLO player detector: {e}")
            self.model = None
    
    def detect_players(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        if not self.model:
            return []
        
        try:
            results = self.model(
                frame,
                conf=self.config['player_detection']['confidence_threshold'],
                iou=self.config['player_detection']['iou_threshold'],
                classes=[0],  # Person class
                verbose=False
            )
            
            detections = []
            for result in results:
                if result.boxes is not None:
                    boxes = result.boxes.xyxy.cpu().numpy()
                    confidences = result.boxes.conf.cpu().numpy()
                    
                    for i, (box, conf) in enumerate(zip(boxes, confidences)):
                        x1, y1, x2, y2 = box.astype(int)
                        detections.append({
                            'bbox': [x1, y1, x2, y2],
                            'confidence': float(conf),
                            'class': 'person',
                            'source': 'yolo'
                        })
            
            return detections
            
        except Exception as e:
            logger.warning(f"YOLO player detection failed: {e}")
            return []

class RFDETRPlayerDetector:
    """RF-DETR-based player detection with tennis-specific filtering"""
    
    def __init__(self, model_path: str, config: Dict[str, Any]):
        self.config = config
        try:
            self.model = RFDETRNano.from_pretrained(model_path)
            logger.info(f"✓ RF-DETR player detector loaded: {model_path}")
        except Exception as e:
            logger.error(f"Failed to load RF-DETR player detector: {e}")
            self.model = None
    
    def detect_players(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        if not self.model:
            logger.warning("🔍 RF-DETR: Model not available")
            return []
        
        try:
            # Run RF-DETR inference
            results = self.model.infer(frame)
            
            detections = []
            if 'boxes' in results and len(results['boxes']) > 0:
                boxes = results['boxes']
                confidences = results['scores']
                labels = results['labels']
                
                for i, (box, conf, label) in enumerate(zip(boxes, confidences, labels)):
                    # Filter for person class (class 0) and tennis-specific criteria
                    if label == 0 and conf > self.config['player_detection']['confidence_threshold']:
                        x1, y1, x2, y2 = box.astype(int)
                        
                        # Tennis-specific filtering
                        width = x2 - x1
                        height = y2 - y1
                        aspect_ratio = height / width if width > 0 else 0
                        
                        # Filter out detections that are too small or have wrong aspect ratio
                        if (width > 50 and height > 100 and 
                            1.5 < aspect_ratio < 4.0 and  # Person should be taller than wide
                            y2 > frame.shape[0] * 0.3):  # Should be in lower 70% of frame
                            
                            detections.append({
                                'bbox': [x1, y1, x2, y2],
                                'confidence': float(conf),
                                'class': 'person',
                                'source': 'rfdetr'
                            })
            
            logger.debug(f"🔍 RF-DETR: Found {len(detections)} players")
            return detections
            
        except Exception as e:
            logger.warning(f"RF-DETR player detection failed: {e}")
            return []

class PoseEstimator:
    """Multi-scale pose estimation for better arm detection"""
    
    def __init__(self, model_path: str):
        """Initialize the multi-scale pose estimator"""
        try:
            from ultralytics import YOLO
            self.model = YOLO(model_path)
            self.history_length = 10
            self.pose_history = {}  # {player_id: deque of poses}
            logger.info(f"✓ Pose estimator loaded: {model_path}")
        except Exception as e:
            logger.error(f"Failed to load pose estimator: {e}")
            self.model = None
    
    def estimate_poses(self, frame: np.ndarray, player_detections: List[Dict]) -> List[Dict]:
        """Estimate poses using multi-scale detection for better arm keypoints"""
        if not self.model or not player_detections:
            return []
        
        try:
            poses = []
            
            for i, player in enumerate(player_detections):
                bbox = player['bbox']
                x1, y1, x2, y2 = bbox
                
                # Add padding around player for better pose detection
                padding = 20
                x1_pad = max(0, x1 - padding)
                y1_pad = max(0, y1 - padding)
                x2_pad = min(frame.shape[1], x2 + padding)
                y2_pad = min(frame.shape[0], y2 + padding)
                
                # Extract player ROI
                player_roi = frame[y1_pad:y2_pad, x1_pad:x2_pad]
                
                if player_roi.size == 0:
                    continue
                
                # Run pose estimation on ROI
                results = self.model(
                    player_roi,
                    conf=0.5,
                    verbose=False
                )
                
                if results and len(results) > 0:
                    result = results[0]
                    if result.keypoints is not None and len(result.keypoints) > 0:
                        keypoints = result.keypoints.xy[0].cpu().numpy()
                        confidences = result.keypoints.conf[0].cpu().numpy()
                        
                        # Convert keypoints back to full frame coordinates
                        full_frame_keypoints = []
                        for kp, conf in zip(keypoints, confidences):
                            if len(kp) >= 2:
                                full_x = kp[0] + x1_pad
                                full_y = kp[1] + y1_pad
                                full_frame_keypoints.append([full_x, full_y, float(conf)])
                            else:
                                full_frame_keypoints.append([0, 0, 0])
                        
                        # Validate pose has hips within bounding box
                        if self._validate_pose_hips({'keypoints': full_frame_keypoints}, bbox):
                            pose = {
                                'player_id': i,
                                'keypoints': full_frame_keypoints,
                                'bbox': bbox,
                                'confidence': float(np.mean(confidences)),
                                'source': 'yolo_pose'
                            }
                            poses.append(pose)
                            
                            # Update pose history for temporal smoothing
                            self._update_pose_history(i, pose)
            
            return poses
            
        except Exception as e:
            logger.warning(f"Pose estimation failed: {e}")
            return []
    
    def _validate_pose_hips(self, pose: Dict, bbox: List[int]) -> bool:
        """Validate that the pose has hips within the player bounding box"""
        x1, y1, x2, y2 = bbox
        keypoints = pose['keypoints']
        
        # Hip keypoints are indices 11 (left hip) and 12 (right hip)
        if len(keypoints) > 12:
            left_hip = keypoints[11]
            right_hip = keypoints[12]
            
            # Check if at least one hip is within the bounding box and has reasonable confidence
            left_hip_valid = (left_hip[2] > 0.3 and 
                            x1 <= left_hip[0] <= x2 and 
                            y1 <= left_hip[1] <= y2)
            
            right_hip_valid = (right_hip[2] > 0.3 and 
                             x1 <= right_hip[0] <= x2 and 
                             y1 <= right_hip[1] <= y2)
            
            return left_hip_valid or right_hip_valid
        
        return False
    
    def _update_pose_history(self, player_id: int, pose: Dict):
        """Update pose history for temporal smoothing"""
        if player_id not in self.pose_history:
            self.pose_history[player_id] = deque(maxlen=self.history_length)
        
        self.pose_history[player_id].append(pose)
    
    def draw_poses(self, frame: np.ndarray, poses: List[Dict]) -> np.ndarray:
        """Draw poses on frame"""
        if not poses:
            return frame
        
        for pose in poses:
            keypoints = pose['keypoints']
            bbox = pose['bbox']
            
            # Draw bounding box
            x1, y1, x2, y2 = bbox
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw keypoints
            for i, kp in enumerate(keypoints):
                if len(kp) >= 3 and kp[2] > 0.3:  # Confidence threshold
                    x, y = int(kp[0]), int(kp[1])
                    cv2.circle(frame, (x, y), 3, (0, 0, 255), -1)
                    cv2.putText(frame, str(i), (x+5, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
            
            # Draw pose connections
            self._draw_pose_connections(frame, keypoints)
            
            # Draw player info
            player_id = pose.get('player_id', 0)
            confidence = pose.get('confidence', 0)
            cv2.putText(frame, f"Player {player_id}: {confidence:.2f}", 
                       (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        return frame
    
    def _draw_pose_connections(self, frame: np.ndarray, keypoints: List[List[float]]):
        """Draw connections between keypoints"""
        if len(keypoints) < 17:
            return
        
        # Define pose connections (simplified)
        connections = [
            (0, 1), (0, 2), (1, 3), (2, 4),  # Head
            (5, 6), (5, 7), (6, 8), (7, 9), (8, 10),  # Arms
            (11, 12), (11, 13), (12, 14), (13, 15), (14, 16)  # Legs
        ]
        
        for start_idx, end_idx in connections:
            if (len(keypoints) > max(start_idx, end_idx) and 
                keypoints[start_idx][2] > 0.3 and keypoints[end_idx][2] > 0.3):
                
                start_point = (int(keypoints[start_idx][0]), int(keypoints[start_idx][1]))
                end_point = (int(keypoints[end_idx][0]), int(keypoints[end_idx][1]))
                cv2.line(frame, start_point, end_point, (255, 0, 0), 2)

class TennisPlayerDetector:
    """Main tennis player detection and pose estimation system"""
    
    def __init__(self, config_path: str = "config.yaml"):
        """Initialize the tennis player detection system"""
        self.config = self._load_config(config_path)
        self.frame_count = 0
        self.start_time = time.time()
        
        # Initialize components
        self.player_detector = None
        self.rfdetr_player_detector = None
        self.pose_estimator = None
        
        # Player tracking state
        self.player_positions = deque(maxlen=30)
        self.player_velocities = deque(maxlen=10)
        
        # Initialize all components
        self._initialize_components()
        
        logger.info("✓ Tennis Player Detector initialized")
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            logger.info(f"Configuration loaded from {config_path}")
            return config
        except Exception as e:
            logger.error(f"Error loading config: {e}")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration"""
        return {
            'models': {
                'yolo_player_model': 'models/playersnball5.pt',
                'rfdetr_model': 'rf-detr-base.pth',
                'pose_model': 'models/yolov8n-pose.pt'
            },
            'player_detection': {
                'confidence_threshold': 0.5,
                'iou_threshold': 0.45,
                'max_players': 2
            },
            'pose_estimation': {
                'confidence_threshold': 0.5,
                'min_keypoints': 5
            }
        }
    
    def _initialize_components(self):
        """Initialize all detection components"""
        try:
            # Initialize RF-DETR player detector (primary)
            if RFDETR_AVAILABLE:
                rfdetr_model_path = self.config['models'].get('rfdetr_model')
                if rfdetr_model_path and Path(rfdetr_model_path).exists():
                    self.rfdetr_player_detector = RFDETRPlayerDetector(rfdetr_model_path, self.config)
                    logger.info("✓ RF-DETR player detector initialized")
                else:
                    logger.warning("RF-DETR model not found, using YOLO fallback")
            
            # Initialize YOLO player detector (fallback)
            yolo_model_path = self.config['models'].get('yolo_player_model')
            if yolo_model_path and Path(yolo_model_path).exists():
                self.player_detector = PlayerDetector(yolo_model_path, self.config)
                logger.info("✓ YOLO player detector initialized")
            else:
                logger.warning("YOLO player model not found")
            
            # Initialize pose estimator
            pose_model_path = self.config['models'].get('pose_model')
            if pose_model_path and Path(pose_model_path).exists():
                self.pose_estimator = PoseEstimator(pose_model_path)
                logger.info("✓ Pose estimator initialized")
            else:
                logger.warning("Pose model not found")
                
        except Exception as e:
            logger.error(f"Error initializing components: {e}")
    
    def process_frame(self, frame: np.ndarray) -> Dict[str, Any]:
        """Process a single frame for player detection and pose estimation"""
        start_time = time.time()
        self.frame_count += 1
        
        # Detect players
        player_detections = self._detect_players(frame)
        
        # Estimate poses
        pose_detections = []
        if self.pose_estimator and player_detections:
            pose_detections = self.pose_estimator.estimate_poses(frame, player_detections)
        
        # Draw visualizations
        output_frame = frame.copy()
        if pose_detections:
            output_frame = self.pose_estimator.draw_poses(output_frame, pose_detections)
        
        # Calculate processing time
        processing_time = time.time() - start_time
        
        # Prepare results
        results = {
            'frame_number': self.frame_count,
            'timestamp': time.time() - self.start_time,
            'player_count': len(player_detections),
            'player_detections': player_detections,
            'pose_count': len(pose_detections),
            'pose_detections': pose_detections,
            'processing_time': processing_time,
            'output_frame': output_frame
        }
        
        return results
    
    def _detect_players(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """Detect players using the best available detector"""
        # Try RF-DETR first if available
        if self.rfdetr_player_detector:
            detections = self.rfdetr_player_detector.detect_players(frame)
            if detections:
                logger.debug(f"RF-DETR detected {len(detections)} players")
                return detections
        
        # Fallback to YOLO
        if self.player_detector:
            detections = self.player_detector.detect_players(frame)
            if detections:
                logger.debug(f"YOLO detected {len(detections)} players")
                return detections
        
        return []
    
    def process_video(self, video_path: str, output_path: Optional[str] = None, show_viewer: bool = True):
        """Process entire video for player detection and pose estimation"""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error(f"Could not open video: {video_path}")
            return
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        logger.info(f"Processing video: {video_path}")
        logger.info(f"Resolution: {width}x{height}, FPS: {fps}, Frames: {total_frames}")
        
        # Setup video writer if output path provided
        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_count = 0
        start_time = time.time()
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Process frame
                results = self.process_frame(frame)
                output_frame = results['output_frame']
                
                # Add frame info
                cv2.putText(output_frame, f"Frame: {frame_count}/{total_frames}", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(output_frame, f"Players: {results['player_count']}", 
                           (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(output_frame, f"Poses: {results['pose_count']}", 
                           (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                # Write frame if output specified
                if writer:
                    writer.write(output_frame)
                
                # Show viewer if requested
                if show_viewer:
                    cv2.imshow('Tennis Player Detection', output_frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                
                frame_count += 1
                
                # Progress update
                if frame_count % 30 == 0:
                    elapsed = time.time() - start_time
                    fps_actual = frame_count / elapsed
                    logger.info(f"Processed {frame_count}/{total_frames} frames ({fps_actual:.1f} FPS)")
        
        except KeyboardInterrupt:
            logger.info("Processing interrupted by user")
        
        finally:
            cap.release()
            if writer:
                writer.release()
            cv2.destroyAllWindows()
            
            # Final stats
            total_time = time.time() - start_time
            avg_fps = frame_count / total_time if total_time > 0 else 0
            logger.info(f"✓ Processing complete: {frame_count} frames in {total_time:.1f}s ({avg_fps:.1f} FPS)")

def main():
    parser = argparse.ArgumentParser(description='Tennis Player Detection and Pose Estimation')
    parser.add_argument('--video', required=True, help='Input video file')
    parser.add_argument('--output', help='Output video file')
    parser.add_argument('--config', default='config.yaml', help='Configuration file')
    parser.add_argument('--viewer', action='store_true', help='Show viewer window')
    
    args = parser.parse_args()
    
    # Initialize detector
    detector = TennisPlayerDetector(args.config)
    
    # Process video
    detector.process_video(
        video_path=args.video,
        output_path=args.output,
        show_viewer=args.viewer
    )

if __name__ == "__main__":
    main()
