#!/usr/bin/env python3
"""
Super Advanced Tennis Analysis Engine
Integrates player detection, pose estimation, ball tracking, bounce detection, AND court detection
for comprehensive tennis video analysis with real-time visualization and geometric validation.
"""

import cv2
import numpy as np
import yaml
import time
import torch
import torch.nn.functional as F
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import logging
from collections import deque
import sys

# Add TennisCourtDetector to path for court detection
sys.path.append('TennisCourtDetector')

# Import from TennisCourtDetector
try:
    from tracknet import BallTrackerNet
    from postprocess import postprocess, refine_kps
    from homography import get_trans_matrix, refer_kps
    COURT_DETECTION_AVAILABLE = True
    logger.info("TennisCourtDetector imports successful - Court detection enabled")
except ImportError as e:
    COURT_DETECTION_AVAILABLE = False
    logger.warning(f"TennisCourtDetector imports failed: {e} - Court detection will be disabled")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TennisAnalysisDemo:
    """Super advanced integrated tennis analysis system with court detection"""
    
    def __init__(self, config_path: str = "config.yaml"):
        """Initialize the super advanced tennis analysis system"""
        self.config = self._load_config(config_path)
        self.frame_count = 0
        self.start_time = time.time()
        
        # Initialize components
        self.player_detector = None
        self.pose_estimator = None
        self.bounce_detector = None
        self.tracknet_model = None
        self.yolo_ball_model = None
        self.court_detector = None  # NEW: Court detection system
        
        # Ball tracking state
        self.ball_positions = deque(maxlen=30)  # Store last 30 ball positions
        self.ball_velocities = deque(maxlen=10)  # Store last 10 velocities
        self.tracknet_predictions = []
        self.yolo_predictions = []
        
        # Court detection state (NEW)
        self.court_keypoints = []
        self.court_confidence = []
        
        # Colinearity-based quality assessment for tennis court lines
        self.court_line_groups = {
            # Horizontal lines (top to bottom)
            'top_horizontal': [0, 4, 6, 1],      # Top baseline + service lines
            'bottom_horizontal': [2, 5, 7, 3],   # Bottom baseline + service lines
            'top_service': [8, 12, 9],            # Top service line + net
            'bottom_service': [10, 13, 11],       # Bottom service line + net
            
            # Vertical lines (left to right)
            'left_vertical': [5, 10, 8, 4],      # Left side lines
            'right_vertical': [6, 9, 11, 7],     # Right side lines
        }
        
        # Parallel line pairs for additional validation
        self.parallel_line_pairs = [
            # Endlines should be parallel to service lines
            ('endline_top', [1, 3], 'service_line_right', [6, 9, 11, 7]),      # Top endline || right service line
            ('endline_bottom', [2, 0], 'service_line_left', [5, 10, 8, 4]),     # Bottom endline || left service line
            
            # All horizontal lines should be parallel to each other
            ('baseline_top', [0, 1], 'baseline_bottom', [2, 3]),                # Top baseline || bottom baseline
            ('baseline_top', [0, 1], 'top_service', [8, 9]),                    # Top baseline || top service line
            ('baseline_top', [0, 1], 'bottom_service', [10, 11]),               # Top baseline || bottom service line
            ('baseline_bottom', [2, 3], 'top_service', [8, 9]),                 # Bottom baseline || top service line
            ('baseline_bottom', [2, 3], 'bottom_service', [10, 11]),            # Bottom baseline || bottom service line
            ('top_service', [8, 9], 'bottom_service', [10, 11]),                # Top service line || bottom service line
            
            # All vertical lines should be parallel to each other
            ('left_vertical', [5, 10, 8, 4], 'right_vertical', [6, 9, 11, 7]), # Left side || right side
            ('left_vertical', [5, 10, 8, 4], 'center_service', [8, 12, 10]),   # Left side || center service
            ('right_vertical', [6, 9, 11, 7], 'center_service', [8, 12, 10]),  # Right side || center service
        ]
        
        # Soft-locked keypoints - only replace when better ones come along
        self.best_keypoints = {}  # keypoint_id -> (x, y, best_score)
        self.quality_threshold = 0.1  # Only consider keypoints with score <= 0.1 as "good enough"
        self.min_history_frames = 3   # Minimum frames before considering a keypoint stable
        
        # Temporal smoothing for static camera
        self.keypoint_history = {}  # Store keypoint positions over time
        self.keypoint_confidence = {}  # Store confidence scores
        self.smoothed_keypoints = {}  # Final smoothed keypoints
        self.history_length = 30  # Number of frames to average over
        self.min_confidence_threshold = 0.3  # Minimum confidence to consider a detection
        
        # Best position tracking - preserve the best predictions we've ever seen
        self.best_positions = {}  # Store the BEST individual prediction ever seen for each keypoint
        self.best_scores = {}  # Store the quality scores for the best positions
        
        # Analysis results
        self.analysis_results = {
            'total_frames': 0,
            'players_detected': 0,
            'poses_estimated': 0,
            'bounces_detected': 0,
            'tracknet_detections': 0,
            'yolo_ball_detections': 0,
            'combined_ball_detections': 0,
            'court_detections': 0,  # NEW: Court detection count
            'keypoints_detected': 0,  # NEW: Court keypoint count
            'processing_times': []
        }
        
        self._initialize_components()
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            logger.info(f"Configuration loaded from {config_path}")
            return config
        except Exception as e:
            logger.error(f"Error loading config: {e}")
            # Fallback to default config
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration if YAML loading fails"""
        return {
            'models': {
                'yolo_player': 'models/playersnball4.pt',
                'yolo_pose': 'models/yolov8n-pose.pt',
                'bounce_detector': 'models/bounce_detector.cbm',
                'tracknet': 'pretrained_ball_detection.pt',
                'yolo_ball': 'models/playersnball4.pt',
                'court_detector': 'model_tennis_court_det.pt'  # NEW: Court detection model
            },
            'yolo_player': {
                'conf_threshold': 0.5,
                'iou_threshold': 0.45,
                'max_det': 10
            },
            'yolo_pose': {
                'conf_threshold': 0.3,
                'iou_threshold': 0.45,
                'max_det': 4,
                'keypoints': 17
            },
            'tracknet': {
                'input_height': 360,
                'input_width': 640,
                'conf_threshold': 0.1,
                'max_dist': 100,
                'max_gap': 4,
                'min_track_length': 5
            },
            'yolo_ball': {
                'conf_threshold': 0.1,
                'iou_threshold': 0.45,
                'max_det': 10
            },
            'ball_tracking': {
                'max_velocity': 200,
                'min_confidence': 0.4,
                'smoothing_factor': 0.7,
                'prediction_frames': 3
            },
            'court_detection': {  # NEW: Court detection settings
                'input_width': 640,
                'input_height': 360,
                'low_threshold': 170,
                'min_radius': 10,
                'max_radius': 25,
                'use_refine_kps': True,
                'use_homography': True
            },
            'video': {
                'fps': 30,
                'frame_skip': 1,
                'resize_width': 1920,
                'resize_height': 1080
            }
        }
    
    def _initialize_components(self):
        """Initialize all analysis components including court detection"""
        try:
            # Initialize player detector
            player_model_path = self.config['models']['yolo_player']
            if Path(player_model_path).exists():
                self.player_detector = PlayerDetector(
                    player_model_path,
                    self.config['yolo_player']
                )
                logger.info("Player detector initialized successfully")
            else:
                logger.warning(f"Player detection model not found: {player_model_path}")
            
            # Initialize pose estimator
            pose_model_path = self.config['models']['yolo_pose']
            if Path(pose_model_path).exists():
                self.pose_estimator = PoseEstimator(
                    pose_model_path,
                    self.config['yolo_pose']
                )
                logger.info("Pose estimator initialized successfully")
            else:
                logger.warning(f"Pose estimation model not found: {pose_model_path}")
            
            # Initialize bounce detector (CatBoost model)
            bounce_model_path = self.config['models'].get('bounce_detector')
            if bounce_model_path and Path(bounce_model_path).exists():
                try:
                    self.bounce_detector = BounceDetector(bounce_model_path)
                    logger.info("Bounce detector initialized successfully")
                except Exception as e:
                    logger.warning(f"Bounce detector initialization failed: {e}")
                    self.bounce_detector = None
            else:
                logger.warning(f"Bounce detection model not found: {bounce_model_path}")
                self.bounce_detector = None
            
            # Initialize TrackNet model for ball detection
            tracknet_path = self.config['models'].get('tracknet')
            if tracknet_path and Path(tracknet_path).exists():
                try:
                    self.tracknet_model = TrackNetDetector(
                        tracknet_path,
                        self.config['tracknet']
                    )
                    logger.info("TrackNet model initialized successfully")
                except Exception as e:
                    logger.warning(f"TrackNet model initialization failed: {e}")
                    self.tracknet_model = None
            else:
                logger.warning(f"TrackNet model not found: {tracknet_path}")
                self.tracknet_model = None
            
            # Initialize YOLO model for ball detection
            yolo_ball_path = self.config['models'].get('yolo_ball')
            if yolo_ball_path and Path(yolo_ball_path).exists():
                try:
                    self.yolo_ball_model = YOLOBallDetector(
                        yolo_ball_path,
                        self.config['yolo_ball']
                    )
                    logger.info("YOLO ball detector initialized successfully")
                except Exception as e:
                    logger.warning(f"YOLO ball model initialization failed: {e}")
                    self.yolo_ball_model = None
            else:
                logger.warning(f"YOLO ball model not found: {yolo_ball_path}")
                self.yolo_ball_model = None
            
            # NEW: Initialize court detector
            if COURT_DETECTION_AVAILABLE:
                court_model_path = self.config['models'].get('court_detector')
                
                if court_model_path and Path(court_model_path).exists():
                    try:
                        self.court_detector = CourtDetector(
                            court_model_path,
                            self.config['court_detection']
                        )
                        if self.court_detector.model is not None:
                            logger.info("Court detector initialized successfully")
                        else:
                            logger.warning("Court detector model failed to load")
                            self.court_detector = None
                    except Exception as e:
                        logger.warning(f"Court detection model initialization failed: {e}")
                        self.court_detector = None
                else:
                    logger.warning(f"Court detection model not found: {court_model_path}")
                    self.court_detector = None
            else:
                logger.warning("Court detection not available - TennisCourtDetector not found")
                self.court_detector = None
                
        except Exception as e:
            logger.error(f"Error initializing components: {e}")
    
    def analyze_video(self, video_path: str, output_path: Optional[str] = None):
        """Analyze tennis video with all models including court detection"""
        if not Path(video_path).exists():
            logger.error(f"Video file not found: {video_path}")
            return
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error(f"Could not open video: {video_path}")
            return
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        logger.info(f"Video: {width}x{height}, {fps} FPS, {total_frames} frames")
        
        # Setup output video if specified
        output_writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            output_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_skip = self.config['video'].get('frame_skip', 1)
        frame_count = 0
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Process every nth frame
                if frame_count % frame_skip != 0:
                    frame_count += 1
                    continue
                
                # Analyze frame with ALL systems
                start_time = time.time()
                annotated_frame = self._analyze_frame(frame)
                processing_time = time.time() - start_time
                
                # Update statistics
                self.analysis_results['total_frames'] += 1
                self.analysis_results['processing_times'].append(processing_time)
                
                # Display frame
                cv2.imshow('Super Advanced Tennis Analysis Engine', annotated_frame)
                
                # Save to output video if specified
                if output_writer:
                    output_writer.write(annotated_frame)
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):  # Quit
                    break
                elif key == ord('p'):  # Pause/Resume
                    cv2.waitKey(0)
                elif key == ord('s'):  # Save current frame
                    cv2.imwrite(f"super_tennis_frame_{frame_count:06d}.jpg", annotated_frame)
                    logger.info(f"Saved frame {frame_count}")
                
                frame_count += 1
                
                # Display progress
                if frame_count % 30 == 0:
                    progress = (frame_count / total_frames) * 100
                    avg_time = np.mean(self.analysis_results['processing_times'])
                    logger.info(f"Progress: {progress:.1f}% | Avg processing time: {avg_time:.3f}s")
        
        finally:
            cap.release()
            if output_writer:
                output_writer.release()
            cv2.destroyAllWindows()
            
            # Print final statistics
            self._print_analysis_summary()
    
    def _analyze_frame(self, frame: np.ndarray) -> np.ndarray:
        """Analyze a single frame with ALL systems including court detection"""
        annotated_frame = frame.copy()
        
        # 1. Player Detection
        player_detections = []
        if self.player_detector:
            try:
                player_detections = self.player_detector.detect_players(frame)
                self.analysis_results['players_detected'] += len(player_detections)
                annotated_frame = self.player_detector.draw_detections(annotated_frame, player_detections)
            except Exception as e:
                logger.error(f"Player detection error: {e}")
        
        # 2. Pose Estimation
        poses = []
        if self.pose_estimator and player_detections:
            try:
                poses = self.pose_estimator.estimate_poses(frame, player_detections)
                self.analysis_results['poses_estimated'] += len(poses)
                annotated_frame = self.pose_estimator.draw_poses(annotated_frame, poses)
            except Exception as e:
                logger.error(f"Pose estimation error: {e}")
        
        # 3. Ball Detection and Tracking
        ball_pred = self._detect_ball_in_frame(frame)
        if ball_pred:
            self.analysis_results['combined_ball_detections'] += 1
            self.ball_positions.append(ball_pred)
            
            # Calculate velocity
            if len(self.ball_positions) >= 2:
                velocity = self._calculate_velocity(self.ball_positions[-2], ball_pred)
                self.ball_velocities.append(velocity)
            
            # Draw ball tracking
            annotated_frame = self._draw_ball_tracking(annotated_frame, ball_pred)
        
        # 4. Ball Bounce Detection
        if self.bounce_detector:
            try:
                bounce_probability = self.bounce_detector.detect_bounce(frame)
                if bounce_probability > 0.7:  # High confidence threshold
                    self.analysis_results['bounces_detected'] += 1
                    # Draw bounce indicator
                    cv2.putText(annotated_frame, f"BOUNCE! ({bounce_probability:.2f})", 
                               (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    # Draw bounce circle
                    cv2.circle(annotated_frame, (100, 100), 30, (0, 0, 255), -1)
            except Exception as e:
                logger.error(f"Bounce detection error: {e}")
        
        # 5. NEW: Court Detection and Analysis
        if self.court_detector:
            try:
                court_points = self.court_detector.detect_court_in_frame(frame)
                
                if court_points:
                    # Update court statistics
                    keypoints_detected = sum(1 for p in court_points if p[0] is not None and p[1] is not None)
                    self.analysis_results['keypoints_detected'] += keypoints_detected
                    
                    # Draw court keypoints and lines
                    annotated_frame = self._draw_court_visualization(annotated_frame, court_points)
                    
                    # Update court detection count if we have enough keypoints
                    if keypoints_detected >= 4:
                        self.analysis_results['court_detections'] += 1
                        
            except Exception as e:
                logger.error(f"Court detection error: {e}")
        
        # Add comprehensive frame information
        self._add_frame_info(annotated_frame)
        
        return annotated_frame
    
    def _draw_court_visualization(self, frame: np.ndarray, court_points: List[Tuple]) -> np.ndarray:
        """Draw court keypoints and lines with sophisticated visualization"""
        annotated_frame = frame.copy()
        
        # Draw keypoints with color coding based on quality
        keypoints_detected = 0
        for j, point in enumerate(court_points):
            if point[0] is not None and point[1] is not None:
                # Color based on soft-lock status and quality
                if j in self.best_keypoints:
                    best_score = self.best_keypoints[j][2]
                    if best_score <= self.quality_threshold:
                        # Soft-locked keypoints get blue color
                        color = (255, 0, 0)  # Blue for soft-locked
                        thickness = 4
                    else:
                        # Best known but not yet soft-locked
                        color = (0, 255, 255)  # Cyan for best known
                        thickness = 3
                else:
                    # Default to green for detected keypoints
                    color = (0, 255, 0)  # Green
                    thickness = 2
                
                # Draw keypoint
                cv2.circle(annotated_frame, (int(point[0]), int(point[1])),
                          radius=5, color=color, thickness=-1)  # Filled circle
                cv2.circle(annotated_frame, (int(point[0]), int(point[1])),
                          radius=8, color=color, thickness=thickness)   # Outline
                
                # Add keypoint number and status info
                if j in self.best_keypoints:
                    best_score = self.best_keypoints[j][2]
                    if best_score <= self.quality_threshold:
                        status_text = f"{j}[LOCKED]"
                    else:
                        status_text = f"{j}[BEST:{best_score:.2f}]"
                else:
                    status_text = f"{j}"
                
                cv2.putText(annotated_frame, status_text, (int(point[0]) + 10, int(point[1]) - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                
                keypoints_detected += 1
        
        # Draw court lines if we have enough keypoints
        if keypoints_detected >= 4:
            self._draw_court_lines(annotated_frame, court_points)
        
        return annotated_frame
    
    def _draw_court_lines(self, frame: np.ndarray, points: List[Tuple]):
        """Draw court lines connecting keypoints based on colinearity groups"""
        try:
            # Define court line connections that match our colinearity groups
            # Horizontal lines (left to right)
            horizontal_lines = [
                (0, 4, 6, 1),      # Top endline: 0 → 4 → 6 → 1
                (2, 5, 7, 3),      # Bottom endline: 2 → 5 → 7 → 3
                (8, 12, 9),        # Top service line: 8 → 12 → 9
                (10, 13, 11),      # Bottom service line: 10 → 13 → 11
            ]
            
            # Vertical lines (top to bottom)
            vertical_lines = [
                (0, 2),             # Left sideline: 0 → 2
                (1, 3),             # Right sideline: 1 → 3
                (5, 10, 8, 4),     # Left doubles alley: 5 → 10 → 8 → 4
                (6, 9, 11, 7),     # Right doubles alley: 6 → 9 → 11 → 7
            ]
            
            # Draw horizontal lines (blue)
            for line_indices in horizontal_lines:
                self._draw_continuous_line(frame, points, line_indices, (255, 0, 0), 3, "horizontal")
            
            # Draw vertical lines (green)
            for line_indices in vertical_lines:
                self._draw_continuous_line(frame, points, line_indices, (0, 255, 0), 3, "vertical")
            
        except Exception as e:
            logger.error(f"Error drawing court lines: {e}")
    
    def _draw_continuous_line(self, frame: np.ndarray, points: List[Tuple], line_indices: List[int], 
                             color: Tuple[int, int, int], thickness: int, line_type: str):
        """Draw a continuous line through multiple points"""
        valid_points = []
        
        # Collect valid points for this line
        for idx in line_indices:
            if (idx < len(points) and 
                points[idx][0] is not None and points[idx][1] is not None):
                valid_points.append((int(points[idx][0]), int(points[idx][1])))
        
        if len(valid_points) < 2:
            return
        
        # Draw line segments connecting all points
        for i in range(len(valid_points) - 1):
            start_point = valid_points[i]
            end_point = valid_points[i + 1]
            cv2.line(frame, start_point, end_point, color, thickness)
    
    def _detect_ball_in_frame(self, frame: np.ndarray) -> Optional[Dict[str, Any]]:
        """Detect ball in a single frame using both models"""
        tracknet_pred = None
        yolo_pred = None
        
        # 1. TrackNet prediction
        if self.tracknet_model:
            try:
                tracknet_pred = self.tracknet_model.detect_ball(frame)
                if tracknet_pred:
                    self.analysis_results['tracknet_detections'] += 1
                    self.tracknet_predictions.append(tracknet_pred)
            except Exception as e:
                logger.error(f"TrackNet error: {e}")
        
        # 2. YOLO ball prediction
        if self.yolo_ball_model:
            try:
                yolo_pred = self.yolo_ball_model.detect_ball(frame)
                if yolo_pred:
                    self.analysis_results['yolo_ball_detections'] += 1
                    self.yolo_predictions.append(yolo_pred)
            except Exception as e:
                logger.error(f"YOLO error: {e}")
        
        # 3. Combine predictions
        return self._combine_predictions(tracknet_pred, yolo_pred)
    
    def _combine_predictions(self, tracknet_pred: Optional[Dict], yolo_pred: Optional[Dict]) -> Optional[Dict]:
        """Combine predictions from both models"""
        if not tracknet_pred and not yolo_pred:
            return None
        
        if tracknet_pred and not yolo_pred:
            return tracknet_pred
        
        if yolo_pred and not tracknet_pred:
            return yolo_pred
        
        # Both predictions exist - combine them
        tracknet_pos = tracknet_pred['position']
        yolo_pos = yolo_pred['position']
        tracknet_conf = tracknet_pred['confidence']
        yolo_conf = yolo_pred['confidence']
        
        # Weighted average based on confidence
        total_conf = tracknet_conf + yolo_conf
        if total_conf > 0:
            combined_x = (tracknet_pos[0] * tracknet_conf + yolo_pos[0] * yolo_conf) / total_conf
            combined_y = (tracknet_pos[1] * tracknet_conf + yolo_pos[1] * yolo_conf) / total_conf
            combined_conf = (tracknet_conf + yolo_conf) / 2
            
            # Apply velocity-based filtering
            if len(self.ball_positions) > 0:
                last_pos = self.ball_positions[-1]['position']
                distance = np.sqrt((combined_x - last_pos[0])**2 + (combined_y - last_pos[1])**2)
                max_velocity = self.config['ball_tracking']['max_velocity']
                
                if distance > max_velocity:
                    # Use the prediction closer to last position
                    tracknet_dist = np.sqrt((tracknet_pos[0] - last_pos[0])**2 + (tracknet_pos[1] - last_pos[1])**2)
                    yolo_dist = np.sqrt((yolo_pos[0] - last_pos[0])**2 + (yolo_pos[1] - last_pos[1])**2)
                    
                    if tracknet_dist < yolo_dist:
                        return tracknet_pred
                    else:
                        return yolo_pred
            
            return {
                'position': [int(combined_x), int(combined_y)],
                'confidence': combined_conf,
                'source': 'combined'
            }
        
        return None
    
    def _calculate_velocity(self, pos1: Dict, pos2: Dict) -> Tuple[float, float]:
        """Calculate velocity between two positions"""
        x1, y1 = pos1['position']
        x2, y2 = pos2['position']
        return (x2 - x1, y2 - y1)
    
    def _draw_ball_tracking(self, frame: np.ndarray, ball_pred: Dict) -> np.ndarray:
        """Draw ball tracking visualization with single color (no trajectory line)"""
        x, y = ball_pred['position']
        conf = ball_pred['confidence']
        
        # Use single color for ball detection (orange)
        color = (0, 165, 255)  # BGR format - orange
        
        # Draw ball position
        cv2.circle(frame, (x, y), 8, color, -1)
        cv2.circle(frame, (x, y), 12, color, 2)
        
        # Draw confidence
        cv2.putText(frame, f"{conf:.2f}", (x + 15, y - 15), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Draw velocity vector
        if len(self.ball_velocities) > 0:
            vx, vy = self.ball_velocities[-1]
            if abs(vx) > 1 or abs(vy) > 1:  # Only draw if there's significant movement
                end_x = int(x + vx * 2)  # Scale for visibility
                end_y = int(y + vy * 2)
                cv2.arrowedLine(frame, (x, y), (end_x, end_y), (0, 255, 0), 2)
        
        return frame
    
    def _add_frame_info(self, frame: np.ndarray):
        """Add comprehensive frame information overlay"""
        # Frame counter
        cv2.putText(frame, f"Frame: {self.analysis_results['total_frames']}", 
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Statistics
        stats_text = [
            f"Players: {self.analysis_results['players_detected']}",
            f"Poses: {self.analysis_results['poses_estimated']}",
            f"Bounces: {self.analysis_results['bounces_detected']}",
            f"TrackNet: {self.analysis_results['tracknet_detections']}",
            f"YOLO Ball: {self.analysis_results['yolo_ball_detections']}",
            f"Combined Ball: {self.analysis_results['combined_ball_detections']}",
            f"Court Detections: {self.analysis_results['court_detections']}",  # NEW
            f"Court Keypoints: {self.analysis_results['keypoints_detected']}"  # NEW
        ]
        
        y_offset = 60
        for stat in stats_text:
            cv2.putText(frame, stat, (10, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            y_offset += 25
        
        # Instructions
        instructions = [
            "Press 'q' to quit",
            "Press 'p' to pause",
            "Press 's' to save frame"
        ]
        
        y_offset = frame.shape[0] - 100
        for instruction in instructions:
            cv2.putText(frame, instruction, (10, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            y_offset += 20
    
    def _print_analysis_summary(self):
        """Print comprehensive analysis summary"""
        print("\n" + "="*60)
        print("SUPER ADVANCED TENNIS ANALYSIS ENGINE - SUMMARY")
        print("="*60)
        print(f"Total frames processed: {self.analysis_results['total_frames']}")
        print(f"Total players detected: {self.analysis_results['players_detected']}")
        print(f"Total poses estimated: {self.analysis_results['poses_estimated']}")
        print(f"Total bounces detected: {self.analysis_results['bounces_detected']}")
        print(f"TrackNet ball detections: {self.analysis_results['tracknet_detections']}")
        print(f"YOLO ball detections: {self.analysis_results['yolo_ball_detections']}")
        print(f"Combined ball detections: {self.analysis_results['combined_ball_detections']}")
        print(f"Court detections: {self.analysis_results['court_detections']}")  # NEW
        print(f"Court keypoints detected: {self.analysis_results['keypoints_detected']}")  # NEW
        
        if self.analysis_results['processing_times']:
            avg_time = np.mean(self.analysis_results['processing_times'])
            min_time = np.min(self.analysis_results['processing_times'])
            max_time = np.max(self.analysis_results['processing_times'])
            print(f"Processing time - Avg: {avg_time:.3f}s, Min: {min_time:.3f}s, Max: {max_time:.3f}s")
        
        total_time = time.time() - self.start_time
        print(f"Total analysis time: {total_time:.2f}s")
        print("="*60)


class PlayerDetector:
    """YOLOv8-based player detection"""
    
    def __init__(self, model_path: str, config: Dict[str, Any]):
        self.config = config
        try:
            from ultralytics import YOLO
            self.model = YOLO(model_path)
            logger.info(f"Player detector initialized with {model_path}")
        except ImportError:
            logger.error("Ultralytics not installed. Install with: pip install ultralytics")
            self.model = None
        except Exception as e:
            logger.error(f"Error loading player detection model: {e}")
            self.model = None
    
    def detect_players(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        if not self.model:
            return []
        
        try:
            results = self.model(
                frame,
                conf=self.config.get('conf_threshold', 0.5),
                iou=self.config.get('iou_threshold', 0.45),
                max_det=self.config.get('max_det', 10),
                verbose=False
            )
            
            detections = []
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        conf = box.conf[0].cpu().numpy()
                        cls = int(box.cls[0].cpu().numpy())
                        
                        # Calculate bounding box dimensions
                        bbox_width = x2 - x1
                        bbox_height = y2 - y1
                        bbox_area = bbox_width * bbox_height
                        
                        # Log detection details
                        logger.debug(f"Detection: class={cls}, conf={conf:.2f}, bbox=[{int(x1)}, {int(y1)}, {int(x2)}, {int(y2)}], area={bbox_area}")
                        
                        # Only keep class 1 (players), filter out class 0 (balls)
                        if cls == 1:  # Player class
                            detection = {
                                'bbox': [int(x1), int(y1), int(x2), int(y2)],
                                'confidence': float(conf),
                                'class': cls,
                                'center': [int((x1 + x2) / 2), int((y1 + y2) / 2)]
                            }
                            detections.append(detection)
                            logger.debug(f"Added player detection (class {cls})")
                        else:
                            logger.debug(f"Filtered out ball detection (class {cls})")
            
            logger.info(f"Detected {len(detections)} players")
            if detections:
                logger.info(f"First detection: class={detections[0]['class']}, bbox={detections[0]['bbox']}")
            return detections
        except Exception as e:
            logger.error(f"Player detection error: {e}")
            return []
    
    def draw_detections(self, frame: np.ndarray, detections: List[Dict[str, Any]]) -> np.ndarray:
        frame_copy = frame.copy()
        
        for detection in detections:
            x1, y1, x2, y2 = detection['bbox']
            conf = detection['confidence']
            cls = detection['class']
            
            # Color for players (green)
            color = (0, 255, 0)
            
            # Draw bounding box
            cv2.rectangle(frame_copy, (x1, y1), (x2, y2), color, 2)
            
            # Draw label
            label = f"Player: {conf:.2f}"
            cv2.putText(frame_copy, label, (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        return frame_copy


class PoseEstimator:
    """YOLOv8-pose-based pose estimation"""
    
    def __init__(self, model_path: str, config: Dict[str, Any]):
        self.config = config
        try:
            from ultralytics import YOLO
            self.model = YOLO(model_path)
            logger.info(f"Pose estimator initialized with {model_path}")
        except ImportError:
            logger.error("Ultralytics not installed. Install with: pip install ultralytics")
            self.model = None
        except Exception as e:
            logger.error(f"Error loading pose estimation model: {e}")
            self.model = None
    
    def estimate_poses(self, frame: np.ndarray, player_detections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not self.model:
            return []
        
        poses = []
        try:
            for player_detection in player_detections:
                x1, y1, x2, y2 = player_detection['bbox']
                
                # Extract player region with padding
                padding = 20
                x1_pad = max(0, x1 - padding)
                y1_pad = max(0, y1 - padding)
                x2_pad = min(frame.shape[1], x2 + padding)
                y2_pad = min(frame.shape[0], y2 + padding)
                
                player_roi = frame[y1_pad:y2_pad, x1_pad:x2_pad]
                
                if player_roi.size == 0:
                    continue
                
                # Run pose estimation
                results = self.model(
                    player_roi,
                    conf=self.config.get('conf_threshold', 0.3),
                    iou=self.config.get('iou_threshold', 0.45),
                    max_det=1,
                    verbose=False
                )
                
                for result in results:
                    try:
                        keypoints = result.keypoints
                        if keypoints is not None and hasattr(keypoints, 'data') and len(keypoints.data) > 0:
                            kpts = keypoints.data[0].cpu().numpy()
                            if len(kpts) > 0:
                                # Adjust keypoint coordinates back to full frame coordinates
                                adjusted_kpts = kpts.copy()
                                adjusted_kpts[:, 0] += x1_pad  # Add x offset
                                adjusted_kpts[:, 1] += y1_pad  # Add y offset
                                
                                pose = {
                                    'keypoints': adjusted_kpts,
                                    'bbox': [x1, y1, x2, y2],
                                    'confidence': float(result.boxes.conf[0].cpu().numpy()) if result.boxes is not None and len(result.boxes.conf) > 0 else 0.0
                                }
                                poses.append(pose)
                    except Exception as e:
                        logger.debug(f"Error processing pose result: {e}")
                        continue
            
            return poses
        except Exception as e:
            logger.error(f"Pose estimation error: {e}")
            return []
    
    def draw_poses(self, frame: np.ndarray, poses: List[Dict[str, Any]]) -> np.ndarray:
        frame_copy = frame.copy()
        
        # COCO keypoint connections for skeleton drawing
        skeleton = [
            (0, 1), (0, 2), (1, 3), (2, 4),  # Head
            (5, 6), (5, 7), (6, 8), (7, 9),  # Arms
            (5, 11), (6, 12), (11, 12),      # Torso
            (11, 13), (13, 15), (12, 14), (14, 16)  # Legs
        ]
        
        for pose in poses:
            keypoints = pose['keypoints']
            bbox = pose['bbox']
            
            # Draw keypoints
            for i, (x, y, conf) in enumerate(keypoints):
                if conf > 0.3:  # Only draw confident keypoints
                    cv2.circle(frame_copy, (int(x), int(y)), 3, (0, 255, 255), -1)
            
            # Draw skeleton
            for start_idx, end_idx in skeleton:
                if (start_idx < len(keypoints) and end_idx < len(keypoints) and 
                    keypoints[start_idx][2] > 0.3 and keypoints[end_idx][2] > 0.3):
                    start_point = (int(keypoints[start_idx][0]), int(keypoints[start_idx][1]))
                    end_point = (int(keypoints[end_idx][0]), int(keypoints[end_idx][1]))
                    cv2.line(frame_copy, start_point, end_point, (255, 255, 0), 2)
            
            # Draw pose confidence
            x1, y1, x2, y2 = bbox
            cv2.putText(frame_copy, f"Pose: {pose['confidence']:.2f}", 
                       (x1, y2+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
        
        return frame_copy


class BounceDetector:
    """CatBoost-based ball bounce detection"""
    
    def __init__(self, model_path: str):
        try:
            import catboost as cb
            self.model = cb.CatBoostClassifier()
            self.model.load_model(model_path)
            logger.info(f"Bounce detector initialized with {model_path}")
        except ImportError:
            logger.error("CatBoost not installed. Install with: pip install catboost")
            self.model = None
        except Exception as e:
            logger.error(f"Error loading bounce detection model: {e}")
            self.model = None
    
    def detect_bounce(self, frame: np.ndarray) -> float:
        if not self.model:
            return 0.0
        
        try:
            # Simple feature extraction (you might want to enhance this)
            # Convert to grayscale and resize for feature extraction
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            resized = cv2.resize(gray, (64, 64))
            
            # Extract basic features (histogram, edges, etc.)
            features = []
            
            # Histogram features
            hist = cv2.calcHist([resized], [0], None, [16], [0, 256])
            hist = hist.flatten() / hist.sum()  # Normalize
            features.extend(hist)
            
            # Edge features
            edges = cv2.Canny(resized, 50, 150)
            edge_density = np.sum(edges > 0) / (64 * 64)
            features.append(edge_density)
            
            # Texture features (simple gradient)
            grad_x = cv2.Sobel(resized, cv2.CV_64F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(resized, cv2.CV_64F, 0, 1, ksize=3)
            gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
            avg_gradient = np.mean(gradient_magnitude)
            features.append(avg_gradient)
            
            # Pad features to expected length (adjust based on your model)
            while len(features) < 20:  # Assuming model expects 20 features
                features.append(0.0)
            features = features[:20]  # Truncate if too long
            
            # Make prediction
            prediction = self.model.predict_proba([features])[0]
            return prediction[1] if len(prediction) > 1 else prediction[0]
            
        except Exception as e:
            logger.error(f"Bounce detection error: {e}")
            return 0.0


class TrackNetDetector:
    """TrackNet-based ball detection"""
    
    def __init__(self, model_path: str, config: Dict[str, Any]):
        self.config = config
        self.device = 'cpu'
        self.frame_buffer = deque(maxlen=3)  # TrackNet needs 3 consecutive frames
        
        try:
            import torch
            import sys
            sys.path.append('TrackNet')
            from TrackNet.model import BallTrackerNet
            
            # Initialize the TrackNet model
            self.model = BallTrackerNet(out_channels=256).to(self.device)
            
            # Load the pretrained weights
            checkpoint = torch.load(model_path, map_location=self.device)
            
            # Handle different checkpoint formats
            if 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            elif 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            else:
                state_dict = checkpoint
            
            self.model.load_state_dict(state_dict)
            self.model.eval()
            logger.info(f"TrackNet model loaded from {model_path}")
            
        except Exception as e:
            logger.error(f"Error loading TrackNet model: {e}")
            self.model = None
    
    def detect_ball(self, frame: np.ndarray) -> Optional[Dict[str, Any]]:
        """Detect ball using TrackNet"""
        if not self.model:
            return None
        
        # Add frame to buffer
        self.frame_buffer.append(frame)
        
        # Need at least 3 frames for TrackNet
        if len(self.frame_buffer) < 3:
            return None
        
        try:
            import torch
            import sys
            sys.path.append('TrackNet')
            from general import postprocess
            
            # Get the 3 consecutive frames
            current_frame = self.frame_buffer[-1]
            prev_frame = self.frame_buffer[-2]
            prev_prev_frame = self.frame_buffer[-3]
            
            # Preprocess frames for TrackNet
            processed_input = self._preprocess_frames(current_frame, prev_frame, prev_prev_frame)
            
            # Run inference
            with torch.no_grad():
                output = self.model(processed_input, testing=True)
                output = output.argmax(dim=1).detach().cpu().numpy()
            
            # Postprocess to get ball position
            ball_pos = self._postprocess_output(output[0], frame.shape)
            
            if ball_pos:
                return {
                    'position': ball_pos,
                    'confidence': 0.8,  # TrackNet doesn't provide confidence, use default
                    'source': 'tracknet'
                }
            
            return None
            
        except Exception as e:
            logger.error(f"TrackNet detection error: {e}")
            return None
    
    def _preprocess_frames(self, frame1: np.ndarray, frame2: np.ndarray, frame3: np.ndarray):
        """Preprocess 3 consecutive frames for TrackNet input"""
        import torch
        
        # Resize frames to TrackNet input size
        height, width = self.config['input_height'], self.config['input_width']
        frame1_resized = cv2.resize(frame1, (width, height))
        frame2_resized = cv2.resize(frame2, (width, height))
        frame3_resized = cv2.resize(frame3, (width, height))
        
        # Convert to RGB and normalize
        frame1_rgb = cv2.cvtColor(frame1_resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        frame2_rgb = cv2.cvtColor(frame2_resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        frame3_rgb = cv2.cvtColor(frame3_resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        
        # Concatenate frames along channel dimension (9 channels total)
        combined = np.concatenate((frame1_rgb, frame2_rgb, frame3_rgb), axis=2)
        
        # Convert to tensor format (C, H, W) and add batch dimension
        tensor = torch.from_numpy(combined).permute(2, 0, 1).unsqueeze(0).float()
        return tensor
    
    def _postprocess_output(self, output: np.ndarray, original_shape: Tuple[int, int, int]) -> Optional[List[int]]:
        """Postprocess TrackNet output to get ball position"""
        import sys
        sys.path.append('TrackNet')
        from general import postprocess
        
        try:
            # Use the original postprocess function
            x, y = postprocess(output)
            
            if x is not None and y is not None:
                # Convert to original image coordinates
                orig_h, orig_w = original_shape[:2]
                
                # The postprocess function returns coordinates scaled by 2, but we need to scale to original image size
                # TrackNet was trained on 360x640, but postprocess scales by 2, so effective output is 720x1280
                scale_h = orig_h / 720  # 360 * 2 = 720
                scale_w = orig_w / 1280  # 640 * 2 = 1280
                
                x_scaled = int(x * scale_w)
                y_scaled = int(y * scale_h)
                
                # Ensure coordinates are within bounds
                x_scaled = max(0, min(x_scaled, orig_w - 1))
                y_scaled = max(0, min(y_scaled, orig_h - 1))
                
                logger.info(f"TrackNet ball detection: pos=({x_scaled}, {y_scaled})")
                return [x_scaled, y_scaled]
            
            return None
            
        except Exception as e:
            logger.error(f"Postprocess error: {e}")
            return None


class YOLOBallDetector:
    """YOLOv8-based ball detection"""
    
    def __init__(self, model_path: str, config: Dict[str, Any]):
        self.config = config
        try:
            from ultralytics import YOLO
            self.model = YOLO(model_path)
            logger.info(f"YOLO ball detector initialized with {model_path}")
        except ImportError:
            logger.error("Ultralytics not installed. Install with: pip install ultralytics")
            self.model = None
        except Exception as e:
            logger.error(f"Error loading YOLO model: {e}")
            self.model = None
    
    def detect_ball(self, frame: np.ndarray) -> Optional[Dict[str, Any]]:
        """Detect ball using YOLO"""
        if not self.model:
            return None
        
        try:
            results = self.model(
                frame,
                conf=self.config.get('conf_threshold', 0.1),
                iou=self.config.get('iou_threshold', 0.45),
                max_det=self.config.get('max_det', 10),
                verbose=False
            )
            
            # Debug: Log all detections
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    logger.debug(f"YOLO found {len(boxes)} detections")
                    for box in boxes:
                        cls = int(box.cls[0].cpu().numpy())
                        conf = float(box.conf[0].cpu().numpy())
                        logger.debug(f"YOLO detection: class={cls}, conf={conf:.3f}")
                        
                        # Look for ball class (class 0) - but also try class 1 if 0 doesn't work
                        if (cls == 0 or cls == 1) and conf > self.config.get('conf_threshold', 0.1):
                            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                            center_x = int((x1 + x2) / 2)
                            center_y = int((y1 + y2) / 2)
                            
                            # Calculate bounding box area to filter out large detections (likely players)
                            bbox_area = (x2 - x1) * (y2 - y1)
                            if bbox_area < 5000:  # Only small detections (likely balls)
                                logger.info(f"YOLO ball detection: class={cls}, conf={conf:.3f}, pos=({center_x}, {center_y}), area={bbox_area}")
                                return {
                                    'position': [center_x, center_y],
                                    'confidence': conf,
                                    'source': 'yolo',
                                    'bbox': [int(x1), int(y1), int(x2), int(y2)]
                                }
            
            return None
            
        except Exception as e:
            logger.error(f"YOLO detection error: {e}")
            return None


# NEW: Court Detection Class (integrated from court_demo.py)
class CourtDetector:
    """Tennis court detection system using deep learning with geometric validation"""
    
    def __init__(self, model_path: str, config: Dict[str, Any]):
        """Initialize the court detection system"""
        self.config = config
        self.device = 'cuda' if hasattr(torch.cuda, 'isAvailable') and torch.cuda.isAvailable() else 'cpu'
        
        # Court detection state
        self.court_keypoints = []
        self.court_confidence = []
        
        # Colinearity-based quality assessment for tennis court lines
        self.court_line_groups = {
            # Horizontal lines (top to bottom)
            'top_horizontal': [0, 4, 6, 1],      # Top baseline + service lines
            'bottom_horizontal': [2, 5, 7, 3],   # Bottom baseline + service lines
            'top_service': [8, 12, 9],            # Top service line + net
            'bottom_service': [10, 13, 11],       # Bottom service line + net
            
            # Vertical lines (left to right)
            'left_vertical': [5, 10, 8, 4],      # Left side lines
            'right_vertical': [6, 9, 11, 7],     # Right side lines
        }
        
        # Parallel line pairs for additional validation
        self.parallel_line_pairs = [
            # Endlines should be parallel to service lines
            ('endline_top', [1, 3], 'service_line_right', [6, 9, 11, 7]),      # Top endline || right service line
            ('endline_bottom', [2, 0], 'service_line_left', [5, 10, 8, 4]),     # Bottom endline || left service line
            
            # All horizontal lines should be parallel to each other
            ('baseline_top', [0, 1], 'baseline_bottom', [2, 3]),                # Top baseline || bottom baseline
            ('baseline_top', [0, 1], 'top_service', [8, 9]),                    # Top baseline || top service line
            ('baseline_top', [0, 1], 'bottom_service', [10, 11]),               # Top baseline || bottom service line
            ('baseline_bottom', [2, 3], 'top_service', [8, 9]),                 # Bottom baseline || top service line
            ('baseline_bottom', [2, 3], 'bottom_service', [10, 11]),            # Bottom baseline || bottom service line
            ('top_service', [8, 9], 'bottom_service', [10, 11]),                # Top service line || bottom service line
            
            # All vertical lines should be parallel to each other
            ('left_vertical', [5, 10, 8, 4], 'right_vertical', [6, 9, 11, 7]), # Left side || right side
            ('left_vertical', [5, 10, 8, 4], 'center_service', [8, 12, 10]),   # Left side || center service
            ('right_vertical', [6, 9, 11, 7], 'center_service', [8, 12, 10]),  # Right side || center service
        ]
        
        # Soft-locked keypoints - only replace when better ones come along
        self.best_keypoints = {}  # keypoint_id -> (x, y, best_score)
        self.quality_threshold = 0.1  # Only consider keypoints with score <= 0.1 as "good enough"
        self.min_history_frames = 3   # Minimum frames before considering a keypoint stable
        
        # Temporal smoothing for static camera
        self.keypoint_history = {}  # Store keypoint positions over time
        self.keypoint_confidence = {}  # Store confidence scores
        self.smoothed_keypoints = {}  # Final smoothed keypoints
        self.history_length = 30  # Number of frames to average over
        self.min_confidence_threshold = 0.3  # Minimum confidence to consider a detection
        
        # Best position tracking - preserve the best predictions we've ever seen
        self.best_positions = {}  # Store the BEST individual prediction ever seen for each keypoint
        self.best_scores = {}  # Store the quality scores for the best positions
        
        # Initialize model
        self.model = None
        self._initialize_model(model_path)
    
    def _initialize_model(self, model_path: str):
        """Initialize the court detection model"""
        try:
            if not Path(model_path).exists():
                logger.error(f"Court detection model not found: {model_path}")
                return
            
            # Initialize model
            self.model = BallTrackerNet(out_channels=15)  # 14 keypoints + 1 center point
            
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            self.model.to(self.device)
            self.model.eval()
            
            logger.info(f"Court detection model loaded from {model_path}")
            logger.info(f"Using device: {self.device}")
            
        except Exception as e:
            logger.error(f"Error initializing court detection model: {e}")
            self.model = None
    
    def detect_court_in_frame(self, frame: np.ndarray) -> List[Tuple]:
        """Detect court keypoints in a single frame - COMPLETE VERSION from court_demo.py"""
        if not self.model:
            return []
        
        try:
            # Get model input dimensions
            input_width = self.config['input_width']
            input_height = self.config['input_height']
            
            # Resize frame for model input
            img_resized = cv2.resize(frame, (input_width, input_height))
            
            # Preprocess input
            inp = (img_resized.astype(np.float32) / 255.)
            inp = torch.tensor(inp).permute(2, 0, 1)  # Convert to (C, H, W) format
            inp = inp.unsqueeze(0).to(self.device)
            
            # Run inference
            with torch.no_grad():
                out = self.model(inp.float())
                
                if isinstance(out, tuple):
                    out = out[0]
                
                # Take the first element to remove batch dimension
                out = out[0]
                pred = F.sigmoid(out).detach().cpu().numpy()
            
            # Extract keypoints from heatmaps
            points = []
            for kps_num in range(14):  # 14 court keypoints
                heatmap = (pred[kps_num] * 255).astype(np.uint8)
                
                # Postprocess heatmap to get keypoint coordinates
                try:
                    x_pred, y_pred = postprocess(heatmap, low_thresh=170, max_radius=25)
                except Exception as e:
                    x_pred, y_pred = None, None
                
                # Refine keypoints if enabled
                if (self.config['use_refine_kps'] and 
                    kps_num not in [8, 12, 9] and x_pred and y_pred):
                    try:
                        x_pred, y_pred = refine_kps(frame, int(y_pred), int(x_pred))
                    except Exception as e:
                        pass
                
                # Scale coordinates back to original video resolution
                if x_pred is not None and y_pred is not None:
                    # Postprocess already scales by 2, so coordinates are at 720x1280
                    scale_x = frame.shape[1] / (input_width * 2)
                    scale_y = frame.shape[0] / (input_height * 2)
                    x_scaled = int(x_pred * scale_x)
                    y_scaled = int(y_pred * scale_y)
                    points.append((x_scaled, y_scaled))
                else:
                    points.append((x_pred, y_pred))
            
            # Now apply the sophisticated processing from court_demo.py
            if len(points) > 0:
                # Assess keypoint quality using colinearity constraints
                colinearity_scores = self._assess_colinearity_quality(points)
                
                # Assess parallelism for additional validation
                parallelism_scores = self._assess_parallelism_quality(points)
                
                # Combine colinearity and parallelism scores
                combined_scores = self._combine_quality_scores(colinearity_scores, parallelism_scores)
                
                # Update best positions with new predictions if they're better
                self._update_best_positions(points, combined_scores)
                
                # Update best keypoints based on quality scores (soft-lock system)
                self._update_best_keypoints(points, combined_scores)
                
                # Apply temporal smoothing to populate keypoint history
                points_with_confidence = []
                for i, (x, y) in enumerate(points):
                    quality_score = combined_scores.get(i, 1.0)
                    # Convert quality score (0=perfect, 1=worst) to confidence (1=perfect, 0=worst)
                    confidence = max(0.0, 1.0 - quality_score)
                    points_with_confidence.append((x, y, confidence))
                
                self._apply_temporal_smoothing(points_with_confidence)
                
                # Apply best keypoints and temporal smoothing for others
                smoothed_points = self._apply_best_keypoints(points, combined_scores)
                
                # Apply homography if enabled (using smoothed points)
                if self.config['use_homography']:
                    try:
                        matrix_trans = get_trans_matrix(smoothed_points)
                        if matrix_trans is not None:
                            points = cv2.perspectiveTransform(refer_kps, matrix_trans)
                            points = [np.squeeze(x) for x in points]
                        else:
                            points = smoothed_points
                    except Exception as e:
                        points = smoothed_points
                else:
                    points = smoothed_points
            
            return points
            
        except Exception as e:
            logger.error(f"Court detection error: {e}")
            return []
    
    def _assess_colinearity_quality(self, points: List[Tuple]) -> Dict[int, float]:
        """Assess keypoint quality based on colinearity with other points on the same line"""
        colinearity_scores = {}
        
        for line_name, point_indices in self.court_line_groups.items():
            # Get valid points for this line
            valid_points = []
            for idx in point_indices:
                if idx < len(points) and points[idx][0] is not None and points[idx][1] is not None:
                    valid_points.append((idx, points[idx]))
            
            if len(valid_points) < 2:
                continue  # Need at least 2 points to assess colinearity
            
            # For lines with only 2 points, we can't calculate colinearity directly
            if len(valid_points) == 2:
                # Store these points for potential use in other validations
                for point_idx, (x, y) in valid_points:
                    if point_idx not in colinearity_scores:
                        colinearity_scores[point_idx] = 0.8  # Moderate error for 2-point lines
                continue
            
            # For lines with 3+ points, we can calculate colinearity
            if len(valid_points) >= 3:
                
                # Calculate colinearity for each point in this line
                for i, (point_idx, (x, y)) in enumerate(valid_points):
                    # Find other points on the same line to test colinearity
                    other_points = [(other_idx, (ox, oy)) for j, (other_idx, (ox, oy)) in enumerate(valid_points) if i != j]
                    
                    if len(other_points) < 2:
                        continue
                    
                    # Test colinearity with different combinations of other points
                    colinearity_measures = []
                    
                    for j in range(len(other_points)):
                        for k in range(j + 1, len(other_points)):
                            p1_idx, (x1, y1) = other_points[j]
                            p2_idx, (x2, y2) = other_points[k]
                            
                            # Calculate colinearity using area of triangle method
                            # If three points are colinear, the area of the triangle they form is 0
                            area = abs((x * (y1 - y2) + x1 * (y2 - y) + x2 * (y - y1)) / 2)
                            
                            # Normalize by the distance between the two reference points
                            distance = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
                            if distance > 0:
                                normalized_area = area / distance
                                # LOWER scores are better - this is the error we want to minimize
                                colinearity_measure = normalized_area / 100  # Scale factor for reasonable range
                                colinearity_measures.append(colinearity_measure)
                    
                    if colinearity_measures:
                        # Average colinearity with all combinations
                        avg_colinearity = np.mean(colinearity_measures)
                        
                        # Update score if this line gives a better result
                        if point_idx not in colinearity_scores or avg_colinearity < colinearity_scores[point_idx]:
                            colinearity_scores[point_idx] = avg_colinearity
        
        # Fill in missing scores with default values
        for i in range(len(points)):
            if i not in colinearity_scores:
                colinearity_scores[i] = 1.0  # Default score (high error = bad)
        
        return colinearity_scores
    
    def _assess_parallelism_quality(self, points: List[Tuple]) -> Dict[int, float]:
        """Assess keypoint quality based on parallelism with other lines"""
        parallelism_scores = {}
        
        for line_name, line1_points, parallel_line_name, line2_points in self.parallel_line_pairs:
            # Get valid points for both lines
            valid_line1 = [(idx, points[idx]) for idx in line1_points if idx < len(points) and points[idx][0] is not None and points[idx][1] is not None]
            valid_line2 = [(idx, points[idx]) for idx in line2_points if idx < len(points) and points[idx][0] is not None and points[idx][1] is not None]
            
            if len(valid_line1) < 2 or len(valid_line2) < 2:
                continue
            
            # Calculate direction vector for line1
            if len(valid_line1) >= 2:
                start_idx, (start_x, start_y) = valid_line1[0]
                end_idx, (end_x, end_y) = valid_line1[-1]
                line1_vector = np.array([end_x - start_x, end_y - start_y])
                line1_length = np.linalg.norm(line1_vector)
                
                if line1_length > 0:
                    line1_unit = line1_vector / line1_length
                    
                    # Calculate parallelism with line2
                    line2_vectors = []
                    for i in range(len(valid_line2) - 1):
                        for j in range(i + 1, len(valid_line2)):
                            p1_idx, (x1, y1) = valid_line2[i]
                            p2_idx, (x2, y2) = valid_line2[j]
                            line2_vector = np.array([x2 - x1, y2 - y1])
                            line2_length = np.linalg.norm(line2_vector)
                            if line2_length > 0:
                                line2_unit = line2_vector / line2_length
                                # Cross product gives area of parallelogram formed by vectors
                                # Parallel lines have cross product = 0
                                # We want to minimize this to 0
                                cross_product = np.cross(line1_unit, line2_unit)
                                # Convert to error: 0 = perfect parallel, higher = less parallel
                                parallelism_error = np.linalg.norm(cross_product)
                                line2_vectors.append(parallelism_error)
                    
                    if line2_vectors:
                        avg_parallelism = np.mean(line2_vectors)
                        
                        # Assign scores to all points in line1
                        for point_idx, _ in valid_line1:
                            if point_idx not in parallelism_scores:
                                parallelism_scores[point_idx] = []
                            parallelism_scores[point_idx].append(avg_parallelism)
        
        # Average all parallelism scores for each point and fill in missing scores
        final_parallelism_scores = {}
        for i in range(len(points)):
            if i in parallelism_scores and parallelism_scores[i]:
                # Average all parallelism scores for this point
                final_parallelism_scores[i] = np.mean(parallelism_scores[i])
            else:
                final_parallelism_scores[i] = 1.0  # Default score (high error = bad)
        
        return final_parallelism_scores
    
    def _combine_quality_scores(self, colinearity_scores: Dict[int, float], parallelism_scores: Dict[int, float]) -> Dict[int, float]:
        """Combine colinearity and parallelism scores into final quality scores"""
        combined_scores = {}
        
        for i in range(max(len(colinearity_scores), len(parallelism_scores))):
            col_score = colinearity_scores.get(i, 0.5)
            par_score = parallelism_scores.get(i, 0.5)
            
            # Weight colinearity more heavily than parallelism
            combined_score = col_score * 0.7 + par_score * 0.3
            combined_scores[i] = combined_score
        
        return combined_scores
    
    def _update_best_positions(self, points: List[Tuple], combined_scores: Dict[int, float]):
        """Update best positions with new predictions if they're better"""
        for point_idx, (x, y) in enumerate(points):
            if x is not None and y is not None:
                current_score = combined_scores.get(point_idx, 1.0)
                
                # Check if this is the best prediction we've ever seen for this keypoint
                if (point_idx not in self.best_scores or 
                    current_score < self.best_scores[point_idx]):
                    
                    # This is a new best prediction!
                    self.best_positions[point_idx] = (x, y)
                    self.best_scores[point_idx] = current_score
    
    def _update_best_keypoints(self, points: List[Tuple], quality_scores: Dict[int, float]):
        """Update best keypoints - only replace when better ones come along (soft-lock)"""
        for kps_num, (x_pred, y_pred) in enumerate(points):
            if x_pred is not None and y_pred is not None:
                quality_score = quality_scores.get(kps_num, 1.0)
                
                if kps_num not in self.keypoint_history:
                    self.keypoint_history[kps_num] = []
                    self.keypoint_confidence[kps_num] = []
                
                # Add current detection to history
                self.keypoint_history[kps_num].append((x_pred, y_pred))
                self.keypoint_confidence[kps_num].append(quality_score)
                
                # Keep only recent history
                if len(self.keypoint_history[kps_num]) > self.history_length:
                    self.keypoint_history[kps_num].pop(0)
                    self.keypoint_confidence[kps_num].pop(0)
                
                # Check if this is a better keypoint than what we have
                if kps_num not in self.best_keypoints:
                    # First time seeing this keypoint
                    if quality_score <= self.quality_threshold:
                        self.best_keypoints[kps_num] = (x_pred, y_pred, quality_score)
                else:
                    # Compare with existing best
                    best_x, best_y, best_score = self.best_keypoints[kps_num]
                    
                    if quality_score < best_score:
                        # This is better! Replace it
                        self.best_keypoints[kps_num] = (x_pred, y_pred, quality_score)
    
    def _apply_best_keypoints(self, points: List[Tuple], quality_scores: Dict[int, float]) -> List[Tuple]:
        """Apply best keypoints and temporal smoothing for others"""
        final_points = []
        
        for kps_num, (x_pred, y_pred) in enumerate(points):
            if kps_num in self.best_keypoints:
                # Use best position - this is our "soft-locked" position
                best_x, best_y, best_score = self.best_keypoints[kps_num]
                final_points.append((best_x, best_y))
            else:
                # Apply temporal smoothing for non-best keypoints
                if len(self.keypoint_history.get(kps_num, [])) > 0:
                    # Weighted average based on confidence
                    total_weight = sum(self.keypoint_confidence.get(kps_num, []))
                    if total_weight > 0:
                        weighted_x = sum(x * w for (x, y), w in zip(self.keypoint_history[kps_num], self.keypoint_confidence[kps_num])) / total_weight
                        weighted_y = sum(y * w for (x, y), w in zip(self.keypoint_history[kps_num], self.keypoint_confidence[kps_num])) / total_weight
                        
                        # Apply outlier rejection
                        if len(self.keypoint_history[kps_num]) >= 5:
                            distances = [np.sqrt((x - weighted_x)**2 + (y - weighted_y)**2) 
                                   for (x, y) in self.keypoint_history[kps_num]]
                            mean_distance = np.mean(distances)
                            std_distance = np.std(distances)
                            
                            # Remove outliers (points too far from mean)
                            filtered_positions = []
                            filtered_weights = []
                            for i, ((x, y), w) in enumerate(zip(self.keypoint_history[kps_num], self.keypoint_confidence[kps_num])):
                                if distances[i] <= mean_distance + 2 * std_distance:  # 2 sigma rule
                                    filtered_positions.append((x, y))
                                    filtered_weights.append(w)
                            
                            if len(filtered_positions) > 0:
                                total_weight = sum(filtered_weights)
                                weighted_x = sum(x * w for (x, y), w in zip(filtered_positions, filtered_weights)) / total_weight
                                weighted_y = sum(y * w for (x, y), w in zip(filtered_positions, filtered_weights)) / total_weight
                        
                        final_points.append((int(weighted_x), int(weighted_y)))
                    else:
                        final_points.append((x_pred, y_pred))
                else:
                    final_points.append((x_pred, y_pred))
        
        return final_points
    
    def _apply_temporal_smoothing(self, points_with_confidence: List[Tuple]) -> List[Tuple]:
        """Apply temporal smoothing to keypoint positions"""
        smoothed_points = []
        
        for kps_num, (x_pred, y_pred, confidence) in enumerate(points_with_confidence):
            if kps_num not in self.keypoint_history:
                self.keypoint_history[kps_num] = []
                self.keypoint_confidence[kps_num] = []
            
            # Add current detection to history
            if x_pred is not None and y_pred is not None and confidence > self.min_confidence_threshold:
                self.keypoint_history[kps_num].append((x_pred, y_pred))
                self.keypoint_confidence[kps_num].append(confidence)
                
                # Keep only recent history
                if len(self.keypoint_history[kps_num]) > self.history_length:
                    self.keypoint_history[kps_num].pop(0)
                    self.keypoint_confidence[kps_num].pop(0)
            
            # Priority 1: Use best position if available and good enough
            if kps_num in self.best_positions and kps_num in self.best_scores:
                best_score = self.best_scores[kps_num]
                if best_score <= self.quality_threshold:
                    # Use the best position we've ever seen
                    best_x, best_y = self.best_positions[kps_num]
                    smoothed_points.append((best_x, best_y))
                    continue
            
            # Priority 2: Calculate smoothed position if we have history
            if len(self.keypoint_history[kps_num]) > 0:
                # Weighted average based on confidence
                total_weight = sum(self.keypoint_confidence[kps_num])
                if total_weight > 0:
                    weighted_x = sum(x * w for (x, y), w in zip(self.keypoint_history[kps_num], self.keypoint_confidence[kps_num])) / total_weight
                    weighted_y = sum(y * w for (x, y), w in zip(self.keypoint_history[kps_num], self.keypoint_confidence[kps_num])) / total_weight
                    
                    # Apply outlier rejection
                    if len(self.keypoint_history[kps_num]) >= 5:
                        distances = [np.sqrt((x - weighted_x)**2 + (y - weighted_y)**2) 
                                   for (x, y) in self.keypoint_history[kps_num]]
                        mean_distance = np.mean(distances)
                        std_distance = np.std(distances)
                        
                        # Remove outliers (points too far from mean)
                        filtered_positions = []
                        filtered_weights = []
                        for i, ((x, y), w) in enumerate(zip(self.keypoint_history[kps_num], self.keypoint_confidence[kps_num])):
                            if distances[i] <= mean_distance + 2 * std_distance:  # 2 sigma rule
                                filtered_positions.append((x, y))
                                filtered_weights.append(w)
                        
                        if len(filtered_positions) > 0:
                            total_weight = sum(filtered_weights)
                            weighted_x = sum(x * w for (x, y), w in zip(filtered_positions, filtered_weights)) / total_weight
                            weighted_y = sum(y * w for (x, y), w in zip(filtered_positions, filtered_weights)) / total_weight
                    
                    smoothed_points.append((int(weighted_x), int(weighted_y)))
                else:
                    smoothed_points.append((None, None))
            else:
                smoothed_points.append((None, None))
        
        return smoothed_points


def main():
    """Main function to run the super advanced tennis analysis engine"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Super Advanced Tennis Analysis Engine')
    parser.add_argument('--video', '-v', type=str, default='tennis_test.mp4',
                       help='Path to input video file')
    parser.add_argument('--output', '-o', type=str, default=None,
                       help='Path to output video file (optional)')
    parser.add_argument('--config', '-c', type=str, default='config.yaml',
                       help='Path to configuration file')
    
    args = parser.parse_args()
    
    # Check if video exists
    if not Path(args.video).exists():
        logger.error(f"Video file '{args.video}' not found!")
        logger.info("Available video files:")
        for video_file in Path('.').glob('*.mp4'):
            logger.info(f"  - {video_file}")
        return
    
    logger.info("🎾🏟️ SUPER ADVANCED TENNIS ANALYSIS ENGINE STARTING...")
    logger.info(f"📹 Input video: {args.video}")
    if args.output:
        logger.info(f"💾 Output video: {args.output}")
    logger.info(f"⚙️  Config file: {args.config}")
    logger.info("🚀 INTEGRATED SYSTEMS: Player Detection, Pose Estimation, Ball Tracking, Bounce Detection, Court Detection")
    logger.info("🎮 Controls: Press 'q' to quit, 'p' to pause/resume, 's' to save frame")
    logger.info("🔧 Initializing all systems...")
    
    # Create and run super advanced analyzer
    analyzer = TennisAnalysisDemo(args.config)
    analyzer.analyze_video(args.video, args.output)


if __name__ == "__main__":
    main()
