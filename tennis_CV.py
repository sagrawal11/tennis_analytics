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
import json
import os

# Configure logging first
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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

# Import RF-DETR for enhanced player and ball detection
try:
    from rfdetr import RFDETRNano
    RFDETR_AVAILABLE = True
    logger.info("RF-DETR imports successful - Enhanced detection enabled")
except ImportError as e:
    RFDETR_AVAILABLE = False
    logger.warning(f"RF-DETR imports failed: {e} - Enhanced detection will be disabled")

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
        
        # NEW: RF-DETR components for enhanced detection
        self.rfdetr_player_detector = None
        self.rfdetr_ball_detector = None
        self.yolo_fallback_detector = None  # Keep YOLO as fallback
        
        # Ball tracking state
        self.ball_positions = deque(maxlen=30)  # Store last 30 ball positions
        self.ball_velocities = deque(maxlen=10)  # Store last 10 velocities
        self.tracknet_predictions = []
        self.yolo_predictions = []
        
        # Player tracking state (NEW)
        self.player_positions = deque(maxlen=30)  # Store last 30 player positions per player
        self.player_velocities = deque(maxlen=10)  # Store last 10 player velocities
        self.player_shot_history = deque(maxlen=20)  # Store shot type history
        
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
            'processing_times': [],
            # NEW: RF-DETR detection counts
            'rfdetr_player_detections': 0,
            'rfdetr_ball_detections': 0,
            'yolo_fallback_detections': 0
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
                'yolo_pose': 'models/yolov8x-pose.pt',
                'bounce_detector': 'models/bounce_detector.cbm',
                'tracknet': 'pretrained_ball_detection.pt',
                'yolo_ball': 'models/playersnball4.pt',
                'court_detector': 'model_tennis_court_det.pt',  # NEW: Court detection model
                'rfdetr_model': 'models/playersnball5.pt'  # NEW: RF-DETR enhanced model
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
            'rfdetr': {  # NEW: RF-DETR settings
                'player_conf_threshold': 0.3,
                'ball_conf_threshold': 0.2,
                'max_players': 2,
                'max_balls': 1
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
            # Initialize RF-DETR player detector (primary)
            logger.info(f"🔍 RF-DETR_AVAILABLE: {RFDETR_AVAILABLE}")
            
            if RFDETR_AVAILABLE:
                rfdetr_model_path = self.config['models'].get('rfdetr_model')
                logger.info(f"🔍 RF-DETR model path: {rfdetr_model_path}")
                logger.info(f"🔍 RF-DETR model path exists: {Path(rfdetr_model_path).exists() if rfdetr_model_path else False}")
                
                if rfdetr_model_path and Path(rfdetr_model_path).exists():
                    try:
                        logger.info("🔍 Attempting to create RFDETRPlayerDetector...")
                        self.rfdetr_player_detector = RFDETRPlayerDetector(
                            rfdetr_model_path,
                            self.config['rfdetr']
                        )
                        logger.info(f"🔍 RFDETRPlayerDetector created: {self.rfdetr_player_detector is not None}")
                        
                        logger.info("🔍 Attempting to create RFDETRBallDetector...")
                        self.rfdetr_ball_detector = RFDETRBallDetector(
                            rfdetr_model_path,
                            self.config['rfdetr']
                        )
                        logger.info(f"🔍 RFDETRBallDetector created: {self.rfdetr_ball_detector is not None}")
                        
                        logger.info("RF-DETR player and ball detectors initialized successfully")
                    except Exception as e:
                        logger.error(f"RF-DETR initialization failed: {e}")
                        import traceback
                        traceback.print_exc()
                        self.rfdetr_player_detector = None
                        self.rfdetr_ball_detector = None
                else:
                    logger.warning(f"RF-DETR model not found: {rfdetr_model_path}")
                    self.rfdetr_player_detector = None
                    self.rfdetr_ball_detector = None
            else:
                logger.warning("RF-DETR not available - using YOLO fallback")
                self.rfdetr_player_detector = None
                self.rfdetr_ball_detector = None
            
            # Initialize YOLO player detector (fallback)
            player_model_path = self.config['models']['yolo_player']
            if Path(player_model_path).exists():
                self.yolo_fallback_detector = PlayerDetector(
                    player_model_path,
                    self.config['yolo_player']
                )
                logger.info("YOLO fallback player detector initialized successfully")
            else:
                logger.warning(f"YOLO fallback player detection model not found: {player_model_path}")
                self.yolo_fallback_detector = None
            
            # Set primary player detector (RF-DETR if available, otherwise YOLO)
            self.player_detector = self.rfdetr_player_detector if self.rfdetr_player_detector else self.yolo_fallback_detector
            
            # Initialize pose estimator
            pose_model_path = self.config['models']['yolo_pose']
            if Path(pose_model_path).exists():
                self.pose_estimator = PoseEstimator(pose_model_path)
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
        
        # Initialize CSV output
        self._initialize_csv_output()
        
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
                    logger.info("🔍 VIDEO: End of video reached, breaking loop")
                    break
                
                # Process every nth frame
                if frame_count % frame_skip != 0:
                    frame_count += 1
                    logger.debug(f"🔍 VIDEO: Skipping frame {frame_count} (frame_skip={frame_skip})")
                    continue
                
                logger.info(f"🔍 VIDEO: Processing frame {frame_count} (shape: {frame.shape})")
                
                # Analyze frame with ALL systems
                logger.info(f"🔍 VIDEO: Starting analysis of frame {frame_count}...")
                start_time = time.time()
                annotated_frame = self._analyze_frame(frame)
                processing_time = time.time() - start_time
                logger.info(f"🔍 VIDEO: Frame {frame_count} analysis completed in {processing_time:.3f}s")
                
                # Update statistics
                self.analysis_results['total_frames'] += 1
                self.analysis_results['processing_times'].append(processing_time)
                logger.info(f"🔍 VIDEO: Statistics updated for frame {frame_count}")
                
                # Display frame
                logger.info(f"🔍 VIDEO: Displaying frame {frame_count}...")
                cv2.imshow('Super Advanced Tennis Analysis Engine', annotated_frame)
                logger.info(f"🔍 VIDEO: Frame {frame_count} displayed successfully")
                
                # Save to output video if specified
                if output_writer:
                    logger.info(f"🔍 VIDEO: Writing frame {frame_count} to output video...")
                    output_writer.write(annotated_frame)
                    logger.info(f"🔍 VIDEO: Frame {frame_count} written to output video")
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):  # Quit
                    logger.info("🔍 VIDEO: Quit key pressed, breaking loop")
                    break
                elif key == ord('p'):  # Pause/Resume
                    logger.info("🔍 VIDEO: Pause key pressed, waiting for user input...")
                    cv2.waitKey(0)
                    logger.info("🔍 VIDEO: Resume key pressed, continuing...")
                elif key == ord('s'):  # Save current frame
                    logger.info(f"🔍 VIDEO: Save key pressed, saving frame {frame_count}...")
                    cv2.imwrite(f"super_tennis_frame_{frame_count:06d}.jpg", annotated_frame)
                    logger.info(f"Saved frame {frame_count}")
                
                frame_count += 1
                logger.info(f"🔍 VIDEO: Frame {frame_count-1} completed, moving to next frame")
                
                # Display progress
                if frame_count % 30 == 0:
                    progress = (frame_count / total_frames) * 100
                    avg_time = np.mean(self.analysis_results['processing_times'])
                    logger.info(f"🔍 VIDEO: Progress: {progress:.1f}% | Avg processing time: {avg_time:.3f}s")
        
        finally:
            cap.release()
            if output_writer:
                output_writer.release()
            cv2.destroyAllWindows()
            
            # Print final statistics
            self._print_analysis_summary()
    
    def _analyze_frame(self, frame: np.ndarray) -> np.ndarray:
        """Analyze a single frame with ALL systems including court detection"""
        frame_num = self.analysis_results['total_frames']
        logger.info(f"🔍 FRAME {frame_num}: Starting frame analysis...")
        
        annotated_frame = frame.copy()
        
        # Data to share with analytics viewer
        frame_data = {
            'frame_number': self.analysis_results['total_frames'],
            'timestamp': time.time(),
            'ball_x': None,
            'ball_y': None,
            'ball_confidence': None,
            'ball_source': None,
            'ball_speed': 0.0,
            'player_count': 0,
            'player_bboxes': [],
            'player_confidences': [],
            'player_speeds': [],
            'player_shot_types': [],
            'pose_count': 0,
            'pose_keypoints': [],
            'court_keypoints': [],
            'bounce_detected': False,
            'bounce_confidence': 0.0,
            'processing_time': 0.0,
            'detection_source': 'unknown'  # NEW: RF-DETR vs YOLO fallback
        }
        
        start_time = time.time()
        logger.info(f"🔍 FRAME {frame_num}: Frame data initialized, starting processing...")
        
        # 1. Player Detection with RF-DETR + YOLO Fallback
        logger.info(f"🔍 FRAME {frame_num}: Starting player detection...")
        player_detections = []
        if self.rfdetr_player_detector:
            logger.info(f"🔍 FRAME {frame_num}: RF-DETR player detector available, attempting detection...")
            try:
                # Try RF-DETR first (primary)
                logger.info(f"🔍 FRAME {frame_num}: Calling RF-DETR detect_players()...")
                rfdetr_detections = self.rfdetr_player_detector.detect_players(frame)
                logger.info(f"🔍 FRAME {frame_num}: RF-DETR detect_players() completed, result: {rfdetr_detections}")
                
                if rfdetr_detections:
                    player_detections = rfdetr_detections
                    self.analysis_results['rfdetr_player_detections'] += len(rfdetr_detections)
                    logger.info(f"🔍 FRAME {frame_num}: RF-DETR detected {len(rfdetr_detections)} players")
                    
                    # Draw RF-DETR detections
                    logger.info(f"🔍 FRAME {frame_num}: Drawing RF-DETR detections...")
                    annotated_frame = self.rfdetr_player_detector.draw_detections(annotated_frame, rfdetr_detections)
                    logger.info(f"🔍 FRAME {frame_num}: RF-DETR detections drawn successfully")
                else:
                    logger.info(f"🔍 FRAME {frame_num}: RF-DETR no players detected, trying YOLO fallback")
            except Exception as e:
                logger.error(f"🔍 FRAME {frame_num}: RF-DETR player detection failed: {e}")
                import traceback
                traceback.print_exc()
                logger.warning(f"RF-DETR player detection failed: {e}, trying YOLO fallback")
        
        # YOLO fallback if RF-DETR failed or no detections
        if not player_detections and self.yolo_fallback_detector:
            logger.info(f"🔍 FRAME {frame_num}: Attempting YOLO fallback detection...")
            try:
                logger.info(f"🔍 FRAME {frame_num}: Calling YOLO detect_players()...")
                yolo_detections = self.yolo_fallback_detector.detect_players(frame)
                logger.info(f"🔍 FRAME {frame_num}: YOLO detect_players() completed, result: {yolo_detections}")
                
                if yolo_detections:
                    player_detections = yolo_detections
                    self.analysis_results['yolo_fallback_detections'] += len(yolo_detections)
                    logger.info(f"🔍 FRAME {frame_num}: YOLO fallback detected {len(yolo_detections)} players")
                    
                    # Draw YOLO fallback detections
                    logger.info(f"🔍 FRAME {frame_num}: Drawing YOLO fallback detections...")
                    annotated_frame = self.yolo_fallback_detector.draw_detections(annotated_frame, yolo_detections)
                    logger.info(f"🔍 FRAME {frame_num}: YOLO fallback detections drawn successfully")
            except Exception as e:
                logger.error(f"🔍 FRAME {frame_num}: YOLO fallback also failed: {e}")
                import traceback
                traceback.print_exc()
        
        # Process detections
        logger.info(f"🔍 FRAME {frame_num}: Processing player detections...")
        if player_detections:
            logger.info(f"🔍 FRAME {frame_num}: Processing {len(player_detections)} player detections...")
            self.analysis_results['players_detected'] += len(player_detections)
            frame_data['player_count'] = len(player_detections)
            
            # Determine detection source
            if hasattr(player_detections[0], 'get') and 'court_score' in player_detections[0]:
                frame_data['detection_source'] = 'rfdetr'
                logger.info(f"🔍 FRAME {frame_num}: Detection source: RF-DETR")
            else:
                frame_data['detection_source'] = 'yolo_fallback'
                logger.info(f"🔍 FRAME {frame_num}: Detection source: YOLO fallback")
            
            # Sort players by y-coordinate (top to bottom) for consistent ordering
            logger.info(f"🔍 FRAME {frame_num}: Sorting players by position for consistent ordering...")
            player_data = []
            for i, detection in enumerate(player_detections):
                if 'bbox' in detection:
                    x1, y1, x2, y2 = detection['bbox']
                    center_y = (y1 + y2) / 2  # Calculate center y-coordinate
                    player_data.append({
                        'bbox': [x1, y1, x2, y2],
                        'center_y': center_y,
                        'confidence': detection.get('confidence', 0.0),
                        'original_index': i
                    })
            
            # Sort by y-coordinate (top player = index 0, bottom player = index 1)
            player_data.sort(key=lambda x: x['center_y'])
            
            # Store player data in consistent order
            for i, player in enumerate(player_data):
                x1, y1, x2, y2 = player['bbox']
                frame_data['player_bboxes'].append(f"{x1},{y1},{x2},{y2}")
                frame_data['player_confidences'].append(player['confidence'])
                
                # Calculate player speed
                player_speed = self._calculate_player_speed([x1, y1, x2, y2], frame_data['timestamp'])
                frame_data['player_speeds'].append(player_speed)
                
                # Detect shot type (will be updated after ball detection)
                frame_data['player_shot_types'].append("unknown")
        else:
            logger.warning(f"🔍 FRAME {frame_num}: No players detected by any model")
            frame_data['detection_source'] = 'none'
        
        logger.info(f"🔍 FRAME {frame_num}: Player detection processing completed")
        
        # 2. Pose Estimation
        logger.info(f"🔍 FRAME {frame_num}: Starting pose estimation...")
        poses = []
        if self.pose_estimator and player_detections:
            logger.info(f"🔍 FRAME {frame_num}: Pose estimator available, estimating poses for {len(player_detections)} players...")
            try:
                logger.info(f"🔍 FRAME {frame_num}: Calling pose_estimator.estimate_poses()...")
                poses = self.pose_estimator.estimate_poses(frame, player_detections)
                logger.info(f"🔍 FRAME {frame_num}: Pose estimation completed, got {len(poses)} poses")
                
                self.analysis_results['poses_estimated'] += len(poses)
                
                logger.info(f"🔍 FRAME {frame_num}: Drawing poses on frame...")
                annotated_frame = self.pose_estimator.draw_poses(annotated_frame, poses)
                logger.info(f"🔍 FRAME {frame_num}: Poses drawn successfully")
                
                # Store pose data
                frame_data['pose_count'] = len(poses)
                logger.info(f"🔍 FRAME {frame_num}: Storing pose data...")
                for i, pose in enumerate(poses):
                    logger.info(f"🔍 FRAME {frame_num}: Processing pose {i+1}/{len(poses)}...")
                    if 'keypoints' in pose:
                        keypoints_str = []
                        for j, kp in enumerate(pose['keypoints']):
                            keypoints_str.append(f"{kp[0]:.1f},{kp[1]:.1f},{kp[2]:.3f}")
                        frame_data['pose_keypoints'].append('|'.join(keypoints_str))
                        logger.info(f"🔍 FRAME {frame_num}: Pose {i+1} processed with {len(pose['keypoints'])} keypoints")
                logger.info(f"🔍 FRAME {frame_num}: Pose data storage completed")
            except Exception as e:
                logger.error(f"🔍 FRAME {frame_num}: Pose estimation error: {e}")
                import traceback
                traceback.print_exc()
        else:
            logger.info(f"🔍 FRAME {frame_num}: Pose estimation skipped (no estimator or no players)")
        
        logger.info(f"🔍 FRAME {frame_num}: Pose estimation section completed")
        
        # 3. Ball Detection and Tracking
        logger.info(f"🔍 FRAME {frame_num}: Starting ball detection and tracking...")
        logger.info(f"🔍 FRAME {frame_num}: Calling _detect_ball_in_frame()...")
        ball_pred = self._detect_ball_in_frame(frame)
        logger.info(f"🔍 FRAME {frame_num}: Ball detection completed, result: {ball_pred}")
        
        if ball_pred:
            logger.info(f"🔍 FRAME {frame_num}: Ball detected, processing ball data...")
            self.analysis_results['combined_ball_detections'] += 1
            frame_data['ball_position'] = ball_pred
            
            # Store ball data
            x, y = ball_pred['position']
            frame_data['ball_x'] = x
            frame_data['ball_y'] = y
            frame_data['ball_confidence'] = ball_pred.get('confidence', 0.0)
            frame_data['ball_source'] = ball_pred.get('source', 'unknown')
            logger.info(f"🔍 FRAME {frame_num}: Ball position stored: ({x}, {y})")
            
            # Calculate ball speed
            logger.info(f"🔍 FRAME {frame_num}: Calculating ball speed...")
            ball_speed = self._calculate_ball_speed([x, y], frame_data['timestamp'])
            frame_data['ball_speed'] = ball_speed
            logger.info(f"🔍 FRAME {frame_num}: Ball speed calculated: {ball_speed}")
            
            # Detect shot types for each player
            logger.info(f"🔍 FRAME {frame_num}: Detecting shot types for {len(frame_data['player_bboxes'])} players...")
            for i, player_bbox_str in enumerate(frame_data['player_bboxes']):
                try:
                    logger.info(f"🔍 FRAME {frame_num}: Processing shot type for player {i+1}...")
                    x1, y1, x2, y2 = map(int, player_bbox_str.split(','))
                    shot_type = self._detect_shot_type([x1, y1, x2, y2], [x, y], poses, self.court_keypoints)
                    frame_data['player_shot_types'][i] = shot_type
                    logger.info(f"🔍 FRAME {frame_num}: Player {i+1} shot type: {shot_type}")
                except Exception as e:
                    logger.error(f"🔍 FRAME {frame_num}: Error detecting shot type for player {i+1}: {e}")
                    continue
            
            # Calculate velocity
            if len(self.ball_positions) >= 2:
                logger.info(f"🔍 FRAME {frame_num}: Calculating ball velocity...")
                velocity = self._calculate_velocity(self.ball_positions[-2], ball_pred)
                self.ball_velocities.append(velocity)
                logger.info(f"🔍 FRAME {frame_num}: Ball velocity calculated: {velocity}")
            
            # Draw ball tracking
            logger.info(f"🔍 FRAME {frame_num}: Drawing ball tracking...")
            annotated_frame = self._draw_ball_tracking(annotated_frame, ball_pred)
            logger.info(f"🔍 FRAME {frame_num}: Ball tracking drawn successfully")
        else:
            logger.info(f"🔍 FRAME {frame_num}: No ball detected in this frame")
        
        logger.info(f"🔍 FRAME {frame_num}: Ball detection and tracking section completed")
        
        # 4. Ball Bounce Detection
        logger.info(f"🔍 FRAME {frame_num}: Starting bounce detection...")
        if self.bounce_detector:
            logger.info(f"🔍 FRAME {frame_num}: Bounce detector available, attempting detection...")
            try:
                logger.info(f"🔍 FRAME {frame_num}: Calling bounce_detector.detect_bounce()...")
                bounce_probability = self.bounce_detector.detect_bounce(frame)
                logger.info(f"🔍 FRAME {frame_num}: Bounce detection completed, probability: {bounce_probability}")
                
                if bounce_probability > 0.7:  # High confidence threshold
                    logger.info(f"🔍 FRAME {frame_num}: Bounce detected with high confidence!")
                    self.analysis_results['bounces_detected'] += 1
                    frame_data['bounce_detected'] = True
                    frame_data['bounce_confidence'] = bounce_probability
                    # Draw bounce indicator
                    cv2.putText(annotated_frame, f"BOUNCE! ({bounce_probability:.2f})", 
                               (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    # Draw bounce circle
                    cv2.circle(annotated_frame, (100, 100), 30, (0, 0, 255), -1)
                    logger.info(f"🔍 FRAME {frame_num}: Bounce indicators drawn successfully")
                else:
                    logger.info(f"🔍 FRAME {frame_num}: No bounce detected (probability: {bounce_probability})")
            except Exception as e:
                logger.error(f"🔍 FRAME {frame_num}: Bounce detection error: {e}")
                import traceback
                traceback.print_exc()
        else:
            logger.info(f"🔍 FRAME {frame_num}: Bounce detector not available")
        
        logger.info(f"🔍 FRAME {frame_num}: Bounce detection section completed")
        
        # 5. NEW: Court Detection and Analysis
        logger.info(f"🔍 FRAME {frame_num}: Starting court detection...")
        if self.court_detector:
            logger.info(f"🔍 FRAME {frame_num}: Court detector available, attempting detection...")
            try:
                logger.info(f"🔍 FRAME {frame_num}: Calling court_detector.detect_court_in_frame()...")
                court_points = self.court_detector.detect_court_in_frame(frame)
                logger.info(f"🔍 FRAME {frame_num}: Court detection completed, got {len(court_points) if court_points else 0} points")
                
                if court_points:
                    logger.info(f"🔍 FRAME {frame_num}: Processing court points...")
                    # Update court statistics
                    keypoints_detected = sum(1 for p in court_points if p[0] is not None and p[1] is not None)
                    self.analysis_results['keypoints_detected'] += keypoints_detected
                    logger.info(f"🔍 FRAME {frame_num}: Court keypoints detected: {keypoints_detected}")
                    
                    # Store court data
                    court_keypoints_str = []
                    for i, point in enumerate(court_points):
                        if point[0] is not None and point[1] is not None:
                            court_keypoints_str.append(f"{point[0]:.1f},{point[1]:.1f}")
                        else:
                            court_keypoints_str.append("None,None")
                    frame_data['court_keypoints'] = court_keypoints_str
                    logger.info(f"🔍 FRAME {frame_num}: Court keypoints stored: {len(court_keypoints_str)}")
                    
                    # Draw court keypoints and lines
                    logger.info(f"🔍 FRAME {frame_num}: Drawing court visualization...")
                    annotated_frame = self._draw_court_visualization(annotated_frame, court_points)
                    logger.info(f"🔍 FRAME {frame_num}: Court visualization drawn successfully")
                    
                    # Update court detection count if we have enough keypoints
                    if keypoints_detected >= 4:
                        self.analysis_results['court_detections'] += 1
                        logger.info(f"🔍 FRAME {frame_num}: Court detection count updated (total: {self.analysis_results['court_detections']})")
                else:
                    logger.info(f"🔍 FRAME {frame_num}: No court points detected")
                        
            except Exception as e:
                logger.error(f"🔍 FRAME {frame_num}: Court detection error: {e}")
                import traceback
                traceback.print_exc()
        else:
            logger.info(f"🔍 FRAME {frame_num}: Court detector not available")
        
        logger.info(f"🔍 FRAME {frame_num}: Court detection section completed")
        
        # Calculate processing time
        processing_time = time.time() - start_time
        frame_data['processing_time'] = processing_time
        self.analysis_results['processing_times'].append(processing_time)
        logger.info(f"🔍 FRAME {frame_num}: Processing time: {processing_time:.3f}s")
        
        # Output data for analytics viewer
        logger.info(f"🔍 FRAME {frame_num}: Outputting frame data to CSV...")
        self._output_frame_data(frame_data)
        logger.info(f"🔍 FRAME {frame_num}: Frame data output completed")
        
        # Add comprehensive frame information
        logger.info(f"🔍 FRAME {frame_num}: Adding frame information overlay...")
        self._add_frame_info(annotated_frame)
        logger.info(f"🔍 FRAME {frame_num}: Frame information overlay added")
        
        logger.info(f"🔍 FRAME {frame_num}: Frame analysis COMPLETED successfully!")
        return annotated_frame
    
    def _output_frame_data(self, frame_data: Dict[str, Any]):
        """Output frame data to CSV for analytics viewer"""
        try:
            # Convert data to CSV format with proper escaping
            csv_line = [
                str(frame_data['frame_number']),
                str(frame_data['timestamp']),
                str(frame_data['ball_x']) if frame_data['ball_x'] is not None else '',
                str(frame_data['ball_y']) if frame_data['ball_y'] is not None else '',
                str(frame_data['ball_confidence']) if frame_data['ball_confidence'] is not None else '',
                str(frame_data['ball_source']) if frame_data['ball_source'] is not None else '',
                str(frame_data['ball_speed']),
                str(frame_data['player_count']),
                f'"{";".join(frame_data["player_bboxes"])}"' if frame_data['player_bboxes'] else '""',
                f'"{";".join([str(c) for c in frame_data["player_confidences"]])}"' if frame_data['player_confidences'] else '""',
                f'"{";".join([str(s) for s in frame_data["player_speeds"]])}"' if frame_data['player_speeds'] else '""',
                f'"{";".join(frame_data["player_shot_types"])}"' if frame_data['player_shot_types'] else '""',
                str(frame_data['pose_count']),
                f'"{";".join(frame_data["pose_keypoints"])}"' if frame_data['pose_keypoints'] else '""',
                f'"{";".join(frame_data["court_keypoints"])}"' if frame_data['court_keypoints'] else '""',
                str(frame_data['bounce_detected']),
                str(frame_data['bounce_confidence']),
                str(frame_data['processing_time']),
                # NEW: RF-DETR detection source
                str(frame_data.get('detection_source', 'unknown'))
            ]
            
            # Write to CSV file with file locking to prevent corruption
            try:
                import fcntl
                with open('tennis_analysis_data.csv', 'a') as f:
                    # Acquire exclusive lock for writing
                    fcntl.flock(f.fileno(), fcntl.LOCK_EX)
                    f.write(','.join(csv_line) + '\n')
                    f.flush()  # Ensure data is written to disk
                    # Release lock
                    fcntl.flock(f.fileno(), fcntl.LOCK_UN)
            except ImportError:
                # fcntl not available on Windows, use regular file operations
                with open('tennis_analysis_data.csv', 'a') as f:
                    f.write(','.join(csv_line) + '\n')
                    f.flush()
            except Exception as e:
                logger.error(f"File locking error: {e}")
                # Fallback to regular file operations
                with open('tennis_analysis_data.csv', 'a') as f:
                    f.write(','.join(csv_line) + '\n')
                    f.flush()
                
        except Exception as e:
            logger.debug(f"Error outputting frame data: {e}")
    
    def _initialize_csv_output(self):
        """Initialize CSV file with headers"""
        try:
            headers = [
                'frame_number', 'timestamp', 'ball_x', 'ball_y', 'ball_confidence', 'ball_source', 'ball_speed',
                'player_count', 'player_bboxes', 'player_confidences', 'player_speeds', 'player_shot_types',
                'pose_count', 'pose_keypoints', 'court_keypoints', 'bounce_detected', 'bounce_confidence', 'processing_time',
                'detection_source'  # NEW: RF-DETR vs YOLO fallback
            ]
            
            # Initialize CSV file with file locking to prevent conflicts
            try:
                import fcntl
                with open('tennis_analysis_data.csv', 'w') as f:
                    # Acquire exclusive lock for writing
                    fcntl.flock(f.fileno(), fcntl.LOCK_EX)
                    f.write(','.join(headers) + '\n')
                    f.flush()  # Ensure data is written to disk
                    # Release lock
                    fcntl.flock(f.fileno(), fcntl.LOCK_UN)
            except ImportError:
                # fcntl not available on Windows, use regular file operations
                with open('tennis_analysis_data.csv', 'w') as f:
                    f.write(','.join(headers) + '\n')
                    f.flush()
            except Exception as e:
                logger.error(f"File locking error during initialization: {e}")
                # Fallback to regular file operations
                with open('tennis_analysis_data.csv', 'w') as f:
                    f.write(','.join(headers) + '\n')
                    f.flush()
                
        except Exception as e:
            logger.error(f"Error initializing CSV: {e}")
    
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
                (12, 13),           # Center service line: 12 → 13
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
        """Detect ball in a single frame using RF-DETR FIRST, then YOLO+TrackNet fallback"""
        tracknet_pred = None
        yolo_pred = None
        rfdetr_pred = None
        
        # Debug logging
        logger.info(f"🔍 Ball detection - RF-DETR detector available: {self.rfdetr_ball_detector is not None}")
        logger.info(f"🔍 Ball detection - RF-DETR player detector available: {self.rfdetr_player_detector is not None}")
        
        # 1. RF-DETR ball prediction (PRIMARY)
        if self.rfdetr_ball_detector:
            try:
                logger.info("🎯 Attempting RF-DETR ball detection...")
                rfdetr_pred = self.rfdetr_ball_detector.detect_ball(frame, self.rfdetr_player_detector)
                logger.info(f"🎯 RF-DETR result: {rfdetr_pred}")
                
                if rfdetr_pred:
                    self.analysis_results['rfdetr_ball_detections'] += 1
                    logger.info(f"🎯 RF-DETR PRIMARY ball detection: {rfdetr_pred}")
                    # RF-DETR detected the ball - use it directly with high confidence
                    return {
                        'position': rfdetr_pred['position'],
                        'confidence': min(1.0, rfdetr_pred['confidence'] * 1.1),  # Boost confidence by 10%
                        'source': 'rfdetr_primary',
                        'bbox': rfdetr_pred['bbox']
                    }
                else:
                    logger.info("🎯 RF-DETR returned None - no ball detected")
            except Exception as e:
                logger.error(f"RF-DETR error: {e}")
                import traceback
                traceback.print_exc()
        
        # 2. RF-DETR failed - fallback to YOLO + TrackNet
        logger.info("🔄 RF-DETR no ball detected, using YOLO+TrackNet fallback")
        
        # TrackNet prediction
        if self.tracknet_model:
            try:
                tracknet_pred = self.tracknet_model.detect_ball(frame)
                if tracknet_pred:
                    self.analysis_results['tracknet_detections'] += 1
                    self.tracknet_predictions.append(tracknet_pred)
            except Exception as e:
                logger.error(f"TrackNet error: {e}")
        
        # YOLO ball prediction
        if self.yolo_ball_model:
            try:
                yolo_pred = self.yolo_ball_model.detect_ball(frame)
                if yolo_pred:
                    self.analysis_results['yolo_ball_detections'] += 1
                    self.yolo_predictions.append(yolo_pred)
            except Exception as e:
                logger.error(f"YOLO error: {e}")
        
        # 3. Combine fallback predictions (YOLO + TrackNet)
        return self._combine_fallback_predictions(tracknet_pred, yolo_pred)
    
    def _combine_fallback_predictions(self, tracknet_pred: Optional[Dict], yolo_pred: Optional[Dict]) -> Optional[Dict]:
        """Combine YOLO + TrackNet predictions when RF-DETR fails (FALLBACK ONLY)"""
        if not tracknet_pred and not yolo_pred:
            return None
        
        if tracknet_pred and not yolo_pred:
            return {**tracknet_pred, 'source': 'tracknet_fallback'}
        
        if yolo_pred and not tracknet_pred:
            return {**yolo_pred, 'source': 'yolo_fallback'}
        
        # Both predictions exist - combine them with consensus
        tracknet_pos = tracknet_pred['position']
        yolo_pos = yolo_pred['position']
        tracknet_conf = tracknet_pred['confidence']
        yolo_conf = yolo_pred['confidence']
        
        # Check if they agree (within 30 pixels)
        distance = np.sqrt((tracknet_pos[0] - yolo_pos[0])**2 + (tracknet_pos[1] - yolo_pos[1])**2)
        
        if distance < 30:  # They agree!
            # Use weighted average
            total_conf = tracknet_conf + yolo_conf
            combined_x = (tracknet_pos[0] * tracknet_conf + yolo_pos[0] * yolo_conf) / total_conf
            combined_y = (tracknet_pos[1] * yolo_conf + yolo_pos[1] * yolo_conf) / total_conf
            combined_conf = (tracknet_conf + yolo_conf) / 2
            
            logger.info(f"🔄 FALLBACK CONSENSUS: TrackNet + YOLO agree at ({int(combined_x)}, {int(combined_y)})")
            
            return {
                'position': [int(combined_x), int(combined_y)],
                'confidence': combined_conf,
                'source': 'fallback_consensus'
            }
        else:
            # They don't agree - use the higher confidence one
            if tracknet_conf > yolo_conf:
                logger.info(f"🔄 FALLBACK: Using TrackNet (higher confidence: {tracknet_conf:.3f})")
                return {**tracknet_pred, 'source': 'tracknet_fallback'}
            else:
                logger.info(f"🔄 FALLBACK: Using YOLO (higher confidence: {yolo_conf:.3f})")
                return {**yolo_pred, 'source': 'yolo_fallback'}
    
    def _combine_predictions(self, tracknet_pred: Optional[Dict], yolo_pred: Optional[Dict]) -> Optional[Dict]:
        """Combine predictions from both models (legacy method - now replaced by _combine_fallback_predictions)"""
        return self._combine_fallback_predictions(tracknet_pred, yolo_pred)
    
    def _combine_triple_predictions(self, tracknet_pred: Optional[Dict], yolo_pred: Optional[Dict], rfdetr_pred: Optional[Dict]) -> Optional[Dict]:
        """Combine predictions from ALL THREE models with weighted fusion and consensus rules"""
        predictions = []
        
        # Collect all valid predictions
        if tracknet_pred:
            predictions.append(('tracknet', tracknet_pred))
        if yolo_pred:
            predictions.append(('yolo', yolo_pred))
        if rfdetr_pred:
            predictions.append(('rfdetr', rfdetr_pred))
        
        if not predictions:
            return None
        
        if len(predictions) == 1:
            # Only one model detected the ball
            source, pred = predictions[0]
            return {**pred, 'source': f'{source}_only'}
        
        # Multiple predictions - implement consensus rules
        if len(predictions) >= 2:
            # Check if 2/3 models agree on ball location
            positions = []
            for source, pred in predictions:
                pos = pred['position']
                positions.append((pos[0], pos[1], pred['confidence'], source))
            
            # Group nearby detections (within 50 pixels)
            clusters = []
            for pos in positions:
                x, y, conf, source = pos
                added_to_cluster = False
                
                for cluster in clusters:
                    cluster_x, cluster_y = cluster['center']
                    distance = np.sqrt((x - cluster_x)**2 + (y - cluster_y)**2)
                    
                    if distance < 50:  # 50 pixel threshold for consensus
                        cluster['positions'].append((x, y, conf, source))
                        cluster['total_conf'] += conf
                        cluster['count'] += 1
                        added_to_cluster = True
                        break
                
                if not added_to_cluster:
                    clusters.append({
                        'center': (x, y),
                        'positions': [(x, y, conf, source)],
                        'total_conf': conf,
                        'count': 1
                    })
            
            # Find the best cluster (highest consensus + confidence)
            best_cluster = None
            best_score = 0
            
            for cluster in clusters:
                # Consensus score: number of agreeing models
                consensus_score = cluster['count'] / len(predictions)
                # Combined confidence score
                avg_conf = cluster['total_conf'] / cluster['count']
                # Final score: consensus + confidence
                final_score = consensus_score * 0.6 + avg_conf * 0.4
                
                if final_score > best_score:
                    best_score = final_score
                    best_cluster = cluster
            
            if best_cluster and best_cluster['count'] >= 2:
                # We have consensus! Boost confidence
                x, y = best_cluster['center']
                boosted_conf = min(1.0, best_cluster['total_conf'] / best_cluster['count'] * 1.2)
                
                logger.info(f"TRIPLE FUSION CONSENSUS: {best_cluster['count']}/{len(predictions)} models agree at ({x}, {y})")
                
                return {
                    'position': [int(x), int(y)],
                    'confidence': boosted_conf,
                    'source': f'consensus_{best_cluster["count"]}models',
                    'consensus_count': best_cluster['count'],
                    'total_models': len(predictions)
                }
        
        # No consensus - use weighted average
        return self._combine_predictions_weighted(tracknet_pred, yolo_pred, rfdetr_pred)
    
    def _combine_predictions_weighted(self, tracknet_pred: Optional[Dict], yolo_pred: Optional[Dict], rfdetr_pred: Optional[Dict]) -> Optional[Dict]:
        """Weighted combination when no consensus is reached"""
        predictions = []
        weights = []
        
        if tracknet_pred:
            predictions.append(tracknet_pred)
            weights.append(0.35)  # TrackNet: 35% weight
        
        if yolo_pred:
            predictions.append(yolo_pred)
            weights.append(0.25)  # YOLO: 25% weight
        
        if rfdetr_pred:
            predictions.append(rfdetr_pred)
            weights.append(0.40)  # RF-DETR: 40% weight (highest weight)
        
        if not predictions:
            return None
        
        if len(predictions) == 1:
            return predictions[0]
        
        # Weighted average
        total_weight = sum(weights)
        weighted_x = 0
        weighted_y = 0
        weighted_conf = 0
        
        for pred, weight in zip(predictions, weights):
            pos = pred['position']
            conf = pred['confidence']
            normalized_weight = weight / total_weight
            
            weighted_x += pos[0] * normalized_weight
            weighted_y += pos[1] * normalized_weight
            weighted_conf += conf * normalized_weight
        
        return {
            'position': [int(weighted_x), int(weighted_y)],
            'confidence': weighted_conf,
            'source': 'weighted_fusion'
        }
    

    
    def _calculate_velocity(self, pos1: Dict, pos2: Dict) -> Tuple[float, float]:
        """Calculate velocity between two positions"""
        x1, y1 = pos1['position']
        x2, y2 = pos2['position']
        return (x2 - x1, y2 - y1)
    
    def _draw_ball_tracking(self, frame: np.ndarray, ball_pred: Dict) -> np.ndarray:
        """Draw ball tracking visualization with RF-DETR priority colors"""
        x, y = ball_pred['position']
        conf = ball_pred['confidence']
        source = ball_pred.get('source', 'unknown')
        
        # Color coding based on detection source
        if 'rfdetr' in source:
            # RF-DETR primary detection - BLUE (highest priority)
            color = (255, 0, 0)  # BGR format - blue
            label = f"RF-DETR: {conf:.2f}"
        elif 'fallback' in source:
            # Fallback detection - ORANGE
            color = (0, 165, 255)  # BGR format - orange
            label = f"Fallback: {conf:.2f}"
        else:
            # Unknown source - GREEN
            color = (0, 255, 0)  # BGR format - green
            label = f"{conf:.2f}"
        
        # Draw ball position
        cv2.circle(frame, (x, y), 8, color, -1)
        cv2.circle(frame, (x, y), 12, color, 2)
        
        # Draw label with source info
        cv2.putText(frame, label, (x + 15, y - 15), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Draw subtle tail using recent ball positions
        if len(self.ball_positions) >= 3:
            # Get last 3 ball positions for tail
            tail_positions = list(self.ball_positions)[-3:]
            for i, pos in enumerate(tail_positions):
                if pos and isinstance(pos, tuple) and len(pos) == 2 and pos != (x, y):
                    try:
                        # Fade the tail (more recent = more opaque)
                        alpha = 0.3 + (i * 0.2)  # 0.3, 0.5, 0.7
                        tail_color = (0, int(165 * alpha), int(255 * alpha))
                        cv2.circle(frame, (int(pos[0]), int(pos[1])), 4, tail_color, -1)
                    except (ValueError, TypeError):
                        # Skip invalid positions
                        continue
        
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
        
        # NEW: RF-DETR statistics
        if self.rfdetr_player_detector:
            stats_text.append(f"RF-DETR Players: {self.analysis_results['rfdetr_player_detections']}")
        if self.rfdetr_ball_detector:
            stats_text.append(f"RF-DETR Balls: {self.analysis_results['rfdetr_ball_detections']}")
        if self.yolo_fallback_detector:
            stats_text.append(f"YOLO Fallback: {self.analysis_results['yolo_fallback_detections']}")
        
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
        
        # NEW: RF-DETR statistics
        if self.rfdetr_player_detector:
            print(f"RF-DETR player detections: {self.analysis_results['rfdetr_player_detections']}")
        if self.rfdetr_ball_detector:
            print(f"RF-DETR ball detections: {self.analysis_results['rfdetr_ball_detections']}")
        if self.yolo_fallback_detector:
            print(f"YOLO fallback detections: {self.analysis_results['yolo_fallback_detections']}")
        
        if self.analysis_results['processing_times']:
            avg_time = np.mean(self.analysis_results['processing_times'])
            min_time = np.min(self.analysis_results['processing_times'])
            max_time = np.max(self.analysis_results['processing_times'])
            print(f"Processing time - Avg: {avg_time:.3f}s, Min: {min_time:.3f}s, Max: {max_time:.3f}s")
        
        total_time = time.time() - self.start_time
        print(f"Total analysis time: {total_time:.2f}s")
        print("="*60)
    
    def _detect_shot_type(self, player_bbox: List[int], ball_position: List[int], poses: List[Dict], court_keypoints: List[Tuple]) -> str:
        """Detect tennis shot type based on pose analysis and game context"""
        try:
            if not poses or not ball_position:
                logger.debug(f"⚠️  No poses ({len(poses) if poses else 0}) or ball position ({ball_position})")
                return "ready_stance"
            
            # Get player center position
            player_center_x = (player_bbox[0] + player_bbox[2]) / 2
            player_center_y = (player_bbox[1] + player_bbox[3]) / 2
            
            # Get ball position
            ball_x, ball_y = ball_position
            
            logger.debug(f"🔍 Player center: ({player_center_x:.1f}, {player_center_y:.1f}), Ball: ({ball_x}, {ball_y})")
            
            # Find the pose that corresponds to this player (closest to player center)
            closest_pose = None
            min_distance = float('inf')
            
            for pose in poses:
                if 'keypoints' in pose and len(pose['keypoints']) > 0:
                    # Use hip keypoints (11, 12) to find player center
                    hip_keypoints = []
                    confidence = pose.get('confidence', [])
                    
                    for i in [11, 12]:  # Left and right hip
                        if i < len(pose['keypoints']) and i < len(confidence) and confidence[i] > 0.3:
                            hip_keypoints.append(pose['keypoints'][i])
                    
                    if len(hip_keypoints) >= 2:
                        # Calculate pose center using hip keypoints
                        pose_center_x = sum(kp[0] for kp in hip_keypoints) / len(hip_keypoints)
                        pose_center_y = sum(kp[1] for kp in hip_keypoints) / len(hip_keypoints)
                        distance = np.sqrt((pose_center_x - player_center_x)**2 + (pose_center_y - player_center_y)**2)
                        
                        if distance < min_distance:
                            min_distance = distance
                            closest_pose = pose
            
            if not closest_pose:
                logger.debug(f"⚠️  No closest pose found")
                return "ready_stance"
            
            # Log pose information for debugging
            pose_scale = closest_pose.get('scale', 'unknown')
            logger.debug(f"🔍 Using pose with scale: {pose_scale}")
            
            # Analyze pose to determine shot type
            keypoints = closest_pose['keypoints']
            confidence = closest_pose.get('confidence', [])
            
            # Get key body keypoints (COCO format)
            # 0: nose, 1: left_eye, 2: right_eye, 3: left_ear, 4: right_ear
            # 5: left_shoulder, 6: right_shoulder, 7: left_elbow, 8: right_elbow
            # 9: left_wrist, 10: right_wrist, 11: left_hip, 12: right_hip
            # 13: left_knee, 14: right_knee, 15: left_ankle, 16: right_ankle
            
            # Extract keypoints with confidence > 0.3
            confident_keypoints = {}
            for i, kp in enumerate(keypoints):
                if i < len(confidence) and confidence[i] > 0.3:  # Only confident keypoints
                    confident_keypoints[i] = kp
            
            # DEBUG: Log keypoint information
            logger.debug(f"🔍 Shot detection - Confident keypoints: {len(confident_keypoints)}/{len(keypoints)}")
            logger.debug(f"🔍 Keypoint indices: {list(confident_keypoints.keys())}")
            
            # Check if we have enough keypoints for analysis
            if len(confident_keypoints) < 8:
                logger.debug(f"⚠️  Not enough confident keypoints ({len(confident_keypoints)} < 8), defaulting to ready_stance")
                return "ready_stance"
            
            # Detect serve - both players near baseline
            if self._is_serve_situation(player_center_y, court_keypoints):
                logger.debug(f"🎾 Serve detected - player at y={player_center_y}")
                return "serve"
            
            # Analyze arm positions for swing detection
            swing_type = self._analyze_swing_pose(confident_keypoints, player_center_x, ball_x, ball_y, player_center_y)
            
            if swing_type != "ready_stance":
                logger.debug(f"🎾 Swing detected: {swing_type}")
                return swing_type
            
            # Check if player is in ready stance (neutral position)
            if self._is_ready_stance(confident_keypoints, player_center_x):
                logger.debug(f"🔄 Ready stance detected")
                return "ready_stance"
            
            # If not swinging and not in ready stance, player is moving
            logger.debug(f"🏃 Player moving - not in ready stance or swinging")
            return "moving"
                
        except Exception as e:
            logger.debug(f"Error detecting shot type: {e}")
            return "ready_stance"
    
    def _is_serve_situation(self, player_y: float, court_keypoints: List[Tuple]) -> bool:
        """Check if this is a serve situation (players near baseline)"""
        try:
            if not court_keypoints or len(court_keypoints) < 4:
                logger.debug(f"⚠️  Not enough court keypoints for serve detection: {len(court_keypoints) if court_keypoints else 0}")
                return False
            
            # Find baseline y-coordinate (assuming court keypoints are ordered)
            # Court keypoints should include baseline coordinates
            baseline_y = None
            for kp in court_keypoints:
                if kp[0] is not None and kp[1] is not None:
                    if baseline_y is None or kp[1] > baseline_y:
                        baseline_y = kp[1]
            
            if baseline_y is None:
                logger.debug(f"⚠️  Could not determine baseline y-coordinate")
                return False
            
            # Check if player is close to baseline (within 50 pixels)
            distance_to_baseline = abs(player_y - baseline_y)
            is_serve = distance_to_baseline < 50
            
            logger.debug(f"🔍 Serve detection - Player y: {player_y:.1f}, Baseline y: {baseline_y:.1f}, Distance: {distance_to_baseline:.1f}, Is serve: {is_serve}")
            
            return is_serve
            
        except Exception as e:
            logger.debug(f"Error checking serve situation: {e}")
            return False
    
    def _analyze_swing_pose(self, keypoints: Dict[int, List], player_center_x: float, ball_x: float, ball_y: float, player_center_y: float) -> str:
        """Analyze pose to determine swing type (forehand, backhand, overhand smash)"""
        try:
            # Get arm keypoints
            left_shoulder = keypoints.get(5)   # Left shoulder
            right_shoulder = keypoints.get(6)  # Right shoulder
            left_elbow = keypoints.get(7)      # Left elbow
            right_elbow = keypoints.get(8)     # Right elbow
            left_wrist = keypoints.get(9)      # Left wrist
            right_wrist = keypoints.get(10)    # Right wrist
            
            # DEBUG: Log arm keypoint availability
            logger.debug(f"🔍 Arm keypoints - Left: S{left_shoulder is not None}, E{left_elbow is not None}, W{left_wrist is not None}")
            logger.debug(f"🔍 Arm keypoints - Right: S{right_shoulder is not None}, E{right_elbow is not None}, W{right_wrist is not None}")
            
            # Check if we have enough arm keypoints
            arm_keypoints = [kp for kp in [left_shoulder, right_shoulder, left_elbow, right_elbow, left_wrist, right_wrist] if kp is not None]
            if len(arm_keypoints) < 4:
                logger.debug(f"⚠️  Not enough arm keypoints ({len(arm_keypoints)} < 4), defaulting to ready_stance")
                return "ready_stance"
            
            # Calculate arm extension and position relative to ball
            left_arm_extended = False
            right_arm_extended = False
            
            # Check left arm extension
            if left_shoulder and left_elbow and left_wrist:
                left_arm_length = np.sqrt((left_elbow[0] - left_shoulder[0])**2 + (left_elbow[1] - left_shoulder[1])**2)
                left_forearm_length = np.sqrt((left_wrist[0] - left_elbow[0])**2 + (left_wrist[1] - left_elbow[1])**2)
                left_arm_extended = left_forearm_length > left_arm_length * 0.8
                logger.debug(f"🔍 Left arm - Upper: {left_arm_length:.1f}, Forearm: {left_forearm_length:.1f}, Extended: {left_arm_extended}")
            
            # Check right arm extension
            if right_shoulder and right_elbow and right_wrist:
                right_arm_length = np.sqrt((right_elbow[0] - right_shoulder[0])**2 + (right_elbow[1] - right_shoulder[1])**2)
                right_forearm_length = np.sqrt((right_wrist[0] - right_elbow[0])**2 + (right_wrist[1] - right_elbow[1])**2)
                right_arm_extended = right_forearm_length > right_arm_length * 0.8
                logger.debug(f"🔍 Right arm - Upper: {right_arm_length:.1f}, Forearm: {right_forearm_length:.1f}, Extended: {right_arm_extended}")
            
            # Determine swing type based on arm extension and ball position
            if left_arm_extended and not right_arm_extended:
                # Left arm extended - likely backhand
                if ball_x < player_center_x:
                    logger.debug(f"🎾 Backhand detected - left arm extended, ball on left")
                    return "backhand"
            elif right_arm_extended and not left_arm_extended:
                # Right arm extended - check ball position for forehand vs overhand
                if ball_y < player_center_y - 50:  # Ball above player - overhand smash
                    logger.debug(f"🎾 Overhand smash detected - right arm extended, ball above")
                    return "overhand_smash"
                else:  # Ball at or below player level - forehand
                    logger.debug(f"🎾 Forehand detected - right arm extended, ball at/below level")
                    return "forehand"
            elif left_arm_extended and right_arm_extended:
                # Both arms extended - could be overhand smash
                if ball_y < player_center_y - 30:
                    logger.debug(f"🎾 Overhand smash detected - both arms extended, ball above")
                    return "overhand_smash"
            
            logger.debug(f"🔄 No swing detected - arms not extended enough")
            return "ready_stance"
            
        except Exception as e:
            logger.debug(f"Error analyzing swing pose: {e}")
            return "ready_stance"
    
    def _is_ready_stance(self, keypoints: Dict[int, List], player_center_x: float) -> bool:
        """Check if player is in neutral ready stance"""
        try:
            # Get key body keypoints
            left_shoulder = keypoints.get(5)   # Left shoulder
            right_shoulder = keypoints.get(6)  # Right shoulder
            left_hip = keypoints.get(11)       # Left hip
            right_hip = keypoints.get(12)      # Right hip
            
            if not all([left_shoulder, right_shoulder, left_hip, right_hip]):
                logger.debug(f"⚠️  Missing key body keypoints for ready stance detection")
                return False
            
            # Check if shoulders and hips are roughly level (neutral stance)
            shoulder_diff = abs(left_shoulder[1] - right_shoulder[1])
            hip_diff = abs(left_hip[1] - right_hip[1])
            
            logger.debug(f"🔍 Ready stance - Shoulder diff: {shoulder_diff:.1f}, Hip diff: {hip_diff:.1f}")
            
            # In ready stance, shoulders and hips should be roughly level
            is_ready = shoulder_diff < 20 and hip_diff < 20
            logger.debug(f"🔍 Ready stance detected: {is_ready}")
            return is_ready
            
        except Exception as e:
            logger.debug(f"Error checking ready stance: {e}")
            return False
    
    def _calculate_player_speed(self, player_bbox: List[int], frame_time: float) -> float:
        """Calculate player speed in pixels per second"""
        try:
            if not self.player_positions:
                return 0.0
            
            # Get current player center
            current_center = [(player_bbox[0] + player_bbox[2]) / 2, (player_bbox[1] + player_bbox[3]) / 2]
            
            # Add to position history
            self.player_positions.append((current_center, frame_time))
            
            # Calculate speed if we have at least 2 positions
            if len(self.player_positions) >= 2:
                prev_pos, prev_time = self.player_positions[-2]
                curr_pos, curr_time = self.player_positions[-1]
                
                # Calculate distance and time difference
                distance = np.sqrt((curr_pos[0] - prev_pos[0])**2 + (curr_pos[1] - prev_pos[1])**2)
                time_diff = curr_time - prev_time
                
                if time_diff > 0:
                    speed = distance / time_diff  # pixels per second
                    self.player_velocities.append(speed)
                    
                    # Return average speed over last few frames
                    if len(self.player_velocities) > 0:
                        return np.mean(list(self.player_velocities)[-5:])  # Average of last 5 velocities
            
            return 0.0
            
        except Exception as e:
            logger.debug(f"Error calculating player speed: {e}")
            return 0.0
    
    def _calculate_ball_speed(self, ball_position: List[int], frame_time: float) -> float:
        """Calculate ball speed in pixels per second"""
        try:
            if not self.ball_positions:
                return 0.0
            
            # Get current ball position
            current_pos = ball_position
            
            # Add to position history with timestamp
            self.ball_positions.append((current_pos, frame_time))
            
            # Calculate speed if we have at least 2 positions
            if len(self.ball_positions) >= 2:
                prev_pos, prev_time = self.ball_positions[-2]
                curr_pos, curr_time = self.ball_positions[-1]
                
                # Calculate distance and time difference
                distance = np.sqrt((curr_pos[0] - prev_pos[0])**2 + (curr_pos[1] - prev_pos[1])**2)
                time_diff = curr_time - prev_time
                
                if time_diff > 0:
                    speed = distance / time_diff  # pixels per second
                    self.ball_velocities.append(speed)
                    
                    # Return average speed over last few frames
                    if len(self.ball_velocities) > 0:
                        return np.mean(list(self.ball_velocities)[-5:])  # Average of last 5 velocities
            
            return 0.0
            
        except Exception as e:
            logger.debug(f"Error calculating ball speed: {e}")
            return 0.0


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


class RFDETRPlayerDetector:
    """RF-DETR-based player detection with tennis-specific filtering"""
    
    def __init__(self, model_path: str, config: Dict[str, Any]):
        self.config = config
        try:
            # Load checkpoint first to get configuration
            checkpoint = torch.load(model_path, map_location='cpu')
            if 'args' in checkpoint and 'model' in checkpoint:
                args = checkpoint['args']
                logger.info(f"RF-DETR model with {args.num_classes} classes: {args.class_names}")
                
                # Create RF-DETR with custom classes
                self.model = RFDETRNano(
                    num_classes=len(args.class_names),  # 2 classes: ball + player
                    pretrain_weights=None  # Don't load default weights
                )
                
                # Load custom state dict
                missing_keys, unexpected_keys = self.model.model.model.load_state_dict(checkpoint['model'], strict=False)
                if missing_keys:
                    logger.warning(f"Missing keys when loading RF-DETR: {len(missing_keys)}")
                if unexpected_keys:
                    logger.warning(f"Unexpected keys when loading RF-DETR: {len(unexpected_keys)}")
                
                # Set class names
                try:
                    self.model.class_names = args.class_names
                except:
                    self.model.model.class_names = args.class_names
                
                self.args = args
                logger.info("RF-DETR player detector initialized successfully")
            else:
                logger.error("Invalid RF-DETR checkpoint format")
                self.model = None
        except Exception as e:
            logger.error(f"Error loading RF-DETR model: {e}")
            self.model = None
    
    def detect_players(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        if not self.model:
            logger.warning("🔍 RF-DETR: Model not available")
            return []
        
        try:
            logger.info("🔍 RF-DETR: Starting player detection...")
            logger.info(f"🔍 RF-DETR: Input frame shape: {frame.shape}")
            
            # Convert BGR to RGB and to PIL
            logger.info("🔍 RF-DETR: Converting BGR to RGB...")
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            logger.info("🔍 RF-DETR: BGR to RGB conversion completed")
            
            from PIL import Image
            logger.info("🔍 RF-DETR: Converting to PIL Image...")
            pil_image = Image.fromarray(frame_rgb)
            logger.info(f"🔍 RF-DETR: PIL Image created, size: {pil_image.size}")
            
            # Run inference
            logger.info("🔍 RF-DETR: Starting model inference...")
            logger.info(f"🔍 RF-DETR: Using threshold: {self.config.get('player_conf_threshold', 0.3)}")
            detections = self.model.predict(pil_image, threshold=self.config.get('player_conf_threshold', 0.3))
            logger.info("🔍 RF-DETR: Model inference completed")
            logger.info(f"🔍 RF-DETR: Raw detections: {len(detections.xyxy) if hasattr(detections, 'xyxy') else 'No xyxy attribute'}")
            
            # Filter for players only
            players = []
            logger.info("🔍 RF-DETR: Filtering detections for players...")
            
            if hasattr(detections, 'xyxy') and len(detections.xyxy) > 0:
                for i in range(len(detections.xyxy)):
                    logger.info(f"🔍 RF-DETR: Processing detection {i+1}/{len(detections.xyxy)}...")
                    bbox = detections.xyxy[i]
                    confidence = detections.confidence[i]
                    class_id = detections.class_id[i]
                    
                    logger.info(f"🔍 RF-DETR: Detection {i+1} - bbox: {bbox}, confidence: {confidence}, class_id: {class_id}")
                    
                    # Only keep players (class_id == 2 for 'player' based on our model)
                    if class_id == 2 and confidence > self.config.get('player_conf_threshold', 0.3):
                        logger.info(f"🔍 RF-DETR: Detection {i+1} is a valid player")
                        x1, y1, x2, y2 = bbox
                        bbox_center_x = (x1 + x2) / 2
                        bbox_center_y = (y1 + y2) / 2
                        
                        # Court position scoring - players should be in center area
                        frame_center_x = frame.shape[1] / 2
                        frame_center_y = frame.shape[0] / 2
                        distance_from_center = abs(bbox_center_x - frame_center_x) + abs(bbox_center_y - frame_center_y)
                        
                        # Combined score: confidence + court position
                        court_score = confidence - (distance_from_center / 2000)
                        
                        detection = {
                            'bbox': [int(x1), int(y1), int(x2), int(y2)],
                            'confidence': float(confidence),
                            'class': int(class_id),
                            'center': [int(bbox_center_x), int(bbox_center_y)],
                            'court_score': court_score
                        }
                        players.append(detection)
                        logger.info(f"🔍 RF-DETR: Player {len(players)} added: bbox={detection['bbox']}, confidence={detection['confidence']:.3f}")
                    else:
                        logger.info(f"🔍 RF-DETR: Detection {i+1} filtered out (class_id: {class_id}, confidence: {confidence})")
            else:
                logger.info("🔍 RF-DETR: No detections found in model output")
            
            # Sort by court score and keep top 2 players
            logger.info(f"🔍 RF-DETR: Sorting {len(players)} players by court score...")
            players.sort(key=lambda x: x['court_score'], reverse=True)
            players = players[:self.config.get('max_players', 2)]
            logger.info(f"🔍 RF-DETR: Final player count: {len(players)}")
            
            logger.info(f"RF-DETR detected {len(players)} players")
            return players
            
        except Exception as e:
            logger.error(f"🔍 RF-DETR: Player detection error: {e}")
            import traceback
            traceback.print_exc()
            return []
    
    def draw_detections(self, frame: np.ndarray, detections: List[Dict[str, Any]]) -> np.ndarray:
        frame_copy = frame.copy()
        
        for detection in detections:
            x1, y1, x2, y2 = detection['bbox']
            conf = detection['confidence']
            court_score = detection.get('court_score', 0.0)
            
            # Color for RF-DETR players (blue to distinguish from YOLO)
            color = (255, 0, 0)  # BGR format - blue
            
            # Draw bounding box
            cv2.rectangle(frame_copy, (x1, y1), (x2, y2), color, 2)
            
            # Draw label with court score
            label = f"RF-Player: {conf:.2f} (CS:{court_score:.2f})"
            cv2.putText(frame_copy, label, (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        return frame_copy


class RFDETRBallDetector:
    """RF-DETR-based ball detection for tennis"""
    
    def __init__(self, model_path: str, config: Dict[str, Any]):
        # Reuse the same model from player detector
        self.config = config
        # The model is already loaded in RFDETRPlayerDetector
        # We'll access it through the player detector instance
    
    def detect_ball(self, frame: np.ndarray, player_detector_instance) -> Optional[Dict[str, Any]]:
        logger.info("🔍 RF-DETR BALL: Starting ball detection...")
        
        if not player_detector_instance or not player_detector_instance.model:
            logger.warning("🔍 RF-DETR BALL: Player detector or model not available")
            return None
        
        try:
            logger.info("🔍 RF-DETR BALL: Converting BGR to RGB...")
            # Convert BGR to RGB and to PIL
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            logger.info("🔍 RF-DETR BALL: BGR to RGB conversion completed")
            
            from PIL import Image
            logger.info("🔍 RF-DETR BALL: Converting to PIL Image...")
            pil_image = Image.fromarray(frame_rgb)
            logger.info(f"🔍 RF-DETR BALL: PIL Image created, size: {pil_image.size}")
            
            # Run inference
            logger.info("🔍 RF-DETR BALL: Starting model inference...")
            logger.info(f"🔍 RF-DETR BALL: Using threshold: {self.config.get('ball_conf_threshold', 0.2)}")
            detections = player_detector_instance.model.predict(pil_image, threshold=self.config.get('ball_conf_threshold', 0.2))
            logger.info("🔍 RF-DETR BALL: Model inference completed")
            logger.info(f"🔍 RF-DETR BALL: Raw detections: {len(detections.xyxy) if hasattr(detections, 'xyxy') else 'No xyxy attribute'} total")
            
            # Filter for ball only
            balls = []
            
            if hasattr(detections, 'xyxy') and len(detections.xyxy) > 0:
                for i in range(len(detections.xyxy)):
                    logger.info(f"🔍 RF-DETR BALL: Processing detection {i+1}/{len(detections.xyxy)}...")
                    bbox = detections.xyxy[i]
                    confidence = detections.confidence[i]
                    class_id = detections.class_id[i]
                    
                    logger.info(f"🔍 RF-DETR BALL: Detection {i+1}: class_id={class_id}, confidence={confidence:.3f}, bbox={bbox}")
                    
                    # Only keep balls (class_id == 1 for 'ball' based on our model)
                    if class_id == 1 and confidence > self.config.get('ball_conf_threshold', 0.2):
                        logger.info(f"🔍 RF-DETR BALL: Detection {i+1} is a valid ball")
                        x1, y1, x2, y2 = bbox
                        center_x = int((x1 + x2) / 2)
                        center_y = int((y1 + y2) / 2)
                        
                        detection = {
                            'position': [center_x, center_y],
                            'confidence': float(confidence),
                            'bbox': [int(x1), int(y1), int(x2), int(y2)],
                            'source': 'rfdetr'
                        }
                        balls.append(detection)
                        logger.info(f"🔍 RF-DETR BALL: Added ball detection: pos=({center_x}, {center_y}), conf={confidence:.3f}")
                    else:
                        logger.info(f"🔍 RF-DETR BALL: Detection {i+1} filtered out: class_id={class_id} (not ball) or confidence={confidence:.3f} < {self.config.get('ball_conf_threshold', 0.2)}")
            else:
                logger.info("🔍 RF-DETR BALL: No detections found in model output")
            
            # Return highest confidence ball
            if balls:
                logger.info(f"🔍 RF-DETR BALL: Sorting {len(balls)} ball detections by confidence...")
                balls.sort(key=lambda x: x['confidence'], reverse=True)
                logger.info(f"🔍 RF-DETR BALL: Best ball detection: pos=({balls[0]['position'][0]}, {balls[0]['position'][1]}), conf={balls[0]['confidence']:.3f}")
                return balls[0]
            else:
                logger.info("🔍 RF-DETR BALL: No valid ball detections found")
                return None
            
        except Exception as e:
            logger.error(f"🔍 RF-DETR BALL: Ball detection error: {e}")
            import traceback
            traceback.print_exc()
            return None


class PoseEstimator:
    """Multi-scale pose estimation for better arm detection"""
    
    def __init__(self, model_path: str):
        """Initialize the multi-scale pose estimator"""
        try:
            from ultralytics import YOLO
            self.model = YOLO(model_path)
            logger.info(f"Pose estimator initialized with {model_path}")
            self.scales = [1.0, 1.5, 2.0]  # Original, 1.5x, 2x zoom
            self.scale_weights = [0.4, 0.35, 0.25]  # Weight for each scale
            
            # Temporal smoothing parameters
            self.pose_history = {}  # player_id -> deque of recent poses
            self.history_length = 5  # Number of frames to remember
            self.max_movement_threshold = 100  # Max pixels a keypoint can move between frames
            self.temporal_weight = 0.7  # How much to weight temporal consistency vs current detection
            
        except Exception as e:
            logger.error(f"Failed to initialize pose estimator: {e}")
            self.model = None
    
    def estimate_poses(self, frame: np.ndarray, player_detections: List[Dict]) -> List[Dict]:
        """Estimate poses using multi-scale detection for better arm keypoints"""
        if not self.model or not player_detections:
            return []
        
        try:
            all_poses = []
            
            # Process each player individually with multi-scale detection
            for player_idx, player in enumerate(player_detections):
                if 'bbox' not in player:
                    continue
                
                x1, y1, x2, y2 = player['bbox']
                
                # Extract player region with moderate padding for limb extension
                padding = 60  # Moderate padding to allow for arm/leg extension during swings
                x1_pad = max(0, x1 - padding)
                y1_pad = max(0, y1 - padding)
                x2_pad = min(frame.shape[1], x2 + padding)
                y2_pad = min(frame.shape[0], y2 + padding)
                
                player_roi = frame[y1_pad:y2_pad, x1_pad:x2_pad]
                
                if player_roi.size == 0:
                    continue
                
                player_poses = []
                
                # Run pose detection at multiple scales for this player
                for scale_idx, scale in enumerate(self.scales):
                    # Resize player ROI for this scale
                    if scale != 1.0:
                        roi_height, roi_width = player_roi.shape[:2]  # Get height and width (first 2 dimensions)
                        new_height, new_width = int(roi_height * scale), int(roi_width * scale)
                        scaled_roi = cv2.resize(player_roi, (new_width, new_height))
                    else:
                        scaled_roi = player_roi
                    
                    # Run pose detection on scaled ROI
                    results = self.model(scaled_roi, verbose=False, max_det=1)
                    
                    # Process results for this scale
                    for result in results:
                        if result.keypoints is not None:
                            keypoints = result.keypoints.data[0].cpu().numpy()
                            confidence = result.keypoints.conf[0].cpu().numpy() if result.keypoints.conf is not None else np.ones(len(keypoints))
                            
                            # Scale keypoints back to original ROI size
                            if scale != 1.0:
                                keypoints[:, 0] /= scale  # x coordinates
                                keypoints[:, 1] /= scale  # y coordinates
                            
                            # Adjust keypoints back to full frame coordinates
                            keypoints[:, 0] += x1_pad  # Add x offset
                            keypoints[:, 1] += y1_pad  # Add y offset
                            
                            # Apply bounding box confidence weighting
                            weighted_confidence = self._apply_bbox_confidence_weighting(
                                keypoints, confidence, [x1, y1, x2, y2]
                            )
                            
                            # Apply temporal consistency weighting
                            temporal_scores = self._calculate_temporal_consistency(
                                player_idx, keypoints, [x1, y1, x2, y2]
                            )
                            
                            # Combine spatial and temporal confidence
                            final_confidence = (
                                weighted_confidence * (1 - self.temporal_weight) +
                                temporal_scores * self.temporal_weight
                            )
                            
                            # Create pose data with scale information
                            pose_data = {
                                'keypoints': keypoints.tolist(),
                                'scale': scale,
                                'scale_weight': self.scale_weights[scale_idx],
                                'confidence': final_confidence.tolist(),
                                'player_idx': player_idx,
                                'bbox': [x1, y1, x2, y2]
                            }
                            player_poses.append(pose_data)
                
                # Combine poses for this player from different scales
                if player_poses:
                    if len(player_poses) > 1:
                        # Multiple scales detected - combine them
                        logger.debug(f"🔍 Player {player_idx}: Combining {len(player_poses)} poses from different scales")
                        combined_pose = self._combine_player_poses(player_poses)
                        
                        # Validate that hips are within bounding box
                        if not self._validate_pose_hips(combined_pose, [x1, y1, x2, y2]):
                            logger.warning(f"⚠️  Player {player_idx}: Invalid pose detected, attempting redraw")
                            redrawn_pose = self._attempt_pose_redraw(frame, player, [x1, y1, x2, y2], player_idx)
                            if redrawn_pose:
                                combined_pose = redrawn_pose
                                logger.info(f"✅ Player {player_idx}: Pose redraw successful")
                            else:
                                logger.warning(f"❌ Player {player_idx}: Pose redraw failed, using original pose")
                        
                        logger.debug(f"🔍 Player {player_idx}: Final pose added to all_poses")
                        all_poses.append(combined_pose)
                        
                        # Update pose history for temporal smoothing
                        self._update_pose_history(player_idx, combined_pose)
                    else:
                        # Only one scale detected - use as is
                        logger.debug(f"🔍 Player {player_idx}: Using single pose from scale {player_poses[0]['scale']}")
                        single_pose = player_poses[0]
                        
                        # Validate that hips are within bounding box
                        if not self._validate_pose_hips(single_pose, [x1, y1, x2, y2]):
                            logger.warning(f"⚠️  Player {player_idx}: Invalid pose detected, attempting redraw")
                            redrawn_pose = self._attempt_pose_redraw(frame, player, [x1, y1, x2, y2], player_idx)
                            if redrawn_pose:
                                single_pose = redrawn_pose
                                logger.info(f"✅ Player {player_idx}: Pose redraw successful")
                            else:
                                logger.warning(f"❌ Player {player_idx}: Pose redraw failed, using original pose")
                        
                        logger.debug(f"🔍 Player {player_idx}: Final pose added to all_poses")
                        all_poses.append(single_pose)
                        
                        # Update pose history for temporal smoothing
                        self._update_pose_history(player_idx, single_pose)
                else:
                    logger.debug(f"⚠️  Player {player_idx}: No poses detected at any scale")
            
            logger.debug(f"🔍 Total poses detected for all players: {len(all_poses)}")
            logger.debug(f"🔍 Player indices in all_poses: {[pose.get('player_idx', 'unknown') for pose in all_poses]}")
            return all_poses
            
        except Exception as e:
            logger.error(f"Error in multi-scale pose estimation: {e}")
            return []
    
    def _apply_bbox_confidence_weighting(self, keypoints: np.ndarray, confidence: np.ndarray, bbox: List[int]) -> np.ndarray:
        """Apply confidence weighting based on distance from player bounding box"""
        x1, y1, x2, y2 = bbox
        weighted_confidence = confidence.copy()
        
        for i, keypoint in enumerate(keypoints):
            kp_x, kp_y = keypoint[0], keypoint[1]  # Extract x, y coordinates
            if kp_x == 0 and kp_y == 0:  # Skip invalid keypoints
                continue
                
            # Check if keypoint is inside the bounding box
            if x1 <= kp_x <= x2 and y1 <= kp_y <= y2:
                # Inside box - keep full confidence
                continue
            else:
                # Outside box - calculate distance penalty
                # Find closest point on bbox boundary
                closest_x = max(x1, min(x2, kp_x))
                closest_y = max(y1, min(y2, kp_y))
                
                distance = np.sqrt((kp_x - closest_x)**2 + (kp_y - closest_y)**2)
                
                # Apply distance penalty: confidence decreases with distance
                # More forgiving for close distances, harsh for far distances
                if distance <= 30:
                    # Very close to box - minimal penalty
                    penalty_factor = 0.9
                elif distance <= 60:
                    # Moderate distance - gradual penalty
                    penalty_factor = max(0.5, 1.0 - (distance - 30) / 60)
                else:
                    # Far from box - heavy penalty
                    penalty_factor = max(0.1, 0.5 - (distance - 60) / 100)
                
                weighted_confidence[i] *= penalty_factor
        
        return weighted_confidence
    
    def _combine_player_poses(self, poses: List[Dict]) -> Dict:
        """Combine poses from different scales for the same player"""
        if not poses:
            return {}
        
        # Sort by scale weight (highest first)
        poses.sort(key=lambda x: x['scale_weight'], reverse=True)
        
        # Start with the highest confidence scale
        best_pose = poses[0]
        combined_keypoints = best_pose['keypoints'].copy()
        combined_confidence = best_pose['confidence'].copy()
        
        # For arm keypoints (5-10), try to improve with other scales
        arm_indices = [5, 6, 7, 8, 9, 10]  # Shoulders, elbows, wrists
        
        for pose in poses[1:]:  # Skip the best one
            for idx in arm_indices:
                if (idx < len(pose['keypoints']) and 
                    idx < len(pose['confidence']) and 
                    pose['confidence'][idx] > combined_confidence[idx]):
                    
                    # This scale has better confidence for this arm keypoint
                    combined_keypoints[idx] = pose['keypoints'][idx]
                    combined_confidence[idx] = pose['confidence'][idx]
        
        return {
            'keypoints': combined_keypoints,
            'confidence': combined_confidence,
            'scale': 'combined',
            'scale_weight': 1.0,
            'player_idx': best_pose.get('player_idx', 0),
            'bbox': best_pose.get('bbox', [])
        }
    
    def draw_poses(self, frame: np.ndarray, poses: List[Dict]) -> np.ndarray:
        """Draw poses on frame"""
        if not poses:
            return frame
        
        for pose in poses:
            if 'keypoints' in pose:
                keypoints = pose['keypoints']
                confidence = pose.get('confidence', [])
                
                # Draw keypoints
                for i, kp in enumerate(keypoints):
                    if len(confidence) > i and confidence[i] > 0.3:
                        x, y = int(kp[0]), int(kp[1])
                        cv2.circle(frame, (x, y), 3, (0, 255, 0), -1)
                        
                        # Label keypoint index for debugging
                        cv2.putText(frame, str(i), (x+5, y-5), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1)
                
                # Draw connections between keypoints (skeleton)
                self._draw_skeleton(frame, keypoints, confidence)
        
        return frame
    
    def _draw_skeleton(self, frame: np.ndarray, keypoints: List, confidence: List):
        """Draw skeleton connections between keypoints"""
        # Define skeleton connections (COCO format)
        skeleton = [
            (0, 1), (0, 2), (1, 3), (2, 4),  # Head connections
            (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),  # Arm connections
            (5, 11), (6, 12), (11, 12),  # Torso connections
            (11, 13), (13, 15), (12, 14), (14, 16)  # Leg connections
        ]
        
        for connection in skeleton:
            if (connection[0] < len(keypoints) and connection[1] < len(keypoints) and
                connection[0] < len(confidence) and connection[1] < len(confidence) and
                confidence[connection[0]] > 0.3 and confidence[connection[1]] > 0.3):
                
                pt1 = (int(keypoints[connection[0]][0]), int(keypoints[connection[0]][1]))
                pt2 = (int(keypoints[connection[1]][0]), int(keypoints[connection[1]][1]))
                cv2.line(frame, pt1, pt2, (0, 255, 0), 2)
    
    def _calculate_temporal_consistency(self, player_id: int, current_keypoints: np.ndarray, current_bbox: List[int]) -> np.ndarray:
        """Calculate temporal consistency scores for each keypoint based on movement history"""
        if player_id not in self.pose_history or len(self.pose_history[player_id]) < 2:
            # Not enough history - return neutral scores
            return np.ones(len(current_keypoints))
        
        temporal_scores = np.ones(len(current_keypoints))
        recent_poses = list(self.pose_history[player_id])[-3:]  # Last 3 poses
        
        for i, keypoint in enumerate(current_keypoints):
            kp_x, kp_y = keypoint[0], keypoint[1]
            if kp_x == 0 and kp_y == 0:  # Skip invalid keypoints
                continue
            
            # Predict where this keypoint should be based on recent movement
            predicted_x, predicted_y = self._predict_keypoint_position(i, recent_poses)
            
            if predicted_x is not None and predicted_y is not None:
                # Calculate distance from predicted position
                distance = np.sqrt((kp_x - predicted_x)**2 + (kp_y - predicted_y)**2)
                
                # Calculate temporal consistency score
                if distance <= self.max_movement_threshold:
                    # Movement is within reasonable bounds
                    temporal_scores[i] = max(0.3, 1.0 - (distance / self.max_movement_threshold))
                else:
                    # Movement is too large - likely a detection error
                    temporal_scores[i] = 0.1
                    
                logger.debug(f"🔍 Keypoint {i}: predicted=({predicted_x:.1f}, {predicted_y:.1f}), "
                           f"actual=({kp_x:.1f}, {kp_y:.1f}), distance={distance:.1f}, "
                           f"temporal_score={temporal_scores[i]:.2f}")
        
        return temporal_scores
    
    def _predict_keypoint_position(self, keypoint_idx: int, recent_poses: List[Dict]) -> Tuple[Optional[float], Optional[float]]:
        """Predict where a keypoint should be based on recent movement history"""
        if len(recent_poses) < 2:
            return None, None
        
        # Extract this keypoint's positions from recent poses
        positions = []
        for pose in recent_poses:
            if (keypoint_idx < len(pose['keypoints']) and 
                pose['keypoints'][keypoint_idx][0] != 0 and 
                pose['keypoints'][keypoint_idx][1] != 0):
                positions.append(pose['keypoints'][keypoint_idx][:2])
        
        if len(positions) < 2:
            return None, None
        
        # Calculate velocity (pixels per frame)
        velocities = []
        for j in range(1, len(positions)):
            dx = positions[j][0] - positions[j-1][0]
            dy = positions[j][1] - positions[j-1][1]
            velocities.append([dx, dy])
        
        if not velocities:
            return None, None
        
        # Average velocity over recent frames
        avg_velocity = np.mean(velocities, axis=0)
        
        # Predict next position based on last known position and average velocity
        last_pos = positions[-1]
        predicted_x = last_pos[0] + avg_velocity[0]
        predicted_y = last_pos[1] + avg_velocity[1]
        
        return predicted_x, predicted_y
    
    def _update_pose_history(self, player_id: int, pose: Dict):
        """Update pose history for temporal smoothing"""
        if player_id not in self.pose_history:
            from collections import deque
            self.pose_history[player_id] = deque(maxlen=self.history_length)
        
        # Store the pose in history
        self.pose_history[player_id].append(pose)
        logger.debug(f"🔍 Updated pose history for player {player_id}: {len(self.pose_history[player_id])} poses stored")
    
    def _validate_pose_hips(self, pose: Dict, bbox: List[int]) -> bool:
        """Validate that the pose has hips within the player bounding box"""
        x1, y1, x2, y2 = bbox
        keypoints = pose['keypoints']
        
        # Hip keypoints are indices 11 (left hip) and 12 (right hip)
        hip_indices = [11, 12]
        
        for hip_idx in hip_indices:
            if hip_idx < len(keypoints):
                hip_x, hip_y = keypoints[hip_idx][0], keypoints[hip_idx][1]
                
                # Skip if hip keypoint is invalid (0, 0)
                if hip_x == 0 and hip_y == 0:
                    continue
                
                # Check if hip is within bounding box
                if not (x1 <= hip_x <= x2 and y1 <= hip_y <= y2):
                    logger.warning(f"⚠️  Hip {hip_idx} at ({hip_x:.1f}, {hip_y:.1f}) outside bbox [{x1}, {y1}, {x2}, {y2}]")
                    return False
        
        logger.debug(f"✅ Pose hips validated within bounding box")
        return True
    
    def _attempt_pose_redraw(self, frame: np.ndarray, player: Dict, bbox: List[int], player_idx: int) -> Optional[Dict]:
        """Attempt to redraw pose detection with stricter constraints if hips are invalid"""
        x1, y1, x2, y2 = bbox
        
        # Try with even tighter ROI (no padding) and higher confidence threshold
        player_roi = frame[y1:y2, x1:x2]
        
        if player_roi.size == 0:
            return None
        
        logger.debug(f"🔄 Attempting pose redraw with tighter constraints for player at {bbox}")
        
        # Try multiple scales with stricter filtering
        for scale in [1.0, 1.5]:
            if scale != 1.0:
                roi_height, roi_width = player_roi.shape[:2]
                new_height, new_width = int(roi_height * scale), int(roi_width * scale)
                scaled_roi = cv2.resize(player_roi, (new_width, new_height))
            else:
                scaled_roi = player_roi
            
            # Run pose detection with higher confidence threshold
            results = self.model(scaled_roi, verbose=False, max_det=1, conf=0.6)  # Higher confidence threshold
            
            for result in results:
                if result.keypoints is not None:
                    keypoints = result.keypoints.data[0].cpu().numpy()
                    confidence = result.keypoints.conf[0].cpu().numpy() if result.keypoints.conf is not None else np.ones(len(keypoints))
                    
                    # Scale keypoints back to original ROI size
                    if scale != 1.0:
                        keypoints[:, 0] /= scale
                        keypoints[:, 1] /= scale
                    
                    # Adjust keypoints back to full frame coordinates
                    keypoints[:, 0] += x1
                    keypoints[:, 1] += y1
                    
                    # Create pose data
                    pose_data = {
                        'keypoints': keypoints.tolist(),
                        'scale': scale,
                        'scale_weight': 1.0,
                        'confidence': confidence.tolist(),
                        'player_idx': player_idx,  # Use the passed player_idx
                        'bbox': bbox
                    }
                    
                    # Validate hips
                    if self._validate_pose_hips(pose_data, bbox):
                        logger.debug(f"✅ Pose redraw successful with scale {scale}")
                        return pose_data
        
        logger.warning(f"❌ Pose redraw failed - no valid poses found")
        return None


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
