#!/usr/bin/env python3
"""
Court Detection Demo Script
Uses the TennisCourtDetector model to detect and overlay tennis court keypoints on videos.
Based on the TennisCourtDetector repository: https://github.com/yastrebksv/TennisCourtDetector
"""

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import yaml
import time
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import logging
import argparse
import sys

# Add TennisCourtDetector to path
sys.path.append('TennisCourtDetector')

# Import from TennisCourtDetector
from tracknet import BallTrackerNet
from postprocess import postprocess, refine_kps
from homography import get_trans_matrix, refer_kps

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CourtDetector:
    """Tennis court detection system using deep learning"""
    
    def __init__(self, config_path: str = "config.yaml"):
        """Initialize the court detection system"""
        self.config = self._load_config(config_path)
        self.frame_count = 0
        self.start_time = time.time()
        
        # Initialize model
        self.model = None
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Court detection state
        self.court_keypoints = []
        self.court_confidence = []
        
        # Colinearity-based quality assessment for tennis court lines
        # Based on user's exact specification of which points should be colinear
        self.court_line_groups = {
            # Horizontal lines (top to bottom)
            'top_horizontal': [0, 4, 6, 1],      # Top baseline + service lines
            'bottom_horizontal': [2, 5, 7, 3],   # Bottom baseline + service lines
            'top_service': [8, 12, 9],            # Top service line + net
            'bottom_service': [10, 13, 11],       # Bottom service line + net
            
            # Vertical lines (left to right)
            'left_vertical': [5, 8, 12, 4],      # Left side lines (including left ends of service lines)
            'right_vertical': [6, 9, 11, 7],     # Right side lines (including right ends of service lines)
        }
        
        # Parallel line pairs for additional validation
        self.parallel_line_pairs = [
            # Endlines should be parallel to service lines
            ('endline_top', [1, 3], 'service_line_right', [6, 9, 11, 7]),      # Top endline || right service line
            ('endline_bottom', [2, 0], 'service_line_left', [5, 8, 12, 4]),     # Bottom endline || left service line (corrected)
            
            # All horizontal lines should be parallel to each other
            ('baseline_top', [0, 1], 'baseline_bottom', [2, 3]),                # Top baseline || bottom baseline
            ('baseline_top', [0, 1], 'top_service', [8, 9]),                    # Top baseline || top service line
            ('baseline_top', [0, 1], 'bottom_service', [10, 11]),               # Top baseline || bottom service line
            ('baseline_bottom', [2, 3], 'top_service', [8, 9]),                 # Bottom baseline || top service line
            ('baseline_bottom', [2, 3], 'bottom_service', [10, 11]),            # Bottom baseline || bottom service line
            ('top_service', [8, 9], 'bottom_service', [10, 11]),                # Top service line || bottom service line
            
            # Service line endpoints should be parallel to each other
            ('service_left', [8, 10], 'service_right', [9, 11]),                # Left service endpoints || right service endpoints
            
            # All vertical lines should be parallel to each other
            ('left_vertical', [5, 8, 12, 4], 'right_vertical', [6, 9, 11, 7]), # Left side || right side (corrected)
            ('left_vertical', [5, 8, 12, 4], 'center_service', [8, 12, 10]),   # Left side || center service (corrected)
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
        
        # Analysis results
        self.analysis_results = {
            'total_frames': 0,
            'court_detections': 0,
            'keypoints_detected': 0,
            'processing_times': []
        }
        
        self._initialize_model()
    
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
        """Get default configuration if YAML loading fails"""
        return {
            'models': {
                'court_detector': 'model_tennis_court_det.pt'
            },
            'court_detection': {
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
                'frame_skip': 1
            }
        }
    
    def _initialize_model(self):
        """Initialize the court detection model"""
        try:
            model_path = self.config['models'].get('court_detector')
            if not model_path or not Path(model_path).exists():
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
    
    def detect_court(self, video_path: str, output_path: Optional[str] = None):
        """Detect court in video and overlay keypoints"""
        if not Path(video_path).exists():
            logger.error(f"Video file not found: {video_path}")
            return
        
        if not self.model:
            logger.error("Court detection model not initialized")
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
                
                # Detect court in frame
                start_time = time.time()
                annotated_frame = self._detect_court_in_frame(frame)
                processing_time = time.time() - start_time
                
                # Update statistics
                self.analysis_results['total_frames'] += 1
                self.analysis_results['processing_times'].append(processing_time)
                
                # Display frame
                cv2.imshow('Court Detection Demo', annotated_frame)
                
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
                    cv2.imwrite(f"court_frame_{frame_count:06d}.jpg", annotated_frame)
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
    
    def _detect_court_in_frame(self, frame: np.ndarray) -> np.ndarray:
        """Detect court keypoints in a single frame"""
        annotated_frame = frame.copy()
        
        try:
            logger.info(f"=== Starting court detection for frame ===")
            logger.info(f"Input frame shape: {frame.shape}")
            
            # Get model input dimensions
            input_width = self.config['court_detection']['input_width']
            input_height = self.config['court_detection']['input_height']
            logger.info(f"Model input dimensions: {input_width}x{input_height}")
            
            # Resize frame for model input
            img_resized = cv2.resize(frame, (input_width, input_height))
            logger.info(f"Resized frame shape: {img_resized.shape}")
            
            # Preprocess input
            inp = (img_resized.astype(np.float32) / 255.)
            logger.info(f"Normalized input shape: {inp.shape}")
            logger.info(f"Input value range: [{inp.min():.3f}, {inp.max():.3f}]")
            
            inp = torch.tensor(inp).permute(2, 0, 1)  # Convert to (C, H, W) format
            logger.info(f"Permuted tensor shape: {inp.shape}")
            
            inp = inp.unsqueeze(0).to(self.device)
            logger.info(f"Final input tensor shape: {inp.shape}")
            logger.info(f"Input tensor dtype: {inp.dtype}")
            logger.info(f"Device: {self.device}")
            
            # Run inference
            logger.info("Running model inference...")
            with torch.no_grad():
                out = self.model(inp.float())
                logger.info(f"Raw model output type: {type(out)}")
                if isinstance(out, tuple):
                    logger.info(f"Model output is tuple with {len(out)} elements")
                    out = out[0]
                    logger.info(f"Using first element, shape: {out.shape}")
                else:
                    logger.info(f"Model output shape: {out.shape}")
                
                # Take the first element to remove batch dimension (like in original code)
                out = out[0]
                logger.info(f"After removing batch dimension, shape: {out.shape}")
                
                pred = F.sigmoid(out).detach().cpu().numpy()
                logger.info(f"Final prediction shape: {pred.shape}")
                logger.info(f"Prediction value range: [{pred.min():.3f}, {pred.max():.3f}]")
            
            # Extract keypoints from heatmaps
            logger.info("=== Extracting keypoints from heatmaps ===")
            points = []
            for kps_num in range(14):  # 14 court keypoints
                logger.info(f"Processing keypoint {kps_num}...")
                
                heatmap = (pred[kps_num] * 255).astype(np.uint8)
                logger.info(f"  Heatmap {kps_num} shape: {heatmap.shape}")
                logger.info(f"  Heatmap {kps_num} value range: [{heatmap.min()}, {heatmap.max()}]")
                
                # Postprocess heatmap to get keypoint coordinates (using original parameters)
                try:
                    x_pred, y_pred = postprocess(heatmap, low_thresh=170, max_radius=25)
                    logger.info(f"  Postprocess result: x={x_pred}, y={y_pred}")
                except Exception as e:
                    logger.error(f"  Postprocess failed for keypoint {kps_num}: {e}")
                    x_pred, y_pred = None, None
                
                # Refine keypoints if enabled
                if (self.config['court_detection']['use_refine_kps'] and 
                    kps_num not in [8, 12, 9] and x_pred and y_pred):
                    logger.info(f"  Attempting to refine keypoint {kps_num}...")
                    try:
                        x_pred, y_pred = refine_kps(frame, int(y_pred), int(x_pred))
                        logger.info(f"  Refined result: x={x_pred}, y={y_pred}")
                    except Exception as e:
                        logger.error(f"  Refine failed for keypoint {kps_num}: {e}")
                
                # Scale coordinates back to original video resolution
                if x_pred is not None and y_pred is not None:
                    # Postprocess already scales by 2, so coordinates are at 720x1280
                    # Scale from postprocess output (720x1280) to original video resolution
                    scale_x = frame.shape[1] / (input_width * 2)  # 3840 / (640 * 2) = 3840 / 1280 = 3
                    scale_y = frame.shape[0] / (input_height * 2)  # 2160 / (360 * 2) = 2160 / 720 = 3
                    x_scaled = int(x_pred * scale_x)
                    y_scaled = int(y_pred * scale_y)
                    logger.info(f"  Scaled keypoint {kps_num}: ({x_pred}, {y_pred}) -> ({x_scaled}, {y_scaled})")
                    points.append((x_scaled, y_scaled))
                else:
                    points.append((x_pred, y_pred))
                    logger.info(f"  Final keypoint {kps_num}: ({x_pred}, {y_pred})")
            
            logger.info(f"All keypoints extracted: {points}")
            
            # Assess keypoint quality using colinearity constraints
            logger.info(f"Original points: {points}")
            colinearity_scores = self._assess_colinearity_quality(points)
            logger.info(f"Colinearity scores: {colinearity_scores}")
            
            # Assess parallelism for additional validation
            parallelism_scores = self._assess_parallelism_quality(points)
            logger.info(f"Parallelism scores: {parallelism_scores}")
            
            # Combine colinearity and parallelism scores
            combined_scores = self._combine_quality_scores(colinearity_scores, parallelism_scores)
            logger.info(f"Combined quality scores: {combined_scores}")
            
            # Store for use in drawing
            self.last_colinearity_scores = combined_scores
            
            # Print detailed frame-by-frame analysis
            self._print_frame_analysis(points, colinearity_scores, parallelism_scores, combined_scores)
            
            # Update best keypoints based on quality scores (soft-lock system)
            self._update_best_keypoints(points, combined_scores)
            
            # Apply best keypoints and temporal smoothing for others
            smoothed_points = self._apply_best_keypoints(points, combined_scores)
            logger.info(f"Final smoothed points: {smoothed_points}")
            
            # Apply homography if enabled (using smoothed points)
            logger.info("=== Applying homography ===")
            if self.config['court_detection']['use_homography']:
                logger.info("Homography is enabled, attempting to get transformation matrix...")
                try:
                    matrix_trans = get_trans_matrix(smoothed_points)
                    logger.info(f"Transformation matrix: {matrix_trans}")
                    if matrix_trans is not None:
                        logger.info("Applying perspective transform...")
                        points = cv2.perspectiveTransform(refer_kps, matrix_trans)
                        points = [np.squeeze(x) for x in points]
                        logger.info(f"Transformed points: {points}")
                    else:
                        logger.info("No transformation matrix found, using smoothed points")
                        points = smoothed_points
                except Exception as e:
                    logger.error(f"Homography failed: {e}")
                    logger.info("Using smoothed points without homography")
                    points = smoothed_points
            else:
                logger.info("Homography is disabled, using smoothed points")
                points = smoothed_points
            
            # Draw keypoints on frame
            logger.info("=== Drawing keypoints and lines ===")
            keypoints_detected = 0
            for j, point in enumerate(points):
                if point[0] is not None and point[1] is not None:
                    logger.info(f"Drawing keypoint {j} at ({int(point[0])}, {int(point[1])})")
                    
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
                    elif hasattr(self, 'last_colinearity_scores') and j in self.last_colinearity_scores:
                        colinearity_score = self.last_colinearity_scores[j]
                        if colinearity_score > 0.7:
                            color = (0, 255, 0)  # Bright green for high colinearity
                            thickness = 3
                        elif colinearity_score > 0.5:
                            color = (0, 200, 0)  # Medium green
                            thickness = 2
                        else:
                            color = (0, 150, 0)  # Dark green for low colinearity
                            thickness = 1
                    else:
                        # Fallback to temporal stability
                        if j in self.keypoint_history and len(self.keypoint_history[j]) >= 10:
                            color = (0, 255, 0)  # Bright green for stable
                            thickness = 3
                        elif j in self.keypoint_history and len(self.keypoint_history[j]) >= 5:
                            color = (0, 200, 0)  # Medium green
                            thickness = 2
                        else:
                            color = (0, 150, 0)  # Dark green
                            thickness = 1
                    
                    # Draw keypoint
                    cv2.circle(annotated_frame, (int(point[0]), int(point[1])),
                              radius=5, color=color, thickness=-1)  # Filled circle
                    cv2.circle(annotated_frame, (int(point[0]), int(point[1])),
                              radius=8, color=color, thickness=thickness)   # Outline
                    
                    # Add keypoint number and status info
                    if j in self.best_keypoints:
                        best_score = self.best_keypoints[j][2]
                        if best_score <= self.quality_threshold:
                            status_text = f"{j}[SOFT_LOCKED]"
                        else:
                            status_text = f"{j}[BEST:{best_score:.2f}]"
                    else:
                        stability_text = f"{j}"
                        if j in self.keypoint_history:
                            stability_text += f"({len(self.keypoint_history[j])})"
                        status_text = stability_text
                    
                    cv2.putText(annotated_frame, status_text, (int(point[0]) + 10, int(point[1]) - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                    
                    keypoints_detected += 1
                else:
                    logger.info(f"Keypoint {j} is None, skipping drawing")
            
            logger.info(f"Total keypoints drawn: {keypoints_detected}/14")
            
            # Draw court lines if we have enough keypoints
            if keypoints_detected >= 4:  # Lowered threshold to show more lines
                logger.info(f"Drawing court lines (have {keypoints_detected} keypoints)")
                self._draw_court_lines(annotated_frame, points)
                self.analysis_results['court_detections'] += 1
            else:
                logger.info(f"Not enough keypoints ({keypoints_detected}) to draw court lines")
            
            self.analysis_results['keypoints_detected'] += keypoints_detected
            
            # Add frame information
            self._add_frame_info(annotated_frame, keypoints_detected)
            
        except Exception as e:
            logger.error(f"Court detection error: {e}")
        
        return annotated_frame
    
    def _calculate_keypoint_confidence(self, points: List[Tuple], pred: np.ndarray) -> List[Tuple]:
        """Calculate confidence for each keypoint based on heatmap intensity"""
        points_with_confidence = []
        
        for kps_num, (x_pred, y_pred) in enumerate(points):
            if x_pred is not None and y_pred is not None:
                # Calculate confidence based on heatmap intensity at the detected position
                # Scale coordinates back to heatmap coordinates
                scale_x = pred.shape[2] / (self.config['court_detection']['input_width'] * 2)
                scale_y = pred.shape[1] / (self.config['court_detection']['input_height'] * 2)
                
                heatmap_x = int(x_pred * scale_x)
                heatmap_y = int(y_pred * scale_y)
                
                # Ensure coordinates are within bounds
                heatmap_x = max(0, min(heatmap_x, pred.shape[2] - 1))
                heatmap_y = max(0, min(heatmap_y, pred.shape[1] - 1))
                
                # Get confidence from heatmap
                confidence = pred[kps_num, heatmap_y, heatmap_x]
                
                # Additional confidence factors
                # 1. Distance from center (court keypoints should be away from edges)
                center_x = pred.shape[2] / 2
                center_y = pred.shape[1] / 2
                distance_from_center = np.sqrt((heatmap_x - center_x)**2 + (heatmap_y - center_y)**2)
                distance_confidence = 1.0 - (distance_from_center / np.sqrt(center_x**2 + center_y**2))
                
                # 2. Local neighborhood consistency
                neighborhood_size = 5
                y_start = max(0, heatmap_y - neighborhood_size)
                y_end = min(pred.shape[1], heatmap_y + neighborhood_size + 1)
                x_start = max(0, heatmap_x - neighborhood_size)
                x_end = min(pred.shape[2], heatmap_x + neighborhood_size + 1)
                
                local_region = pred[kps_num, y_start:y_end, x_start:x_end]
                local_consistency = np.std(local_region)  # Lower std = more consistent
                consistency_confidence = 1.0 - min(local_consistency, 0.5) / 0.5
                
                # Combine confidence factors
                final_confidence = (confidence * 0.6 + distance_confidence * 0.2 + consistency_confidence * 0.2)
                final_confidence = max(0.0, min(1.0, final_confidence))
                
                points_with_confidence.append((x_pred, y_pred, final_confidence))
                logger.debug(f"Keypoint {kps_num} confidence: {final_confidence:.3f}")
            else:
                points_with_confidence.append((x_pred, y_pred, 0.0))
        
        return points_with_confidence
    
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
            # but we can use them to validate other points
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
                                
                                # Additional validation: check if points are on the same side of the court
                                # This helps prevent points from being placed on the wrong side
                                if line_name in ['left_vertical', 'right_vertical']:
                                    # For vertical lines, x-coordinates should be similar
                                    x_coords = [x, x1, x2]
                                    x_std = np.std(x_coords)
                                    if x_std < 50:  # Points should be roughly aligned vertically
                                        colinearity_measure *= 0.8  # Reduce score (better) for good alignment
                                    else:
                                        colinearity_measure *= 1.2  # Increase score (worse) for poor alignment
                                
                                colinearity_measures.append(colinearity_measure)
                    
                    if colinearity_measures:
                        # Average colinearity with all combinations
                        avg_colinearity = np.mean(colinearity_measures)
                        
                        # Update score if this line gives a better result
                        if point_idx not in colinearity_scores or avg_colinearity > colinearity_scores[point_idx]:
                            colinearity_scores[point_idx] = avg_colinearity
                            logger.info(f"Point {point_idx} on {line_name}: NEW BEST colinearity = {avg_colinearity:.3f} (from {len(colinearity_measures)} combinations)")
                        else:
                            logger.debug(f"Point {point_idx} on {line_name}: colinearity = {avg_colinearity:.3f} (from {len(colinearity_measures)} combinations)")
                        
                        # Show detailed colinearity breakdown for debugging
                        if line_name in ['top_horizontal', 'bottom_horizontal', 'left_vertical', 'right_vertical']:
                            logger.debug(f"  Point {point_idx} colinearity details:")
                            for j, measure in enumerate(colinearity_measures[:3]):  # Show first 3 measures
                                logger.debug(f"    Combination {j+1}: {measure:.3f}")
                    else:
                        logger.debug(f"Point {point_idx} on {line_name}: no colinearity measures calculated")
        
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
            
            # Debug: Check if point 10 is in any parallelism checks
            if 10 in line1_points or 10 in line2_points:
                logger.info(f"Point 10 found in parallelism check: {line_name} ({line1_points}) || {parallel_line_name} ({line2_points})")
                logger.info(f"  Valid line1: {valid_line1}")
                logger.info(f"  Valid line2: {valid_line2}")
            
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
                                # Dot product gives cosine of angle between vectors
                                # Parallel lines have dot product close to ±1
                                # We want to minimize the angle difference, so convert to error score
                                dot_product = abs(np.dot(line1_unit, line2_unit))
                                # Convert to error: 0 = perfect parallel, 1 = perpendicular
                                parallelism_error = 1.0 - dot_product
                                line2_vectors.append(parallelism_error)
                    
                    if line2_vectors:
                        avg_parallelism = np.mean(line2_vectors)
                        
                        # Assign scores to all points in line1
                        for point_idx, _ in valid_line1:
                            if point_idx not in parallelism_scores:
                                parallelism_scores[point_idx] = []
                            parallelism_scores[point_idx].append(avg_parallelism)
                        
                        logger.info(f"{line_name} || {parallel_line_name}: parallelism = {avg_parallelism:.3f}")
                        
                        # Show detailed parallelism breakdown for key lines
                        if line_name in ['endline_top', 'endline_bottom', 'baseline_top', 'baseline_bottom']:
                            logger.debug(f"  {line_name} ({start_idx}->{end_idx}) parallelism details:")
                            for j, score in enumerate(line2_vectors[:3]):  # Show first 3 scores
                                logger.debug(f"    {parallel_line_name} segment {j+1}: {score:.3f}")
                            logger.debug(f"  Line vector: [{line1_vector[0]:.1f}, {line1_vector[1]:.1f}]")
        
        # Average all parallelism scores for each point and fill in missing scores
        final_parallelism_scores = {}
        for i in range(len(points)):
            if i in parallelism_scores and parallelism_scores[i]:
                # Average all parallelism scores for this point
                final_parallelism_scores[i] = np.mean(parallelism_scores[i])
            else:
                final_parallelism_scores[i] = 1.0  # Default score (high error = bad)
        
        return final_parallelism_scores
    
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
                        logger.info(f"🌟 NEW BEST keypoint {kps_num} at ({x_pred}, {y_pred}) with score {quality_score:.3f}")
                else:
                    # Compare with existing best
                    best_x, best_y, best_score = self.best_keypoints[kps_num]
                    
                    if quality_score < best_score:
                        # This is better! Replace it
                        old_score = best_score
                        self.best_keypoints[kps_num] = (x_pred, y_pred, quality_score)
                        logger.info(f"🔄 IMPROVED keypoint {kps_num}: {old_score:.3f} → {quality_score:.3f}")
                        logger.info(f"   Position: ({best_x}, {best_y}) → ({x_pred}, {y_pred})")
                    elif quality_score == best_score:
                        # Same quality, could average positions for stability
                        if len(self.keypoint_history[kps_num]) >= self.min_history_frames:
                            # Calculate average of recent good detections
                            good_positions = [(x, y) for (x, y), conf in zip(self.keypoint_history[kps_num], self.keypoint_confidence[kps_num]) 
                                            if conf <= self.quality_threshold]
                            if len(good_positions) >= 2:
                                avg_x = sum(x for x, y in good_positions) / len(good_positions)
                                avg_y = sum(y for x, y in good_positions) / len(good_positions)
                                self.best_keypoints[kps_num] = (int(avg_x), int(avg_y), quality_score)
                                logger.debug(f"Stabilized keypoint {kps_num} at ({int(avg_x)}, {int(avg_y)})")
    
    def _apply_best_keypoints(self, points: List[Tuple], quality_scores: Dict[int, float]) -> List[Tuple]:
        """Apply best keypoints and temporal smoothing for others"""
        final_points = []
        
        for kps_num, (x_pred, y_pred) in enumerate(points):
            if kps_num in self.best_keypoints:
                # Use best position - this is our "soft-locked" position
                best_x, best_y, best_score = self.best_keypoints[kps_num]
                final_points.append((best_x, best_y))
                logger.debug(f"Using BEST keypoint {kps_num}: ({best_x}, {best_y}) - score: {best_score:.3f}")
            else:
                # Apply temporal smoothing for non-best keypoints
                if len(self.keypoint_history.get(kps_num, [])) > 0:
                    # Weighted average based on quality scores (lower is better, so invert for weights)
                    weights = [1.0 / (conf + 0.1) for conf in self.keypoint_confidence.get(kps_num, [])]  # Avoid division by zero
                    total_weight = sum(weights)
                    if total_weight > 0:
                        weighted_x = sum(x * w for (x, y), w in zip(self.keypoint_history[kps_num], weights)) / total_weight
                        weighted_y = sum(y * w for (x, y), w in zip(self.keypoint_history[kps_num], weights)) / total_weight
                        
                        final_points.append((int(weighted_x), int(weighted_y)))
                        logger.debug(f"Using SMOOTHED keypoint {kps_num}: ({int(weighted_x)}, {int(weighted_y)})")
                    else:
                        final_points.append((x_pred, y_pred))
                        logger.debug(f"Using ORIGINAL keypoint {kps_num}: ({x_pred}, {y_pred})")
                else:
                    final_points.append((x_pred, y_pred))
                    logger.debug(f"Using ORIGINAL keypoint {kps_num}: ({x_pred}, {y_pred})")
        
        return final_points
    
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
    
    def _print_frame_analysis(self, points: List[Tuple], colinearity_scores: Dict[int, float], parallelism_scores: Dict[int, float], combined_scores: Dict[int, float]):
        """Print detailed frame-by-frame analysis of keypoint quality"""
        logger.info("=" * 80)
        logger.info("FRAME ANALYSIS")
        logger.info("=" * 80)
        
        # Print header
        logger.info(f"{'KP':<3} {'X':<6} {'Y':<6} {'Col':<6} {'Par':<6} {'Comb':<6} {'Status':<15} {'History':<10}")
        logger.info("-" * 80)
        
        for i in range(len(points)):
            if points[i][0] is not None and points[i][1] is not None:
                x, y = int(points[i][0]), int(points[i][1])
                col_score = colinearity_scores.get(i, 0.0)
                par_score = parallelism_scores.get(i, 0.0)
                comb_score = combined_scores.get(i, 0.0)
                
                # Determine status
                if i in self.best_keypoints:
                    best_score = self.best_keypoints[i][2]
                    if best_score <= self.quality_threshold:
                        status = "SOFT_LOCKED"
                    else:
                        status = "BEST_KNOWN"
                elif i in self.keypoint_history and len(self.keypoint_history[i]) >= self.min_history_frames:
                    if comb_score <= self.quality_threshold:
                        status = "READY_FOR_BEST"
                    else:
                        status = "STABLE"
                elif i in self.keypoint_history and len(self.keypoint_history[i]) >= 3:
                    status = "BUILDING"
                else:
                    status = "NEW"
                
                # History info
                history_count = len(self.keypoint_history.get(i, []))
                history_str = f"{history_count}f"
                
                logger.info(f"{i:<3} {x:<6} {y:<6} {col_score:<6.3f} {par_score:<6.3f} {comb_score:<6.3f} {status:<15} {history_str:<10}")
            else:
                logger.info(f"{i:<3} {'None':<6} {'None':<6} {'N/A':<6} {'N/A':<6} {'N/A':<6} {'MISSING':<15} {'N/A':<10}")
        
        # Print summary statistics
        soft_locked_count = sum(1 for i in range(len(points)) 
                               if i in self.best_keypoints and self.best_keypoints[i][2] <= self.quality_threshold)
        best_known_count = sum(1 for i in range(len(points)) 
                              if i in self.best_keypoints and self.best_keypoints[i][2] > self.quality_threshold)
        ready_count = sum(1 for i in range(len(points)) 
                         if i in self.keypoint_history and len(self.keypoint_history[i]) >= self.min_history_frames 
                         and combined_scores.get(i, 0) <= self.quality_threshold 
                         and i not in self.best_keypoints)
        stable_count = sum(1 for i in range(len(points)) 
                          if i in self.keypoint_history and len(self.keypoint_history[i]) >= 3 
                          and i not in self.best_keypoints)
        
        logger.info("-" * 80)
        logger.info(f"SUMMARY: {soft_locked_count} SOFT_LOCKED, {best_known_count} BEST_KNOWN, {ready_count} READY, {stable_count} STABLE")
        logger.info("=" * 80)
    
    def _update_locked_keypoints(self, points: List[Tuple], colinearity_scores: Dict[int, float]):
        """Update which keypoints should be locked based on colinearity scores"""
        for kps_num, (x_pred, y_pred) in enumerate(points):
            if x_pred is not None and y_pred is not None:
                colinearity_score = colinearity_scores.get(kps_num, 0.0)
                
                if kps_num not in self.keypoint_history:
                    self.keypoint_history[kps_num] = []
                    self.keypoint_confidence[kps_num] = []
                
                # Add current detection to history
                self.keypoint_history[kps_num].append((x_pred, y_pred))
                self.keypoint_confidence[kps_num].append(colinearity_score)
                
                # Keep only recent history
                if len(self.keypoint_history[kps_num]) > self.history_length:
                    self.keypoint_history[kps_num].pop(0)
                    self.keypoint_confidence[kps_num].pop(0)
                
                # Check if we should lock this keypoint
                # Now LOWER scores are better, so we lock when score is BELOW threshold
                if (len(self.keypoint_history[kps_num]) >= self.min_lock_frames and 
                    colinearity_score <= self.lock_threshold):
                    
                    # Calculate average position from recent high-confidence detections
                    high_conf_positions = [(x, y) for (x, y), conf in zip(self.keypoint_history[kps_num], self.keypoint_confidence[kps_num]) 
                                         if conf >= self.lock_threshold]
                    
                    if len(high_conf_positions) >= self.min_lock_frames:
                        avg_x = sum(x for x, y in high_conf_positions) / len(high_conf_positions)
                        avg_y = sum(y for x, y in high_conf_positions) / len(high_conf_positions)
                        
                        self.locked_keypoints[kps_num] = (int(avg_x), int(avg_y), colinearity_score)
                        logger.info(f"🔒 LOCKED keypoint {kps_num} at ({int(avg_x)}, {int(avg_y)}) with confidence {colinearity_score:.3f}")
                        logger.info(f"   Position averaged from {len(high_conf_positions)} high-confidence detections")
                        logger.info(f"   History: {len(self.keypoint_history[kps_num])} frames, threshold: {self.lock_threshold}")
                
                # Unlock if confidence gets worse significantly
                elif kps_num in self.locked_keypoints:
                    current_confidence = colinearity_score
                    locked_confidence = self.locked_keypoints[kps_num][2]
                    
                    # Now LOWER scores are better, so unlock if score gets much worse
                    if current_confidence > locked_confidence * 2.0:  # Score gets 2x worse
                        logger.info(f"🔓 UNLOCKING keypoint {kps_num} due to confidence drop: {current_confidence:.3f} > {locked_confidence:.3f}")
                        logger.info(f"   Previous locked confidence: {locked_confidence:.3f}, current: {current_confidence:.3f}")
                        del self.locked_keypoints[kps_num]
    
    def _apply_locked_keypoints(self, points: List[Tuple], colinearity_scores: Dict[int, float]) -> List[Tuple]:
        """Apply locked keypoints and temporal smoothing for unlocked ones"""
        final_points = []
        
        for kps_num, (x_pred, y_pred) in enumerate(points):
            if kps_num in self.locked_keypoints:
                # Use locked position - NEVER change this
                locked_x, locked_y, locked_conf = self.locked_keypoints[kps_num]
                final_points.append((locked_x, locked_y))
                logger.debug(f"Using LOCKED keypoint {kps_num}: ({locked_x}, {locked_y}) - POSITION FIXED")
            else:
                # Apply temporal smoothing for unlocked keypoints
                if len(self.keypoint_history.get(kps_num, [])) > 0:
                    # Weighted average based on colinearity scores
                    total_weight = sum(self.keypoint_confidence.get(kps_num, []))
                    if total_weight > 0:
                        weighted_x = sum(x * w for (x, y), w in zip(self.keypoint_history[kps_num], self.keypoint_confidence[kps_num])) / total_weight
                        weighted_y = sum(y * w for (x, y), w in zip(self.keypoint_history[kps_num], self.keypoint_confidence[kps_num])) / total_weight
                        
                        final_points.append((int(weighted_x), int(weighted_y)))
                        logger.debug(f"Using SMOOTHED keypoint {kps_num}: ({int(weighted_x)}, {int(weighted_y)})")
                    else:
                        final_points.append((x_pred, y_pred))
                        logger.debug(f"Using ORIGINAL keypoint {kps_num}: ({x_pred}, {y_pred})")
                else:
                    final_points.append((x_pred, y_pred))
                    logger.debug(f"Using ORIGINAL keypoint {kps_num}: ({x_pred}, {y_pred})")
        
        return final_points
    
    def _apply_colinearity_smoothing(self, points: List[Tuple], colinearity_scores: Dict[int, float]) -> List[Tuple]:
        """Apply temporal smoothing with colinearity-based filtering"""
        smoothed_points = []
        
        for kps_num, (x_pred, y_pred) in enumerate(points):
            if kps_num not in self.keypoint_history:
                self.keypoint_history[kps_num] = []
                self.keypoint_confidence[kps_num] = []
            
            # Add current detection to history if it has good colinearity
            if x_pred is not None and y_pred is not None:
                colinearity_score = colinearity_scores.get(kps_num, 0.5)
                
                # Only add to history if colinearity is good enough
                if colinearity_score < 0.8:  # Threshold for acceptable colinearity (lower is better)
                    self.keypoint_history[kps_num].append((x_pred, y_pred))
                    self.keypoint_confidence[kps_num].append(colinearity_score)
                    
                    # Keep only recent history
                    if len(self.keypoint_history[kps_num]) > self.history_length:
                        self.keypoint_history[kps_num].pop(0)
                        self.keypoint_confidence[kps_num].pop(0)
            
            # Calculate smoothed position
            if len(self.keypoint_history[kps_num]) > 0:
                # Weighted average based on colinearity scores
                total_weight = sum(self.keypoint_confidence[kps_num])
                if total_weight > 0:
                    weighted_x = sum(x * w for (x, y), w in zip(self.keypoint_history[kps_num], self.keypoint_confidence[kps_num])) / total_weight
                    weighted_y = sum(y * w for (x, y), w in zip(self.keypoint_history[kps_num], self.keypoint_confidence[kps_num])) / total_weight
                    
                    # Apply outlier rejection based on colinearity
                    if len(self.keypoint_history[kps_num]) >= 5:
                        # Calculate distances from current weighted average
                        distances = [np.sqrt((x - weighted_x)**2 + (y - weighted_y)**2) 
                                   for (x, y) in self.keypoint_history[kps_num]]
                        mean_distance = np.mean(distances)
                        std_distance = np.std(distances)
                        
                        # Filter out points that are too far from the mean
                        filtered_positions = []
                        filtered_weights = []
                        for i, ((x, y), w) in enumerate(zip(self.keypoint_history[kps_num], self.keypoint_confidence[kps_num])):
                            if distances[i] <= mean_distance + 1.5 * std_distance:  # Less strict outlier rejection
                                filtered_positions.append((x, y))
                                filtered_weights.append(w)
                        
                        if len(filtered_positions) > 0:
                            total_weight = sum(filtered_weights)
                            weighted_x = sum(x * w for (x, y), w in zip(filtered_positions, filtered_weights)) / total_weight
                            weighted_y = sum(y * w for (x, y), w in zip(filtered_positions, filtered_weights)) / total_weight
                    
                    smoothed_points.append((int(weighted_x), int(weighted_y)))
                    logger.debug(f"Smoothed keypoint {kps_num}: ({int(weighted_x)}, {int(weighted_y)}) from {len(self.keypoint_history[kps_num])} frames")
                else:
                    smoothed_points.append((None, None))
            else:
                smoothed_points.append((None, None))
        
        return smoothed_points
    
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
            
            # Calculate smoothed position
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
                    logger.debug(f"Smoothed keypoint {kps_num}: ({int(weighted_x)}, {int(weighted_y)}) from {len(self.keypoint_history[kps_num])} frames")
                else:
                    smoothed_points.append((None, None))
            else:
                smoothed_points.append((None, None))
        
        return smoothed_points
    
    def _draw_court_lines(self, frame: np.ndarray, points: List[Tuple]):
        """Draw court lines connecting keypoints"""
        logger.info("=== Drawing court lines ===")
        logger.info(f"Received {len(points)} points for line drawing")
        
        try:
            # Define court line connections based on keypoint indices
            # These are the standard tennis court line connections
            court_lines = [
                (0, 1),   # Baseline top
                (2, 3),   # Baseline bottom  
                (4, 5),   # Left inner line
                (6, 7),   # Right inner line
                (8, 9),   # Top inner line
                (10, 11), # Bottom inner line
                (12, 13)  # Middle line
            ]
            
            logger.info(f"Attempting to draw {len(court_lines)} court lines")
            
            # Draw lines
            for start_idx, end_idx in court_lines:
                logger.info(f"Processing line {start_idx}-{end_idx}")
                
                if (start_idx < len(points) and end_idx < len(points) and
                    points[start_idx][0] is not None and points[start_idx][1] is not None and
                    points[end_idx][0] is not None and points[end_idx][1] is not None):
                    
                    start_point = (int(points[start_idx][0]), int(points[start_idx][1]))
                    end_point = (int(points[end_idx][0]), int(points[end_idx][1]))
                    
                    # Draw thicker, more visible lines
                    cv2.line(frame, start_point, end_point, (255, 0, 0), 3)  # Blue lines, thickness 3
                    logger.info(f"Drew court line from {start_point} to {end_point}")
                else:
                    logger.info(f"Could not draw line {start_idx}-{end_idx}: points not available")
                    if start_idx < len(points):
                        logger.info(f"  Start point {start_idx}: {points[start_idx]}")
                    if end_idx < len(points):
                        logger.info(f"  End point {end_idx}: {points[end_idx]}")
            
            # Also draw some additional court structure lines for better visibility
            # Draw service boxes
            logger.info("Attempting to draw service box lines...")
            if (len(points) >= 14 and 
                points[8][0] is not None and points[8][1] is not None and
                points[9][0] is not None and points[9][1] is not None and
                points[10][0] is not None and points[10][1] is not None and
                points[11][0] is not None and points[11][1] is not None):
                
                # Service box lines
                cv2.line(frame, 
                         (int(points[8][0]), int(points[8][1])), 
                         (int(points[10][0]), int(points[10][1])), 
                         (0, 255, 255), 2)  # Yellow lines
                cv2.line(frame, 
                         (int(points[9][0]), int(points[9][1])), 
                         (int(points[11][0]), int(points[11][1])), 
                         (0, 255, 255), 2)  # Yellow lines
                logger.info("Drew service box lines")
            else:
                logger.info("Could not draw service box lines - missing required keypoints")
        
        except Exception as e:
            logger.error(f"Error drawing court lines: {e}")
            logger.error(f"Points: {points}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
    
    def _add_frame_info(self, frame: np.ndarray, keypoints_detected: int):
        """Add frame information overlay"""
        # Frame counter
        cv2.putText(frame, f"Frame: {self.analysis_results['total_frames']}", 
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Statistics
        stats_text = [
            f"Court Detections: {self.analysis_results['court_detections']}",
            f"Keypoints: {keypoints_detected}/14",
            f"Total Keypoints: {self.analysis_results['keypoints_detected']}"
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
        """Print analysis summary"""
        print("\n" + "="*50)
        print("COURT DETECTION ANALYSIS SUMMARY")
        print("="*50)
        print(f"Total frames processed: {self.analysis_results['total_frames']}")
        print(f"Court detections: {self.analysis_results['court_detections']}")
        print(f"Total keypoints detected: {self.analysis_results['keypoints_detected']}")
        
        if self.analysis_results['processing_times']:
            avg_time = np.mean(self.analysis_results['processing_times'])
            min_time = np.min(self.analysis_results['processing_times'])
            max_time = np.max(self.analysis_results['processing_times'])
            print(f"Processing time - Avg: {avg_time:.3f}s, Min: {min_time:.3f}s, Max: {max_time:.3f}s")
        
        total_time = time.time() - self.start_time
        print(f"Total analysis time: {total_time:.2f}s")
        print("="*50)


def main():
    """Main function to run the court detection demo"""
    parser = argparse.ArgumentParser(description='Court Detection Demo')
    parser.add_argument('--video', '-v', type=str, default='tennis_test.mp4',
                       help='Path to input video file')
    parser.add_argument('--output', '-o', type=str, default=None,
                       help='Path to output video file (optional)')
    parser.add_argument('--config', '-c', type=str, default='config.yaml',
                       help='Path to configuration file')
    
    args = parser.parse_args()
    
    # Check if video exists
    if not Path(args.video).exists():
        print(f"Error: Video file '{args.video}' not found!")
        print("Available video files:")
        for video_file in Path('.').glob('*.mp4'):
            print(f"  - {video_file}")
        return
    
    print("🏟️ Court Detection Demo Starting...")
    print(f"📹 Input video: {args.video}")
    if args.output:
        print(f"💾 Output video: {args.output}")
    print(f"⚙️  Config file: {args.config}")
    print("\nControls:")
    print("  - Press 'q' to quit")
    print("  - Press 'p' to pause/resume")
    print("  - Press 's' to save current frame")
    print("\nInitializing model...")
    
    # Create and run detector
    detector = CourtDetector(args.config)
    detector.detect_court(args.video, args.output)


if __name__ == "__main__":
    main()
