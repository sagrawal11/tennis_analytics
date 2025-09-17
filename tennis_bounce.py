#!/usr/bin/env python3
"""
Tennis Ball Bounce Detection System

Uses a pre-trained CatBoost model to detect ball bounces from trajectory data.
Analyzes ball position changes over time to identify when the ball hits the ground.

Usage:
    python tennis_bounce.py --video tennis_test5.mp4 --csv tennis_analysis_data.csv --viewer
"""

import cv2
import numpy as np
import pandas as pd
import catboost as ctb
import argparse
import logging
from typing import List, Tuple, Optional, Dict
from collections import deque
import os
import torch
import torch.nn.functional as F

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')
logger = logging.getLogger(__name__)


class BounceDetector:
    """Ball bounce detection using CatBoost model with court boundary validation"""
    
    def __init__(self, model_path: str):
        """Initialize bounce detector with pre-trained model"""
        self.model_path = model_path
        self.model = None
        self.num_feature_frames = 2  # Model expects 12 features = 2 frames × 6 features
        self.ball_trajectory = deque(maxlen=50)  # Store more ball positions for temporal analysis
        self.bounce_threshold = 0.95  # Threshold for bounce detection (much higher = very selective)
        self.min_bounce_gap_frames = 30  # Minimum frames between bounces (1 second at 30fps)
        self.last_bounce_frame = -1  # Track last bounce frame
        
        # Court boundary validation
        self.court_keypoints = None  # Will be set from CSV data
        self.court_boundary_polygon = None  # Computed court boundary
        
        # Load the model
        self._load_model()
        
        logger.info(f"Bounce detector initialized with model: {model_path}")
    
    def set_court_keypoints(self, court_keypoints: List[Tuple]):
        """Set court keypoints for boundary validation"""
        self.court_keypoints = court_keypoints
        self._compute_court_boundary()
    
    def _compute_court_boundary(self):
        """Compute court boundary polygon from keypoints"""
        if not self.court_keypoints or len(self.court_keypoints) < 4:
            self.court_boundary_polygon = None
            return
        
        # Extract valid keypoints (corners of the court)
        # Keypoints 0-3 are typically the court corners
        valid_points = []
        for i in range(min(4, len(self.court_keypoints))):
            point = self.court_keypoints[i]
            if point[0] is not None and point[1] is not None:
                valid_points.append((int(point[0]), int(point[1])))
        
        if len(valid_points) >= 4:
            # Create a polygon from the court corners
            # Order: top-left, top-right, bottom-right, bottom-left
            self.court_boundary_polygon = np.array(valid_points, dtype=np.int32)
            logger.info(f"Court boundary computed with {len(valid_points)} points")
        else:
            self.court_boundary_polygon = None
            logger.warning("Not enough valid court keypoints for boundary computation")
    
    def _is_within_court(self, x: float, y: float) -> bool:
        """Check if a point is within the court boundaries"""
        if self.court_boundary_polygon is None:
            return True  # If no court boundary, allow all bounces
        
        try:
            point = (int(x), int(y))
            result = cv2.pointPolygonTest(self.court_boundary_polygon, point, False)
            return result >= 0  # Inside or on the boundary
        except Exception as e:
            logger.warning(f"Error checking court boundary: {e}")
            return True  # If error, allow the bounce
    
    def _load_model(self):
        """Load the pre-trained CatBoost model"""
        try:
            if not os.path.exists(self.model_path):
                raise FileNotFoundError(f"Model file not found: {self.model_path}")
            
            self.model = ctb.CatBoostRegressor()
            self.model.load_model(self.model_path)
            logger.info("CatBoost model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def add_ball_position(self, ball_x: Optional[float], ball_y: Optional[float], frame_number: int):
        """Add ball position to trajectory history"""
        if ball_x is not None and ball_y is not None:
            self.ball_trajectory.append((ball_x, ball_y, frame_number))
        else:
            # Add None for missing ball positions
            self.ball_trajectory.append((None, None, frame_number))
    
    def detect_bounce(self, current_frame: int) -> Tuple[bool, float]:
        """
        Detect if there's a bounce in the current trajectory within court boundaries
        
        IMPORTANT: The model predicts status==2 (ball NOT bouncing), so we need to invert the logic!
        - High confidence = consistent movement (no bounce)
        - Low confidence = bounce or direction change
        
        Returns:
            Tuple of (is_bounce, confidence)
        """
        if len(self.ball_trajectory) < self.num_feature_frames * 2:
            return False, 0.0
        
        # Check minimum time gap between bounces
        if current_frame - self.last_bounce_frame < self.min_bounce_gap_frames:
            return False, 0.0
        
        # Extract features for the model
        features = self._extract_features()
        if features is None:
            return False, 0.0
        
        try:
            # Predict status==2 probability (ball NOT bouncing)
            status2_confidence = self.model.predict([features])[0]
            
            # Invert the logic: low confidence in "not bouncing" = high confidence in "bouncing"
            bounce_confidence = 1.0 - status2_confidence
            is_bounce = bounce_confidence > self.bounce_threshold
            
            # COURT BOUNDARY VALIDATION: Only allow bounces within court boundaries
            if is_bounce:
                # Get the current ball position for court boundary check
                current_ball_pos = None
                for x, y, frame in reversed(self.ball_trajectory):
                    if x is not None and y is not None and frame == current_frame:
                        current_ball_pos = (x, y)
                        break
                
                if current_ball_pos:
                    x, y = current_ball_pos
                    if not self._is_within_court(x, y):
                        logger.info(f"Frame {current_frame}: Bounce detected but outside court boundaries at ({x:.1f}, {y:.1f})")
                        is_bounce = False
                        bounce_confidence = 0.0
                    else:
                        logger.info(f"Frame {current_frame}: Valid bounce within court at ({x:.1f}, {y:.1f})")
            
            # If bounce detected, update last bounce frame
            if is_bounce:
                self.last_bounce_frame = current_frame
            
            # Debug: Log only high confidence predictions
            if bounce_confidence > 0.9:  # Only log very high confidence predictions
                logger.info(f"Frame {current_frame}: Status2={status2_confidence:.3f}, Bounce={bounce_confidence:.3f}, is_bounce={is_bounce}")
            
            return is_bounce, bounce_confidence
            
        except Exception as e:
            logger.warning(f"Error in bounce prediction: {e}")
            return False, 0.0
    
    def _extract_features(self) -> Optional[List[float]]:
        """
        Extract features for the CatBoost model using trajectory analysis.
        
        From high-up perspective looking down at court:
        - Ball bounces happen when it hits the ground within court bounds
        - Bounces show as changes in trajectory direction/velocity, not Y-coordinate patterns
        - We look for sharp changes in ball movement direction or speed
        """
        # Get all valid ball positions
        valid_positions = [(x, y, frame) for x, y, frame in self.ball_trajectory 
                          if x is not None and y is not None]
        
        if len(valid_positions) < 5:  # Need at least 5 frames for analysis
            return None
        
        # Use a larger temporal window for better bounce detection
        window_size = min(10, len(valid_positions))
        recent_positions = valid_positions[-window_size:]
        
        # Extract x, y coordinates
        x_coords = [pos[0] for pos in recent_positions]
        y_coords = [pos[1] for pos in recent_positions]
        
        # Find the point with the most dramatic trajectory change
        # Look for sharp changes in direction or velocity
        max_change_idx = 0
        max_change = 0
        
        for i in range(1, len(x_coords) - 1):
            # Calculate velocity vectors before and after this point
            prev_dx = x_coords[i] - x_coords[i-1]
            prev_dy = y_coords[i] - y_coords[i-1]
            next_dx = x_coords[i+1] - x_coords[i]
            next_dy = y_coords[i+1] - y_coords[i]
            
            # Calculate change in direction (angle change)
            if prev_dx != 0 or prev_dy != 0:
                prev_angle = np.arctan2(prev_dy, prev_dx)
                next_angle = np.arctan2(next_dy, next_dx)
                angle_change = abs(next_angle - prev_angle)
                # Normalize angle change to 0-π
                if angle_change > np.pi:
                    angle_change = 2*np.pi - angle_change
                
                # Calculate change in speed
                prev_speed = np.sqrt(prev_dx**2 + prev_dy**2)
                next_speed = np.sqrt(next_dx**2 + next_dy**2)
                speed_change = abs(next_speed - prev_speed)
                
                # Combined change metric (weighted)
                total_change = angle_change * 10 + speed_change  # Angle change is more important
                
                if total_change > max_change:
                    max_change = total_change
                    max_change_idx = i
        
        # Ensure we have enough frames before and after the change point
        if max_change_idx < 2 or max_change_idx >= len(x_coords) - 2:
            # If change point is too close to edges, use the middle frame
            max_change_idx = len(x_coords) // 2
        
        # Use the trajectory change point as our "current" frame
        current_x = x_coords[max_change_idx]
        current_y = y_coords[max_change_idx]
        
        features = []
        eps = 1e-15  # Epsilon to prevent division by zero
        
        # Generate features for 2 frames (i=1, i=2) relative to the change point
        for i in range(1, 3):  # i=1, i=2
            # Past frames: change point - i
            x_lag_i = x_coords[max_change_idx - i]
            y_lag_i = y_coords[max_change_idx - i]
            
            # Future frames: change point + i  
            x_lag_inv_i = x_coords[max_change_idx + i]
            y_lag_inv_i = y_coords[max_change_idx + i]
            
            # Calculate differences
            x_diff_i = abs(x_lag_i - current_x)
            y_diff_i = y_lag_i - current_y  # y_diff is NOT abs (can be negative)
            x_diff_inv_i = abs(x_lag_inv_i - current_x)
            y_diff_inv_i = y_lag_inv_i - current_y  # y_diff_inv is NOT abs
            
            # Calculate ratios
            x_div_i = abs(x_diff_i / (x_diff_inv_i + eps))
            y_div_i = y_diff_i / (y_diff_inv_i + eps)
            
            # Add features in the order expected by the model
            features.extend([x_diff_i, x_diff_inv_i, x_div_i, y_diff_i, y_diff_inv_i, y_div_i])
        
        return features
    
    def get_trajectory_info(self) -> Dict:
        """Get information about current trajectory"""
        valid_positions = [(x, y, frame) for x, y, frame in self.ball_trajectory 
                          if x is not None and y is not None]
        
        return {
            'total_frames': len(self.ball_trajectory),
            'valid_frames': len(valid_positions),
            'recent_positions': valid_positions[-5:] if valid_positions else []
        }


class TennisBounceProcessor:
    """Main processor for tennis bounce detection"""
    
    def __init__(self, model_path: str):
        """Initialize processor with bounce detector"""
        self.bounce_detector = BounceDetector(model_path)
        self.bounce_history = []  # Store bounce detections
        
    def process_video(self, video_file: str, csv_file: str, output_file: str = None, show_viewer: bool = False):
        """Process video with bounce detection and court boundary validation"""
        # Load CSV data
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
        
        # Load court keypoints from CSV for boundary validation
        self._load_court_keypoints(df)
        
        # Setup video writer if output specified
        out = None
        if output_file:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_file, fourcc, fps, (width, height))
        
        # Setup viewer
        if show_viewer:
            cv2.namedWindow('Tennis Bounce Detection', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('Tennis Bounce Detection', 1200, 800)
        
        try:
            for idx, row in df.iterrows():
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Get ball position from CSV
                ball_x = self._parse_float(row.get('ball_x', ''))
                ball_y = self._parse_float(row.get('ball_y', ''))
                
                # Debug: Log ball data for first few frames
                if idx < 10:
                    logger.info(f"Frame {idx}: ball_x={ball_x}, ball_y={ball_y}")
                
                # Add ball position to detector
                self.bounce_detector.add_ball_position(ball_x, ball_y, idx)
                
                # Detect bounce
                is_bounce, confidence = self.bounce_detector.detect_bounce(idx)
                
                # Store bounce detection
                if is_bounce and ball_x is not None and ball_y is not None:
                    self.bounce_history.append({
                        'frame': idx,
                        'confidence': confidence,
                        'ball_x': ball_x,
                        'ball_y': ball_y
                    })
                
                # Add overlays
                frame_with_overlays = self._add_overlays(frame, idx, ball_x, ball_y, is_bounce, confidence)
                
                # Write frame
                if out:
                    out.write(frame_with_overlays)
                
                # Show in viewer
                if show_viewer:
                    cv2.imshow('Tennis Bounce Detection', frame_with_overlays)
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
        
        # Print bounce detection summary
        self._print_summary()
        
        logger.info("Processing completed!")
    
    def _parse_float(self, value: str) -> Optional[float]:
        """Parse float value from CSV string"""
        try:
            if pd.isna(value) or value == '':
                return None
            return float(value)
        except (ValueError, TypeError):
            return None
    
    def _add_overlays(self, frame: np.ndarray, frame_number: int, ball_x: Optional[float], 
                     ball_y: Optional[float], is_bounce: bool, confidence: float) -> np.ndarray:
        """Add bounce detection overlays to frame"""
        frame = frame.copy()
        
        # Add frame info
        cv2.putText(frame, f"Frame: {frame_number}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(frame, f"Bounce Detection", (10, 70), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Add ball position
        if ball_x is not None and ball_y is not None:
            bx, by = int(ball_x), int(ball_y)
            
            # Draw ball
            if is_bounce:
                # Red circle for bounce
                cv2.circle(frame, (bx, by), 12, (0, 0, 255), -1)
                cv2.putText(frame, "BOUNCE!", (bx + 15, by), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            else:
                # Yellow circle for normal ball
                cv2.circle(frame, (bx, by), 8, (0, 255, 255), -1)
            
            # Add confidence
            cv2.putText(frame, f"Conf: {confidence:.3f}", (bx + 15, by + 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Add trajectory info
        traj_info = self.bounce_detector.get_trajectory_info()
        cv2.putText(frame, f"Trajectory: {traj_info['valid_frames']}/{traj_info['total_frames']} frames", 
                   (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # Add bounce count
        cv2.putText(frame, f"Bounces detected: {len(self.bounce_history)}", 
                   (10, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        return frame
    
    def _load_court_keypoints(self, df: pd.DataFrame):
        """Load court keypoints from CSV data"""
        try:
            # Look for court keypoints in the CSV
            if 'court_keypoints' in df.columns:
                # Get the first valid court keypoints
                for idx, row in df.iterrows():
                    court_keypoints_str = row.get('court_keypoints', '')
                    if court_keypoints_str and court_keypoints_str != '':
                        # Parse court keypoints string
                        court_keypoints = self._parse_court_keypoints(court_keypoints_str)
                        if court_keypoints and len(court_keypoints) >= 4:
                            self.bounce_detector.set_court_keypoints(court_keypoints)
                            logger.info(f"Loaded court keypoints from frame {idx}")
                            return
                logger.warning("No valid court keypoints found in CSV")
            else:
                logger.warning("No court_keypoints column found in CSV")
        except Exception as e:
            logger.warning(f"Error loading court keypoints: {e}")
    
    def _parse_court_keypoints(self, keypoints_str: str) -> List[Tuple]:
        """Parse court keypoints from CSV string format"""
        try:
            if not keypoints_str or keypoints_str == '':
                return []
            
            # Parse the keypoints string (format: "x1,y1|x2,y2|...")
            keypoints = []
            points = keypoints_str.split('|')
            for point in points:
                if ',' in point:
                    x, y = point.split(',')
                    try:
                        x_val = float(x) if x != 'nan' and x != '' else None
                        y_val = float(y) if y != 'nan' and y != '' else None
                        keypoints.append((x_val, y_val))
                    except ValueError:
                        keypoints.append((None, None))
                else:
                    keypoints.append((None, None))
            
            return keypoints
        except Exception as e:
            logger.warning(f"Error parsing court keypoints: {e}")
            return []
    
    def _print_summary(self):
        """Print bounce detection summary"""
        logger.info("=== BOUNCE DETECTION SUMMARY ===")
        logger.info(f"Total bounces detected: {len(self.bounce_history)}")
        
        if self.bounce_history:
            logger.info("Bounce details:")
            for i, bounce in enumerate(self.bounce_history):
                logger.info(f"  Bounce {i+1}: Frame {bounce['frame']}, "
                           f"Confidence {bounce['confidence']:.3f}, "
                           f"Position ({bounce['ball_x']:.1f}, {bounce['ball_y']:.1f})")
        else:
            logger.info("No bounces detected in this video")


def main():
    """Main function for standalone usage"""
    parser = argparse.ArgumentParser(description='Tennis Ball Bounce Detection System')
    parser.add_argument('--video', required=True, help='Input video file')
    parser.add_argument('--csv', required=True, help='Input CSV file with ball tracking data')
    parser.add_argument('--model', default='models/bounce_detector.cbm', 
                       help='Path to bounce detection model (.cbm file)')
    parser.add_argument('--output', help='Output video file (optional)')
    parser.add_argument('--viewer', action='store_true', 
                       help='Show real-time viewer')
    
    args = parser.parse_args()
    
    # Create processor and run analysis
    processor = TennisBounceProcessor(args.model)
    processor.process_video(args.video, args.csv, args.output, show_viewer=args.viewer)


if __name__ == "__main__":
    main()
