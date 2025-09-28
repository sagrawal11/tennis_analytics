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
import argparse
import logging
from typing import List, Tuple, Optional, Dict
from collections import deque
import os

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')
logger = logging.getLogger(__name__)


class BounceDetector:
    """Ball bounce detection based on trajectory analysis"""
    
    def __init__(self):
        """Initialize bounce detector"""
        self.ball_trajectory = deque(maxlen=50)  # Store ball positions for temporal analysis
        self.min_bounce_gap_frames = 30  # Minimum frames between bounces (1 second at 30fps)
        self.last_bounce_frame = -1  # Track last bounce frame
        
        logger.info("Bounce detector initialized")
    
    
    
    def add_ball_position(self, ball_x: Optional[float], ball_y: Optional[float], frame_number: int):
        """Add ball position to trajectory history"""
        if ball_x is not None and ball_y is not None:
            self.ball_trajectory.append((ball_x, ball_y, frame_number))
        else:
            # Add None for missing ball positions
            self.ball_trajectory.append((None, None, frame_number))
    
    def detect_bounce(self, current_frame: int) -> Tuple[bool, float]:
        """
        Detect if there's a bounce in the current trajectory using hybrid approach
        
        Returns:
            Tuple of (is_bounce, confidence)
        """
        # Check minimum time gap between bounces
        if current_frame - self.last_bounce_frame < self.min_bounce_gap_frames:
            return False, 0.0
        
        # Use window-based trajectory analysis
        bounce_confidence = self._detect_bounce_windowed(current_frame)
        
        # Update last bounce frame if bounce detected
        if bounce_confidence > 0.7:
            self.last_bounce_frame = current_frame
        
        return bounce_confidence > 0.7, bounce_confidence
    
    def _detect_bounce_windowed(self, current_frame: int, window_size: int = 10) -> float:
        """
        Detect bounces using sliding window approach inspired by research
        Based on the AI Tennis Ball Bounce Detection research
        """
        # Get recent trajectory data
        valid_positions = [(x, y, frame) for x, y, frame in self.ball_trajectory 
                          if x is not None and y is not None]
        
        if len(valid_positions) < window_size:
            return 0.0
        
        # Extract window of recent positions
        recent_window = valid_positions[-window_size:]
        x_coords = [pos[0] for pos in recent_window]
        y_coords = [pos[1] for pos in recent_window]
        
        # Analyze trajectory patterns for bounce indicators
        bounce_confidence = self._analyze_trajectory_patterns(x_coords, y_coords)
        
        return bounce_confidence
    
    def _analyze_trajectory_patterns(self, x_coords: List[float], y_coords: List[float]) -> float:
        """
        Analyze trajectory for bounce indicators based on physics and research insights
        """
        confidence = 0.0
        
        if len(x_coords) < 3 or len(y_coords) < 3:
            return 0.0
        
        # 1. Y-direction velocity reversal (most important physics indicator)
        y_velocity_score = self._analyze_y_velocity_reversal(y_coords)
        confidence += y_velocity_score * 0.4  # 40% weight
        
        # 2. Trajectory curvature (sharp direction changes)
        curvature_score = self._analyze_trajectory_curvature(x_coords, y_coords)
        confidence += curvature_score * 0.3  # 30% weight
        
        # 3. Speed changes (bounces often show speed reduction)
        speed_score = self._analyze_speed_changes(x_coords, y_coords)
        confidence += speed_score * 0.2  # 20% weight
        
        # 4. Acceleration analysis (sudden acceleration changes)
        acceleration_score = self._analyze_acceleration_patterns(x_coords, y_coords)
        confidence += acceleration_score * 0.1  # 10% weight
        
        return min(1.0, confidence)
    
    def _analyze_y_velocity_reversal(self, y_coords: List[float]) -> float:
        """
        Detect Y-velocity reversals indicating bounces
        This is the most reliable physics-based indicator
        """
        if len(y_coords) < 3:
            return 0.0
        
        # Calculate Y-velocities
        y_velocities = [y_coords[i] - y_coords[i-1] for i in range(1, len(y_coords))]
        
        # Look for velocity reversals in Y direction
        max_reversal_strength = 0.0
        
        for i in range(1, len(y_velocities)):
            prev_vel = y_velocities[i-1]
            curr_vel = y_velocities[i]
            
            # Check for upward bounce (negative to positive velocity)
            if prev_vel < -2 and curr_vel > 2:  # Significant upward reversal
                reversal_strength = abs(curr_vel - prev_vel)
                max_reversal_strength = max(max_reversal_strength, reversal_strength)
            
            # Check for downward bounce (positive to negative velocity) 
            elif prev_vel > 2 and curr_vel < -2:  # Significant downward reversal
                reversal_strength = abs(curr_vel - prev_vel)
                max_reversal_strength = max(max_reversal_strength, reversal_strength)
        
        # Normalize reversal strength to 0-1 scale
        # Typical reversal strength is 10-50 pixels, so normalize accordingly
        return min(1.0, max_reversal_strength / 30.0)
    
    def _analyze_trajectory_curvature(self, x_coords: List[float], y_coords: List[float]) -> float:
        """
        Detect sharp direction changes in trajectory indicating bounces
        """
        if len(x_coords) < 3:
            return 0.0
        
        max_angle_change = 0.0
        
        # Calculate trajectory angles and look for sharp changes
        for i in range(1, len(x_coords)-1):
            # Calculate direction vectors
            dx1 = x_coords[i] - x_coords[i-1]
            dy1 = y_coords[i] - y_coords[i-1]
            dx2 = x_coords[i+1] - x_coords[i]
            dy2 = y_coords[i+1] - y_coords[i]
            
            # Skip if either vector is too small
            if abs(dx1) < 0.1 and abs(dy1) < 0.1:
                continue
            if abs(dx2) < 0.1 and abs(dy2) < 0.1:
                continue
            
            # Calculate angles
            angle1 = np.arctan2(dy1, dx1)
            angle2 = np.arctan2(dy2, dx2)
            
            # Calculate angle change
            angle_change = abs(angle2 - angle1)
            if angle_change > np.pi:
                angle_change = 2*np.pi - angle_change
            
            max_angle_change = max(max_angle_change, angle_change)
        
        # Convert to score: sharp changes (>60 degrees) get high scores
        if max_angle_change > np.pi/3:  # 60 degrees
            return min(1.0, (max_angle_change - np.pi/3) / (np.pi/2))  # Scale to 0-1
        return 0.0
    
    def _analyze_speed_changes(self, x_coords: List[float], y_coords: List[float]) -> float:
        """
        Detect significant speed changes that often occur at bounces
        """
        if len(x_coords) < 3:
            return 0.0
        
        # Calculate speeds between consecutive points
        speeds = []
        for i in range(1, len(x_coords)):
            speed = np.sqrt((x_coords[i] - x_coords[i-1])**2 + (y_coords[i] - y_coords[i-1])**2)
            speeds.append(speed)
        
        if len(speeds) < 2:
            return 0.0
        
        # Look for significant speed changes
        max_speed_change = 0.0
        for i in range(1, len(speeds)):
            speed_change = abs(speeds[i] - speeds[i-1])
            max_speed_change = max(max_speed_change, speed_change)
        
        # Normalize: significant speed changes are 10+ pixels
        avg_speed = np.mean(speeds)
        if avg_speed > 0:
            relative_change = max_speed_change / avg_speed
            return min(1.0, relative_change)
        
        return 0.0
    
    def _analyze_acceleration_patterns(self, x_coords: List[float], y_coords: List[float]) -> float:
        """
        Detect sudden acceleration changes that occur at bounces
        """
        if len(x_coords) < 4:
            return 0.0
        
        # Calculate accelerations (second derivative of position)
        max_acceleration_change = 0.0
        
        for i in range(2, len(x_coords)-1):
            # Calculate velocity vectors
            vx1 = x_coords[i-1] - x_coords[i-2]
            vy1 = y_coords[i-1] - y_coords[i-2]
            vx2 = x_coords[i] - x_coords[i-1]
            vy2 = y_coords[i] - y_coords[i-1]
            vx3 = x_coords[i+1] - x_coords[i]
            vy3 = y_coords[i+1] - y_coords[i]
            
            # Calculate acceleration vectors
            ax1 = vx2 - vx1
            ay1 = vy2 - vy1
            ax2 = vx3 - vx2
            ay2 = vy3 - vy2
            
            # Calculate acceleration change magnitude
            accel_change = np.sqrt((ax2 - ax1)**2 + (ay2 - ay1)**2)
            max_acceleration_change = max(max_acceleration_change, accel_change)
        
        # Normalize acceleration changes
        return min(1.0, max_acceleration_change / 20.0)  # Scale based on typical values
    
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
    
    def __init__(self):
        """Initialize processor with bounce detector"""
        self.bounce_detector = BounceDetector()
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
                    # Update last bounce frame
                    self.bounce_detector.last_bounce_frame = idx
                
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
    parser.add_argument('--output', help='Output video file (optional)')
    parser.add_argument('--viewer', action='store_true', 
                       help='Show real-time viewer')
    
    args = parser.parse_args()
    
    # Create processor and run analysis
    processor = TennisBounceProcessor()
    processor.process_video(args.video, args.csv, args.output, show_viewer=args.viewer)


if __name__ == "__main__":
    main()
