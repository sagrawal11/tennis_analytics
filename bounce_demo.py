#!/usr/bin/env python3
"""
Bounce Detection Demo Script
Properly implements temporal ball trajectory analysis for detecting tennis ball bounces
using the correct feature extraction approach from TrackNet.
"""

import cv2
import numpy as np
import yaml
import time
import logging
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Deque
from collections import deque
import catboost as cb
import argparse
import sys

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BounceDetector:
    """Proper CatBoost-based ball bounce detection using temporal trajectory analysis"""
    
    def __init__(self, model_path: str, num_frames: int = 3):
        """
        Initialize bounce detector
        
        Args:
            model_path: Path to the trained CatBoost model (.cbm file)
            num_frames: Number of consecutive frames to analyze (default: 3)
        """
        self.num_frames = num_frames
        self.ball_positions = deque(maxlen=num_frames + 2)  # Store positions for analysis
        self.frame_timestamps = deque(maxlen=num_frames + 2)  # Store timestamps
        
        try:
            # Load the CatBoost model
            self.model = cb.CatBoostClassifier()
            self.model.load_model(model_path)
            logger.info(f"Bounce detector initialized with {model_path}")
            logger.info(f"Model expects {self.model.feature_names_} features")
            
            # Verify we have the right number of features
            expected_features = (num_frames - 1) * 6  # 6 features per frame difference
            if len(self.model.feature_names_) != expected_features:
                logger.warning(f"Model expects {len(self.model.feature_names_)} features, but we'll generate {expected_features}")
                
        except Exception as e:
            logger.error(f"Error loading bounce detection model: {e}")
            self.model = None
    
    def add_ball_position(self, x: float, y: float, timestamp: float = None):
        """
        Add a new ball position for temporal analysis
        
        Args:
            x: X coordinate of ball
            y: Y coordinate of ball  
            timestamp: Frame timestamp (optional)
        """
        if timestamp is None:
            timestamp = time.time()
            
        self.ball_positions.append((x, y))
        self.frame_timestamps.append(timestamp)
        
        # Keep only the frames we need
        while len(self.ball_positions) > self.num_frames + 2:
            self.ball_positions.popleft()
            self.frame_timestamps.popleft()
    
    def extract_trajectory_features(self) -> List[float]:
        """
        Extract trajectory features for bounce detection
        Based on TrackNet bounce_train.py approach
        """
        if len(self.ball_positions) < self.num_frames:
            return []
        
        features = []
        eps = 1e-15  # Small epsilon to avoid division by zero
        
        # Get the last num_frames positions
        positions = list(self.ball_positions)[-self.num_frames:]
        
        # Calculate features for each frame difference
        for i in range(1, self.num_frames):
            # Current frame
            curr_x, curr_y = positions[i]
            # Previous frame
            prev_x, prev_y = positions[i-1]
            # Next frame (if available)
            next_x, next_y = positions[i+1] if i+1 < len(positions) else (curr_x, curr_y)
            
            # X and Y differences
            x_diff = abs(curr_x - prev_x)
            y_diff = curr_y - prev_y  # Keep sign for y (gravity direction matters)
            x_diff_inv = abs(next_x - curr_x)
            y_diff_inv = next_y - curr_y
            
            # Ratio features (avoid division by zero)
            x_div = abs(x_diff / (x_diff_inv + eps))
            y_div = y_diff / (y_diff_inv + eps)
            
            # Add all 6 features for this frame difference
            features.extend([x_diff, x_diff_inv, x_div, y_diff, y_diff_inv, y_div])
        
        return features
    
    def detect_bounce(self) -> Tuple[bool, float]:
        """
        Detect if a bounce occurred based on temporal trajectory analysis
        
        Returns:
            Tuple of (bounce_detected, confidence)
        """
        if not self.model or len(self.ball_positions) < self.num_frames:
            return False, 0.0
        
        try:
            # Extract trajectory features
            features = self.extract_trajectory_features()
            
            if len(features) == 0:
                return False, 0.0
            
            # Ensure we have the right number of features
            expected_features = (self.num_frames - 1) * 6
            if len(features) != expected_features:
                logger.warning(f"Expected {expected_features} features, got {len(features)}")
                # Pad or truncate to match expected length
                while len(features) < expected_features:
                    features.append(0.0)
                features = features[:expected_features]
            
            # Make prediction
            prediction = self.model.predict_proba([features])[0]
            confidence = prediction[1] if len(prediction) > 1 else prediction[0]
            
            # Determine if bounce occurred (threshold can be adjusted)
            bounce_detected = confidence > 0.6
            
            return bounce_detected, confidence
            
        except Exception as e:
            logger.error(f"Bounce detection error: {e}")
            return False, 0.0
    
    def get_trajectory_info(self) -> Dict[str, any]:
        """Get current trajectory information for debugging"""
        if len(self.ball_positions) < 2:
            return {}
        
        positions = list(self.ball_positions)
        velocities = []
        
        for i in range(1, len(positions)):
            prev_x, prev_y = positions[i-1]
            curr_x, curr_y = positions[i]
            
            dx = curr_x - prev_x
            dy = curr_y - prev_y
            velocity = np.sqrt(dx**2 + dy**2)
            velocities.append(velocity)
        
        return {
            'num_positions': len(self.ball_positions),
            'positions': positions,
            'velocities': velocities,
            'avg_velocity': np.mean(velocities) if velocities else 0,
            'velocity_change': velocities[-1] - velocities[0] if len(velocities) > 1 else 0
        }


class BallTracker:
    """Simple ball tracking for demo purposes"""
    
    def __init__(self, model_path: str = None):
        """
        Initialize ball tracker
        
        Args:
            model_path: Path to ball detection model (optional)
        """
        self.model_path = model_path
        self.ball_positions = deque(maxlen=30)
        
        # Simple color-based ball detection as fallback
        self.ball_color_lower = np.array([0, 100, 100])  # HSV lower bound for yellow/white balls
        self.ball_color_upper = np.array([30, 255, 255])  # HSV upper bound
        
    def detect_ball_simple(self, frame: np.ndarray) -> Optional[Tuple[float, float, float]]:
        """
        Simple color-based ball detection
        
        Returns:
            Tuple of (x, y, confidence) or None if no ball detected
        """
        try:
            # Convert to HSV
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            
            # Create mask for ball colors
            mask = cv2.inRange(hsv, self.ball_color_lower, self.ball_color_upper)
            
            # Find contours
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if not contours:
                return None
            
            # Find the largest contour (likely the ball)
            largest_contour = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(largest_contour)
            
            # Filter by area (avoid noise)
            if area < 50 or area > 5000:
                return None
            
            # Get centroid
            M = cv2.moments(largest_contour)
            if M["m00"] == 0:
                return None
                
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m10"] / M["m00"])
            
            # Calculate confidence based on area and circularity
            perimeter = cv2.arcLength(largest_contour, True)
            circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0
            
            confidence = min(1.0, (area / 1000) * circularity)
            
            return (cx, cy, confidence)
            
        except Exception as e:
            logger.error(f"Simple ball detection error: {e}")
            return None


class BounceDemo:
    """Main demo class for testing bounce detection"""
    
    def __init__(self, video_path: str, model_path: str = "models/bounce_detector.cbm"):
        """
        Initialize bounce detection demo
        
        Args:
            video_path: Path to input video
            model_path: Path to bounce detection model
        """
        self.video_path = video_path
        self.model_path = model_path
        
        # Initialize components
        self.bounce_detector = BounceDetector(model_path)
        self.ball_tracker = BallTracker()
        
        # Video capture
        self.cap = None
        
        # Visualization settings
        self.show_trajectory = True
        self.show_features = True
        self.show_confidence = True
        
        # Statistics
        self.stats = {
            'total_frames': 0,
            'balls_detected': 0,
            'bounces_detected': 0,
            'avg_confidence': 0.0
        }
        
    def run_demo(self):
        """Run the bounce detection demo"""
        if not Path(self.video_path).exists():
            logger.error(f"Video file not found: {self.video_path}")
            return
        
        if not Path(self.model_path).exists():
            logger.error(f"Model file not found: {self.model_path}")
            return
        
        try:
            self.cap = cv2.VideoCapture(self.video_path)
            if not self.cap.isOpened():
                logger.error(f"Could not open video: {self.video_path}")
                return
            
            logger.info("🎾 Starting Bounce Detection Demo")
            logger.info("Controls:")
            logger.info("  't' - Toggle trajectory display")
            logger.info("  'f' - Toggle feature display") 
            logger.info("  'c' - Toggle confidence display")
            logger.info("  'q' - Quit")
            logger.info("  SPACE - Pause/Resume")
            
            paused = False
            
            while True:
                if not paused:
                    ret, frame = self.cap.read()
                    if not ret:
                        break
                    
                    # Process frame
                    processed_frame = self.process_frame(frame)
                    
                    # Display frame
                    cv2.imshow("Bounce Detection Demo", processed_frame)
                
                # Handle key presses
                key = cv2.waitKey(1 if not paused else 0) & 0xFF
                
                if key == ord('q'):
                    break
                elif key == ord(' '):
                    paused = not paused
                    logger.info("Paused" if paused else "Resumed")
                elif key == ord('t'):
                    self.show_trajectory = not self.show_trajectory
                    logger.info(f"Trajectory display: {'ON' if self.show_trajectory else 'OFF'}")
                elif key == ord('f'):
                    self.show_features = not self.show_features
                    logger.info(f"Feature display: {'ON' if self.show_features else 'OFF'}")
                elif key == ord('c'):
                    self.show_confidence = not self.show_confidence
                    logger.info(f"Confidence display: {'ON' if self.show_confidence else 'OFF'}")
            
        except KeyboardInterrupt:
            logger.info("Demo interrupted by user")
        finally:
            if self.cap:
                self.cap.release()
            cv2.destroyAllWindows()
            
            # Print final statistics
            self.print_statistics()
    
    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """Process a single frame for bounce detection"""
        self.stats['total_frames'] += 1
        
        # Create output frame
        output_frame = frame.copy()
        
        # Detect ball
        ball_detection = self.ball_tracker.detect_ball_simple(frame)
        
        if ball_detection:
            x, y, confidence = ball_detection
            self.stats['balls_detected'] += 1
            
            # Add ball position to bounce detector
            self.bounce_detector.add_ball_position(x, y, time.time())
            
            # Draw ball
            cv2.circle(output_frame, (int(x), int(y)), 10, (0, 255, 255), -1)
            cv2.circle(output_frame, (int(x), int(y)), 12, (0, 0, 0), 2)
            
            # Check for bounce
            bounce_detected, bounce_confidence = self.bounce_detector.detect_bounce()
            
            if bounce_detected:
                self.stats['bounces_detected'] += 1
                self.stats['avg_confidence'] = (self.stats['avg_confidence'] + bounce_confidence) / 2
                
                # Draw bounce indicator
                cv2.putText(output_frame, f"BOUNCE! ({bounce_confidence:.2f})", 
                           (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                cv2.circle(output_frame, (int(x), int(y)), 30, (0, 0, 255), 3)
            
            # Draw trajectory if enabled
            if self.show_trajectory:
                self.draw_trajectory(output_frame)
            
            # Draw feature information if enabled
            if self.show_features:
                self.draw_feature_info(output_frame)
            
            # Draw confidence if enabled
            if self.show_confidence:
                cv2.putText(output_frame, f"Ball Conf: {confidence:.2f}", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                if bounce_detected:
                    cv2.putText(output_frame, f"Bounce Conf: {bounce_confidence:.2f}", 
                               (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Draw frame counter
        cv2.putText(output_frame, f"Frame: {self.stats['total_frames']}", 
                   (10, output_frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return output_frame
    
    def draw_trajectory(self, frame: np.ndarray):
        """Draw ball trajectory"""
        if len(self.bounce_detector.ball_positions) < 2:
            return
        
        positions = list(self.bounce_detector.ball_positions)
        
        # Draw trajectory line
        for i in range(1, len(positions)):
            prev_x, prev_y = positions[i-1]
            curr_x, curr_y = positions[i]
            
            # Color based on recency (newer = brighter)
            alpha = i / len(positions)
            color = (0, int(255 * alpha), int(255 * (1 - alpha)))
            
            cv2.line(frame, (int(prev_x), int(prev_y)), (int(curr_x), int(curr_y)), color, 2)
            
            # Draw position markers
            cv2.circle(frame, (int(curr_x), int(curr_y)), 3, color, -1)
    
    def draw_feature_info(self, frame: np.ndarray):
        """Draw feature information"""
        trajectory_info = self.bounce_detector.get_trajectory_info()
        
        if not trajectory_info:
            return
        
        y_offset = 100
        cv2.putText(frame, f"Positions: {trajectory_info['num_positions']}", 
                   (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        y_offset += 25
        
        if trajectory_info['velocities']:
            cv2.putText(frame, f"Avg Velocity: {trajectory_info['avg_velocity']:.1f}", 
                       (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            y_offset += 25
            
            cv2.putText(frame, f"Velocity Change: {trajectory_info['velocity_change']:.1f}", 
                       (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    def print_statistics(self):
        """Print demo statistics"""
        logger.info("\n" + "="*50)
        logger.info("BOUNCE DETECTION DEMO STATISTICS")
        logger.info("="*50)
        logger.info(f"Total Frames Processed: {self.stats['total_frames']}")
        logger.info(f"Balls Detected: {self.stats['balls_detected']}")
        logger.info(f"Bounces Detected: {self.stats['bounces_detected']}")
        logger.info(f"Average Bounce Confidence: {self.stats['avg_confidence']:.3f}")
        
        if self.stats['total_frames'] > 0:
            detection_rate = (self.stats['balls_detected'] / self.stats['total_frames']) * 100
            bounce_rate = (self.stats['bounces_detected'] / self.stats['total_frames']) * 100
            logger.info(f"Ball Detection Rate: {detection_rate:.1f}%")
            logger.info(f"Bounce Detection Rate: {bounce_rate:.1f}%")
        
        logger.info("="*50)


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Bounce Detection Demo")
    parser.add_argument("--video", type=str, default="tennis_test5.mp4", 
                       help="Path to input video file")
    parser.add_argument("--model", type=str, default="models/bounce_detector.cbm",
                       help="Path to bounce detection model")
    
    args = parser.parse_args()
    
    # Check if video exists
    if not Path(args.video).exists():
        logger.error(f"Video file not found: {args.video}")
        logger.info("Available video files:")
        for video_file in Path(".").glob("*.mp4"):
            logger.info(f"  - {video_file}")
        return
    
    # Check if model exists
    if not Path(args.model).exists():
        logger.error(f"Model file not found: {args.model}")
        logger.info("Available model files:")
        for model_file in Path("models").glob("*.cbm"):
            logger.info(f"  - {model_file}")
        return
    
    # Run demo
    demo = BounceDemo(args.video, args.model)
    demo.run_demo()


if __name__ == "__main__":
    main()
