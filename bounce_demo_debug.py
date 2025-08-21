#!/usr/bin/env python3
"""
Bounce Detection Debug Script
Enhanced version with debugging capabilities and parameter tuning
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
    
    def detect_bounce(self, confidence_threshold: float = 0.6) -> Tuple[bool, float, List[float]]:
        """
        Detect if a bounce occurred based on temporal trajectory analysis
        
        Args:
            confidence_threshold: Threshold for bounce detection
            
        Returns:
            Tuple of (bounce_detected, confidence, features)
        """
        if not self.model or len(self.ball_positions) < self.num_frames:
            return False, 0.0, []
        
        try:
            # Extract trajectory features
            features = self.extract_trajectory_features()
            
            if len(features) == 0:
                return False, 0.0, []
            
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
            
            # Determine if bounce occurred
            bounce_detected = confidence > confidence_threshold
            
            return bounce_detected, confidence, features
            
        except Exception as e:
            logger.error(f"Bounce detection error: {e}")
            return False, 0.0, []
    
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
    """Enhanced ball tracking with multiple detection methods"""
    
    def __init__(self, model_path: str = None):
        """
        Initialize ball tracker
        
        Args:
            model_path: Path to ball detection model (optional)
        """
        self.model_path = model_path
        self.ball_positions = deque(maxlen=30)
        
        # Multiple color ranges for different ball types
        self.ball_color_ranges = [
            # Yellow/white balls (tennis)
            (np.array([0, 100, 100]), np.array([30, 255, 255])),
            # Green balls
            (np.array([35, 50, 50]), np.array([85, 255, 255])),
            # Orange balls
            (np.array([5, 100, 100]), np.array([25, 255, 255]))
        ]
        
    def detect_ball_enhanced(self, frame: np.ndarray) -> Optional[Tuple[float, float, float]]:
        """
        Enhanced ball detection using multiple color ranges and contour analysis
        
        Returns:
            Tuple of (x, y, confidence) or None if no ball detected
        """
        try:
            # Convert to HSV
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            
            best_detection = None
            best_confidence = 0
            
            # Try multiple color ranges
            for color_lower, color_upper in self.ball_color_ranges:
                # Create mask for this color range
                mask = cv2.inRange(hsv, color_lower, color_upper)
                
                # Apply morphological operations to clean up the mask
                kernel = np.ones((3, 3), np.uint8)
                mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
                mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
                
                # Find contours
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                for contour in contours:
                    area = cv2.contourArea(contour)
                    
                    # Filter by area (avoid noise and too large objects)
                    if area < 30 or area > 8000:
                        continue
                    
                    # Get centroid
                    M = cv2.moments(contour)
                    if M["m00"] == 0:
                        continue
                        
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m10"] / M["m00"])
                    
                    # Calculate confidence based on multiple factors
                    perimeter = cv2.arcLength(contour, True)
                    circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0
                    
                    # Area-based confidence (prefer medium-sized objects)
                    area_confidence = 1.0 - abs(area - 500) / 500  # Peak at 500 pixels
                    area_confidence = max(0, min(1, area_confidence))
                    
                    # Combined confidence
                    confidence = (circularity * 0.6 + area_confidence * 0.4)
                    
                    if confidence > best_confidence:
                        best_confidence = confidence
                        best_detection = (cx, cy, confidence)
            
            return best_detection
            
        except Exception as e:
            logger.error(f"Enhanced ball detection error: {e}")
            return None


class BounceDemoDebug:
    """Enhanced demo class with debugging capabilities"""
    
    def __init__(self, video_path: str, model_path: str = "models/bounce_detector.cbm"):
        """
        Initialize bounce detection debug demo
        
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
        self.show_debug_info = True
        
        # Bounce detection parameters
        self.confidence_threshold = 0.1  # Much more sensitive threshold
        self.show_all_predictions = True  # Show confidence even when below threshold
        
        # Statistics
        self.stats = {
            'total_frames': 0,
            'balls_detected': 0,
            'bounces_detected': 0,
            'avg_confidence': 0.0,
            'confidence_values': [],
            'feature_values': []
        }
        
    def run_demo(self):
        """Run the bounce detection debug demo"""
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
            
            logger.info("🎾 Starting Bounce Detection Debug Demo")
            logger.info("Controls:")
            logger.info("  't' - Toggle trajectory display")
            logger.info("  'f' - Toggle feature display") 
            logger.info("  'c' - Toggle confidence display")
            logger.info("  'd' - Toggle debug info")
            logger.info("  '+' - Increase confidence threshold (0.01)")
            logger.info("  '-' - Decrease confidence threshold (0.01)")
            logger.info("  '0' - Reset threshold to 0.0 (most sensitive)")
            logger.info("  '1' - Set threshold to 0.1")
            logger.info("  '2' - Set threshold to 0.2")
            logger.info("  'a' - Toggle show all predictions")
            logger.info("  'q' - Quit")
            logger.info("  SPACE - Pause/Resume")
            logger.info(f"  Current threshold: {self.confidence_threshold:.2f}")
            
            paused = False
            
            while True:
                if not paused:
                    ret, frame = self.cap.read()
                    if not ret:
                        break
                    
                    # Process frame
                    processed_frame = self.process_frame(frame)
                    
                    # Display frame
                    cv2.imshow("Bounce Detection Debug Demo", processed_frame)
                
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
                elif key == ord('d'):
                    self.show_debug_info = not self.show_debug_info
                    logger.info(f"Debug info: {'ON' if self.show_debug_info else 'OFF'}")
                elif key == ord('+') or key == ord('='):
                    self.confidence_threshold = min(1.0, self.confidence_threshold + 0.01)  # Smaller increments
                    logger.info(f"Confidence threshold: {self.confidence_threshold:.3f}")
                elif key == ord('-') or key == ord('_'):
                    self.confidence_threshold = max(0.0, self.confidence_threshold - 0.01)  # Smaller increments
                    logger.info(f"Confidence threshold: {self.confidence_threshold:.3f}")
                elif key == ord('0'):
                    self.confidence_threshold = 0.0  # Reset to most sensitive
                    logger.info(f"Confidence threshold reset to: {self.confidence_threshold:.3f}")
                elif key == ord('1'):
                    self.confidence_threshold = 0.1  # Quick preset
                    logger.info(f"Confidence threshold set to: {self.confidence_threshold:.3f}")
                elif key == ord('2'):
                    self.confidence_threshold = 0.2  # Quick preset
                    logger.info(f"Confidence threshold set to: {self.confidence_threshold:.3f}")
                elif key == ord('a'):
                    self.show_all_predictions = not self.show_all_predictions
                    logger.info(f"Show all predictions: {'ON' if self.show_all_predictions else 'OFF'}")
            
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
        ball_detection = self.ball_tracker.detect_ball_enhanced(frame)
        
        if ball_detection:
            x, y, confidence = ball_detection
            self.stats['balls_detected'] += 1
            
            # Add ball position to bounce detector
            self.bounce_detector.add_ball_position(x, y, time.time())
            
            # Draw ball
            cv2.circle(output_frame, (int(x), int(y)), 10, (0, 255, 255), -1)
            cv2.circle(output_frame, (int(x), int(y)), 12, (0, 0, 0), 2)
            
            # Check for bounce
            bounce_detected, bounce_confidence, features = self.bounce_detector.detect_bounce(self.confidence_threshold)
            
            # Store confidence and features for analysis
            self.stats['confidence_values'].append(bounce_confidence)
            if features:
                self.stats['feature_values'].append(features)
            
            # Always show bounce detection with red dot and confidence
            if bounce_confidence > 0.05:  # Very low threshold to show all potential bounces
                # Draw red dot for bounce detection
                cv2.circle(output_frame, (int(x), int(y)), 8, (0, 0, 255), -1)  # Solid red dot
                cv2.circle(output_frame, (int(x), int(y)), 10, (255, 255, 255), 2)  # White outline
                
                # Show confidence level
                confidence_text = f"Bounce: {bounce_confidence:.3f}"
                cv2.putText(output_frame, confidence_text, 
                           (int(x) + 15, int(y) - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            
            if bounce_detected:
                self.stats['bounces_detected'] += 1
                self.stats['avg_confidence'] = (self.stats['avg_confidence'] + bounce_confidence) / 2
                
                # Draw prominent bounce indicator
                cv2.putText(output_frame, f"BOUNCE! ({bounce_confidence:.2f})", 
                           (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                cv2.circle(output_frame, (int(x), int(y)), 40, (0, 0, 255), 3)  # Larger red circle
            
            # Show confidence even when below threshold if enabled
            if self.show_all_predictions or bounce_detected:
                if self.show_confidence:
                    color = (0, 0, 255) if bounce_detected else (255, 255, 255)
                    cv2.putText(output_frame, f"Bounce Conf: {bounce_confidence:.3f}", 
                               (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            # Draw trajectory if enabled
            if self.show_trajectory:
                self.draw_trajectory(output_frame)
            
            # Draw feature information if enabled
            if self.show_features:
                self.draw_feature_info(output_frame)
            
            # Draw debug information if enabled
            if self.show_debug_info:
                self.draw_debug_info(output_frame, features)
            
            # Draw ball confidence
            cv2.putText(output_frame, f"Ball Conf: {confidence:.2f}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Draw frame counter and threshold
        cv2.putText(output_frame, f"Frame: {self.stats['total_frames']}", 
                   (10, output_frame.shape[0] - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(output_frame, f"Threshold: {self.confidence_threshold:.3f}", 
                   (10, output_frame.shape[0] - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Show detection status
        if self.stats['bounces_detected'] > 0:
            status_color = (0, 255, 0)  # Green if we've detected bounces
        else:
            status_color = (0, 0, 255)  # Red if no bounces detected
        cv2.putText(output_frame, f"Bounces: {self.stats['bounces_detected']}", 
                   (10, output_frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
        
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
    
    def draw_debug_info(self, frame: np.ndarray, features: List[float]):
        """Draw debug information including features"""
        if not features:
            return
        
        y_offset = 200
        cv2.putText(frame, "Features:", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        y_offset += 20
        
        # Show feature names and values
        feature_names = ['x_diff_1', 'x_diff_2', 'x_diff_inv_1', 'x_diff_inv_2', 'x_div_1', 'x_div_2',
                        'y_diff_1', 'y_diff_2', 'y_diff_inv_1', 'y_diff_inv_2', 'y_div_1', 'y_div_2']
        
        for i, (name, value) in enumerate(zip(feature_names, features)):
            if i % 3 == 0 and i > 0:  # New line every 3 features
                y_offset += 20
            x_offset = 10 + (i % 3) * 200
            
            text = f"{name}: {value:.2f}"
            cv2.putText(frame, text, (x_offset, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    
    def print_statistics(self):
        """Print demo statistics with enhanced analysis"""
        logger.info("\n" + "="*60)
        logger.info("BOUNCE DETECTION DEBUG DEMO STATISTICS")
        logger.info("="*60)
        logger.info(f"Total Frames Processed: {self.stats['total_frames']}")
        logger.info(f"Balls Detected: {self.stats['balls_detected']}")
        logger.info(f"Bounces Detected: {self.stats['bounces_detected']}")
        logger.info(f"Average Bounce Confidence: {self.stats['avg_confidence']:.3f}")
        
        if self.stats['total_frames'] > 0:
            detection_rate = (self.stats['balls_detected'] / self.stats['total_frames']) * 100
            bounce_rate = (self.stats['bounces_detected'] / self.stats['total_frames']) * 100
            logger.info(f"Ball Detection Rate: {detection_rate:.1f}%")
            logger.info(f"Bounce Detection Rate: {bounce_rate:.1f}%")
        
        # Confidence analysis
        if self.stats['confidence_values']:
            confidences = np.array(self.stats['confidence_values'])
            logger.info(f"\nConfidence Analysis:")
            logger.info(f"  Min Confidence: {confidences.min():.3f}")
            logger.info(f"  Max Confidence: {confidences.max():.3f}")
            logger.info(f"  Mean Confidence: {confidences.mean():.3f}")
            logger.info(f"  Std Confidence: {confidences.std():.3f}")
            
            # Show confidence distribution
            thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
            logger.info(f"\nConfidence Distribution:")
            for threshold in thresholds:
                count = np.sum(confidences > threshold)
                percentage = (count / len(confidences)) * 100
                logger.info(f"  >{threshold:.1f}: {count} frames ({percentage:.1f}%)")
        
        logger.info("="*60)


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Bounce Detection Debug Demo")
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
    demo = BounceDemoDebug(args.video, args.model)
    demo.run_demo()


if __name__ == "__main__":
    main()
