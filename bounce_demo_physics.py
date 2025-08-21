#!/usr/bin/env python3
"""
Physics-Based Bounce Detection Demo
Uses actual tennis ball physics to detect real bounces, not just movement patterns
"""

import cv2
import numpy as np
import yaml
import time
import logging
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Deque, Any
from collections import deque
import argparse
import sys

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PhysicsBounceDetector:
    """Physics-based bounce detection using actual tennis ball physics"""
    
    def __init__(self, num_frames: int = 10):
        """
        Initialize physics-based bounce detector
        
        Args:
            num_frames: Number of frames to analyze for physics calculations
        """
        self.num_frames = num_frames
        self.ball_positions = deque(maxlen=num_frames)
        self.frame_timestamps = deque(maxlen=num_frames)
        
        # Physics parameters for tennis ball
        self.gravity = 9.8  # m/s^2
        self.pixels_per_meter = 100  # Approximate scale (adjust based on your video)
        self.bounce_coefficient = 0.7  # Tennis ball bounces back to ~70% of height
        
        # Bounce detection parameters
        self.min_bounce_height = 20  # Minimum height change to consider a bounce
        self.velocity_threshold = 50  # Minimum velocity to consider significant movement
        self.direction_change_threshold = 45  # Degrees - minimum direction change for bounce
        
        logger.info("Physics-based bounce detector initialized")
        logger.info(f"Gravity: {self.gravity} m/s^2")
        logger.info(f"Bounce coefficient: {self.bounce_coefficient}")
    
    def add_ball_position(self, x: float, y: float, timestamp: float = None):
        """Add a new ball position for physics analysis"""
        if timestamp is None:
            timestamp = time.time()
            
        self.ball_positions.append((x, y))
        self.frame_timestamps.append(timestamp)
        
        # Keep only the frames we need
        while len(self.ball_positions) > self.num_frames:
            self.ball_positions.popleft()
            self.frame_timestamps.popleft()
    
    def calculate_physics_features(self) -> Dict[str, float]:
        """Calculate physics-based features for bounce detection"""
        if len(self.ball_positions) < 3:
            return {}
        
        positions = list(self.ball_positions)
        timestamps = list(self.frame_timestamps)
        
        features = {}
        
        # Calculate velocities
        velocities = []
        for i in range(1, len(positions)):
            prev_x, prev_y = positions[i-1]
            curr_x, curr_y = positions[i]
            dt = timestamps[i] - timestamps[i-1]
            
            if dt > 0:
                vx = (curr_x - prev_x) / dt
                vy = (curr_y - prev_y) / dt
                velocity = np.sqrt(vx**2 + vy**2)
                velocities.append(velocity)
        
        if len(velocities) < 2:
            return {}
        
        # Calculate acceleration (change in velocity)
        accelerations = []
        for i in range(1, len(velocities)):
            dt = timestamps[i+1] - timestamps[i-1]
            if dt > 0:
                acceleration = (velocities[i] - velocities[i-1]) / dt
                accelerations.append(acceleration)
        
        # Calculate trajectory features
        features['avg_velocity'] = np.mean(velocities)
        features['velocity_change'] = velocities[-1] - velocities[0] if len(velocities) > 1 else 0
        features['max_velocity'] = np.max(velocities)
        features['min_velocity'] = np.min(velocities)
        
        if accelerations:
            features['avg_acceleration'] = np.mean(accelerations)
            features['acceleration_change'] = accelerations[-1] - accelerations[0] if len(accelerations) > 1 else 0
        
        # Calculate height-based features (y-coordinate analysis)
        y_positions = [pos[1] for pos in positions]
        features['height_range'] = np.max(y_positions) - np.min(y_positions)
        features['height_change'] = y_positions[-1] - y_positions[0]
        features['avg_height'] = np.mean(y_positions)
        
        # Calculate direction changes
        directions = []
        for i in range(1, len(positions)):
            prev_x, prev_y = positions[i-1]
            curr_x, curr_y = positions[i]
            
            if prev_x != curr_x or prev_y != curr_y:
                angle = np.arctan2(curr_y - prev_y, curr_x - prev_x) * 180 / np.pi
                directions.append(angle)
        
        if len(directions) >= 2:
            # Calculate direction change
            direction_changes = []
            for i in range(1, len(directions)):
                change = abs(directions[i] - directions[i-1])
                # Handle angle wrapping (e.g., 359° to 1° = 2° change, not 358°)
                if change > 180:
                    change = 360 - change
                direction_changes.append(change)
            
            features['avg_direction_change'] = np.mean(direction_changes)
            features['max_direction_change'] = np.max(direction_changes)
        
        return features
    
    def detect_bounce_physics(self) -> Tuple[bool, float, Dict[str, float]]:
        """
        Detect bounce using physics-based analysis
        
        Returns:
            Tuple of (bounce_detected, confidence, features)
        """
        if len(self.ball_positions) < 5:  # Need enough frames for physics analysis
            return False, 0.0, {}
        
        features = self.calculate_physics_features()
        if not features:
            return False, 0.0, {}
        
        # Physics-based bounce detection logic
        bounce_indicators = []
        
        # 1. Height-based bounce detection
        if 'height_range' in features and features['height_range'] > self.min_bounce_height:
            # Ball is moving vertically significantly
            if 'height_change' in features:
                # Check if ball is moving downward (positive y change in image coordinates)
                if features['height_change'] > 10:  # Ball moving down
                    bounce_indicators.append(("height_movement", 0.3))
        
        # 2. Velocity-based bounce detection
        if 'velocity_change' in features and abs(features['velocity_change']) > self.velocity_threshold:
            # Significant velocity change detected
            bounce_indicators.append(("velocity_change", 0.4))
        
        # 3. Direction change bounce detection
        if 'max_direction_change' in features and features['max_direction_change'] > self.direction_change_threshold:
            # Significant direction change (ball hitting ground and bouncing)
            bounce_indicators.append(("direction_change", 0.5))
        
        # 4. Acceleration-based bounce detection
        if 'acceleration_change' in features and abs(features['acceleration_change']) > 100:
            # Sudden acceleration change (impact with ground)
            bounce_indicators.append(("acceleration_change", 0.6))
        
        # 5. Trajectory analysis - check for parabolic motion interruption
        if len(self.ball_positions) >= 7:
            # Analyze if trajectory follows expected physics
            trajectory_score = self._analyze_trajectory_physics()
            if trajectory_score > 0.5:
                bounce_indicators.append(("trajectory_physics", trajectory_score))
        
        # Calculate overall confidence
        if bounce_indicators:
            # Weight the indicators
            total_confidence = sum(score for _, score in bounce_indicators)
            avg_confidence = total_confidence / len(bounce_indicators)
            
            # Additional confidence boost for multiple indicators
            if len(bounce_indicators) >= 3:
                avg_confidence *= 1.2
            
            bounce_detected = avg_confidence > 0.4
            return bounce_detected, avg_confidence, features
        
        return False, 0.0, features
    
    def _analyze_trajectory_physics(self) -> float:
        """Analyze if trajectory follows expected physics patterns"""
        if len(self.ball_positions) < 7:
            return 0.0
        
        positions = list(self.ball_positions)
        
        # Check for parabolic motion (ball should follow gravity)
        y_positions = [pos[1] for pos in positions]
        x_positions = [pos[0] for pos in positions]
        
        # Simple parabola fitting (y = ax² + bx + c)
        try:
            # Fit quadratic curve to y vs x
            coeffs = np.polyfit(x_positions, y_positions, 2)
            a, b, c = coeffs
            
            # Calculate R-squared (goodness of fit)
            y_pred = np.polyval(coeffs, x_positions)
            ss_res = np.sum((y_positions - y_pred) ** 2)
            ss_tot = np.sum((y_positions - np.mean(y_positions)) ** 2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
            
            # Check if parabola opens upward (a > 0) which indicates ball moving upward
            if a > 0 and r_squared > 0.7:
                return 0.8  # Good parabolic motion
            elif a < 0 and r_squared > 0.7:
                return 0.6  # Good parabolic motion (ball moving down)
            else:
                return 0.3  # Poor fit to parabola
                
        except:
            return 0.0
    
    def get_physics_info(self) -> Dict[str, any]:
        """Get physics information for debugging"""
        if len(self.ball_positions) < 2:
            return {}
        
        features = self.calculate_physics_features()
        
        return {
            'num_positions': len(self.ball_positions),
            'positions': list(self.ball_positions),
            'features': features,
            'bounce_coefficient': self.bounce_coefficient,
            'gravity': self.gravity
        }


class WorkingBallTracker:
    """Ball tracker that uses the working system from tennis_CV.py"""
    
    def __init__(self, config_path: str = "config.yaml"):
        """
        Initialize working ball tracker
        
        Args:
            config_path: Path to configuration file
        """
        self.config = self._load_config(config_path)
        self.ball_positions = deque(maxlen=30)
        
        # Initialize models (simplified version of tennis_CV.py)
        self.rfdetr_ball_detector = None
        self.tracknet_model = None
        self.yolo_ball_model = None
        
        self._initialize_models()
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            return config
        except Exception as e:
            logger.error(f"Error loading config: {e}")
            return {}
    
    def _initialize_models(self):
        """Initialize ball detection models"""
        try:
            # Try to initialize RF-DETR
            if Path("models/playersnball5.pt").exists():
                try:
                    from RF_ball_detector import CustomPyTorchDetector
                    self.rfdetr_ball_detector = CustomPyTorchDetector("models/playersnball5.pt")
                    logger.info("RF-DETR ball detector initialized")
                except Exception as e:
                    logger.warning(f"RF-DETR initialization failed: {e}")
            
            # Try to initialize YOLO
            if Path("models/playersnball4.pt").exists():
                try:
                    from ultralytics import YOLO
                    self.yolo_ball_model = YOLO("models/playersnball4.pt")
                    logger.info("YOLO ball model initialized")
                except Exception as e:
                    logger.warning(f"YOLO initialization failed: {e}")
                    
        except Exception as e:
            logger.error(f"Model initialization error: {e}")
    
    def detect_ball(self, frame: np.ndarray) -> Optional[Tuple[float, float, float]]:
        """
        Detect ball using the working system from tennis_CV.py
        
        Returns:
            Tuple of (x, y, confidence) or None if no ball detected
        """
        try:
            # 1. Try RF-DETR first
            if self.rfdetr_ball_detector:
                detections = self.rfdetr_ball_detector.detect(frame, conf_threshold=0.2)
                for detection in detections:
                    if detection['class_name'] == 'ball':
                        x, y = detection['bbox'][0] + detection['bbox'][2] // 2, detection['bbox'][1] + detection['bbox'][3] // 2
                        return (x, y, detection['confidence'])
            
            # 2. Try YOLO
            if self.yolo_ball_model:
                results = self.yolo_ball_model(frame, verbose=False)
                for result in results:
                    if result.boxes is not None:
                        boxes = result.boxes
                        for box in boxes:
                            # Assuming class 0 is ball (adjust based on your model)
                            if box.cls == 0 and box.conf > 0.2:
                                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                                x, y = (x1 + x2) / 2, (y1 + y2) / 2
                                return (x, y, float(box.conf))
            
            return None
            
        except Exception as e:
            logger.error(f"Ball detection error: {e}")
            return None


class PhysicsBounceDemo:
    """Physics-based bounce detection demo"""
    
    def __init__(self, video_path: str):
        """
        Initialize physics-based bounce detection demo
        
        Args:
            video_path: Path to input video
        """
        self.video_path = video_path
        
        # Initialize components
        self.bounce_detector = PhysicsBounceDetector()
        self.ball_tracker = WorkingBallTracker()
        
        # Video capture
        self.cap = None
        
        # Visualization settings
        self.show_trajectory = True
        self.show_physics = True
        self.show_confidence = True
        
        # Statistics
        self.stats = {
            'total_frames': 0,
            'balls_detected': 0,
            'bounces_detected': 0,
            'avg_confidence': 0.0,
            'confidence_values': [],
            'physics_features': []
        }
        
    def run_demo(self):
        """Run the physics-based bounce detection demo"""
        if not Path(self.video_path).exists():
            logger.error(f"Video file not found: {self.video_path}")
            return
        
        try:
            self.cap = cv2.VideoCapture(self.video_path)
            if not self.cap.isOpened():
                logger.error(f"Could not open video: {self.video_path}")
                return
            
            logger.info("🎾 Starting Physics-Based Bounce Detection Demo")
            logger.info("This version uses ACTUAL tennis ball physics!")
            logger.info("Controls:")
            logger.info("  't' - Toggle trajectory display")
            logger.info("  'p' - Toggle physics info display")
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
                    cv2.imshow("Physics-Based Bounce Detection", processed_frame)
                
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
                elif key == ord('p'):
                    self.show_physics = not self.show_physics
                    logger.info(f"Physics info: {'ON' if self.show_physics else 'OFF'}")
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
        """Process a single frame for physics-based bounce detection"""
        self.stats['total_frames'] += 1
        
        # Create output frame
        output_frame = frame.copy()
        
        # Detect ball using working system
        ball_detection = self.ball_tracker.detect_ball(frame)
        
        if ball_detection:
            x, y, confidence = ball_detection
            self.stats['balls_detected'] += 1
            
            # Add ball position to physics detector
            self.bounce_detector.add_ball_position(x, y, time.time())
            
            # Draw ball
            cv2.circle(output_frame, (int(x), int(y)), 10, (0, 255, 255), -1)
            cv2.circle(output_frame, (int(x), int(y)), 12, (0, 0, 0), 2)
            
            # Check for bounce using physics
            bounce_detected, bounce_confidence, physics_features = self.bounce_detector.detect_bounce_physics()
            
            # Store confidence and features for analysis
            self.stats['confidence_values'].append(bounce_confidence)
            if physics_features:
                self.stats['physics_features'].append(physics_features)
            
            # Show bounce detection with red dot and confidence
            if bounce_confidence > 0.1:  # Show all potential bounces
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
                cv2.putText(output_frame, f"PHYSICS BOUNCE! ({bounce_confidence:.2f})", 
                           (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                cv2.circle(output_frame, (int(x), int(y)), 40, (0, 0, 255), 3)  # Larger red circle
            
            # Show confidence
            if self.show_confidence:
                color = (0, 0, 255) if bounce_detected else (255, 255, 255)
                cv2.putText(output_frame, f"Bounce Conf: {bounce_confidence:.3f}", 
                           (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            # Draw trajectory if enabled
            if self.show_trajectory:
                self.draw_trajectory(output_frame)
            
            # Draw physics information if enabled
            if self.show_physics:
                self.draw_physics_info(output_frame, physics_features)
            
            # Draw ball confidence
            cv2.putText(output_frame, f"Ball Conf: {confidence:.2f}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Draw frame counter
        cv2.putText(output_frame, f"Frame: {self.stats['total_frames']}", 
                   (10, output_frame.shape[0] - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Show detection status
        if self.stats['bounces_detected'] > 0:
            status_color = (0, 255, 0)  # Green if we've detected bounces
        else:
            status_color = (0, 0, 255)  # Red if no bounces detected
        cv2.putText(output_frame, f"Physics Bounces: {self.stats['bounces_detected']}", 
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
    
    def draw_physics_info(self, frame: np.ndarray, features: Dict[str, float]):
        """Draw physics information"""
        if not features:
            return
        
        y_offset = 100
        cv2.putText(frame, "Physics Features:", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        y_offset += 25
        
        # Show key physics features
        key_features = ['avg_velocity', 'velocity_change', 'height_range', 'height_change', 'max_direction_change']
        
        for feature in key_features:
            if feature in features:
                value = features[feature]
                text = f"{feature}: {value:.2f}"
                cv2.putText(frame, text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                y_offset += 20
    
    def print_statistics(self):
        """Print demo statistics"""
        logger.info("\n" + "="*60)
        logger.info("PHYSICS-BASED BOUNCE DETECTION STATISTICS")
        logger.info("="*60)
        logger.info(f"Total Frames Processed: {self.stats['total_frames']}")
        logger.info(f"Balls Detected: {self.stats['balls_detected']}")
        logger.info(f"Physics Bounces Detected: {self.stats['bounces_detected']}")
        logger.info(f"Average Bounce Confidence: {self.stats['avg_confidence']:.3f}")
        
        if self.stats['total_frames'] > 0:
            detection_rate = (self.stats['balls_detected'] / self.stats['total_frames']) * 100
            bounce_rate = (self.stats['bounces_detected'] / self.stats['total_frames']) * 100
            logger.info(f"Ball Detection Rate: {detection_rate:.1f}%")
            logger.info(f"Bounce Detection Rate: {bounce_rate:.1f}%")
        
        logger.info("="*60)


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Physics-Based Bounce Detection Demo")
    parser.add_argument("--video", type=str, default="tennis_test5.mp4", 
                       help="Path to input video file")
    
    args = parser.parse_args()
    
    # Check if video exists
    if not Path(args.video).exists():
        logger.error(f"Video file not found: {args.video}")
        logger.info("Available video files:")
        for video_file in Path(".").glob("*.mp4"):
            logger.info(f"  - {video_file}")
        return
    
    # Run demo
    demo = PhysicsBounceDemo(args.video)
    demo.run_demo()


if __name__ == "__main__":
    main()
