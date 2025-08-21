#!/usr/bin/env python3
"""
Trajectory-Based Bounce Detection Demo
Focuses on sharp angle changes in ball trajectory when away from players
"""

import cv2
import numpy as np
import argparse
import logging
from collections import deque
from typing import List, Dict, Tuple, Optional, Deque, Any
import math
from ultralytics import YOLO
import sys
import os

# Add the src directory to the path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from RF_ball_detector import CustomPyTorchDetector

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(s)s')
logger = logging.getLogger(__name__)

class TrajectoryBounceDetector:
    """Detects bounces by analyzing sharp trajectory angle changes when ball is away from players"""
    
    def __init__(self, min_trajectory_length: int = 5, angle_threshold: float = 45.0):
        self.min_trajectory_length = min_trajectory_length
        self.angle_threshold = angle_threshold  # degrees
        self.ball_positions = deque(maxlen=30)  # Store last 30 positions
        self.frame_numbers = deque(maxlen=30)
        self.player_positions = deque(maxlen=10)  # Store recent player positions
        self.bounce_history = deque(maxlen=50)  # Track detected bounces
        
        # Physics constants
        self.min_bounce_height = 15  # Minimum height change for bounce
        self.max_player_distance = 80  # Maximum distance to consider ball "near player"
        self.min_trajectory_speed = 5  # Minimum movement to consider trajectory valid
        
        logger.info(f"Trajectory bounce detector initialized")
        logger.info(f"Angle threshold: {angle_threshold}°")
        logger.info(f"Min trajectory length: {min_trajectory_length}")
    
    def add_ball_position(self, x: float, y: float, frame_num: int):
        """Add new ball position to trajectory"""
        if x > 0 and y > 0:  # Valid position
            self.ball_positions.append((x, y))
            self.frame_numbers.append(frame_num)
    
    def add_player_positions(self, positions: List[Tuple[float, float]]):
        """Add detected player positions"""
        if positions:
            # Use center of all players or closest player
            if len(positions) == 1:
                self.player_positions.append(positions[0])
            else:
                # Use center of all players
                avg_x = sum(p[0] for p in positions) / len(positions)
                avg_y = sum(p[1] for p in positions) / len(positions)
                self.player_positions.append((avg_x, avg_y))
    
    def _calculate_trajectory_angles(self) -> List[float]:
        """Calculate angles between consecutive trajectory segments"""
        if len(self.ball_positions) < 3:
            return []
        
        angles = []
        positions = list(self.ball_positions)
        
        for i in range(1, len(positions) - 1):
            # Vector 1: from i-1 to i
            dx1 = positions[i][0] - positions[i-1][0]
            dy1 = positions[i][1] - positions[i-1][1]
            
            # Vector 2: from i to i+1
            dx2 = positions[i+1][0] - positions[i][0]
            dy2 = positions[i+1][1] - positions[i][1]
            
            # Calculate angle between vectors
            if abs(dx1) > 0.1 or abs(dy1) > 0.1:  # Avoid division by zero
                if abs(dx2) > 0.1 or abs(dy2) > 0.1:
                    # Calculate angle using dot product
                    dot_product = dx1 * dx2 + dy1 * dy2
                    mag1 = math.sqrt(dx1**2 + dy1**2)
                    mag2 = math.sqrt(dx2**2 + dy2**2)
                    
                    if mag1 > 0 and mag2 > 0:
                        cos_angle = dot_product / (mag1 * mag2)
                        cos_angle = max(-1.0, min(1.0, cos_angle))  # Clamp to valid range
                        angle = math.degrees(math.acos(cos_angle))
                        angles.append(angle)
        
        return angles
    
    def _is_ball_away_from_players(self, ball_pos: Tuple[float, float]) -> bool:
        """Check if ball is far enough from players to consider for bounce detection"""
        if not self.player_positions:
            return True  # No players detected, assume ball is away
        
        # Check distance to closest player
        min_distance = float('inf')
        for player_pos in self.player_positions:
            distance = math.sqrt((ball_pos[0] - player_pos[0])**2 + (ball_pos[1] - player_pos[1])**2)
            min_distance = min(min_distance, distance)
        
        return min_distance > self.max_player_distance
    
    def _has_significant_height_change(self, recent_positions: List[Tuple[float, float]]) -> bool:
        """Check if there's a significant height change indicating a bounce"""
        if len(recent_positions) < 3:
            return False
        
        # Calculate height range in recent positions
        y_coords = [pos[1] for pos in recent_positions]
        height_range = max(y_coords) - min(y_coords)
        
        return height_range > self.min_bounce_height
    
    def _is_trajectory_valid(self, positions: List[Tuple[float, float]]) -> bool:
        """Check if trajectory has enough movement to be valid"""
        if len(positions) < 2:
            return False
        
        total_distance = 0
        for i in range(1, len(positions)):
            dx = positions[i][0] - positions[i-1][0]
            dy = positions[i][1] - positions[i-1][1]
            total_distance += math.sqrt(dx**2 + dy**2)
        
        return total_distance > self.min_trajectory_speed * len(positions)
    
    def detect_bounce(self) -> Tuple[bool, float, Dict[str, Any]]:
        """Detect bounce based on trajectory analysis"""
        if len(self.ball_positions) < self.min_trajectory_length:
            return False, 0.0, {}
        
        # Get recent positions for analysis
        recent_positions = list(self.ball_positions)[-self.min_trajectory_length:]
        
        # Check if ball is away from players
        ball_away_from_players = self._is_ball_away_from_players(recent_positions[-1])
        if not ball_away_from_players:
            return False, 0.0, {"reason": "ball_near_player"}
        
        # Check if trajectory has valid movement
        if not self._is_trajectory_valid(recent_positions):
            return False, 0.0, {"reason": "insufficient_movement"}
        
        # Calculate trajectory angles
        angles = self._calculate_trajectory_angles()
        if not angles:
            return False, 0.0, {"reason": "no_angles_calculated"}
        
        # Look for sharp angle changes
        sharp_angles = [angle for angle in angles if angle > self.angle_threshold]
        
        if not sharp_angles:
            return False, 0.0, {"reason": "no_sharp_angles"}
        
        # Check for height change
        has_height_change = self._has_significant_height_change(recent_positions)
        
        # Calculate confidence based on angle sharpness and height change
        max_angle = max(sharp_angles)
        angle_confidence = min(1.0, max_angle / 90.0)  # Normalize to 0-1
        height_confidence = 1.0 if has_height_change else 0.5
        
        # Final confidence is combination of both
        final_confidence = (angle_confidence + height_confidence) / 2
        
        # Additional checks for bounce validity
        is_bounce = (
            len(sharp_angles) >= 1 and  # At least one sharp angle
            has_height_change and        # Significant height change
            final_confidence > 0.6      # High enough confidence
        )
        
        bounce_info = {
            "sharp_angles": sharp_angles,
            "max_angle": max_angle,
            "angle_confidence": angle_confidence,
            "height_confidence": height_confidence,
            "height_change": has_height_change,
            "trajectory_length": len(recent_positions),
            "ball_away_from_players": ball_away_from_players
        }
        
        return is_bounce, final_confidence, bounce_info
    
    def get_trajectory_info(self) -> Dict[str, Any]:
        """Get current trajectory information for debugging"""
        if len(self.ball_positions) < 2:
            return {"positions": 0, "angles": [], "recent_movement": 0}
        
        positions = list(self.ball_positions)
        angles = self._calculate_trajectory_angles()
        
        # Calculate recent movement
        if len(positions) >= 2:
            recent_dx = positions[-1][0] - positions[-2][0]
            recent_dy = positions[-1][1] - positions[-2][1]
            recent_movement = math.sqrt(recent_dx**2 + recent_dy**2)
        else:
            recent_movement = 0
        
        return {
            "positions": len(positions),
            "angles": angles,
            "recent_movement": recent_movement,
            "ball_away_from_players": self._is_ball_away_from_players(positions[-1]) if positions else False
        }

class WorkingBallTracker:
    """Ball tracker that uses the working system from tennis_CV.py"""
    
    def __init__(self, config_path: str = "config.yaml"):
        self.config = self._load_config(config_path)
        self.ball_positions = deque(maxlen=30)
        self.rfdetr_ball_detector = None
        self.yolo_ball_model = None
        self._initialize_models()
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        try:
            import yaml
            with open(config_path, 'r') as file:
                return yaml.safe_load(file)
        except Exception as e:
            logger.error(f"Error loading config: {e}")
            return {}
    
    def _initialize_models(self):
        """Initialize detection models"""
        try:
            # Initialize RF-DETR
            rfdetr_path = self.config.get('models', {}).get('rfdetr_ball', 'models/playersnball5.pt')
            if os.path.exists(rfdetr_path):
                self.rfdetr_ball_detector = CustomPyTorchDetector(rfdetr_path)
                logger.info("RF-DETR ball detector initialized")
            else:
                logger.warning(f"RF-DETR model not found at {rfdetr_path}")
        except Exception as e:
            logger.error(f"Error initializing RF-DETR: {e}")
        
        try:
            # Initialize YOLO
            yolo_path = self.config.get('models', {}).get('yolo_ball', 'models/best_ball.pt')
            if os.path.exists(yolo_path):
                self.yolo_ball_model = YOLO(yolo_path)
                logger.info("YOLO ball model initialized")
            else:
                logger.warning(f"YOLO model not found at {yolo_path}")
        except Exception as e:
            logger.error(f"Error initializing YOLO: {e}")
    
    def detect_ball(self, frame: np.ndarray) -> Optional[Tuple[float, float, float]]:
        """Detect ball using RF-DETR first, then YOLO as fallback"""
        try:
            # Try RF-DETR first
            if self.rfdetr_ball_detector:
                try:
                    detections = self.rfdetr_ball_detector.detect(frame)
                    if detections and len(detections) > 0:
                        # Look for ball detection (class 0)
                        for det in detections:
                            if det[4] == 0:  # Ball class
                                x, y, w, h, conf = det[:5]
                                center_x = x + w/2
                                center_y = y + h/2
                                return center_x, center_y, conf
                except Exception as e:
                    logger.debug(f"RF-DETR detection failed: {e}")
            
            # Fallback to YOLO
            if self.yolo_ball_model:
                try:
                    results = self.yolo_ball_model(frame, verbose=False)
                    if results and len(results) > 0:
                        result = results[0]
                        if result.boxes is not None and len(result.boxes) > 0:
                            # Get the first detection
                            box = result.boxes[0]
                            if box.conf > 0.3:  # Confidence threshold
                                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                                center_x = (x1 + x2) / 2
                                center_y = (y1 + y2) / 2
                                return center_x, center_y, float(box.conf)
                except Exception as e:
                    logger.debug(f"YOLO detection failed: {e}")
            
            return None
            
        except Exception as e:
            logger.error(f"Ball detection error: {e}")
            return None

class TrajectoryBounceDemo:
    """Demo class for trajectory-based bounce detection"""
    
    def __init__(self, video_path: str):
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)
        self.trajectory_detector = TrajectoryBounceDetector()
        self.ball_tracker = WorkingBallTracker()
        
        # Display toggles
        self.show_trajectory = True
        self.show_physics = True
        self.show_confidence = True
        
        # Statistics
        self.frame_count = 0
        self.bounce_count = 0
        self.ball_detection_count = 0
        
        if not self.cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        logger.info(f"🎾 Starting Trajectory-Based Bounce Detection Demo")
        logger.info(f"Video: {video_path}")
        logger.info(f"Controls:")
        logger.info(f"  't' - Toggle trajectory display")
        logger.info(f"  'p' - Toggle physics info display")
        logger.info(f"  'c' - Toggle confidence display")
        logger.info(f"  'q' - Quit")
        logger.info(f"  SPACE - Pause/Resume")
    
    def run(self):
        """Run the demo"""
        paused = False
        
        while True:
            if not paused:
                ret, frame = self.cap.read()
                if not ret:
                    break
                
                self.frame_count += 1
                
                # Detect ball
                ball_result = self.ball_tracker.detect_ball(frame)
                if ball_result:
                    x, y, conf = ball_result
                    self.ball_detection_count += 1
                    
                    # Add ball position to trajectory detector
                    self.trajectory_detector.add_ball_position(x, y, self.frame_count)
                    
                    # Detect bounce
                    is_bounce, confidence, bounce_info = self.trajectory_detector.detect_bounce()
                    
                    if is_bounce:
                        self.bounce_count += 1
                        # Draw red dot for bounce
                        cv2.circle(frame, (int(x), int(y)), 8, (0, 0, 255), -1)
                        cv2.putText(frame, f"BOUNCE! {confidence:.2f}", 
                                  (int(x) + 10, int(y) - 10), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                    
                    # Draw ball trajectory
                    if self.show_trajectory:
                        self._draw_trajectory(frame)
                
                # Add overlays
                self._add_overlays(frame)
                
                # Display frame
                cv2.imshow('Trajectory-Based Bounce Detection', frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord(' '):
                paused = not paused
            elif key == ord('t'):
                self.show_trajectory = not self.show_trajectory
            elif key == ord('p'):
                self.show_physics = not self.show_physics
            elif key == ord('c'):
                self.show_confidence = not self.show_confidence
        
        self._cleanup()
        self._print_statistics()
    
    def _draw_trajectory(self, frame):
        """Draw ball trajectory on frame"""
        if len(self.trajectory_detector.ball_positions) < 2:
            return
        
        positions = list(self.trajectory_detector.ball_positions)
        
        # Draw trajectory line
        for i in range(1, len(positions)):
            pt1 = (int(positions[i-1][0]), int(positions[i-1][1]))
            pt2 = (int(positions[i][0]), int(positions[i][1]))
            cv2.line(frame, pt1, pt2, (0, 255, 0), 2)
        
        # Draw position points
        for i, pos in enumerate(positions):
            color = (255, 0, 0) if i == len(positions) - 1 else (0, 255, 255)
            cv2.circle(frame, (int(pos[0]), int(pos[1])), 3, color, -1)
    
    def _add_overlays(self, frame):
        """Add information overlays to frame"""
        # Frame info
        cv2.putText(frame, f"Frame: {self.frame_count}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Ball detection info
        detection_rate = (self.ball_detection_count / self.frame_count * 100) if self.frame_count > 0 else 0
        cv2.putText(frame, f"Ball Detection: {detection_rate:.1f}%", (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Bounce info
        cv2.putText(frame, f"Bounces: {self.bounce_count}", (10, 90), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Trajectory info
        if self.show_physics:
            try:
                trajectory_info = self.trajectory_detector.get_trajectory_info()
                y_offset = 120
                
                cv2.putText(frame, f"Trajectory Length: {trajectory_info.get('positions', 0)}", (10, y_offset), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                y_offset += 25
                
                angles = trajectory_info.get('angles', [])
                if angles:
                    avg_angle = sum(angles) / len(angles)
                    cv2.putText(frame, f"Avg Angle: {avg_angle:.1f}°", (10, y_offset), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    y_offset += 25
                
                cv2.putText(frame, f"Recent Movement: {trajectory_info.get('recent_movement', 0):.1f}", (10, y_offset), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                y_offset += 25
                
                cv2.putText(frame, f"Away from Players: {trajectory_info.get('ball_away_from_players', False)}", (10, y_offset), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            except Exception as e:
                logger.debug(f"Error displaying trajectory info: {e}")
    
    def _cleanup(self):
        """Clean up resources"""
        self.cap.release()
        cv2.destroyAllWindows()
    
    def _print_statistics(self):
        """Print final statistics"""
        logger.info("\n" + "="*60)
        logger.info("TRAJECTORY-BASED BOUNCE DETECTION STATISTICS")
        logger.info("="*60)
        logger.info(f"Total Frames Processed: {self.frame_count}")
        logger.info(f"Balls Detected: {self.ball_detection_count}")
        logger.info(f"Bounces Detected: {self.bounce_count}")
        logger.info(f"Ball Detection Rate: {self.ball_detection_count/self.frame_count*100:.1f}%")
        logger.info(f"Bounce Detection Rate: {self.bounce_count/self.frame_count*100:.1f}%")
        logger.info("="*60)

def main():
    parser = argparse.ArgumentParser(description="Trajectory-Based Bounce Detection Demo")
    parser.add_argument("--video", required=True, help="Path to input video file")
    args = parser.parse_args()
    
    try:
        demo = TrajectoryBounceDemo(args.video)
        demo.run()
    except Exception as e:
        logger.error(f"Demo failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
