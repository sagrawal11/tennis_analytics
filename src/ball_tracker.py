"""
Ball Tracking Module using TrackNet
Tracks tennis ball trajectory and movement patterns
"""

import cv2
import numpy as np
import tensorflow as tf
from typing import List, Tuple, Dict, Any, Optional
import logging
from dataclasses import dataclass
from collections import deque
import math

logger = logging.getLogger(__name__)

@dataclass
class BallPosition:
    """Represents a ball position at a specific time"""
    x: float
    y: float
    confidence: float
    frame_number: int
    timestamp: float

@dataclass
class BallTrajectory:
    """Represents a complete ball trajectory"""
    positions: List[BallPosition]
    start_frame: int
    end_frame: int
    trajectory_id: int
    is_valid: bool = True

class TrackNet:
    """TrackNet-based ball tracking for tennis analysis"""
    
    def __init__(self, model_path: str, config: Dict[str, Any]):
        """
        Initialize TrackNet ball tracker
        
        Args:
            model_path: Path to TrackNet model weights
            config: Configuration dictionary
        """
        self.config = config
        self.model = tf.keras.models.load_model(model_path)
        self.input_size = config.get('input_size', 256)
        self.sequence_length = config.get('sequence_length', 3)
        self.conf_threshold = config.get('conf_threshold', 0.5)
        self.smooth_factor = config.get('smooth_factor', 0.8)
        
        # Ball tracking state
        self.ball_positions = deque(maxlen=30)  # Keep last 30 positions
        self.current_trajectory = None
        self.trajectory_id_counter = 0
        self.frame_buffer = deque(maxlen=self.sequence_length)
        
        logger.info(f"TrackNet ball tracker initialized with model: {model_path}")
    
    def preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Preprocess frame for TrackNet input
        
        Args:
            frame: Input frame (BGR format)
            
        Returns:
            Preprocessed frame
        """
        try:
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Resize to input size
            frame_resized = cv2.resize(frame_rgb, (self.input_size, self.input_size))
            
            # Normalize to [0, 1]
            frame_normalized = frame_resized.astype(np.float32) / 255.0
            
            return frame_normalized
            
        except Exception as e:
            logger.error(f"Error in frame preprocessing: {e}")
            return np.zeros((self.input_size, self.input_size, 3), dtype=np.float32)
    
    def predict_ball_position(self, frame: np.ndarray) -> Optional[BallPosition]:
        """
        Predict ball position in a single frame
        
        Args:
            frame: Input frame
            
        Returns:
            Predicted ball position or None if not detected
        """
        try:
            # Add frame to buffer
            self.frame_buffer.append(frame)
            
            # Need at least sequence_length frames
            if len(self.frame_buffer) < self.sequence_length:
                return None
            
            # Prepare input sequence
            input_sequence = np.array(list(self.frame_buffer))
            
            # Reshape for model input: (1, sequence_length, height, width, channels)
            input_sequence = np.expand_dims(input_sequence, axis=0)
            
            # Run inference
            prediction = self.model.predict(input_sequence, verbose=0)
            
            # Process prediction (assuming output is heatmap)
            heatmap = prediction[0, -1]  # Take last frame prediction
            
            # Find ball position from heatmap
            ball_pos = self._extract_ball_from_heatmap(heatmap, frame.shape)
            
            if ball_pos:
                # Apply smoothing
                smoothed_pos = self._apply_smoothing(ball_pos)
                
                # Create ball position object
                ball_position = BallPosition(
                    x=smoothed_pos[0],
                    y=smoothed_pos[1],
                    confidence=ball_pos[2],
                    frame_number=len(self.ball_positions),
                    timestamp=len(self.ball_positions) / 30.0  # Assuming 30 FPS
                )
                
                return ball_position
            
        except Exception as e:
            logger.error(f"Error in ball position prediction: {e}")
        
        return None
    
    def _extract_ball_from_heatmap(self, heatmap: np.ndarray, original_shape: Tuple[int, ...]) -> Optional[Tuple[float, float, float]]:
        """
        Extract ball position from TrackNet heatmap
        
        Args:
            heatmap: Model output heatmap
            original_shape: Original frame shape
            
        Returns:
            Tuple of (x, y, confidence) or None
        """
        try:
            # Find maximum value in heatmap
            max_val = np.max(heatmap)
            
            if max_val < self.conf_threshold:
                return None
            
            # Find position of maximum value
            max_pos = np.unravel_index(np.argmax(heatmap), heatmap.shape)
            y_pred, x_pred = max_pos
            
            # Convert to original frame coordinates
            h_orig, w_orig = original_shape[:2]
            x_orig = (x_pred / self.input_size) * w_orig
            y_orig = (y_pred / self.input_size) * h_orig
            
            return (x_orig, y_orig, max_val)
            
        except Exception as e:
            logger.error(f"Error extracting ball from heatmap: {e}")
            return None
    
    def _apply_smoothing(self, ball_pos: Tuple[float, float, float]) -> Tuple[float, float]:
        """
        Apply temporal smoothing to ball position
        
        Args:
            ball_pos: Current ball position (x, y, confidence)
            
        Returns:
            Smoothed position (x, y)
        """
        x, y, _ = ball_pos
        
        if len(self.ball_positions) > 0:
            # Get last position
            last_pos = self.ball_positions[-1]
            
            # Apply exponential smoothing
            x_smooth = self.smooth_factor * x + (1 - self.smooth_factor) * last_pos.x
            y_smooth = self.smooth_factor * y + (1 - self.smooth_factor) * last_pos.y
            
            return (x_smooth, y_smooth)
        
        return (x, y)
    
    def update_trajectory(self, ball_position: BallPosition) -> None:
        """
        Update ball trajectory with new position
        
        Args:
            ball_position: New ball position
        """
        # Add to position history
        self.ball_positions.append(ball_position)
        
        # Update or create trajectory
        if self.current_trajectory is None:
            self.current_trajectory = BallTrajectory(
                positions=[ball_position],
                start_frame=ball_position.frame_number,
                end_frame=ball_position.frame_number,
                trajectory_id=self.trajectory_id_counter
            )
        else:
            # Check if this is a continuation of current trajectory
            if self._is_trajectory_continuous(ball_position):
                self.current_trajectory.positions.append(ball_position)
                self.current_trajectory.end_frame = ball_position.frame_number
            else:
                # Start new trajectory
                self._finalize_trajectory()
                self.current_trajectory = BallTrajectory(
                    positions=[ball_position],
                    start_frame=ball_position.frame_number,
                    end_frame=ball_position.frame_number,
                    trajectory_id=self.trajectory_id_counter
                )
    
    def _is_trajectory_continuous(self, ball_position: BallPosition) -> bool:
        """
        Check if new ball position continues current trajectory
        
        Args:
            ball_position: New ball position
            
        Returns:
            True if trajectory is continuous
        """
        if not self.current_trajectory or not self.current_trajectory.positions:
            return False
        
        # Get last position
        last_pos = self.current_trajectory.positions[-1]
        
        # Calculate distance
        distance = math.sqrt((ball_position.x - last_pos.x)**2 + (ball_position.y - last_pos.y)**2)
        
        # Check if distance is reasonable (ball shouldn't jump too far)
        max_distance = 100  # pixels, adjust based on your setup
        max_time_gap = 5  # frames
        
        time_gap = ball_position.frame_number - last_pos.frame_number
        
        return distance <= max_distance and time_gap <= max_time_gap
    
    def _finalize_trajectory(self) -> None:
        """Finalize current trajectory and prepare for next one"""
        if self.current_trajectory:
            # Validate trajectory
            if len(self.current_trajectory.positions) >= self.config.get('min_trajectory_length', 10):
                self.current_trajectory.is_valid = True
            else:
                self.current_trajectory.is_valid = False
            
            # Increment trajectory ID
            self.trajectory_id_counter += 1
    
    def get_current_trajectory(self) -> Optional[BallTrajectory]:
        """Get current ball trajectory"""
        return self.current_trajectory
    
    def analyze_trajectory(self, trajectory: BallTrajectory) -> Dict[str, Any]:
        """
        Analyze ball trajectory for tennis insights
        
        Args:
            trajectory: Ball trajectory to analyze
            
        Returns:
            Dictionary with trajectory analysis
        """
        if not trajectory or not trajectory.positions:
            return {}
        
        analysis = {}
        
        try:
            positions = trajectory.positions
            
            # Calculate trajectory statistics
            x_coords = [pos.x for pos in positions]
            y_coords = [pos.y for pos in positions]
            
            # Trajectory length
            total_distance = 0
            for i in range(1, len(positions)):
                dx = positions[i].x - positions[i-1].x
                dy = positions[i].y - positions[i-1].y
                total_distance += math.sqrt(dx*dx + dy*dy)
            
            # Velocity analysis
            velocities = []
            for i in range(1, len(positions)):
                dt = positions[i].timestamp - positions[i-1].timestamp
                if dt > 0:
                    dx = positions[i].x - positions[i-1].x
                    dy = positions[i].y - positions[i-1].y
                    velocity = math.sqrt(dx*dx + dy*dy) / dt
                    velocities.append(velocity)
            
            # Direction analysis
            if len(positions) >= 2:
                start_pos = positions[0]
                end_pos = positions[-1]
                dx = end_pos.x - start_pos.x
                dy = end_pos.y - start_pos.y
                direction_angle = math.degrees(math.atan2(dy, dx))
            else:
                direction_angle = 0
            
            analysis = {
                'trajectory_id': trajectory.trajectory_id,
                'duration': positions[-1].timestamp - positions[0].timestamp,
                'total_distance': total_distance,
                'average_velocity': np.mean(velocities) if velocities else 0,
                'max_velocity': np.max(velocities) if velocities else 0,
                'direction_angle': direction_angle,
                'start_position': (positions[0].x, positions[0].y),
                'end_position': (positions[-1].x, positions[-1].y),
                'num_positions': len(positions),
                'is_valid': trajectory.is_valid
            }
            
        except Exception as e:
            logger.error(f"Error in trajectory analysis: {e}")
        
        return analysis
    
    def draw_trajectory(self, frame: np.ndarray, trajectory: BallTrajectory) -> np.ndarray:
        """
        Draw ball trajectory on frame
        
        Args:
            frame: Input frame
            trajectory: Ball trajectory to draw
            
        Returns:
            Frame with drawn trajectory
        """
        frame_copy = frame.copy()
        
        if not trajectory or not trajectory.positions:
            return frame_copy
        
        # Draw trajectory line
        for i in range(1, len(trajectory.positions)):
            pos1 = trajectory.positions[i-1]
            pos2 = trajectory.positions[i]
            
            # Color based on velocity (red = fast, blue = slow)
            velocity = math.sqrt((pos2.x - pos1.x)**2 + (pos2.y - pos1.y)**2)
            color = self._velocity_to_color(velocity)
            
            cv2.line(frame_copy, 
                     (int(pos1.x), int(pos1.y)), 
                     (int(pos2.x), int(pos2.y)), 
                     color, 2)
        
        # Draw ball positions
        for i, pos in enumerate(trajectory.positions):
            # Size based on confidence
            radius = max(3, int(pos.confidence * 10))
            
            # Color based on frame number (gradient)
            color = self._frame_to_color(i, len(trajectory.positions))
            
            cv2.circle(frame_copy, (int(pos.x), int(pos.y)), radius, color, -1)
            
            # Draw frame number
            cv2.putText(frame_copy, str(i), (int(pos.x)+5, int(pos.y)-5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)
        
        return frame_copy
    
    def _velocity_to_color(self, velocity: float) -> Tuple[int, int, int]:
        """Convert velocity to BGR color (red = fast, blue = slow)"""
        # Normalize velocity to [0, 1] range
        max_vel = 1000  # Adjust based on your setup
        normalized_vel = min(velocity / max_vel, 1.0)
        
        # Red (fast) to Blue (slow)
        blue = int(255 * (1 - normalized_vel))
        red = int(255 * normalized_vel)
        green = 0
        
        return (blue, green, red)
    
    def _frame_to_color(self, frame_idx: int, total_frames: int) -> Tuple[int, int, int]:
        """Convert frame index to BGR color (gradient)"""
        if total_frames <= 1:
            return (255, 0, 0)
        
        # Normalize to [0, 1]
        normalized = frame_idx / (total_frames - 1)
        
        # Blue to Red gradient
        blue = int(255 * (1 - normalized))
        red = int(255 * normalized)
        green = 0
        
        return (blue, green, red)
