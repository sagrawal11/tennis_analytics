"""
Ball Tracking Module using PyTorch TrackNet
Tracks tennis ball trajectory using PyTorch implementation
"""

import cv2
import numpy as np
import torch
import torch.nn as nn
from typing import List, Tuple, Dict, Any, Optional
import logging
from dataclasses import dataclass
from collections import deque
import math
from scipy.spatial import distance

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
    trajectory_id: int
    positions: List[BallPosition]
    start_frame: int
    end_frame: int
    is_valid: bool = True

class ConvBlock(nn.Module):
    """Convolutional block for TrackNet"""
    def __init__(self, in_channels, out_channels, kernel_size=3, pad=1, stride=1, bias=True):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=pad, bias=bias),
            nn.ReLU(),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        return self.block(x)

# Import the original model architecture
from model_original import BallTrackerNet                       
    
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.uniform_(module.weight, -0.05, 0.05)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)

class TrackNetPyTorch:
    """PyTorch-based TrackNet ball tracking for tennis analysis"""
    
    def __init__(self, model_path: str, config: Dict[str, Any]):
        """
        Initialize TrackNet ball tracker
        
        Args:
            model_path: Path to TrackNet model weights
            config: Configuration dictionary
        """
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize model
        self.model = BallTrackerNet()
        if model_path and model_path != "models/tracknet.h5":  # Skip if it's the old TensorFlow path
            try:
                self.model.load_state_dict(torch.load(model_path, map_location=self.device))
                logger.info(f"Loaded TrackNet model from {model_path}")
            except Exception as e:
                logger.warning(f"Could not load model from {model_path}: {e}")
                logger.info("Using randomly initialized model")
        
        self.model.to(self.device)
        self.model.eval()
        
        # Configuration
        self.input_height = config.get('input_height', 360)
        self.input_width = config.get('input_width', 640)
        self.conf_threshold = config.get('conf_threshold', 0.5)
        self.max_dist = config.get('max_dist', 100)
        self.max_gap = config.get('max_gap', 4)
        self.min_track_length = config.get('min_track_length', 5)
        
        # Ball tracking state
        self.frame_buffer = deque(maxlen=3)  # TrackNet needs 3 consecutive frames
        self.ball_positions = deque(maxlen=30)
        self.current_trajectory = None
        self.trajectory_id_counter = 0
        
        logger.info(f"TrackNet PyTorch ball tracker initialized")
    
    def preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Preprocess frame for TrackNet input
        
        Args:
            frame: Input frame (BGR format)
            
        Returns:
            Preprocessed frame
        """
        try:
            # Resize to input size
            frame_resized = cv2.resize(frame, (self.input_width, self.input_height))
            
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
            
            # Normalize to [0, 1]
            frame_normalized = frame_rgb.astype(np.float32) / 255.0
            
            return frame_normalized
            
        except Exception as e:
            logger.error(f"Error in frame preprocessing: {e}")
            return np.zeros((self.input_height, self.input_width, 3), dtype=np.float32)
    
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
            
            # Need at least 3 frames
            if len(self.frame_buffer) < 3:
                return None
            
            # Prepare input sequence (3 consecutive frames)
            frames = list(self.frame_buffer)
            img = self.preprocess_frame(frames[-1])
            img_prev = self.preprocess_frame(frames[-2])
            img_preprev = self.preprocess_frame(frames[-3])
            
            # Concatenate frames along channel dimension
            imgs = np.concatenate((img, img_prev, img_preprev), axis=2)
            
            # Convert to PyTorch format: (batch, channels, height, width)
            imgs = np.rollaxis(imgs, 2, 0)
            inp = np.expand_dims(imgs, axis=0)
            
            # Run inference
            with torch.no_grad():
                out = self.model(torch.from_numpy(inp).float().to(self.device), testing=True)
                output = out.argmax(dim=1).detach().cpu().numpy()
            
            # Postprocess to get ball position
            x_pred, y_pred = self._postprocess(output[0], frame.shape)
            
            if x_pred is not None and y_pred is not None:
                # Create ball position object
                ball_position = BallPosition(
                    x=x_pred,
                    y=y_pred,
                    confidence=1.0,  # TrackNet doesn't provide confidence scores
                    frame_number=len(self.ball_positions),
                    timestamp=len(self.ball_positions) / 30.0  # Assuming 30 FPS
                )
                
                return ball_position
            
        except Exception as e:
            logger.error(f"Error in ball position prediction: {e}")
        
        return None
    
    def _postprocess(self, feature_map: np.ndarray, original_shape: Tuple[int, ...]) -> Tuple[Optional[float], Optional[float]]:
        """
        Postprocess TrackNet output to extract ball position
        
        Args:
            feature_map: Model output feature map
            original_shape: Original frame shape
            
        Returns:
            Tuple of (x, y) coordinates or (None, None)
        """
        try:
            # Reshape feature map
            feature_map *= 255
            feature_map = feature_map.reshape((self.input_height, self.input_width))
            feature_map = feature_map.astype(np.uint8)
            
            # Apply threshold
            ret, heatmap = cv2.threshold(feature_map, 127, 255, cv2.THRESH_BINARY)
            
            # Find circles (ball candidates)
            circles = cv2.HoughCircles(
                heatmap, cv2.HOUGH_GRADIENT, dp=1, minDist=1, 
                param1=50, param2=2, minRadius=2, maxRadius=7
            )
            
            if circles is not None and len(circles) > 0:
                # Take the first (strongest) circle
                x_circle, y_circle = circles[0][0][:2]
                
                # Scale back to original frame coordinates
                h_orig, w_orig = original_shape[:2]
                x_orig = (x_circle / self.input_width) * w_orig
                y_orig = (y_circle / self.input_height) * h_orig
                
                return x_orig, y_orig
            
        except Exception as e:
            logger.error(f"Error in postprocessing: {e}")
        
        return None, None
    
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
        if not self.current_trajectory.positions:
            return True
        
        last_pos = self.current_trajectory.positions[-1]
        
        # Calculate distance between positions
        dist = distance.euclidean(
            (ball_position.x, ball_position.y),
            (last_pos.x, last_pos.y)
        )
        
        # Check if distance is reasonable
        return dist <= self.max_dist
    
    def _finalize_trajectory(self) -> None:
        """Finalize current trajectory and start new one"""
        if self.current_trajectory and len(self.current_trajectory.positions) >= self.min_track_length:
            self.trajectory_id_counter += 1
    
    def get_current_trajectory(self) -> Optional[BallTrajectory]:
        """Get current ball trajectory"""
        return self.current_trajectory
    
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
        
        if trajectory and len(trajectory.positions) > 1:
            # Draw trajectory line
            points = [(int(pos.x), int(pos.y)) for pos in trajectory.positions if pos.x is not None and pos.y is not None]
            
            if len(points) > 1:
                # Draw line connecting points
                for i in range(1, len(points)):
                    cv2.line(frame_copy, points[i-1], points[i], (0, 255, 255), 2)
                
                # Draw current ball position
                if points:
                    cv2.circle(frame_copy, points[-1], 5, (0, 255, 0), -1)
        
        return frame_copy
    
    def analyze_trajectory(self, trajectory: BallTrajectory) -> Dict[str, Any]:
        """
        Analyze ball trajectory
        
        Args:
            trajectory: Ball trajectory to analyze
            
        Returns:
            Dictionary with trajectory analysis
        """
        if not trajectory or len(trajectory.positions) < 2:
            return {}
        
        positions = [pos for pos in trajectory.positions if pos.x is not None and pos.y is not None]
        
        if len(positions) < 2:
            return {}
        
        # Calculate trajectory statistics
        distances = []
        velocities = []
        
        for i in range(1, len(positions)):
            dist = distance.euclidean(
                (positions[i].x, positions[i].y),
                (positions[i-1].x, positions[i-1].y)
            )
            distances.append(dist)
            
            # Calculate velocity (pixels per frame)
            time_diff = positions[i].timestamp - positions[i-1].timestamp
            if time_diff > 0:
                velocity = dist / time_diff
                velocities.append(velocity)
        
        analysis = {
            'trajectory_id': trajectory.trajectory_id,
            'start_frame': trajectory.start_frame,
            'end_frame': trajectory.end_frame,
            'num_positions': len(positions),
            'total_distance': sum(distances),
            'average_distance': np.mean(distances) if distances else 0,
            'max_distance': max(distances) if distances else 0,
            'average_velocity': np.mean(velocities) if velocities else 0,
            'max_velocity': max(velocities) if velocities else 0,
            'is_valid': trajectory.is_valid
        }
        
        return analysis
