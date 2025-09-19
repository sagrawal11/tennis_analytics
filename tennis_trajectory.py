#!/usr/bin/env python3
"""
Tennis Ball Trajectory Visualizer
Shows the ball's path over time with a color gradient from red (start) to blue (end)
"""

import cv2
import numpy as np
import pandas as pd
import argparse
import logging
from pathlib import Path
from typing import List, Tuple, Optional

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class BallTrajectoryVisualizer:
    def __init__(self, video_path: str, csv_path: str, output_path: str):
        self.video_path = video_path
        self.csv_path = csv_path
        self.output_path = output_path
        
        # Load video
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        # Get video properties
        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        logger.info(f"Video: {self.width}x{self.height} @ {self.fps}fps, {self.total_frames} frames")
        
        # Load CSV data
        self.df = pd.read_csv(csv_path)
        logger.info(f"Loaded {len(self.df)} rows from CSV")
        
        # Ball trajectory data
        self.ball_trajectory = []
        self.max_trajectory_length = 100  # Keep last 100 points for performance
        
        # Video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.out = cv2.VideoWriter(output_path, fourcc, self.fps, (self.width, self.height))
        
    
    def get_color_for_time(self, time_ratio: float) -> Tuple[int, int, int]:
        """Get BGR color based on time ratio (0.0 = red, 1.0 = blue)"""
        # Clamp to [0, 1]
        time_ratio = max(0.0, min(1.0, time_ratio))
        
        # Red to Blue gradient
        red = int(255 * (1.0 - time_ratio))
        blue = int(255 * time_ratio)
        green = 0
        
        return (blue, green, red)  # BGR format for OpenCV
    
    def draw_trajectory(self, frame: np.ndarray, current_frame: int) -> np.ndarray:
        """Draw the ball trajectory with color gradient"""
        if len(self.ball_trajectory) < 2:
            return frame
        
        # Draw trajectory lines
        for i in range(1, len(self.ball_trajectory)):
            prev_point = self.ball_trajectory[i-1]
            curr_point = self.ball_trajectory[i]
            
            # Calculate time ratio for color
            time_ratio = i / (len(self.ball_trajectory) - 1)
            color = self.get_color_for_time(time_ratio)
            
            # Draw line segment
            cv2.line(frame, 
                    (int(prev_point[0]), int(prev_point[1])),
                    (int(curr_point[0]), int(curr_point[1])),
                    color, 2)
        
        # Draw trajectory points
        for i, point in enumerate(self.ball_trajectory):
            time_ratio = i / (len(self.ball_trajectory) - 1)
            color = self.get_color_for_time(time_ratio)
            
            # Draw point
            cv2.circle(frame, (int(point[0]), int(point[1])), 3, color, -1)
            
            # Draw frame number for recent points
            if i >= len(self.ball_trajectory) - 10:  # Last 10 points
                cv2.putText(frame, f"{i}", 
                           (int(point[0]) + 5, int(point[1]) - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        return frame
    
    def process_video(self):
        """Process the entire video and create trajectory visualization"""
        frame_count = 0
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            # Get ball position for this frame
            if frame_count < len(self.df):
                ball_x = self.df.iloc[frame_count]['ball_x']
                ball_y = self.df.iloc[frame_count]['ball_y']
                ball_confidence = self.df.iloc[frame_count]['ball_confidence']
                
                # Only use ball position if confidence is reasonable
                if not pd.isna(ball_x) and not pd.isna(ball_y) and ball_confidence > 0.3:
                    ball_pos = (float(ball_x), float(ball_y))
                else:
                    ball_pos = None
                
                if ball_pos is not None:
                    # Add to trajectory
                    self.ball_trajectory.append(ball_pos)
                    
                    # Keep trajectory length manageable
                    if len(self.ball_trajectory) > self.max_trajectory_length:
                        self.ball_trajectory.pop(0)
            
            # Draw trajectory on frame
            frame_with_trajectory = self.draw_trajectory(frame.copy(), frame_count)
            
            # Add info text
            info_text = f"Frame: {frame_count}/{self.total_frames} | Trajectory Points: {len(self.ball_trajectory)}"
            cv2.putText(frame_with_trajectory, info_text, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Add color legend
            legend_text = "Red = Start, Blue = End"
            cv2.putText(frame_with_trajectory, legend_text, (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Write frame
            self.out.write(frame_with_trajectory)
            
            frame_count += 1
            if frame_count % 30 == 0:
                logger.info(f"Processed {frame_count}/{self.total_frames} frames ({frame_count/self.total_frames*100:.1f}%)")
        
        # Cleanup
        self.cap.release()
        self.out.release()
        
        logger.info(f"Trajectory visualization saved to: {self.output_path}")
        logger.info(f"Total trajectory points: {len(self.ball_trajectory)}")

def main():
    parser = argparse.ArgumentParser(description='Visualize tennis ball trajectory with color gradient')
    parser.add_argument('--video', required=True, help='Input video file')
    parser.add_argument('--csv', required=True, help='CSV file with ball positions')
    parser.add_argument('--output', required=True, help='Output video file')
    
    args = parser.parse_args()
    
    # Check if files exist
    if not Path(args.video).exists():
        logger.error(f"Video file not found: {args.video}")
        return
    
    if not Path(args.csv).exists():
        logger.error(f"CSV file not found: {args.csv}")
        return
    
    try:
        visualizer = BallTrajectoryVisualizer(args.video, args.csv, args.output)
        visualizer.process_video()
        logger.info("Trajectory visualization completed successfully!")
        
    except Exception as e:
        logger.error(f"Error processing video: {e}")
        raise

if __name__ == "__main__":
    main()
