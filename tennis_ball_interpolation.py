#!/usr/bin/env python3
"""
Tennis Ball Position Interpolation
Attempts to fill in missing ball positions during bounces using trajectory interpolation
"""

import cv2
import numpy as np
import pandas as pd
import argparse
import logging
from pathlib import Path
from typing import List, Tuple, Optional
from scipy.interpolate import interp1d
from scipy.optimize import minimize_scalar

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class BallInterpolator:
    def __init__(self, video_path: str, csv_path: str):
        self.video_path = video_path
        self.csv_path = csv_path
        
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
        self.original_trajectory = []
        self.interpolated_trajectory = []
        self.missing_frames = []
        self.filtered_detections = 0
        self.total_detections = 0
        
    def get_ball_position(self, frame_idx: int, last_valid_pos: Optional[Tuple[float, float]] = None, last_valid_frame: int = -1) -> Optional[Tuple[float, float, float]]:
        """Get ball position and confidence for a frame with advanced temporal smoothing"""
        if frame_idx >= len(self.df):
            return None
        
        ball_x = self.df.iloc[frame_idx]['ball_x']
        ball_y = self.df.iloc[frame_idx]['ball_y']
        ball_confidence = self.df.iloc[frame_idx]['ball_confidence']
        
        if pd.isna(ball_x) or pd.isna(ball_y) or ball_confidence < 0.3:
            return None
        
        current_pos = (float(ball_x), float(ball_y))
        self.total_detections += 1
        
        # Apply advanced temporal smoothing using 10 frames before and after
        smoothed_pos = self._apply_temporal_smoothing(frame_idx, current_pos)
        
        if smoothed_pos != current_pos:
            self.filtered_detections += 1
            logger.info(f"Frame {frame_idx}: Ball position smoothed using temporal analysis")
            # Reduce confidence for smoothed positions
            ball_confidence *= 0.8
        
        return (smoothed_pos[0], smoothed_pos[1], float(ball_confidence))
    
    def _apply_temporal_smoothing(self, frame_idx: int, current_pos: Tuple[float, float]) -> Tuple[float, float]:
        """Apply physics-based temporal smoothing with trajectory modeling"""
        # Adaptive window size based on ball speed
        window_size = self._calculate_adaptive_window(frame_idx)
        
        start_frame = max(0, frame_idx - window_size)
        end_frame = min(len(self.df) - 1, frame_idx + window_size)
        
        # Collect valid positions in the temporal window
        valid_positions = []
        for f in range(start_frame, end_frame + 1):
            if f == frame_idx:
                continue  # Skip current frame
                
            ball_x = self.df.iloc[f]['ball_x']
            ball_y = self.df.iloc[f]['ball_y']
            ball_confidence = self.df.iloc[f]['ball_confidence']
            
            if not pd.isna(ball_x) and not pd.isna(ball_y) and ball_confidence > 0.3:
                valid_positions.append({
                    'frame': f,
                    'x': float(ball_x),
                    'y': float(ball_y),
                    'confidence': float(ball_confidence),
                    'distance_from_current': abs(f - frame_idx)
                })
        
        if len(valid_positions) < 3:  # Need at least 3 points for meaningful smoothing
            return current_pos
        
        # Try multiple smoothing methods and take the best one
        methods = [
            self._physics_based_smoothing(frame_idx, current_pos, valid_positions),
            self._velocity_based_smoothing(frame_idx, current_pos, valid_positions),
            self._confidence_weighted_smoothing(current_pos, valid_positions)
        ]
        
        # Choose the method with the best confidence score
        best_method = max(methods, key=lambda x: x[1])  # x[1] is confidence
        smoothed_pos, confidence = best_method
        
        # Only apply smoothing if confidence is high and difference is significant
        distance = np.sqrt((current_pos[0] - smoothed_pos[0])**2 + (current_pos[1] - smoothed_pos[1])**2)
        
        if distance > 50 and confidence > 0.6:  # More sensitive threshold
            return smoothed_pos
        else:
            return current_pos
    
    def _calculate_adaptive_window(self, frame_idx: int) -> int:
        """Calculate adaptive window size based on ball speed"""
        # Look at recent ball movement to estimate speed
        recent_frames = min(5, frame_idx)
        if recent_frames < 2:
            return 10  # Default window
        
        # Calculate average speed over recent frames
        total_distance = 0
        valid_frames = 0
        
        for i in range(1, recent_frames + 1):
            if frame_idx - i >= 0:
                prev_x = self.df.iloc[frame_idx - i]['ball_x']
                prev_y = self.df.iloc[frame_idx - i]['ball_y']
                curr_x = self.df.iloc[frame_idx - i + 1]['ball_x']
                curr_y = self.df.iloc[frame_idx - i + 1]['ball_y']
                
                if not pd.isna(prev_x) and not pd.isna(prev_y) and not pd.isna(curr_x) and not pd.isna(curr_y):
                    distance = np.sqrt((curr_x - prev_x)**2 + (curr_y - prev_y)**2)
                    total_distance += distance
                    valid_frames += 1
        
        if valid_frames == 0:
            return 10
        
        avg_speed = total_distance / valid_frames
        
        # Adaptive window: faster balls need larger windows
        if avg_speed > 50:  # High speed
            return 15
        elif avg_speed > 20:  # Medium speed
            return 12
        else:  # Low speed
            return 8
    
    def _physics_based_smoothing(self, frame_idx: int, current_pos: Tuple[float, float], valid_positions: List[dict]) -> Tuple[Tuple[float, float], float]:
        """Physics-based smoothing using parabolic trajectory fitting"""
        if len(valid_positions) < 4:
            return current_pos, 0.0
        
        # Separate x and y coordinates with frame numbers
        frames = [pos['frame'] for pos in valid_positions]
        x_coords = [pos['x'] for pos in valid_positions]
        y_coords = [pos['y'] for pos in valid_positions]
        confidences = [pos['confidence'] for pos in valid_positions]
        
        try:
            # Fit parabolas to x and y trajectories
            x_poly = np.polyfit(frames, x_coords, 2)  # Quadratic fit
            y_poly = np.polyfit(frames, y_coords, 2)  # Quadratic fit
            
            # Predict position at current frame
            predicted_x = np.polyval(x_poly, frame_idx)
            predicted_y = np.polyval(y_poly, frame_idx)
            
            # Calculate confidence based on how well the parabola fits
            x_residuals = [abs(np.polyval(x_poly, f) - x) for f, x in zip(frames, x_coords)]
            y_residuals = [abs(np.polyval(y_poly, f) - y) for f, y in zip(frames, y_coords)]
            
            avg_x_error = np.mean(x_residuals)
            avg_y_error = np.mean(y_residuals)
            
            # Confidence based on fit quality (lower error = higher confidence)
            confidence = max(0.0, 1.0 - (avg_x_error + avg_y_error) / 200.0)
            
            return (predicted_x, predicted_y), confidence
            
        except:
            return current_pos, 0.0
    
    def _velocity_based_smoothing(self, frame_idx: int, current_pos: Tuple[float, float], valid_positions: List[dict]) -> Tuple[Tuple[float, float], float]:
        """Velocity-based smoothing using recent movement patterns"""
        if len(valid_positions) < 3:
            return current_pos, 0.0
        
        # Sort by frame number
        sorted_positions = sorted(valid_positions, key=lambda x: x['frame'])
        
        # Calculate velocities
        velocities = []
        for i in range(1, len(sorted_positions)):
            prev = sorted_positions[i-1]
            curr = sorted_positions[i]
            
            dx = curr['x'] - prev['x']
            dy = curr['y'] - prev['y']
            dt = curr['frame'] - prev['frame']
            
            if dt > 0:
                velocities.append({
                    'frame': curr['frame'],
                    'vx': dx / dt,
                    'vy': dy / dt,
                    'confidence': min(prev['confidence'], curr['confidence'])
                })
        
        if len(velocities) < 2:
            return current_pos, 0.0
        
        # Find the most recent velocity
        recent_velocity = velocities[-1]
        
        # Predict position based on velocity
        frames_since_last = frame_idx - sorted_positions[-1]['frame']
        predicted_x = sorted_positions[-1]['x'] + recent_velocity['vx'] * frames_since_last
        predicted_y = sorted_positions[-1]['y'] + recent_velocity['vy'] * frames_since_last
        
        # Confidence based on velocity consistency
        velocity_consistency = 1.0 - np.std([v['vx'] for v in velocities]) / 100.0
        confidence = max(0.0, min(1.0, velocity_consistency * recent_velocity['confidence']))
        
        return (predicted_x, predicted_y), confidence
    
    def _confidence_weighted_smoothing(self, current_pos: Tuple[float, float], valid_positions: List[dict]) -> Tuple[Tuple[float, float], float]:
        """Original confidence-weighted smoothing as fallback"""
        total_weight = 0
        weighted_x = 0
        weighted_y = 0
        
        for pos in valid_positions:
            # Weight by confidence and inverse temporal distance
            temporal_weight = 1.0 / (pos['distance_from_current'] + 1)
            confidence_weight = pos['confidence']
            total_weight += temporal_weight * confidence_weight
            
            weighted_x += pos['x'] * temporal_weight * confidence_weight
            weighted_y += pos['y'] * temporal_weight * confidence_weight
        
        if total_weight == 0:
            return current_pos, 0.0
        
        smoothed_x = weighted_x / total_weight
        smoothed_y = weighted_y / total_weight
        
        # Confidence based on how many points contributed
        confidence = min(1.0, len(valid_positions) / 10.0)
        
        return (smoothed_x, smoothed_y), confidence
    
    def find_trajectory_segments(self) -> List[dict]:
        """Find continuous segments of ball detections with distance validation"""
        segments = []
        current_segment = []
        last_valid_pos = None
        last_valid_frame = -1
        
        for frame_idx in range(len(self.df)):
            # Only apply distance constraint to consecutive frames
            if last_valid_frame != -1 and frame_idx - last_valid_frame > 1:
                # There's a gap, don't apply distance constraint
                last_valid_pos = None
            
            ball_pos = self.get_ball_position(frame_idx, last_valid_pos, last_valid_frame)
            
            if ball_pos is not None:
                current_segment.append({
                    'frame': frame_idx,
                    'x': ball_pos[0],
                    'y': ball_pos[1],
                    'confidence': ball_pos[2]
                })
                last_valid_pos = (ball_pos[0], ball_pos[1])
                last_valid_frame = frame_idx
            else:
                if current_segment:
                    segments.append(current_segment)
                    current_segment = []
                # Reset tracking when we lose detection
                last_valid_pos = None
                last_valid_frame = -1
        
        # Add final segment if exists
        if current_segment:
            segments.append(current_segment)
        
        logger.info(f"Found {len(segments)} trajectory segments (with distance validation)")
        logger.info(f"Smoothed {self.filtered_detections}/{self.total_detections} detections ({self.filtered_detections/self.total_detections*100:.1f}%) due to distance > 100px")
        for i, seg in enumerate(segments):
            logger.info(f"  Segment {i+1}: frames {seg[0]['frame']}-{seg[-1]['frame']} ({len(seg)} points)")
        
        return segments
    
    def interpolate_parabolic_trajectory(self, segment: List[dict], gap_start: int, gap_end: int) -> List[dict]:
        """Interpolate ball position using parabolic fitting"""
        if len(segment) < 3:
            return []
        
        # Extract x, y, frame data
        frames = [p['frame'] for p in segment]
        x_coords = [p['x'] for p in segment]
        y_coords = [p['y'] for p in segment]
        
        # Fit parabolas for x and y separately
        x_poly = np.polyfit(frames, x_coords, 2)  # Quadratic fit
        y_poly = np.polyfit(frames, y_coords, 2)  # Quadratic fit
        
        # Interpolate for gap frames
        interpolated = []
        for frame in range(gap_start, gap_end + 1):
            if frame not in frames:  # Only interpolate missing frames
                x_interp = np.polyval(x_poly, frame)
                y_interp = np.polyval(y_poly, frame)
                
                # Ensure coordinates are within video bounds
                x_interp = max(0, min(self.width - 1, x_interp))
                y_interp = max(0, min(self.height - 1, y_interp))
                
                interpolated.append({
                    'frame': frame,
                    'x': x_interp,
                    'y': y_interp,
                    'confidence': 0.5,  # Lower confidence for interpolated points
                    'interpolated': True
                })
        
        return interpolated
    
    def interpolate_linear_trajectory(self, segment: List[dict], gap_start: int, gap_end: int) -> List[dict]:
        """Interpolate ball position using linear interpolation"""
        if len(segment) < 2:
            return []
        
        # Find points before and after gap
        before_points = [p for p in segment if p['frame'] < gap_start]
        after_points = [p for p in segment if p['frame'] > gap_end]
        
        if not before_points or not after_points:
            return []
        
        # Use last point before and first point after
        p1 = before_points[-1]
        p2 = after_points[0]
        
        # Linear interpolation
        interpolated = []
        total_frames = p2['frame'] - p1['frame']
        
        for frame in range(gap_start, gap_end + 1):
            if frame not in [p['frame'] for p in segment]:
                # Calculate interpolation factor
                t = (frame - p1['frame']) / total_frames
                
                x_interp = p1['x'] + t * (p2['x'] - p1['x'])
                y_interp = p1['y'] + t * (p2['y'] - p1['y'])
                
                # Ensure coordinates are within video bounds
                x_interp = max(0, min(self.width - 1, x_interp))
                y_interp = max(0, min(self.height - 1, y_interp))
                
                interpolated.append({
                    'frame': frame,
                    'x': x_interp,
                    'y': y_interp,
                    'confidence': 0.3,  # Lower confidence for interpolated points
                    'interpolated': True
                })
        
        return interpolated
    
    def interpolate_trajectory(self):
        """Main interpolation logic"""
        segments = self.find_trajectory_segments()
        
        if len(segments) < 2:
            logger.warning("Not enough trajectory segments for interpolation")
            return
        
        # Find gaps between segments
        gaps = []
        for i in range(len(segments) - 1):
            current_end = segments[i][-1]['frame']
            next_start = segments[i + 1][0]['frame']
            
            if next_start - current_end > 1:  # Gap exists
                gaps.append({
                    'start': current_end + 1,
                    'end': next_start - 1,
                    'before_segment': segments[i],
                    'after_segment': segments[i + 1]
                })
        
        logger.info(f"Found {len(gaps)} gaps to interpolate")
        
        # Interpolate each gap
        all_interpolated = []
        for gap in gaps:
            logger.info(f"Interpolating gap: frames {gap['start']}-{gap['end']}")
            
            # Try parabolic interpolation first (more accurate for ball trajectories)
            interpolated = self.interpolate_parabolic_trajectory(
                gap['before_segment'] + gap['after_segment'],
                gap['start'],
                gap['end']
            )
            
            # Fall back to linear if parabolic fails
            if not interpolated:
                interpolated = self.interpolate_linear_trajectory(
                    gap['before_segment'] + gap['after_segment'],
                    gap['start'],
                    gap['end']
                )
            
            if interpolated:
                all_interpolated.extend(interpolated)
                logger.info(f"  Interpolated {len(interpolated)} points")
            else:
                logger.warning(f"  Failed to interpolate gap {gap['start']}-{gap['end']}")
        
        # Combine original and interpolated points
        all_points = []
        
        # Add original points
        for segment in segments:
            for point in segment:
                point['interpolated'] = False
                all_points.append(point)
        
        # Add interpolated points
        all_points.extend(all_interpolated)
        
        # Sort by frame number
        all_points.sort(key=lambda p: p['frame'])
        
        self.interpolated_trajectory = all_points
        logger.info(f"Total trajectory points: {len(all_points)} (original + interpolated)")
        logger.info(f"Interpolated points: {len(all_interpolated)}")
    
    def draw_trajectory(self, frame: np.ndarray, current_frame: int) -> np.ndarray:
        """Draw the interpolated trajectory up to current frame only"""
        if not self.interpolated_trajectory:
            return frame
        
        # Get points up to current frame
        visible_points = [p for p in self.interpolated_trajectory if p['frame'] <= current_frame]
        
        if len(visible_points) < 2:
            return frame
        
        # Draw trajectory lines
        for i in range(1, len(visible_points)):
            prev_point = visible_points[i-1]
            curr_point = visible_points[i]
            
            # Skip if points are too far apart in time
            if curr_point['frame'] - prev_point['frame'] > 5:
                continue
            
            # Color based on whether point is interpolated
            if curr_point.get('interpolated', False):
                color = (0, 255, 255)  # Yellow for interpolated
                thickness = 1
            else:
                color = (0, 255, 0)  # Green for original
                thickness = 2
            
            # Draw line segment
            cv2.line(frame, 
                    (int(prev_point['x']), int(prev_point['y'])),
                    (int(curr_point['x']), int(curr_point['y'])),
                    color, thickness)
        
        # Draw trajectory points
        for point in visible_points:
            # Color based on whether point is interpolated
            if point.get('interpolated', False):
                color = (0, 255, 255)  # Yellow for interpolated
                radius = 2
            else:
                color = (0, 255, 0)  # Green for original
                radius = 3
            
            # Draw point
            cv2.circle(frame, (int(point['x']), int(point['y'])), radius, color, -1)
        
        return frame
    
    def run_viewer(self):
        """Run the interactive viewer"""
        # First, do the interpolation
        self.interpolate_trajectory()
        
        print("Ball Interpolation Viewer")
        print("Controls:")
        print("  'q' - Quit")
        print("  'p' - Pause/Resume")
        print("  'r' - Restart")
        print("  's' - Step frame by frame")
        print("  'f' - Fast forward (skip 10 frames)")
        print("  'b' - Backward (go back 10 frames)")
        print("  'g' - Go to specific frame")
        print("  'h' - Show/hide help")
        
        paused = False
        step_mode = False
        show_help = True
        current_frame = 0
        
        # Set initial frame
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("End of video")
                break
            
            # Draw interpolated trajectory
            frame_with_trajectory = self.draw_trajectory(frame.copy(), current_frame)
            
            # Add info text
            info_text = f"Frame: {current_frame}/{self.total_frames}"
            cv2.putText(frame_with_trajectory, info_text, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Add legend
            legend_text = "Green = Original, Yellow = Interpolated"
            cv2.putText(frame_with_trajectory, legend_text, (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Add help text
            if show_help:
                help_text = [
                    "Controls: q=quit, p=pause, r=restart, s=step",
                    "f=forward, b=backward, g=goto, h=help"
                ]
                for i, text in enumerate(help_text):
                    cv2.putText(frame_with_trajectory, text, (10, 90 + i*20),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Show frame
            cv2.imshow('Ball Interpolation Viewer', frame_with_trajectory)
            
            # Handle keyboard input
            key = cv2.waitKey(30) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('p'):
                paused = not paused
                step_mode = False
                print("Paused" if paused else "Resumed")
            elif key == ord('r'):
                current_frame = 0
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
                paused = False
                step_mode = False
                print("Restarted")
            elif key == ord('s'):
                step_mode = True
                paused = False
                print("Step mode enabled")
            elif key == ord('f'):
                current_frame = min(self.total_frames - 1, current_frame + 10)
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
                print(f"Fast forward to frame {current_frame}")
            elif key == ord('b'):
                current_frame = max(0, current_frame - 10)
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
                print(f"Backward to frame {current_frame}")
            elif key == ord('g'):
                try:
                    frame_num = int(input("Enter frame number: "))
                    if 0 <= frame_num < self.total_frames:
                        current_frame = frame_num
                        self.cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
                        print(f"Jumped to frame {current_frame}")
                    else:
                        print(f"Frame number must be between 0 and {self.total_frames-1}")
                except ValueError:
                    print("Invalid frame number")
            elif key == ord('h'):
                show_help = not show_help
                print("Help", "shown" if show_help else "hidden")
            
            # Advance frame if not paused and not in step mode
            if not paused and not step_mode:
                current_frame += 1
            elif step_mode:
                # In step mode, only advance when 's' is pressed again
                pass
            else:
                # Paused, don't advance
                pass
        
        # Cleanup
        self.cap.release()
        cv2.destroyAllWindows()
        print("Viewer closed")

def main():
    parser = argparse.ArgumentParser(description='Interpolate missing ball positions in trajectory')
    parser.add_argument('--video', required=True, help='Input video file')
    parser.add_argument('--csv', required=True, help='CSV file with ball positions')
    
    args = parser.parse_args()
    
    # Check if files exist
    if not Path(args.video).exists():
        logger.error(f"Video file not found: {args.video}")
        return
    
    if not Path(args.csv).exists():
        logger.error(f"CSV file not found: {args.csv}")
        return
    
    try:
        interpolator = BallInterpolator(args.video, args.csv)
        interpolator.run_viewer()
        logger.info("Ball interpolation viewer completed!")
        
    except Exception as e:
        logger.error(f"Error processing video: {e}")
        raise

if __name__ == "__main__":
    main()
