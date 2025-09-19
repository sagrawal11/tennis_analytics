#!/usr/bin/env python3
"""
Robust Tennis Ball Bounce Detection
Works directly with raw ball data to detect bounces using multiple independent methods
"""

import cv2
import numpy as np
import pandas as pd
import argparse
import logging
from pathlib import Path
from typing import List, Tuple, Optional, Dict
from collections import deque
import math

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RobustBounceDetector:
    def __init__(self, video_path: str, csv_path: str, output_path: str = None):
        self.video_path = video_path
        self.csv_path = csv_path
        self.output_path = output_path or video_path.replace('.mp4', '_bounce_detection.mp4')
        
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
        
        # Video writer for saving
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.out = cv2.VideoWriter(self.output_path, fourcc, self.fps, (self.width, self.height))
        
        # Bounce detection parameters
        self.min_velocity_change = 0.3  # Minimum velocity change for bounce
        self.min_direction_change = 30  # Minimum direction change in degrees
        self.analysis_window = 10  # Frames to analyze around potential bounce
        self.min_bounce_gap = 15  # Minimum frames between bounces
        self.confidence_threshold = 0.6  # Minimum confidence for bounce (back to strict)
        
        # Court geometry (from overhead perspective)
        self.court_height = self.height * 0.6  # Approximate court height in pixels
        self.net_position = self.height * 0.4  # Approximate net position
        
        # Bounce history
        self.detected_bounces = []
        self.last_bounce_frame = -1
        self.bounce_locations = []  # Store bounce locations for marking
        
        # Ball position smoothing for better bounce locations
        self.ball_position_history = []
        self.max_history = 10
        
    def get_ball_position(self, frame_idx: int) -> Optional[Tuple[float, float, float]]:
        """Get ball position and confidence for a frame"""
        if frame_idx >= len(self.df):
            return None
        
        ball_x = self.df.iloc[frame_idx]['ball_x']
        ball_y = self.df.iloc[frame_idx]['ball_y']
        ball_confidence = self.df.iloc[frame_idx]['ball_confidence']
        
        if pd.isna(ball_x) or pd.isna(ball_y) or ball_confidence < 0.2:
            return None
        
        return (float(ball_x), float(ball_y), float(ball_confidence))
    
    def get_smoothed_ball_position(self, frame_idx: int) -> Optional[Tuple[float, float, float]]:
        """Get smoothed ball position using recent history"""
        raw_pos = self.get_ball_position(frame_idx)
        if raw_pos is None:
            return None
        
        x, y, confidence = raw_pos
        
        # Add to history
        self.ball_position_history.append((x, y, confidence, frame_idx))
        
        # Keep only recent history
        if len(self.ball_position_history) > self.max_history:
            self.ball_position_history.pop(0)
        
        # If we have enough history, smooth the position
        if len(self.ball_position_history) >= 3:
            # Use weighted average based on confidence
            total_weight = 0
            weighted_x = 0
            weighted_y = 0
            
            for hist_x, hist_y, hist_conf, hist_frame in self.ball_position_history:
                # Weight by confidence and recency
                weight = hist_conf * (1.0 - abs(hist_frame - frame_idx) / 10.0)
                weight = max(0, weight)  # Don't use negative weights
                
                weighted_x += hist_x * weight
                weighted_y += hist_y * weight
                total_weight += weight
            
            if total_weight > 0:
                smoothed_x = weighted_x / total_weight
                smoothed_y = weighted_y / total_weight
                return (smoothed_x, smoothed_y, confidence)
        
        return raw_pos
    
    def get_ball_trajectory_window(self, center_frame: int) -> List[Dict]:
        """Get ball positions in a window around center_frame"""
        window = []
        start_frame = max(0, center_frame - self.analysis_window)
        end_frame = min(len(self.df) - 1, center_frame + self.analysis_window)
        
        for frame_idx in range(start_frame, end_frame + 1):
            ball_pos = self.get_ball_position(frame_idx)
            if ball_pos is not None:
                window.append({
                    'frame': frame_idx,
                    'x': ball_pos[0],
                    'y': ball_pos[1],
                    'confidence': ball_pos[2]
                })
        
        return window
    
    def detect_velocity_bounce(self, trajectory: List[Dict], center_frame: int) -> Tuple[bool, float]:
        """Detect bounce based on velocity direction changes"""
        if len(trajectory) < 5:
            return False, 0.0
        
        # Sort by frame number
        sorted_traj = sorted(trajectory, key=lambda x: x['frame'])
        
        # Find center point in trajectory
        center_idx = None
        for i, point in enumerate(sorted_traj):
            if point['frame'] == center_frame:
                center_idx = i
                break
        
        if center_idx is None or center_idx < 2 or center_idx >= len(sorted_traj) - 2:
            return False, 0.0
        
        # Calculate velocities before and after center
        before_velocities = []
        after_velocities = []
        
        # Before center
        for i in range(max(0, center_idx - 3), center_idx):
            if i + 1 < center_idx:
                prev = sorted_traj[i]
                curr = sorted_traj[i + 1]
                dt = curr['frame'] - prev['frame']
                if dt > 0:
                    vx = (curr['x'] - prev['x']) / dt
                    vy = (curr['y'] - prev['y']) / dt
                    before_velocities.append((vx, vy))
        
        # After center
        for i in range(center_idx, min(len(sorted_traj) - 1, center_idx + 3)):
            if i + 1 < len(sorted_traj):
                curr = sorted_traj[i]
                next_ = sorted_traj[i + 1]
                dt = next_['frame'] - curr['frame']
                if dt > 0:
                    vx = (next_['x'] - curr['x']) / dt
                    vy = (next_['y'] - curr['y']) / dt
                    after_velocities.append((vx, vy))
        
        if len(before_velocities) < 2 or len(after_velocities) < 2:
            return False, 0.0
        
        # Calculate average velocities
        avg_before_vx = np.mean([v[0] for v in before_velocities])
        avg_before_vy = np.mean([v[1] for v in before_velocities])
        avg_after_vx = np.mean([v[0] for v in after_velocities])
        avg_after_vy = np.mean([v[1] for v in after_velocities])
        
        # Check for velocity direction change
        before_direction = math.atan2(avg_before_vy, avg_before_vx)
        after_direction = math.atan2(avg_after_vy, avg_after_vx)
        
        direction_change = abs(after_direction - before_direction)
        if direction_change > math.pi:
            direction_change = 2 * math.pi - direction_change
        
        # Convert to degrees
        direction_change_deg = math.degrees(direction_change)
        
        # Check for significant direction change
        if direction_change_deg > self.min_direction_change:
            # Calculate confidence based on direction change magnitude
            confidence = min(1.0, direction_change_deg / 90.0)
            return True, confidence
        
        return False, 0.0
    
    def detect_height_bounce(self, trajectory: List[Dict], center_frame: int) -> Tuple[bool, float]:
        """Detect bounce based on height changes (overhead perspective)"""
        if len(trajectory) < 5:
            return False, 0.0
        
        # Sort by frame number
        sorted_traj = sorted(trajectory, key=lambda x: x['frame'])
        
        # Find center point
        center_idx = None
        for i, point in enumerate(sorted_traj):
            if point['frame'] == center_frame:
                center_idx = i
                break
        
        if center_idx is None or center_idx < 2 or center_idx >= len(sorted_traj) - 2:
            return False, 0.0
        
        center_point = sorted_traj[center_idx]
        
        # Check if ball is near court level (back baseline area)
        court_distance = abs(center_point['y'] - self.court_height)
        if court_distance > 100:  # Too far from court
            return False, 0.0
        
        # Check trajectory before and after
        before_points = sorted_traj[max(0, center_idx - 3):center_idx]
        after_points = sorted_traj[center_idx + 1:min(len(sorted_traj), center_idx + 4)]
        
        if len(before_points) < 2 or len(after_points) < 2:
            return False, 0.0
        
        # Check if ball was going towards court before and away after
        before_avg_y = np.mean([p['y'] for p in before_points])
        after_avg_y = np.mean([p['y'] for p in after_points])
        
        # From overhead: Y decreases towards back baseline (court level)
        going_towards_court = before_avg_y > center_point['y']
        going_away_from_court = after_avg_y > center_point['y']
        
        if going_towards_court and going_away_from_court:
            # Calculate confidence based on how close to court level
            confidence = max(0.0, 1.0 - court_distance / 100.0)
            return True, confidence
        
        return False, 0.0
    
    def detect_acceleration_bounce(self, trajectory: List[Dict], center_frame: int) -> Tuple[bool, float]:
        """Detect bounce based on acceleration changes"""
        if len(trajectory) < 7:
            return False, 0.0
        
        # Sort by frame number
        sorted_traj = sorted(trajectory, key=lambda x: x['frame'])
        
        # Find center point
        center_idx = None
        for i, point in enumerate(sorted_traj):
            if point['frame'] == center_frame:
                center_idx = i
                break
        
        if center_idx is None or center_idx < 3 or center_idx >= len(sorted_traj) - 3:
            return False, 0.0
        
        # Calculate accelerations
        accelerations = []
        for i in range(1, len(sorted_traj) - 1):
            prev = sorted_traj[i - 1]
            curr = sorted_traj[i]
            next_ = sorted_traj[i + 1]
            
            # Calculate velocities
            v1x = (curr['x'] - prev['x']) / (curr['frame'] - prev['frame'])
            v1y = (curr['y'] - prev['y']) / (curr['frame'] - prev['frame'])
            v2x = (next_['x'] - curr['x']) / (next_['frame'] - curr['frame'])
            v2y = (next_['y'] - curr['y']) / (next_['frame'] - curr['frame'])
            
            # Calculate acceleration
            ax = v2x - v1x
            ay = v2y - v1y
            acceleration = math.sqrt(ax**2 + ay**2)
            
            accelerations.append({
                'frame': curr['frame'],
                'acceleration': acceleration,
                'index': i
            })
        
        # Find the acceleration at center frame
        center_accel = None
        for accel in accelerations:
            if accel['frame'] == center_frame:
                center_accel = accel
                break
        
        if center_accel is None:
            return False, 0.0
        
        # Check if acceleration is significantly higher than surrounding
        center_idx_accel = center_accel['index']
        before_accels = [a['acceleration'] for a in accelerations[max(0, center_idx_accel - 2):center_idx_accel]]
        after_accels = [a['acceleration'] for a in accelerations[center_idx_accel + 1:min(len(accelerations), center_idx_accel + 3)]]
        
        if len(before_accels) < 1 or len(after_accels) < 1:
            return False, 0.0
        
        avg_before_accel = np.mean(before_accels)
        avg_after_accel = np.mean(after_accels)
        center_accel_value = center_accel['acceleration']
        
        # Check if center acceleration is significantly higher
        if center_accel_value > avg_before_accel * 1.5 and center_accel_value > avg_after_accel * 1.5:
            confidence = min(1.0, (center_accel_value - max(avg_before_accel, avg_after_accel)) / 10.0)
            return True, confidence
        
        return False, 0.0
    
    def detect_trajectory_pattern_bounce(self, trajectory: List[Dict], center_frame: int) -> Tuple[bool, float]:
        """Detect bounce based on V-shaped trajectory pattern (down then up)"""
        if len(trajectory) < 7:
            return False, 0.0
        
        # Sort by frame number
        sorted_traj = sorted(trajectory, key=lambda x: x['frame'])
        
        # Find center point
        center_idx = None
        for i, point in enumerate(sorted_traj):
            if point['frame'] == center_frame:
                center_idx = i
                break
        
        if center_idx is None or center_idx < 3 or center_idx >= len(sorted_traj) - 3:
            return False, 0.0
        
        # Get points before and after center
        before_points = sorted_traj[max(0, center_idx - 3):center_idx]
        after_points = sorted_traj[center_idx + 1:min(len(sorted_traj), center_idx + 4)]
        
        if len(before_points) < 2 or len(after_points) < 2:
            return False, 0.0
        
        center_point = sorted_traj[center_idx]
        
        # Check if ball is near court level
        court_distance = abs(center_point['y'] - self.court_height)
        if court_distance > 120:  # More lenient for pattern detection
            return False, 0.0
        
        # Calculate trajectory slopes
        before_slopes = []
        for i in range(len(before_points) - 1):
            prev = before_points[i]
            curr = before_points[i + 1]
            if curr['frame'] != prev['frame']:
                slope = (curr['y'] - prev['y']) / (curr['frame'] - prev['frame'])
                before_slopes.append(slope)
        
        after_slopes = []
        for i in range(len(after_points) - 1):
            curr = after_points[i]
            next_ = after_points[i + 1]
            if next_['frame'] != curr['frame']:
                slope = (next_['y'] - curr['y']) / (next_['frame'] - curr['frame'])
                after_slopes.append(slope)
        
        if len(before_slopes) < 1 or len(after_slopes) < 1:
            return False, 0.0
        
        # Check for V-pattern: negative slope before (going down), positive slope after (going up)
        avg_before_slope = np.mean(before_slopes)
        avg_after_slope = np.mean(after_slopes)
        
        # From overhead: Y decreases towards court, so negative slope = going towards court
        going_down_before = avg_before_slope < -1.0  # Ball moving towards court (stricter)
        going_up_after = avg_after_slope > 1.0      # Ball moving away from court (stricter)
        
        if going_down_before and going_up_after:
            # Calculate confidence based on slope magnitude and court distance
            slope_confidence = min(1.0, (abs(avg_before_slope) + abs(avg_after_slope)) / 10.0)
            court_confidence = max(0.0, 1.0 - court_distance / 120.0)
            confidence = (slope_confidence + court_confidence) / 2.0
            return True, confidence
        
        return False, 0.0
    
    def detect_bounce_at_frame(self, frame_idx: int) -> Tuple[bool, float, Dict]:
        """Detect if there's a bounce at the given frame using multiple methods"""
        # Skip if too close to last bounce
        if frame_idx - self.last_bounce_frame < self.min_bounce_gap:
            return False, 0.0, {}
        
        # Get trajectory window
        trajectory = self.get_ball_trajectory_window(frame_idx)
        if len(trajectory) < 5:
            return False, 0.0, {}
        
        # Try all detection methods
        methods = [
            ("velocity", self.detect_velocity_bounce),
            ("height", self.detect_height_bounce),
            ("acceleration", self.detect_acceleration_bounce),
            ("pattern", self.detect_trajectory_pattern_bounce)
        ]
        
        results = {}
        total_confidence = 0.0
        method_count = 0
        
        for method_name, method_func in methods:
            try:
                is_bounce, confidence = method_func(trajectory, frame_idx)
                results[method_name] = {
                    'detected': is_bounce,
                    'confidence': confidence
                }
                
                if is_bounce:
                    total_confidence += confidence
                    method_count += 1
            except Exception as e:
                logger.debug(f"Method {method_name} failed: {e}")
                results[method_name] = {'detected': False, 'confidence': 0.0}
        
        # Require at least 2 methods to agree (more strict)
        if method_count >= 2:
            avg_confidence = total_confidence / method_count
            if avg_confidence >= self.confidence_threshold:
                return True, avg_confidence, results
        
        return False, 0.0, results
    
    def run_viewer(self):
        """Run the interactive bounce detection viewer"""
        print("Robust Bounce Detection Viewer")
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
            
            # Get ball position for current frame (use smoothed for better accuracy)
            ball_pos = self.get_smoothed_ball_position(current_frame)
            
            # Check for bounce at current frame
            is_bounce, bounce_confidence, bounce_info = self.detect_bounce_at_frame(current_frame)
            
            # Draw ball if detected
            if ball_pos is not None:
                x, y, confidence = ball_pos
                
                # Make ball red during bounce, otherwise green/yellow based on confidence
                if is_bounce:
                    color = (0, 0, 255)  # Red for bounce
                    radius = 8  # Larger during bounce
                else:
                    color = (0, 255, 0) if confidence > 0.5 else (0, 255, 255)  # Green/yellow
                    radius = 5
                
                cv2.circle(frame, (int(x), int(y)), radius, color, -1)
                cv2.putText(frame, f"Conf: {confidence:.2f}", (int(x) + 10, int(y) - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            
            if is_bounce:
                # Draw bounce indicator
                cv2.circle(frame, (50, 50), 20, (0, 0, 255), -1)
                cv2.putText(frame, "BOUNCE!", (80, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                cv2.putText(frame, f"Conf: {bounce_confidence:.2f}", (80, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                # Store bounce and location
                if current_frame not in [b['frame'] for b in self.detected_bounces]:
                    bounce_location = None
                    if ball_pos is not None:
                        bounce_location = (ball_pos[0], ball_pos[1])
                        self.bounce_locations.append(bounce_location)
                    
                    self.detected_bounces.append({
                        'frame': current_frame,
                        'confidence': bounce_confidence,
                        'methods': bounce_info,
                        'location': bounce_location
                    })
                    self.last_bounce_frame = current_frame
                    logger.info(f"BOUNCE DETECTED at frame {current_frame} - Confidence: {bounce_confidence:.3f}")
            
            # Draw all previous bounce locations as yellow circles on court
            for i, bounce_loc in enumerate(self.bounce_locations):
                if bounce_loc is not None:
                    # Draw yellow circle at bounce location
                    cv2.circle(frame, (int(bounce_loc[0]), int(bounce_loc[1])), 15, (0, 255, 255), 3)
                    # Add bounce number
                    cv2.putText(frame, f"B{i+1}", (int(bounce_loc[0]) - 10, int(bounce_loc[1]) - 20),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            
            # Draw court level line
            cv2.line(frame, (0, int(self.court_height)), (self.width, int(self.court_height)), (255, 255, 255), 2)
            cv2.putText(frame, "Court Level", (10, int(self.court_height) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            
            # Add info text
            info_text = f"Frame: {current_frame}/{self.total_frames} | Bounces: {len(self.detected_bounces)}"
            cv2.putText(frame, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Add help text
            if show_help:
                help_text = [
                    "Controls: q=quit, p=pause, r=restart, s=step",
                    "f=forward, b=backward, g=goto, h=help"
                ]
                for i, text in enumerate(help_text):
                    cv2.putText(frame, text, (10, 90 + i*20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Write frame to output video
            self.out.write(frame)
            
            # Show frame
            cv2.imshow('Robust Bounce Detection', frame)
            
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
        self.out.release()
        cv2.destroyAllWindows()
        
        # Print summary
        print(f"\n=== BOUNCE DETECTION SUMMARY ===")
        print(f"Total bounces detected: {len(self.detected_bounces)}")
        for i, bounce in enumerate(self.detected_bounces):
            print(f"  Bounce {i+1}: Frame {bounce['frame']}, Confidence {bounce['confidence']:.3f}")
        print(f"Video saved to: {self.output_path}")
        print("Viewer closed")

def main():
    parser = argparse.ArgumentParser(description='Robust tennis ball bounce detection')
    parser.add_argument('--video', required=True, help='Input video file')
    parser.add_argument('--csv', required=True, help='CSV file with ball positions')
    parser.add_argument('--output', help='Output video file (optional)')
    
    args = parser.parse_args()
    
    # Check if files exist
    if not Path(args.video).exists():
        logger.error(f"Video file not found: {args.video}")
        return
    
    if not Path(args.csv).exists():
        logger.error(f"CSV file not found: {args.csv}")
        return
    
    try:
        detector = RobustBounceDetector(args.video, args.csv, args.output)
        detector.run_viewer()
        logger.info("Robust bounce detection completed!")
        
    except Exception as e:
        logger.error(f"Error processing video: {e}")
        raise

if __name__ == "__main__":
    main()
