#!/usr/bin/env python3
"""
Tennis Analytics Viewer
Reads data from CSV file created by tennis_CV.py and visualizes overlays on black background.
"""

import cv2
import numpy as np
import yaml
import time
import threading
import queue
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import logging
from collections import deque
import csv
import pandas as pd
import sys

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TennisAnalyticsViewer:
    """Analytics viewer that reads CSV data and visualizes overlays"""
    
    def __init__(self, config_path: str = "config.yaml", output_path: str = None):
        """Initialize the analytics viewer"""
        self.config = self._load_config(config_path)
        self.frame_width = 1920
        self.frame_height = 1080
        
        # Video output setup
        self.output_path = output_path
        self.video_writer = None
        
        # Data storage
        self.csv_data = None
        self.current_frame_idx = 0
        self.total_frames = 0
        self.is_playing = True
        self.playback_speed = 1.0  # 1.0 = real-time, 2.0 = 2x speed, etc.
        
        # Initialize analytics data
        self.analytics_data = {
            'total_frames': 0,
            'combined_ball_detections': 0,
            'bounces_detected': 0,
            'shot_analysis': {
                'serve_count': 0,
                'forehand_count': 0,
                'backhand_count': 0,
                'overhand_smash_count': 0,
                'ready_stance_count': 0,
                'moving_count': 0
            },
            'processing_times': []
        }
        
        # Visualization settings
        self.overlay_alpha = 0.8
        self.trail_length = 15
        self.show_trajectories = True
        self.show_analytics = True
        self.show_heatmap = False
        
        # Color scheme (matching CV system)
        self.colors = {
            'ball': (0, 165, 255),      # Orange
            'player': (0, 255, 0),      # Green
            'court': (255, 255, 255),   # White
            'bounce': (0, 0, 255),      # Red
            'trajectory': (255, 255, 0), # Yellow
            'analytics': (255, 255, 255), # White
            'background': (0, 0, 0),     # Black
            'pose_skeleton': (255, 255, 0), # Yellow for pose lines
            'pose_keypoints': (0, 255, 255), # Cyan for pose keypoints
            'court_lines_horizontal': (255, 0, 0), # Blue
            'court_lines_vertical': (0, 255, 0),   # Green
            'court_keypoints_locked': (255, 0, 0), # Blue for locked
            'court_keypoints_best': (0, 255, 255), # Cyan for best
            'court_keypoints_default': (0, 255, 0) # Green for default
        }
        
        # Court line definitions (matching CV system)
        self.horizontal_lines = [
            (0, 4, 6, 1),      # Top endline: 0 → 4 → 6 → 1
            (2, 5, 7, 3),      # Bottom endline: 2 → 5 → 7 → 3
            (8, 12, 9),        # Top service line: 8 → 12 → 9
            (10, 13, 11),      # Bottom service line: 10 → 13 → 11
        ]
        
        self.vertical_lines = [
            (0, 2),             # Left sideline: 0 → 2
            (1, 3),             # Right sideline: 1 → 3
            (5, 10, 8, 4),     # Left doubles alley: 5 → 10 → 8 → 4
            (6, 9, 11, 7),     # Right doubles alley: 6 → 9 → 11 → 7
        ]
        
        # Pose skeleton connections (COCO format)
        self.pose_skeleton = [
            (0, 1), (0, 2), (1, 3), (2, 4),  # Head
            (5, 6), (5, 7), (6, 8), (7, 9),  # Arms
            (5, 11), (6, 12), (11, 12),      # Torso
            (11, 13), (13, 15), (12, 14), (14, 16)  # Legs
        ]
        
        # Initialize display
        self.setup_display()
        
    def _initialize_video_writer(self):
        """Initialize video writer if output path is specified"""
        if not self.output_path:
            return
        
        try:
            # Define codec and create VideoWriter object
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            fps = 30.0  # Match the original video FPS
            frame_size = (self.frame_width, self.frame_height)
            
            self.video_writer = cv2.VideoWriter(
                self.output_path,
                fourcc,
                fps,
                frame_size
            )
            
            if self.video_writer.isOpened():
                logger.info(f"✅ Video writer initialized: {self.output_path}")
            else:
                logger.error(f"❌ Failed to initialize video writer: {self.output_path}")
                self.video_writer = None
                
        except Exception as e:
            logger.error(f"Error initializing video writer: {e}")
            self.video_writer = None
    
    def _release_video_writer(self):
        """Release video writer resources"""
        if self.video_writer:
            self.video_writer.release()
            logger.info(f"✅ Video saved: {self.output_path}")
        
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.error(f"Error loading config: {e}")
            return {}
    
    def setup_display(self):
        """Setup the analytics display window"""
        cv2.namedWindow('Tennis Analytics Viewer', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Tennis Analytics Viewer', self.frame_width, self.frame_height)
        
        # Create initial blank canvas
        self.canvas = np.full((self.frame_height, self.frame_width, 3), 
                             self.colors['background'], dtype=np.uint8)
    
    def load_csv_data(self, csv_path: str = "tennis_analysis_data.csv"):
        """Load data from CSV file"""
        try:
            if not Path(csv_path).exists():
                logger.error(f"CSV file not found: {csv_path}")
                return False
            
            logger.info(f"📊 Loading data from {csv_path}...")
            self.csv_data = pd.read_csv(csv_path)
            self.total_frames = len(self.csv_data)
            logger.info(f"✅ Loaded {self.total_frames} frames of data")
            
            # Calculate analytics
            self._calculate_analytics()
            
            return True
            
        except Exception as e:
            logger.error(f"Error loading CSV data: {e}")
            return False
    
    def _calculate_analytics(self):
        """Calculate analytics from CSV data"""
        if self.csv_data is None:
            return
        
        # Count detections
        self.analytics_data['total_frames'] = len(self.csv_data)
        self.analytics_data['combined_ball_detections'] = len(self.csv_data[self.csv_data['ball_x'].notna()])
        self.analytics_data['bounces_detected'] = len(self.csv_data[self.csv_data['bounce_detected'] == True])
        
        # Calculate shot type statistics
        shot_counts = {
            'serve': 0, 'forehand': 0, 'backhand': 0, 'overhand_smash': 0, 
            'ready_stance': 0, 'moving': 0
        }
        
        for _, row in self.csv_data.iterrows():
            if pd.notna(row['player_shot_types']) and row['player_shot_types'] != '':
                shot_types = row['player_shot_types'].split(';')
                for shot_type in shot_types:
                    if shot_type in shot_counts:
                        shot_counts[shot_type] += 1
        
        self.analytics_data['shot_analysis']['serve_count'] = shot_counts['serve']
        self.analytics_data['shot_analysis']['forehand_count'] = shot_counts['forehand']
        self.analytics_data['shot_analysis']['backhand_count'] = shot_counts['backhand']
        self.analytics_data['shot_analysis']['overhand_smash_count'] = shot_counts['overhand_smash']
        self.analytics_data['shot_analysis']['ready_stance_count'] = shot_counts['ready_stance']
        self.analytics_data['shot_analysis']['moving_count'] = shot_counts['moving']
        
        # Calculate average processing time
        if 'processing_time' in self.csv_data.columns:
            self.analytics_data['processing_times'] = self.csv_data['processing_time'].dropna().tolist()
        
        logger.info(f"📈 Analytics calculated: {self.analytics_data['combined_ball_detections']} ball detections, {self.analytics_data['bounces_detected']} bounces")
        logger.info(f"🎾 Shot analysis: {shot_counts['serve']} serves, {shot_counts['forehand']} forehands, {shot_counts['backhand']} backhands, {shot_counts['overhand_smash']} smashes")
        logger.info(f"🔄 Player states: {shot_counts['ready_stance']} ready stance, {shot_counts['moving']} moving")
    
    def get_frame_data(self, frame_idx: int) -> Optional[Dict[str, Any]]:
        """Get data for a specific frame"""
        if self.csv_data is None or frame_idx >= len(self.csv_data):
            return None
        
        row = self.csv_data.iloc[frame_idx]
        
        # Parse ball data
        ball_data = None
        if pd.notna(row['ball_x']) and pd.notna(row['ball_y']):
            ball_data = {
                'position': [int(row['ball_x']), int(row['ball_y'])],
                'confidence': float(row['ball_confidence']) if pd.notna(row['ball_confidence']) else 0.0,
                'source': row['ball_source'] if pd.notna(row['ball_source']) else 'unknown',
                'speed': float(row['ball_speed']) if pd.notna(row['ball_speed']) else 0.0
            }
        
        # Parse player data
        player_detections = []
        if pd.notna(row['player_bboxes']) and row['player_bboxes'] != '':
            bboxes = row['player_bboxes'].split(';')
            confidences = row['player_confidences'].split(';') if pd.notna(row['player_confidences']) else []
            speeds = row['player_speeds'].split(';') if pd.notna(row['player_speeds']) else []
            shot_types = row['player_shot_types'].split(';') if pd.notna(row['player_shot_types']) else []
            
            for i, bbox_str in enumerate(bboxes):
                if bbox_str and bbox_str != '':
                    try:
                        x1, y1, x2, y2 = map(int, bbox_str.split(','))
                        conf = float(confidences[i]) if i < len(confidences) else 0.0
                        speed = float(speeds[i]) if i < len(speeds) else 0.0
                        shot_type = shot_types[i] if i < len(shot_types) else 'unknown'
                        
                        player_detections.append({
                            'bbox': [x1, y1, x2, y2],
                            'confidence': conf,
                            'speed': speed,
                            'shot_type': shot_type
                        })
                    except:
                        continue
        
        # Parse pose data
        poses = []
        if pd.notna(row['pose_keypoints']) and row['pose_keypoints'] != '':
            pose_strings = row['pose_keypoints'].split(';')
            for pose_str in pose_strings:
                if pose_str and pose_str != '':
                    try:
                        keypoints = []
                        kp_strings = pose_str.split('|')
                        for kp_str in kp_strings:
                            if kp_str and kp_str != '':
                                x, y, conf = map(float, kp_str.split(','))
                                keypoints.append([x, y, conf])
                        poses.append({'keypoints': keypoints})
                    except:
                        continue
        
        # Parse court data
        court_keypoints = []
        if pd.notna(row['court_keypoints']) and row['court_keypoints'] != '':
            kp_strings = row['court_keypoints'].split(';')
            for kp_str in kp_strings:
                if kp_str and kp_str != '' and kp_str != 'None,None':
                    try:
                        x, y = map(float, kp_str.split(','))
                        court_keypoints.append((x, y))
                    except:
                        court_keypoints.append((None, None))
                else:
                    court_keypoints.append((None, None))
        
        return {
            'frame_number': int(row['frame_number']),
            'timestamp': float(row['timestamp']),
            'ball_position': ball_data,
            'player_detections': player_detections,
            'poses': poses,
            'court_keypoints': court_keypoints,
            'bounce_detected': bool(row['bounce_detected']),
            'bounce_confidence': float(row['bounce_confidence']) if pd.notna(row['bounce_confidence']) else 0.0,
            'processing_time': float(row['processing_time']) if pd.notna(row['processing_time']) else 0.0
        }
    
    def draw_court_keypoints(self, frame: np.ndarray, court_keypoints: List[Tuple]) -> np.ndarray:
        """Draw court keypoints exactly like CV system"""
        if not court_keypoints:
            return frame
        
        keypoints_detected = 0
        for j, point in enumerate(court_keypoints):
            if point[0] is not None and point[1] is not None:
                # Default to green for detected keypoints
                color = self.colors['court_keypoints_default']
                thickness = 2
                
                # Draw keypoint (matching CV system)
                cv2.circle(frame, (int(point[0]), int(point[1])),
                          radius=5, color=color, thickness=-1)  # Filled circle
                cv2.circle(frame, (int(point[0]), int(point[1])),
                          radius=8, color=color, thickness=thickness)   # Outline
                
                # Add keypoint number
                cv2.putText(frame, str(j), (int(point[0]) + 10, int(point[1]) - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                
                keypoints_detected += 1
        
        # Draw court lines if we have enough keypoints
        if keypoints_detected >= 4:
            self.draw_court_lines(frame, court_keypoints)
        
        return frame
    
    def draw_court_lines(self, frame: np.ndarray, points: List[Tuple]):
        """Draw court lines exactly like CV system"""
        try:
            # Draw horizontal lines (blue)
            for line_indices in self.horizontal_lines:
                self.draw_continuous_line(frame, points, line_indices, 
                                        self.colors['court_lines_horizontal'], 3, "horizontal")
            
            # Draw vertical lines (green)
            for line_indices in self.vertical_lines:
                self.draw_continuous_line(frame, points, line_indices, 
                                        self.colors['court_lines_vertical'], 3, "vertical")
            
        except Exception as e:
            logger.error(f"Error drawing court lines: {e}")
    
    def draw_continuous_line(self, frame: np.ndarray, points: List[Tuple], line_indices: List[int], 
                           color: Tuple[int, int, int], thickness: int, line_type: str):
        """Draw a continuous line through multiple points (matching CV system)"""
        valid_points = []
        
        # Collect valid points for this line
        for idx in line_indices:
            if (idx < len(points) and 
                points[idx][0] is not None and points[idx][1] is not None):
                valid_points.append((int(points[idx][0]), int(points[idx][1])))
        
        if len(valid_points) < 2:
            return
        
        # Draw line segments connecting all points
        for i in range(len(valid_points) - 1):
            start_point = valid_points[i]
            end_point = valid_points[i + 1]
            cv2.line(frame, start_point, end_point, color, thickness)
    
    def draw_ball_tracking(self, frame: np.ndarray, ball_data: Dict) -> np.ndarray:
        """Draw ball tracking exactly like CV system"""
        if not ball_data:
            return frame
        
        x, y = ball_data['position']
        conf = ball_data.get('confidence', 0.0)
        speed = ball_data.get('speed', 0.0)
        
        # Use single color for ball detection (orange) - matching CV system
        color = self.colors['ball']
        
        # Draw ball position
        cv2.circle(frame, (x, y), 8, color, -1)
        cv2.circle(frame, (x, y), 12, color, 2)
        
        # Draw confidence
        cv2.putText(frame, f"{conf:.2f}", (x + 15, y - 15), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Draw ball speed above ball
        speed_text = f"Speed: {speed:.1f} px/s"
        cv2.putText(frame, speed_text, (x - 30, y - 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)  # Yellow for speed
        
        return frame
    
    def draw_player_detections(self, frame: np.ndarray, player_detections: List[Dict]) -> np.ndarray:
        """Draw player detections exactly like CV system"""
        if not player_detections:
            return frame
        
        for detection in player_detections:
            if 'bbox' in detection:
                x1, y1, x2, y2 = detection['bbox']
                conf = detection.get('confidence', 0.0)
                speed = detection.get('speed', 0.0)
                shot_type = detection.get('shot_type', 'ready_stance')
                
                # Color for players (green) - matching CV system
                color = self.colors['player']
                
                # Draw bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                
                # Draw player label with confidence
                label = f"Player: {conf:.2f}"
                cv2.putText(frame, label, (x1, y1-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                
                # Draw player speed above player
                speed_text = f"Speed: {speed:.1f} px/s"
                cv2.putText(frame, speed_text, (x1, y1-30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)  # Yellow for speed
                
                # Draw shot type below player with appropriate colors
                if shot_type != 'ready_stance':
                    shot_colors = {
                        'forehand': (0, 255, 255),      # Cyan
                        'backhand': (255, 0, 255),      # Magenta
                        'serve': (255, 255, 0),         # Yellow
                        'overhand_smash': (0, 255, 0),  # Green
                        'moving': (128, 128, 128)       # Gray
                    }
                    shot_color = shot_colors.get(shot_type, (255, 255, 255))  # White default
                    shot_text = f"Shot: {shot_type.replace('_', ' ').title()}"
                    cv2.putText(frame, shot_text, (x1, y2+20), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, shot_color, 2)
        
        return frame
    
    def draw_poses(self, frame: np.ndarray, poses: List[Dict]) -> np.ndarray:
        """Draw pose estimation exactly like CV system"""
        if not poses:
            return frame
        
        for pose in poses:
            if 'keypoints' in pose:
                keypoints = pose['keypoints']
                
                # Draw keypoints
                for i, (x, y, conf) in enumerate(keypoints):
                    if conf > 0.3:  # Only draw confident keypoints
                        cv2.circle(frame, (int(x), int(y)), 3, self.colors['pose_keypoints'], -1)
                
                # Draw skeleton
                for start_idx, end_idx in self.pose_skeleton:
                    if (start_idx < len(keypoints) and end_idx < len(keypoints) and 
                        keypoints[start_idx][2] > 0.3 and keypoints[end_idx][2] > 0.3):
                        start_point = (int(keypoints[start_idx][0]), int(keypoints[start_idx][1]))
                        end_point = (int(keypoints[end_idx][0]), int(keypoints[end_idx][1]))
                        cv2.line(frame, start_point, end_point, self.colors['pose_skeleton'], 2)
        
        return frame
    
    def draw_analytics_panel(self, frame: np.ndarray, frame_data: Optional[Dict]) -> np.ndarray:
        """Draw analytics information panel"""
        # Create semi-transparent overlay for analytics
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (500, 450), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # Analytics title
        cv2.putText(frame, "TENNIS ANALYTICS", (20, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, self.colors['analytics'], 2)
        
        # Frame information
        if frame_data:
            cv2.putText(frame, f"Frame: {frame_data.get('frame_number', 0)} / {self.total_frames}", (20, 70),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.colors['analytics'], 2)
        
        # Statistics
        stats = [
            f"Total Frames: {self.analytics_data['total_frames']}",
            f"Ball Detections: {self.analytics_data['combined_ball_detections']}",
            f"Bounces: {self.analytics_data['bounces_detected']}",
            f"Serves: {self.analytics_data['shot_analysis']['serve_count']}",
            f"Forehands: {self.analytics_data['shot_analysis']['forehand_count']}",
            f"Backhands: {self.analytics_data['shot_analysis']['backhand_count']}",
            f"Overhand Smashes: {self.analytics_data['shot_analysis']['overhand_smash_count']}",
            f"Ready Stance: {self.analytics_data['shot_analysis']['ready_stance_count']}",
            f"Moving: {self.analytics_data['shot_analysis']['moving_count']}",
            f"Current Frame: {self.current_frame_idx}",
            f"Playback Speed: {self.playback_speed}x"
        ]
        
        y_offset = 100
        for stat in stats:
            cv2.putText(frame, stat, (20, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.colors['analytics'], 2)
            y_offset += 25
        
        # Performance metrics
        if self.analytics_data['processing_times']:
            avg_time = np.mean(self.analytics_data['processing_times'])
            fps = 1.0 / avg_time if avg_time > 0 else 0
            cv2.putText(frame, f"Avg FPS: {fps:.1f}", (20, y_offset + 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.colors['analytics'], 2)
        
        return frame
    
    def render_frame(self) -> np.ndarray:
        """Render the complete analytics frame with all overlays"""
        # Create fresh black canvas
        frame = np.full((self.frame_height, self.frame_width, 3), 
                       self.colors['background'], dtype=np.uint8)
        
        # Get current frame data
        frame_data = self.get_frame_data(self.current_frame_idx)
        
        # Debug info
        if frame_data:
            debug_info = f"Frame: {frame_data.get('frame_number', 0)} | Ball: {frame_data.get('ball_position') is not None} | Players: {len(frame_data.get('player_detections', []))}"
        else:
            debug_info = f"Frame: {self.current_frame_idx} / {self.total_frames} | No data"
        
        cv2.putText(frame, debug_info, (10, self.frame_height - 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        if frame_data:
            # Draw court keypoints and lines
            frame = self.draw_court_keypoints(frame, frame_data.get('court_keypoints', []))
            
            # Draw ball tracking
            if self.show_trajectories:
                frame = self.draw_ball_tracking(frame, frame_data.get('ball_position'))
            
            # Draw player detections
            frame = self.draw_player_detections(frame, frame_data.get('player_detections', []))
            
            # Draw pose estimation
            frame = self.draw_poses(frame, frame_data.get('poses', []))
            
            # Draw bounce events
            if frame_data.get('bounce_detected'):
                ball_pos = frame_data.get('ball_position')
                if ball_pos:
                    pos = ball_pos['position']
                    cv2.circle(frame, (int(pos[0]), int(pos[1])), 30, self.colors['bounce'], 3)
                    cv2.putText(frame, f"BOUNCE! ({frame_data.get('bounce_confidence', 0.0):.2f})", 
                               (int(pos[0]) - 30, int(pos[1]) - 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.colors['bounce'], 2)
        
        # Draw analytics panel
        if self.show_analytics:
            frame = self.draw_analytics_panel(frame, frame_data)
        
        return frame
    
    def run_viewer(self):
        """Run the analytics viewer"""
        logger.info("🎾 Starting Tennis Analytics Viewer...")
        logger.info("Controls: Press 'q' to quit, 't' to toggle trajectories, 'a' to toggle analytics")
        logger.info("Playback: Press 'space' to pause/resume, 'left/right' arrows to step, 'up/down' arrows for speed")
        
        # Load CSV data
        if not self.load_csv_data():
            logger.error("Failed to load CSV data. Make sure tennis_CV.py has finished running.")
            return
        
        # Initialize video writer if output path is specified
        self._initialize_video_writer()
        
        last_frame_time = time.time()
        frame_interval = 1.0 / 30.0  # 30 FPS default
        
        try:
            while True:
                current_time = time.time()
                
                # Update frame if playing and enough time has passed
                if self.is_playing and current_time - last_frame_time >= frame_interval / self.playback_speed:
                    self.current_frame_idx += 1
                    last_frame_time = current_time
                    
                    # Loop back to beginning
                    if self.current_frame_idx >= self.total_frames:
                        self.current_frame_idx = 0
                
                # Render frame
                frame = self.render_frame()
                
                # Write frame to video if output is enabled
                if self.video_writer and self.video_writer.isOpened():
                    self.video_writer.write(frame)
                
                # Display frame
                cv2.imshow('Tennis Analytics Viewer', frame)
                
                # Handle key presses
                key = cv2.waitKey(30) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('t'):
                    self.show_trajectories = not self.show_trajectories
                    logger.info(f"Trajectories: {'ON' if self.show_trajectories else 'OFF'}")
                elif key == ord('a'):
                    self.show_analytics = not self.show_analytics
                    logger.info(f"Analytics: {'ON' if self.show_analytics else 'OFF'}")
                elif key == ord(' '):  # Spacebar
                    self.is_playing = not self.is_playing
                    logger.info(f"Playback: {'PAUSED' if not self.is_playing else 'PLAYING'}")
                elif key == 83:  # Right arrow
                    self.current_frame_idx = min(self.current_frame_idx + 1, self.total_frames - 1)
                elif key == 81:  # Left arrow
                    self.current_frame_idx = max(self.current_frame_idx - 1, 0)
                elif key == 82:  # Up arrow
                    self.playback_speed = min(self.playback_speed * 1.5, 10.0)
                    logger.info(f"Playback speed: {self.playback_speed:.1f}x")
                elif key == 84:  # Down arrow
                    self.playback_speed = max(self.playback_speed / 1.5, 0.1)
                    logger.info(f"Playback speed: {self.playback_speed:.1f}x")
        
        except KeyboardInterrupt:
            logger.info("Analytics viewer interrupted by user")
        finally:
            # Release video writer
            self._release_video_writer()
            cv2.destroyAllWindows()
            logger.info("Analytics viewer closed")

def main():
    """Main function to run the analytics viewer"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Tennis Analytics Viewer")
    parser.add_argument("--csv", type=str, default="data/processed/csv/tennis_analysis_data.csv", 
                       help="Path to CSV data file")
    parser.add_argument("--width", type=int, default=1280, help="Window width")
    parser.add_argument("--height", type=int, default=720, help="Window height")
    parser.add_argument("--output", "-o", type=str, default=None,
                       help="Output video file path (optional)")
    
    args = parser.parse_args()
    
    # Create and run analytics viewer
    viewer = TennisAnalyticsViewer(output_path=args.output)
    
    if viewer.load_csv_data(args.csv):
        logger.info(f"✅ Loaded CSV data: {args.csv}")
        logger.info("🎮 Starting analytics viewer...")
        logger.info("Controls:")
        logger.info("  - Press 'q' to quit")
        logger.info("  - Press 'space' to pause/resume")
        logger.info("  - Press 'left/right' arrows to step through frames")
        logger.info("  - Press 'up/down' arrows to change playback speed")
        logger.info("  - Press 't' to toggle ball trajectories")
        logger.info("  - Press 'a' to toggle analytics panel")
        
        viewer.run_viewer()
    else:
        logger.error(f"❌ Failed to load CSV data: {args.csv}")
        sys.exit(1)

if __name__ == "__main__":
    main()
