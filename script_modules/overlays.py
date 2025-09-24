#!/usr/bin/env python3
"""
Tennis Analysis Overlay System
Overlays all analysis outputs onto a single video viewer
Designed to be easily extensible - just add new script names to integrate them
"""

import cv2
import numpy as np
import pandas as pd
import argparse
import logging
from typing import List, Dict, Optional, Tuple
import os
import sys

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')
logger = logging.getLogger(__name__)


class OverlayRenderer:
    """Renders overlays from different analysis scripts"""
    
    def __init__(self):
        """Initialize overlay renderer"""
        self.overlay_data = {}
        self.overlay_colors = {
            'ball': (0, 255, 255),      # Yellow
            'court': (255, 0, 255),     # Magenta  
            'positioning': (0, 255, 0), # Green
            'pose': (255, 0, 0),        # Red
            'default': (255, 255, 255)  # White
        }
        
    def load_overlay_data(self, script_name: str, csv_file: str) -> bool:
        """Load overlay data from a CSV file"""
        try:
            if not os.path.exists(csv_file):
                logger.warning(f"CSV file not found: {csv_file}")
                return False
                
            df = pd.read_csv(csv_file)
            self.overlay_data[script_name] = df
            logger.info(f"Loaded {len(df)} records from {script_name}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading {script_name} data: {e}")
            return False
    
    def render_ball_overlay(self, frame: np.ndarray, frame_number: int) -> np.ndarray:
        """Render ball detection overlay"""
        if 'ball' not in self.overlay_data:
            return frame
            
        df = self.overlay_data['ball']
        frame_data = df[df['frame'] == frame_number]
        
        for _, row in frame_data.iterrows():
            if pd.notna(row['x']) and pd.notna(row['y']):
                x, y = int(row['x']), int(row['y'])
                confidence = row.get('confidence', 0.5)
                
                # Draw ball circle with confidence-based size
                radius = max(3, int(confidence * 10))
                cv2.circle(frame, (x, y), radius, self.overlay_colors['ball'], 2)
                
                # Draw confidence text
                cv2.putText(frame, f"Ball: {confidence:.2f}", (x+10, y-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.overlay_colors['ball'], 1)
        
        return frame
    
    def render_court_overlay(self, frame: np.ndarray, frame_number: int) -> np.ndarray:
        """Render court keypoints overlay"""
        if 'court' not in self.overlay_data:
            return frame
            
        df = self.overlay_data['court']
        
        # Use the AVERAGE row (last row) for all frames since court keypoints are static
        average_row = df.iloc[-1]  # Get the last row which should be the AVERAGE
        keypoints_drawn = 0
        
        # Draw court keypoints
        for i in range(15):  # 15 keypoints
            x_col = f'keypoint_{i}_x'
            y_col = f'keypoint_{i}_y'
            
            if x_col in average_row and y_col in average_row and pd.notna(average_row[x_col]) and pd.notna(average_row[y_col]):
                x, y = int(average_row[x_col]), int(average_row[y_col])
                # Make court keypoints much more visible
                cv2.circle(frame, (x, y), 8, (0, 255, 255), -1)  # Bright yellow, larger
                cv2.circle(frame, (x, y), 12, (0, 0, 0), 2)     # Black outline
                cv2.putText(frame, str(i), (x+15, y-15), cv2.FONT_HERSHEY_SIMPLEX, 0.8, 
                           (0, 0, 0), 2)  # Black text, larger
                keypoints_drawn += 1
        
        # Debug output
        if frame_number % 30 == 0:  # Print every 30 frames
            print(f"Frame {frame_number}: Drew {keypoints_drawn} court keypoints from AVERAGE row")
        
        return frame
    
    def render_court_zones_overlay(self, frame: np.ndarray, frame_number: int) -> np.ndarray:
        """Render court zones overlay"""
        try:
            if 'court_zones' not in self.overlay_data:
                return frame
                
            df = self.overlay_data['court_zones']
            
            # Use the AVERAGE row (last row) for all frames since court zones are static
            average_row = df.iloc[-1]  # Get the last row which should be the AVERAGE
            
            # Extract keypoints for zone calculation
            keypoints = []
            for i in range(15):
                x_col = f'keypoint_{i}_x'
                y_col = f'keypoint_{i}_y'
                if x_col in average_row and y_col in average_row and pd.notna(average_row[x_col]) and pd.notna(average_row[y_col]):
                    keypoints.append((int(average_row[x_col]), int(average_row[y_col])))
                else:
                    keypoints.append(None)
            
            # Draw court zones based on tennis court layout
            zones_drawn = 0
            
            # Service boxes - use keypoints 4,5,6,7 (service box corners)
            if all(keypoints[i] for i in [4, 5, 6, 7]):
                self._draw_service_box_zones(frame, keypoints[4], keypoints[5], keypoints[6], keypoints[7])
                zones_drawn += 4
            
            # Baseline zones - use keypoints 8,9,10,11 (baseline corners)  
            if all(keypoints[i] for i in [8, 9, 10, 11]):
                self._draw_baseline_zones(frame, keypoints[8], keypoints[9], keypoints[10], keypoints[11])
                zones_drawn += 3
            
            # Doubles lanes - use keypoints 2,3 (doubles lane corners)
            if all(keypoints[i] for i in [2, 3]):
                self._draw_doubles_lanes(frame, keypoints[2], keypoints[3])
                zones_drawn += 2
                
        except Exception as e:
            if frame_number == 0:
                print(f"DEBUG: Error in render_court_zones_overlay: {e}")
            return frame
        
        return frame
    
    def _draw_service_box_zones(self, frame, p1, p2, p3, p4):
        """Draw service box zones (A, B, C, D)"""
        # Create 4 vertical zones
        zone_colors = [(255, 100, 100), (100, 255, 100), (100, 100, 255), (255, 255, 100)]
        zone_names = ['A', 'B', 'C', 'D']
        
        # Calculate zone boundaries
        x_coords = [p1[0], p2[0], p3[0], p4[0]]
        y_coords = [p1[1], p2[1], p3[1], p4[1]]
        
        min_x, max_x = min(x_coords), max(x_coords)
        min_y, max_y = min(y_coords), max(y_coords)
        
        # Draw 4 vertical zones
        zone_width = (max_x - min_x) // 4
        for i in range(4):
            x1 = min_x + i * zone_width
            x2 = min_x + (i + 1) * zone_width
            
            # Draw zone rectangle
            cv2.rectangle(frame, (x1, min_y), (x2, max_y), zone_colors[i], 2)
            
            # Draw zone label
            center_x = (x1 + x2) // 2
            center_y = (min_y + max_y) // 2
            cv2.putText(frame, zone_names[i], (center_x - 10, center_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, zone_colors[i], 2)
    
    def _draw_baseline_zones(self, frame, p1, p2, p3, p4):
        """Draw baseline zones (WIDE, BODY, TEE)"""
        zone_colors = [(255, 100, 255), (100, 255, 255), (255, 200, 100)]
        zone_names = ['WIDE', 'BODY', 'TEE']
        
        # Calculate zone boundaries
        x_coords = [p1[0], p2[0], p3[0], p4[0]]
        y_coords = [p1[1], p2[1], p3[1], p4[1]]
        
        min_x, max_x = min(x_coords), max(x_coords)
        min_y, max_y = min(y_coords), max(y_coords)
        
        # Draw 3 vertical zones
        zone_width = (max_x - min_x) // 3
        for i in range(3):
            x1 = min_x + i * zone_width
            x2 = min_x + (i + 1) * zone_width
            
            # Draw zone rectangle
            cv2.rectangle(frame, (x1, min_y), (x2, max_y), zone_colors[i], 2)
            
            # Draw zone label
            center_x = (x1 + x2) // 2
            center_y = (min_y + max_y) // 2
            cv2.putText(frame, zone_names[i], (center_x - 20, center_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, zone_colors[i], 2)
    
    def _draw_doubles_lanes(self, frame, p1, p2):
        """Draw doubles lanes (AA, DD)"""
        # Draw left doubles lane (AA) - use p1 as reference
        if p1:
            # Create a rectangle for the left doubles lane
            lane_width = 50
            lane_height = 200
            x1 = p1[0] - lane_width
            y1 = p1[1] - lane_height // 2
            x2 = p1[0]
            y2 = p1[1] + lane_height // 2
            cv2.rectangle(frame, (x1, y1), (x2, y2), (200, 200, 200), 3)
            cv2.putText(frame, 'AA', (x1 + 5, y1 + 25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)
        
        # Draw right doubles lane (DD) - use p2 as reference
        if p2:
            # Create a rectangle for the right doubles lane
            lane_width = 50
            lane_height = 200
            x1 = p2[0]
            y1 = p2[1] - lane_height // 2
            x2 = p2[0] + lane_width
            y2 = p2[1] + lane_height // 2
            cv2.rectangle(frame, (x1, y1), (x2, y2), (150, 150, 150), 3)
            cv2.putText(frame, 'DD', (x1 + 5, y1 + 25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (150, 150, 150), 2)
    
    def render_positioning_overlay(self, frame: np.ndarray, frame_number: int) -> np.ndarray:
        """Render player positioning overlay"""
        if 'positioning' not in self.overlay_data:
            return frame
            
        df = self.overlay_data['positioning']
        frame_data = df[df['frame'] == frame_number]
        
        for _, row in frame_data.iterrows():
            player_id = row['player_id']
            position = row['position']
            feet_x = int(row['feet_x'])
            feet_y = int(row['feet_y'])
            
            # Color based on position
            color = self.overlay_colors['positioning']
            if position == 'FRONT':
                color = (0, 255, 0)  # Green
            elif position == 'BACK':
                color = (0, 0, 255)  # Red
            elif position == 'DOUBLES':
                color = (255, 0, 255)  # Magenta
            else:
                color = (128, 128, 128)  # Gray
            
            # Draw player feet position
            cv2.circle(frame, (feet_x, feet_y), 8, color, -1)
            
            # Draw position label
            label = f"P{player_id}: {position}"
            cv2.putText(frame, label, (feet_x+10, feet_y-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        return frame
    
    def render_pose_overlay(self, frame: np.ndarray, frame_number: int) -> np.ndarray:
        """Render pose keypoints overlay"""
        if 'pose' not in self.overlay_data:
            return frame
            
        df = self.overlay_data['pose']
        frame_data = df[df['frame'] == frame_number]
        
        for _, row in frame_data.iterrows():
            player_id = row['player_id']
            keypoints_str = row['keypoints']
            
            if pd.notna(keypoints_str):
                # Parse keypoints string
                keypoints = []
                for kp_str in keypoints_str.split('|'):
                    if ',' in kp_str:
                        x, y = kp_str.split(',')
                        keypoints.append((int(float(x)), int(float(y))))
                
                # Draw keypoints
                for i, (x, y) in enumerate(keypoints):
                    cv2.circle(frame, (x, y), 3, self.overlay_colors['pose'], -1)
                    cv2.putText(frame, str(i), (x+3, y-3), cv2.FONT_HERSHEY_SIMPLEX, 0.3, 
                               self.overlay_colors['pose'], 1)
                
                # Draw skeleton connections (basic)
                if len(keypoints) >= 17:  # COCO format
                    # Head connections
                    if keypoints[0][0] > 0 and keypoints[1][0] > 0:
                        cv2.line(frame, keypoints[0], keypoints[1], self.overlay_colors['pose'], 2)
                    
                    # Shoulder connections
                    if keypoints[5][0] > 0 and keypoints[6][0] > 0:
                        cv2.line(frame, keypoints[5], keypoints[6], self.overlay_colors['pose'], 2)
        
        return frame


class TennisOverlayProcessor:
    """Main processor for tennis analysis overlays"""
    
    def __init__(self):
        """Initialize processor"""
        self.renderer = OverlayRenderer()
        
        # Define available scripts and their CSV files
        # To add a new script, just add it here!
        self.available_scripts = {
            'ball': 'ball_detection.csv',
            'court': 'court_keypoints.csv', 
            'court_zones': 'court_keypoints.csv',  # Uses same CSV as court keypoints
            'positioning': 'player_positioning.csv',
            'pose': 'player_poses.csv'
        }
        
        # Script-specific render functions
        self.render_functions = {
            'ball': self.renderer.render_ball_overlay,
            'court': self.renderer.render_court_overlay,
            'court_zones': self.renderer.render_court_zones_overlay,
            'positioning': self.renderer.render_positioning_overlay,
            'pose': self.renderer.render_pose_overlay
        }
    
    def load_script_data(self, script_name: str) -> bool:
        """Load data for a specific script"""
        if script_name not in self.available_scripts:
            logger.error(f"Unknown script: {script_name}. Available: {list(self.available_scripts.keys())}")
            return False
        
        csv_file = self.available_scripts[script_name]
        return self.renderer.load_overlay_data(script_name, csv_file)
    
    def add_new_script(self, script_name: str, csv_file: str, render_function):
        """Add a new script to the overlay system"""
        self.available_scripts[script_name] = csv_file
        self.render_functions[script_name] = render_function
        logger.info(f"Added new script: {script_name}")
    
    def process_video(self, video_file: str, output_file: str = None, 
                     active_scripts: List[str] = None, show_viewer: bool = True):
        """Process video with overlays from specified scripts"""
        
        # Default to all scripts if none specified
        if active_scripts is None:
            active_scripts = list(self.available_scripts.keys())
        
        # Load data for active scripts
        loaded_scripts = []
        for script_name in active_scripts:
            if self.load_script_data(script_name):
                loaded_scripts.append(script_name)
        
        logger.info(f"Loaded overlays for: {loaded_scripts}")
        
        # Open video
        cap = cv2.VideoCapture(video_file)
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_file}")
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        logger.info(f"Processing video: {width}x{height} @ {fps}fps, {total_frames} frames")
        
        # Setup video writer if output specified
        out = None
        if output_file:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_file, fourcc, fps, (width, height))
        
        # Setup viewer
        if show_viewer:
            cv2.namedWindow('Tennis Analysis Overlays', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('Tennis Analysis Overlays', 1200, 800)
        
        frame_number = 0
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Apply overlays from each active script
                for script_name in loaded_scripts:
                    if script_name in self.render_functions:
                        frame = self.render_functions[script_name](frame, frame_number)
                
                # Add frame info
                cv2.putText(frame, f"Frame: {frame_number}", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                
                # Add script info
                script_info = f"Scripts: {', '.join(loaded_scripts)}"
                cv2.putText(frame, script_info, (10, 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                # Write frame
                if out:
                    out.write(frame)
                
                # Show in viewer
                if show_viewer:
                    cv2.imshow('Tennis Analysis Overlays', frame)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        break
                    elif key == ord(' '):  # Space to pause
                        cv2.waitKey(0)
                
                # Progress update
                if frame_number % 30 == 0:
                    logger.info(f"Processed {frame_number}/{total_frames} frames ({frame_number/total_frames*100:.1f}%)")
                
                frame_number += 1
        
        finally:
            # Cleanup
            cap.release()
            if out:
                out.release()
            if show_viewer:
                cv2.destroyAllWindows()
        
        logger.info("Overlay processing completed!")


def main():
    """Main function for standalone usage"""
    parser = argparse.ArgumentParser(description='Tennis Analysis Overlay System')
    parser.add_argument('--video', required=True, help='Input video file')
    parser.add_argument('--output', default='tennis_overlays.mp4', help='Output video file')
    parser.add_argument('--scripts', nargs='+', 
                       choices=['ball', 'court', 'court_zones', 'positioning', 'pose'],
                       default=['ball', 'court', 'court_zones', 'positioning', 'pose'],
                       help='Scripts to include in overlay (default: all)')
    parser.add_argument('--no-viewer', action='store_true', help='Disable real-time viewer')
    
    args = parser.parse_args()
    
    try:
        processor = TennisOverlayProcessor()
        processor.process_video(
            args.video, 
            args.output, 
            args.scripts, 
            show_viewer=not args.no_viewer
        )
        
    except Exception as e:
        logger.error(f"Error running overlay system: {e}")


if __name__ == "__main__":
    main()
