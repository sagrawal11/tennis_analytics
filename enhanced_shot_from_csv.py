#!/usr/bin/env python3
"""
Enhanced Shot Classification from CSV Data
Reads pose data from tennis_CV.py CSV output and applies enhanced shot classification
"""

import cv2
import numpy as np
import pandas as pd
import yaml
import logging
import argparse
import sys
import os
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import time
import ast

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from enhanced_shot_classifier import EnhancedShotClassifier

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EnhancedShotFromCSV:
    """Enhanced shot classification using CSV pose data"""
    
    def __init__(self, config_path: str = "config.yaml"):
        """Initialize the classifier"""
        self.config_path = config_path
        self.config = self._load_config()
        
        # Initialize shot classifier
        self.shot_classifier = EnhancedShotClassifier()
        
        # Shot type colors
        self.shot_colors = {
            'serve': (0, 255, 0),        # Green
            'forehand': (255, 0, 0),     # Blue
            'backhand': (0, 0, 255),     # Red
            'overhead_smash': (255, 255, 0), # Cyan
            'volley': (255, 0, 255),     # Magenta
            'ready_stance': (128, 128, 128), # Gray
            'moving': (0, 255, 255),     # Yellow
            'unknown': (255, 255, 255)   # White
        }
        
        logger.info("Enhanced Shot from CSV initialized")
    
    def _load_config(self) -> Dict:
        """Load configuration from YAML file"""
        try:
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
            logger.info(f"Configuration loaded from {self.config_path}")
            return config
        except Exception as e:
            logger.error(f"Error loading config: {e}")
            return {}
    
    def _parse_poses_from_csv(self, poses_str: str) -> List[Dict]:
        """Parse poses string from CSV into pose dictionaries"""
        try:
            if pd.isna(poses_str) or poses_str == '':
                return []
            
            parsed_poses = []
            # Split by semicolon for different people
            person_poses = poses_str.split(';')
            
            for person_pose in person_poses:
                if not person_pose.strip():
                    continue
                
                # Split keypoints by pipe
                keypoint_strs = person_pose.split('|')
                keypoints = []
                confidence = []
                
                for kp_str in keypoint_strs:
                    if not kp_str.strip():
                        continue
                    try:
                        parts = kp_str.split(',')
                        if len(parts) >= 3:
                            x, y, conf = map(float, parts[:3])
                            keypoints.append([x, y])
                            confidence.append(conf)
                    except (ValueError, IndexError):
                        keypoints.append([0.0, 0.0])
                        confidence.append(0.0)
                
                if keypoints:
                    parsed_poses.append({
                        'keypoints': keypoints,
                        'confidence': confidence
                    })
            
            return parsed_poses
            
        except Exception as e:
            logger.debug(f"Error parsing poses: {e}")
            return []
    
    def _parse_ball_position_from_csv(self, ball_x: str, ball_y: str) -> Optional[List[int]]:
        """Parse ball position from CSV"""
        try:
            if pd.isna(ball_x) or pd.isna(ball_y) or ball_x == '' or ball_y == '':
                return None
            
            # Parse ball position from separate x, y columns
            x = int(float(ball_x))
            y = int(float(ball_y))
            return [x, y]
            
        except Exception as e:
            logger.debug(f"Error parsing ball position: {e}")
            return None
    
    def _parse_court_keypoints_from_csv(self, court_str: str) -> List[Tuple]:
        """Parse court keypoints from CSV"""
        try:
            if pd.isna(court_str) or court_str == '':
                return []
            
            # Parse court keypoints format: "x1,y1;x2,y2;x3,y3;..."
            court_kps = []
            pairs = court_str.split(';')
            for pair in pairs:
                if pair.strip():
                    try:
                        x, y = map(float, pair.split(','))
                        court_kps.append((int(x), int(y)))
                    except (ValueError, IndexError):
                        continue
            
            return court_kps
            
        except Exception as e:
            logger.debug(f"Error parsing court keypoints: {e}")
            return []
    
    def _parse_player_bboxes_from_csv(self, bboxes_str: str) -> List[List[int]]:
        """Parse player bounding boxes from CSV"""
        try:
            if pd.isna(bboxes_str) or bboxes_str == '':
                return []
            
            # Parse bounding boxes format: "x1,y1,x2,y2;x1,y1,x2,y2;..."
            bboxes = []
            boxes = bboxes_str.split(';')
            for box in boxes:
                if box.strip():
                    try:
                        x1, y1, x2, y2 = map(int, box.split(','))
                        bboxes.append([x1, y1, x2, y2])
                    except (ValueError, IndexError):
                        continue
            
            return bboxes
            
        except Exception as e:
            logger.debug(f"Error parsing player bboxes: {e}")
            return []
    
    def process_csv_data(self, csv_path: str, video_path: str, output_path: str = None):
        """Process CSV data and create enhanced shot classification video"""
        try:
            # Load CSV data
            logger.info(f"Loading CSV data from {csv_path}")
            df = pd.read_csv(csv_path)
            logger.info(f"Loaded {len(df)} frames of data")
            
            # Open video
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                logger.error(f"Could not open video: {video_path}")
                return
            
            # Get video properties
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            logger.info(f"Video: {width}x{height} @ {fps}fps, {total_frames} frames")
            
            # Setup output video
            output_writer = None
            if output_path:
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                output_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
                logger.info(f"Output video: {output_path}")
            
            # Create window
            cv2.namedWindow("Enhanced Shot Classification", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Enhanced Shot Classification", 1280, 720)
            
            frame_count = 0
            start_time = time.time()
            
            # Process each frame
            for index, row in df.iterrows():
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_count += 1
                
                # Parse data from CSV row
                poses = self._parse_poses_from_csv(row.get('pose_keypoints', ''))
                ball_position = self._parse_ball_position_from_csv(row.get('ball_x', ''), row.get('ball_y', ''))
                court_keypoints = self._parse_court_keypoints_from_csv(row.get('court_keypoints', ''))
                player_bboxes = self._parse_player_bboxes_from_csv(row.get('player_bboxes', ''))
                
                # Apply enhanced shot classification
                enhanced_shot_types = []
                for i, bbox in enumerate(player_bboxes):
                    # Calculate player speed (simplified)
                    player_speed = 0.0  # Could be enhanced with position tracking
                    
                    # Classify shot
                    shot_type = self.shot_classifier.classify_shot(
                        bbox, ball_position, poses, court_keypoints, player_speed, frame_count
                    )
                    enhanced_shot_types.append(shot_type)
                
                # Draw enhanced classifications on frame
                self._draw_enhanced_classifications(
                    frame, player_bboxes, enhanced_shot_types, 
                    ball_position, court_keypoints, poses
                )
                
                # Add frame counter
                cv2.putText(frame, f"Frame: {frame_count}/{total_frames}", (10, height - 20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                # Show frame
                cv2.imshow("Enhanced Shot Classification", frame)
                
                # Write to output video
                if output_writer:
                    output_writer.write(frame)
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord(' '):
                    # Pause/unpause
                    cv2.waitKey(0)
                
                # Progress update
                if frame_count % 30 == 0:
                    elapsed = time.time() - start_time
                    fps_processed = frame_count / elapsed
                    logger.info(f"Processed {frame_count}/{total_frames} frames ({fps_processed:.1f} fps)")
            
            # Cleanup
            cap.release()
            if output_writer:
                output_writer.release()
            cv2.destroyAllWindows()
            
            # Print final statistics
            final_stats = self.shot_classifier.get_shot_statistics()
            logger.info("Final Enhanced Shot Statistics:")
            for shot_type, count in final_stats.items():
                if count > 0:
                    logger.info(f"  {shot_type}: {count}")
            
            logger.info("Enhanced shot classification completed!")
            
        except Exception as e:
            logger.error(f"Error processing CSV data: {e}")
    
    def _draw_enhanced_classifications(self, frame: np.ndarray, player_bboxes: List[List[int]], 
                                     enhanced_shot_types: List[str], ball_position: Optional[List[int]], 
                                     court_keypoints: List[Tuple], poses: List[Dict]):
        """Draw enhanced shot classifications on frame"""
        try:
            # Draw player bounding boxes and shot types
            for i, (bbox, shot_type) in enumerate(zip(player_bboxes, enhanced_shot_types)):
                x1, y1, x2, y2 = bbox
                
                # Get shot color
                shot_color = self.shot_colors.get(shot_type, (255, 255, 255))
                
                # Draw player bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), shot_color, 2)
                
                # Draw enhanced shot type text below player
                shot_text = f"Enhanced: {shot_type.replace('_', ' ').title()}"
                text_size = cv2.getTextSize(shot_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                text_x = x1
                text_y = y2 + text_size[1] + 10
                
                # Draw text background
                cv2.rectangle(frame, (text_x - 5, text_y - text_size[1] - 5), 
                             (text_x + text_size[0] + 5, text_y + 5), (0, 0, 0), -1)
                
                # Draw text
                cv2.putText(frame, shot_text, (text_x, text_y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, shot_color, 2)
                
                # Draw player number
                player_text = f"Player {i+1}"
                player_y = text_y + 25
                cv2.putText(frame, player_text, (text_x, player_y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            
            # Draw ball position
            if ball_position:
                ball_x, ball_y = ball_position
                cv2.circle(frame, (ball_x, ball_y), 5, (0, 255, 255), -1)
                cv2.putText(frame, "BALL", (ball_x + 10, ball_y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
            
            # Draw court keypoints
            for kp in court_keypoints:
                cv2.circle(frame, kp, 3, (255, 255, 255), -1)
            
            # Draw pose keypoints
            for pose in poses:
                if 'keypoints' in pose:
                    keypoints = pose['keypoints']
                    confidence = pose.get('confidence', [])
                    
                    for i, kp in enumerate(keypoints):
                        if i < len(confidence) and confidence[i] > 0.3:
                            x, y = int(kp[0]), int(kp[1])
                            cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)
            
            # Draw shot statistics
            stats = self.shot_classifier.get_shot_statistics()
            y_offset = 30
            for shot_type, count in stats.items():
                if count > 0:
                    color = self.shot_colors.get(shot_type, (255, 255, 255))
                    text = f"Enhanced {shot_type}: {count}"
                    cv2.putText(frame, text, (10, y_offset), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                    y_offset += 20
                    
        except Exception as e:
            logger.error(f"Error drawing enhanced classifications: {e}")

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Enhanced Shot Classification from CSV")
    parser.add_argument("--csv", required=True, help="CSV file with pose data")
    parser.add_argument("--video", required=True, help="Input video path")
    parser.add_argument("--config", default="config.yaml", help="Configuration file")
    parser.add_argument("--output", help="Output video path")
    
    args = parser.parse_args()
    
    # Check if files exist
    if not os.path.exists(args.csv):
        logger.error(f"CSV file not found: {args.csv}")
        return
    
    if not os.path.exists(args.video):
        logger.error(f"Video file not found: {args.video}")
        return
    
    # Create processor
    processor = EnhancedShotFromCSV(args.config)
    
    # Process data
    processor.process_csv_data(args.csv, args.video, args.output)

if __name__ == "__main__":
    main()
