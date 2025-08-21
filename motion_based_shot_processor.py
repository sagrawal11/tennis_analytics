#!/usr/bin/env python3
"""
Motion-Based Shot Classification Processor

This script processes tennis analysis data from CSV and applies motion-based
shot classification to create enhanced analysis videos.
"""

import pandas as pd
import cv2
import numpy as np
import argparse
import logging
import ast
from typing import List, Dict, Optional, Tuple
from motion_based_shot_classifier import MotionBasedShotClassifier

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

class MotionBasedShotProcessor:
    def __init__(self):
        """Initialize the motion-based shot processor"""
        self.shot_classifier = MotionBasedShotClassifier()
        self.enhanced_shot_types = []
        
    def _parse_ball_position_from_csv(self, ball_x: str, ball_y: str) -> Optional[List[int]]:
        """Parse ball position from CSV columns"""
        try:
            if pd.isna(ball_x) or pd.isna(ball_y) or ball_x == '' or ball_y == '':
                return None
            
            x = int(float(ball_x))
            y = int(float(ball_y))
            return [x, y]
        except (ValueError, TypeError):
            return None
    
    def _parse_court_keypoints_from_csv(self, court_str: str) -> List[Tuple]:
        """Parse court keypoints from CSV string format"""
        try:
            if pd.isna(court_str) or court_str == '':
                return []
            
            # Parse format like "x1,y1;x2,y2;..."
            keypoints = []
            pairs = court_str.split(';')
            for pair in pairs:
                if pair.strip():
                    x, y = pair.split(',')
                    keypoints.append((int(float(x)), int(float(y))))
            return keypoints
        except (ValueError, AttributeError):
            return []
    
    def _parse_player_bboxes_from_csv(self, bbox_str: str) -> List[List[int]]:
        """Parse player bounding boxes from CSV string format"""
        try:
            if pd.isna(bbox_str) or bbox_str == '':
                return []
            
            # Parse format like "x1,y1,x2,y2;x1,y1,x2,y2;..."
            bboxes = []
            boxes = bbox_str.split(';')
            for box in boxes:
                if box.strip():
                    coords = box.split(',')
                    bbox = [int(float(coords[0])), int(float(coords[1])), 
                           int(float(coords[2])), int(float(coords[3]))]
                    bboxes.append(bbox)
            return bboxes
        except (ValueError, AttributeError):
            return []
    
    def _parse_poses_from_csv(self, pose_str: str) -> List[Dict[int, List]]:
        """Parse pose keypoints from CSV string format"""
        try:
            if pd.isna(pose_str) or pose_str == '':
                return []
            
            # Parse format like "x,y,confidence|x,y,confidence;x,y,confidence|x,y,confidence;..."
            poses = []
            players = pose_str.split(';')
            
            for player_pose in players:
                if not player_pose.strip():
                    continue
                
                keypoints = {}
                points = player_pose.split('|')
                
                for i, point in enumerate(points):
                    if point.strip():
                        try:
                            x, y, conf = point.split(',')
                            keypoints[i] = [int(float(x)), int(float(y)), float(conf)]
                        except (ValueError, IndexError):
                            continue
                
                if keypoints:
                    poses.append(keypoints)
            
            return poses
        except (ValueError, AttributeError):
            return []
    
    def process_csv_data(self, csv_file: str, video_file: str, output_file: str):
        """Process CSV data and create enhanced analysis video"""
        try:
            # Load CSV data
            logger.info(f"Loading CSV data from {csv_file}")
            df = pd.read_csv(csv_file)
            logger.info(f"Loaded {len(df)} frames of data")
            
            # Load video
            cap = cv2.VideoCapture(video_file)
            if not cap.isOpened():
                raise ValueError(f"Could not open video file: {video_file}")
            
            # Get video properties
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            logger.info(f"Video: {width}x{height} @ {fps}fps, {total_frames} frames")
            logger.info(f"Output video: {output_file}")
            
            # Setup video writer
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_file, fourcc, fps, (width, height))
            
            # Process each frame
            for frame_count, (_, row) in enumerate(df.iterrows()):
                # Read frame
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Parse data from CSV row
                poses = self._parse_poses_from_csv(row.get('pose_keypoints', ''))
                ball_position = self._parse_ball_position_from_csv(row.get('ball_x', ''), row.get('ball_y', ''))
                court_keypoints = self._parse_court_keypoints_from_csv(row.get('court_keypoints', ''))
                player_bboxes = self._parse_player_bboxes_from_csv(row.get('player_bboxes', ''))
                
                # Apply motion-based shot classification
                enhanced_shot_types = []
                for i, bbox in enumerate(player_bboxes):
                    # Classify shot for each player
                    shot_type = self.shot_classifier.classify_shot(
                        bbox, ball_position, poses, court_keypoints, frame_count, player_id=i
                    )
                    enhanced_shot_types.append(shot_type)
                
                # Add overlays to frame
                frame_with_overlays = self._add_overlays(
                    frame, frame_count, player_bboxes, enhanced_shot_types, 
                    ball_position, court_keypoints, poses
                )
                
                # Write frame
                out.write(frame_with_overlays)
                
                # Progress update
                if frame_count % 30 == 0:
                    logger.info(f"Processed {frame_count}/{len(df)} frames ({frame_count/len(df)*100:.1f}%)")
            
            # Cleanup
            cap.release()
            out.release()
            
            # Print final statistics
            self._print_final_statistics()
            logger.info(f"Motion-based analysis video created: {output_file}")
            
        except Exception as e:
            logger.error(f"Error processing CSV data: {e}")
            raise
    
    def _add_overlays(self, frame: np.ndarray, frame_count: int, 
                     player_bboxes: List[List[int]], shot_types: List[str],
                     ball_position: Optional[List[int]], court_keypoints: List[Tuple],
                     poses: List[Dict[int, List]]) -> np.ndarray:
        """Add overlays to the frame"""
        # Copy frame to avoid modifying original
        frame_with_overlays = frame.copy()
        
        # Add frame number
        cv2.putText(frame_with_overlays, f"Frame: {frame_count}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Add player bounding boxes and shot types
        for i, (bbox, shot_type) in enumerate(zip(player_bboxes, shot_types)):
            if bbox:
                x1, y1, x2, y2 = bbox
                
                # Draw bounding box
                color = (0, 255, 0) if i == 0 else (255, 0, 0)  # Green for player 0, red for player 1
                cv2.rectangle(frame_with_overlays, (x1, y1), (x2, y2), color, 2)
                
                # Add player ID and shot type
                label = f"Player {i}: {shot_type}"
                cv2.putText(frame_with_overlays, label, (x1, y1-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        # Add ball position
        if ball_position:
            ball_x, ball_y = ball_position
            cv2.circle(frame_with_overlays, (ball_x, ball_y), 5, (0, 255, 255), -1)
            cv2.putText(frame_with_overlays, "Ball", (ball_x+10, ball_y-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        
        # Add court keypoints
        for i, (x, y) in enumerate(court_keypoints):
            cv2.circle(frame_with_overlays, (x, y), 3, (255, 255, 0), -1)
            if i < 4:  # Label first 4 keypoints
                cv2.putText(frame_with_overlays, f"C{i}", (x+5, y-5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
        
        # Add pose keypoints (skeleton)
        for player_id, pose in enumerate(poses):
            if player_id >= len(player_bboxes):
                continue
                
            color = (0, 255, 0) if player_id == 0 else (255, 0, 0)
            
            # Draw key skeleton connections
            connections = [
                (5, 6),   # shoulders
                (6, 8),   # right shoulder to right elbow
                (8, 10),  # right elbow to right wrist
                (5, 7),   # left shoulder to left elbow
                (7, 9),   # left elbow to left wrist
                (6, 12),  # right shoulder to right hip
                (5, 11),  # left shoulder to left hip
                (12, 14), # right hip to right knee
                (14, 16), # right knee to right ankle
                (11, 13), # left hip to left knee
                (13, 15)  # left knee to left ankle
            ]
            
            for connection in connections:
                if connection[0] in pose and connection[1] in pose:
                    pt1 = (int(pose[connection[0]][0]), int(pose[connection[0]][1]))
                    pt2 = (int(pose[connection[1]][0]), int(pose[connection[1]][1]))
                    cv2.line(frame_with_overlays, pt1, pt2, color, 2)
            
            # Draw keypoints
            for keypoint_id, keypoint in pose.items():
                if keypoint_id in [5, 6, 7, 8, 9, 10]:  # Only show arm keypoints
                    x, y = int(keypoint[0]), int(keypoint[1])
                    cv2.circle(frame_with_overlays, (x, y), 4, color, -1)
        
        return frame_with_overlays
    
    def _print_final_statistics(self):
        """Print final shot classification statistics"""
        logger.info("Final Motion-Based Shot Statistics:")
        
        # Count shot types for each player
        for player_id in range(2):  # Assuming 2 players
            player_shots = []
            for frame_data in self.shot_classifier.player_motion_history.get(player_id, []):
                # This would need to be tracked during processing
                pass
            
            # For now, just show that processing completed
            logger.info(f"  Player {player_id}: Motion analysis completed")

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Motion-based tennis shot classification')
    parser.add_argument('--csv', required=True, help='Input CSV file with analysis data')
    parser.add_argument('--video', required=True, help='Input video file')
    parser.add_argument('--output', required=True, help='Output video file')
    
    args = parser.parse_args()
    
    # Create processor and run analysis
    processor = MotionBasedShotProcessor()
    processor.process_csv_data(args.csv, args.video, args.output)

if __name__ == "__main__":
    main()
