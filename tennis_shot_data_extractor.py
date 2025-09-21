#!/usr/bin/env python3
"""
Tennis Shot Data Extractor
Extracts features for machine learning-based shot classification
"""

import pandas as pd
import numpy as np
import cv2
import argparse
import logging
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import ast

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ShotDataExtractor:
    def __init__(self):
        self.extracted_data = []
        
    def extract_features_from_csv(self, csv_path: str, video_path: str) -> None:
        """Extract features from CSV data for machine learning"""
        logger.info(f"Loading data from {csv_path}")
        df = pd.read_csv(csv_path)
        
        logger.info(f"Processing {len(df)} frames")
        
        for idx, row in df.iterrows():
            frame_data = self._parse_frame_data(row, idx)
            if frame_data:
                self.extracted_data.append(frame_data)
                
        logger.info(f"Extracted {len(self.extracted_data)} data points")
        
    def _parse_frame_data(self, row: pd.Series, frame_idx: int) -> Optional[Dict]:
        """Parse a single frame of data and extract features"""
        try:
            # Basic frame info
            frame_data = {
                'frame_number': frame_idx,
                'ball_x': row.get('ball_x', 0),
                'ball_y': row.get('ball_y', 0),
                'ball_confidence': row.get('ball_confidence', 0),
            }
            
            # Parse player data
            player_bboxes = self._parse_player_bboxes(row.get('player_bboxes', ''))
            player_poses = self._parse_player_poses(row.get('pose_keypoints', ''))
            
            if len(player_bboxes) >= 2 and len(player_poses) >= 2:
                # Player 0 (near player)
                p0_data = self._extract_player_features(
                    player_bboxes[0], player_poses[0], 0, frame_data
                )
                if p0_data:
                    frame_data.update({f'p0_{k}': v for k, v in p0_data.items()})
                
                # Player 1 (far player)  
                p1_data = self._extract_player_features(
                    player_bboxes[1], player_poses[1], 1, frame_data
                )
                if p1_data:
                    frame_data.update({f'p1_{k}': v for k, v in p1_data.items()})
                    
                return frame_data
                
        except Exception as e:
            logger.warning(f"Error parsing frame {frame_idx}: {e}")
            return None
            
    def _parse_player_bboxes(self, bbox_str: str) -> List[List[float]]:
        """Parse player bounding boxes from CSV string"""
        if not bbox_str or bbox_str == 'nan':
            return []
        try:
            # Format: "x1,y1,x2,y2;x1,y1,x2,y2"
            bbox_parts = bbox_str.split(';')
            bboxes = []
            for part in bbox_parts:
                coords = [float(x) for x in part.split(',')]
                if len(coords) == 4:
                    bboxes.append(coords)
            return bboxes
        except:
            return []
            
    def _parse_player_poses(self, pose_str: str) -> List[List[float]]:
        """Parse player pose keypoints from CSV string"""
        if not pose_str or pose_str == 'nan':
            return []
        try:
            # Format: "x,y,conf|x,y,conf|...;x,y,conf|x,y,conf|..."
            player_poses = []
            players = pose_str.split(';')
            for player in players:
                keypoints = []
                kp_parts = player.split('|')
                for kp in kp_parts:
                    coords = [float(x) for x in kp.split(',')]
                    if len(coords) == 3:
                        keypoints.append(coords)
                player_poses.append(keypoints)
            return player_poses
        except:
            return []
            
    def _extract_player_features(self, bbox: List[float], pose: List[List[float]], 
                                player_id: int, frame_data: Dict) -> Optional[Dict]:
        """Extract features for a single player"""
        try:
            if len(bbox) < 4 or len(pose) < 17:
                return None
                
            x1, y1, x2, y2 = bbox
            features = {}
            
            # Basic bounding box features
            features['bbox_center_x'] = (x1 + x2) / 2
            features['bbox_center_y'] = (y1 + y2) / 2
            features['bbox_width'] = x2 - x1
            features['bbox_height'] = y2 - y1
            features['bbox_area'] = (x2 - x1) * (y2 - y1)
            
            # Feet position (bottom 10% of bbox)
            features['feet_y'] = y2
            features['feet_x'] = (x1 + x2) / 2
            
            # Ball distance
            ball_x = frame_data.get('ball_x', 0)
            ball_y = frame_data.get('ball_y', 0)
            if ball_x > 0 and ball_y > 0:
                features['ball_distance'] = np.sqrt((features['bbox_center_x'] - ball_x)**2 + 
                                                   (features['bbox_center_y'] - ball_y)**2)
            else:
                features['ball_distance'] = float('inf')
                
            # Pose keypoint features
            if len(pose) >= 17:
                # Key points: 5=left_shoulder, 6=right_shoulder, 8=right_elbow, 10=right_wrist
                keypoints = {
                    'left_shoulder': pose[5] if len(pose) > 5 else [0, 0, 0],
                    'right_shoulder': pose[6] if len(pose) > 6 else [0, 0, 0],
                    'right_elbow': pose[8] if len(pose) > 8 else [0, 0, 0],
                    'right_wrist': pose[10] if len(pose) > 10 else [0, 0, 0],
                }
                
                # Extract keypoint coordinates and confidence
                for name, kp in keypoints.items():
                    if len(kp) >= 3:
                        features[f'{name}_x'] = kp[0]
                        features[f'{name}_y'] = kp[1] 
                        features[f'{name}_conf'] = kp[2]
                    else:
                        features[f'{name}_x'] = 0
                        features[f'{name}_y'] = 0
                        features[f'{name}_conf'] = 0
                
                # Calculate body center (midpoint of shoulders)
                if (features['left_shoulder_conf'] > 0.5 and 
                    features['right_shoulder_conf'] > 0.5):
                    features['body_center_x'] = (features['left_shoulder_x'] + 
                                               features['right_shoulder_x']) / 2
                    features['body_center_y'] = (features['left_shoulder_y'] + 
                                               features['right_shoulder_y']) / 2
                else:
                    features['body_center_x'] = features['bbox_center_x']
                    features['body_center_y'] = features['bbox_center_y']
                
                # Arm extension (distance from elbow to wrist)
                if (features['right_elbow_conf'] > 0.5 and 
                    features['right_wrist_conf'] > 0.5):
                    features['arm_extension'] = np.sqrt(
                        (features['right_wrist_x'] - features['right_elbow_x'])**2 +
                        (features['right_wrist_y'] - features['right_elbow_y'])**2
                    )
                else:
                    features['arm_extension'] = 0
                
                # Wrist position relative to body center
                if (features['right_wrist_conf'] > 0.5 and 
                    features['body_center_x'] > 0):
                    features['wrist_relative_x'] = (features['right_wrist_x'] - 
                                                  features['body_center_x'])
                    features['wrist_relative_y'] = (features['right_wrist_y'] - 
                                                  features['body_center_y'])
                    features['wrist_distance_from_center'] = np.sqrt(
                        features['wrist_relative_x']**2 + features['wrist_relative_y']**2
                    )
                else:
                    features['wrist_relative_x'] = 0
                    features['wrist_relative_y'] = 0
                    features['wrist_distance_from_center'] = 0
                
                # Arm angle (angle of arm from elbow to wrist)
                if (features['right_elbow_conf'] > 0.5 and 
                    features['right_wrist_conf'] > 0.5):
                    dx = features['right_wrist_x'] - features['right_elbow_x']
                    dy = features['right_wrist_y'] - features['right_elbow_y']
                    features['arm_angle'] = np.arctan2(dy, dx) * 180 / np.pi
                else:
                    features['arm_angle'] = 0
                    
            return features
            
        except Exception as e:
            logger.warning(f"Error extracting features for player {player_id}: {e}")
            return None
            
    def save_to_csv(self, output_path: str) -> None:
        """Save extracted features to CSV"""
        if not self.extracted_data:
            logger.warning("No data to save")
            return
            
        df = pd.DataFrame(self.extracted_data)
        df.to_csv(output_path, index=False)
        logger.info(f"Saved {len(df)} data points to {output_path}")
        
        # Print summary statistics
        logger.info(f"Data shape: {df.shape}")
        logger.info(f"Columns: {list(df.columns)}")
        
        # Show some sample data
        logger.info("Sample data:")
        print(df.head())

def main():
    parser = argparse.ArgumentParser(description='Extract tennis shot features for ML')
    parser.add_argument('--csv', required=True, help='Input CSV file path')
    parser.add_argument('--video', required=True, help='Input video file path')
    parser.add_argument('--output', default='tennis_shot_features.csv', help='Output CSV file path')
    
    args = parser.parse_args()
    
    # Check if files exist
    if not Path(args.csv).exists():
        logger.error(f"CSV file not found: {args.csv}")
        return
        
    if not Path(args.video).exists():
        logger.error(f"Video file not found: {args.video}")
        return
    
    # Extract features
    extractor = ShotDataExtractor()
    extractor.extract_features_from_csv(args.csv, args.video)
    extractor.save_to_csv(args.output)
    
    logger.info("Feature extraction completed!")

if __name__ == "__main__":
    main()
