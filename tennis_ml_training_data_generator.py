#!/usr/bin/env python3
"""
Tennis ML Training Data Generator
Processes multiple videos with manual annotations to create a comprehensive training dataset
"""

import pandas as pd
import numpy as np
import cv2
import argparse
import logging
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import ast
from dataclasses import dataclass
from enum import Enum

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ShotType(Enum):
    FOREHAND = "forehand"
    BACKHAND = "backhand"
    OVERHEAD_SMASH = "overhead_smash"
    SERVE = "serve"
    UNKNOWN = "unknown"

@dataclass
class Annotation:
    video_file: str
    start_frame: int
    end_frame: int
    player_id: int
    shot_type: str
    notes: str

@dataclass
class TrainingSample:
    video_file: str
    frame_number: int
    player_id: int
    true_shot_type: str
    features: Dict
    ball_x: float
    ball_y: float
    ball_confidence: float

class TennisMLTrainingDataGenerator:
    def __init__(self, annotations_csv: str):
        self.annotations_csv = annotations_csv
        self.annotations = []
        self.training_samples = []
        
    def load_annotations(self):
        """Load annotations from CSV file"""
        logger.info(f"Loading annotations from {self.annotations_csv}")
        df = pd.read_csv(self.annotations_csv)
        
        for _, row in df.iterrows():
            annotation = Annotation(
                video_file=row['video_file'],
                start_frame=int(row['start_frame']),
                end_frame=int(row['end_frame']),
                player_id=int(row['player_id']),
                shot_type=row['shot_type'].lower(),
                notes=row.get('notes', '')
            )
            self.annotations.append(annotation)
        
        logger.info(f"Loaded {len(self.annotations)} annotations")
        
        # Group by video
        video_annotations = {}
        for ann in self.annotations:
            if ann.video_file not in video_annotations:
                video_annotations[ann.video_file] = []
            video_annotations[ann.video_file].append(ann)
        
        return video_annotations
    
    def process_video(self, video_path: str, video_annotations: List[Annotation]):
        """Process a single video and extract features for annotated frames"""
        logger.info(f"Processing {video_path}")
        
        if not Path(video_path).exists():
            logger.error(f"Video file not found: {video_path}")
            return
        
        # Load corresponding CSV data
        csv_path = video_path.replace('.mp4', '_analysis_data.csv')
        if not Path(csv_path).exists():
            logger.error(f"CSV file not found: {csv_path}")
            return
        
        df = pd.read_csv(csv_path)
        logger.info(f"Loaded {len(df)} frames from {csv_path}")
        
        # Process each annotation
        for annotation in video_annotations:
            logger.info(f"Processing annotation: {annotation.shot_type} for player {annotation.player_id} "
                       f"frames {annotation.start_frame}-{annotation.end_frame}")
            
            for frame_idx in range(annotation.start_frame, annotation.end_frame + 1):
                if frame_idx >= len(df):
                    continue
                
                row = df.iloc[frame_idx]
                training_sample = self._extract_training_sample(
                    video_path, frame_idx, annotation, row
                )
                
                if training_sample:
                    self.training_samples.append(training_sample)
    
    def _extract_training_sample(self, video_path: str, frame_idx: int, 
                               annotation: Annotation, row: pd.Series) -> Optional[TrainingSample]:
        """Extract features for a single training sample"""
        try:
            # Basic frame info
            ball_x = row.get('ball_x', 0)
            ball_y = row.get('ball_y', 0)
            ball_confidence = row.get('ball_confidence', 0)
            
            # Parse player data
            player_bboxes = self._parse_player_bboxes(row.get('player_bboxes', ''))
            player_poses = self._parse_player_poses(row.get('pose_keypoints', ''))
            
            if len(player_bboxes) <= annotation.player_id or len(player_poses) <= annotation.player_id:
                return None
            
            # Extract features for the specific player
            bbox = player_bboxes[annotation.player_id]
            pose = player_poses[annotation.player_id]
            
            features = self._extract_ml_features(bbox, pose, ball_x, ball_y)
            
            if not features:
                return None
            
            return TrainingSample(
                video_file=video_path,
                frame_number=frame_idx,
                player_id=annotation.player_id,
                true_shot_type=annotation.shot_type,
                features=features,
                ball_x=ball_x,
                ball_y=ball_y,
                ball_confidence=ball_confidence
            )
            
        except Exception as e:
            logger.warning(f"Error extracting training sample from {video_path} frame {frame_idx}: {e}")
            return None
    
    def _parse_player_bboxes(self, bbox_str: str) -> List[List[float]]:
        """Parse player bounding boxes from CSV string"""
        if not bbox_str or bbox_str == 'nan':
            return []
        try:
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
    
    def _extract_ml_features(self, bbox: List[float], pose: List[List[float]], 
                           ball_x: float, ball_y: float) -> Optional[Dict]:
        """Extract ML features from player data"""
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
            
            # Ball distance (relative to player width)
            if ball_x > 0 and ball_y > 0:
                ball_distance_px = np.sqrt((features['bbox_center_x'] - ball_x)**2 + 
                                         (features['bbox_center_y'] - ball_y)**2)
                features['ball_distance'] = ball_distance_px / features['bbox_width'] if features['bbox_width'] > 0 else float('inf')
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
                
                # Arm extension (distance from elbow to wrist, relative to player width)
                if (features['right_elbow_conf'] > 0.5 and 
                    features['right_wrist_conf'] > 0.5):
                    arm_extension_px = np.sqrt(
                        (features['right_wrist_x'] - features['right_elbow_x'])**2 +
                        (features['right_wrist_y'] - features['right_elbow_y'])**2
                    )
                    features['arm_extension'] = arm_extension_px / features['bbox_width'] if features['bbox_width'] > 0 else 0
                else:
                    features['arm_extension'] = 0
                
                # Wrist position relative to body center (in pixels first)
                if (features['right_wrist_conf'] > 0.5 and 
                    features['body_center_x'] > 0):
                    wrist_relative_x_px = features['right_wrist_x'] - features['body_center_x']
                    wrist_relative_y_px = features['right_wrist_y'] - features['body_center_y']
                    
                    # Convert to relative measurements
                    features['wrist_relative_x'] = wrist_relative_x_px / features['bbox_width'] if features['bbox_width'] > 0 else 0
                    features['wrist_relative_y'] = wrist_relative_y_px / features['bbox_width'] if features['bbox_width'] > 0 else 0
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
            logger.warning(f"Error extracting ML features: {e}")
            return None
    
    def save_training_data(self, output_path: str):
        """Save training data to CSV"""
        if not self.training_samples:
            logger.warning("No training samples to save")
            return
        
        # Convert to DataFrame
        data = []
        for sample in self.training_samples:
            row = {
                'video_file': sample.video_file,
                'frame_number': sample.frame_number,
                'player_id': sample.player_id,
                'true_shot_type': sample.true_shot_type,
                'ball_x': sample.ball_x,
                'ball_y': sample.ball_y,
                'ball_confidence': sample.ball_confidence,
                **sample.features
            }
            data.append(row)
        
        df = pd.DataFrame(data)
        df.to_csv(output_path, index=False)
        
        logger.info(f"Saved {len(df)} training samples to {output_path}")
        logger.info(f"Data shape: {df.shape}")
        
        # Show distribution
        logger.info("Shot type distribution:")
        logger.info(df['true_shot_type'].value_counts())
        
        # Show player distribution
        logger.info("Player distribution:")
        logger.info(df['player_id'].value_counts())
        
        return df

def main():
    parser = argparse.ArgumentParser(description='Generate ML training data from video annotations')
    parser.add_argument('--annotations', required=True, help='CSV file with annotations')
    parser.add_argument('--output', default='tennis_ml_training_data.csv', help='Output CSV file')
    parser.add_argument('--video-dir', default='.', help='Directory containing video files')
    
    args = parser.parse_args()
    
    if not Path(args.annotations).exists():
        logger.error(f"Annotations file not found: {args.annotations}")
        return
    
    # Generate training data
    generator = TennisMLTrainingDataGenerator(args.annotations)
    video_annotations = generator.load_annotations()
    
    # Process each video
    for video_file, annotations in video_annotations.items():
        video_path = Path(args.video_dir) / video_file
        generator.process_video(str(video_path), annotations)
    
    # Save training data
    df = generator.save_training_data(args.output)
    
    logger.info("Training data generation completed!")
    logger.info(f"Total samples: {len(generator.training_samples)}")
    
    # Show sample data
    if len(df) > 0:
        logger.info("Sample training data:")
        print(df.head())

if __name__ == "__main__":
    main()
