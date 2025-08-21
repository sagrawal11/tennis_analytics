#!/usr/bin/env python3
"""
Analyze pose differences between serving player and ready stance player
to understand why classification is failing
"""

import pandas as pd
import numpy as np
import logging
from typing import List, Dict, Tuple

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def parse_poses_from_csv(poses_str: str) -> List[Dict]:
    """Parse poses string from CSV into pose dictionaries"""
    try:
        if pd.isna(poses_str) or poses_str == '':
            return []
        
        parsed_poses = []
        person_poses = poses_str.split(';')
        
        for person_pose in person_poses:
            if not person_pose.strip():
                continue
            
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

def parse_player_bboxes_from_csv(bboxes_str: str) -> List[List[int]]:
    """Parse player bounding boxes from CSV"""
    try:
        if pd.isna(bboxes_str) or bboxes_str == '':
            return []
        
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

def analyze_pose_differences():
    """Analyze pose differences between serving and ready stance players"""
    
    # Load CSV data
    df = pd.read_csv('tennis_analysis_data.csv')
    logger.info(f"Loaded {len(df)} frames of data")
    
    # Analyze first few frames where we know:
    # - Bottom player (higher y-coordinate) is serving
    # - Top player (lower y-coordinate) is in ready stance
    frames_to_analyze = [0, 1, 2, 3, 4]
    
    for frame_idx in frames_to_analyze:
        if frame_idx >= len(df):
            break
            
        row = df.iloc[frame_idx]
        logger.info(f"\n{'='*60}")
        logger.info(f"FRAME {frame_idx}")
        logger.info(f"{'='*60}")
        
        # Parse data
        poses = parse_poses_from_csv(row.get('pose_keypoints', ''))
        player_bboxes = parse_player_bboxes_from_csv(row.get('player_bboxes', ''))
        
        logger.info(f"Found {len(poses)} poses and {len(player_bboxes)} players")
        
        # Analyze each player
        for player_idx, (bbox, pose) in enumerate(zip(player_bboxes, poses)):
            x1, y1, x2, y2 = bbox
            player_center_y = (y1 + y2) / 2
            
            # Determine player position (top vs bottom)
            player_position = "BOTTOM" if player_center_y > 500 else "TOP"  # Rough threshold
            expected_shot = "SERVE" if player_position == "BOTTOM" else "READY_STANCE"
            
            logger.info(f"\nPlayer {player_idx + 1} ({player_position} - Expected: {expected_shot}):")
            logger.info(f"  Bbox: {bbox}")
            logger.info(f"  Center Y: {player_center_y:.1f}")
            
            # Analyze keypoints
            keypoints = pose['keypoints']
            confidence = pose['confidence']
            
            # Key keypoints for tennis analysis (YOLO pose format)
            keypoint_names = {
                0: "nose", 1: "left_eye", 2: "right_eye", 3: "left_ear", 4: "right_ear",
                5: "left_shoulder", 6: "right_shoulder", 7: "left_elbow", 8: "right_elbow",
                9: "left_wrist", 10: "right_wrist", 11: "left_hip", 12: "right_hip",
                13: "left_knee", 14: "right_knee", 15: "left_ankle", 16: "right_ankle"
            }
            
            # Focus on arm keypoints
            arm_keypoints = [5, 6, 7, 8, 9, 10]  # shoulders, elbows, wrists
            
            logger.info("  Arm Keypoints:")
            for kp_idx in arm_keypoints:
                if kp_idx < len(keypoints) and kp_idx < len(confidence):
                    kp = keypoints[kp_idx]
                    conf = confidence[kp_idx]
                    name = keypoint_names.get(kp_idx, f"kp_{kp_idx}")
                    logger.info(f"    {name}: ({kp[0]:.1f}, {kp[1]:.1f}) conf={conf:.3f}")
            
            # Calculate arm metrics
            left_shoulder = keypoints[5] if 5 < len(keypoints) and confidence[5] > 0.3 else None
            right_shoulder = keypoints[6] if 6 < len(keypoints) and confidence[6] > 0.3 else None
            left_elbow = keypoints[7] if 7 < len(keypoints) and confidence[7] > 0.3 else None
            right_elbow = keypoints[8] if 8 < len(keypoints) and confidence[8] > 0.3 else None
            left_wrist = keypoints[9] if 9 < len(keypoints) and confidence[9] > 0.3 else None
            right_wrist = keypoints[10] if 10 < len(keypoints) and confidence[10] > 0.3 else None
            
            logger.info("  Arm Analysis:")
            
            # Left arm analysis
            if all([left_shoulder, left_elbow, left_wrist]):
                left_upper_arm_length = np.sqrt((left_elbow[0] - left_shoulder[0])**2 + (left_elbow[1] - left_shoulder[1])**2)
                left_forearm_length = np.sqrt((left_wrist[0] - left_elbow[0])**2 + (left_wrist[1] - left_elbow[1])**2)
                left_extension_ratio = left_forearm_length / left_upper_arm_length if left_upper_arm_length > 0 else 0
                left_wrist_height = left_wrist[1]  # Lower Y = higher position
                left_shoulder_height = left_shoulder[1]
                left_arm_raised = left_wrist[1] < left_shoulder[1] - 20  # Wrist above shoulder
                
                logger.info(f"    Left Arm:")
                logger.info(f"      Upper arm length: {left_upper_arm_length:.1f}")
                logger.info(f"      Forearm length: {left_forearm_length:.1f}")
                logger.info(f"      Extension ratio: {left_extension_ratio:.3f}")
                logger.info(f"      Wrist height: {left_wrist_height:.1f}")
                logger.info(f"      Shoulder height: {left_shoulder_height:.1f}")
                logger.info(f"      Arm raised: {left_arm_raised}")
            
            # Right arm analysis
            if all([right_shoulder, right_elbow, right_wrist]):
                right_upper_arm_length = np.sqrt((right_elbow[0] - right_shoulder[0])**2 + (right_elbow[1] - right_shoulder[1])**2)
                right_forearm_length = np.sqrt((right_wrist[0] - right_elbow[0])**2 + (right_wrist[1] - right_elbow[1])**2)
                right_extension_ratio = right_forearm_length / right_upper_arm_length if right_upper_arm_length > 0 else 0
                right_wrist_height = right_wrist[1]  # Lower Y = higher position
                right_shoulder_height = right_shoulder[1]
                right_arm_raised = right_wrist[1] < right_shoulder[1] - 20  # Wrist above shoulder
                
                logger.info(f"    Right Arm:")
                logger.info(f"      Upper arm length: {right_upper_arm_length:.1f}")
                logger.info(f"      Forearm length: {right_forearm_length:.1f}")
                logger.info(f"      Extension ratio: {right_extension_ratio:.3f}")
                logger.info(f"      Wrist height: {right_wrist_height:.1f}")
                logger.info(f"      Shoulder height: {right_shoulder_height:.1f}")
                logger.info(f"      Arm raised: {right_arm_raised}")
            
            # Compare with current classification logic
            logger.info("  Current Classification Logic Analysis:")
            
            # Current arm extension threshold
            ARM_EXTENSION_THRESHOLD = 0.7
            
            left_extended = left_extension_ratio > ARM_EXTENSION_THRESHOLD if 'left_extension_ratio' in locals() else False
            right_extended = right_extension_ratio > ARM_EXTENSION_THRESHOLD if 'right_extension_ratio' in locals() else False
            
            logger.info(f"    Left arm extended (>0.7): {left_extended}")
            logger.info(f"    Right arm extended (>0.7): {right_extended}")
            
            # This is why both are classified as forehand!
            if left_extended and not right_extended:
                logger.info("    → Would classify as BACKHAND")
            elif right_extended and not left_extended:
                logger.info("    → Would classify as FOREHAND")
            elif left_extended and right_extended:
                logger.info("    → Would classify as BACKHAND (two-handed)")
            else:
                logger.info("    → Would classify as READY_STANCE")
            
            logger.info(f"    → But we know it should be: {expected_shot}")

if __name__ == "__main__":
    analyze_pose_differences()
