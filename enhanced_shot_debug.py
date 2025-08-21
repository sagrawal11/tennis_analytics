#!/usr/bin/env python3
"""
Enhanced Shot Classification Debug Tool
Shows detailed information about classification decisions
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

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from enhanced_shot_classifier import EnhancedShotClassifier

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class EnhancedShotDebug:
    """Debug tool for enhanced shot classification"""
    
    def __init__(self, config_path: str = "config.yaml"):
        """Initialize the debug tool"""
        self.config_path = config_path
        self.config = self._load_config()
        
        # Initialize shot classifier
        self.shot_classifier = EnhancedShotClassifier()
        
        # Debug colors
        self.debug_colors = {
            'keypoint': (0, 255, 0),     # Green
            'bbox': (255, 255, 0),       # Cyan
            'ball': (0, 255, 255),       # Yellow
            'court': (255, 255, 255),    # White
            'text': (255, 255, 255),     # White
            'arm_line': (255, 0, 255),   # Magenta
            'debug_info': (0, 255, 255)  # Yellow
        }
        
        logger.info("Enhanced Shot Debug tool initialized")
    
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
    
    def _parse_ball_position_from_csv(self, ball_x: str, ball_y: str) -> Optional[List[int]]:
        """Parse ball position from CSV"""
        try:
            if pd.isna(ball_x) or pd.isna(ball_y) or ball_x == '' or ball_y == '':
                return None
            
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
    
    def debug_shot_classification(self, csv_path: str, video_path: str, start_frame: int = 0, max_frames: int = 50):
        """Debug shot classification with detailed visualization"""
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
            
            # Skip to start frame
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            
            # Create window
            cv2.namedWindow("Shot Classification Debug", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Shot Classification Debug", 1600, 900)
            
            frame_count = start_frame
            processed_frames = 0
            
            # Process frames
            for index in range(start_frame, min(start_frame + max_frames, len(df))):
                ret, frame = cap.read()
                if not ret:
                    break
                
                row = df.iloc[index]
                frame_count += 1
                processed_frames += 1
                
                # Parse data from CSV row
                poses = self._parse_poses_from_csv(row.get('pose_keypoints', ''))
                ball_position = self._parse_ball_position_from_csv(row.get('ball_x', ''), row.get('ball_y', ''))
                court_keypoints = self._parse_court_keypoints_from_csv(row.get('court_keypoints', ''))
                player_bboxes = self._parse_player_bboxes_from_csv(row.get('player_bboxes', ''))
                
                # Debug each player
                for i, bbox in enumerate(player_bboxes):
                    debug_info = self._debug_single_player(bbox, ball_position, poses, court_keypoints)
                    self._draw_debug_visualization(frame, bbox, poses, ball_position, court_keypoints, debug_info, i)
                
                # Add frame info
                cv2.putText(frame, f"Frame: {frame_count}/{total_frames} (Debug)", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                
                cv2.putText(frame, "Press 'q' to quit, 'space' to pause, 'n' for next frame", (10, height - 20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                # Show frame
                cv2.imshow("Shot Classification Debug", frame)
                
                # Handle key presses
                key = cv2.waitKey(0) & 0xFF  # Wait for key press
                if key == ord('q'):
                    break
                elif key == ord('n'):
                    continue  # Next frame
                elif key == ord(' '):
                    # Pause/unpause
                    cv2.waitKey(0)
            
            # Cleanup
            cap.release()
            cv2.destroyAllWindows()
            
            logger.info(f"Debug completed! Processed {processed_frames} frames")
            
        except Exception as e:
            logger.error(f"Error in debug tool: {e}")
    
    def _debug_single_player(self, player_bbox: List[int], ball_position: Optional[List[int]], 
                           poses: List[Dict], court_keypoints: List[Tuple]) -> Dict:
        """Debug a single player's shot classification"""
        try:
            # Get player center
            player_center_x = (player_bbox[0] + player_bbox[2]) / 2
            player_center_y = (player_bbox[1] + player_bbox[3]) / 2
            
            debug_info = {
                'player_center': (player_center_x, player_center_y),
                'closest_pose': None,
                'confident_keypoints': {},
                'arm_analysis': {},
                'classification_steps': [],
                'final_shot': 'unknown'
            }
            
            # Find closest pose
            closest_pose = self.shot_classifier._find_closest_pose(poses, player_center_x, player_center_y)
            debug_info['closest_pose'] = closest_pose
            
            if not closest_pose:
                debug_info['classification_steps'].append("No pose found -> ready_stance")
                debug_info['final_shot'] = 'ready_stance'
                return debug_info
            
            # Extract confident keypoints
            confident_keypoints = self.shot_classifier._extract_confident_keypoints(closest_pose)
            debug_info['confident_keypoints'] = confident_keypoints
            
            if len(confident_keypoints) < 8:
                debug_info['classification_steps'].append(f"Too few keypoints ({len(confident_keypoints)}) -> ready_stance")
                debug_info['final_shot'] = 'ready_stance'
                return debug_info
            
            # Analyze arms
            debug_info['arm_analysis'] = self._analyze_arms(confident_keypoints)
            
            # Step through classification logic
            debug_info['classification_steps'].append(f"Found {len(confident_keypoints)} confident keypoints")
            
            # Check serve
            if self.shot_classifier._is_serve(confident_keypoints, player_center_y, court_keypoints, ball_position):
                debug_info['classification_steps'].append("Serve detected")
                debug_info['final_shot'] = 'serve'
                return debug_info
            else:
                debug_info['classification_steps'].append("Not a serve")
            
            # Check overhead smash
            if self.shot_classifier._is_overhead_smash(confident_keypoints, player_center_y, ball_position):
                debug_info['classification_steps'].append("Overhead smash detected")
                debug_info['final_shot'] = 'overhead_smash'
                return debug_info
            else:
                debug_info['classification_steps'].append("Not overhead smash")
            
            # Check volley
            if self.shot_classifier._is_volley(confident_keypoints, player_center_y, court_keypoints):
                debug_info['classification_steps'].append("Volley detected")
                debug_info['final_shot'] = 'volley'
                return debug_info
            else:
                debug_info['classification_steps'].append("Not a volley")
            
            # Check groundstroke
            groundstroke = self.shot_classifier._classify_groundstroke(confident_keypoints, player_center_x, ball_position)
            if groundstroke != "ready_stance":
                debug_info['classification_steps'].append(f"Groundstroke: {groundstroke}")
                debug_info['final_shot'] = groundstroke
                return debug_info
            else:
                debug_info['classification_steps'].append("No groundstroke detected")
            
            # Check movement
            if self.shot_classifier._is_moving(confident_keypoints, 0.0):
                debug_info['classification_steps'].append("Movement detected")
                debug_info['final_shot'] = 'moving'
                return debug_info
            else:
                debug_info['classification_steps'].append("No movement detected")
            
            # Default
            debug_info['classification_steps'].append("Default to ready_stance")
            debug_info['final_shot'] = 'ready_stance'
            
            return debug_info
            
        except Exception as e:
            logger.error(f"Error in debug analysis: {e}")
            return {'final_shot': 'error', 'classification_steps': [f"Error: {e}"]}
    
    def _analyze_arms(self, keypoints: Dict[int, List]) -> Dict:
        """Analyze arm positions for debugging"""
        arm_analysis = {
            'left_arm': {'shoulder': None, 'elbow': None, 'wrist': None, 'extended': False},
            'right_arm': {'shoulder': None, 'elbow': None, 'wrist': None, 'extended': False},
            'arm_angles': {},
            'extension_ratios': {}
        }
        
        try:
            # Left arm
            left_shoulder = keypoints.get(5)
            left_elbow = keypoints.get(7)
            left_wrist = keypoints.get(9)
            
            if left_shoulder and left_elbow and left_wrist:
                arm_analysis['left_arm'] = {
                    'shoulder': left_shoulder,
                    'elbow': left_elbow,
                    'wrist': left_wrist,
                    'extended': self.shot_classifier._is_arm_extended(left_shoulder, left_elbow, left_wrist)
                }
            
            # Right arm
            right_shoulder = keypoints.get(6)
            right_elbow = keypoints.get(8)
            right_wrist = keypoints.get(10)
            
            if right_shoulder and right_elbow and right_wrist:
                arm_analysis['right_arm'] = {
                    'shoulder': right_shoulder,
                    'elbow': right_elbow,
                    'wrist': right_wrist,
                    'extended': self.shot_classifier._is_arm_extended(right_shoulder, right_elbow, right_wrist)
                }
            
        except Exception as e:
            logger.debug(f"Error analyzing arms: {e}")
        
        return arm_analysis
    
    def _draw_debug_visualization(self, frame: np.ndarray, bbox: List[int], poses: List[Dict], 
                                ball_position: Optional[List[int]], court_keypoints: List[Tuple], 
                                debug_info: Dict, player_idx: int):
        """Draw comprehensive debug visualization"""
        try:
            x1, y1, x2, y2 = bbox
            
            # Draw player bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), self.debug_colors['bbox'], 2)
            
            # Draw player center
            center_x, center_y = debug_info['player_center']
            cv2.circle(frame, (int(center_x), int(center_y)), 5, (255, 0, 0), -1)
            
            # Draw pose keypoints with confidence
            closest_pose = debug_info['closest_pose']
            if closest_pose and 'keypoints' in closest_pose:
                keypoints = closest_pose['keypoints']
                confidence = closest_pose.get('confidence', [])
                
                for i, kp in enumerate(keypoints):
                    if i < len(confidence):
                        conf = confidence[i]
                        if conf > 0.3:
                            x, y = int(kp[0]), int(kp[1])
                            # Color code by confidence
                            color_intensity = int(255 * min(conf, 1.0))
                            color = (0, color_intensity, 0)
                            cv2.circle(frame, (x, y), 3, color, -1)
                            # Draw keypoint number
                            cv2.putText(frame, str(i), (x + 5, y), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
            
            # Draw arm analysis
            arm_analysis = debug_info.get('arm_analysis', {})
            
            # Left arm
            left_arm = arm_analysis.get('left_arm', {})
            if all(left_arm.get(k) for k in ['shoulder', 'elbow', 'wrist']):
                shoulder = left_arm['shoulder']
                elbow = left_arm['elbow']
                wrist = left_arm['wrist']
                
                # Draw arm lines
                color = (0, 255, 0) if left_arm.get('extended') else (0, 0, 255)
                cv2.line(frame, (int(shoulder[0]), int(shoulder[1])), (int(elbow[0]), int(elbow[1])), color, 2)
                cv2.line(frame, (int(elbow[0]), int(elbow[1])), (int(wrist[0]), int(wrist[1])), color, 2)
            
            # Right arm  
            right_arm = arm_analysis.get('right_arm', {})
            if all(right_arm.get(k) for k in ['shoulder', 'elbow', 'wrist']):
                shoulder = right_arm['shoulder']
                elbow = right_arm['elbow']
                wrist = right_arm['wrist']
                
                # Draw arm lines
                color = (0, 255, 0) if right_arm.get('extended') else (0, 0, 255)
                cv2.line(frame, (int(shoulder[0]), int(shoulder[1])), (int(elbow[0]), int(elbow[1])), color, 2)
                cv2.line(frame, (int(elbow[0]), int(elbow[1])), (int(wrist[0]), int(wrist[1])), color, 2)
            
            # Draw ball position
            if ball_position:
                ball_x, ball_y = ball_position
                cv2.circle(frame, (ball_x, ball_y), 8, self.debug_colors['ball'], -1)
                cv2.putText(frame, "BALL", (ball_x + 10, ball_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
            
            # Draw classification info
            text_x = x1
            text_y = y2 + 25
            
            # Final classification
            final_shot = debug_info.get('final_shot', 'unknown')
            cv2.putText(frame, f"Player {player_idx + 1}: {final_shot.upper()}", (text_x, text_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            text_y += 25
            
            # Classification steps
            steps = debug_info.get('classification_steps', [])
            for i, step in enumerate(steps[-3:]):  # Show last 3 steps
                cv2.putText(frame, f"• {step}", (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
                text_y += 15
            
            # Arm extension info
            if left_arm.get('extended') or right_arm.get('extended'):
                arm_text = f"Arms: L={left_arm.get('extended', False)}, R={right_arm.get('extended', False)}"
                cv2.putText(frame, arm_text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
                
        except Exception as e:
            logger.error(f"Error drawing debug visualization: {e}")

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Enhanced Shot Classification Debug Tool")
    parser.add_argument("--csv", default="tennis_analysis_data.csv", help="CSV file with pose data")
    parser.add_argument("--video", default="tennis_test5.mp4", help="Input video path")
    parser.add_argument("--config", default="config.yaml", help="Configuration file")
    parser.add_argument("--start", type=int, default=0, help="Start frame number")
    parser.add_argument("--frames", type=int, default=50, help="Number of frames to debug")
    
    args = parser.parse_args()
    
    # Check if files exist
    if not os.path.exists(args.csv):
        logger.error(f"CSV file not found: {args.csv}")
        return
    
    if not os.path.exists(args.video):
        logger.error(f"Video file not found: {args.video}")
        return
    
    # Create debug tool
    debug_tool = EnhancedShotDebug(args.config)
    
    # Run debug analysis
    debug_tool.debug_shot_classification(args.csv, args.video, args.start, args.frames)

if __name__ == "__main__":
    main()
