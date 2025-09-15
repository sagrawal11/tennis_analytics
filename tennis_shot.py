#!/usr/bin/env python3
"""
Tennis Shot Classification System
Combines both legacy pose-based and motion-based shot classification methods.
This is a standalone script that can be used independently or integrated into other systems.
"""

import numpy as np
import logging
import math
import pandas as pd
import cv2
import argparse
from typing import Dict, List, Optional, Tuple, Deque
from collections import deque

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

class TennisShotClassifier:
    """
    Unified tennis shot classification system combining:
    1. Legacy pose-based classification (static analysis)
    2. Motion-based classification (dynamic analysis)
    """
    
    def __init__(self, use_motion_based: bool = True):
        """Initialize the tennis shot classifier"""
        self.use_motion_based = use_motion_based
        
        # Shot type definitions
        self.shot_types = {
            'serve': 'Serve',
            'forehand': 'Forehand', 
            'backhand': 'Backhand',
            'overhead_smash': 'Overhead Smash',
            'ready_stance': 'Ready Stance',
            'moving': 'Moving',
            'unknown': 'Unknown'
        }
        
        # Legacy pose-based parameters
        self.arm_extension_threshold = 0.8
        self.swing_confidence_threshold = 0.3
        self.baseline_proximity_threshold = 50
        
        # Motion-based parameters (tuned for better accuracy)
        self.motion_history_length = 15  # Reduced for more responsive detection
        self.swing_detection_window = 6   # Reduced for faster detection
        self.velocity_threshold = 20.0    # Lowered for more sensitive detection
        self.acceleration_threshold = 12.0  # Lowered for more sensitive detection
        self.swing_duration_frames = 4    # Reduced for faster detection
        self.backswing_ratio = 0.4
        self.contact_zone_threshold = 80  # Increased for more lenient ball proximity
        self.shot_persistence_frames = 6  # Reduced for less persistence
        
        # Motion tracking (for motion-based classification)
        self.player_motion_history = {}
        self.swing_sequences = {}
        self.last_shot_frame = {}
        self.current_shot_type = {}
        self.shot_start_frame = {}
        
        # Legacy pose tracking
        self.player_shot_history = deque(maxlen=20)
        
        logger.info(f"Tennis shot classifier initialized (motion-based: {use_motion_based})")
    
    def classify_shot(self, player_bbox: List[int], ball_position: Optional[List[int]], 
                     poses: List[Dict], court_keypoints: List[Tuple], 
                     frame_number: int = 0, player_id: int = 0) -> str:
        """
        Main shot classification method - chooses between legacy and motion-based
        
        Args:
            player_bbox: [x1, y1, x2, y2] player bounding box
            ball_position: [x, y] ball position (can be None)
            poses: List of pose dictionaries with keypoints
            court_keypoints: List of court keypoint tuples
            frame_number: Current frame number
            player_id: Player identifier
            
        Returns:
            Shot type string
        """
        try:
            if self.use_motion_based:
                return self._classify_shot_motion_based(
                    player_bbox, ball_position, poses, court_keypoints, frame_number, player_id
                )
            else:
                return self._classify_shot_legacy(
                    player_bbox, ball_position, poses, court_keypoints, frame_number, player_id
                )
        except Exception as e:
            logger.error(f"Error in shot classification: {e}")
            return "unknown"
    
    def _classify_shot_legacy(self, player_bbox: List[int], ball_position: Optional[List[int]], 
                             poses: List[Dict], court_keypoints: List[Tuple], 
                             frame_number: int = 0, player_id: int = 0) -> str:
        """Legacy pose-based shot classification"""
        try:
            # Get player center
            player_center_x = (player_bbox[0] + player_bbox[2]) / 2
            player_center_y = (player_bbox[1] + player_bbox[3]) / 2
            
            # Get ball position
            ball_x, ball_y = ball_position if ball_position else (0, 0)
            
            # Extract pose keypoints for this player
            if not poses or player_id >= len(poses):
                return "unknown"
            
            keypoints = poses[player_id]
            
            # Filter for confident keypoints only
            confident_keypoints = {}
            for i, kp in keypoints.items():
                if len(kp) >= 3 and kp[2] > self.swing_confidence_threshold:
                    confident_keypoints[i] = kp
            
            # Check if we have enough keypoints
            if len(confident_keypoints) < 8:
                return "ready_stance"
            
            # Detect serve situation
            if self._is_serve_situation_legacy(player_center_y, court_keypoints):
                return "serve"
            
            # Analyze swing pose
            swing_type = self._analyze_swing_pose_legacy(
                confident_keypoints, player_center_x, ball_x, ball_y, player_center_y
            )
            
            if swing_type != "ready_stance":
                return swing_type
            
            # Check ready stance
            if self._is_ready_stance_legacy(confident_keypoints, player_center_x):
                return "ready_stance"
            
            # Default to moving
            return "moving"
            
        except Exception as e:
            logger.error(f"Error in legacy classification: {e}")
            return "ready_stance"
    
    def _classify_shot_motion_based(self, player_bbox: List[int], ball_position: Optional[List[int]], 
                                   poses: List[Dict], court_keypoints: List[Tuple], 
                                   frame_number: int = 0, player_id: int = 0) -> str:
        """Motion-based shot classification"""
        try:
            # Get player center
            player_center_x = (player_bbox[0] + player_bbox[2]) / 2
            player_center_y = (player_bbox[1] + player_bbox[3]) / 2
            player_center = [player_center_x, player_center_y]
            
            # Extract pose keypoints for this player
            if not poses or player_id >= len(poses):
                return "unknown"
            
            keypoints = poses[player_id]
            
            # Update motion history
            self._update_motion_history(player_id, keypoints, player_center, frame_number)
            
            # Analyze motion patterns
            motion_analysis = self._analyze_motion_patterns(player_id)
            
            # Detect swing sequences
            swing_detection = self._detect_swing_sequence(player_id, motion_analysis, ball_position, player_center)
            
            # Classify shot type based on swing analysis
            shot_type = self._classify_swing_type(player_id, swing_detection, keypoints, player_center, ball_position)
            
            # Apply tennis game flow rules
            final_shot_type = self._apply_tennis_rules(shot_type, frame_number, player_id, ball_position, player_center)
            
            # Update shot state
            self._update_shot_state(player_id, final_shot_type, frame_number)
            
            return final_shot_type
            
        except Exception as e:
            logger.error(f"Error in motion-based classification: {e}")
            return "unknown"
    
    # ==================== LEGACY POSE-BASED METHODS ====================
    
    def _is_serve_situation_legacy(self, player_y: float, court_keypoints: List[Tuple]) -> bool:
        """Check if this is a serve situation (players near baseline)"""
        try:
            if not court_keypoints or len(court_keypoints) < 4:
                return False
            
            # Find baseline y-coordinate
            baseline_y = None
            for kp in court_keypoints:
                if kp[0] is not None and kp[1] is not None:
                    if baseline_y is None or kp[1] > baseline_y:
                        baseline_y = kp[1]
            
            if baseline_y is None:
                return False
            
            # Check if player is close to baseline
            distance_to_baseline = abs(player_y - baseline_y)
            return distance_to_baseline < self.baseline_proximity_threshold
            
        except Exception as e:
            logger.error(f"Error checking serve situation: {e}")
            return False
    
    def _analyze_swing_pose_legacy(self, keypoints: Dict[int, List], player_center_x: float, 
                                  ball_x: float, ball_y: float, player_center_y: float) -> str:
        """Analyze pose to determine swing type (forehand, backhand, overhand smash)"""
        try:
            # Get arm keypoints
            left_shoulder = keypoints.get(5)
            right_shoulder = keypoints.get(6)
            left_elbow = keypoints.get(7)
            right_elbow = keypoints.get(8)
            left_wrist = keypoints.get(9)
            right_wrist = keypoints.get(10)
            
            # Check if we have enough arm keypoints
            arm_keypoints = [kp for kp in [left_shoulder, right_shoulder, left_elbow, right_elbow, left_wrist, right_wrist] if kp is not None]
            if len(arm_keypoints) < 4:
                return "ready_stance"
            
            # Calculate arm extension
            left_arm_extended = False
            right_arm_extended = False
            
            # Check left arm extension
            if left_shoulder and left_elbow and left_wrist:
                left_arm_length = np.sqrt((left_elbow[0] - left_shoulder[0])**2 + (left_elbow[1] - left_shoulder[1])**2)
                left_forearm_length = np.sqrt((left_wrist[0] - left_elbow[0])**2 + (left_wrist[1] - left_elbow[1])**2)
                left_arm_extended = left_forearm_length > left_arm_length * self.arm_extension_threshold
            
            # Check right arm extension
            if right_shoulder and right_elbow and right_wrist:
                right_arm_length = np.sqrt((right_elbow[0] - right_shoulder[0])**2 + (right_elbow[1] - right_shoulder[1])**2)
                right_forearm_length = np.sqrt((right_wrist[0] - right_elbow[0])**2 + (right_wrist[1] - right_elbow[1])**2)
                right_arm_extended = right_forearm_length > right_arm_length * self.arm_extension_threshold
            
            # Check for overhead smash (both arms raised)
            if left_arm_extended and right_arm_extended:
                if left_wrist and right_wrist:
                    avg_wrist_y = (left_wrist[1] + right_wrist[1]) / 2
                    if avg_wrist_y < player_center_y - 50:  # Wrists above shoulders
                        return "overhead_smash"
            
            # Check for groundstrokes
            if right_arm_extended:
                # Determine forehand vs backhand based on arm position relative to body
                if right_wrist:
                    if right_wrist[0] < player_center_x - 20:  # Right hand left of center
                        return "backhand"
                    else:
                        return "forehand"
            
            if left_arm_extended:
                # Left arm extended - could be backhand or forehand
                if left_wrist:
                    if left_wrist[0] < player_center_x - 20:  # Left hand left of center
                        return "forehand"
                    else:
                        return "backhand"
            
            return "ready_stance"
            
        except Exception as e:
            logger.error(f"Error analyzing swing pose: {e}")
            return "ready_stance"
    
    def _is_ready_stance_legacy(self, keypoints: Dict[int, List], player_center_x: float) -> bool:
        """Check if player is in ready stance (neutral position)"""
        try:
            # Get shoulder keypoints
            left_shoulder = keypoints.get(5)
            right_shoulder = keypoints.get(6)
            
            if not (left_shoulder and right_shoulder):
                return False
            
            # Check if shoulders are level (ready stance)
            shoulder_height_diff = abs(left_shoulder[1] - right_shoulder[1])
            return shoulder_height_diff < 30  # Shoulders roughly level
            
        except Exception as e:
            logger.error(f"Error checking ready stance: {e}")
            return False
    
    # ==================== MOTION-BASED METHODS ====================
    
    def _update_motion_history(self, player_id: int, keypoints: Dict[int, List], 
                              player_center: List[float], frame_number: int):
        """Update motion history for a player"""
        if player_id not in self.player_motion_history:
            self.player_motion_history[player_id] = deque(maxlen=self.motion_history_length)
        
        # Extract key arm keypoints
        right_shoulder = keypoints.get(6)
        right_elbow = keypoints.get(8) 
        right_wrist = keypoints.get(10)
        left_shoulder = keypoints.get(5)
        left_elbow = keypoints.get(7)
        left_wrist = keypoints.get(9)
        
        # Calculate arm motion data
        motion_data = {
            'frame': frame_number,
            'player_center': player_center,
            'right_arm': {
                'shoulder': right_shoulder,
                'elbow': right_elbow,
                'wrist': right_wrist,
                'arm_center': self._calculate_arm_center(right_shoulder, right_elbow, right_wrist)
            },
            'left_arm': {
                'shoulder': left_shoulder,
                'elbow': left_elbow,
                'wrist': left_wrist,
                'arm_center': self._calculate_arm_center(left_shoulder, left_elbow, left_wrist)
            }
        }
        
        self.player_motion_history[player_id].append(motion_data)
    
    def _calculate_arm_center(self, shoulder: Optional[List], elbow: Optional[List], 
                            wrist: Optional[List]) -> Optional[List[float]]:
        """Calculate the center point of an arm"""
        if not all([shoulder, elbow, wrist]):
            return None
        
        # Weighted center (wrist has more influence for swing detection)
        center_x = (shoulder[0] * 0.2 + elbow[0] * 0.3 + wrist[0] * 0.5)
        center_y = (shoulder[1] * 0.2 + elbow[1] * 0.3 + wrist[1] * 0.5)
        
        return [center_x, center_y]
    
    def _analyze_motion_patterns(self, player_id: int) -> Dict:
        """Analyze motion patterns to detect swing characteristics"""
        if player_id not in self.player_motion_history:
            return {'has_motion': False}
        
        history = list(self.player_motion_history[player_id])
        if len(history) < 3:
            return {'has_motion': False}
        
        # Analyze right arm motion (primary swing arm)
        right_arm_motion = self._analyze_arm_motion(history, 'right_arm')
        
        # Analyze left arm motion (support arm)
        left_arm_motion = self._analyze_arm_motion(history, 'left_arm')
        
        # Analyze player center motion
        player_motion = self._analyze_player_motion(history)
        
        return {
            'has_motion': right_arm_motion['has_motion'] or left_arm_motion['has_motion'],
            'right_arm': right_arm_motion,
            'left_arm': left_arm_motion,
            'player_motion': player_motion,
            'swing_characteristics': self._detect_swing_characteristics(right_arm_motion, left_arm_motion)
        }
    
    def _analyze_arm_motion(self, history: List[Dict], arm_key: str) -> Dict:
        """Analyze motion of a specific arm"""
        if len(history) < 3:
            return {'has_motion': False}
        
        # Extract arm centers
        arm_centers = []
        for frame_data in history:
            arm_center = frame_data[arm_key]['arm_center']
            if arm_center:
                arm_centers.append(arm_center)
        
        if len(arm_centers) < 3:
            return {'has_motion': False}
        
        # Calculate velocities
        velocities = []
        for i in range(1, len(arm_centers)):
            dx = arm_centers[i][0] - arm_centers[i-1][0]
            dy = arm_centers[i][1] - arm_centers[i-1][1]
            velocity = math.sqrt(dx*dx + dy*dy)
            velocities.append(velocity)
        
        # Calculate accelerations
        accelerations = []
        for i in range(1, len(velocities)):
            acceleration = velocities[i] - velocities[i-1]
            accelerations.append(acceleration)
        
        # Analyze motion characteristics
        avg_velocity = np.mean(velocities) if velocities else 0
        max_velocity = max(velocities) if velocities else 0
        avg_acceleration = np.mean(accelerations) if accelerations else 0
        max_acceleration = max(accelerations) if accelerations else 0
        
        # Detect swing-like motion (more lenient criteria)
        has_swing_motion = (max_velocity > self.velocity_threshold and 
                           len(velocities) >= 3)  # Removed acceleration requirement
        
        # Analyze motion direction
        motion_direction = self._analyze_motion_direction(arm_centers)
        
        return {
            'has_motion': max_velocity > 5.0,
            'has_swing_motion': has_swing_motion,
            'avg_velocity': avg_velocity,
            'max_velocity': max_velocity,
            'avg_acceleration': avg_acceleration,
            'max_acceleration': max_acceleration,
            'motion_direction': motion_direction,
            'arm_centers': arm_centers
        }
    
    def _analyze_player_motion(self, history: List[Dict]) -> Dict:
        """Analyze overall player motion"""
        if len(history) < 3:
            return {'has_motion': False}
        
        player_centers = [frame_data['player_center'] for frame_data in history]
        
        # Calculate player velocity
        velocities = []
        for i in range(1, len(player_centers)):
            dx = player_centers[i][0] - player_centers[i-1][0]
            dy = player_centers[i][1] - player_centers[i-1][1]
            velocity = math.sqrt(dx*dx + dy*dy)
            velocities.append(velocity)
        
        avg_velocity = np.mean(velocities) if velocities else 0
        max_velocity = max(velocities) if velocities else 0
        
        return {
            'has_motion': avg_velocity > 10.0,
            'avg_velocity': avg_velocity,
            'max_velocity': max_velocity
        }
    
    def _analyze_motion_direction(self, arm_centers: List[List[float]]) -> str:
        """Analyze the direction of arm motion"""
        if len(arm_centers) < 3:
            return "unknown"
        
        # Calculate overall motion vector
        start_x, start_y = arm_centers[0]
        end_x, end_y = arm_centers[-1]
        
        dx = end_x - start_x
        dy = end_y - start_y
        
        # Determine primary direction
        if abs(dx) > abs(dy):
            return "horizontal" if dx > 0 else "horizontal_reverse"
        else:
            return "vertical" if dy > 0 else "vertical_reverse"
    
    def _detect_swing_characteristics(self, right_arm: Dict, left_arm: Dict) -> Dict:
        """Detect tennis swing characteristics"""
        characteristics = {
            'has_swing': False,
            'swing_type': 'unknown',
            'swing_phase': 'unknown',
            'confidence': 0.0
        }
        
        # Check if there's a swing motion
        if not (right_arm['has_swing_motion'] or left_arm['has_swing_motion']):
            return characteristics
        
        # Additional check: require significant motion (lowered threshold)
        if right_arm['max_velocity'] < 25.0 and left_arm['max_velocity'] < 25.0:
            return characteristics
        
        characteristics['has_swing'] = True
        
        # Analyze swing type based on motion patterns
        if right_arm['has_swing_motion']:
            motion_dir = right_arm['motion_direction']
            velocity_profile = self._analyze_velocity_profile(right_arm['arm_centers'])
            
            if motion_dir == "horizontal":
                characteristics['swing_type'] = "forehand"
                characteristics['confidence'] = 0.8
            elif motion_dir == "horizontal_reverse":
                characteristics['swing_type'] = "backhand"
                characteristics['confidence'] = 0.8
            elif motion_dir == "vertical":
                characteristics['swing_type'] = "overhead"
                characteristics['confidence'] = 0.7
            else:
                characteristics['swing_type'] = "groundstroke"
                characteristics['confidence'] = 0.6
            
            # Determine swing phase
            characteristics['swing_phase'] = self._determine_swing_phase(velocity_profile)
        
        return characteristics
    
    def _analyze_velocity_profile(self, arm_centers: List[List[float]]) -> Dict:
        """Analyze the velocity profile of arm motion"""
        if len(arm_centers) < 4:
            return {'profile': 'unknown'}
        
        # Calculate velocities at each point
        velocities = []
        for i in range(1, len(arm_centers)):
            dx = arm_centers[i][0] - arm_centers[i-1][0]
            dy = arm_centers[i][1] - arm_centers[i-1][1]
            velocity = math.sqrt(dx*dx + dy*dy)
            velocities.append(velocity)
        
        # Analyze velocity pattern
        if len(velocities) >= 3:
            first_half = velocities[:len(velocities)//2]
            second_half = velocities[len(velocities)//2:]
            
            avg_first = np.mean(first_half) if first_half else 0
            avg_second = np.mean(second_half) if second_half else 0
            
            if avg_first < avg_second:
                return {'profile': 'accelerating', 'peak_velocity': max(velocities)}
            elif avg_first > avg_second:
                return {'profile': 'decelerating', 'peak_velocity': max(velocities)}
            else:
                return {'profile': 'constant', 'peak_velocity': max(velocities)}
        
        return {'profile': 'unknown'}
    
    def _determine_swing_phase(self, velocity_profile: Dict) -> str:
        """Determine the phase of the swing based on velocity profile"""
        profile = velocity_profile.get('profile', 'unknown')
        peak_velocity = velocity_profile.get('peak_velocity', 0)
        
        if profile == 'accelerating' and peak_velocity > self.velocity_threshold:
            return 'backswing'
        elif profile == 'decelerating' and peak_velocity > self.velocity_threshold:
            return 'follow_through'
        elif peak_velocity > self.velocity_threshold * 1.5:
            return 'contact'
        else:
            return 'unknown'
    
    def _detect_swing_sequence(self, player_id: int, motion_analysis: Dict, 
                              ball_position: Optional[List[int]], 
                              player_center: List[float]) -> Dict:
        """Detect complete swing sequences"""
        swing_detection = {
            'has_swing': False,
            'swing_type': 'unknown',
            'swing_phase': 'unknown',
            'ball_proximity': False,
            'confidence': 0.0
        }
        
        # Check for swing characteristics
        if not motion_analysis.get('swing_characteristics', {}).get('has_swing', False):
            return swing_detection
        
        swing_chars = motion_analysis['swing_characteristics']
        swing_detection.update(swing_chars)
        
        # Check ball proximity
        if ball_position and player_center:
            ball_x, ball_y = ball_position
            player_x, player_y = player_center
            
            distance = math.sqrt((ball_x - player_x)**2 + (ball_y - player_y)**2)
            swing_detection['ball_proximity'] = distance < 200
        
        return swing_detection
    
    def _classify_swing_type(self, player_id: int, swing_detection: Dict, 
                           keypoints: Dict[int, List], player_center: List[float],
                           ball_position: Optional[List[int]]) -> str:
        """Classify the type of swing based on motion analysis"""
        
        # If no swing detected, classify based on motion
        if not swing_detection['has_swing']:
            return self._classify_non_swing_motion(player_id, keypoints, player_center)
        
        # Get swing type from detection
        swing_type = swing_detection['swing_type']
        
        # Refine classification based on player orientation and ball position
        if swing_type in ['forehand', 'backhand']:
            refined_type = self._refine_groundstroke_classification(
                player_id, swing_type, keypoints, player_center, ball_position
            )
            return refined_type
        elif swing_type == 'overhead':
            return 'overhead_smash'
        else:
            return swing_type
    
    def _classify_non_swing_motion(self, player_id: int, keypoints: Dict[int, List], 
                                 player_center: List[float]) -> str:
        """Classify motion when no swing is detected"""
        
        # Check if player is moving
        if player_id in self.player_motion_history:
            history = list(self.player_motion_history[player_id])
            if len(history) >= 3:
                recent_motion = history[-3:]
                player_centers = [frame['player_center'] for frame in recent_motion]
                
                # Calculate recent movement
                total_movement = 0
                for i in range(1, len(player_centers)):
                    dx = player_centers[i][0] - player_centers[i-1][0]
                    dy = player_centers[i][1] - player_centers[i-1][1]
                    movement = math.sqrt(dx*dx + dy*dy)
                    total_movement += movement
                
                # More lenient movement thresholds
                threshold = 15.0  # Same threshold for both players
                
                if total_movement > threshold:
                    return 'moving'
        
        return 'ready_stance'
    
    def _refine_groundstroke_classification(self, player_id: int, swing_type: str,
                                          keypoints: Dict[int, List], player_center: List[float],
                                          ball_position: Optional[List[int]]) -> str:
        """Refine forehand/backhand classification based on player orientation"""
        
        # Get right hand position (assuming right-handed players)
        right_wrist = keypoints.get(10)
        right_shoulder = keypoints.get(6)
        
        if not (right_wrist and right_shoulder):
            return swing_type
        
        # Calculate body center
        body_center_x = player_center[0]
        right_hand_x = right_wrist[0]
        
        # Adjust classification based on player orientation
        if player_id == 1:  # Far player
            if right_hand_x < body_center_x - 15:
                return 'backhand'
            else:
                return 'forehand'
        else:  # Near player
            if right_hand_x < body_center_x - 15:
                return 'forehand'
            else:
                return 'backhand'
    
    def _apply_tennis_rules(self, shot_type: str, frame_number: int, player_id: int,
                           ball_position: Optional[List[int]], 
                           player_center: List[float]) -> str:
        """Apply tennis game flow rules"""
        
        # Rule 1: Shot persistence
        if player_id in self.current_shot_type and self.current_shot_type[player_id]:
            if frame_number - self.shot_start_frame.get(player_id, 0) < self.shot_persistence_frames:
                return self.current_shot_type[player_id]
        
        # Rule 2: Ball proximity for shots (more lenient)
        if shot_type in ['forehand', 'backhand', 'overhead_smash']:
            if ball_position and player_center:
                ball_x, ball_y = ball_position
                player_x, player_y = player_center
                distance = math.sqrt((ball_x - player_x)**2 + (ball_y - player_y)**2)
                
                if distance > 300:  # Increased threshold for ball proximity
                    return 'moving'
        
        # Rule 3: Serve only at point start
        if shot_type == 'serve' and frame_number > 50:
            return 'moving'
        
        return shot_type
    
    def _update_shot_state(self, player_id: int, shot_type: str, frame_number: int):
        """Update shot state tracking"""
        if shot_type in ['forehand', 'backhand', 'overhead_smash', 'serve']:
            self.current_shot_type[player_id] = shot_type
            self.shot_start_frame[player_id] = frame_number
            self.last_shot_frame[player_id] = frame_number
        else:
            # Clear shot state for non-shot classifications
            self.current_shot_type[player_id] = None
            self.shot_start_frame[player_id] = None


    


class TennisShotProcessor:
    """Processor for applying shot classification to CSV data and creating videos"""
    
    def __init__(self, use_motion_based: bool = True):
        """Initialize the processor"""
        self.shot_classifier = TennisShotClassifier(use_motion_based=use_motion_based)
        self.use_motion_based = use_motion_based
        
    
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
    
    def process_csv_data(self, csv_file: str, video_file: str, output_file: str = None, show_viewer: bool = False):
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
            
            logger.info(f"Video: {width}x{height} @ {fps}fps")
            logger.info(f"Using {'motion-based' if self.use_motion_based else 'legacy pose-based'} classification")
            
            # Setup video writer only if output file specified
            out = None
            if output_file:
                logger.info(f"Output video: {output_file}")
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out = cv2.VideoWriter(output_file, fourcc, fps, (width, height))
            else:
                logger.info("No output video - viewer only mode")
            
            # Viewer setup
            if show_viewer:
                cv2.namedWindow('Tennis Shot Classification', cv2.WINDOW_NORMAL)
                cv2.resizeWindow('Tennis Shot Classification', 1200, 800)
                logger.info("🎮 Viewer Controls:")
                logger.info("   - Press 'q' to quit")
                logger.info("   - Press 'space' to pause/resume")
                logger.info("   - Press 'left/right' arrows to step frame by frame")
                logger.info("   - Press 's' to save current frame")
                logger.info("   - Press 'r' to reset to beginning")
            
            # Viewer state
            paused = False
            current_frame = 0
            step_mode = False
            
            # Process each frame
            while current_frame < len(df):
                if not paused or step_mode:
                    # Read frame
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    # Parse data from CSV row
                    row = df.iloc[current_frame]
                    poses = self._parse_poses_from_csv(row.get('pose_keypoints', ''))
                    ball_position = self._parse_ball_position_from_csv(row.get('ball_x', ''), row.get('ball_y', ''))
                    court_keypoints = self._parse_court_keypoints_from_csv(row.get('court_keypoints', ''))
                    player_bboxes = self._parse_player_bboxes_from_csv(row.get('player_bboxes', ''))
                    
                    # Apply consistent player tracking
                    # CSV now has consistent ordering, so we can use bboxes directly
                    tracked_bboxes = player_bboxes
                    
                    # Apply shot classification with consistent player IDs
                    shot_types = []
                    for i, bbox in enumerate(tracked_bboxes):
                        if bbox:
                            shot_type = self.shot_classifier.classify_shot(
                                bbox, ball_position, poses, court_keypoints, current_frame, player_id=i
                            )
                            shot_types.append(shot_type)
                        else:
                            shot_types.append("unknown")
                    
                    # Add overlays to frame
                    frame_with_overlays = self._add_overlays(
                        frame, current_frame, tracked_bboxes, shot_types, 
                        ball_position, court_keypoints, poses
                    )
                    
                    # Write frame only if output file specified
                    if out is not None:
                        out.write(frame_with_overlays)
                    
                    # Show in viewer
                    if show_viewer:
                        cv2.imshow('Tennis Shot Classification', frame_with_overlays)
                    
                    # Progress update
                    if current_frame % 30 == 0:
                        logger.info(f"Processed {current_frame}/{len(df)} frames ({current_frame/len(df)*100:.1f}%)")
                    
                    current_frame += 1
                    step_mode = False
                
                if show_viewer:
                    # Handle keyboard input
                    key = cv2.waitKey(30) & 0xFF
                    
                    if key == ord('q'):
                        logger.info("Quitting viewer...")
                        break
                    elif key == ord(' '):  # Space bar
                        paused = not paused
                        logger.info(f"{'Paused' if paused else 'Resumed'}")
                    elif key == 83:  # Right arrow
                        step_mode = True
                        logger.info("Step forward")
                    elif key == 81:  # Left arrow
                        current_frame = max(0, current_frame - 1)
                        cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
                        step_mode = True
                        logger.info(f"Step backward to frame {current_frame}")
                    elif key == ord('s'):
                        # Save current frame
                        save_name = f"frame_{current_frame:04d}.jpg"
                        cv2.imwrite(save_name, frame_with_overlays)
                        logger.info(f"Saved frame {current_frame} as {save_name}")
                    elif key == ord('r'):
                        # Reset to beginning
                        current_frame = 0
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        logger.info("Reset to beginning")
                else:
                    # If no viewer, just process normally
                    pass
            
            # Cleanup
            cap.release()
            if out is not None:
                out.release()
                logger.info(f"Shot classification video created: {output_file}")
            
            if show_viewer:
                cv2.destroyAllWindows()
            
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
        
        # Add frame number and method
        method_text = "Motion-Based" if self.use_motion_based else "Legacy Pose-Based"
        cv2.putText(frame_with_overlays, f"Frame: {frame_count} | {method_text}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Add player bounding boxes and shot types with enhanced debugging
        for i, (bbox, shot_type) in enumerate(zip(player_bboxes, shot_types)):
            if bbox:
                x1, y1, x2, y2 = bbox
                
                # Draw bounding box
                color = (0, 255, 0) if i == 0 else (255, 0, 0)
                cv2.rectangle(frame_with_overlays, (x1, y1), (x2, y2), color, 2)
                
                # Add player ID and shot type with confidence info
                label = f"Player {i}: {shot_type}"
                cv2.putText(frame_with_overlays, label, (x1, y1-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                
                # Add motion analysis info for motion-based method
                if self.use_motion_based and i in self.shot_classifier.player_motion_history:
                    history = list(self.shot_classifier.player_motion_history[i])
                    if len(history) >= 3:
                        # Show recent motion data
                        recent_frames = history[-3:]
                        motion_info = []
                        
                        # Calculate recent velocity
                        if len(recent_frames) >= 2:
                            centers = [f['player_center'] for f in recent_frames]
                            velocities = []
                            for j in range(1, len(centers)):
                                dx = centers[j][0] - centers[j-1][0]
                                dy = centers[j][1] - centers[j-1][1]
                                vel = math.sqrt(dx*dx + dy*dy)
                                velocities.append(vel)
                            
                            if velocities:
                                avg_vel = np.mean(velocities)
                                max_vel = max(velocities)
                                motion_info.append(f"Vel: {avg_vel:.1f}")
                                motion_info.append(f"Max: {max_vel:.1f}")
                        
                        # Show arm motion if available
                        if recent_frames:
                            right_arm = recent_frames[-1]['right_arm']
                            if right_arm['arm_center']:
                                arm_center = right_arm['arm_center']
                                cv2.circle(frame_with_overlays, 
                                          (int(arm_center[0]), int(arm_center[1])), 
                                          3, (255, 255, 0), -1)
                        
                        # Display motion info
                        if motion_info:
                            info_text = " | ".join(motion_info)
                            cv2.putText(frame_with_overlays, info_text, (x1, y1+25), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        # Add ball position with distance info
        if ball_position:
            ball_x, ball_y = ball_position
            cv2.circle(frame_with_overlays, (ball_x, ball_y), 5, (0, 255, 255), -1)
            cv2.putText(frame_with_overlays, "Ball", (ball_x+10, ball_y-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            
            # Show ball distance to each player
            for i, bbox in enumerate(player_bboxes):
                if bbox:
                    player_center_x = (bbox[0] + bbox[2]) / 2
                    player_center_y = (bbox[1] + bbox[3]) / 2
                    distance = math.sqrt((ball_x - player_center_x)**2 + (ball_y - player_center_y)**2)
                    
                    color = (0, 255, 0) if i == 0 else (255, 0, 0)
                    cv2.putText(frame_with_overlays, f"Dist: {distance:.0f}px", 
                               (bbox[0], bbox[3]+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        # Add court keypoints
        for i, (x, y) in enumerate(court_keypoints):
            cv2.circle(frame_with_overlays, (x, y), 3, (255, 255, 0), -1)
            if i < 4:
                cv2.putText(frame_with_overlays, f"C{i}", (x+5, y-5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
        
        # Add pose skeleton for debugging
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
            ]
            
            for connection in connections:
                if connection[0] in pose and connection[1] in pose:
                    pt1 = (int(pose[connection[0]][0]), int(pose[connection[0]][1]))
                    pt2 = (int(pose[connection[1]][0]), int(pose[connection[1]][1]))
                    cv2.line(frame_with_overlays, pt1, pt2, color, 2)
            
            # Draw key arm keypoints
            for keypoint_id, keypoint in pose.items():
                if keypoint_id in [5, 6, 7, 8, 9, 10]:  # Arm keypoints
                    x, y = int(keypoint[0]), int(keypoint[1])
                    cv2.circle(frame_with_overlays, (x, y), 4, color, -1)
        
        return frame_with_overlays


def main():
    """Main function for standalone usage"""
    parser = argparse.ArgumentParser(description='Tennis Shot Classification System')
    parser.add_argument('--csv', required=True, help='Input CSV file with analysis data')
    parser.add_argument('--video', required=True, help='Input video file')
    parser.add_argument('--output', help='Output video file (optional - viewer only if not specified)')
    parser.add_argument('--method', choices=['motion', 'legacy'], default='motion',
                       help='Classification method: motion-based or legacy pose-based')
    parser.add_argument('--viewer', action='store_true', 
                       help='Show real-time viewer for observation and fine-tuning')
    
    args = parser.parse_args()
    
    # Create processor and run analysis
    use_motion_based = (args.method == 'motion')
    processor = TennisShotProcessor(use_motion_based=use_motion_based)
    processor.process_csv_data(args.csv, args.video, args.output, show_viewer=args.viewer)


if __name__ == "__main__":
    main()
