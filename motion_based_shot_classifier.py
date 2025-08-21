#!/usr/bin/env python3
"""
Motion-Based Tennis Shot Classifier

This classifier focuses on detecting actual tennis swing motion rather than static poses.
It analyzes arm velocity, acceleration, swing patterns, and tennis mechanics to identify
shots at the correct timing.
"""

import numpy as np
import logging
from typing import Dict, List, Optional, Tuple, Deque
from collections import deque
import math

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

class MotionBasedShotClassifier:
    def __init__(self):
        """Initialize motion-based shot classifier"""
        
        # Motion analysis parameters
        self.motion_history_length = 20  # Frames to track for motion analysis
        self.swing_detection_window = 8   # Frames to analyze for swing detection
        self.velocity_threshold = 25.0    # Minimum velocity for swing detection (increased)
        self.acceleration_threshold = 15.0  # Minimum acceleration for swing detection (increased)
        
        # Tennis-specific parameters
        self.swing_duration_frames = 6     # Expected duration of a tennis swing
        self.backswing_ratio = 0.4        # Backswing should be ~40% of total swing
        self.contact_zone_threshold = 50   # Pixels around expected contact point
        
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
        
        # Motion tracking
        self.player_motion_history = {}  # {player_id: deque of motion data}
        self.swing_sequences = {}        # {player_id: current swing sequence}
        self.last_shot_frame = {}        # {player_id: last frame with shot}
        
        # Shot detection state
        self.current_shot_type = {}
        self.shot_start_frame = {}
        self.shot_persistence_frames = 8
        
        logger.info("Motion-based shot classifier initialized")
    
    def classify_shot(self, player_bbox: List[int], ball_position: Optional[List[int]], 
                     poses: List[Dict], court_keypoints: List[Tuple], 
                     frame_number: int = 0, player_id: int = 0) -> str:
        """
        Motion-based shot classification using swing analysis
        
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
            
            logger.info(f"🎾 Player {player_id} Frame {frame_number}: {final_shot_type}")
            return final_shot_type
            
        except Exception as e:
            logger.error(f"Error in motion-based classification: {e}")
            return "unknown"
    
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
        
        # Detect swing-like motion (more restrictive)
        has_swing_motion = (max_velocity > self.velocity_threshold and 
                           max_acceleration > self.acceleration_threshold and
                           len(velocities) >= 4)  # Require minimum motion history
        
        # Analyze motion direction
        motion_direction = self._analyze_motion_direction(arm_centers)
        
        return {
            'has_motion': max_velocity > 5.0,  # Basic motion threshold
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
        
        # Check if there's a swing motion (more restrictive)
        if not (right_arm['has_swing_motion'] or left_arm['has_swing_motion']):
            return characteristics
        
        # Additional check: require significant motion
        if right_arm['max_velocity'] < 30.0 and left_arm['max_velocity'] < 30.0:
            return characteristics
        
        characteristics['has_swing'] = True
        
        # Analyze swing type based on motion patterns
        if right_arm['has_swing_motion']:
            # Primary swing arm analysis
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
            # Check for acceleration-deceleration pattern (typical of tennis swing)
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
            swing_detection['ball_proximity'] = distance < 200  # Ball near player
        
        # Update swing sequence tracking
        if player_id not in self.swing_sequences:
            self.swing_sequences[player_id] = {
                'start_frame': None,
                'swing_type': None,
                'phases': []
            }
        
        # Track swing sequence
        if swing_detection['has_swing']:
            if self.swing_sequences[player_id]['start_frame'] is None:
                self.swing_sequences[player_id]['start_frame'] = len(self.player_motion_history[player_id])
                self.swing_sequences[player_id]['swing_type'] = swing_detection['swing_type']
            
            self.swing_sequences[player_id]['phases'].append(swing_detection['swing_phase'])
        else:
            # Reset swing sequence if no swing detected
            self.swing_sequences[player_id] = {
                'start_frame': None,
                'swing_type': None,
                'phases': []
            }
        
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
                
                # Different thresholds for different players
                threshold = 25.0 if player_id == 0 else 2.5
                
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
            # Far player: right hand left of center = backhand, right of center = forehand
            if right_hand_x < body_center_x - 15:
                return 'backhand'
            else:
                return 'forehand'
        else:  # Near player
            # Near player: right hand left of center = forehand, right of center = backhand
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
        
        # Rule 2: Ball proximity for shots
        if shot_type in ['forehand', 'backhand', 'overhead_smash']:
            if ball_position and player_center:
                ball_x, ball_y = ball_position
                player_x, player_y = player_center
                distance = math.sqrt((ball_x - player_x)**2 + (ball_y - player_y)**2)
                
                if distance > 200:  # Ball too far from player
                    return 'moving'
        
        # Rule 3: Serve only at point start (simplified)
        if shot_type == 'serve' and frame_number > 50:  # Assume serve happens early
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
