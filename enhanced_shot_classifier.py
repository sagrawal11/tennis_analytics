#!/usr/bin/env python3
"""
Enhanced Tennis Shot Classification System
Provides refined detection of tennis shots: forehand, backhand, overhead smash, ready stance, and moving.
"""

import numpy as np
import logging
from typing import Dict, List, Tuple, Optional
from collections import deque
import math

logger = logging.getLogger(__name__)

class EnhancedShotClassifier:
    """Enhanced tennis shot classification with improved accuracy"""
    
    def __init__(self):
        # Shot classification thresholds
        self.ARM_EXTENSION_THRESHOLD = 0.7  # Arm extension ratio for swing detection
        self.SHOULDER_LEVEL_THRESHOLD = 15  # Pixels for shoulder level detection
        self.HIP_LEVEL_THRESHOLD = 15       # Pixels for hip level detection
        self.MOVEMENT_THRESHOLD = 10        # Pixels for movement detection
        self.SERVE_HEIGHT_THRESHOLD = 100   # Pixels above player for serve detection
        self.OVERHEAD_HEIGHT_THRESHOLD = 80 # Pixels above player for overhead detection
        
        # Shot history for temporal analysis
        self.shot_history = deque(maxlen=10)
        self.pose_history = deque(maxlen=5)
        
        # Tennis game flow state
        self.point_started = False
        self.last_shot_frame = 0
        self.current_frame = 0
        
        # Shot persistence tracking
        self.current_shot_type = {}  # Track current shot type for each player
        self.shot_start_frame = {}  # Track when shot started for each player
        self.shot_persistence_frames = 8  # How many frames to persist a shot classification
        
        # Movement tracking
        self.player_positions = {}  # Track player positions over time
        self.movement_threshold = 25  # Pixels of movement over 15 frames to consider "moving"
        self.movement_window = 15  # Number of frames to analyze for movement (longer window)
        self.player_position_history = {}  # Store position history for each player
        
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
        
        logger.info("Enhanced shot classifier initialized")
    
    def classify_shot(self, player_bbox: List[int], ball_position: Optional[List[int]], 
                     poses: List[Dict], court_keypoints: List[Tuple], 
                     frame_number: int = 0, player_id: int = 0) -> str:
        """
        Enhanced shot classification with multiple analysis methods
        
        Args:
            player_bbox: [x1, y1, x2, y2] player bounding box
            ball_position: [x, y] ball position (can be None)
            poses: List of pose dictionaries with keypoints
            court_keypoints: List of court keypoint tuples
            player_speed: Player movement speed in pixels/frame
            
        Returns:
            Shot type string
        """
        try:
            # Get player center
            player_center_x = (player_bbox[0] + player_bbox[2]) / 2
            player_center_y = (player_bbox[1] + player_bbox[3]) / 2
            
            # Find closest pose to player
            closest_pose = self._find_closest_pose(poses, player_center_x, player_center_y)
            if not closest_pose:
                return "ready_stance"
            
            # Extract confident keypoints
            confident_keypoints = self._extract_confident_keypoints(closest_pose)
            if len(confident_keypoints) < 8:
                return "ready_stance"
            
            # Multi-stage classification
            shot_type = self._classify_shot_multi_stage(
                confident_keypoints, player_center_x, player_center_y,
                ball_position, court_keypoints, player_bbox, player_id
            )
            
            # Add to history for temporal analysis
            self.shot_history.append(shot_type)
            self.pose_history.append(confident_keypoints)
            
            # Apply temporal smoothing
            smoothed_shot_type = self._apply_temporal_smoothing(shot_type)
            
            # Apply tennis game flow rules
            player_center = [player_center_x, player_center_y]
            final_shot_type = self._enforce_tennis_game_flow(smoothed_shot_type, frame_number, ball_position, player_center, player_id)
            
            logger.debug(f"🎾 Shot classification: {shot_type} -> {smoothed_shot_type} -> {final_shot_type}")
            return final_shot_type
            
        except Exception as e:
            logger.error(f"Error in shot classification: {e}")
            return "ready_stance"
    
    def _find_closest_pose(self, poses: List[Dict], player_x: float, player_y: float) -> Optional[Dict]:
        """Find the pose closest to the player center"""
        if not poses:
            return None
        
        closest_pose = None
        min_distance = float('inf')
        
        for pose in poses:
            if 'keypoints' in pose and len(pose['keypoints']) > 0:
                # Use hip keypoints (11, 12) to find pose center
                hip_keypoints = []
                confidence = pose.get('confidence', [])
                
                for i in [11, 12]:  # Left and right hip
                    if i < len(pose['keypoints']) and i < len(confidence) and confidence[i] > 0.3:
                        hip_keypoints.append(pose['keypoints'][i])
                
                if len(hip_keypoints) >= 2:
                    pose_center_x = sum(kp[0] for kp in hip_keypoints) / len(hip_keypoints)
                    pose_center_y = sum(kp[1] for kp in hip_keypoints) / len(hip_keypoints)
                    distance = np.sqrt((pose_center_x - player_x)**2 + (pose_center_y - player_y)**2)
                    
                    if distance < min_distance:
                        min_distance = distance
                        closest_pose = pose
        
        return closest_pose
    
    def _extract_confident_keypoints(self, pose: Dict) -> Dict[int, List]:
        """Extract keypoints with confidence > 0.3"""
        keypoints = pose['keypoints']
        confidence = pose.get('confidence', [])
        
        confident_keypoints = {}
        for i, kp in enumerate(keypoints):
            if i < len(confidence) and confidence[i] > 0.3:
                confident_keypoints[i] = kp
        
        return confident_keypoints
    
    def _classify_shot_multi_stage(self, keypoints: Dict[int, List], player_x: float, 
                                 player_y: float, ball_position: Optional[List[int]], 
                                 court_keypoints: List[Tuple], player_bbox: List[int], 
                                 player_id: int = 0) -> str:
        """Multi-stage shot classification"""
        
        # Stage 1: Check for serve
        if self._is_serve(keypoints, player_y, court_keypoints, ball_position):
            return "serve"
        
        # Stage 2: Check for overhead smash
        if self._is_overhead_smash(keypoints, player_y, court_keypoints, ball_position):
            return "overhead_smash"
        
        # Stage 3: Check for groundstrokes (forehand/backhand) - PRIORITY OVER MOVEMENT
        groundstroke = self._classify_groundstroke(keypoints, player_x, ball_position, player_id)
        if groundstroke != "ready_stance":
            return groundstroke
        
        # Stage 4: Check for movement (only if not hitting a shot)
        if self._is_moving(player_bbox, player_id):
            return "moving"
        
        # Stage 5: Default to ready stance
        
        # Stage 6: Default to ready stance
        return "ready_stance"
    
    def _is_serve(self, keypoints: Dict[int, List], player_y: float, 
                 court_keypoints: List[Tuple], ball_position: Optional[List[int]]) -> bool:
        """Enhanced serve detection"""
        try:
            # Check if player is near baseline
            if not self._is_near_baseline(player_y, court_keypoints):
                return False
            
            # Check for serve pose characteristics
            left_shoulder = keypoints.get(5)
            right_shoulder = keypoints.get(6)
            left_elbow = keypoints.get(7)
            right_elbow = keypoints.get(8)
            left_wrist = keypoints.get(9)
            right_wrist = keypoints.get(10)
            
            # Serve typically has one arm raised high above shoulder
            if right_shoulder and right_wrist:
                # Check if right arm is raised significantly above shoulder
                arm_raised = right_wrist[1] < right_shoulder[1] - 20  # More lenient threshold
                if arm_raised:
                    return True
            
            if left_shoulder and left_wrist:
                # Check if left arm is raised significantly above shoulder
                arm_raised = left_wrist[1] < left_shoulder[1] - 20  # More lenient threshold
                if arm_raised:
                    return True
            
            # Check ball position for serve (if available) - more lenient
            if ball_position:
                ball_x, ball_y = ball_position
                # Ball should be high above player for serve (more lenient threshold)
                if ball_y < player_y - 50:  # Lower threshold since video starts at peak
                    return True
            
            return False
            
        except Exception as e:
            logger.debug(f"Error in serve detection: {e}")
            return False
    
    def _is_overhead_smash(self, keypoints: Dict[int, List], player_y: float, 
                          court_keypoints: List[Tuple], ball_position: Optional[List[int]]) -> bool:
        """Enhanced overhead smash detection"""
        try:
            # Check if player is near net (NOT near baseline)
            if self._is_near_baseline(player_y, court_keypoints):
                return False
            
            # Check ball position first
            if ball_position:
                ball_x, ball_y = ball_position
                if ball_y < player_y - self.OVERHEAD_HEIGHT_THRESHOLD:
                    # Ball is high - check for overhead pose
                    left_shoulder = keypoints.get(5)
                    right_shoulder = keypoints.get(6)
                    left_wrist = keypoints.get(9)
                    right_wrist = keypoints.get(10)
                    
                    # Overhead typically has one arm raised high above shoulder
                    if right_shoulder and right_wrist:
                        # Check if right arm is raised significantly above shoulder
                        arm_raised = right_wrist[1] < right_shoulder[1] - 30
                        if arm_raised:
                            return True
                    
                    if left_shoulder and left_wrist:
                        # Check if left arm is raised significantly above shoulder
                        arm_raised = left_wrist[1] < left_shoulder[1] - 30
                        if arm_raised:
                            return True
            
            return False
            
        except Exception as e:
            logger.debug(f"Error in overhead detection: {e}")
            return False
    

    
    def _classify_groundstroke(self, keypoints: Dict[int, List], player_x: float, 
                             ball_position: Optional[List[int]], player_id: int = 0) -> str:
        """Enhanced groundstroke classification based on right hand position (accounting for player orientation)"""
        try:
            # Focus on right hand position since all players are assumed to be right-handed
            right_shoulder = keypoints.get(6)
            right_elbow = keypoints.get(8)
            right_wrist = keypoints.get(10)
            
            # Need reliable right arm keypoints
            if not all([right_shoulder, right_elbow, right_wrist]):
                return "ready_stance"
            
            # Check if player is in a swing position
            if not self._is_arm_extended(right_shoulder, right_elbow, right_wrist):
                return "ready_stance"
            
            # Get player body center (use shoulders for reference)
            left_shoulder = keypoints.get(5)
            if not left_shoulder:
                # Use player_x as fallback
                body_center_x = player_x
            else:
                # Calculate body center from shoulders
                body_center_x = (left_shoulder[0] + right_shoulder[0]) / 2
            
            # Determine shot type based on right hand position relative to body
            right_hand_x = right_wrist[0]
            
            # Account for player orientation:
            # Player 0 (bottom): Facing away from camera - normal orientation
            # Player 1 (top): Facing toward camera - flipped orientation
            if player_id == 1:
                # Player facing camera: right hand on left side of video = forehand
                # Right hand on right side of video = backhand
                if right_hand_x < body_center_x - 15:  # Right hand on left side of video
                    return "forehand"
                else:  # Right hand on right side of video
                    return "backhand"
            else:
                # Player facing away from camera: normal orientation
                # Right hand on right side of body = forehand
                # Right hand crossed to left side = backhand
                if right_hand_x < body_center_x - 15:  # Right hand crossed over to left side
                    return "backhand"
                else:  # Right hand on natural right side
                    return "forehand"
            
        except Exception as e:
            logger.debug(f"Error in groundstroke classification: {e}")
            return "ready_stance"
    
    def _is_arm_extended(self, shoulder: Optional[List], elbow: Optional[List], 
                        wrist: Optional[List]) -> bool:
        """Check if arm is extended for swing"""
        if not all([shoulder, elbow, wrist]):
            return False
        
        # Calculate arm segments
        upper_arm_length = np.sqrt((elbow[0] - shoulder[0])**2 + (elbow[1] - shoulder[1])**2)
        forearm_length = np.sqrt((wrist[0] - elbow[0])**2 + (wrist[1] - elbow[1])**2)
        
        # Arm is extended if forearm is significantly longer than upper arm
        if upper_arm_length > 0:
            extension_ratio = forearm_length / upper_arm_length
            return extension_ratio > self.ARM_EXTENSION_THRESHOLD
        
        return False
    
    def _is_arm_in_swing_position(self, shoulder: Optional[List], elbow: Optional[List], 
                                 wrist: Optional[List], is_right_arm: bool = True) -> bool:
        """Check if arm is in a swing position for forehand/backhand"""
        if not all([shoulder, elbow, wrist]):
            return False
        
        # Check if arm is extended
        if not self._is_arm_extended(shoulder, elbow, wrist):
            return False
        
        # For tennis shots, we want to see the arm extended out to the side
        # Check if wrist is positioned away from the body (extended laterally)
        shoulder_x = shoulder[0]
        wrist_x = wrist[0]
        
        if is_right_arm:
            # Right arm should be extended to the right (wrist_x > shoulder_x)
            return wrist_x > shoulder_x + 20  # At least 20 pixels to the right
        else:
            # Left arm should be extended to the left (wrist_x < shoulder_x)
            return wrist_x < shoulder_x - 20  # At least 20 pixels to the left
    
    def _is_moving(self, player_bbox: List[int], player_id: int = 0) -> bool:
        """Detect if player is moving based on bounding box changes over 15 frames"""
        try:
            # Get current player center
            x1, y1, x2, y2 = player_bbox
            current_center_x = (x1 + x2) / 2
            current_center_y = (y1 + y2) / 2
            
            # Initialize position history for this player if needed
            if player_id not in self.player_position_history:
                self.player_position_history[player_id] = []
            
            # Add current position to history
            self.player_position_history[player_id].append((current_center_x, current_center_y))
            
            # Keep only the last movement_window frames
            if len(self.player_position_history[player_id]) > self.movement_window:
                self.player_position_history[player_id] = self.player_position_history[player_id][-self.movement_window:]
            
            # Need at least 2 positions to calculate movement
            if len(self.player_position_history[player_id]) < 2:
                return True  # Default to moving until we have enough data
            
            # Calculate movement from first to last position in window
            if len(self.player_position_history[player_id]) >= self.movement_window:
                positions = self.player_position_history[player_id]
                first_x, first_y = positions[0]
                last_x, last_y = positions[-1]
                window_movement = np.sqrt((last_x - first_x)**2 + (last_y - first_y)**2)
                
                # Adjust threshold based on player distance (further players move fewer pixels)
                # Player 0 (bottom): closer to camera, normal threshold
                # Player 1 (top): further from camera, lower threshold
                if player_id == 1:
                    # Further player - more sensitive to movement
                    adjusted_threshold = self.movement_threshold * 0.6  # 60% of normal threshold
                else:
                    # Closer player - normal threshold
                    adjusted_threshold = self.movement_threshold
                
                # Player is moving if they've moved significantly over the window
                is_moving = window_movement > adjusted_threshold
                
                logger.debug(f"Player {player_id}: window_movement={window_movement:.1f}, threshold={adjusted_threshold:.1f}, moving={is_moving}")
                return is_moving
            
            # Default to moving if we don't have enough frames yet
            # In tennis, players are usually moving more than stationary
            return True
            
        except Exception as e:
            logger.debug(f"Error in movement detection: {e}")
            return True  # Default to moving on error
    
    def _is_near_baseline(self, player_y: float, court_keypoints: List[Tuple]) -> bool:
        """Check if player is near baseline"""
        try:
            if not court_keypoints or len(court_keypoints) < 4:
                return False
            
            # Find baseline y-coordinate (highest y value)
            baseline_y = max(kp[1] for kp in court_keypoints if kp[1] is not None)
            
            # Check if player is close to baseline (within 50 pixels)
            distance_to_baseline = abs(player_y - baseline_y)
            return distance_to_baseline < 50
            
        except Exception as e:
            logger.debug(f"Error checking baseline proximity: {e}")
            return False
    

    
    def _apply_temporal_smoothing(self, current_shot: str) -> str:
        """Apply temporal smoothing to reduce shot type flickering"""
        try:
            if len(self.shot_history) < 3:
                return current_shot
            
            # Get recent shot types
            recent_shots = list(self.shot_history)[-3:]
            
            # If all recent shots are the same, use that
            if len(set(recent_shots)) == 1:
                return recent_shots[0]
            
            # If current shot is different from previous, check if it's a brief change
            if current_shot != recent_shots[-2]:
                # Count occurrences of each shot type
                shot_counts = {}
                for shot in recent_shots:
                    shot_counts[shot] = shot_counts.get(shot, 0) + 1
                
                # Return the most common shot type
                most_common = max(shot_counts.items(), key=lambda x: x[1])[0]
                return most_common
            
            return current_shot
            
        except Exception as e:
            logger.debug(f"Error in temporal smoothing: {e}")
            return current_shot
    
    def _enforce_tennis_game_flow(self, shot_type: str, frame_number: int, 
                                ball_position: Optional[List[int]] = None, 
                                player_center: Optional[List[float]] = None, 
                                player_id: int = 0) -> str:
        """Enforce tennis game flow rules with ball proximity and shot persistence"""
        try:
            # Update frame counter
            self.current_frame = frame_number
            
            # Ball proximity check
            ball_near_player = False
            if ball_position and player_center:
                ball_x, ball_y = ball_position
                player_x, player_y = player_center
                
                # Calculate distance between ball and player
                distance = np.sqrt((ball_x - player_x)**2 + (ball_y - player_y)**2)
                ball_near_player = distance < 200  # Within 200 pixels (more restrictive)
            
            # Rule 1: Serve can only happen at the beginning of a point
            if shot_type == "serve":
                if self.point_started:
                    # Point already started, can't serve again
                    return "ready_stance"
                else:
                    # Start a new point
                    self.point_started = True
                    self.last_shot_frame = frame_number
                    return "serve"
            
            # Rule 2: Shot persistence - if we're in the middle of a shot, continue it
            if player_id in self.current_shot_type and player_id in self.shot_start_frame:
                current_shot = self.current_shot_type[player_id]
                shot_start = self.shot_start_frame[player_id]
                frames_in_shot = frame_number - shot_start
                
                # If we're in a shot and it hasn't been too long, persist the shot
                if current_shot in ["forehand", "backhand", "overhead_smash"] and frames_in_shot < self.shot_persistence_frames:
                    return current_shot
            
            # Rule 3: Only classify shots when ball is near player
            if not ball_near_player:
                # Ball is far away - can't be hitting a shot
                if shot_type in ["forehand", "backhand", "overhead_smash"]:
                    # Clear any ongoing shot
                    if player_id in self.current_shot_type:
                        del self.current_shot_type[player_id]
                    if player_id in self.shot_start_frame:
                        del self.shot_start_frame[player_id]
                    return "ready_stance"
            
            # Rule 4: Start a new shot when ball is near and pose indicates hitting
            if ball_near_player and shot_type in ["forehand", "backhand", "overhead_smash"]:
                # Start tracking this shot
                self.current_shot_type[player_id] = shot_type
                self.shot_start_frame[player_id] = frame_number
                self.last_shot_frame = frame_number
                return shot_type
            
            # Rule 5: Strokes must be separated by ready stance or movement
            if shot_type in ["forehand", "backhand", "overhead_smash"]:
                # Check if we had a stroke recently without transition
                if len(self.shot_history) > 0:
                    last_shot = self.shot_history[-1]
                    frames_since_last_shot = frame_number - self.last_shot_frame
                    
                    # If last shot was also a stroke and not enough time passed, force transition
                    if last_shot in ["forehand", "backhand", "overhead_smash"] and frames_since_last_shot < 10:
                        return "ready_stance"
                
                # Valid stroke, update last shot frame
                self.last_shot_frame = frame_number
                return shot_type
            
            # Rule 6: Ready stance or movement can happen anytime
            if shot_type in ["ready_stance", "moving"]:
                # Clear any ongoing shot when transitioning to non-shot states
                if player_id in self.current_shot_type:
                    del self.current_shot_type[player_id]
                if player_id in self.shot_start_frame:
                    del self.shot_start_frame[player_id]
                return shot_type
            
            # Default case
            return "ready_stance"
            
        except Exception as e:
            logger.debug(f"Error in tennis game flow enforcement: {e}")
            return shot_type
    
    def get_shot_statistics(self) -> Dict[str, int]:
        """Get shot type statistics from history"""
        shot_counts = {}
        for shot_type in self.shot_history:
            shot_counts[shot_type] = shot_counts.get(shot_type, 0) + 1
        return shot_counts
    
    def reset_history(self):
        """Reset shot and pose history"""
        self.shot_history.clear()
        self.pose_history.clear()

# Example usage and testing
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.DEBUG)
    
    # Create classifier
    classifier = EnhancedShotClassifier()
    
    print("Enhanced Shot Classifier Test")
    print("=" * 40)
    print("Available shot types:")
    for shot_type, description in classifier.shot_types.items():
        print(f"  - {shot_type}: {description}")
    print("=" * 40)
