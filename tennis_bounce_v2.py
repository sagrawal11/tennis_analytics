#!/usr/bin/env python3
"""
Tennis Ball Bounce Detection System V2
Physics-based approach with multiple validation methods
"""

import cv2
import numpy as np
import pandas as pd
import argparse
import logging
from typing import List, Tuple, Optional, Dict
from collections import deque
import os
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
import math

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')
logger = logging.getLogger(__name__)


class PhysicsBounceDetector:
    """Physics-based tennis ball bounce detection system"""
    
    def __init__(self):
        """Initialize bounce detector with physics-based parameters"""
        self.ball_trajectory = deque(maxlen=100)  # Store ball positions
        self.court_homography = None
        self.court_keypoints = None
        self.ground_plane_height = None
        
        # Physics parameters
        self.min_trajectory_length = 10  # Minimum frames for analysis
        self.velocity_window = 5  # Frames to calculate velocity
        self.acceleration_window = 3  # Frames to calculate acceleration
        self.bounce_validation_window = 15  # Frames around potential bounce
        
        # Detection thresholds - More sensitive
        self.velocity_change_threshold = 0.3  # Lower threshold for velocity changes
        self.height_tolerance = 15.0  # More lenient tolerance for ground level
        self.min_bounce_gap_frames = 20  # Shorter gap between bounces
        self.min_confidence_threshold = 0.5  # Lower confidence requirement
        
        # Bounce history
        self.detected_bounces = []
        self.last_bounce_frame = -1
        
        logger.info("Physics-based bounce detector initialized")
    
    def set_court_data(self, court_keypoints: List[Tuple], homography_matrix: np.ndarray = None):
        """Set court keypoints and homography for ground plane detection"""
        self.court_keypoints = court_keypoints
        self.court_homography = homography_matrix
        self._calculate_ground_plane()
        logger.info(f"Court data set with {len(court_keypoints)} keypoints")
    
    def _calculate_ground_plane(self):
        """Calculate ground plane height from court keypoints"""
        if not self.court_keypoints:
            return
        
        # Use court corners to estimate ground plane
        # For high-up camera, ground plane is roughly at the bottom of the court
        valid_points = [p for p in self.court_keypoints if p[0] is not None and p[1] is not None]
        
        if len(valid_points) >= 4:
            # Use bottom court corners to estimate ground level
            y_coords = [p[1] for p in valid_points]
            self.ground_plane_height = max(y_coords)  # Bottom of court = ground level
            
            # Increase tolerance since we're being too strict
            self.height_tolerance = 25.0  # Much more lenient
            
            logger.info(f"Ground plane height estimated at y={self.ground_plane_height:.1f}")
            logger.info(f"Height tolerance set to {self.height_tolerance} pixels")
        else:
            logger.warning("Not enough court keypoints for ground plane calculation")
    
    def add_ball_position(self, ball_x: Optional[float], ball_y: Optional[float], frame_number: int):
        """Add ball position to trajectory history"""
        if ball_x is not None and ball_y is not None:
            self.ball_trajectory.append({
                'x': ball_x,
                'y': ball_y,
                'frame': frame_number,
                'valid': True
            })
        else:
            # Add placeholder for missing data
            self.ball_trajectory.append({
                'x': None,
                'y': None,
                'frame': frame_number,
                'valid': False
            })
    
    def detect_bounce(self, current_frame: int) -> Tuple[bool, float, Dict]:
        """
        Detect ball bounce using multiple physics-based methods
        
        Returns:
            (is_bounce, confidence, bounce_info)
        """
        if len(self.ball_trajectory) < self.min_trajectory_length:
            return False, 0.0, {}
        
        # Check minimum time gap between bounces
        if current_frame - self.last_bounce_frame < self.min_bounce_gap_frames:
            return False, 0.0, {}
        
        # Get valid trajectory points
        valid_points = [p for p in self.ball_trajectory if p['valid']]
        if len(valid_points) < self.min_trajectory_length:
            return False, 0.0, {}
        
        # Method 1: Height-based detection (ball touching ground)
        height_bounce, height_confidence = self._detect_height_bounce(valid_points, current_frame)
        
        # Method 2: Velocity change detection
        velocity_bounce, velocity_confidence = self._detect_velocity_bounce(valid_points, current_frame)
        
        # Method 3: Parabolic trajectory analysis
        parabola_bounce, parabola_confidence = self._detect_parabola_bounce(valid_points, current_frame)
        
        # Method 4: Multi-frame trajectory analysis
        trajectory_bounce, trajectory_confidence = self._detect_trajectory_bounce(valid_points, current_frame)
        
        # Combine all methods with weighted confidence (reduced height weight)
        methods = [
            (height_bounce, height_confidence, 0.1),      # Reduced weight - ball touching ground
            (velocity_bounce, velocity_confidence, 0.4),  # Most important - physics validation
            (parabola_bounce, parabola_confidence, 0.3),  # Trajectory validation
            (trajectory_bounce, trajectory_confidence, 0.2)  # Additional validation
        ]
        
        # Calculate combined confidence
        total_confidence = 0.0
        total_weight = 0.0
        bounce_count = 0
        
        for is_bounce, confidence, weight in methods:
            if is_bounce:
                bounce_count += 1
                total_confidence += confidence * weight
            total_weight += weight
        
        # Require at least 1 method to agree AND very low confidence (very sensitive)
        combined_confidence = total_confidence / total_weight if total_weight > 0 else 0.0
        is_bounce = (bounce_count >= 1 and 
                    combined_confidence > 0.1)  # Very low threshold for maximum sensitivity
        
        # Debug logging for potential bounces
        if bounce_count >= 1 or current_frame < 50:
            logger.info(f"Frame {current_frame}: Methods={bounce_count}/4, Conf={combined_confidence:.3f}, "
                       f"H={height_bounce}({height_confidence:.2f}), V={velocity_bounce}({velocity_confidence:.2f}), "
                       f"P={parabola_bounce}({parabola_confidence:.2f}), T={trajectory_bounce}({trajectory_confidence:.2f})")
        
        bounce_info = {
            'height_method': height_bounce,
            'velocity_method': velocity_bounce,
            'parabola_method': parabola_bounce,
            'trajectory_method': trajectory_bounce,
            'height_confidence': height_confidence,
            'velocity_confidence': velocity_confidence,
            'parabola_confidence': parabola_confidence,
            'trajectory_confidence': trajectory_confidence,
            'combined_confidence': combined_confidence,
            'method_agreement': bounce_count
        }
        
        if is_bounce:
            self.last_bounce_frame = current_frame
            self.detected_bounces.append({
                'frame': current_frame,
                'confidence': combined_confidence,
                'methods': bounce_count,
                'info': bounce_info
            })
            logger.info(f"BOUNCE DETECTED at frame {current_frame} - Methods: {bounce_count}/4, Confidence: {combined_confidence:.3f}")
        
        return is_bounce, combined_confidence, bounce_info
    
    def _detect_height_bounce(self, valid_points: List[Dict], current_frame: int) -> Tuple[bool, float]:
        """Detect bounce based on ball reaching minimum height (ground contact) - OVERHEAD PERSPECTIVE"""
        if not self.ground_plane_height:
            return False, 0.0
        
        # Look for ball reaching minimum height (ground level)
        recent_points = valid_points[-self.bounce_validation_window:]
        if len(recent_points) < 5:
            return False, 0.0
        
        # From overhead perspective, Y-coordinate represents depth on court
        # Find the point with the LOWEST y-coordinate (closest to back baseline)
        min_y = float('inf')
        bounce_frame = None
        bounce_point = None
        
        for point in recent_points:
            if point['y'] < min_y:
                min_y = point['y']
                bounce_frame = point['frame']
                bounce_point = point
        
        # Check if this minimum point is close to ground level (back baseline)
        distance_to_ground = abs(min_y - self.ground_plane_height)
        if distance_to_ground > self.height_tolerance:
            return False, 0.0
        
        # Must be recent (within 3 frames)
        if abs(bounce_frame - current_frame) > 3:
            return False, 0.0
        
        # Additional validation: check if ball was going towards back baseline before and away after
        if bounce_point:
            bounce_idx = recent_points.index(bounce_point)
            
            # Check trajectory before bounce (should be going towards back baseline - decreasing Y)
            going_towards_back = False
            if bounce_idx >= 2:
                y_before = recent_points[bounce_idx - 2]['y']
                y_at_bounce = bounce_point['y']
                going_towards_back = y_before > y_at_bounce  # Y decreases towards back baseline
            
            # Check trajectory after bounce (should be going away from back baseline - increasing Y)
            going_away_from_back = False
            if bounce_idx < len(recent_points) - 2:
                y_at_bounce = bounce_point['y']
                y_after = recent_points[bounce_idx + 2]['y']
                going_away_from_back = y_after > y_at_bounce  # Y increases away from back baseline
            
            # Only detect bounce if trajectory shows proper bounce pattern
            if not (going_towards_back and going_away_from_back):
                return False, 0.0
        
        # Calculate confidence based on distance to ground and trajectory validation
        confidence = max(0.0, 1.0 - (distance_to_ground / self.height_tolerance))
        
        return True, confidence
    
    def _detect_velocity_bounce(self, valid_points: List[Dict], current_frame: int) -> Tuple[bool, float]:
        """Detect bounce based on velocity changes at ground contact (not peaks)"""
        if len(valid_points) < self.velocity_window * 2:
            return False, 0.0
        
        # Calculate velocities
        velocities = []
        for i in range(self.velocity_window, len(valid_points)):
            prev_point = valid_points[i - self.velocity_window]
            curr_point = valid_points[i]
            
            dx = curr_point['x'] - prev_point['x']
            dy = curr_point['y'] - prev_point['y']
            dt = curr_point['frame'] - prev_point['frame']
            
            if dt > 0:
                velocity = math.sqrt(dx*dx + dy*dy) / dt
                velocities.append({
                    'velocity': velocity,
                    'frame': curr_point['frame'],
                    'dx': dx,
                    'dy': dy,
                    'x': curr_point['x'],
                    'y': curr_point['y']
                })
        
        if len(velocities) < 5:
            return False, 0.0
        
        # Find the point with minimum y-coordinate (ground contact)
        min_y = float('inf')
        ground_contact_idx = -1
        
        for i, vel in enumerate(velocities):
            if vel['y'] < min_y:
                min_y = vel['y']
                ground_contact_idx = i
        
        if ground_contact_idx == -1 or ground_contact_idx < 2 or ground_contact_idx >= len(velocities) - 2:
            return False, 0.0
        
        # Check velocity changes around the ground contact point
        vel_before = velocities[ground_contact_idx - 1]['velocity']
        vel_at_contact = velocities[ground_contact_idx]['velocity']
        vel_after = velocities[ground_contact_idx + 1]['velocity']
        
        # At ground contact, we expect:
        # 1. Significant velocity change (deceleration on impact)
        # 2. Ball was going down before (negative dy)
        # 3. Ball is going up after (positive dy)
        
        # Check if ball was going down before contact
        dy_before = velocities[ground_contact_idx - 1]['dy']
        dy_after = velocities[ground_contact_idx + 1]['dy']
        
        going_down_before = dy_before > 0  # Positive dy means going down in image coordinates
        going_up_after = dy_after < 0      # Negative dy means going up in image coordinates
        
        if not (going_down_before and going_up_after):
            return False, 0.0
        
        # Check for significant velocity change at ground contact
        if vel_before > 0:
            velocity_change = abs(vel_at_contact - vel_before) / vel_before
            
            if velocity_change > self.velocity_change_threshold:
                # Check if this is close to current frame
                if abs(velocities[ground_contact_idx]['frame'] - current_frame) <= 3:
                    confidence = min(1.0, velocity_change / self.velocity_change_threshold)
                    return True, confidence
        
        return False, 0.0
    
    def _detect_parabola_bounce(self, valid_points: List[Dict], current_frame: int) -> Tuple[bool, float]:
        """Detect bounce using parabolic trajectory fitting - find minimum point"""
        if len(valid_points) < 10:
            return False, 0.0
        
        # Use recent trajectory for parabola fitting
        recent_points = valid_points[-20:]  # Last 20 valid points
        if len(recent_points) < 8:
            return False, 0.0
        
        try:
            # Extract x, y coordinates and frame numbers
            x_coords = np.array([p['x'] for p in recent_points])
            y_coords = np.array([p['y'] for p in recent_points])
            frames = np.array([p['frame'] for p in recent_points])
            
            # Fit parabola to y vs frame (height over time)
            def parabola_func(t, a, b, c):
                return a * t**2 + b * t + c
            
            # Normalize frame numbers for better fitting
            frame_offset = frames[0]
            t_normalized = frames - frame_offset
            
            popt, pcov = curve_fit(parabola_func, t_normalized, y_coords)
            
            # Find minimum of parabola (bounce point)
            # For parabola ax^2 + bx + c, minimum is at x = -b/(2a)
            if abs(popt[0]) > 1e-10:  # Avoid division by zero
                min_frame_offset = -popt[1] / (2 * popt[0])
                min_frame = min_frame_offset + frame_offset
                
                # Check if minimum is within recent frames and close to current
                if (min_frame >= recent_points[0]['frame'] and 
                    min_frame <= recent_points[-1]['frame'] and
                    abs(min_frame - current_frame) <= 5):
                    
                    # Additional validation: check if this minimum point is actually the lowest point
                    min_y_actual = min(y_coords)
                    min_y_predicted = parabola_func(min_frame_offset, *popt)
                    
                    # The predicted minimum should be close to the actual minimum
                    if abs(min_y_predicted - min_y_actual) < 10:  # Within 10 pixels
                        # Calculate confidence based on fit quality
                        y_pred = parabola_func(t_normalized, *popt)
                        r_squared = 1 - (np.sum((y_coords - y_pred)**2) / np.sum((y_coords - np.mean(y_coords))**2))
                        confidence = max(0.0, r_squared)
                        
                        return True, confidence
            
        except Exception as e:
            logger.debug(f"Parabola fitting failed: {e}")
        
        return False, 0.0
    
    def _detect_trajectory_bounce(self, valid_points: List[Dict], current_frame: int) -> Tuple[bool, float]:
        """Detect bounce using trajectory analysis at ground contact point"""
        if len(valid_points) < 15:
            return False, 0.0
        
        # Find the point with minimum y-coordinate (ground contact)
        recent_points = valid_points[-15:]
        min_y = float('inf')
        ground_contact_idx = -1
        
        for i, point in enumerate(recent_points):
            if point['y'] < min_y:
                min_y = point['y']
                ground_contact_idx = i
        
        if ground_contact_idx == -1 or ground_contact_idx < 2 or ground_contact_idx >= len(recent_points) - 2:
            return False, 0.0
        
        # Check trajectory direction changes around ground contact
        # Before contact: should be going down
        # At contact: direction change
        # After contact: should be going up
        
        # Calculate direction vectors
        dx_before = recent_points[ground_contact_idx]['x'] - recent_points[ground_contact_idx - 1]['x']
        dy_before = recent_points[ground_contact_idx]['y'] - recent_points[ground_contact_idx - 1]['y']
        
        dx_after = recent_points[ground_contact_idx + 1]['x'] - recent_points[ground_contact_idx]['x']
        dy_after = recent_points[ground_contact_idx + 1]['y'] - recent_points[ground_contact_idx]['y']
        
        # Check if ball was going down before contact and up after
        going_down_before = dy_before > 0  # Positive dy means going down in image coordinates
        going_up_after = dy_after < 0      # Negative dy means going up in image coordinates
        
        if not (going_down_before and going_up_after):
            return False, 0.0
        
        # Calculate angle change at ground contact
        direction_before = math.atan2(dy_before, dx_before)
        direction_after = math.atan2(dy_after, dx_after)
        
        angle_change = abs(direction_after - direction_before)
        if angle_change > math.pi:
            angle_change = 2 * math.pi - angle_change
        
        # Significant direction change (bounce typically causes > 30 degrees)
        if angle_change > math.pi / 6:  # 30 degrees (lowered from 60)
            if abs(recent_points[ground_contact_idx]['frame'] - current_frame) <= 3:
                confidence = min(1.0, angle_change / (math.pi / 2))  # Normalize to 90 degrees
                return True, confidence
        
        return False, 0.0
    
    def get_bounce_summary(self) -> Dict:
        """Get summary of detected bounces"""
        return {
            'total_bounces': len(self.detected_bounces),
            'bounces': self.detected_bounces,
            'average_confidence': np.mean([b['confidence'] for b in self.detected_bounces]) if self.detected_bounces else 0.0
        }


class TennisBounceProcessorV2:
    """Processes video with new physics-based bounce detection"""
    
    def __init__(self):
        """Initialize processor with new bounce detector"""
        self.bounce_detector = PhysicsBounceDetector()
        self.bounce_history = []
    
    def process_video(self, video_file: str, csv_file: str, output_file: str = None, show_viewer: bool = False):
        """Process video with new bounce detection system"""
        # Load CSV data
        df = pd.read_csv(csv_file)
        cap = cv2.VideoCapture(video_file)
        
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_file}")
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        logger.info(f"Processing {len(df)} frames from {video_file}")
        logger.info(f"Video: {width}x{height} @ {fps}fps")
        
        # Load court data for ground plane detection
        self._load_court_data(df)
        
        # Setup video writer if output specified
        out = None
        if output_file:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_file, fourcc, fps, (width, height))
        
        # Setup viewer
        if show_viewer:
            cv2.namedWindow('Tennis Bounce Detection V2', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('Tennis Bounce Detection V2', 1200, 800)
        
        try:
            for idx, row in df.iterrows():
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Get ball position from CSV
                ball_x = self._parse_float(row.get('ball_x', ''))
                ball_y = self._parse_float(row.get('ball_y', ''))
                
                # Add ball position to detector
                self.bounce_detector.add_ball_position(ball_x, ball_y, idx)
                
                # Detect bounce
                is_bounce, confidence, bounce_info = self.bounce_detector.detect_bounce(idx)
                
                # Store bounce detection
                if is_bounce and ball_x is not None and ball_y is not None:
                    self.bounce_history.append({
                        'frame': idx,
                        'confidence': confidence,
                        'ball_x': ball_x,
                        'ball_y': ball_y,
                        'info': bounce_info
                    })
                
                # Add overlays
                frame_with_overlays = self._add_overlays(frame, idx, ball_x, ball_y, is_bounce, confidence, bounce_info)
                
                # Write frame
                if out:
                    out.write(frame_with_overlays)
                
                # Show in viewer
                if show_viewer:
                    cv2.imshow('Tennis Bounce Detection V2', frame_with_overlays)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        break
                    elif key == ord(' '):  # Space to pause
                        cv2.waitKey(0)
                
                # Progress update
                if idx % 30 == 0:
                    logger.info(f"Processed {idx}/{len(df)} frames ({idx/len(df)*100:.1f}%)")
        
        finally:
            # Cleanup
            cap.release()
            if out:
                out.release()
            if show_viewer:
                cv2.destroyAllWindows()
        
        # Print bounce detection summary
        self._print_summary()
    
    def _load_court_data(self, df: pd.DataFrame):
        """Load court keypoints from CSV data"""
        try:
            if 'court_keypoints' in df.columns:
                for idx, row in df.iterrows():
                    court_keypoints_str = row.get('court_keypoints', '')
                    if court_keypoints_str and court_keypoints_str != '':
                        court_keypoints = self._parse_court_keypoints(court_keypoints_str)
                        if court_keypoints and len(court_keypoints) >= 4:
                            self.bounce_detector.set_court_data(court_keypoints)
                            logger.info(f"Loaded court keypoints from frame {idx}")
                            return
                logger.warning("No valid court keypoints found in CSV")
        except Exception as e:
            logger.warning(f"Error loading court data: {e}")
    
    def _parse_court_keypoints(self, keypoints_str: str) -> List[Tuple]:
        """Parse court keypoints from CSV string format"""
        try:
            if not keypoints_str or keypoints_str == '':
                return []
            
            keypoints = []
            points = keypoints_str.split(';')
            for point in points:
                if ',' in point:
                    x, y = point.split(',')
                    try:
                        x_val = float(x) if x != 'nan' and x != '' else None
                        y_val = float(y) if y != 'nan' and y != '' else None
                        keypoints.append((x_val, y_val))
                    except ValueError:
                        keypoints.append((None, None))
                else:
                    keypoints.append((None, None))
            
            return keypoints
        except Exception as e:
            logger.warning(f"Error parsing court keypoints: {e}")
            return []
    
    def _parse_float(self, value: str) -> Optional[float]:
        """Parse float value from CSV string"""
        try:
            if pd.isna(value) or value == '':
                return None
            return float(value)
        except (ValueError, TypeError):
            return None
    
    def _add_overlays(self, frame: np.ndarray, frame_number: int, ball_x: Optional[float], ball_y: Optional[float], 
                     is_bounce: bool, confidence: float, bounce_info: Dict) -> np.ndarray:
        """Add overlays to frame for visualization"""
        frame = frame.copy()
        
        # Frame info
        cv2.putText(frame, f"Frame: {frame_number}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Ground plane indicator
        if self.bounce_detector.ground_plane_height:
            y_ground = int(self.bounce_detector.ground_plane_height)
            cv2.line(frame, (0, y_ground), (frame.shape[1], y_ground), (100, 100, 100), 1)
            cv2.putText(frame, "Ground Level", (10, y_ground - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1)
        
        # Ball position
        if ball_x is not None and ball_y is not None:
            bx, by = int(ball_x), int(ball_y)
            cv2.circle(frame, (bx, by), 8, (0, 255, 255), -1)  # Yellow ball
            cv2.putText(frame, "Ball", (bx + 10, by), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            
            # Distance to ground
            if self.bounce_detector.ground_plane_height:
                distance_to_ground = abs(ball_y - self.bounce_detector.ground_plane_height)
                cv2.putText(frame, f"Ground Dist: {distance_to_ground:.1f}px", (bx + 10, by + 20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
            
            # Bounce indicator
            if is_bounce:
                cv2.circle(frame, (bx, by), 15, (0, 0, 255), 3)  # Red circle for bounce
                cv2.putText(frame, f"BOUNCE! ({confidence:.2f})", (bx - 50, by - 20), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                # Show method agreement
                if 'method_agreement' in bounce_info:
                    cv2.putText(frame, f"Methods: {bounce_info['method_agreement']}/4", (bx - 50, by - 40), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        
        return frame
    
    def _print_summary(self):
        """Print bounce detection summary"""
        summary = self.bounce_detector.get_bounce_summary()
        
        logger.info("=== PHYSICS-BASED BOUNCE DETECTION SUMMARY ===")
        logger.info(f"Total bounces detected: {summary['total_bounces']}")
        logger.info(f"Average confidence: {summary['average_confidence']:.3f}")
        
        if summary['bounces']:
            logger.info("Bounce details:")
            for i, bounce in enumerate(summary['bounces']):
                logger.info(f"  Bounce {i+1}: Frame {bounce['frame']}, "
                           f"Confidence {bounce['confidence']:.3f}, "
                           f"Methods {bounce['methods']}/4")
        else:
            logger.info("No bounces detected in this video")


def main():
    """Main function for standalone usage"""
    parser = argparse.ArgumentParser(description='Tennis Ball Bounce Detection System V2')
    parser.add_argument('--video', required=True, help='Input video file')
    parser.add_argument('--csv', required=True, help='Input CSV file with ball tracking data')
    parser.add_argument('--output', default='tennis_bounce_v2_analysis.mp4', help='Output video file')
    parser.add_argument('--viewer', action='store_true', help='Show real-time viewer')
    
    args = parser.parse_args()
    
    processor = TennisBounceProcessorV2()
    processor.process_video(args.video, args.csv, args.output, show_viewer=args.viewer)


if __name__ == "__main__":
    main()
