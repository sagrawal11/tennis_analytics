#!/usr/bin/env python3
"""
Enhanced Shot Classification Demo
Shows real-time shot classification with visual overlays
"""

import cv2
import numpy as np
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
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EnhancedShotDemo:
    """Demo for enhanced shot classification with visual feedback"""
    
    def __init__(self, config_path: str = "config.yaml"):
        """Initialize the demo"""
        self.config_path = config_path
        self.config = self._load_config()
        
        # Initialize shot classifier
        self.shot_classifier = EnhancedShotClassifier()
        
        # Initialize models
        self.player_detector = None
        self.pose_estimator = None
        self.ball_detector = None
        self.court_detector = None
        
        # Initialize tracking
        self.player_positions = []
        self.ball_positions = []
        
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
        
        logger.info("Enhanced Shot Demo initialized")
    
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
    
    def _initialize_models(self):
        """Initialize AI models for detection"""
        try:
            from ultralytics import YOLO
            
            # Initialize player detector
            if 'player_model' in self.config:
                self.player_detector = YOLO(self.config['player_model'])
                logger.info("Player detector initialized")
            
            # Initialize pose estimator
            if 'pose_model' in self.config:
                self.pose_estimator = YOLO(self.config['pose_model'])
                logger.info("Pose estimator initialized")
            
            # Initialize ball detector (RF-DETR)
            if 'ball_model' in self.config:
                # Import RF-DETR ball detector
                from RF_ball_detector import RFBallDetector
                self.ball_detector = RFBallDetector(self.config['ball_model'])
                logger.info("Ball detector initialized")
            
            # Initialize court detector
            if 'court_model' in self.config:
                self.court_detector = YOLO(self.config['court_model'])
                logger.info("Court detector initialized")
                
        except Exception as e:
            logger.error(f"Error initializing models: {e}")
    
    def _detect_players(self, frame: np.ndarray) -> List[Dict]:
        """Detect players in frame"""
        try:
            if self.player_detector is None:
                return []
            
            results = self.player_detector(frame, verbose=False)
            players = []
            
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        conf = box.conf[0].cpu().numpy()
                        cls = box.cls[0].cpu().numpy()
                        
                        # Filter for person class (0) with high confidence
                        if cls == 0 and conf > 0.5:
                            players.append({
                                'bbox': [int(x1), int(y1), int(x2), int(y2)],
                                'confidence': float(conf)
                            })
            
            return players
            
        except Exception as e:
            logger.error(f"Error detecting players: {e}")
            return []
    
    def _detect_poses(self, frame: np.ndarray, players: List[Dict]) -> List[Dict]:
        """Detect poses for players"""
        try:
            if self.pose_estimator is None:
                return []
            
            poses = []
            for player in players:
                bbox = player['bbox']
                x1, y1, x2, y2 = bbox
                
                # Crop player region
                player_region = frame[y1:y2, x1:x2]
                if player_region.size == 0:
                    continue
                
                # Detect pose in player region
                results = self.pose_estimator(player_region, verbose=False)
                
                for result in results:
                    if result.keypoints is not None:
                        keypoints = result.keypoints.data[0].cpu().numpy()
                        confidence = result.keypoints.conf[0].cpu().numpy()
                        
                        # Convert keypoints back to full frame coordinates
                        keypoints[:, 0] += x1
                        keypoints[:, 1] += y1
                        
                        poses.append({
                            'keypoints': keypoints.tolist(),
                            'confidence': confidence.tolist(),
                            'bbox': bbox
                        })
            
            return poses
            
        except Exception as e:
            logger.error(f"Error detecting poses: {e}")
            return []
    
    def _detect_ball(self, frame: np.ndarray) -> Optional[List[int]]:
        """Detect ball position"""
        try:
            if self.ball_detector is None:
                return None
            
            # Use RF-DETR ball detector
            ball_bbox = self.ball_detector.detect_ball(frame)
            if ball_bbox:
                x1, y1, x2, y2 = ball_bbox
                ball_center = [int((x1 + x2) / 2), int((y1 + y2) / 2)]
                return ball_center
            
            return None
            
        except Exception as e:
            logger.error(f"Error detecting ball: {e}")
            return None
    
    def _detect_court(self, frame: np.ndarray) -> List[Tuple]:
        """Detect court keypoints"""
        try:
            if self.court_detector is None:
                return []
            
            results = self.court_detector(frame, verbose=False)
            court_keypoints = []
            
            for result in results:
                if result.keypoints is not None:
                    keypoints = result.keypoints.data[0].cpu().numpy()
                    court_keypoints = [(int(kp[0]), int(kp[1])) for kp in keypoints if kp[2] > 0.3]
                    break
            
            return court_keypoints
            
        except Exception as e:
            logger.error(f"Error detecting court: {e}")
            return []
    
    def _calculate_player_speed(self, player_bbox: List[int]) -> float:
        """Calculate player movement speed"""
        try:
            player_center = [(player_bbox[0] + player_bbox[2]) / 2, 
                           (player_bbox[1] + player_bbox[3]) / 2]
            
            if self.player_positions:
                prev_center = self.player_positions[-1]
                distance = np.sqrt((player_center[0] - prev_center[0])**2 + 
                                 (player_center[1] - prev_center[1])**2)
                return distance
            else:
                self.player_positions.append(player_center)
                return 0.0
                
        except Exception as e:
            logger.error(f"Error calculating player speed: {e}")
            return 0.0
    
    def _draw_shot_classification(self, frame: np.ndarray, players: List[Dict], 
                                poses: List[Dict], ball_position: Optional[List[int]], 
                                court_keypoints: List[Tuple]):
        """Draw shot classifications on frame"""
        try:
            for i, player in enumerate(players):
                bbox = player['bbox']
                x1, y1, x2, y2 = bbox
                
                # Calculate player speed
                player_speed = self._calculate_player_speed(bbox)
                
                # Classify shot
                shot_type = self.shot_classifier.classify_shot(
                    bbox, ball_position, poses, court_keypoints, player_speed
                )
                
                # Get shot color
                shot_color = self.shot_colors.get(shot_type, (255, 255, 255))
                
                # Draw player bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), shot_color, 2)
                
                # Draw shot type text below player
                shot_text = f"Shot: {shot_type.replace('_', ' ').title()}"
                text_size = cv2.getTextSize(shot_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                text_x = x1
                text_y = y2 + text_size[1] + 10
                
                # Draw text background
                cv2.rectangle(frame, (text_x - 5, text_y - text_size[1] - 5), 
                             (text_x + text_size[0] + 5, text_y + 5), (0, 0, 0), -1)
                
                # Draw text
                cv2.putText(frame, shot_text, (text_x, text_y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, shot_color, 2)
                
                # Draw speed indicator
                speed_text = f"Speed: {player_speed:.1f}"
                speed_y = text_y + 25
                cv2.putText(frame, speed_text, (text_x, speed_y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                
                # Update player positions
                if len(self.player_positions) > 0:
                    self.player_positions[-1] = [(x1 + x2) / 2, (y1 + y2) / 2]
                else:
                    self.player_positions.append([(x1 + x2) / 2, (y1 + y2) / 2])
            
            # Draw ball position
            if ball_position:
                ball_x, ball_y = ball_position
                cv2.circle(frame, (ball_x, ball_y), 5, (0, 255, 255), -1)
                cv2.putText(frame, "BALL", (ball_x + 10, ball_y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
            
            # Draw court keypoints
            for kp in court_keypoints:
                cv2.circle(frame, kp, 3, (255, 255, 255), -1)
            
            # Draw shot statistics
            stats = self.shot_classifier.get_shot_statistics()
            y_offset = 30
            for shot_type, count in stats.items():
                if count > 0:
                    color = self.shot_colors.get(shot_type, (255, 255, 255))
                    text = f"{shot_type}: {count}"
                    cv2.putText(frame, text, (10, y_offset), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                    y_offset += 20
                    
        except Exception as e:
            logger.error(f"Error drawing shot classification: {e}")
    
    def run_demo(self, video_path: str, output_path: str = None):
        """Run the enhanced shot classification demo"""
        try:
            # Initialize models
            self._initialize_models()
            
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
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_count += 1
                
                # Detect players, poses, ball, and court
                players = self._detect_players(frame)
                poses = self._detect_poses(frame, players)
                ball_position = self._detect_ball(frame)
                court_keypoints = self._detect_court(frame)
                
                # Draw shot classifications
                self._draw_shot_classification(frame, players, poses, ball_position, court_keypoints)
                
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
            logger.info("Final Shot Statistics:")
            for shot_type, count in final_stats.items():
                if count > 0:
                    logger.info(f"  {shot_type}: {count}")
            
            logger.info("Demo completed!")
            
        except Exception as e:
            logger.error(f"Error in demo: {e}")

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Enhanced Shot Classification Demo")
    parser.add_argument("--video", required=True, help="Input video path")
    parser.add_argument("--config", default="config.yaml", help="Configuration file")
    parser.add_argument("--output", help="Output video path")
    
    args = parser.parse_args()
    
    # Check if video exists
    if not os.path.exists(args.video):
        logger.error(f"Video file not found: {args.video}")
        return
    
    # Create demo
    demo = EnhancedShotDemo(args.config)
    
    # Run demo
    demo.run_demo(args.video, args.output)

if __name__ == "__main__":
    main()
