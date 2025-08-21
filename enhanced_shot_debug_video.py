#!/usr/bin/env python3
"""
Enhanced Shot Classification Debug Video
Creates a debug video for frame-by-frame analysis of shot classifications
"""

import cv2
import pandas as pd
import numpy as np
import logging
import time
from enhanced_shot_classifier import EnhancedShotClassifier

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EnhancedShotDebugVideo:
    """Debug video creator for enhanced shot classification"""
    
    def __init__(self):
        self.shot_classifier = EnhancedShotClassifier()
        
        # Shot type colors for visualization
        self.shot_colors = {
            'serve': (0, 255, 0),      # Green
            'forehand': (255, 0, 0),   # Blue
            'backhand': (0, 0, 255),   # Red
            'overhead_smash': (255, 255, 0),  # Cyan
            'ready_stance': (255, 255, 255),  # White
            'moving': (128, 128, 128),  # Gray
            'unknown': (0, 255, 255)   # Yellow
        }
    
    def _parse_poses_from_csv(self, poses_str: str):
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
    
    def _parse_ball_position_from_csv(self, ball_x: str, ball_y: str):
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
    
    def _parse_court_keypoints_from_csv(self, court_str: str):
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
    
    def _parse_player_bboxes_from_csv(self, bboxes_str: str):
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
    
    def create_debug_video(self, csv_path: str, video_path: str, output_path: str = "shot_debugging.mp4"):
        """Create debug video with detailed shot classification information"""
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
            logger.info(f"Output debug video: {output_path}")
            
            # Setup output video
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            output_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            
            # Process each frame
            start_time = time.time()
            
            for frame_count in range(min(len(df), total_frames)):
                # Read frame
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Get CSV data for this frame
                row = df.iloc[frame_count]
                
                # Parse data
                poses = self._parse_poses_from_csv(row.get('pose_keypoints', ''))
                ball_position = self._parse_ball_position_from_csv(row.get('ball_x', ''), row.get('ball_y', ''))
                court_keypoints = self._parse_court_keypoints_from_csv(row.get('court_keypoints', ''))
                player_bboxes = self._parse_player_bboxes_from_csv(row.get('player_bboxes', ''))
                
                # Apply enhanced shot classification
                enhanced_shot_types = []
                for i, bbox in enumerate(player_bboxes):
                    # Classify shot for each player
                    shot_type = self.shot_classifier.classify_shot(
                        bbox, ball_position, poses, court_keypoints, frame_count, player_id=i
                    )
                    enhanced_shot_types.append(shot_type)
                
                # Draw debug information on frame
                self._draw_debug_info(
                    frame, player_bboxes, enhanced_shot_types, 
                    ball_position, court_keypoints, poses, frame_count, total_frames
                )
                
                # Write to output video
                output_writer.write(frame)
                
                # Progress update
                if frame_count % 30 == 0:
                    elapsed = time.time() - start_time
                    fps_processed = frame_count / elapsed
                    logger.info(f"Processed {frame_count}/{total_frames} frames ({fps_processed:.1f} fps)")
            
            # Cleanup
            cap.release()
            output_writer.release()
            
            # Print final statistics
            final_stats = self.shot_classifier.get_shot_statistics()
            logger.info("Final Enhanced Shot Statistics:")
            for shot_type, count in final_stats.items():
                if count > 0:
                    logger.info(f"  {shot_type}: {count}")
            
            logger.info(f"Debug video created: {output_path}")
            
        except Exception as e:
            logger.error(f"Error creating debug video: {e}")
    
    def _draw_debug_info(self, frame: np.ndarray, player_bboxes, enhanced_shot_types, 
                        ball_position, court_keypoints, poses, frame_count, total_frames):
        """Draw detailed debug information on frame"""
        try:
            # Draw frame counter
            cv2.putText(frame, f"Frame: {frame_count}/{total_frames}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Draw player bounding boxes and shot types
            for i, (bbox, shot_type) in enumerate(zip(player_bboxes, enhanced_shot_types)):
                x1, y1, x2, y2 = bbox
                
                # Get shot color
                shot_color = self.shot_colors.get(shot_type, (255, 255, 255))
                
                # Draw player bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), shot_color, 2)
                
                # Draw enhanced shot type text below player
                shot_text = f"Player {i+1}: {shot_type.replace('_', ' ').title()}"
                text_size = cv2.getTextSize(shot_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                text_x = x1
                text_y = y2 + text_size[1] + 10
                
                # Draw text background
                cv2.rectangle(frame, (text_x - 5, text_y - text_size[1] - 5), 
                             (text_x + text_size[0] + 5, text_y + 5), (0, 0, 0), -1)
                
                # Draw text
                cv2.putText(frame, shot_text, (text_x, text_y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, shot_color, 2)
                
                # Draw player center point
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2
                cv2.circle(frame, (center_x, center_y), 3, (0, 255, 255), -1)
            
            # Draw ball position
            if ball_position:
                ball_x, ball_y = ball_position
                cv2.circle(frame, (ball_x, ball_y), 8, (0, 255, 255), -1)
                cv2.putText(frame, "BALL", (ball_x + 10, ball_y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
            
            # Draw court keypoints
            for i, (x, y) in enumerate(court_keypoints):
                cv2.circle(frame, (x, y), 3, (255, 0, 255), -1)
                cv2.putText(frame, f"C{i}", (x + 5, y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 0, 255), 1)
            
            # Draw pose keypoints for first player
            if poses and len(poses) > 0:
                pose = poses[0]  # First player's pose
                keypoints = pose['keypoints']
                confidence = pose.get('confidence', [])
                
                # Draw key arm keypoints
                arm_keypoints = [5, 6, 7, 8, 9, 10]  # shoulders, elbows, wrists
                for kp_idx in arm_keypoints:
                    if kp_idx < len(keypoints) and kp_idx < len(confidence):
                        kp = keypoints[kp_idx]
                        conf = confidence[kp_idx]
                        if conf > 0.3:  # Only draw confident keypoints
                            cv2.circle(frame, (int(kp[0]), int(kp[1])), 4, (255, 255, 0), -1)
                            cv2.putText(frame, f"KP{kp_idx}", (int(kp[0]) + 5, int(kp[1])), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 0), 1)
            
        except Exception as e:
            logger.debug(f"Error drawing debug info: {e}")

def main():
    """Main function"""
    import sys
    
    if len(sys.argv) < 3:
        print("Usage: python3 enhanced_shot_debug_video.py <csv_path> <video_path> [output_path]")
        print("Example: python3 enhanced_shot_debug_video.py tennis_analysis_data.csv tennis_test5.mp4")
        return
    
    csv_path = sys.argv[1]
    video_path = sys.argv[2]
    output_path = sys.argv[3] if len(sys.argv) > 3 else "shot_debugging.mp4"
    
    debug_video = EnhancedShotDebugVideo()
    debug_video.create_debug_video(csv_path, video_path, output_path)

if __name__ == "__main__":
    main()
