#!/usr/bin/env python3
"""
Tennis Analysis Demo Script
Integrates player detection, pose estimation, and ball bounce detection
for comprehensive tennis video analysis with real-time visualization.
"""

import cv2
import numpy as np
import yaml
import time
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TennisAnalysisDemo:
    """Integrated tennis analysis system with real-time visualization"""
    
    def __init__(self, config_path: str = "config.yaml"):
        """Initialize the tennis analysis demo system"""
        self.config = self._load_config(config_path)
        self.frame_count = 0
        self.start_time = time.time()
        
        # Initialize components
        self.player_detector = None
        self.pose_estimator = None
        self.bounce_detector = None
        
        # Analysis results
        self.analysis_results = {
            'total_frames': 0,
            'players_detected': 0,
            'poses_estimated': 0,
            'bounces_detected': 0,
            'processing_times': []
        }
        
        self._initialize_components()
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            logger.info(f"Configuration loaded from {config_path}")
            return config
        except Exception as e:
            logger.error(f"Error loading config: {e}")
            # Fallback to default config
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration if YAML loading fails"""
        return {
            'models': {
                'yolo_player': 'models/playersnball4.pt',
                'yolo_pose': 'models/yolov8n-pose.pt',
                'bounce_detector': 'models/bounce_detector.cbm'
            },
            'yolo_player': {
                'conf_threshold': 0.5,
                'iou_threshold': 0.45,
                'max_det': 10
            },
            'yolo_pose': {
                'conf_threshold': 0.3,
                'iou_threshold': 0.45,
                'max_det': 4,
                'keypoints': 17
            },
            'video': {
                'fps': 30,
                'frame_skip': 1,
                'resize_width': 1920,
                'resize_height': 1080
            }
        }
    
    def _initialize_components(self):
        """Initialize all analysis components"""
        try:
            # Initialize player detector
            player_model_path = self.config['models']['yolo_player']
            if Path(player_model_path).exists():
                self.player_detector = PlayerDetector(
                    player_model_path,
                    self.config['yolo_player']
                )
                logger.info("Player detector initialized successfully")
            else:
                logger.warning(f"Player detection model not found: {player_model_path}")
            
            # Initialize pose estimator
            pose_model_path = self.config['models']['yolo_pose']
            if Path(pose_model_path).exists():
                self.pose_estimator = PoseEstimator(
                    pose_model_path,
                    self.config['yolo_pose']
                )
                logger.info("Pose estimator initialized successfully")
            else:
                logger.warning(f"Pose estimation model not found: {pose_model_path}")
            
            # Initialize bounce detector (CatBoost model)
            bounce_model_path = self.config['models'].get('bounce_detector')
            if bounce_model_path and Path(bounce_model_path).exists():
                try:
                    self.bounce_detector = BounceDetector(bounce_model_path)
                    logger.info("Bounce detector initialized successfully")
                except Exception as e:
                    logger.warning(f"Bounce detector initialization failed: {e}")
                    self.bounce_detector = None
            else:
                logger.warning(f"Bounce detection model not found: {bounce_model_path}")
                self.bounce_detector = None
                
        except Exception as e:
            logger.error(f"Error initializing components: {e}")
    
    def analyze_video(self, video_path: str, output_path: Optional[str] = None):
        """Analyze tennis video with all three models"""
        if not Path(video_path).exists():
            logger.error(f"Video file not found: {video_path}")
            return
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error(f"Could not open video: {video_path}")
            return
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        logger.info(f"Video: {width}x{height}, {fps} FPS, {total_frames} frames")
        
        # Setup output video if specified
        output_writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            output_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_skip = self.config['video'].get('frame_skip', 1)
        frame_count = 0
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Process every nth frame
                if frame_count % frame_skip != 0:
                    frame_count += 1
                    continue
                
                # Analyze frame
                start_time = time.time()
                annotated_frame = self._analyze_frame(frame)
                processing_time = time.time() - start_time
                
                # Update statistics
                self.analysis_results['total_frames'] += 1
                self.analysis_results['processing_times'].append(processing_time)
                
                # Display frame
                cv2.imshow('Tennis Analysis Demo', annotated_frame)
                
                # Save to output video if specified
                if output_writer:
                    output_writer.write(annotated_frame)
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):  # Quit
                    break
                elif key == ord('p'):  # Pause/Resume
                    cv2.waitKey(0)
                elif key == ord('s'):  # Save current frame
                    cv2.imwrite(f"frame_{frame_count:06d}.jpg", annotated_frame)
                    logger.info(f"Saved frame {frame_count}")
                
                frame_count += 1
                
                # Display progress
                if frame_count % 30 == 0:
                    progress = (frame_count / total_frames) * 100
                    avg_time = np.mean(self.analysis_results['processing_times'])
                    logger.info(f"Progress: {progress:.1f}% | Avg processing time: {avg_time:.3f}s")
        
        finally:
            cap.release()
            if output_writer:
                output_writer.release()
            cv2.destroyAllWindows()
            
            # Print final statistics
            self._print_analysis_summary()
    
    def _analyze_frame(self, frame: np.ndarray) -> np.ndarray:
        """Analyze a single frame with all three models"""
        annotated_frame = frame.copy()
        
        # 1. Player Detection
        player_detections = []
        if self.player_detector:
            try:
                player_detections = self.player_detector.detect_players(frame)
                self.analysis_results['players_detected'] += len(player_detections)
                annotated_frame = self.player_detector.draw_detections(annotated_frame, player_detections)
            except Exception as e:
                logger.error(f"Player detection error: {e}")
        
        # 2. Pose Estimation
        poses = []
        if self.pose_estimator and player_detections:
            try:
                poses = self.pose_estimator.estimate_poses(frame, player_detections)
                self.analysis_results['poses_estimated'] += len(poses)
                annotated_frame = self.pose_estimator.draw_poses(annotated_frame, poses)
            except Exception as e:
                logger.error(f"Pose estimation error: {e}")
        
        # 3. Ball Bounce Detection
        if self.bounce_detector:
            try:
                bounce_probability = self.bounce_detector.detect_bounce(frame)
                if bounce_probability > 0.7:  # High confidence threshold
                    self.analysis_results['bounces_detected'] += 1
                    # Draw bounce indicator
                    cv2.putText(annotated_frame, f"BOUNCE! ({bounce_probability:.2f})", 
                               (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    # Draw bounce circle
                    cv2.circle(annotated_frame, (100, 100), 30, (0, 0, 255), -1)
            except Exception as e:
                logger.error(f"Bounce detection error: {e}")
        
        # Add frame counter and statistics
        self._add_frame_info(annotated_frame)
        
        return annotated_frame
    
    def _add_frame_info(self, frame: np.ndarray):
        """Add frame information overlay"""
        # Frame counter
        cv2.putText(frame, f"Frame: {self.analysis_results['total_frames']}", 
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Statistics
        stats_text = [
            f"Players: {self.analysis_results['players_detected']}",
            f"Poses: {self.analysis_results['poses_estimated']}",
            f"Bounces: {self.analysis_results['bounces_detected']}"
        ]
        
        y_offset = 60
        for stat in stats_text:
            cv2.putText(frame, stat, (10, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            y_offset += 25
        
        # Instructions
        instructions = [
            "Press 'q' to quit",
            "Press 'p' to pause",
            "Press 's' to save frame"
        ]
        
        y_offset = frame.shape[0] - 100
        for instruction in instructions:
            cv2.putText(frame, instruction, (10, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            y_offset += 20
    
    def _print_analysis_summary(self):
        """Print analysis summary"""
        print("\n" + "="*50)
        print("TENNIS ANALYSIS SUMMARY")
        print("="*50)
        print(f"Total frames processed: {self.analysis_results['total_frames']}")
        print(f"Total players detected: {self.analysis_results['players_detected']}")
        print(f"Total poses estimated: {self.analysis_results['poses_estimated']}")
        print(f"Total bounces detected: {self.analysis_results['bounces_detected']}")
        
        if self.analysis_results['processing_times']:
            avg_time = np.mean(self.analysis_results['processing_times'])
            min_time = np.min(self.analysis_results['processing_times'])
            max_time = np.max(self.analysis_results['processing_times'])
            print(f"Processing time - Avg: {avg_time:.3f}s, Min: {min_time:.3f}s, Max: {max_time:.3f}s")
        
        total_time = time.time() - self.start_time
        print(f"Total analysis time: {total_time:.2f}s")
        print("="*50)


class PlayerDetector:
    """YOLOv8-based player detection"""
    
    def __init__(self, model_path: str, config: Dict[str, Any]):
        self.config = config
        try:
            from ultralytics import YOLO
            self.model = YOLO(model_path)
            logger.info(f"Player detector initialized with {model_path}")
        except ImportError:
            logger.error("Ultralytics not installed. Install with: pip install ultralytics")
            self.model = None
        except Exception as e:
            logger.error(f"Error loading player detection model: {e}")
            self.model = None
    
    def detect_players(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        if not self.model:
            return []
        
        try:
            results = self.model(
                frame,
                conf=self.config.get('conf_threshold', 0.5),
                iou=self.config.get('iou_threshold', 0.45),
                max_det=self.config.get('max_det', 10),
                verbose=False
            )
            
            detections = []
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        conf = box.conf[0].cpu().numpy()
                        cls = int(box.cls[0].cpu().numpy())
                        
                        # Calculate bounding box dimensions
                        bbox_width = x2 - x1
                        bbox_height = y2 - y1
                        bbox_area = bbox_width * bbox_height
                        
                        # Log detection details
                        logger.debug(f"Detection: class={cls}, conf={conf:.2f}, bbox=[{int(x1)}, {int(y1)}, {int(x2)}, {int(y2)}], area={bbox_area}")
                        
                        # Only keep class 1 (players), filter out class 0 (balls)
                        if cls == 1:  # Player class
                            detection = {
                                'bbox': [int(x1), int(y1), int(x2), int(y2)],
                                'confidence': float(conf),
                                'class': cls,
                                'center': [int((x1 + x2) / 2), int((y1 + y2) / 2)]
                            }
                            detections.append(detection)
                            logger.debug(f"Added player detection (class {cls})")
                        else:
                            logger.debug(f"Filtered out ball detection (class {cls})")
            
            logger.info(f"Detected {len(detections)} players")
            if detections:
                logger.info(f"First detection: class={detections[0]['class']}, bbox={detections[0]['bbox']}")
            return detections
        except Exception as e:
            logger.error(f"Player detection error: {e}")
            return []
    
    def draw_detections(self, frame: np.ndarray, detections: List[Dict[str, Any]]) -> np.ndarray:
        frame_copy = frame.copy()
        
        for detection in detections:
            x1, y1, x2, y2 = detection['bbox']
            conf = detection['confidence']
            cls = detection['class']
            
            # Color for players (green)
            color = (0, 255, 0)
            
            # Draw bounding box
            cv2.rectangle(frame_copy, (x1, y1), (x2, y2), color, 2)
            
            # Draw label
            label = f"Player: {conf:.2f}"
            cv2.putText(frame_copy, label, (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        return frame_copy


class PoseEstimator:
    """YOLOv8-pose-based pose estimation"""
    
    def __init__(self, model_path: str, config: Dict[str, Any]):
        self.config = config
        try:
            from ultralytics import YOLO
            self.model = YOLO(model_path)
            logger.info(f"Pose estimator initialized with {model_path}")
        except ImportError:
            logger.error("Ultralytics not installed. Install with: pip install ultralytics")
            self.model = None
        except Exception as e:
            logger.error(f"Error loading pose estimation model: {e}")
            self.model = None
    
    def estimate_poses(self, frame: np.ndarray, player_detections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not self.model:
            return []
        
        poses = []
        try:
            for player_detection in player_detections:
                x1, y1, x2, y2 = player_detection['bbox']
                
                # Extract player region with padding
                padding = 20
                x1_pad = max(0, x1 - padding)
                y1_pad = max(0, y1 - padding)
                x2_pad = min(frame.shape[1], x2 + padding)
                y2_pad = min(frame.shape[0], y2 + padding)
                
                player_roi = frame[y1_pad:y2_pad, x1_pad:x2_pad]
                
                if player_roi.size == 0:
                    continue
                
                # Run pose estimation
                results = self.model(
                    player_roi,
                    conf=self.config.get('conf_threshold', 0.3),
                    iou=self.config.get('iou_threshold', 0.45),
                    max_det=1,
                    verbose=False
                )
                
                for result in results:
                    try:
                        keypoints = result.keypoints
                        if keypoints is not None and hasattr(keypoints, 'data') and len(keypoints.data) > 0:
                            kpts = keypoints.data[0].cpu().numpy()
                            if len(kpts) > 0:
                                # Adjust keypoint coordinates back to full frame coordinates
                                adjusted_kpts = kpts.copy()
                                adjusted_kpts[:, 0] += x1_pad  # Add x offset
                                adjusted_kpts[:, 1] += y1_pad  # Add y offset
                                
                                pose = {
                                    'keypoints': adjusted_kpts,
                                    'bbox': [x1, y1, x2, y2],
                                    'confidence': float(result.boxes.conf[0].cpu().numpy()) if result.boxes is not None and len(result.boxes.conf) > 0 else 0.0
                                }
                                poses.append(pose)
                    except Exception as e:
                        logger.debug(f"Error processing pose result: {e}")
                        continue
            
            return poses
        except Exception as e:
            logger.error(f"Pose estimation error: {e}")
            return []
    
    def draw_poses(self, frame: np.ndarray, poses: List[Dict[str, Any]]) -> np.ndarray:
        frame_copy = frame.copy()
        
        # COCO keypoint connections for skeleton drawing
        skeleton = [
            (0, 1), (0, 2), (1, 3), (2, 4),  # Head
            (5, 6), (5, 7), (6, 8), (7, 9),  # Arms
            (5, 11), (6, 12), (11, 12),      # Torso
            (11, 13), (13, 15), (12, 14), (14, 16)  # Legs
        ]
        
        for pose in poses:
            keypoints = pose['keypoints']
            bbox = pose['bbox']
            
            # Draw keypoints
            for i, (x, y, conf) in enumerate(keypoints):
                if conf > 0.3:  # Only draw confident keypoints
                    cv2.circle(frame_copy, (int(x), int(y)), 3, (0, 255, 255), -1)
            
            # Draw skeleton
            for start_idx, end_idx in skeleton:
                if (start_idx < len(keypoints) and end_idx < len(keypoints) and 
                    keypoints[start_idx][2] > 0.3 and keypoints[end_idx][2] > 0.3):
                    start_point = (int(keypoints[start_idx][0]), int(keypoints[start_idx][1]))
                    end_point = (int(keypoints[end_idx][0]), int(keypoints[end_idx][1]))
                    cv2.line(frame_copy, start_point, end_point, (255, 255, 0), 2)
            
            # Draw pose confidence
            x1, y1, x2, y2 = bbox
            cv2.putText(frame_copy, f"Pose: {pose['confidence']:.2f}", 
                       (x1, y2+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
        
        return frame_copy


class BounceDetector:
    """CatBoost-based ball bounce detection"""
    
    def __init__(self, model_path: str):
        try:
            import catboost as cb
            self.model = cb.CatBoostClassifier()
            self.model.load_model(model_path)
            logger.info(f"Bounce detector initialized with {model_path}")
        except ImportError:
            logger.error("CatBoost not installed. Install with: pip install catboost")
            self.model = None
        except Exception as e:
            logger.error(f"Error loading bounce detection model: {e}")
            self.model = None
    
    def detect_bounce(self, frame: np.ndarray) -> float:
        if not self.model:
            return 0.0
        
        try:
            # Simple feature extraction (you might want to enhance this)
            # Convert to grayscale and resize for feature extraction
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            resized = cv2.resize(gray, (64, 64))
            
            # Extract basic features (histogram, edges, etc.)
            features = []
            
            # Histogram features
            hist = cv2.calcHist([resized], [0], None, [16], [0, 256])
            hist = hist.flatten() / hist.sum()  # Normalize
            features.extend(hist)
            
            # Edge features
            edges = cv2.Canny(resized, 50, 150)
            edge_density = np.sum(edges > 0) / (64 * 64)
            features.append(edge_density)
            
            # Texture features (simple gradient)
            grad_x = cv2.Sobel(resized, cv2.CV_64F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(resized, cv2.CV_64F, 0, 1, ksize=3)
            gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
            avg_gradient = np.mean(gradient_magnitude)
            features.append(avg_gradient)
            
            # Pad features to expected length (adjust based on your model)
            while len(features) < 20:  # Assuming model expects 20 features
                features.append(0.0)
            features = features[:20]  # Truncate if too long
            
            # Make prediction
            prediction = self.model.predict_proba([features])[0]
            return prediction[1] if len(prediction) > 1 else prediction[0]
            
        except Exception as e:
            logger.error(f"Bounce detection error: {e}")
            return 0.0


def main():
    """Main function to run the tennis analysis demo"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Tennis Analysis Demo')
    parser.add_argument('--video', '-v', type=str, default='tennis_test.mp4',
                       help='Path to input video file')
    parser.add_argument('--output', '-o', type=str, default=None,
                       help='Path to output video file (optional)')
    parser.add_argument('--config', '-c', type=str, default='config.yaml',
                       help='Path to configuration file')
    
    args = parser.parse_args()
    
    # Check if video exists
    if not Path(args.video).exists():
        print(f"Error: Video file '{args.video}' not found!")
        print("Available video files:")
        for video_file in Path('.').glob('*.mp4'):
            print(f"  - {video_file}")
        return
    
    print("🎾 Tennis Analysis Demo Starting...")
    print(f"📹 Input video: {args.video}")
    if args.output:
        print(f"💾 Output video: {args.output}")
    print(f"⚙️  Config file: {args.config}")
    print("\nControls:")
    print("  - Press 'q' to quit")
    print("  - Press 'p' to pause/resume")
    print("  - Press 's' to save current frame")
    print("\nInitializing models...")
    
    # Create and run analyzer
    analyzer = TennisAnalysisDemo(args.config)
    analyzer.analyze_video(args.video, args.output)


if __name__ == "__main__":
    main()
