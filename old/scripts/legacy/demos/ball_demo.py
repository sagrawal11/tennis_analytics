#!/usr/bin/env python3
"""
Ball Tracking Demo Script
Combines TrackNet and YOLO ball detection models for robust tennis ball tracking.
"""

import cv2
import numpy as np
import yaml
import time
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import logging
from collections import deque

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)



class BallTracker:
    """Combined ball tracking system using TrackNet and YOLO"""
    
    def __init__(self, config_path: str = "config.yaml"):
        """Initialize the ball tracking system"""
        self.config = self._load_config(config_path)
        self.frame_count = 0
        self.start_time = time.time()
        
        # Initialize models
        self.tracknet_model = None
        self.yolo_model = None
        
        # Ball tracking state
        self.ball_positions = deque(maxlen=30)  # Store last 30 ball positions
        self.ball_velocities = deque(maxlen=10)  # Store last 10 velocities
        self.tracknet_predictions = []
        self.yolo_predictions = []
        
        # Analysis results
        self.analysis_results = {
            'total_frames': 0,
            'tracknet_detections': 0,
            'yolo_detections': 0,
            'combined_detections': 0,
            'processing_times': []
        }
        
        self._initialize_models()
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            logger.info(f"Configuration loaded from {config_path}")
            return config
        except Exception as e:
            logger.error(f"Error loading config: {e}")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration if YAML loading fails"""
        return {
            'models': {
                'tracknet': 'pretrained_ball_detection.pt',  # This exists in root directory
                'yolo_player': 'models/playersnball4.pt'     # This exists in models directory
            },
            'yolo': {
                'conf_threshold': 0.3,
                'iou_threshold': 0.45,
                'max_det': 10
            },
            'tracknet': {
                'input_height': 360,
                'input_width': 640,
                'conf_threshold': 0.1,  # Lower threshold for TrackNet
                'max_dist': 100,
                'max_gap': 4,
                'min_track_length': 5
            },
            'yolo': {
                'conf_threshold': 0.1,  # Lower threshold for YOLO
                'iou_threshold': 0.45,
                'max_det': 10
            },
            'ball_tracking': {
                'max_velocity': 200,  # pixels per frame
                'min_confidence': 0.4,
                'smoothing_factor': 0.7,
                'prediction_frames': 3
            },
            'video': {
                'fps': 30,
                'frame_skip': 1,
                'resize_width': 1920,
                'resize_height': 1080
            }
        }
    
    def _initialize_models(self):
        """Initialize both ball detection models"""
        try:
            # Initialize TrackNet model
            tracknet_path = self.config['models'].get('tracknet')
            logger.info(f"Looking for TrackNet model at: {tracknet_path}")
            if tracknet_path and Path(tracknet_path).exists():
                logger.info(f"TrackNet model file found at: {tracknet_path}")
                try:
                    self.tracknet_model = TrackNetDetector(
                        tracknet_path,
                        self.config['tracknet']
                    )
                    logger.info("TrackNet model initialized successfully")
                except Exception as e:
                    logger.warning(f"TrackNet model initialization failed: {e}")
                    self.tracknet_model = None
            else:
                logger.warning(f"TrackNet model not found: {tracknet_path}")
                self.tracknet_model = None
            
            # Initialize YOLO model for ball detection
            yolo_path = self.config['models'].get('yolo_player')
            logger.info(f"Looking for YOLO model at: {yolo_path}")
            if yolo_path and Path(yolo_path).exists():
                logger.info(f"YOLO model file found at: {yolo_path}")
                try:
                    self.yolo_model = YOLOBallDetector(
                        yolo_path,
                        self.config['yolo']
                    )
                    logger.info("YOLO ball detector initialized successfully")
                except Exception as e:
                    logger.warning(f"YOLO model initialization failed: {e}")
                    self.yolo_model = None
            else:
                logger.warning(f"YOLO model not found: {yolo_path}")
                self.yolo_model = None
                
        except Exception as e:
            logger.error(f"Error initializing models: {e}")
    
    def track_ball(self, video_path: str, output_path: Optional[str] = None):
        """Track ball in video using both models"""
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
                
                # Track ball in frame
                start_time = time.time()
                annotated_frame = self._track_ball_in_frame(frame)
                processing_time = time.time() - start_time
                
                # Update statistics
                self.analysis_results['total_frames'] += 1
                self.analysis_results['processing_times'].append(processing_time)
                
                # Display frame
                cv2.imshow('Ball Tracking Demo', annotated_frame)
                
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
                    cv2.imwrite(f"ball_frame_{frame_count:06d}.jpg", annotated_frame)
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
    
    def _track_ball_in_frame(self, frame: np.ndarray) -> np.ndarray:
        """Track ball in a single frame using both models"""
        annotated_frame = frame.copy()
        
        # Get predictions from both models
        tracknet_pred = None
        yolo_pred = None
        
        # 1. TrackNet prediction
        if self.tracknet_model:
            try:
                tracknet_pred = self.tracknet_model.detect_ball(frame)
                if tracknet_pred:
                    self.analysis_results['tracknet_detections'] += 1
                    self.tracknet_predictions.append(tracknet_pred)
            except Exception as e:
                logger.error(f"TrackNet error: {e}")
        
        # 2. YOLO ball prediction
        if self.yolo_model:
            try:
                yolo_pred = self.yolo_model.detect_ball(frame)
                if yolo_pred:
                    self.analysis_results['yolo_detections'] += 1
                    self.yolo_predictions.append(yolo_pred)
            except Exception as e:
                logger.error(f"YOLO error: {e}")
        
        # 3. Combine predictions
        combined_pred = self._combine_predictions(tracknet_pred, yolo_pred)
        
        if combined_pred:
            self.analysis_results['combined_detections'] += 1
            self.ball_positions.append(combined_pred)
            
            # Calculate velocity
            if len(self.ball_positions) >= 2:
                velocity = self._calculate_velocity(self.ball_positions[-2], combined_pred)
                self.ball_velocities.append(velocity)
            
            # Draw ball tracking
            annotated_frame = self._draw_ball_tracking(annotated_frame, combined_pred)
        
        # Add frame information
        self._add_frame_info(annotated_frame)
        
        return annotated_frame
    
    def _combine_predictions(self, tracknet_pred: Optional[Dict], yolo_pred: Optional[Dict]) -> Optional[Dict]:
        """Combine predictions from both models"""
        if not tracknet_pred and not yolo_pred:
            return None
        
        if tracknet_pred and not yolo_pred:
            return tracknet_pred
        
        if yolo_pred and not tracknet_pred:
            return yolo_pred
        
        # Both predictions exist - combine them
        tracknet_pos = tracknet_pred['position']
        yolo_pos = yolo_pred['position']
        tracknet_conf = tracknet_pred['confidence']
        yolo_conf = yolo_pred['confidence']
        
        # Weighted average based on confidence
        total_conf = tracknet_conf + yolo_conf
        if total_conf > 0:
            combined_x = (tracknet_pos[0] * tracknet_conf + yolo_pos[0] * yolo_conf) / total_conf
            combined_y = (tracknet_pos[1] * tracknet_conf + yolo_pos[1] * yolo_conf) / total_conf
            combined_conf = (tracknet_conf + yolo_conf) / 2
            
            # Apply velocity-based filtering
            if len(self.ball_positions) > 0:
                last_pos = self.ball_positions[-1]['position']
                distance = np.sqrt((combined_x - last_pos[0])**2 + (combined_y - last_pos[1])**2)
                max_velocity = self.config['ball_tracking']['max_velocity']
                
                if distance > max_velocity:
                    # Use the prediction closer to last position
                    tracknet_dist = np.sqrt((tracknet_pos[0] - last_pos[0])**2 + (tracknet_pos[1] - last_pos[1])**2)
                    yolo_dist = np.sqrt((yolo_pos[0] - last_pos[0])**2 + (yolo_pos[1] - last_pos[1])**2)
                    
                    if tracknet_dist < yolo_dist:
                        return tracknet_pred
                    else:
                        return yolo_pred
            
            return {
                'position': [int(combined_x), int(combined_y)],
                'confidence': combined_conf,
                'source': 'combined'
            }
        
        return None
    
    def _calculate_velocity(self, pos1: Dict, pos2: Dict) -> Tuple[float, float]:
        """Calculate velocity between two positions"""
        x1, y1 = pos1['position']
        x2, y2 = pos2['position']
        return (x2 - x1, y2 - y1)
    
    def _draw_ball_tracking(self, frame: np.ndarray, ball_pred: Dict) -> np.ndarray:
        """Draw ball tracking visualization"""
        x, y = ball_pred['position']
        conf = ball_pred['confidence']
        source = ball_pred.get('source', 'unknown')
        
        # Draw ball position
        color = (0, 255, 255) if source == 'combined' else (255, 0, 0) if source == 'tracknet' else (0, 0, 255)
        cv2.circle(frame, (x, y), 8, color, -1)
        cv2.circle(frame, (x, y), 12, color, 2)
        
        # Draw confidence
        cv2.putText(frame, f"{conf:.2f}", (x + 15, y - 15), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Draw source indicator
        cv2.putText(frame, source.upper(), (x - 20, y + 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        # Draw ball trajectory
        if len(self.ball_positions) > 1:
            points = [pred['position'] for pred in list(self.ball_positions)[-10:]]
            for i in range(1, len(points)):
                cv2.line(frame, tuple(points[i-1]), tuple(points[i]), (255, 255, 0), 2)
        
        # Draw velocity vector
        if len(self.ball_velocities) > 0:
            vx, vy = self.ball_velocities[-1]
            if abs(vx) > 1 or abs(vy) > 1:  # Only draw if there's significant movement
                end_x = int(x + vx * 2)  # Scale for visibility
                end_y = int(y + vy * 2)
                cv2.arrowedLine(frame, (x, y), (end_x, end_y), (0, 255, 0), 2)
        
        return frame
    
    def _add_frame_info(self, frame: np.ndarray):
        """Add frame information overlay"""
        # Frame counter
        cv2.putText(frame, f"Frame: {self.analysis_results['total_frames']}", 
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Statistics
        stats_text = [
            f"TrackNet: {self.analysis_results['tracknet_detections']}",
            f"YOLO: {self.analysis_results['yolo_detections']}",
            f"Combined: {self.analysis_results['combined_detections']}",
            f"Ball Positions: {len(self.ball_positions)}"
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
        print("BALL TRACKING ANALYSIS SUMMARY")
        print("="*50)
        print(f"Total frames processed: {self.analysis_results['total_frames']}")
        print(f"TrackNet detections: {self.analysis_results['tracknet_detections']}")
        print(f"YOLO detections: {self.analysis_results['yolo_detections']}")
        print(f"Combined detections: {self.analysis_results['combined_detections']}")
        
        if self.analysis_results['processing_times']:
            avg_time = np.mean(self.analysis_results['processing_times'])
            min_time = np.min(self.analysis_results['processing_times'])
            max_time = np.max(self.analysis_results['processing_times'])
            print(f"Processing time - Avg: {avg_time:.3f}s, Min: {min_time:.3f}s, Max: {max_time:.3f}s")
        
        total_time = time.time() - self.start_time
        print(f"Total analysis time: {total_time:.2f}s")
        print("="*50)


class TrackNetDetector:
    """TrackNet-based ball detection"""
    
    def __init__(self, model_path: str, config: Dict[str, Any]):
        self.config = config
        self.device = 'cpu'
        self.frame_buffer = deque(maxlen=3)  # TrackNet needs 3 consecutive frames
        
        try:
            import torch
            import sys
            sys.path.append('TrackNet')
            from model import BallTrackerNet
            
            # Initialize the TrackNet model
            self.model = BallTrackerNet(out_channels=256).to(self.device)
            
            # Load the pretrained weights
            checkpoint = torch.load(model_path, map_location=self.device)
            
            # Handle different checkpoint formats
            if 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            elif 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            else:
                state_dict = checkpoint
            
            self.model.load_state_dict(state_dict)
            self.model.eval()
            logger.info(f"TrackNet model loaded from {model_path}")
            
        except Exception as e:
            logger.error(f"Error loading TrackNet model: {e}")
            self.model = None
    
    def detect_ball(self, frame: np.ndarray) -> Optional[Dict[str, Any]]:
        """Detect ball using TrackNet"""
        if not self.model:
            return None
        
        # Add frame to buffer
        self.frame_buffer.append(frame)
        
        # Need at least 3 frames for TrackNet
        if len(self.frame_buffer) < 3:
            return None
        
        try:
            import torch
            import sys
            sys.path.append('TrackNet')
            from general import postprocess
            
            # Get the 3 consecutive frames
            current_frame = self.frame_buffer[-1]
            prev_frame = self.frame_buffer[-2]
            prev_prev_frame = self.frame_buffer[-3]
            
            # Preprocess frames for TrackNet
            processed_input = self._preprocess_frames(current_frame, prev_frame, prev_prev_frame)
            
            # Run inference
            with torch.no_grad():
                output = self.model(processed_input, testing=True)
                output = output.argmax(dim=1).detach().cpu().numpy()
            
            # Postprocess to get ball position
            ball_pos = self._postprocess_output(output[0], frame.shape)
            
            if ball_pos:
                return {
                    'position': ball_pos,
                    'confidence': 0.8,  # TrackNet doesn't provide confidence, use default
                    'source': 'tracknet'
                }
            
            return None
            
        except Exception as e:
            logger.error(f"TrackNet detection error: {e}")
            return None
    
    def _preprocess_frames(self, frame1: np.ndarray, frame2: np.ndarray, frame3: np.ndarray):
        """Preprocess 3 consecutive frames for TrackNet input"""
        import torch
        
        # Resize frames to TrackNet input size
        height, width = self.config['input_height'], self.config['input_width']
        frame1_resized = cv2.resize(frame1, (width, height))
        frame2_resized = cv2.resize(frame2, (width, height))
        frame3_resized = cv2.resize(frame3, (width, height))
        
        # Convert to RGB and normalize
        frame1_rgb = cv2.cvtColor(frame1_resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        frame2_rgb = cv2.cvtColor(frame2_resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        frame3_rgb = cv2.cvtColor(frame3_resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        
        # Concatenate frames along channel dimension (9 channels total)
        combined = np.concatenate((frame1_rgb, frame2_rgb, frame3_rgb), axis=2)
        
        # Convert to tensor format (C, H, W) and add batch dimension
        tensor = torch.from_numpy(combined).permute(2, 0, 1).unsqueeze(0).float()
        return tensor
    
    def _postprocess_output(self, output: np.ndarray, original_shape: Tuple[int, int, int]) -> Optional[List[int]]:
        """Postprocess TrackNet output to get ball position"""
        import sys
        sys.path.append('TrackNet')
        from general import postprocess
        
        try:
            # Use the original postprocess function
            x, y = postprocess(output)
            
            if x is not None and y is not None:
                # Convert to original image coordinates
                orig_h, orig_w = original_shape[:2]
                
                # The postprocess function returns coordinates scaled by 2, but we need to scale to original image size
                # TrackNet was trained on 360x640, but postprocess scales by 2, so effective output is 720x1280
                scale_h = orig_h / 720  # 360 * 2 = 720
                scale_w = orig_w / 1280  # 640 * 2 = 1280
                
                x_scaled = int(x * scale_w)
                y_scaled = int(y * scale_h)
                
                # Ensure coordinates are within bounds
                x_scaled = max(0, min(x_scaled, orig_w - 1))
                y_scaled = max(0, min(y_scaled, orig_h - 1))
                
                logger.info(f"TrackNet ball detection: pos=({x_scaled}, {y_scaled})")
                return [x_scaled, y_scaled]
            
            return None
            
        except Exception as e:
            logger.error(f"Postprocess error: {e}")
            return None
    

    



class YOLOBallDetector:
    """YOLO-based ball detection"""
    
    def __init__(self, model_path: str, config: Dict[str, Any]):
        self.config = config
        try:
            from ultralytics import YOLO
            self.model = YOLO(model_path)
            logger.info(f"YOLO ball detector initialized with {model_path}")
        except ImportError:
            logger.error("Ultralytics not installed. Install with: pip install ultralytics")
            self.model = None
        except Exception as e:
            logger.error(f"Error loading YOLO model: {e}")
            self.model = None
    
    def detect_ball(self, frame: np.ndarray) -> Optional[Dict[str, Any]]:
        """Detect ball using YOLO"""
        if not self.model:
            return None
        
        try:
            results = self.model(
                frame,
                conf=self.config.get('conf_threshold', 0.3),
                iou=self.config.get('iou_threshold', 0.45),
                max_det=self.config.get('max_det', 10),
                verbose=False
            )
            
            # Debug: Log all detections
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    logger.debug(f"YOLO found {len(boxes)} detections")
                    for box in boxes:
                        cls = int(box.cls[0].cpu().numpy())
                        conf = float(box.conf[0].cpu().numpy())
                        logger.debug(f"YOLO detection: class={cls}, conf={conf:.3f}")
                        
                        # Look for ball class (class 0) - but also try class 1 if 0 doesn't work
                        if (cls == 0 or cls == 1) and conf > self.config.get('conf_threshold', 0.3):
                            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                            center_x = int((x1 + x2) / 2)
                            center_y = int((y1 + y2) / 2)
                            
                            # Calculate bounding box area to filter out large detections (likely players)
                            bbox_area = (x2 - x1) * (y2 - y1)
                            if bbox_area < 5000:  # Only small detections (likely balls)
                                logger.info(f"YOLO ball detection: class={cls}, conf={conf:.3f}, pos=({center_x}, {center_y}), area={bbox_area}")
                                return {
                                    'position': [center_x, center_y],
                                    'confidence': conf,
                                    'source': 'yolo',
                                    'bbox': [int(x1), int(y1), int(x2), int(y2)]
                                }
            
            return None
            
        except Exception as e:
            logger.error(f"YOLO detection error: {e}")
            return None


def main():
    """Main function to run the ball tracking demo"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Ball Tracking Demo')
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
    
    print("🏀 Ball Tracking Demo Starting...")
    print(f"📹 Input video: {args.video}")
    if args.output:
        print(f"💾 Output video: {args.output}")
    print(f"⚙️  Config file: {args.config}")
    print("\nControls:")
    print("  - Press 'q' to quit")
    print("  - Press 'p' to pause/resume")
    print("  - Press 's' to save current frame")
    print("\nInitializing models...")
    
    # Create and run tracker
    tracker = BallTracker(args.config)
    tracker.track_ball(args.video, args.output)


if __name__ == "__main__":
    main()
