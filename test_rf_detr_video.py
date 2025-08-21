#!/usr/bin/env python3
"""
Test RF-DETR model on tennis_test5.mp4 video using official package
Compare ball detection performance with current system
"""

import cv2
import numpy as np
import torch
import logging
from pathlib import Path
from typing import List, Dict
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RFDETRVideoTester:
    """Test RF-DETR model on video for ball detection using official package"""
    
    def __init__(self, model_path: str, video_path: str):
        """Initialize the video tester"""
        self.model_path = model_path
        self.video_path = video_path
        self.model = None
        
        try:
            self._load_model()
            logger.info(f"RF-DETR model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def _load_model(self):
        """Load the RF-DETR model with custom tennis weights"""
        try:
            # Load checkpoint first to get configuration
            logger.info(f"Loading checkpoint from {self.model_path}")
            checkpoint = torch.load(self.model_path, map_location='cpu')
            
            if 'args' in checkpoint and 'model' in checkpoint:
                # Extract configuration from checkpoint
                args = checkpoint['args']
                logger.info(f"Found custom tennis model with {args.num_classes} classes")
                logger.info(f"Class names: {args.class_names}")
                
                # Import RF-DETR with custom configuration
                from rfdetr import RFDETRNano  # Use Nano since that's what our model appears to be
                
                # Create model with custom classes (RF-DETR adds background automatically)
                # Our checkpoint has 3 classes total (background + ball + player)
                # But RF-DETR expects num_classes to be just the object classes (ball + player = 2)
                self.model = RFDETRNano(
                    num_classes=len(args.class_names),  # 2 classes: ball + player
                    pretrain_weights=None  # Don't load default weights
                )
                
                # Load our custom state dict into the underlying PyTorch model
                missing_keys, unexpected_keys = self.model.model.model.load_state_dict(checkpoint['model'], strict=False)
                if missing_keys:
                    logger.warning(f"Missing keys when loading model: {missing_keys[:5]}...")  # Show first 5
                if unexpected_keys:
                    logger.warning(f"Unexpected keys when loading model: {unexpected_keys[:5]}...")  # Show first 5
                
                # Set class names (try different approaches)
                try:
                    self.model.class_names = args.class_names
                except:
                    # Fallback: manually set it
                    self.model.model.class_names = args.class_names
                
                # Store args for later use
                self.args = args
                logger.info("Custom tennis RF-DETR model loaded successfully!")
                
            else:
                logger.error("Invalid checkpoint format - missing 'args' or 'model'")
                raise ValueError("Invalid checkpoint format")
                
        except ImportError:
            logger.error("RF-DETR package not found. Install with: pip install rfdetr")
            raise
        except Exception as e:
            logger.error(f"Error loading RF-DETR model: {e}")
            raise
    
    def detect_objects(self, frame: np.ndarray, conf_threshold: float = 0.3) -> List[Dict]:
        """Detect objects in frame using RF-DETR"""
        if self.model is None:
            return []
        
        try:
            # Convert BGR to RGB (RF-DETR expects RGB)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Convert to PIL Image (RF-DETR expects PIL)
            from PIL import Image
            pil_image = Image.fromarray(frame_rgb)
            
            # Run inference
            detections = self.model.predict(pil_image, threshold=conf_threshold)
            
            # Use our custom tennis class names
            class_names = self.model.class_names
            logger.info(f"Using tennis classes: {class_names}")
            

            
            # Convert to our format
            converted_detections = []
            for i in range(len(detections.xyxy)):
                bbox = detections.xyxy[i]
                confidence = detections.confidence[i]
                class_id = detections.class_id[i]
                
                # Get class name - handle the mapping correctly
                if class_id == 0:  # Background class
                    continue  # Skip background detections
                elif class_id in class_names:
                    class_name = class_names[class_id]
                else:
                    class_name = f"unknown_{class_id}"
                
                # Convert to our format
                detection = {
                    'bbox': [int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])],
                    'class': int(class_id),
                    'confidence': float(confidence),
                    'class_name': class_name
                }
                converted_detections.append(detection)
            
            return converted_detections
            
        except Exception as e:
            logger.error(f"Error in detection: {e}")
            return []
    
    def _filter_detections(self, detections: List[Dict]) -> List[Dict]:
        """Filter detections to only keep exactly 2 players and 1 ball"""
        # Separate detections by type
        players = []
        balls = []
        
        for detection in detections:
            class_name = detection['class_name']
            confidence = detection['confidence']
            bbox = detection['bbox']
            
            if class_name == 'player' and confidence > 0.3:  # Tennis players
                x1, y1, x2, y2 = bbox
                bbox_center_x = (x1 + x2) / 2
                bbox_center_y = (y1 + y2) / 2
                
                # Court position scoring - players should be in center area
                frame_center_x = 1920 / 2
                frame_center_y = 1080 / 2
                distance_from_center = abs(bbox_center_x - frame_center_x) + abs(bbox_center_y - frame_center_y)
                
                # Combined score: confidence + court position
                detection['court_score'] = confidence - (distance_from_center / 2000)
                players.append(detection)
                
            elif class_name == 'ball' and confidence > 0.2:  # Tennis ball (lower threshold)
                balls.append(detection)
        
        # Select best 2 players and 1 ball
        filtered = []
        
        # Keep top 2 players by court score
        players.sort(key=lambda x: x['court_score'], reverse=True)
        filtered.extend(players[:2])
        
        # Keep highest confidence ball
        if balls:
            balls.sort(key=lambda x: x['confidence'], reverse=True)
            filtered.append(balls[0])
        
        return filtered
    
    def test_video(self, output_path: str = None):
        """Test the model on the video"""
        if not Path(self.video_path).exists():
            logger.error(f"Video not found: {self.video_path}")
            return
        
        # Open video
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            logger.error(f"Could not open video: {self.video_path}")
            return
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        logger.info(f"Video: {width}x{height}, {fps} FPS, {total_frames} frames")
        
        # Setup output video if specified
        out = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_count = 0
        start_time = time.time()
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Run detection
            detection_start = time.time()
            detections = self.detect_objects(frame)
            detection_time = time.time() - detection_start
            
            # Filter detections
            filtered_detections = self._filter_detections(detections)
            
            # Draw detections
            result_frame = self._draw_detections(frame, filtered_detections)
            
            # Add info overlay
            info_text = f"Frame: {frame_count}/{total_frames} | Detection: {detection_time*1000:.1f}ms | Objects: {len(filtered_detections)}"
            cv2.putText(result_frame, info_text, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Display
            cv2.imshow("RF-DETR Test", result_frame)
            
            # Write to output if specified
            if out:
                out.write(result_frame)
            
            # Progress update
            if frame_count % 30 == 0:
                elapsed = time.time() - start_time
                fps_actual = frame_count / elapsed
                logger.info(f"Progress: {frame_count}/{total_frames} | FPS: {fps_actual:.1f} | Objects detected: {len(filtered_detections)}")
            
            # Check for quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        # Cleanup
        cap.release()
        if out:
            out.release()
        cv2.destroyAllWindows()
        
        # Final stats
        total_time = time.time() - start_time
        avg_fps = frame_count / total_time
        logger.info(f"Completed: {frame_count} frames in {total_time:.1f}s | Avg FPS: {avg_fps:.1f}")
    
    def _draw_detections(self, frame: np.ndarray, detections: List[Dict]) -> np.ndarray:
        """Draw detection boxes on frame"""
        frame_copy = frame.copy()
        
        # Define colors for our custom tennis classes
        class_colors = {
            'player': (0, 255, 0),      # Green for tennis players
            'ball': (0, 255, 255),      # Yellow for tennis ball
        }
        
        for detection in detections:
            bbox = detection['bbox']
            x1, y1, x2, y2 = bbox
            confidence = detection['confidence']
            class_name = detection['class_name']
            
            # Choose color based on class
            color = class_colors.get(class_name, (128, 128, 128))  # Gray for unknown
            
            # Draw bounding box
            cv2.rectangle(frame_copy, (x1, y1), (x2, y2), color, 2)
            
            # Create clean label
            label = f"{class_name.title()}: {confidence:.2f}"
            
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            
            # Draw label background
            cv2.rectangle(frame_copy, (x1, y1 - label_size[1] - 10), 
                         (x1 + label_size[0], y1), color, -1)
            
            # Draw label text
            cv2.putText(frame_copy, label, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        
        return frame_copy


def main():
    """Main function to test RF-DETR on tennis video"""
    model_path = "models/playersnball5.pt"
    video_path = "tennis_test5.mp4"
    output_path = "rf_detr_official_test_output.mp4"
    
    if not Path(model_path).exists():
        logger.error(f"Model not found: {model_path}")
        return
    
    if not Path(video_path).exists():
        logger.error(f"Video not found: {video_path}")
        return
    
    try:
        # Initialize tester
        tester = RFDETRVideoTester(model_path, video_path)
        
        # Test on video
        logger.info("Starting RF-DETR video test with official package...")
        tester.test_video(output_path)
        
    except Exception as e:
        logger.error(f"Test failed: {e}")


if __name__ == "__main__":
    main()
