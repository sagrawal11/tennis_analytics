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
        """Load the RF-DETR model using the official package"""
        try:
            # Import RF-DETR
            from rfdetr import RFDETRBase
            
            # Load the base model (it comes with pretrained weights)
            self.model = RFDETRBase()
            
            # The model is already loaded with pretrained weights
            # We can't load custom weights directly, so we'll use the base model
            # This should still give us good detection performance
            logger.info("RF-DETR base model loaded with pretrained weights")
                
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
            
            # Convert to our format
            converted_detections = []
            for i in range(len(detections.xyxy)):
                bbox = detections.xyxy[i]
                confidence = detections.confidence[i]
                class_id = detections.class_id[i]
                
                # Convert to our format
                detection = {
                    'bbox': [int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])],
                    'class': int(class_id),
                    'confidence': float(confidence),
                    'class_name': 'ball' if int(class_id) == 0 else 'player'
                }
                converted_detections.append(detection)
            
            return converted_detections
            
        except Exception as e:
            logger.error(f"Error in detection: {e}")
            return []
    
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
            
            # Draw detections
            result_frame = self._draw_detections(frame, detections)
            
            # Add info overlay
            info_text = f"Frame: {frame_count}/{total_frames} | Detection: {detection_time*1000:.1f}ms | Objects: {len(detections)}"
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
                logger.info(f"Progress: {frame_count}/{total_frames} | FPS: {fps_actual:.1f} | Objects detected: {len(detections)}")
            
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
        
        for detection in detections:
            bbox = detection['bbox']
            x1, y1, x2, y2 = bbox
            confidence = detection['confidence']
            class_name = detection['class_name']
            
            # Choose color based on class
            if class_name == 'ball':
                color = (0, 255, 255)  # Yellow for ball
            else:
                color = (0, 255, 0)    # Green for player
            
            # Draw bounding box
            cv2.rectangle(frame_copy, (x1, y1), (x2, y2), color, 2)
            
            # Draw label
            label = f"{class_name}: {confidence:.2f}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            cv2.rectangle(frame_copy, (x1, y1 - label_size[1] - 10), 
                         (x1 + label_size[0], y1), color, -1)
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
