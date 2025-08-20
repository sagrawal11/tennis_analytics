#!/usr/bin/env python3
"""
Custom PyTorch Ball and Player Detector
A detector for custom trained models (like playersnball5.pt)
"""

import cv2
import numpy as np
import torch
import logging
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CustomPyTorchDetector:
    """Custom PyTorch detector for players and ball detection"""
    
    def __init__(self, model_path: str, device: str = "auto"):
        """Initialize custom PyTorch detector"""
        self.model_path = model_path
        self.device = self._get_device(device)
        self.model = None
        self.transform = None
        self.class_names = ["ball", "player"]  # Custom class names
        
        try:
            self._load_model()
            logger.info(f"Custom PyTorch detector initialized with {model_path}")
        except Exception as e:
            logger.error(f"Failed to initialize custom PyTorch detector: {e}")
            self.model = None
    
    def _get_device(self, device: str) -> str:
        """Determine the best available device"""
        if device == "auto":
            if torch.cuda.is_available():
                return "cuda"
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                return "mps"
            else:
                return "cpu"
        return device
    
    def _load_model(self):
        """Load custom PyTorch model"""
        try:
            # Load the checkpoint
            checkpoint = torch.load(self.model_path, map_location=self.device)
            logger.info(f"Loaded checkpoint with keys: {checkpoint.keys()}")
            
            # Extract model and args
            if 'model' in checkpoint:
                # The model is stored as a state dict, we need to reconstruct the architecture
                # For now, let's try to use it directly as a model (some checkpoints work this way)
                try:
                    self.model = checkpoint['model']
                    # Check if it has the necessary methods for inference
                    if hasattr(self.model, 'forward'):
                        logger.info("Model has forward method, using directly")
                    else:
                        logger.warning("Model doesn't have forward method, this may not work")
                except Exception as e:
                    logger.error(f"Error using model directly: {e}")
                    raise
                
                if 'args' in checkpoint:
                    self.args = checkpoint['args']
                    logger.info(f"Model args: {self.args}")
                    logger.info(f"Class names: {getattr(self.args, 'class_names', 'Unknown')}")
            else:
                # Assume the checkpoint is the model itself
                self.model = checkpoint
            
            # Move to device
            try:
                self.model.to(self.device)
                self.model.eval()
                logger.info(f"Custom PyTorch model loaded successfully on {self.device}")
            except Exception as e:
                logger.warning(f"Could not move model to device: {e}")
                # Try to use the model as-is
                pass
            
        except Exception as e:
            logger.error(f"Error loading custom PyTorch model: {e}")
            raise
    
    def _preprocess_frame(self, frame: np.ndarray) -> torch.Tensor:
        """Preprocess frame for model input"""
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Normalize to [0, 1] and convert to tensor
        frame_tensor = torch.from_numpy(frame_rgb).float() / 255.0
        
        # Add batch dimension and move to device
        frame_tensor = frame_tensor.unsqueeze(0).to(self.device)
        
        # Ensure correct shape (B, H, W, C) -> (B, C, H, W) if needed
        if frame_tensor.shape[-1] == 3:
            frame_tensor = frame_tensor.permute(0, 3, 1, 2)
        
        return frame_tensor
    
    def detect(self, frame: np.ndarray, conf_threshold: float = 0.5) -> List[Dict]:
        """Detect objects in frame using custom PyTorch model"""
        if self.model is None:
            logger.warning("Custom PyTorch model not loaded")
            return []
        
        try:
            # Preprocess frame
            input_tensor = self._preprocess_frame(frame)
            
            # Run inference
            with torch.no_grad():
                outputs = self.model(input_tensor)
            
            # Process outputs (this will depend on your model's output format)
            # For now, let's assume a standard detection output format
            detections = self._process_outputs(outputs, conf_threshold, frame.shape)
            
            logger.debug(f"Custom PyTorch model detected {len(detections)} objects")
            return detections
            
        except Exception as e:
            logger.error(f"Error in custom PyTorch detection: {e}")
            return []
    
    def _process_outputs(self, outputs, conf_threshold: float, frame_shape) -> List[Dict]:
        """Process model outputs to extract detections"""
        detections = []
        
        try:
            # This is a placeholder - you'll need to adapt this based on your model's output format
            # Common formats include:
            # - YOLO-style: [batch, num_detections, 6] where 6 = [x, y, w, h, conf, class]
            # - DETR-style: [batch, num_queries, num_classes + 4] where 4 = [x1, y1, x2, y2]
            
            if isinstance(outputs, (list, tuple)):
                # Handle multi-output models
                outputs = outputs[0]
            
            if isinstance(outputs, torch.Tensor):
                # Handle tensor outputs
                if outputs.dim() == 3:  # [batch, num_detections, features]
                    batch_size, num_detections, features = outputs.shape
                    
                    for i in range(num_detections):
                        detection = outputs[0, i]  # First batch
                        
                        if features >= 6:  # YOLO-style output
                            x, y, w, h, conf, cls = detection[:6]
                            
                            if conf >= conf_threshold:
                                # Convert center coordinates to corners
                                x1 = int(x - w/2)
                                y1 = int(y - h/2)
                                x2 = int(x + w/2)
                                y2 = int(y + h/2)
                                
                                detections.append({
                                    'bbox': [x1, y1, x2, y2],
                                    'class': int(cls),
                                    'confidence': float(conf),
                                    'class_name': self.class_names[int(cls)] if int(cls) < len(self.class_names) else 'unknown'
                                })
                
                elif outputs.dim() == 2:  # [num_detections, features]
                    num_detections, features = outputs.shape
                    
                    for i in range(num_detections):
                        detection = outputs[i]
                        
                        if features >= 6:  # YOLO-style output
                            x, y, w, h, conf, cls = detection[:6]
                            
                            if conf >= conf_threshold:
                                # Convert center coordinates to corners
                                x1 = int(x - w/2)
                                y1 = int(y - h/2)
                                x2 = int(x + w/2)
                                y2 = int(y + h/2)
                                
                                detections.append({
                                    'bbox': [x1, y1, x2, y2],
                                    'class': int(cls),
                                    'confidence': float(conf),
                                    'class_name': self.class_names[int(cls)] if int(cls) < len(self.class_names) else 'unknown'
                                })
            
            logger.info(f"Processed outputs shape: {outputs.shape if hasattr(outputs, 'shape') else 'unknown'}")
            
        except Exception as e:
            logger.error(f"Error processing outputs: {e}")
            logger.info(f"Outputs type: {type(outputs)}")
            if hasattr(outputs, 'shape'):
                logger.info(f"Outputs shape: {outputs.shape}")
        
        return detections
    
    def detect_players(self, frame: np.ndarray, conf_threshold: float = 0.5) -> List[Dict]:
        """Detect only players in frame"""
        detections = self.detect(frame, conf_threshold)
        # Filter for player class (assuming player is class 1)
        player_detections = [d for d in detections if d['class'] == 1]
        return player_detections
    
    def detect_ball(self, frame: np.ndarray, conf_threshold: float = 0.3) -> List[Dict]:
        """Detect only ball in frame"""
        detections = self.detect(frame, conf_threshold)
        # Filter for ball class (assuming ball is class 0)
        ball_detections = [d for d in detections if d['class'] == 0]
        return ball_detections
    
    def draw_detections(self, frame: np.ndarray, detections: List[Dict]) -> np.ndarray:
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


def test_custom_detector():
    """Test custom PyTorch detector with a sample image or video"""
    model_path = "models/playersnball5.pt"
    
    if not Path(model_path).exists():
        logger.error(f"Model not found: {model_path}")
        return
    
    # Initialize detector
    detector = CustomPyTorchDetector(model_path)
    
    if detector.model is None:
        logger.error("Failed to initialize detector")
        return
    
    # Test with webcam or sample image
    cap = cv2.VideoCapture(0)  # Use 0 for webcam
    
    if not cap.isOpened():
        logger.info("Webcam not available, trying to open sample image...")
        # You can replace this with a sample image path
        sample_image = np.zeros((480, 640, 3), dtype=np.uint8)
        sample_image[:] = (128, 128, 128)  # Gray image
        
        # Test detection on sample image
        detections = detector.detect(sample_image)
        logger.info(f"Sample image detections: {detections}")
        
        # Draw and display
        result_image = detector.draw_detections(sample_image, detections)
        cv2.imshow("Custom PyTorch Test", result_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return
    
    logger.info("Testing Custom PyTorch detector with webcam. Press 'q' to quit.")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Run detection
        start_time = time.time()
        detections = detector.detect(frame)
        inference_time = time.time() - start_time
        
        # Draw detections
        result_frame = detector.draw_detections(frame, detections)
        
        # Add FPS info
        fps_text = f"FPS: {1.0/inference_time:.1f}"
        cv2.putText(result_frame, fps_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Display
        cv2.imshow("Custom PyTorch Test", result_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    test_custom_detector()
