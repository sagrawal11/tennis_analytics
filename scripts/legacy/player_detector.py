"""
Player Detection Module using YOLOv8
Detects tennis players in video frames
"""

import cv2
import numpy as np
from ultralytics import YOLO
from typing import List, Tuple, Dict, Any
import logging

logger = logging.getLogger(__name__)

class PlayerDetector:
    """YOLOv8-based player detection for tennis analysis"""
    
    def __init__(self, model_path: str, config: Dict[str, Any]):
        """
        Initialize player detector
        
        Args:
            model_path: Path to YOLOv8 model weights
            config: Configuration dictionary
        """
        self.config = config
        self.model = YOLO(model_path)
        self.conf_threshold = config.get('conf_threshold', 0.5)
        self.iou_threshold = config.get('iou_threshold', 0.45)
        self.max_det = config.get('max_det', 10)
        
        logger.info(f"Player detector initialized with model: {model_path}")
    
    def detect_players(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """
        Detect players in a single frame
        
        Args:
            frame: Input frame (BGR format)
            
        Returns:
            List of player detections with bounding boxes and confidence scores
        """
        try:
            # Run inference
            results = self.model(
                frame,
                conf=self.conf_threshold,
                iou=self.iou_threshold,
                max_det=self.max_det,
                verbose=False
            )
            
            detections = []
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        # Get bounding box coordinates
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        conf = box.conf[0].cpu().numpy()
                        cls = int(box.cls[0].cpu().numpy())
                        
                        # Only keep player detections (assuming class 1 is player)
                        if cls == 1:  # Player class
                            detection = {
                                'bbox': [int(x1), int(y1), int(x2), int(y2)],
                                'confidence': float(conf),
                                'class': cls,
                                'center': [int((x1 + x2) / 2), int((y1 + y2) / 2)]
                            }
                            detections.append(detection)
            
            logger.debug(f"Detected {len(detections)} players")
            return detections
            
        except Exception as e:
            logger.error(f"Error in player detection: {e}")
            return []
    
    def draw_detections(self, frame: np.ndarray, detections: List[Dict[str, Any]]) -> np.ndarray:
        """
        Draw player detection bounding boxes on frame
        
        Args:
            frame: Input frame
            detections: List of player detections
            
        Returns:
            Frame with drawn detections
        """
        frame_copy = frame.copy()
        
        for detection in detections:
            x1, y1, x2, y2 = detection['bbox']
            conf = detection['confidence']
            
            # Draw bounding box
            cv2.rectangle(frame_copy, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw confidence score
            label = f"Player: {conf:.2f}"
            cv2.putText(frame_copy, label, (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # Draw center point
            center_x, center_y = detection['center']
            cv2.circle(frame_copy, (center_x, center_y), 3, (255, 0, 0), -1)
        
        return frame_copy
    
    def filter_detections_by_confidence(self, detections: List[Dict[str, Any]], 
                                      min_confidence: float) -> List[Dict[str, Any]]:
        """
        Filter detections by minimum confidence threshold
        
        Args:
            detections: List of detections
            min_confidence: Minimum confidence threshold
            
        Returns:
            Filtered list of detections
        """
        return [det for det in detections if det['confidence'] >= min_confidence]
    
    def get_player_rois(self, frame: np.ndarray, detections: List[Dict[str, Any]]) -> List[np.ndarray]:
        """
        Extract regions of interest (ROIs) for detected players
        
        Args:
            frame: Input frame
            detections: List of player detections
            
        Returns:
            List of player ROIs
        """
        rois = []
        for detection in detections:
            x1, y1, x2, y2 = detection['bbox']
            roi = frame[y1:y2, x1:x2]
            if roi.size > 0:  # Check if ROI is valid
                rois.append(roi)
        
        return rois
