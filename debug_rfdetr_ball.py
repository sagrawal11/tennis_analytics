#!/usr/bin/env python3
"""
Debug script to test RF-DETR ball detection
"""

import cv2
import torch
import logging
from rfdetr import RFDETRNano

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_rfdetr_ball_detection():
    """Test RF-DETR ball detection on a single frame"""
    
    # Load the model
    model_path = 'models/playersnball5.pt'
    logger.info(f"Loading model from {model_path}")
    
    try:
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location='cpu')
        if 'args' in checkpoint and 'model' in checkpoint:
            args = checkpoint['args']
            logger.info(f"Model classes: {args.class_names}")
            logger.info(f"Num classes: {args.num_classes}")
            
            # Create model
            model = RFDETRNano(
                num_classes=len(args.class_names),
                pretrain_weights=None
            )
            
            # Load weights
            missing_keys, unexpected_keys = model.model.model.load_state_dict(checkpoint['model'], strict=False)
            logger.info(f"Missing keys: {len(missing_keys)}")
            logger.info(f"Unexpected keys: {len(unexpected_keys)}")
            
            # Set class names
            try:
                model.class_names = args.class_names
            except:
                model.model.class_names = args.class_names
            
            # Load a test frame
            cap = cv2.VideoCapture('tennis_test5.mp4')
            ret, frame = cap.read()
            cap.release()
            
            if not ret:
                logger.error("Could not read frame from video")
                return
            
            logger.info(f"Frame shape: {frame.shape}")
            
            # Convert BGR to RGB and to PIL
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            from PIL import Image
            pil_image = Image.fromarray(frame_rgb)
            
            # Run inference
            logger.info("Running RF-DETR inference...")
            detections = model.predict(pil_image, threshold=0.1)  # Low threshold to see all detections
            
            logger.info(f"Total detections: {len(detections.xyxy)}")
            
            # Check all detections
            for i in range(len(detections.xyxy)):
                bbox = detections.xyxy[i]
                confidence = detections.confidence[i]
                class_id = detections.class_id[i]
                
                logger.info(f"Detection {i}: class_id={class_id}, confidence={confidence:.3f}, bbox={bbox}")
                
                # Try to get class name
                try:
                    if hasattr(model, 'class_names') and class_id in model.class_names:
                        class_name = model.class_names[class_id]
                    elif hasattr(model.model, 'class_names') and class_id in model.model.class_names:
                        class_name = model.model.class_names[class_id]
                    else:
                        class_name = f"unknown_{class_id}"
                    logger.info(f"  Class name: {class_name}")
                except:
                    logger.info(f"  Could not get class name")
            
        else:
            logger.error("Invalid checkpoint format")
            
    except Exception as e:
        logger.error(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_rfdetr_ball_detection()
