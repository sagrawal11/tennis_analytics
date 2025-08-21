#!/usr/bin/env python3
"""
Debug script to test ball detection logic specifically
"""

import cv2
import torch
import logging
from rfdetr import RFDETRNano

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_ball_detection_logic():
    """Test the exact ball detection logic from tennis_CV.py"""
    
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
            
            # Test different confidence thresholds
            thresholds = [0.1, 0.2, 0.3, 0.4, 0.5]
            
            for threshold in thresholds:
                logger.info(f"\n🔍 Testing threshold: {threshold}")
                
                # Run inference
                detections = model.predict(pil_image, threshold=threshold)
                logger.info(f"Total detections: {len(detections.xyxy)}")
                
                # Check all detections
                balls = []
                players = []
                
                for i in range(len(detections.xyxy)):
                    bbox = detections.xyxy[i]
                    confidence = detections.confidence[i]
                    class_id = detections.class_id[i]
                    
                    logger.info(f"  Detection {i}: class_id={class_id}, confidence={confidence:.3f}")
                    
                    # Test our ball detection logic
                    if class_id == 1 and confidence > threshold:  # Ball
                        x1, y1, x2, y2 = bbox
                        center_x = int((x1 + x2) / 2)
                        center_y = int((y1 + y2) / 2)
                        balls.append({
                            'position': [center_x, center_y],
                            'confidence': float(confidence),
                            'bbox': [int(x1), int(y1), int(x2), int(y2)]
                        })
                        logger.info(f"    ✅ BALL detected at ({center_x}, {center_y})")
                    
                    # Test our player detection logic  
                    elif class_id == 2 and confidence > threshold:  # Player
                        x1, y1, x2, y2 = bbox
                        center_x = int((x1 + x2) / 2)
                        center_y = int((y1 + y2) / 2)
                        players.append({
                            'bbox': [int(x1), int(y1), int(x2), int(y2)],
                            'confidence': float(confidence),
                            'center': [center_x, center_y]
                        })
                        logger.info(f"    ✅ PLAYER detected at ({center_x}, {center_y})")
                
                logger.info(f"  Summary: {len(balls)} balls, {len(players)} players at threshold {threshold}")
                
                if balls:
                    logger.info(f"  🎯 BALLS FOUND! RF-DETR should be working as primary!")
                    break
            
        else:
            logger.error("Invalid checkpoint format")
            
    except Exception as e:
        logger.error(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_ball_detection_logic()
