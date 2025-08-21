#!/usr/bin/env python3
"""
Debug script to test RF-DETR model inference and identify hanging issues
"""

import cv2
import numpy as np
import time
import signal
import sys
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TimeoutError(Exception):
    pass

def timeout_handler(signum, frame):
    raise TimeoutError("Operation timed out")

def test_rfdetr_inference(video_path: str, model_path: str, timeout_seconds: int = 30):
    """Test RF-DETR inference with timeout protection"""
    
    logger.info(f"🎾 Testing RF-DETR inference on video: {video_path}")
    logger.info(f"🔧 Model path: {model_path}")
    logger.info(f"⏰ Timeout: {timeout_seconds} seconds")
    
    # Check if files exist
    if not Path(video_path).exists():
        logger.error(f"❌ Video file not found: {video_path}")
        return False
    
    if not Path(model_path).exists():
        logger.error(f"❌ Model file not found: {model_path}")
        return False
    
    # Load video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error(f"❌ Could not open video: {video_path}")
        return False
    
    # Load RF-DETR model
    try:
        logger.info("🔧 Loading RF-DETR model...")
        from rfdetr import RFDETRNano
        import torch
        
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location='cpu')
        if 'args' in checkpoint and 'model' in checkpoint:
            args = checkpoint['args']
            logger.info(f"✅ RF-DETR model loaded: {args.num_classes} classes: {args.class_names}")
            
            # Create model
            model = RFDETRNano(
                num_classes=len(args.class_names),
                pretrain_weights=None
            )
            
            # Load weights
            model.load_state_dict(checkpoint['model'])
            model.eval()
            logger.info("✅ RF-DETR model weights loaded successfully")
        else:
            logger.error("❌ Invalid checkpoint format")
            return False
            
    except Exception as e:
        logger.error(f"❌ Failed to load RF-DETR model: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test inference on first few frames
    frame_count = 0
    max_test_frames = 10
    
    while frame_count < max_test_frames:
        ret, frame = cap.read()
        if not ret:
            logger.info("🔍 End of video reached")
            break
        
        logger.info(f"🔍 Testing frame {frame_count + 1}...")
        
        try:
            # Set timeout for inference
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(timeout_seconds)
            
            # Convert frame
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            from PIL import Image
            pil_image = Image.fromarray(frame_rgb)
            
            logger.info(f"🔍 Frame {frame_count + 1}: Starting inference...")
            start_time = time.time()
            
            # Run inference with timeout
            detections = model.predict(pil_image, threshold=0.3)
            
            # Cancel timeout
            signal.alarm(0)
            
            inference_time = time.time() - start_time
            logger.info(f"✅ Frame {frame_count + 1}: Inference completed in {inference_time:.3f}s")
            
            # Check detections
            if hasattr(detections, 'xyxy') and len(detections.xyxy) > 0:
                logger.info(f"🔍 Frame {frame_count + 1}: Found {len(detections.xyxy)} detections")
                for i in range(min(3, len(detections.xyxy))):  # Show first 3 detections
                    bbox = detections.xyxy[i]
                    confidence = detections.confidence[i]
                    class_id = detections.class_id[i]
                    logger.info(f"  Detection {i+1}: class={class_id}, conf={confidence:.3f}, bbox={bbox}")
            else:
                logger.info(f"🔍 Frame {frame_count + 1}: No detections found")
            
        except TimeoutError:
            logger.error(f"❌ Frame {frame_count + 1}: Inference timed out after {timeout_seconds}s")
            signal.alarm(0)
            break
        except Exception as e:
            logger.error(f"❌ Frame {frame_count + 1}: Inference failed: {e}")
            signal.alarm(0)
            import traceback
            traceback.print_exc()
            break
        
        frame_count += 1
    
    cap.release()
    logger.info(f"✅ RF-DETR inference test completed. Processed {frame_count} frames.")
    return frame_count > 0

def main():
    """Main function"""
    video_path = "tennis_test5.mp4"
    model_path = "models/playersnball5.pt"
    
    logger.info("🚀 Starting RF-DETR inference debug test...")
    
    success = test_rfdetr_inference(video_path, model_path, timeout_seconds=30)
    
    if success:
        logger.info("✅ RF-DETR inference test passed!")
    else:
        logger.error("❌ RF-DETR inference test failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()
