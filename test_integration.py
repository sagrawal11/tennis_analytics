#!/usr/bin/env python3
"""
Test script to verify RF-DETR integration in tennis_CV.py
"""

import sys
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_rfdetr_import():
    """Test if RF-DETR can be imported"""
    try:
        from rfdetr import RFDETRNano
        logger.info("✅ RF-DETR import successful")
        return True
    except ImportError as e:
        logger.error(f"❌ RF-DETR import failed: {e}")
        return False

def test_tennis_cv_import():
    """Test if tennis_CV.py can be imported with RF-DETR classes"""
    try:
        # Import the classes we added
        from tennis_CV import RFDETRPlayerDetector, RFDETRBallDetector
        logger.info("✅ RF-DETR classes import successful")
        return True
    except ImportError as e:
        logger.error(f"❌ RF-DETR classes import failed: {e}")
        return False

def test_model_loading():
    """Test if the RF-DETR model can be loaded"""
    try:
        import torch
        checkpoint = torch.load('models/playersnball5.pt', map_location='cpu')
        
        if 'args' in checkpoint and 'model' in checkpoint:
            args = checkpoint['args']
            logger.info(f"✅ Model checkpoint loaded successfully")
            logger.info(f"   Classes: {args.class_names}")
            logger.info(f"   Num classes: {args.num_classes}")
            return True
        else:
            logger.error("❌ Invalid checkpoint format")
            return False
    except Exception as e:
        logger.error(f"❌ Model loading failed: {e}")
        return False

def main():
    """Run all tests"""
    logger.info("🧪 Testing RF-DETR Integration...")
    
    tests = [
        ("RF-DETR Import", test_rfdetr_import),
        ("Tennis CV Classes", test_tennis_cv_import),
        ("Model Loading", test_model_loading),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        logger.info(f"\n--- Testing: {test_name} ---")
        if test_func():
            passed += 1
            logger.info(f"✅ {test_name}: PASSED")
        else:
            logger.error(f"❌ {test_name}: FAILED")
    
    logger.info(f"\n🎯 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 All tests passed! RF-DETR integration is ready!")
        return True
    else:
        logger.error("💥 Some tests failed. Check the errors above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
