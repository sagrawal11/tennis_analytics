#!/usr/bin/env python3
"""
Test script to validate the tennis analytics system
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

def test_imports():
    """Test that all modules can be imported"""
    print("Testing module imports...")
    
    try:
        from player_detector import PlayerDetector
        print("✓ PlayerDetector imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import PlayerDetector: {e}")
        return False
    
    try:
        from pose_estimator import PoseEstimator
        print("✓ PoseEstimator imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import PoseEstimator: {e}")
        return False
    
    try:
        from ball_tracker import TrackNet
        print("✓ TrackNet imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import TrackNet: {e}")
        return False
    
    try:
        from tennis_analyzer import TennisAnalyzer
        print("✓ TennisAnalyzer imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import TennisAnalyzer: {e}")
        return False
    
    return True

def test_config_loading():
    """Test configuration file loading"""
    print("\nTesting configuration loading...")
    
    config_path = "config.yaml"
    if not os.path.exists(config_path):
        print(f"✗ Configuration file not found: {config_path}")
        return False
    
    try:
        import yaml
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        required_keys = ['models', 'data', 'yolo_player', 'yolo_pose', 'tracknet']
        for key in required_keys:
            if key not in config:
                print(f"✗ Missing required config key: {key}")
                return False
        
        print("✓ Configuration loaded successfully")
        return True
        
    except Exception as e:
        print(f"✗ Failed to load configuration: {e}")
        return False

def test_directory_structure():
    """Test that required directories exist"""
    print("\nTesting directory structure...")
    
    required_dirs = [
        "src",
        "models",
        "data/raw_videos",
        "data/processed_frames",
        "data/annotations",
        "data/output"
    ]
    
    all_exist = True
    for dir_path in required_dirs:
        if os.path.exists(dir_path):
            print(f"✓ Directory exists: {dir_path}")
        else:
            print(f"✗ Directory missing: {dir_path}")
            all_exist = False
    
    return all_exist

def test_model_files():
    """Test that model files are present (or provide download instructions)"""
    print("\nTesting model files...")
    
    model_files = [
        "models/yolov8n.pt",
        "models/yolov11-pose.pt", 
        "models/tracknet.h5"
    ]
    
    missing_models = []
    for model_path in model_files:
        if os.path.exists(model_path):
            size_mb = os.path.getsize(model_path) / (1024 * 1024)
            print(f"✓ Model found: {model_path} ({size_mb:.1f} MB)")
        else:
            print(f"✗ Model missing: {model_path}")
            missing_models.append(model_path)
    
    if missing_models:
        print(f"\n⚠️  Missing models: {len(missing_models)}")
        print("Please download the following models:")
        print("1. YOLOv8n.pt - https://github.com/ultralytics/assets/releases")
        print("2. YOLOv11-pose.pt - https://github.com/ultralytics/assets/releases")
        print("3. TrackNet.h5 - https://github.com/yu4u/tracknet")
        print("\nPlace them in the 'models/' directory")
        return False
    
    return True

def test_dependencies():
    """Test that required Python packages are installed"""
    print("\nTesting Python dependencies...")
    
    required_packages = [
        'torch',
        'torchvision', 
        'ultralytics',
        'opencv-python',
        'numpy',
        'tensorflow',
        'keras',
        'pyyaml'
    ]
    
    all_installed = True
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
            print(f"✓ {package} installed")
        except ImportError:
            print(f"✗ {package} not installed")
            all_installed = False
    
    return all_installed

def main():
    """Run all tests"""
    print("🎾 Tennis Analytics System - System Test")
    print("=" * 50)
    
    tests = [
        ("Dependencies", test_dependencies),
        ("Directory Structure", test_directory_structure),
        ("Configuration", test_config_loading),
        ("Model Files", test_model_files),
        ("Module Imports", test_imports)
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n{test_name}:")
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"✗ Test failed with error: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 50)
    print("TEST SUMMARY")
    print("=" * 50)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "PASS" if result else "FAIL"
        print(f"{test_name}: {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! System is ready to use.")
        print("\nNext steps:")
        print("1. Place a tennis video in data/raw_videos/")
        print("2. Run: python main.py --video data/raw_videos/your_video.mp4")
    else:
        print("❌ Some tests failed. Please fix the issues above.")
        print("\nCommon solutions:")
        print("1. Run: python main.py --setup")
        print("2. Install missing dependencies: pip install -r requirements.txt")
        print("3. Download required models to models/ directory")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
