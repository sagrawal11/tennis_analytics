#!/usr/bin/env python3
"""
Debug script to test shot detection on a few frames
"""

import cv2
import numpy as np
import logging
from tennis_CV import TennisAnalysisDemo

# Configure logging to see debug messages
logging.basicConfig(level=logging.DEBUG)

def test_shot_detection():
    """Test shot detection on a few frames"""
    
    # Initialize the analyzer
    analyzer = TennisAnalysisDemo("config.yaml")
    
    # Open video
    cap = cv2.VideoCapture("tennis_test5.mp4")
    
    frame_count = 0
    max_frames = 10  # Only test first 10 frames
    
    while cap.isOpened() and frame_count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
            
        print(f"\n{'='*60}")
        print(f"FRAME {frame_count}")
        print(f"{'='*60}")
        
        # Process frame
        try:
            # Get player detections
            if analyzer.player_detector:
                player_detections = analyzer.player_detector.detect_players(frame)
                print(f"Players detected: {len(player_detections)}")
                
                # Get poses
                poses = []
                if analyzer.pose_estimator:
                    poses = analyzer.pose_estimator.estimate_poses(frame, player_detections)
                    print(f"Poses estimated: {len(poses)}")
                
                # Get ball position
                ball_pred = analyzer._detect_ball_in_frame(frame)
                ball_position = None
                if ball_pred:
                    ball_position = ball_pred['position']
                    print(f"Ball position: {ball_position}")
                
                # Test shot detection for each player
                for i, detection in enumerate(player_detections):
                    if 'bbox' in detection:
                        bbox = detection['bbox']
                        print(f"\nPlayer {i+1} bbox: {bbox}")
                        
                        # Test shot detection
                        shot_type = analyzer._detect_shot_type(bbox, ball_position, poses, analyzer.court_keypoints)
                        print(f"Shot type: {shot_type}")
                        
        except Exception as e:
            print(f"Error processing frame: {e}")
        
        frame_count += 1
        
        # Press 'q' to quit early
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    test_shot_detection()
