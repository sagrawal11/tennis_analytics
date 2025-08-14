#!/usr/bin/env python3
"""
Test Tennis Analytics System with tennis_test.mp4
"""

import cv2
import sys
import time
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from tennis_analyzer import TennisAnalyzer

def test_tennis_analysis():
    """Test the tennis analytics system with the test video"""
    print("🎾 Testing Tennis Analytics System")
    
    # Check if test video exists
    video_path = "tennis_test.mp4"
    if not Path(video_path).exists():
        print(f"❌ Test video not found: {video_path}")
        return
    
    # Check if player model exists
    player_model = "playersnball4.pt"
    if not Path(player_model).exists():
        print(f"❌ Player model not found: {player_model}")
        return
    
    try:
        # Initialize analyzer
        print("🔧 Initializing Tennis Analyzer...")
        analyzer = TennisAnalyzer("config.yaml")
        
        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"❌ Could not open video: {video_path}")
            return
        
        # Get video info
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print(f"📹 Video: {width}x{height} @ {fps}fps, {total_frames} frames")
        
        frame_count = 0
        start_time = time.time()
        
        print("\n🚀 Starting analysis... Press 'q' to quit, 's' to save results")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Analyze frame
            frame_results = analyzer.analyze_frame(frame)
            
            # Draw analysis
            annotated_frame = analyzer.draw_analysis_on_frame(frame, frame_results)
            
            # Display
            cv2.imshow('Tennis Analytics Test', annotated_frame)
            
            # Handle keys
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("⏹️  Test stopped by user")
                break
            elif key == ord('s'):
                print("💾 Saving current results...")
                analyzer.save_analysis_results("test_results.json")
                analyzer.print_summary_report()
            
            # Progress update
            frame_count += 1
            if frame_count % 30 == 0:
                elapsed = time.time() - start_time
                fps_actual = frame_count / elapsed
                progress = (frame_count / total_frames) * 100
                print(f"📊 Progress: {progress:.1f}% | FPS: {fps_actual:.1f}")
        
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        
        # Final results
        print(f"\n✅ Test complete! Processed {frame_count} frames")
        analyzer.print_summary_report()
        
        # Save results
        analyzer.save_analysis_results("test_results.json")
        print("💾 Results saved to test_results.json")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_tennis_analysis()
