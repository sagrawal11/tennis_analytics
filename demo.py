#!/usr/bin/env python3
"""
Demo script for the Tennis Analytics System
Shows how to use the system programmatically
"""

import cv2
import sys
import time
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from tennis_analyzer import TennisAnalyzer

def demo_with_video(video_path: str, config_path: str = "config.yaml"):
    """Demo the system with a video file"""
    print(f"🎾 Starting Tennis Analytics Demo with video: {video_path}")
    
    try:
        # Initialize analyzer
        analyzer = TennisAnalyzer(config_path)
        
        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"❌ Could not open video: {video_path}")
            return
        
        # Get video info
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"📹 Video: {fps}fps, {total_frames} frames")
        
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
            cv2.imshow('Tennis Analytics Demo', annotated_frame)
            
            # Handle keys
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("⏹️  Demo stopped by user")
                break
            elif key == ord('s'):
                print("💾 Saving current results...")
                analyzer.save_analysis_results("demo_results.json")
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
        print(f"\n✅ Demo complete! Processed {frame_count} frames")
        analyzer.print_summary_report()
        
        # Save results
        analyzer.save_analysis_results("demo_results.json")
        print("💾 Results saved to demo_results.json")
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        raise

def demo_with_webcam(config_path: str = "config.yaml"):
    """Demo the system with webcam input"""
    print("🎾 Starting Tennis Analytics Demo with webcam")
    
    try:
        # Initialize analyzer
        analyzer = TennisAnalyzer(config_path)
        
        # Open webcam
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("❌ Could not open webcam")
            return
        
        print("\n📹 Webcam active. Press 'q' to quit, 's' to save results")
        print("💡 Try moving around to see pose estimation and ball tracking!")
        
        frame_count = 0
        start_time = time.time()
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("❌ Failed to read from webcam")
                break
            
            # Analyze frame
            frame_results = analyzer.analyze_frame(frame)
            
            # Draw analysis
            annotated_frame = analyzer.draw_analysis_on_frame(frame, frame_results)
            
            # Display
            cv2.imshow('Tennis Analytics Demo (Webcam)', annotated_frame)
            
            # Handle keys
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("⏹️  Demo stopped by user")
                break
            elif key == ord('s'):
                print("💾 Saving current results...")
                analyzer.save_analysis_results("webcam_demo_results.json")
                analyzer.print_summary_report()
            
            # Performance update
            frame_count += 1
            if frame_count % 30 == 0:
                elapsed = time.time() - start_time
                fps_actual = frame_count / elapsed
                print(f"📊 FPS: {fps_actual:.1f} | Frames: {frame_count}")
        
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        
        # Final results
        print(f"\n✅ Webcam demo complete! Processed {frame_count} frames")
        analyzer.print_summary_report()
        
        # Save results
        analyzer.save_analysis_results("webcam_demo_results.json")
        print("💾 Results saved to webcam_demo_results.json")
        
    except Exception as e:
        print(f"❌ Webcam demo failed: {e}")
        raise

def demo_offline_analysis(config_path: str = "config.yaml"):
    """Demo offline analysis capabilities"""
    print("🎾 Starting Offline Analysis Demo")
    
    try:
        # Initialize analyzer
        analyzer = TennisAnalyzer(config_path)
        
        print("📊 Offline analysis demo shows:")
        print("  • Player detection statistics")
        print("  • Pose estimation capabilities")
        print("  • Ball tracking analysis")
        print("  • Swing mechanics classification")
        print("  • Performance metrics")
        
        # Create sample analysis data
        print("\n🔧 Creating sample analysis session...")
        
        # Simulate some analysis results
        analyzer.analysis_results['summary_stats'] = {
            'total_frames': 1000,
            'total_players_detected': 850,
            'total_poses_estimated': 800,
            'total_ball_detections': 750,
            'swing_phases': {
                'backswing': 200,
                'contact': 150,
                'follow_through': 180,
                'ready_position': 270
            },
            'average_processing_time': 0.045,
            'processing_times': [0.04, 0.05, 0.045] * 333
        }
        
        # Print summary
        print("\n📈 Sample Analysis Results:")
        analyzer.print_summary_report()
        
        # Save results
        analyzer.save_analysis_results("offline_demo_results.json")
        print("\n💾 Sample results saved to offline_demo_results.json")
        
        print("\n✅ Offline demo complete!")
        
    except Exception as e:
        print(f"❌ Offline demo failed: {e}")
        raise

def main():
    """Main demo function"""
    print("🎾 Tennis Analytics System - Demo")
    print("=" * 50)
    
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python demo.py video <video_path>     # Demo with video file")
        print("  python demo.py webcam                 # Demo with webcam")
        print("  python demo.py offline                # Demo offline analysis")
        print("\nExamples:")
        print("  python demo.py video tennis_match.mp4")
        print("  python demo.py webcam")
        print("  python demo.py offline")
        return
    
    mode = sys.argv[1].lower()
    
    try:
        if mode == "video":
            if len(sys.argv) < 3:
                print("❌ Please provide video path: python demo.py video <video_path>")
                return
            video_path = sys.argv[2]
            demo_with_video(video_path)
            
        elif mode == "webcam":
            demo_with_webcam()
            
        elif mode == "offline":
            demo_offline_analysis()
            
        else:
            print(f"❌ Unknown mode: {mode}")
            print("Available modes: video, webcam, offline")
            
    except KeyboardInterrupt:
        print("\n⏹️  Demo interrupted by user")
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        print("\n💡 Make sure you have:")
        print("  1. Installed all dependencies: pip install -r requirements.txt")
        print("  2. Downloaded required models to models/ directory")
        print("  3. Run setup: python main.py --setup")

if __name__ == "__main__":
    main()
