#!/usr/bin/env python3
"""
Main Tennis Analytics Script
Demonstrates the complete tennis analytics system
"""

import cv2
import argparse
import logging
import os
from pathlib import Path
import sys

# Add src to path for imports
sys.path.append(str(Path(__file__).parent / "src"))

from tennis_analyzer import TennisAnalyzer

def setup_logging(verbose: bool = False):
    """Setup logging configuration"""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

def create_directories():
    """Create necessary directories for the project"""
    dirs = [
        "data/raw_videos",
        "data/processed_frames", 
        "data/annotations",
        "data/output",
        "models"
    ]
    
    for dir_path in dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        print(f"Created directory: {dir_path}")

def download_models():
    """Download required models (placeholder for now)"""
    print("\n" + "="*60)
    print("MODEL DOWNLOAD REQUIRED")
    print("="*60)
    print("Please download the following models to the 'models/' directory:")
    print("1. YOLOv8n.pt - Player detection model")
    print("   Download from: https://github.com/ultralytics/assets/releases")
    print("2. YOLOv11-pose.pt - Pose estimation model")
    print("   Download from: https://github.com/ultralytics/assets/releases")
    print("3. TrackNet.h5 - Ball tracking model")
    print("   Download from: https://github.com/yu4u/tracknet")
    print("\nAfter downloading, place them in the 'models/' directory")
    print("="*60)

def analyze_video(video_path: str, config_path: str, output_path: str, 
                  save_video: bool = False, show_display: bool = True):
    """
    Analyze tennis video using the complete system
    
    Args:
        video_path: Path to input video
        config_path: Path to configuration file
        output_path: Path to save analysis results
        save_video: Whether to save annotated video
        show_display: Whether to show real-time display
    """
    try:
        # Initialize tennis analyzer
        print(f"Initializing Tennis Analyzer...")
        analyzer = TennisAnalyzer(config_path)
        
        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"Video: {width}x{height} @ {fps}fps, {total_frames} frames")
        
        # Setup video writer if saving
        video_writer = None
        if save_video:
            output_video_path = output_path.replace('.json', '_annotated.mp4')
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
            print(f"Will save annotated video to: {output_video_path}")
        
        frame_count = 0
        print(f"\nStarting analysis...")
        print("Press 'q' to quit, 's' to save current results")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Analyze frame
            frame_results = analyzer.analyze_frame(frame)
            
            # Draw analysis on frame
            annotated_frame = analyzer.draw_analysis_on_frame(frame, frame_results)
            
            # Save frame if requested
            if video_writer:
                video_writer.write(annotated_frame)
            
            # Display frame
            if show_display:
                cv2.imshow('Tennis Analysis', annotated_frame)
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("Analysis stopped by user")
                    break
                elif key == ord('s'):
                    print("Saving current results...")
                    analyzer.save_analysis_results(output_path)
                    analyzer.print_summary_report()
            
            # Progress update
            frame_count += 1
            if frame_count % 30 == 0:  # Every 30 frames
                progress = (frame_count / total_frames) * 100
                print(f"Progress: {progress:.1f}% ({frame_count}/{total_frames})")
        
        # Cleanup
        cap.release()
        if video_writer:
            video_writer.release()
        if show_display:
            cv2.destroyAllWindows()
        
        # Save final results
        print(f"\nSaving analysis results to: {output_path}")
        analyzer.save_analysis_results(output_path)
        
        # Print summary report
        analyzer.print_summary_report()
        
        print(f"\nAnalysis complete! Results saved to: {output_path}")
        
    except Exception as e:
        logging.error(f"Error during video analysis: {e}")
        raise

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Tennis Analytics System")
    parser.add_argument("--video", "-v", type=str, help="Path to input video file")
    parser.add_argument("--config", "-c", type=str, default="config.yaml", 
                       help="Path to configuration file")
    parser.add_argument("--output", "-o", type=str, default="data/output/analysis_results.json",
                       help="Path to save analysis results")
    parser.add_argument("--save-video", action="store_true", 
                       help="Save annotated video output")
    parser.add_argument("--no-display", action="store_true", 
                       help="Disable real-time display")
    parser.add_argument("--setup", action="store_true", 
                       help="Setup project directories and download models")
    parser.add_argument("--verbose", action="store_true", 
                       help="Enable verbose logging")
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.verbose)
    
    # Setup project if requested
    if args.setup:
        create_directories()
        download_models()
        return
    
    # Check if video file is provided
    if not args.video:
        print("Error: Please provide a video file using --video")
        print("Use --setup to initialize the project first")
        return
    
    # Check if video file exists
    if not os.path.exists(args.video):
        print(f"Error: Video file not found: {args.video}")
        return
    
    # Check if config file exists
    if not os.path.exists(args.config):
        print(f"Error: Configuration file not found: {args.config}")
        print("Please create config.yaml or use --setup to initialize the project")
        return
    
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Run analysis
    try:
        analyze_video(
            video_path=args.video,
            config_path=args.config,
            output_path=args.output,
            save_video=args.save_video,
            show_display=not args.no_display
        )
    except KeyboardInterrupt:
        print("\nAnalysis interrupted by user")
    except Exception as e:
        logging.error(f"Analysis failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
