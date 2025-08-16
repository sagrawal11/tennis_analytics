#!/usr/bin/env python3
"""
Test the pretrained ball detection model on tennis_test.mp4
"""

import torch
import cv2
import numpy as np
import argparse
from pathlib import Path
import time
import sys

# Add the trackNet_version2 directory to path to import the model
sys.path.append('trackNet_version2')

def load_tracknet_model(model_path):
    """Load the pretrained TrackNet model"""
    print(f"🔄 Loading TrackNet model from {model_path}")
    
    try:
        # Import the TrackNet model architecture
        from model import BallTrackerNet
        
        # Create model instance
        model = BallTrackerNet(out_channels=1)
        print(f"✅ TrackNet model created")
        
        # Load the pretrained weights
        state_dict = torch.load(model_path, map_location='cpu')
        model.load_state_dict(state_dict)
        print(f"✅ Pretrained weights loaded successfully")
        
        # Set to evaluation mode
        model.eval()
        
        return model
        
    except Exception as e:
        print(f"❌ Error loading TrackNet model: {e}")
        print(f"   Make sure you're in the root directory and trackNet_version2/model.py exists")
        return None

def preprocess_frames(frames, target_height=720, target_width=1280):
    """Preprocess frames for TrackNet input"""
    processed_frames = []
    
    for frame in frames:
        # Resize to target resolution
        frame_resized = cv2.resize(frame, (target_width, target_height))
        
        # Normalize to [0, 1]
        frame_normalized = frame_resized.astype(np.float32) / 255.0
        
        # Convert to tensor and add batch dimension
        frame_tensor = torch.from_numpy(frame_normalized).permute(2, 0, 1).unsqueeze(0)
        processed_frames.append(frame_tensor)
    
    return processed_frames

def create_tracknet_input(frames, target_height=720, target_width=1280):
    """Create 9-channel input for TrackNet (3 consecutive frames)"""
    if len(frames) < 3:
        print(f"❌ Need at least 3 frames, got {len(frames)}")
        return None
    
    # Take the last 3 frames
    recent_frames = frames[-3:]
    
    # Preprocess frames
    processed_frames = preprocess_frames(recent_frames, target_height, target_width)
    
    # Concatenate along channel dimension
    # Each frame is (1, 3, H, W), so concatenated becomes (1, 9, H, W)
    tracknet_input = torch.cat(processed_frames, dim=1)
    
    return tracknet_input

def detect_balls(heatmap, threshold=0.5):
    """Extract ball positions from heatmap using Hough circles"""
    # Convert heatmap to numpy and scale to 0-255
    heatmap_np = heatmap.squeeze().cpu().numpy()
    heatmap_scaled = (heatmap_np * 255).astype(np.uint8)
    
    # Apply threshold
    _, binary = cv2.threshold(heatmap_scaled, int(threshold * 255), 255, cv2.THRESH_BINARY)
    
    # Find circles (balls)
    circles = cv2.HoughCircles(
        binary, cv2.HOUGH_GRADIENT, dp=1, minDist=20,
        param1=50, param2=30, minRadius=5, maxRadius=50
    )
    
    ball_positions = []
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        for (x, y, r) in circles:
            ball_positions.append((x, y, r))
    
    return ball_positions, heatmap_scaled

def process_frame(frame, model, frame_buffer, frame_num, target_height=720, target_width=1280):
    """Process a single frame with ball detection"""
    # Add frame to buffer
    frame_buffer.append(frame)
    
    # Keep only last 3 frames
    if len(frame_buffer) > 3:
        frame_buffer.pop(0)
    
    # Create TrackNet input if we have enough frames
    if len(frame_buffer) == 3:
        try:
            # Create input tensor
            tracknet_input = create_tracknet_input(frame_buffer, target_height, target_width)
            
            if tracknet_input is not None:
                # Run inference
                with torch.no_grad():
                    heatmap = model(tracknet_input)
                
                # Extract ball positions
                ball_positions, heatmap_vis = detect_balls(heatmap)
                
                # Draw balls on frame
                for (x, y, r) in ball_positions:
                    # Scale coordinates back to original frame size
                    x_scaled = int(x * frame.shape[1] / target_width)
                    y_scaled = int(y * frame.shape[0] / target_height)
                    r_scaled = int(r * frame.shape[1] / target_width)
                    
                    # Draw circle
                    cv2.circle(frame, (x_scaled, y_scaled), r_scaled, (0, 255, 0), 2)
                    cv2.circle(frame, (x_scaled, y_scaled), 2, (0, 0, 255), -1)
                
                # Add frame number and ball count
                cv2.putText(frame, f"Frame: {frame_num} | Balls: {len(ball_positions)}", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                return frame, ball_positions
            else:
                # Fallback if input creation fails
                cv2.putText(frame, f"Frame: {frame_num} | Processing...", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                return frame, []
                
        except Exception as e:
            print(f"❌ Error processing frame {frame_num}: {e}")
            cv2.putText(frame, f"Frame: {frame_num} | Error", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            return frame, []
    else:
        # Not enough frames yet
        cv2.putText(frame, f"Frame: {frame_num} | Buffering...", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        return frame, []

def process_video(video_path, model, output_path=None, max_frames=None, target_height=720, target_width=1280):
    """Process the video with the ball detection model"""
    print(f"🎬 Processing video: {video_path}")
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ Could not open video: {video_path}")
        return
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"📹 Video Info:")
    print(f"  Resolution: {width}x{height}")
    print(f"  FPS: {fps}")
    print(f"  Total Frames: {total_frames}")
    print(f"  Target Processing: {target_width}x{target_height}")
    
    # Setup output video if specified
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        print(f"💾 Output video: {output_path}")
    
    # Process frames
    frame_count = 0
    start_time = time.time()
    frame_buffer = []
    total_balls_detected = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_count += 1
        if max_frames and frame_count > max_frames:
            break
            
        # Process frame
        processed_frame, ball_positions = process_frame(frame, model, frame_buffer, frame_count, target_height, target_width)
        total_balls_detected += len(ball_positions)
        
        # Write to output if specified
        if output_path:
            out.write(processed_frame)
        
        # Show progress
        if frame_count % 30 == 0:  # Every 30 frames
            elapsed = time.time() - start_time
            fps_processed = frame_count / elapsed
            print(f"📊 Frame {frame_count}/{total_frames} | FPS: {fps_processed:.1f} | Balls: {total_balls_detected}")
    
    # Cleanup
    cap.release()
    if output_path:
        out.release()
    
    total_time = time.time() - start_time
    print(f"✅ Processing complete!")
    print(f"  Total frames: {frame_count}")
    print(f"  Total balls detected: {total_balls_detected}")
    print(f"  Total time: {total_time:.1f}s")
    print(f"  Average FPS: {frame_count/total_time:.1f}")

def main():
    parser = argparse.ArgumentParser(description='Test pretrained TrackNet ball detection model')
    parser.add_argument('--model', type=str, default='pretrained_ball_detection.pt',
                       help='Path to pretrained model')
    parser.add_argument('--video', type=str, default='tennis_test.mp4',
                       help='Path to input video')
    parser.add_argument('--output', type=str, default='tennis_test_tracknet_output.mp4',
                       help='Path to output video')
    parser.add_argument('--max-frames', type=int, default=None,
                       help='Maximum frames to process (for testing)')
    parser.add_argument('--height', type=int, default=720,
                       help='Target processing height')
    parser.add_argument('--width', type=int, default=1280,
                       help='Target processing width')
    
    args = parser.parse_args()
    
    print("🎾 Testing Pretrained TrackNet Ball Detection Model")
    print("=" * 60)
    
    # Check files exist
    if not Path(args.model).exists():
        print(f"❌ Model not found: {args.model}")
        return
    
    if not Path(args.video).exists():
        print(f"❌ Video not found: {args.video}")
        return
    
    # Check if TrackNet model file exists
    if not Path('trackNet_version2/model.py').exists():
        print(f"❌ TrackNet model.py not found: trackNet_version2/model.py")
        print(f"   Make sure you're in the root directory")
        return
    
    # Load model
    model = load_tracknet_model(args.model)
    if model is None:
        return
    
    # Process video
    process_video(args.video, model, args.output, args.max_frames, args.height, args.width)

if __name__ == "__main__":
    main()
