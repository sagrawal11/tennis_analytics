#!/usr/bin/env python3
"""
Improved Ball Tracking System based on TRACE project
Integrates TRACE's better TrackNet implementation with our tennis analytics system
"""

import cv2
import numpy as np
import torch
import torch.nn as nn
import pandas as pd
import logging
from typing import Optional, Tuple, List, Dict
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ConvBlock(nn.Module):
    """Convolutional block with optional batch normalization"""
    def __init__(self, in_channels, out_channels, kernel_size, pad, bias=True, bn=True):
        super().__init__()
        if bn:
            self.block = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size, padding=pad, bias=bias),
                nn.ReLU(),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.block = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size, padding=pad, bias=bias),
                nn.ReLU()
            )

    def forward(self, x):
        return self.block(x)

class BallTrackerNet(nn.Module):
    """Improved TrackNet implementation based on TRACE"""
    def __init__(self, out_channels=256, bn=True):
        super().__init__()
        self.out_channels = out_channels

        # Encoder layers
        layer_1 = ConvBlock(in_channels=9, out_channels=64, kernel_size=3, pad=1, bias=True, bn=bn)
        layer_2 = ConvBlock(in_channels=64, out_channels=64, kernel_size=3, pad=1, bias=True, bn=bn)
        layer_3 = nn.MaxPool2d(kernel_size=2, stride=2)
        layer_4 = ConvBlock(in_channels=64, out_channels=128, kernel_size=3, pad=1, bias=True, bn=bn)
        layer_5 = ConvBlock(in_channels=128, out_channels=128, kernel_size=3, pad=1, bias=True, bn=bn)
        layer_6 = nn.MaxPool2d(kernel_size=2, stride=2)
        layer_7 = ConvBlock(in_channels=128, out_channels=256, kernel_size=3, pad=1, bias=True, bn=bn)
        layer_8 = ConvBlock(in_channels=256, out_channels=256, kernel_size=3, pad=1, bias=True, bn=bn)
        layer_9 = ConvBlock(in_channels=256, out_channels=256, kernel_size=3, pad=1, bias=True, bn=bn)
        layer_10 = nn.MaxPool2d(kernel_size=2, stride=2)
        layer_11 = ConvBlock(in_channels=256, out_channels=512, kernel_size=3, pad=1, bias=True, bn=bn)
        layer_12 = ConvBlock(in_channels=512, out_channels=512, kernel_size=3, pad=1, bias=True, bn=bn)
        layer_13 = ConvBlock(in_channels=512, out_channels=512, kernel_size=3, pad=1, bias=True, bn=bn)

        self.encoder = nn.Sequential(layer_1, layer_2, layer_3, layer_4, layer_5, layer_6, layer_7, layer_8, layer_9,
                                     layer_10, layer_11, layer_12, layer_13)

        # Decoder layers
        layer_14 = nn.Upsample(scale_factor=2)
        layer_15 = ConvBlock(in_channels=512, out_channels=256, kernel_size=3, pad=1, bias=True, bn=bn)
        layer_16 = ConvBlock(in_channels=256, out_channels=256, kernel_size=3, pad=1, bias=True, bn=bn)
        layer_17 = ConvBlock(in_channels=256, out_channels=256, kernel_size=3, pad=1, bias=True, bn=bn)
        layer_18 = nn.Upsample(scale_factor=2)
        layer_19 = ConvBlock(in_channels=256, out_channels=128, kernel_size=3, pad=1, bias=True, bn=bn)
        layer_20 = ConvBlock(in_channels=128, out_channels=128, kernel_size=3, pad=1, bias=True, bn=bn)
        layer_21 = nn.Upsample(scale_factor=2)
        layer_22 = ConvBlock(in_channels=128, out_channels=64, kernel_size=3, pad=1, bias=True, bn=bn)
        layer_23 = ConvBlock(in_channels=64, out_channels=64, kernel_size=3, pad=1, bias=True, bn=bn)
        layer_24 = ConvBlock(in_channels=64, out_channels=self.out_channels, kernel_size=3, pad=1, bias=True, bn=bn)

        self.decoder = nn.Sequential(layer_14, layer_15, layer_16, layer_17, layer_18, layer_19, layer_20, layer_21,
                                     layer_22, layer_23, layer_24)

        self.softmax = nn.Softmax(dim=1)
        self._init_weights()

    def forward(self, x, testing=False):
        batch_size = x.size(0)
        features = self.encoder(x)
        scores_map = self.decoder(features)
        output = scores_map.reshape(batch_size, self.out_channels, -1)
        if testing:
            output = self.softmax(output)
        return output

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.uniform_(module.weight, -0.05, 0.05)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)

    def inference(self, frames: torch.Tensor):
        self.eval()
        with torch.no_grad():
            if len(frames.shape) == 3:
                frames = frames.unsqueeze(0)
            if next(self.parameters()).is_cuda:
                frames.cuda()
            # Forward pass
            output = self(frames, True)
            output = output.argmax(dim=1).detach().cpu().numpy()
            if self.out_channels == 2:
                output *= 255
            x, y = self.get_center_ball(output)
        return x, y

    def get_center_ball(self, output):
        """Detect the center of the ball using Hough circle transform"""
        output = output.reshape((360, 640))
        output = output.astype(np.uint8)
        heatmap = cv2.resize(output, (640, 360))
        
        # Convert to binary image
        ret, heatmap = cv2.threshold(heatmap, 127, 255, cv2.THRESH_BINARY)
        
        # Find circles with radius 2-7 pixels
        circles = cv2.HoughCircles(heatmap, cv2.HOUGH_GRADIENT, dp=1, minDist=1, 
                                 param1=50, param2=8, minRadius=2, maxRadius=7)
        
        if circles is not None and len(circles) == 1:
            x = int(circles[0][0][0])
            y = int(circles[0][0][1])
            return x, y
        return None, None

def combine_three_frames(frame1, frame2, frame3, width, height):
    """Combine three consecutive frames for TrackNet input"""
    # Resize and convert to float
    img = cv2.resize(frame1, (width, height)).astype(np.float32)
    img1 = cv2.resize(frame2, (width, height)).astype(np.float32)
    img2 = cv2.resize(frame3, (width, height)).astype(np.float32)
    
    # Combine frames (width, height, rgb*3)
    imgs = np.concatenate((img, img1, img2), axis=2)
    
    # Change to channels_first for TrackNet
    imgs = np.rollaxis(imgs, 2, 0)
    return np.array(imgs)

class ImprovedBallDetector:
    """Improved ball detector based on TRACE implementation"""
    
    def __init__(self, model_path: str = "TRACE/TrackNet/Weights.pth", out_channels: int = 2):
        self.device = torch.device("cpu")
        
        # Load TrackNet model
        self.detector = BallTrackerNet(out_channels=out_channels)
        try:
            saved_state_dict = torch.load(model_path, map_location=torch.device("cpu"))
            self.detector.load_state_dict(saved_state_dict['model_state'])
            logger.info(f"Loaded TrackNet model from {model_path}")
        except Exception as e:
            logger.error(f"Failed to load model from {model_path}: {e}")
            raise
        
        self.detector.eval().to(self.device)
        
        # Frame history for 3-frame input
        self.current_frame = None
        self.last_frame = None
        self.before_last_frame = None
        
        # Video dimensions
        self.video_width = None
        self.video_height = None
        self.model_input_width = 640
        self.model_input_height = 360
        
        # Detection parameters
        self.threshold_dist = 100  # Maximum distance between consecutive detections
        self.xy_coordinates = np.array([[None, None], [None, None]])
        
        # Detection history
        self.detection_history = []
        self.max_history = 50

    def detect_ball(self, frame: np.ndarray) -> Tuple[Optional[float], Optional[float], float]:
        """
        Detect ball in current frame using 3-frame TrackNet approach
        Returns: (x, y, confidence)
        """
        # Save frame dimensions
        if self.video_width is None:
            self.video_width = frame.shape[1]
            self.video_height = frame.shape[0]
        
        # Update frame history
        self.last_frame = self.before_last_frame
        self.before_last_frame = self.current_frame
        self.current_frame = frame.copy()
        
        # Only detect if we have 3 frames
        if self.last_frame is not None:
            # Combine frames for TrackNet input
            frames = combine_three_frames(self.current_frame, self.before_last_frame, self.last_frame,
                                        self.model_input_width, self.model_input_height)
            frames = (torch.from_numpy(frames) / 255).to(self.device)
            
            # Run inference
            x, y = self.detector.inference(frames)
            
            if x is not None and y is not None:
                # Scale coordinates back to video dimensions
                x = int(x * (self.video_width / self.model_input_width))
                y = int(y * (self.video_height / self.model_input_height))
                
                # Check distance from previous detection (outlier filtering)
                if self.xy_coordinates[-1][0] is not None:
                    prev_x, prev_y = self.xy_coordinates[-1]
                    distance = np.linalg.norm(np.array([x, y]) - np.array([prev_x, prev_y]))
                    
                    if distance > self.threshold_dist:
                        # Outlier detected, use previous position
                        x, y = prev_x, prev_y
                        confidence = 0.3  # Lower confidence for outlier
                    else:
                        confidence = 0.9  # High confidence for valid detection
                else:
                    confidence = 0.8  # Medium confidence for first detection
                
                # Update coordinate history
                self.xy_coordinates = np.append(self.xy_coordinates, np.array([[x, y]]), axis=0)
                
                # Keep only recent history
                if len(self.xy_coordinates) > self.max_history:
                    self.xy_coordinates = self.xy_coordinates[-self.max_history:]
                
                # Add to detection history
                self.detection_history.append({
                    'x': x, 'y': y, 'confidence': confidence
                })
                
                return float(x), float(y), confidence
            else:
                # No detection, return None with low confidence
                return None, None, 0.0
        else:
            return None, None, 0.0

class TraceBallProcessor:
    """Main processor for improved ball tracking using TRACE approach"""
    
    def __init__(self, video_path: str, csv_path: str, output_path: str = "tennis_trace_ball.mp4"):
        self.video_path = video_path
        self.csv_path = csv_path
        self.output_path = output_path
        
        # Initialize video capture
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Initialize ball detector
        self.ball_detector = ImprovedBallDetector()
        
        # Initialize video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.out = cv2.VideoWriter(output_path, fourcc, self.fps, (self.width, self.height))
        
        # Results storage
        self.detection_results = []
        
        logger.info(f"Initialized TraceBallProcessor for {video_path}")
        logger.info(f"Video: {self.width}x{self.height} @ {self.fps}fps")

    def process_video(self):
        """Process entire video with improved ball tracking"""
        frame_count = 0
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            # Detect ball using TRACE approach
            x, y, confidence = self.ball_detector.detect_ball(frame)
            
            # Store results
            self.detection_results.append({
                'frame': frame_count,
                'x': x,
                'y': y,
                'confidence': confidence
            })
            
            # Draw ball if detected
            if x is not None and y is not None:
                # Color based on confidence
                if confidence > 0.7:
                    color = (0, 255, 0)  # Green for high confidence
                elif confidence > 0.4:
                    color = (0, 255, 255)  # Yellow for medium confidence
                else:
                    color = (0, 0, 255)  # Red for low confidence
                
                # Draw ball
                cv2.circle(frame, (int(x), int(y)), 8, color, -1)
                cv2.putText(frame, f"Conf: {confidence:.2f}", (int(x) + 10, int(y) - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            # Write frame
            self.out.write(frame)
            frame_count += 1
            
            if frame_count % 30 == 0:
                logger.info(f"Processed {frame_count} frames")
        
        # Cleanup
        self.cap.release()
        self.out.release()
        
        logger.info(f"Processing complete! Processed {frame_count} frames")
        logger.info(f"Output saved to: {self.output_path}")
        
        return self.detection_results

    def save_results_to_csv(self, output_csv: str = "tennis_trace_ball_results.csv"):
        """Save detection results to CSV"""
        df = pd.DataFrame(self.detection_results)
        df.to_csv(output_csv, index=False)
        logger.info(f"Results saved to: {output_csv}")

def main():
    """Main function to run improved ball tracking"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Improved Ball Tracking using TRACE approach")
    parser.add_argument("--video", required=True, help="Input video path")
    parser.add_argument("--csv", help="Input CSV path (optional)")
    parser.add_argument("--output", default="tennis_trace_ball.mp4", help="Output video path")
    parser.add_argument("--results-csv", default="tennis_trace_ball_results.csv", help="Output CSV path")
    
    args = parser.parse_args()
    
    try:
        # Initialize processor
        processor = TraceBallProcessor(args.video, args.csv, args.output)
        
        # Process video
        results = processor.process_video()
        
        # Save results
        processor.save_results_to_csv(args.results_csv)
        
        print(f"\n=== IMPROVED BALL TRACKING RESULTS ===")
        print(f"Total frames processed: {len(results)}")
        detections = [r for r in results if r['x'] is not None]
        print(f"Ball detections: {len(detections)}")
        print(f"Detection rate: {len(detections)/len(results)*100:.1f}%")
        print(f"Output video: {args.output}")
        print(f"Results CSV: {args.results_csv}")
        
    except Exception as e:
        logger.error(f"Error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
