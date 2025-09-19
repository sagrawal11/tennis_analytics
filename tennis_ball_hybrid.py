#!/usr/bin/env python3
"""
Hybrid Ball Detection System
Combines TRACE TrackNet with our existing ball detection for optimal results
"""

import cv2
import numpy as np
import torch
import pandas as pd
import logging
from typing import Optional, Tuple, List, Dict
from pathlib import Path
import sys
import os

# Add current directory to path to import our existing modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import our existing ball detection
from tennis_ball_trace_improved import ImprovedBallDetector, TraceBallProcessor

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class HybridBallDetector:
    """Hybrid ball detector combining TRACE and our existing system"""
    
    def __init__(self, trace_model_path: str = "TRACE/TrackNet/Weights.pth"):
        # Initialize TRACE detector
        self.trace_detector = ImprovedBallDetector(trace_model_path)
        
        # Detection history for quality assessment
        self.detection_history = []
        self.max_history = 20
        
        # Quality metrics
        self.trace_quality_score = 0.5
        self.our_quality_score = 0.5
        
        # Detection parameters
        self.min_confidence_threshold = 0.3
        self.max_jump_distance = 150  # Maximum reasonable jump between frames
        
        logger.info("Initialized HybridBallDetector")

    def get_our_ball_detection(self, frame: np.ndarray) -> Tuple[Optional[float], Optional[float], float]:
        """Get ball detection using our existing system (from CSV data)"""
        # This would typically read from CSV, but for now we'll simulate
        # In practice, you'd read from the existing tennis_analysis_data.csv
        return None, None, 0.0

    def assess_detection_quality(self, x: float, y: float, confidence: float, 
                               method: str) -> float:
        """Assess the quality of a ball detection"""
        if x is None or y is None or confidence < self.min_confidence_threshold:
            return 0.0
        
        # Base quality from confidence
        quality = confidence
        
        # Check for reasonable movement (not too far from previous detections)
        if len(self.detection_history) > 0:
            last_detection = self.detection_history[-1]
            if last_detection['x'] is not None and last_detection['y'] is not None:
                distance = np.sqrt((x - last_detection['x'])**2 + (y - last_detection['y'])**2)
                
                # Penalize large jumps
                if distance > self.max_jump_distance:
                    quality *= 0.3  # Heavy penalty for large jumps
                elif distance > 100:
                    quality *= 0.7  # Moderate penalty for medium jumps
                else:
                    quality *= 1.1  # Bonus for reasonable movement
        
        # Method-specific adjustments
        if method == "trace":
            # TRACE tends to get stuck, so penalize if it hasn't moved much recently
            if len(self.detection_history) >= 5:
                recent_positions = [d for d in self.detection_history[-5:] if d['method'] == 'trace' and d['x'] is not None]
                if len(recent_positions) >= 3:
                    # Check if TRACE has been stuck in same area
                    positions = [(d['x'], d['y']) for d in recent_positions]
                    variance = np.var(positions, axis=0).sum()
                    if variance < 100:  # Low variance = stuck
                        quality *= 0.5  # Penalize stuck detections
        
        return min(1.0, max(0.0, quality))

    def detect_ball_hybrid(self, frame: np.ndarray, our_x: Optional[float] = None, 
                          our_y: Optional[float] = None, our_conf: float = 0.0) -> Tuple[Optional[float], Optional[float], float, str]:
        """
        Hybrid ball detection combining TRACE and our system
        Returns: (x, y, confidence, method_used)
        """
        # Get TRACE detection
        trace_x, trace_y, trace_conf = self.trace_detector.detect_ball(frame)
        
        # Use provided our system detection or get it
        if our_x is None or our_y is None:
            our_x, our_y, our_conf = self.get_our_ball_detection(frame)
        
        # Assess quality of both detections
        trace_quality = self.assess_detection_quality(trace_x, trace_y, trace_conf, "trace")
        our_quality = self.assess_detection_quality(our_x, our_y, our_conf, "our")
        
        # Choose the better detection
        if trace_quality > our_quality and trace_x is not None:
            chosen_x, chosen_y, chosen_conf = trace_x, trace_y, trace_conf
            chosen_method = "trace"
            chosen_quality = trace_quality
        elif our_x is not None and our_y is not None:
            chosen_x, chosen_y, chosen_conf = our_x, our_y, our_conf
            chosen_method = "our"
            chosen_quality = our_quality
        else:
            # Neither detection is good enough
            chosen_x, chosen_y, chosen_conf = None, None, 0.0
            chosen_method = "none"
            chosen_quality = 0.0
        
        # Update quality scores for adaptive behavior
        if chosen_method == "trace":
            self.trace_quality_score = 0.7 * self.trace_quality_score + 0.3 * chosen_quality
        elif chosen_method == "our":
            self.our_quality_score = 0.7 * self.our_quality_score + 0.3 * chosen_quality
        
        # Store detection in history
        self.detection_history.append({
            'x': chosen_x,
            'y': chosen_y,
            'confidence': chosen_conf,
            'method': chosen_method,
            'quality': chosen_quality,
            'trace_x': trace_x,
            'trace_y': trace_y,
            'trace_conf': trace_conf,
            'our_x': our_x,
            'our_y': our_y,
            'our_conf': our_conf
        })
        
        # Keep only recent history
        if len(self.detection_history) > self.max_history:
            self.detection_history.pop(0)
        
        return chosen_x, chosen_y, chosen_conf, chosen_method

class HybridBallProcessor:
    """Main processor using hybrid ball detection"""
    
    def __init__(self, video_path: str, csv_path: str, output_path: str = "tennis_hybrid_ball.mp4"):
        self.video_path = video_path
        self.csv_path = csv_path
        self.output_path = output_path
        
        # Load CSV data
        self.df = pd.read_csv(csv_path)
        logger.info(f"Loaded {len(self.df)} rows from CSV")
        
        # Initialize video capture
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Initialize hybrid detector
        self.hybrid_detector = HybridBallDetector()
        
        # Initialize video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.out = cv2.VideoWriter(output_path, fourcc, self.fps, (self.width, self.height))
        
        # Results storage
        self.detection_results = []
        
        logger.info(f"Initialized HybridBallProcessor for {video_path}")
        logger.info(f"Video: {self.width}x{self.height} @ {self.fps}fps")

    def get_our_ball_position(self, frame_idx: int) -> Tuple[Optional[float], Optional[float], float]:
        """Get ball position from our existing CSV data"""
        if frame_idx >= len(self.df):
            return None, None, 0.0
        
        ball_x = self.df.iloc[frame_idx]['ball_x']
        ball_y = self.df.iloc[frame_idx]['ball_y']
        ball_confidence = self.df.iloc[frame_idx]['ball_confidence']
        
        if pd.isna(ball_x) or pd.isna(ball_y) or ball_confidence < 0.2:
            return None, None, 0.0
        
        return float(ball_x), float(ball_y), float(ball_confidence)

    def process_video(self):
        """Process entire video with hybrid ball detection"""
        frame_count = 0
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            # Get our system's detection from CSV
            our_x, our_y, our_conf = self.get_our_ball_position(frame_count)
            
            # Get hybrid detection
            hybrid_x, hybrid_y, hybrid_conf, method = self.hybrid_detector.detect_ball_hybrid(
                frame, our_x, our_y, our_conf
            )
            
            # Store results
            self.detection_results.append({
                'frame': frame_count,
                'x': hybrid_x,
                'y': hybrid_y,
                'confidence': hybrid_conf,
                'method': method,
                'our_x': our_x,
                'our_y': our_y,
                'our_conf': our_conf
            })
            
            # Draw ball if detected
            if hybrid_x is not None and hybrid_y is not None:
                # Color based on method and confidence
                if method == "trace":
                    if hybrid_conf > 0.7:
                        color = (0, 255, 0)  # Green for good TRACE
                    else:
                        color = (0, 255, 255)  # Yellow for okay TRACE
                elif method == "our":
                    if hybrid_conf > 0.7:
                        color = (255, 0, 0)  # Blue for good our system
                    else:
                        color = (255, 255, 0)  # Cyan for okay our system
                else:
                    color = (0, 0, 255)  # Red for low confidence
                
                # Draw ball
                cv2.circle(frame, (int(hybrid_x), int(hybrid_y)), 8, color, -1)
                cv2.putText(frame, f"{method.upper()}: {hybrid_conf:.2f}", 
                           (int(hybrid_x) + 10, int(hybrid_y) - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            # Draw method comparison info
            cv2.putText(frame, f"TRACE Quality: {self.hybrid_detector.trace_quality_score:.2f}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"Our Quality: {self.hybrid_detector.our_quality_score:.2f}", 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            
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

    def save_results_to_csv(self, output_csv: str = "tennis_hybrid_ball_results.csv"):
        """Save detection results to CSV"""
        df = pd.DataFrame(self.detection_results)
        df.to_csv(output_csv, index=False)
        logger.info(f"Results saved to: {output_csv}")

    def print_summary(self):
        """Print detection summary"""
        total_frames = len(self.detection_results)
        detections = [r for r in self.detection_results if r['x'] is not None]
        trace_detections = [r for r in detections if r['method'] == 'trace']
        our_detections = [r for r in detections if r['method'] == 'our']
        
        print(f"\n=== HYBRID BALL DETECTION SUMMARY ===")
        print(f"Total frames: {total_frames}")
        print(f"Total detections: {len(detections)} ({len(detections)/total_frames*100:.1f}%)")
        print(f"TRACE detections: {len(trace_detections)} ({len(trace_detections)/len(detections)*100:.1f}%)")
        print(f"Our system detections: {len(our_detections)} ({len(our_detections)/len(detections)*100:.1f}%)")
        print(f"Final TRACE quality score: {self.hybrid_detector.trace_quality_score:.3f}")
        print(f"Final our quality score: {self.hybrid_detector.our_quality_score:.3f}")

def main():
    """Main function to run hybrid ball detection"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Hybrid Ball Detection combining TRACE and our system")
    parser.add_argument("--video", required=True, help="Input video path")
    parser.add_argument("--csv", required=True, help="Input CSV path")
    parser.add_argument("--output", default="tennis_hybrid_ball.mp4", help="Output video path")
    parser.add_argument("--results-csv", default="tennis_hybrid_ball_results.csv", help="Output CSV path")
    
    args = parser.parse_args()
    
    try:
        # Initialize processor
        processor = HybridBallProcessor(args.video, args.csv, args.output)
        
        # Process video
        results = processor.process_video()
        
        # Save results
        processor.save_results_to_csv(args.results_csv)
        
        # Print summary
        processor.print_summary()
        
    except Exception as e:
        logger.error(f"Error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
