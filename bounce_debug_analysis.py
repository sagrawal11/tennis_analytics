#!/usr/bin/env python3
"""
Bounce Detection Analysis Script
Focused debugging to understand why the model is over-detecting bounces
"""

import cv2
import numpy as np
import time
import logging
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from collections import deque
import catboost as cb
import argparse
import pandas as pd

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BounceAnalyzer:
    """Analyze bounce detection to understand over-detection issues"""
    
    def __init__(self, model_path: str, num_frames: int = 3):
        self.num_frames = num_frames
        self.ball_positions = deque(maxlen=num_frames + 2)
        self.frame_timestamps = deque(maxlen=num_frames + 2)
        
        try:
            self.model = cb.CatBoostClassifier()
            self.model.load_model(model_path)
            logger.info(f"Model loaded: {model_path}")
            logger.info(f"Feature names: {self.model.feature_names_}")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            self.model = None
    
    def add_ball_position(self, x: float, y: float, timestamp: float = None):
        if timestamp is None:
            timestamp = time.time()
        self.ball_positions.append((x, y))
        self.frame_timestamps.append(timestamp)
        
        while len(self.ball_positions) > self.num_frames + 2:
            self.ball_positions.popleft()
            self.frame_timestamps.popleft()
    
    def extract_trajectory_features(self) -> List[float]:
        """Extract trajectory features for analysis"""
        if len(self.ball_positions) < self.num_frames:
            return []
        
        features = []
        eps = 1e-15
        
        positions = list(self.ball_positions)[-self.num_frames:]
        
        for i in range(1, self.num_frames):
            curr_x, curr_y = positions[i]
            prev_x, prev_y = positions[i-1]
            next_x, next_y = positions[i+1] if i+1 < len(positions) else (curr_x, curr_y)
            
            x_diff = abs(curr_x - prev_x)
            y_diff = curr_y - prev_y
            x_diff_inv = abs(next_x - curr_x)
            y_diff_inv = next_y - curr_y
            
            x_div = abs(x_diff / (x_diff_inv + eps))
            y_div = y_diff / (y_diff_inv + eps)
            
            features.extend([x_diff, x_diff_inv, x_div, y_diff, y_diff_inv, y_div])
        
        return features
    
    def analyze_prediction(self) -> Tuple[float, List[float], Dict[str, any]]:
        """Analyze model prediction and features"""
        if not self.model or len(self.ball_positions) < self.num_frames:
            return 0.0, [], {}
        
        features = self.extract_trajectory_features()
        if len(features) == 0:
            return 0.0, [], {}
        
        # Ensure correct feature count
        expected_features = (self.num_frames - 1) * 6
        while len(features) < expected_features:
            features.append(0.0)
        features = features[:expected_features]
        
        # Get prediction
        prediction = self.model.predict_proba([features])[0]
        confidence = prediction[1] if len(prediction) > 1 else prediction[0]
        
        # Analyze features
        feature_analysis = self.analyze_features(features)
        
        return confidence, features, feature_analysis
    
    def analyze_features(self, features: List[float]) -> Dict[str, any]:
        """Analyze what the features are telling us"""
        if len(features) != 12:  # 2 frame differences * 6 features each
            return {}
        
        # Group features by type
        x_diffs = [features[0], features[1]]      # x_diff_1, x_diff_2
        x_diff_invs = [features[2], features[3]]  # x_diff_inv_1, x_diff_inv_2
        x_divs = [features[4], features[5]]       # x_div_1, x_div_2
        y_diffs = [features[6], features[7]]      # y_diff_1, y_diff_2
        y_diff_invs = [features[8], features[9]]  # y_diff_inv_1, y_diff_inv_2
        y_divs = [features[10], features[11]]     # y_div_1, y_div_2
        
        # Analyze patterns
        analysis = {
            'x_movement': {
                'avg_x_diff': np.mean(x_diffs),
                'avg_x_diff_inv': np.mean(x_diff_invs),
                'x_direction_change': abs(np.mean(x_diffs) - np.mean(x_diff_invs)),
                'x_ratio_avg': np.mean(x_divs)
            },
            'y_movement': {
                'avg_y_diff': np.mean(y_diffs),
                'avg_y_diff_inv': np.mean(y_diff_invs),
                'y_direction_change': abs(np.mean(y_diffs) - np.mean(y_diff_invs)),
                'y_ratio_avg': np.mean(y_divs)
            },
            'movement_patterns': {
                'high_x_movement': np.mean(x_diffs) > 10,
                'high_y_movement': np.mean(y_diffs) > 10,
                'direction_reversal': abs(np.mean(y_diffs) + np.mean(y_diff_invs)) < 5,
                'smooth_trajectory': np.mean(x_divs) < 2 and np.mean(y_divs) < 2
            }
        }
        
        return analysis


class BallTracker:
    """Simple ball tracking for analysis"""
    
    def __init__(self):
        self.ball_color_ranges = [
            (np.array([0, 100, 100]), np.array([30, 255, 255])),  # Yellow/white
            (np.array([35, 50, 50]), np.array([85, 255, 255])),   # Green
            (np.array([5, 100, 100]), np.array([25, 255, 255]))   # Orange
        ]
    
    def detect_ball(self, frame: np.ndarray) -> Optional[Tuple[float, float, float]]:
        try:
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            
            best_detection = None
            best_confidence = 0
            
            for color_lower, color_upper in self.ball_color_ranges:
                mask = cv2.inRange(hsv, color_lower, color_upper)
                
                kernel = np.ones((3, 3), np.uint8)
                mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
                mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
                
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                for contour in contours:
                    area = cv2.contourArea(contour)
                    if area < 30 or area > 8000:
                        continue
                    
                    M = cv2.moments(contour)
                    if M["m00"] == 0:
                        continue
                    
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m10"] / M["m00"])
                    
                    perimeter = cv2.arcLength(contour, True)
                    circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0
                    
                    area_confidence = 1.0 - abs(area - 500) / 500
                    area_confidence = max(0, min(1, area_confidence))
                    
                    confidence = (circularity * 0.6 + area_confidence * 0.4)
                    
                    if confidence > best_confidence:
                        best_confidence = confidence
                        best_detection = (cx, cy, confidence)
            
            return best_detection
            
        except Exception as e:
            logger.error(f"Ball detection error: {e}")
            return None


class BounceAnalysisDemo:
    """Demo to analyze bounce detection issues"""
    
    def __init__(self, video_path: str, model_path: str):
        self.video_path = video_path
        self.model_path = model_path
        
        self.bounce_analyzer = BounceAnalyzer(model_path)
        self.ball_tracker = BallTracker()
        
        # Analysis data
        self.frame_data = []
        self.high_confidence_frames = []
        
    def run_analysis(self):
        """Run the analysis"""
        if not Path(self.video_path).exists():
            logger.error(f"Video not found: {self.video_path}")
            return
        
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            logger.error(f"Could not open video: {self.video_path}")
            return
        
        logger.info("🔍 Starting Bounce Detection Analysis")
        logger.info("This will analyze why the model is over-detecting bounces")
        
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Detect ball
            ball_detection = self.ball_tracker.detect_ball(frame)
            
            if ball_detection:
                x, y, confidence = ball_detection
                
                # Add to analyzer
                self.bounce_analyzer.add_ball_position(x, y, time.time())
                
                # Get prediction analysis
                bounce_confidence, features, feature_analysis = self.bounce_analyzer.analyze_prediction()
                
                # Store frame data
                frame_info = {
                    'frame': frame_count,
                    'ball_x': x,
                    'ball_y': y,
                    'ball_confidence': confidence,
                    'bounce_confidence': bounce_confidence,
                    'features': features,
                    'feature_analysis': feature_analysis
                }
                
                self.frame_data.append(frame_info)
                
                # Track high confidence frames
                if bounce_confidence > 0.4:
                    self.high_confidence_frames.append(frame_info)
                
                # Show progress every 50 frames
                if frame_count % 50 == 0:
                    logger.info(f"Frame {frame_count}: Bounce confidence: {bounce_confidence:.3f}")
        
        cap.release()
        
        # Analyze results
        self.analyze_results()
    
    def analyze_results(self):
        """Analyze the analysis results"""
        logger.info("\n" + "="*60)
        logger.info("BOUNCE DETECTION ANALYSIS RESULTS")
        logger.info("="*60)
        
        if not self.frame_data:
            logger.info("No frame data collected")
            return
        
        # Basic statistics
        total_frames = len(self.frame_data)
        confidences = [f['bounce_confidence'] for f in self.frame_data]
        
        logger.info(f"Total Frames Analyzed: {total_frames}")
        logger.info(f"Average Bounce Confidence: {np.mean(confidences):.3f}")
        logger.info(f"Min Confidence: {np.min(confidences):.3f}")
        logger.info(f"Max Confidence: {np.max(confidences):.3f}")
        logger.info(f"Std Confidence: {np.std(confidences):.3f}")
        
        # Confidence distribution
        logger.info(f"\nConfidence Distribution:")
        thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
        for threshold in thresholds:
            count = sum(1 for c in confidences if c > threshold)
            percentage = (count / total_frames) * 100
            logger.info(f"  >{threshold:.1f}: {count} frames ({percentage:.1f}%)")
        
        # Feature analysis
        if self.frame_data:
            logger.info(f"\nFeature Analysis (first frame):")
            first_frame = self.frame_data[0]
            if first_frame['feature_analysis']:
                analysis = first_frame['feature_analysis']
                
                logger.info(f"  X Movement:")
                logger.info(f"    Avg X diff: {analysis['x_movement']['avg_x_diff']:.2f}")
                logger.info(f"    Avg X diff inv: {analysis['x_movement']['avg_x_diff_inv']:.2f}")
                logger.info(f"    X direction change: {analysis['x_movement']['x_direction_change']:.2f}")
                logger.info(f"    X ratio avg: {analysis['x_movement']['x_ratio_avg']:.2f}")
                
                logger.info(f"  Y Movement:")
                logger.info(f"    Avg Y diff: {analysis['y_movement']['avg_y_diff']:.2f}")
                logger.info(f"    Avg Y diff inv: {analysis['y_movement']['avg_y_diff_inv']:.2f}")
                logger.info(f"    Y direction change: {analysis['y_movement']['y_direction_change']:.2f}")
                logger.info(f"    Y ratio avg: {analysis['y_movement']['y_ratio_avg']:.2f}")
                
                logger.info(f"  Movement Patterns:")
                logger.info(f"    High X movement: {analysis['movement_patterns']['high_x_movement']}")
                logger.info(f"    High Y movement: {analysis['movement_patterns']['high_y_movement']}")
                logger.info(f"    Direction reversal: {analysis['movement_patterns']['direction_reversal']}")
                logger.info(f"    Smooth trajectory: {analysis['movement_patterns']['smooth_trajectory']}")
        
        # High confidence frame analysis
        if self.high_confidence_frames:
            logger.info(f"\nHigh Confidence Frames (>0.4): {len(self.high_confidence_frames)}")
            logger.info("These frames have the highest bounce confidence:")
            
            # Sort by confidence
            high_conf_sorted = sorted(self.high_confidence_frames, 
                                    key=lambda x: x['bounce_confidence'], reverse=True)
            
            for i, frame_info in enumerate(high_conf_sorted[:5]):  # Show top 5
                logger.info(f"  Frame {frame_info['frame']}: {frame_info['bounce_confidence']:.3f}")
        
        logger.info("="*60)
        
        # Save detailed data to CSV for further analysis
        self.save_analysis_data()
    
    def save_analysis_data(self):
        """Save analysis data to CSV for further investigation"""
        if not self.frame_data:
            return
        
        # Prepare data for CSV
        csv_data = []
        for frame_info in self.frame_data:
            row = {
                'frame': frame_info['frame'],
                'ball_x': frame_info['ball_x'],
                'ball_y': frame_info['ball_y'],
                'ball_confidence': frame_info['ball_confidence'],
                'bounce_confidence': frame_info['bounce_confidence']
            }
            
            # Add features
            if frame_info['features']:
                for i, feature in enumerate(frame_info['features']):
                    row[f'feature_{i}'] = feature
            
            # Add feature analysis
            if frame_info['feature_analysis']:
                analysis = frame_info['feature_analysis']
                row['avg_x_diff'] = analysis['x_movement']['avg_x_diff']
                row['avg_y_diff'] = analysis['y_movement']['avg_y_diff']
                row['x_direction_change'] = analysis['x_movement']['x_direction_change']
                row['y_direction_change'] = analysis['y_movement']['y_direction_change']
                row['x_ratio_avg'] = analysis['x_movement']['x_ratio_avg']
                row['y_ratio_avg'] = analysis['y_movement']['y_ratio_avg']
            
            csv_data.append(row)
        
        # Save to CSV
        df = pd.DataFrame(csv_data)
        output_file = 'bounce_analysis_data.csv'
        df.to_csv(output_file, index=False)
        logger.info(f"Detailed analysis data saved to: {output_file}")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Bounce Detection Analysis")
    parser.add_argument("--video", type=str, default="tennis_test5.mp4", 
                       help="Path to input video file")
    parser.add_argument("--model", type=str, default="models/bounce_detector.cbm",
                       help="Path to bounce detection model")
    
    args = parser.parse_args()
    
    # Check files
    if not Path(args.video).exists():
        logger.error(f"Video file not found: {args.video}")
        return
    
    if not Path(args.model).exists():
        logger.error(f"Model file not found: {args.model}")
        return
    
    # Run analysis
    demo = BounceAnalysisDemo(args.video, args.model)
    demo.run_analysis()


if __name__ == "__main__":
    main()
