#!/usr/bin/env python3
"""
Tennis Court Segmentation System

This script segments the tennis court into different zones for analytics.
Based on discussions with Duke tennis coaches, the court is divided into
strategic zones for shot analysis and player positioning.

Usage:
    python court_segmenter.py --video tennis_test5.mp4 --csv tennis_analysis_data.csv --viewer
"""

import cv2
import numpy as np
import pandas as pd
import argparse
import logging
from typing import List, Tuple, Optional, Dict
import os

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')
logger = logging.getLogger(__name__)


class CourtSegmenter:
    """Tennis court segmentation system for zone-based analytics"""
    
    def __init__(self):
        """Initialize court segmenter"""
        self.court_keypoints = None
        self.court_zones = {}
        
        # Define court zones based on tennis strategy
        # These will be calculated from court keypoints
        self.zone_definitions = {
            'service_box_left': None,
            'service_box_right': None,
            'baseline_left': None,
            'baseline_right': None,
            'net_area': None,
            'no_mans_land': None
        }
        
        logger.info("Court segmenter initialized")
    
    def load_court_keypoints(self, csv_file: str):
        """Load court keypoints from CSV data"""
        try:
            df = pd.read_csv(csv_file)
            
            # Look for court keypoints in the CSV
            if 'court_keypoints' in df.columns:
                # Get the first valid court keypoints
                for idx, row in df.iterrows():
                    court_keypoints_str = row.get('court_keypoints', '')
                    if court_keypoints_str and court_keypoints_str != '':
                        # Parse court keypoints string
                        court_keypoints = self._parse_court_keypoints(court_keypoints_str)
                        if court_keypoints and len(court_keypoints) >= 4:
                            self.court_keypoints = court_keypoints
                            self._calculate_court_zones()
                            logger.info(f"Loaded court keypoints from frame {idx}")
                            return True
                logger.warning("No valid court keypoints found in CSV")
            else:
                logger.warning("No court_keypoints column found in CSV")
                
        except Exception as e:
            logger.error(f"Error loading court keypoints: {e}")
        
        return False
    
    def _parse_court_keypoints(self, keypoints_str: str) -> List[Tuple]:
        """Parse court keypoints from CSV string format"""
        try:
            if not keypoints_str or keypoints_str == '':
                return []
            
            # Parse the keypoints string (format: "x1,y1;x2,y2;...")
            keypoints = []
            points = keypoints_str.split(';')
            for point in points:
                if ',' in point:
                    # Handle x,y format
                    parts = point.split(',')
                    if len(parts) >= 2:
                        try:
                            x_val = float(parts[0]) if parts[0] != 'nan' and parts[0] != '' else None
                            y_val = float(parts[1]) if parts[1] != 'nan' and parts[1] != '' else None
                            keypoints.append((x_val, y_val))
                        except ValueError:
                            keypoints.append((None, None))
                    else:
                        keypoints.append((None, None))
                else:
                    keypoints.append((None, None))
            
            return keypoints
        except Exception as e:
            logger.warning(f"Error parsing court keypoints: {e}")
            return []
    
    def _calculate_court_zones(self):
        """Calculate court zones based on keypoints"""
        if not self.court_keypoints or len(self.court_keypoints) < 4:
            logger.warning("Not enough court keypoints for zone calculation")
            return
        
        # Extract valid keypoints
        valid_points = []
        for point in self.court_keypoints:
            if point[0] is not None and point[1] is not None:
                valid_points.append((int(point[0]), int(point[1])))
        
        if len(valid_points) < 4:
            logger.warning("Not enough valid court keypoints for zone calculation")
            return
        
        # For now, use a simple rectangular court assumption
        # In a real implementation, we'd use the actual court keypoints
        # to define proper service boxes, baselines, etc.
        
        # Get court bounds
        x_coords = [p[0] for p in valid_points]
        y_coords = [p[1] for p in valid_points]
        
        min_x, max_x = min(x_coords), max(x_coords)
        min_y, max_y = min(y_coords), max(y_coords)
        
        # Calculate court dimensions
        court_width = max_x - min_x
        court_height = max_y - min_y
        
        # Define zones (simplified rectangular approximation)
        # Service boxes (top half, divided into left and right)
        service_y = min_y + court_height * 0.4  # Service line approximately 40% down
        service_mid_x = min_x + court_width * 0.5  # Middle of court
        
        self.zone_definitions = {
            'service_box_left': {
                'points': np.array([
                    [min_x, min_y],
                    [service_mid_x, min_y],
                    [service_mid_x, service_y],
                    [min_x, service_y]
                ], dtype=np.int32),
                'color': (0, 255, 255),  # Yellow
                'name': 'Service Box Left'
            },
            'service_box_right': {
                'points': np.array([
                    [service_mid_x, min_y],
                    [max_x, min_y],
                    [max_x, service_y],
                    [service_mid_x, service_y]
                ], dtype=np.int32),
                'color': (0, 255, 255),  # Yellow
                'name': 'Service Box Right'
            },
            'baseline_left': {
                'points': np.array([
                    [min_x, service_y],
                    [service_mid_x, service_y],
                    [service_mid_x, max_y],
                    [min_x, max_y]
                ], dtype=np.int32),
                'color': (255, 0, 255),  # Magenta
                'name': 'Baseline Left'
            },
            'baseline_right': {
                'points': np.array([
                    [service_mid_x, service_y],
                    [max_x, service_y],
                    [max_x, max_y],
                    [service_mid_x, max_y]
                ], dtype=np.int32),
                'color': (255, 0, 255),  # Magenta
                'name': 'Baseline Right'
            },
            'net_area': {
                'points': np.array([
                    [min_x, service_y - 20],
                    [max_x, service_y - 20],
                    [max_x, service_y + 20],
                    [min_x, service_y + 20]
                ], dtype=np.int32),
                'color': (0, 255, 0),  # Green
                'name': 'Net Area'
            }
        }
        
        logger.info(f"Court zones calculated with {len(valid_points)} keypoints")
        logger.info(f"Court bounds: ({min_x}, {min_y}) to ({max_x}, {max_y})")
    
    def get_zone_for_point(self, x: float, y: float) -> Optional[str]:
        """Get the zone name for a given point"""
        if not self.zone_definitions:
            return None
        
        point = (int(x), int(y))
        
        for zone_name, zone_data in self.zone_definitions.items():
            if zone_data and 'points' in zone_data:
                result = cv2.pointPolygonTest(zone_data['points'], point, False)
                if result >= 0:  # Inside or on the boundary
                    return zone_name
        
        return 'out_of_court'
    
    def draw_court_overlay(self, frame: np.ndarray) -> np.ndarray:
        """Draw court zones overlay on frame"""
        if not self.zone_definitions:
            return frame
        
        overlay = frame.copy()
        
        # Draw each zone
        for zone_name, zone_data in self.zone_definitions.items():
            if zone_data and 'points' in zone_data:
                # Draw filled polygon with transparency
                cv2.fillPoly(overlay, [zone_data['points']], zone_data['color'])
                
                # Draw zone boundary
                cv2.polylines(overlay, [zone_data['points']], True, (255, 255, 255), 2)
                
                # Add zone label
                if 'name' in zone_data:
                    # Get center of zone for label
                    center = np.mean(zone_data['points'], axis=0).astype(int)
                    cv2.putText(overlay, zone_data['name'], (center[0] - 50, center[1]), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Blend overlay with original frame
        alpha = 0.3  # Transparency
        result = cv2.addWeighted(frame, 1 - alpha, overlay, alpha, 0)
        
        return result
    
    def draw_court_keypoints(self, frame: np.ndarray) -> np.ndarray:
        """Draw court keypoints on frame"""
        if not self.court_keypoints:
            return frame
        
        for i, point in enumerate(self.court_keypoints):
            if point[0] is not None and point[1] is not None:
                x, y = int(point[0]), int(point[1])
                # Draw keypoint
                cv2.circle(frame, (x, y), 8, (0, 0, 255), -1)  # Red filled circle
                cv2.circle(frame, (x, y), 12, (255, 255, 255), 2)  # White outline
                # Add keypoint number
                cv2.putText(frame, str(i), (x + 15, y - 15), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return frame


class CourtSegmentationProcessor:
    """Processes video with court segmentation overlay"""
    
    def __init__(self):
        """Initialize processor"""
        self.court_segmenter = CourtSegmenter()
        self.zone_analytics = {}  # Track analytics per zone
    
    def process_video(self, video_file: str, csv_file: str, output_file: str = None, show_viewer: bool = False):
        """Process video with court segmentation"""
        # Load court keypoints
        if not self.court_segmenter.load_court_keypoints(csv_file):
            logger.error("Failed to load court keypoints")
            return
        
        # Load CSV data
        df = pd.read_csv(csv_file)
        cap = cv2.VideoCapture(video_file)
        
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_file}")
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        logger.info(f"Processing {len(df)} frames from {video_file}")
        logger.info(f"Video: {width}x{height} @ {fps}fps")
        
        # Setup video writer if output specified
        out = None
        if output_file:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_file, fourcc, fps, (width, height))
        
        # Setup viewer
        if show_viewer:
            cv2.namedWindow('Tennis Court Segmentation', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('Tennis Court Segmentation', 1200, 800)
        
        try:
            for idx, row in df.iterrows():
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Get ball position from CSV
                ball_x = self._parse_float(row.get('ball_x', ''))
                ball_y = self._parse_float(row.get('ball_y', ''))
                
                # Draw court overlay
                frame_with_overlay = self.court_segmenter.draw_court_overlay(frame)
                
                # Draw court keypoints
                frame_with_overlay = self.court_segmenter.draw_court_keypoints(frame_with_overlay)
                
                # Add ball position and zone info
                frame_with_overlay = self._add_ball_info(frame_with_overlay, ball_x, ball_y, idx)
                
                # Write frame
                if out:
                    out.write(frame_with_overlay)
                
                # Show in viewer
                if show_viewer:
                    cv2.imshow('Tennis Court Segmentation', frame_with_overlay)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        break
                    elif key == ord(' '):  # Space to pause
                        cv2.waitKey(0)
                
                # Progress update
                if idx % 30 == 0:
                    logger.info(f"Processed {idx}/{len(df)} frames ({idx/len(df)*100:.1f}%)")
        
        finally:
            # Cleanup
            cap.release()
            if out:
                out.release()
            if show_viewer:
                cv2.destroyAllWindows()
        
        logger.info("Court segmentation processing completed!")
    
    def _parse_float(self, value: str) -> Optional[float]:
        """Parse float value from CSV string"""
        try:
            if pd.isna(value) or value == '':
                return None
            return float(value)
        except (ValueError, TypeError):
            return None
    
    def _add_ball_info(self, frame: np.ndarray, ball_x: Optional[float], ball_y: Optional[float], frame_number: int) -> np.ndarray:
        """Add ball position and zone information to frame"""
        # Frame info
        cv2.putText(frame, f"Frame: {frame_number}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Ball position and zone
        if ball_x is not None and ball_y is not None:
            bx, by = int(ball_x), int(ball_y)
            cv2.circle(frame, (bx, by), 8, (0, 255, 255), -1)  # Yellow ball
            cv2.putText(frame, "Ball", (bx + 10, by), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            
            # Get zone for ball
            zone = self.court_segmenter.get_zone_for_point(ball_x, ball_y)
            if zone:
                cv2.putText(frame, f"Zone: {zone}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        return frame


def main():
    """Main function for standalone usage"""
    parser = argparse.ArgumentParser(description='Tennis Court Segmentation System')
    parser.add_argument('--video', required=True, help='Input video file')
    parser.add_argument('--csv', required=True, help='Input CSV file with court keypoints data')
    parser.add_argument('--output', default='tennis_court_segmentation.mp4', help='Output video file')
    parser.add_argument('--viewer', action='store_true', help='Show real-time viewer')
    
    args = parser.parse_args()
    
    processor = CourtSegmentationProcessor()
    processor.process_video(args.video, args.csv, args.output, show_viewer=args.viewer)


if __name__ == "__main__":
    main()
