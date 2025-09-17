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
        self.zone_definitions = {}
        
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
        """Calculate court zones based on actual court keypoints"""
        if not self.court_keypoints or len(self.court_keypoints) < 14:
            logger.warning("Not enough court keypoints for zone calculation (need 14)")
            return
        
        # Extract valid keypoints (court keypoints are 0-indexed)
        valid_points = []
        for point in self.court_keypoints:
            if point[0] is not None and point[1] is not None:
                valid_points.append((int(point[0]), int(point[1])))
            else:
                valid_points.append(None)
        
        if len([p for p in valid_points if p is not None]) < 14:
            logger.warning("Not enough valid court keypoints for zone calculation")
            return
        
        # Define colors for different zone types
        colors_4_zones = [(255, 100, 100), (100, 255, 100), (100, 100, 255), (255, 255, 100)]  # Red, Green, Blue, Yellow
        colors_3_zones = [(255, 100, 255), (100, 255, 255), (255, 200, 100)]  # Magenta, Cyan, Orange
        
        # Main court zones with Duke tennis terminology
        # Region 1: Points 10, 5, 11, 7 (4 vertical zones) - A, B, C, D
        if (valid_points[10] and valid_points[5] and valid_points[11] and valid_points[7]):
            p10, p5, p11, p7 = valid_points[10], valid_points[5], valid_points[11], valid_points[7]
            self._create_4_vertical_zones(p10, p5, p11, p7, colors_4_zones, 'region1', ['A', 'B', 'C', 'D'])
        
        # Region 2: Points 4, 8, 6, 9 (4 vertical zones) - A, B, C, D
        if (valid_points[4] and valid_points[8] and valid_points[6] and valid_points[9]):
            p4, p8, p6, p9 = valid_points[4], valid_points[8], valid_points[6], valid_points[9]
            self._create_4_vertical_zones(p4, p8, p6, p9, colors_4_zones, 'region2', ['A', 'B', 'C', 'D'])
        
        # Region 3: Points 8, 10, 12, 13 (3 vertical zones) - BROAD, MID, TEE
        if (valid_points[8] and valid_points[10] and valid_points[12] and valid_points[13]):
            p8, p10, p12, p13 = valid_points[8], valid_points[10], valid_points[12], valid_points[13]
            self._create_3_vertical_zones(p8, p10, p12, p13, colors_3_zones, 'region3', ['BROAD', 'MID', 'TEE'])
        
        # Region 4: Points 9, 11, 12, 13 (3 vertical zones) - BROAD, MID, TEE
        if (valid_points[9] and valid_points[11] and valid_points[12] and valid_points[13]):
            p9, p11, p12, p13 = valid_points[9], valid_points[11], valid_points[12], valid_points[13]
            self._create_3_vertical_zones(p9, p11, p12, p13, colors_3_zones, 'region4', ['BROAD', 'MID', 'TEE'])
        
        # Doubles lane zones with Duke tennis terminology
        doubles_colors = [(200, 200, 200), (150, 150, 150), (100, 100, 100), (50, 50, 50)]  # Gray shades
        
        # Left doubles lane: Points 2, 5, 10 (need to find 4th point) - AA
        if (valid_points[2] and valid_points[5] and valid_points[10]):
            p2, p5, p10 = valid_points[2], valid_points[5], valid_points[10]
            p4th = self._find_rectangle_4th_point(p2, p5, p10, "left_doubles")
            self._create_single_zone([p2, p5, p10, p4th], doubles_colors[0], 'doubles_left', 'AA')
        
        # Right doubles lane: Points 3, 7, 11 (need to find 4th point) - DD
        if (valid_points[3] and valid_points[7] and valid_points[11]):
            p3, p7, p11 = valid_points[3], valid_points[7], valid_points[11]
            p4th = self._find_rectangle_4th_point(p3, p7, p11, "right_doubles")
            self._create_single_zone([p3, p7, p11, p4th], doubles_colors[1], 'doubles_right', 'DD')
        
        # Top doubles lane: Points 0, 4, 8 (need to find 4th point) - AA
        if (valid_points[0] and valid_points[4] and valid_points[8]):
            p0, p4, p8 = valid_points[0], valid_points[4], valid_points[8]
            p4th = self._find_rectangle_4th_point(p0, p4, p8, "top_doubles")
            self._create_single_zone([p0, p4, p8, p4th], doubles_colors[2], 'doubles_top', 'AA')
        
        # Bottom doubles lane: Points 1, 6, 9 (need to find 4th point) - DD
        if (valid_points[1] and valid_points[6] and valid_points[9]):
            p1, p6, p9 = valid_points[1], valid_points[6], valid_points[9]
            p4th = self._find_rectangle_4th_point(p1, p6, p9, "bottom_doubles")
            self._create_single_zone([p1, p6, p9, p4th], doubles_colors[3], 'doubles_bottom', 'DD')
        
        logger.info(f"Court zones calculated with {len([p for p in valid_points if p is not None])} keypoints")
        logger.info("Main court zones: A, B, C, D (service boxes) + BROAD, MID, TEE (baseline)")
        logger.info("Doubles lane zones: AA (left/top) + DD (right/bottom)")
    
    def _create_4_vertical_zones(self, p1, p2, p3, p4, colors, region_name, zone_names):
        """Create 4 vertical zones from 4 corner points using homography"""
        # Define the source rectangle (ideal court geometry)
        # We'll create a perfect rectangle and then transform it to match the actual court
        src_width = 1000  # Arbitrary width for ideal rectangle
        src_height = 500  # Arbitrary height for ideal rectangle
        
        # Source rectangle corners (ideal geometry)
        src_corners = np.array([
            [0, 0],                    # Top-left
            [src_width, 0],            # Top-right
            [src_width, src_height],   # Bottom-right
            [0, src_height]            # Bottom-left
        ], dtype=np.float32)
        
        # Destination corners (actual court keypoints)
        # We need to order them properly: top-left, top-right, bottom-right, bottom-left
        dst_corners = self._order_rectangle_corners([p1, p2, p3, p4])
        dst_corners = np.array(dst_corners, dtype=np.float32)
        
        # Calculate homography matrix
        homography = cv2.getPerspectiveTransform(src_corners, dst_corners)
        
        # Create 4 vertical zones in the ideal space
        quarter_width = src_width / 4
        
        for i in range(4):
            zone_name = f'{region_name}_{zone_names[i]}'
            left_x = i * quarter_width
            right_x = (i + 1) * quarter_width
            
            # Define zone corners in ideal space
            ideal_zone_corners = np.array([
                [left_x, 0],
                [right_x, 0],
                [right_x, src_height],
                [left_x, src_height]
            ], dtype=np.float32)
            
            # Transform to actual court coordinates
            transformed_corners = cv2.perspectiveTransform(
                ideal_zone_corners.reshape(-1, 1, 2), homography
            ).reshape(-1, 2)
            
            self.zone_definitions[zone_name] = {
                'points': transformed_corners.astype(np.int32),
                'color': colors[i],
                'name': zone_names[i]
            }
    
    def _create_3_vertical_zones(self, p1, p2, p3, p4, colors, region_name, zone_names):
        """Create 3 vertical zones from 4 corner points using homography"""
        # Define the source rectangle (ideal court geometry)
        src_width = 1000  # Arbitrary width for ideal rectangle
        src_height = 500  # Arbitrary height for ideal rectangle
        
        # Source rectangle corners (ideal geometry)
        src_corners = np.array([
            [0, 0],                    # Top-left
            [src_width, 0],            # Top-right
            [src_width, src_height],   # Bottom-right
            [0, src_height]            # Bottom-left
        ], dtype=np.float32)
        
        # Destination corners (actual court keypoints)
        dst_corners = self._order_rectangle_corners([p1, p2, p3, p4])
        dst_corners = np.array(dst_corners, dtype=np.float32)
        
        # Calculate homography matrix
        homography = cv2.getPerspectiveTransform(src_corners, dst_corners)
        
        # Create 3 vertical zones in the ideal space
        third_width = src_width / 3
        
        for i in range(3):
            zone_name = f'{region_name}_{zone_names[i]}'
            left_x = i * third_width
            right_x = (i + 1) * third_width
            
            # Define zone corners in ideal space
            ideal_zone_corners = np.array([
                [left_x, 0],
                [right_x, 0],
                [right_x, src_height],
                [left_x, src_height]
            ], dtype=np.float32)
            
            # Transform to actual court coordinates
            transformed_corners = cv2.perspectiveTransform(
                ideal_zone_corners.reshape(-1, 1, 2), homography
            ).reshape(-1, 2)
            
            self.zone_definitions[zone_name] = {
                'points': transformed_corners.astype(np.int32),
                'color': colors[i],
                'name': zone_names[i]
            }
    
    def _order_rectangle_corners(self, corners):
        """Order 4 corner points as top-left, top-right, bottom-right, bottom-left"""
        # Convert to numpy array for easier manipulation
        corners = np.array(corners)
        
        # Calculate centroid
        centroid = np.mean(corners, axis=0)
        
        # Sort by angle from centroid
        def angle_from_centroid(point):
            return np.arctan2(point[1] - centroid[1], point[0] - centroid[0])
        
        # Sort corners by angle
        sorted_indices = np.argsort([angle_from_centroid(corner) for corner in corners])
        sorted_corners = corners[sorted_indices]
        
        # Now we need to identify which is which
        # Find the point with minimum y (top) and minimum x (left) among top points
        top_points = sorted_corners[sorted_corners[:, 1] <= np.median(sorted_corners[:, 1])]
        bottom_points = sorted_corners[sorted_corners[:, 1] > np.median(sorted_corners[:, 1])]
        
        # Order top points by x
        top_points = top_points[top_points[:, 0].argsort()]
        # Order bottom points by x
        bottom_points = bottom_points[bottom_points[:, 0].argsort()]
        
        # Return in order: top-left, top-right, bottom-right, bottom-left
        return [top_points[0], top_points[1], bottom_points[1], bottom_points[0]]
    
    def _find_rectangle_4th_point(self, p1, p2, p3, zone_type="general"):
        """Find the 4th point to complete a rectangle given 3 points"""
        # Convert to numpy arrays
        p1, p2, p3 = np.array(p1), np.array(p2), np.array(p3)
        
        if zone_type == "left_doubles":
            # Left 4th: extend horizontally from point 10 (p3) and on line from 2 to 0
            # We need to get points 2 and 0 from the court keypoints
            if hasattr(self, 'court_keypoints') and self.court_keypoints:
                valid_points = []
                for point in self.court_keypoints:
                    if point[0] is not None and point[1] is not None:
                        valid_points.append((int(point[0]), int(point[1])))
                    else:
                        valid_points.append(None)
                
                if valid_points[2] and valid_points[0]:
                    p2_ref = np.array(valid_points[2])
                    p0_ref = np.array(valid_points[0])
                    # Find intersection of horizontal line from p3 with line from p2_ref to p0_ref
                    p4 = self._find_line_intersection(p3, p2_ref, p0_ref, horizontal=True)
                else:
                    p4 = p1 + p2 - p3  # Fallback
            else:
                p4 = p1 + p2 - p3  # Fallback
                
        elif zone_type == "right_doubles":
            # Right 4th: extend horizontally from point 11 (p3) and on line from 1 to 3
            if hasattr(self, 'court_keypoints') and self.court_keypoints:
                valid_points = []
                for point in self.court_keypoints:
                    if point[0] is not None and point[1] is not None:
                        valid_points.append((int(point[0]), int(point[1])))
                    else:
                        valid_points.append(None)
                
                if valid_points[1] and valid_points[3]:
                    p1_ref = np.array(valid_points[1])
                    p3_ref = np.array(valid_points[3])
                    # Find intersection of horizontal line from p3 with line from p1_ref to p3_ref
                    p4 = self._find_line_intersection(p3, p1_ref, p3_ref, horizontal=True)
                else:
                    p4 = p1 + p2 - p3  # Fallback
            else:
                p4 = p1 + p2 - p3  # Fallback
                
        elif zone_type == "top_doubles":
            # Top 4th: extend horizontally from point 8 (p3) and on line from 0 to 2
            if hasattr(self, 'court_keypoints') and self.court_keypoints:
                valid_points = []
                for point in self.court_keypoints:
                    if point[0] is not None and point[1] is not None:
                        valid_points.append((int(point[0]), int(point[1])))
                    else:
                        valid_points.append(None)
                
                if valid_points[0] and valid_points[2]:
                    p0_ref = np.array(valid_points[0])
                    p2_ref = np.array(valid_points[2])
                    # Find intersection of horizontal line from p3 with line from p0_ref to p2_ref
                    p4 = self._find_line_intersection(p3, p0_ref, p2_ref, horizontal=True)
                else:
                    p4 = p1 + p2 - p3  # Fallback
            else:
                p4 = p1 + p2 - p3  # Fallback
                
        elif zone_type == "bottom_doubles":
            # Bottom 4th: extend horizontally from point 9 (p3) and on line from 1 to 3
            if hasattr(self, 'court_keypoints') and self.court_keypoints:
                valid_points = []
                for point in self.court_keypoints:
                    if point[0] is not None and point[1] is not None:
                        valid_points.append((int(point[0]), int(point[1])))
                    else:
                        valid_points.append(None)
                
                if valid_points[1] and valid_points[3]:
                    p1_ref = np.array(valid_points[1])
                    p3_ref = np.array(valid_points[3])
                    # Find intersection of horizontal line from p3 with line from p1_ref to p3_ref
                    p4 = self._find_line_intersection(p3, p1_ref, p3_ref, horizontal=True)
                else:
                    p4 = p1 + p2 - p3  # Fallback
            else:
                p4 = p1 + p2 - p3  # Fallback
        else:
            # Fallback to diagonal method for any other zone types
            d12 = np.linalg.norm(p1 - p2)
            d13 = np.linalg.norm(p1 - p3)
            d23 = np.linalg.norm(p2 - p3)
            
            if d12 >= d13 and d12 >= d23:
                p4 = p1 + p2 - p3
            elif d13 >= d12 and d13 >= d23:
                p4 = p1 + p3 - p2
            else:
                p4 = p2 + p3 - p1
        
        return p4.astype(int).tolist()
    
    def _find_line_intersection(self, point, line_start, line_end, horizontal=True):
        """Find intersection of horizontal/vertical line from point with another line"""
        if horizontal:
            # Horizontal line from point: y = point[1]
            y = point[1]
            # Line from line_start to line_end
            if abs(line_end[0] - line_start[0]) < 1e-6:  # Vertical line
                x = line_start[0]
            else:
                # Calculate x where y = point[1] on the line
                slope = (line_end[1] - line_start[1]) / (line_end[0] - line_start[0])
                x = line_start[0] + (y - line_start[1]) / slope
        else:
            # Vertical line from point: x = point[0]
            x = point[0]
            # Line from line_start to line_end
            if abs(line_end[1] - line_start[1]) < 1e-6:  # Horizontal line
                y = line_start[1]
            else:
                # Calculate y where x = point[0] on the line
                slope = (line_end[1] - line_start[1]) / (line_end[0] - line_start[0])
                y = line_start[1] + (x - line_start[0]) * slope
        
        return np.array([x, y])
    
    def _create_single_zone(self, corners, color, zone_name, zone_label):
        """Create a single zone from 4 corner points using homography"""
        # Define the source rectangle (ideal court geometry)
        src_width = 1000
        src_height = 500
        
        # Source rectangle corners (ideal geometry)
        src_corners = np.array([
            [0, 0],                    # Top-left
            [src_width, 0],            # Top-right
            [src_width, src_height],   # Bottom-right
            [0, src_height]            # Bottom-left
        ], dtype=np.float32)
        
        # Destination corners (actual court keypoints)
        dst_corners = self._order_rectangle_corners(corners)
        dst_corners = np.array(dst_corners, dtype=np.float32)
        
        # Calculate homography matrix
        homography = cv2.getPerspectiveTransform(src_corners, dst_corners)
        
        # Define the entire zone in ideal space
        ideal_zone_corners = np.array([
            [0, 0],
            [src_width, 0],
            [src_width, src_height],
            [0, src_height]
        ], dtype=np.float32)
        
        # Transform to actual court coordinates
        transformed_corners = cv2.perspectiveTransform(
            ideal_zone_corners.reshape(-1, 1, 2), homography
        ).reshape(-1, 2)
        
        self.zone_definitions[zone_name] = {
            'points': transformed_corners.astype(np.int32),
            'color': color,
            'name': zone_label
        }
    
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
                    
                    # Draw zone name with larger, more visible text
                    font_scale = 1.2
                    thickness = 3
                    text_color = (255, 255, 255)  # White text
                    outline_color = (0, 0, 0)     # Black outline
                    
                    # Get text size for centering
                    text_size = cv2.getTextSize(zone_data['name'], cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]
                    text_x = center[0] - text_size[0] // 2
                    text_y = center[1] + text_size[1] // 2
                    
                    # Draw black outline
                    cv2.putText(overlay, zone_data['name'], (text_x - 1, text_y - 1), 
                               cv2.FONT_HERSHEY_SIMPLEX, font_scale, outline_color, thickness + 2)
                    cv2.putText(overlay, zone_data['name'], (text_x + 1, text_y + 1), 
                               cv2.FONT_HERSHEY_SIMPLEX, font_scale, outline_color, thickness + 2)
                    
                    # Draw white text
                    cv2.putText(overlay, zone_data['name'], (text_x, text_y), 
                               cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, thickness)
        
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
        
        # Draw calculated 4th points for doubles lanes
        self._draw_calculated_4th_points(frame)
        
        return frame
    
    def _draw_calculated_4th_points(self, frame: np.ndarray):
        """Draw the calculated 4th points for doubles lane zones"""
        if not self.court_keypoints:
            return
        
        # Get valid keypoints
        valid_points = []
        for point in self.court_keypoints:
            if point[0] is not None and point[1] is not None:
                valid_points.append((int(point[0]), int(point[1])))
            else:
                valid_points.append(None)
        
        # Calculate and draw 4th points for each doubles lane
        doubles_lanes = [
            ([2, 5, 10], "Left", "left_doubles"),
            ([3, 7, 11], "Right", "right_doubles"), 
            ([0, 4, 8], "Top", "top_doubles"),
            ([1, 6, 9], "Bottom", "bottom_doubles")
        ]
        
        colors = [(255, 0, 255), (0, 255, 255), (255, 255, 0), (255, 128, 0)]  # Magenta, Cyan, Yellow, Orange
        
        for i, (point_indices, name, zone_type) in enumerate(doubles_lanes):
            if all(valid_points[idx] for idx in point_indices):
                p1, p2, p3 = [valid_points[idx] for idx in point_indices]
                p4th = self._find_rectangle_4th_point(p1, p2, p3, zone_type)
                
                # Draw the 4th point
                x, y = int(p4th[0]), int(p4th[1])
                cv2.circle(frame, (x, y), 10, colors[i], -1)  # Filled circle
                cv2.circle(frame, (x, y), 14, (255, 255, 255), 2)  # White outline
                cv2.putText(frame, f"{name}4th", (x + 15, y - 15), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                
                # Draw lines connecting the 4 points to show the quadrilateral
                points = [p1, p2, p3, p4th]
                for j in range(4):
                    pt1 = tuple(points[j])
                    pt2 = tuple(points[(j + 1) % 4])
                    cv2.line(frame, pt1, pt2, colors[i], 2)


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
                # Draw zone info with larger, more visible text
                font_scale = 1.0
                thickness = 2
                text_color = (255, 255, 255)  # White text
                outline_color = (0, 0, 0)     # Black outline
                
                zone_text = f"Ball Zone: {zone}"
                text_size = cv2.getTextSize(zone_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]
                
                # Draw black outline
                cv2.putText(frame, zone_text, (9, 69), 
                           cv2.FONT_HERSHEY_SIMPLEX, font_scale, outline_color, thickness + 2)
                cv2.putText(frame, zone_text, (11, 71), 
                           cv2.FONT_HERSHEY_SIMPLEX, font_scale, outline_color, thickness + 2)
                
                # Draw white text
                cv2.putText(frame, zone_text, (10, 70), 
                           cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, thickness)
        
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
