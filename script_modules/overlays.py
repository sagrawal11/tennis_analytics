#!/usr/bin/env python3
"""
Tennis Analysis Overlay System
Overlays all analysis outputs onto a single video viewer
Designed to be easily extensible - just add new script names to integrate them
"""

import cv2
import numpy as np
import pandas as pd
import argparse
import logging
from typing import List, Dict, Optional, Tuple
import os
import sys

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')
logger = logging.getLogger(__name__)


class OverlayRenderer:
    """Renders overlays from different analysis scripts"""
    
    def __init__(self):
        """Initialize overlay renderer"""
        self.overlay_data = {}
        self.overlay_colors = {
            'ball': (0, 255, 255),      # Yellow
            'court': (255, 0, 255),     # Magenta  
            'positioning': (0, 255, 0), # Green
            'pose': (255, 0, 0),        # Red
            'default': (255, 255, 255)  # White
        }
        
    def load_overlay_data(self, script_name: str, csv_file: str) -> bool:
        """Load overlay data from a CSV file"""
        try:
            if not os.path.exists(csv_file):
                logger.warning(f"CSV file not found: {csv_file}")
                return False
                
            df = pd.read_csv(csv_file)
            self.overlay_data[script_name] = df
            logger.info(f"Loaded {len(df)} records from {script_name}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading {script_name} data: {e}")
            return False
    
    def render_ball_overlay(self, frame: np.ndarray, frame_number: int) -> np.ndarray:
        """Render ball detection overlay"""
        if 'ball' not in self.overlay_data:
            return frame
            
        df = self.overlay_data['ball']
        frame_data = df[df['frame'] == frame_number]
        
        for _, row in frame_data.iterrows():
            if pd.notna(row['x']) and pd.notna(row['y']):
                x, y = int(row['x']), int(row['y'])
                confidence = row.get('confidence', 0.5)
                
                # Draw ball circle with confidence-based size
                radius = max(3, int(confidence * 10))
                cv2.circle(frame, (x, y), radius, self.overlay_colors['ball'], 2)
                
                # Draw confidence text
                cv2.putText(frame, f"Ball: {confidence:.2f}", (x+10, y-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.overlay_colors['ball'], 1)
        
        return frame
    
    def render_court_overlay(self, frame: np.ndarray, frame_number: int) -> np.ndarray:
        """Render court keypoints overlay"""
        if 'court' not in self.overlay_data:
            return frame
            
        df = self.overlay_data['court']
        
        # Use the AVERAGE row (last row) for all frames since court keypoints are static
        average_row = df.iloc[-1]  # Get the last row which should be the AVERAGE
        keypoints_drawn = 0
        
        # Draw court keypoints
        for i in range(15):  # 15 keypoints
            x_col = f'keypoint_{i}_x'
            y_col = f'keypoint_{i}_y'
            
            if x_col in average_row and y_col in average_row and pd.notna(average_row[x_col]) and pd.notna(average_row[y_col]):
                x, y = int(average_row[x_col]), int(average_row[y_col])
                # Make court keypoints much more visible
                cv2.circle(frame, (x, y), 8, (0, 255, 255), -1)  # Bright yellow, larger
                cv2.circle(frame, (x, y), 12, (0, 0, 0), 2)     # Black outline
                cv2.putText(frame, str(i), (x+15, y-15), cv2.FONT_HERSHEY_SIMPLEX, 0.8, 
                           (0, 0, 0), 2)  # Black text, larger
                keypoints_drawn += 1
        
        # Debug output
        if frame_number % 30 == 0:  # Print every 30 frames
            print(f"Frame {frame_number}: Drew {keypoints_drawn} court keypoints from AVERAGE row")
        
        return frame
    
    def render_court_zones_overlay(self, frame: np.ndarray, frame_number: int) -> np.ndarray:
        """Render court zones overlay"""
        try:
            if 'court_zones' not in self.overlay_data:
                return frame
                
            df = self.overlay_data['court_zones']
            
            # Use the AVERAGE row (last row) for all frames since court zones are static
            average_row = df.iloc[-1]  # Get the last row which should be the AVERAGE
            
            # Extract keypoints for zone calculation
            keypoints = []
            for i in range(15):
                x_col = f'keypoint_{i}_x'
                y_col = f'keypoint_{i}_y'
                if x_col in average_row and y_col in average_row and pd.notna(average_row[x_col]) and pd.notna(average_row[y_col]):
                    keypoints.append((int(average_row[x_col]), int(average_row[y_col])))
                else:
                    keypoints.append(None)
            
            # Draw the 4 main court regions with proper subdivisions
            
            # 1. Near Side (Baseline to Service Line) - Bounded by 5, 7, 10, 11
            if all(keypoints[i] for i in [5, 7, 10, 11]):
                self._draw_near_side_zones(frame, keypoints[5], keypoints[7], keypoints[10], keypoints[11])
            
            # 2. Left Service Box - Bounded by 10, 13, 8, 12
            if all(keypoints[i] for i in [10, 13, 8, 12]):
                self._draw_left_service_box_zones(frame, keypoints[10], keypoints[13], keypoints[8], keypoints[12])
            
            # 3. Right Service Box - Bounded by 13, 11, 9, 12
            if all(keypoints[i] for i in [13, 11, 9, 12]):
                self._draw_right_service_box_zones(frame, keypoints[13], keypoints[11], keypoints[9], keypoints[12])
            
            # 4. Far Side (Service Line to Baseline) - Bounded by 4, 8, 6, 9
            if all(keypoints[i] for i in [4, 8, 6, 9]):
                self._draw_far_side_zones(frame, keypoints[4], keypoints[8], keypoints[6], keypoints[9])
            
            # 5. Near Side Left Doubles Alley (AA) - Bounded by 2, 5, 10, and left doubles alley
            if all(keypoints[i] for i in [2, 5, 10]):
                self._draw_near_left_doubles_alley(frame, keypoints[2], keypoints[5], keypoints[10])
            
            # 6. Near Side Right Doubles Alley (DD) - Bounded by 11, 7, 3, and right doubles alley  
            if all(keypoints[i] for i in [11, 7, 3]):
                self._draw_near_right_doubles_alley(frame, keypoints[11], keypoints[7], keypoints[3])
            
            # 7. Far Side Left Doubles Alley (AA) - Bounded by 0, 4, 8, and left doubles alley
            if all(keypoints[i] for i in [0, 4, 8]):
                self._draw_far_left_doubles_alley(frame, keypoints[0], keypoints[4], keypoints[8])
            
            # 8. Far Side Right Doubles Alley (DD) - Bounded by 1, 6, 9, and right doubles alley
            if all(keypoints[i] for i in [1, 6, 9]):
                self._draw_far_right_doubles_alley(frame, keypoints[1], keypoints[6], keypoints[9])
                
        except Exception as e:
            if frame_number == 0:
                print(f"DEBUG: Error in render_court_zones_overlay: {e}")
            return frame
        
        return frame
    
    def _draw_near_side_zones(self, frame, p5, p7, p10, p11):
        """Draw near side zones (A, B, C, D) - bounded by keypoints 5, 7, 10, 11"""
        zone_colors = [(100, 100, 255), (100, 255, 100), (255, 100, 100), (100, 255, 255)]
        zone_names = ['A', 'B', 'C', 'D']
        
        # Create the main quadrilateral
        main_quad = np.array([p5, p7, p11, p10], np.int32)
        
        # Draw 4 equal subdivisions
        for i in range(4):
            # Calculate subdivision points
            t = (i + 1) / 4.0
            
            # Interpolate along the top edge (p5 to p7)
            top_point = (
                int(p5[0] + t * (p7[0] - p5[0])),
                int(p5[1] + t * (p7[1] - p5[1]))
            )
            
            # Interpolate along the bottom edge (p10 to p11)
            bottom_point = (
                int(p10[0] + t * (p11[0] - p10[0])),
                int(p10[1] + t * (p11[1] - p10[1]))
            )
            
            # Create zone polygon
            if i == 0:
                zone_poly = np.array([p5, top_point, bottom_point, p10], np.int32)
            else:
                prev_t = i / 4.0
                prev_top = (
                    int(p5[0] + prev_t * (p7[0] - p5[0])),
                    int(p5[1] + prev_t * (p7[1] - p5[1]))
                )
                prev_bottom = (
                    int(p10[0] + prev_t * (p11[0] - p10[0])),
                    int(p10[1] + prev_t * (p11[1] - p10[1]))
                )
                zone_poly = np.array([prev_top, top_point, bottom_point, prev_bottom], np.int32)
            
            # Draw zone polygon with transparency
            overlay = frame.copy()
            cv2.fillPoly(overlay, [zone_poly], zone_colors[i])
            cv2.addWeighted(overlay, 0.2, frame, 0.8, 0, frame)
            cv2.polylines(frame, [zone_poly], True, (255, 255, 255), 2)
            
            # Draw zone label
            center = np.mean(zone_poly, axis=0).astype(int)
            cv2.putText(frame, zone_names[i], tuple(center), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    def _draw_left_service_box_zones(self, frame, p10, p13, p8, p12):
        """Draw left service box zones (WIDE, BODY, TEE) - bounded by keypoints 10, 13, 8, 12"""
        zone_colors = [(255, 100, 255), (255, 255, 100), (100, 200, 255)]
        zone_names = ['WIDE', 'BODY', 'TEE']
        
        # Draw 3 equal vertical subdivisions (from service line to baseline)
        for i in range(3):
            t = (i + 1) / 3.0
            
            # Interpolate along the service line edge (p10 to p13)
            service_point = (
                int(p10[0] + t * (p13[0] - p10[0])),
                int(p10[1] + t * (p13[1] - p10[1]))
            )
            
            # Interpolate along the baseline edge (p8 to p12)
            baseline_point = (
                int(p8[0] + t * (p12[0] - p8[0])),
                int(p8[1] + t * (p12[1] - p8[1]))
            )
            
            # Create zone polygon
            if i == 0:
                zone_poly = np.array([p10, service_point, baseline_point, p8], np.int32)
            else:
                prev_t = i / 3.0
                prev_service = (
                    int(p10[0] + prev_t * (p13[0] - p10[0])),
                    int(p10[1] + prev_t * (p13[1] - p10[1]))
                )
                prev_baseline = (
                    int(p8[0] + prev_t * (p12[0] - p8[0])),
                    int(p8[1] + prev_t * (p12[1] - p8[1]))
                )
                zone_poly = np.array([prev_service, service_point, baseline_point, prev_baseline], np.int32)
            
            # Draw zone polygon with transparency
            overlay = frame.copy()
            cv2.fillPoly(overlay, [zone_poly], zone_colors[i])
            cv2.addWeighted(overlay, 0.2, frame, 0.8, 0, frame)
            cv2.polylines(frame, [zone_poly], True, (255, 255, 255), 2)
            
            # Draw zone label
            center = np.mean(zone_poly, axis=0).astype(int)
            cv2.putText(frame, zone_names[i], tuple(center), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    def _draw_right_service_box_zones(self, frame, p13, p11, p9, p12):
        """Draw right service box zones (TEE, BODY, WIDE) - bounded by keypoints 13, 11, 9, 12"""
        zone_colors = [(100, 200, 255), (255, 255, 100), (255, 100, 255)]
        zone_names = ['TEE', 'BODY', 'WIDE']
        
        # Draw 3 equal vertical subdivisions (from service line to baseline)
        for i in range(3):
            t = (i + 1) / 3.0
            
            # Interpolate along the service line edge (p13 to p11)
            service_point = (
                int(p13[0] + t * (p11[0] - p13[0])),
                int(p13[1] + t * (p11[1] - p13[1]))
            )
            
            # Interpolate along the baseline edge (p12 to p9)
            baseline_point = (
                int(p12[0] + t * (p9[0] - p12[0])),
                int(p12[1] + t * (p9[1] - p12[1]))
            )
            
            # Create zone polygon
            if i == 0:
                zone_poly = np.array([p13, service_point, baseline_point, p12], np.int32)
            else:
                prev_t = i / 3.0
                prev_service = (
                    int(p13[0] + prev_t * (p11[0] - p13[0])),
                    int(p13[1] + prev_t * (p11[1] - p13[1]))
                )
                prev_baseline = (
                    int(p12[0] + prev_t * (p9[0] - p12[0])),
                    int(p12[1] + prev_t * (p9[1] - p12[1]))
                )
                zone_poly = np.array([prev_service, service_point, baseline_point, prev_baseline], np.int32)
            
            # Draw zone polygon with transparency
            overlay = frame.copy()
            cv2.fillPoly(overlay, [zone_poly], zone_colors[i])
            cv2.addWeighted(overlay, 0.2, frame, 0.8, 0, frame)
            cv2.polylines(frame, [zone_poly], True, (255, 255, 255), 2)
            
            # Draw zone label
            center = np.mean(zone_poly, axis=0).astype(int)
            cv2.putText(frame, zone_names[i], tuple(center), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    def _draw_far_side_zones(self, frame, p4, p8, p6, p9):
        """Draw far side zones (A, B, C, D) - bounded by keypoints 4, 8, 6, 9"""
        zone_colors = [(100, 100, 255), (100, 255, 100), (255, 100, 100), (100, 255, 255)]
        zone_names = ['A', 'B', 'C', 'D']
        
        # Draw 4 equal subdivisions
        for i in range(4):
            t = (i + 1) / 4.0
            
            # Interpolate along the top edge (p4 to p6)
            top_point = (
                int(p4[0] + t * (p6[0] - p4[0])),
                int(p4[1] + t * (p6[1] - p4[1]))
            )
            
            # Interpolate along the bottom edge (p8 to p9)
            bottom_point = (
                int(p8[0] + t * (p9[0] - p8[0])),
                int(p8[1] + t * (p9[1] - p8[1]))
            )
            
            # Create zone polygon
            if i == 0:
                zone_poly = np.array([p4, top_point, bottom_point, p8], np.int32)
            else:
                prev_t = i / 4.0
                prev_top = (
                    int(p4[0] + prev_t * (p6[0] - p4[0])),
                    int(p4[1] + prev_t * (p6[1] - p4[1]))
                )
                prev_bottom = (
                    int(p8[0] + prev_t * (p9[0] - p8[0])),
                    int(p8[1] + prev_t * (p9[1] - p8[1]))
                )
                zone_poly = np.array([prev_top, top_point, bottom_point, prev_bottom], np.int32)
            
            # Draw zone polygon with transparency
            overlay = frame.copy()
            cv2.fillPoly(overlay, [zone_poly], zone_colors[i])
            cv2.addWeighted(overlay, 0.2, frame, 0.8, 0, frame)
            cv2.polylines(frame, [zone_poly], True, (255, 255, 255), 2)
            
            # Draw zone label
            center = np.mean(zone_poly, axis=0).astype(int)
            cv2.putText(frame, zone_names[i], tuple(center), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    def _draw_near_left_doubles_alley(self, frame, p2, p5, p10):
        """Draw near side left doubles alley (AA) - bounded by keypoints 2, 5, 10"""
        zone_color = (200, 200, 200)  # Light gray for doubles alley
        
        # The doubles alley should be bounded by:
        # - Outside doubles sideline: keypoint 2 (vertical line)
        # - Inside doubles alley sideline: keypoints 5 to 10 (singles sideline)
        # - Line from 10 to outside doubles sideline (parallel to line from 2 to 5)
        # - Baseline: from p2 to p5
        
        # For a proper parallelogram, we need both sets of opposite sides to be parallel:
        # 1. Line from p2 to p5 should be parallel to line from p10 to sideline
        # 2. Line from p5 to p10 should be parallel to line from p2 to sideline
        
        # Calculate the direction vector from p2 to p5 (baseline direction)
        baseline_dx = p5[0] - p2[0]
        baseline_dy = p5[1] - p2[1]
        
        # Calculate the direction vector from p5 to p10 (service line direction)
        service_dx = p10[0] - p5[0]
        service_dy = p10[1] - p5[1]
        
        # The outside doubles sideline is vertical at p2[0]
        outside_sideline_x = p2[0]
        
        # Find the intersection point that satisfies both parallel constraints
        # We need to solve for the point (outside_sideline_x, y) such that:
        # 1. Line from p10 to this point is parallel to line from p2 to p5
        # 2. Line from p2 to this point is parallel to line from p5 to p10
        
        # For constraint 1: Line p10->(outside_sideline_x, y) parallel to p2->p5
        # Direction vector: (outside_sideline_x - p10[0], y - p10[1]) parallel to (baseline_dx, baseline_dy)
        # This gives us: (y - p10[1]) / (outside_sideline_x - p10[0]) = baseline_dy / baseline_dx
        
        # For constraint 2: Line p2->(outside_sideline_x, y) parallel to p5->p10  
        # Direction vector: (outside_sideline_x - p2[0], y - p2[1]) parallel to (service_dx, service_dy)
        # This gives us: (y - p2[1]) / (outside_sideline_x - p2[0]) = service_dy / service_dx
        
        # Solving these two equations for y:
        # From constraint 1: y = p10[1] + (baseline_dy/baseline_dx) * (outside_sideline_x - p10[0])
        # From constraint 2: y = p2[1] + (service_dy/service_dx) * (outside_sideline_x - p2[0])
        
        # Set them equal and solve for outside_sideline_x, then find y
        
        # Since outside_sideline_x = p2[0], we can substitute:
        if baseline_dx != 0 and service_dx != 0:
            # Using constraint 1 (since outside_sideline_x = p2[0]):
            outside_net_y = p10[1] + (baseline_dy / baseline_dx) * (p2[0] - p10[0])
        else:
            # Fallback if lines are vertical
            outside_net_y = p10[1]
        
        # Create the parallelogram with correct geometry
        zone_poly = np.array([
            p2,  # Outside baseline corner
            p5,  # Inside baseline corner
            p10, # Inside service line corner
            (int(outside_sideline_x), int(outside_net_y))  # Outside net corner
        ], np.int32)
        
        # Draw zone polygon with transparency
        overlay = frame.copy()
        cv2.fillPoly(overlay, [zone_poly], zone_color)
        cv2.addWeighted(overlay, 0.2, frame, 0.8, 0, frame)
        cv2.polylines(frame, [zone_poly], True, (255, 255, 255), 2)
        
        # Draw zone label
        center = np.mean(zone_poly, axis=0).astype(int)
        cv2.putText(frame, 'AA', tuple(center), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    def _draw_near_right_doubles_alley(self, frame, p11, p7, p3):
        """Draw near side right doubles alley (DD) - bounded by keypoints 11, 7, 3"""
        zone_color = (150, 150, 150)  # Darker gray for doubles alley
        
        # The doubles alley should be bounded by:
        # - Outside doubles sideline: keypoint 3 (vertical line)
        # - Inside doubles alley sideline: keypoints 7 to 11 (singles sideline)
        # - Line from 11 to outside doubles sideline (parallel to line from 3 to 7)
        # - Baseline: from p3 to p7
        
        # For a proper parallelogram, we need both sets of opposite sides to be parallel:
        # 1. Line from p3 to p7 should be parallel to line from p11 to sideline
        # 2. Line from p7 to p11 should be parallel to line from p3 to sideline
        
        # Calculate the direction vector from p3 to p7 (baseline direction)
        baseline_dx = p7[0] - p3[0]
        baseline_dy = p7[1] - p3[1]
        
        # Calculate the direction vector from p7 to p11 (service line direction)
        service_dx = p11[0] - p7[0]
        service_dy = p11[1] - p7[1]
        
        # The outside doubles sideline is vertical at p3[0]
        outside_sideline_x = p3[0]
        
        # Find the intersection point that satisfies both parallel constraints
        if baseline_dx != 0 and service_dx != 0:
            # Using constraint 1: line from p11 parallel to p3->p7
            outside_net_y = p11[1] + (baseline_dy / baseline_dx) * (p3[0] - p11[0])
        else:
            # Fallback if lines are vertical
            outside_net_y = p11[1]
        
        # Create the parallelogram with correct geometry
        zone_poly = np.array([
            p3,  # Outside baseline corner
            p7,  # Inside baseline corner
            p11, # Inside service line corner
            (int(outside_sideline_x), int(outside_net_y))  # Outside net corner
        ], np.int32)
        
        # Draw zone polygon with transparency
        overlay = frame.copy()
        cv2.fillPoly(overlay, [zone_poly], zone_color)
        cv2.addWeighted(overlay, 0.2, frame, 0.8, 0, frame)
        cv2.polylines(frame, [zone_poly], True, (255, 255, 255), 2)
        
        # Draw zone label
        center = np.mean(zone_poly, axis=0).astype(int)
        cv2.putText(frame, 'DD', tuple(center), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    def _draw_far_left_doubles_alley(self, frame, p0, p4, p8):
        """Draw far side left doubles alley (AA) - bounded by keypoints 0, 4, 8"""
        zone_color = (200, 200, 200)  # Light gray for doubles alley
        
        # The doubles alley should be bounded by:
        # - Outside doubles sideline: keypoint 0 (vertical line)
        # - Inside doubles alley sideline: keypoints 4 to 8 (singles sideline)
        # - Line from 8 to outside doubles sideline (parallel to line from 0 to 4)
        # - Baseline: from p0 to p4
        
        # For a proper parallelogram, we need both sets of opposite sides to be parallel:
        # 1. Line from p0 to p4 should be parallel to line from p8 to sideline
        # 2. Line from p4 to p8 should be parallel to line from p0 to sideline
        
        # Calculate the direction vector from p0 to p4 (baseline direction)
        baseline_dx = p4[0] - p0[0]
        baseline_dy = p4[1] - p0[1]
        
        # Calculate the direction vector from p4 to p8 (service line direction)
        service_dx = p8[0] - p4[0]
        service_dy = p8[1] - p4[1]
        
        # The outside doubles sideline is vertical at p0[0]
        outside_sideline_x = p0[0]
        
        # Find the intersection point that satisfies both parallel constraints
        if baseline_dx != 0 and service_dx != 0:
            # Using constraint 1: line from p8 parallel to p0->p4
            outside_net_y = p8[1] + (baseline_dy / baseline_dx) * (p0[0] - p8[0])
        else:
            # Fallback if lines are vertical
            outside_net_y = p8[1]
        
        # Create the parallelogram with correct geometry
        zone_poly = np.array([
            p0,  # Outside baseline corner
            p4,  # Inside baseline corner
            p8,  # Inside service line corner
            (int(outside_sideline_x), int(outside_net_y))  # Outside net corner
        ], np.int32)
        
        # Draw zone polygon with transparency
        overlay = frame.copy()
        cv2.fillPoly(overlay, [zone_poly], zone_color)
        cv2.addWeighted(overlay, 0.2, frame, 0.8, 0, frame)
        cv2.polylines(frame, [zone_poly], True, (255, 255, 255), 2)
        
        # Draw zone label
        center = np.mean(zone_poly, axis=0).astype(int)
        cv2.putText(frame, 'AA', tuple(center), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    def _draw_far_right_doubles_alley(self, frame, p1, p6, p9):
        """Draw far side right doubles alley (DD) - bounded by keypoints 1, 6, 9"""
        zone_color = (150, 150, 150)  # Darker gray for doubles alley
        
        # The doubles alley should be bounded by:
        # - Outside doubles sideline: keypoint 1 (vertical line)
        # - Inside doubles alley sideline: keypoints 6 to 9 (singles sideline)
        # - Line from 9 to outside doubles sideline (parallel to line from 1 to 6)
        # - Baseline: from p1 to p6
        
        # For a proper parallelogram, we need both sets of opposite sides to be parallel:
        # 1. Line from p1 to p6 should be parallel to line from p9 to sideline
        # 2. Line from p6 to p9 should be parallel to line from p1 to sideline
        
        # Calculate the direction vector from p1 to p6 (baseline direction)
        baseline_dx = p6[0] - p1[0]
        baseline_dy = p6[1] - p1[1]
        
        # Calculate the direction vector from p6 to p9 (service line direction)
        service_dx = p9[0] - p6[0]
        service_dy = p9[1] - p6[1]
        
        # The outside doubles sideline is vertical at p1[0]
        outside_sideline_x = p1[0]
        
        # Find the intersection point that satisfies both parallel constraints
        if baseline_dx != 0 and service_dx != 0:
            # Using constraint 1: line from p9 parallel to p1->p6
            outside_net_y = p9[1] + (baseline_dy / baseline_dx) * (p1[0] - p9[0])
        else:
            # Fallback if lines are vertical
            outside_net_y = p9[1]
        
        # Create the parallelogram with correct geometry
        zone_poly = np.array([
            p1,  # Outside baseline corner
            p6,  # Inside baseline corner
            p9,  # Inside service line corner
            (int(outside_sideline_x), int(outside_net_y))  # Outside net corner
        ], np.int32)
        
        # Draw zone polygon with transparency
        overlay = frame.copy()
        cv2.fillPoly(overlay, [zone_poly], zone_color)
        cv2.addWeighted(overlay, 0.2, frame, 0.8, 0, frame)
        cv2.polylines(frame, [zone_poly], True, (255, 255, 255), 2)
        
        # Draw zone label
        center = np.mean(zone_poly, axis=0).astype(int)
        cv2.putText(frame, 'DD', tuple(center), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    def render_positioning_overlay(self, frame: np.ndarray, frame_number: int) -> np.ndarray:
        """Render player positioning overlay"""
        if 'positioning' not in self.overlay_data:
            return frame
            
        df = self.overlay_data['positioning']
        frame_data = df[df['frame'] == frame_number]
        
        for _, row in frame_data.iterrows():
            player_id = row['player_id']
            position = row['position']
            feet_x = int(row['feet_x'])
            feet_y = int(row['feet_y'])
            
            # Color based on position
            color = self.overlay_colors['positioning']
            if position == 'FRONT':
                color = (0, 255, 0)  # Green
            elif position == 'BACK':
                color = (0, 0, 255)  # Red
            elif position == 'DOUBLES':
                color = (255, 0, 255)  # Magenta
            else:
                color = (128, 128, 128)  # Gray
            
            # Draw player feet position
            cv2.circle(frame, (feet_x, feet_y), 8, color, -1)
            
            # Draw position label
            label = f"P{player_id}: {position}"
            cv2.putText(frame, label, (feet_x+10, feet_y-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        return frame
    
    def render_pose_overlay(self, frame: np.ndarray, frame_number: int) -> np.ndarray:
        """Render pose keypoints overlay"""
        if 'pose' not in self.overlay_data:
            return frame
            
        df = self.overlay_data['pose']
        frame_data = df[df['frame'] == frame_number]
        
        for _, row in frame_data.iterrows():
            player_id = row['player_id']
            keypoints_str = row['keypoints']
            
            if pd.notna(keypoints_str):
                # Parse keypoints string
                keypoints = []
                for kp_str in keypoints_str.split('|'):
                    if ',' in kp_str:
                        x, y = kp_str.split(',')
                        keypoints.append((int(float(x)), int(float(y))))
                
                # Draw keypoints
                for i, (x, y) in enumerate(keypoints):
                    cv2.circle(frame, (x, y), 3, self.overlay_colors['pose'], -1)
                    cv2.putText(frame, str(i), (x+3, y-3), cv2.FONT_HERSHEY_SIMPLEX, 0.3, 
                               self.overlay_colors['pose'], 1)
                
                # Draw skeleton connections (basic)
                if len(keypoints) >= 17:  # COCO format
                    # Head connections
                    if keypoints[0][0] > 0 and keypoints[1][0] > 0:
                        cv2.line(frame, keypoints[0], keypoints[1], self.overlay_colors['pose'], 2)
                    
                    # Shoulder connections
                    if keypoints[5][0] > 0 and keypoints[6][0] > 0:
                        cv2.line(frame, keypoints[5], keypoints[6], self.overlay_colors['pose'], 2)
        
        return frame


class TennisOverlayProcessor:
    """Main processor for tennis analysis overlays"""
    
    def __init__(self):
        """Initialize processor"""
        self.renderer = OverlayRenderer()
        
        # Define available scripts and their CSV files
        # To add a new script, just add it here!
        self.available_scripts = {
            'ball': 'ball_detection.csv',
            'court': 'court_keypoints.csv', 
            'court_zones': 'court_keypoints.csv',  # Uses same CSV as court keypoints
            'positioning': 'player_positioning.csv',
            'pose': 'player_poses.csv'
        }
        
        # Script-specific render functions
        self.render_functions = {
            'ball': self.renderer.render_ball_overlay,
            'court': self.renderer.render_court_overlay,
            'court_zones': self.renderer.render_court_zones_overlay,
            'positioning': self.renderer.render_positioning_overlay,
            'pose': self.renderer.render_pose_overlay
        }
    
    def load_script_data(self, script_name: str) -> bool:
        """Load data for a specific script"""
        if script_name not in self.available_scripts:
            logger.error(f"Unknown script: {script_name}. Available: {list(self.available_scripts.keys())}")
            return False
        
        csv_file = self.available_scripts[script_name]
        return self.renderer.load_overlay_data(script_name, csv_file)
    
    def add_new_script(self, script_name: str, csv_file: str, render_function):
        """Add a new script to the overlay system"""
        self.available_scripts[script_name] = csv_file
        self.render_functions[script_name] = render_function
        logger.info(f"Added new script: {script_name}")
    
    def process_video(self, video_file: str, output_file: str = None, 
                     active_scripts: List[str] = None, show_viewer: bool = True):
        """Process video with overlays from specified scripts"""
        
        # Default to all scripts if none specified
        if active_scripts is None:
            active_scripts = list(self.available_scripts.keys())
        
        # Load data for active scripts
        loaded_scripts = []
        for script_name in active_scripts:
            if self.load_script_data(script_name):
                loaded_scripts.append(script_name)
        
        logger.info(f"Loaded overlays for: {loaded_scripts}")
        
        # Open video
        cap = cv2.VideoCapture(video_file)
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_file}")
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        logger.info(f"Processing video: {width}x{height} @ {fps}fps, {total_frames} frames")
        
        # Setup video writer if output specified
        out = None
        if output_file:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_file, fourcc, fps, (width, height))
        
        # Setup viewer
        if show_viewer:
            cv2.namedWindow('Tennis Analysis Overlays', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('Tennis Analysis Overlays', 1200, 800)
        
        frame_number = 0
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Apply overlays from each active script
                for script_name in loaded_scripts:
                    if script_name in self.render_functions:
                        frame = self.render_functions[script_name](frame, frame_number)
                
                # Add frame info
                cv2.putText(frame, f"Frame: {frame_number}", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                
                # Add script info
                script_info = f"Scripts: {', '.join(loaded_scripts)}"
                cv2.putText(frame, script_info, (10, 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                # Write frame
                if out:
                    out.write(frame)
                
                # Show in viewer
                if show_viewer:
                    cv2.imshow('Tennis Analysis Overlays', frame)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        break
                    elif key == ord(' '):  # Space to pause
                        cv2.waitKey(0)
                
                # Progress update
                if frame_number % 30 == 0:
                    logger.info(f"Processed {frame_number}/{total_frames} frames ({frame_number/total_frames*100:.1f}%)")
                
                frame_number += 1
        
        finally:
            # Cleanup
            cap.release()
            if out:
                out.release()
            if show_viewer:
                cv2.destroyAllWindows()
        
        logger.info("Overlay processing completed!")


def main():
    """Main function for standalone usage"""
    parser = argparse.ArgumentParser(description='Tennis Analysis Overlay System')
    parser.add_argument('--video', required=True, help='Input video file')
    parser.add_argument('--output', default='tennis_overlays.mp4', help='Output video file')
    parser.add_argument('--scripts', nargs='+', 
                       choices=['ball', 'court', 'court_zones', 'positioning', 'pose'],
                       default=['ball', 'court', 'court_zones', 'positioning', 'pose'],
                       help='Scripts to include in overlay (default: all)')
    parser.add_argument('--no-viewer', action='store_true', help='Disable real-time viewer')
    
    args = parser.parse_args()
    
    try:
        processor = TennisOverlayProcessor()
        processor.process_video(
            args.video, 
            args.output, 
            args.scripts, 
            show_viewer=not args.no_viewer
        )
        
    except Exception as e:
        logger.error(f"Error running overlay system: {e}")


if __name__ == "__main__":
    main()
