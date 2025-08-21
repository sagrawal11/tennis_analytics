#!/usr/bin/env python3
"""
Debug script to analyze far player movement data and understand why
it's being classified as ready stance instead of moving.
"""

import pandas as pd
import numpy as np
import ast
from typing import List, Tuple, Dict
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

class FarPlayerMovementDebugger:
    def __init__(self):
        self.movement_threshold = 25
        self.movement_window = 15
        self.player_position_history = {}
    
    def _parse_player_bboxes_from_csv(self, bbox_str: str) -> List[List[int]]:
        """Parse player bounding boxes from CSV string format"""
        try:
            if pd.isna(bbox_str) or bbox_str == '':
                return []
            
            # Format: "x1,y1,x2,y2;x1,y1,x2,y2;..."
            bboxes = []
            for bbox_str_part in bbox_str.split(';'):
                if bbox_str_part.strip():
                    coords = [int(x.strip()) for x in bbox_str_part.split(',')]
                    if len(coords) == 4:
                        bboxes.append(coords)
            return bboxes
        except Exception as e:
            logger.error(f"Error parsing bbox: {e}")
            return []
    
    def analyze_far_player_movement(self, csv_file: str):
        """Analyze far player movement data from CSV"""
        try:
            # Read CSV data
            df = pd.read_csv(csv_file)
            logger.info(f"Loaded {len(df)} frames from CSV")
            
            # Track far player (player_id=1) movement
            far_player_movements = []
            
            for frame_idx, row in df.iterrows():
                frame_count = frame_idx + 1
                
                # Parse player bounding boxes
                bboxes = self._parse_player_bboxes_from_csv(row.get('player_bboxes', ''))
                
                if len(bboxes) < 2:
                    continue
                
                # Get far player bbox (player_id=1)
                far_player_bbox = bboxes[1] if len(bboxes) > 1 else None
                
                if far_player_bbox:
                    # Calculate center
                    x1, y1, x2, y2 = far_player_bbox
                    center_x = (x1 + x2) / 2
                    center_y = (y1 + y2) / 2
                    
                    # Track position history
                    if 1 not in self.player_position_history:
                        self.player_position_history[1] = []
                    
                    self.player_position_history[1].append((center_x, center_y))
                    
                    # Keep only last movement_window frames
                    if len(self.player_position_history[1]) > self.movement_window:
                        self.player_position_history[1] = self.player_position_history[1][-self.movement_window:]
                    
                    # Calculate movement if we have enough frames
                    if len(self.player_position_history[1]) >= self.movement_window:
                        positions = self.player_position_history[1]
                        first_x, first_y = positions[0]
                        last_x, last_y = positions[-1]
                        window_movement = np.sqrt((last_x - first_x)**2 + (last_y - first_y)**2)
                        
                        # Use same threshold as classifier
                        adjusted_threshold = self.movement_threshold * 0.1  # 2.5 pixels
                        is_moving = window_movement > adjusted_threshold
                        
                        far_player_movements.append({
                            'frame': frame_count,
                            'center_x': center_x,
                            'center_y': center_y,
                            'window_movement': window_movement,
                            'threshold': adjusted_threshold,
                            'is_moving': is_moving,
                            'bbox': far_player_bbox
                        })
                        
                        # Log significant movements
                        if window_movement > 5:  # Log movements > 5 pixels
                            logger.info(f"Frame {frame_count}: Far player movement = {window_movement:.1f} pixels, threshold = {adjusted_threshold:.1f}, moving = {is_moving}")
            
            # Analyze results
            if far_player_movements:
                movements = [m['window_movement'] for m in far_player_movements]
                moving_frames = [m for m in far_player_movements if m['is_moving']]
                ready_frames = [m for m in far_player_movements if not m['is_moving']]
                
                logger.info(f"\n=== FAR PLAYER MOVEMENT ANALYSIS ===")
                logger.info(f"Total frames analyzed: {len(far_player_movements)}")
                logger.info(f"Frames classified as moving: {len(moving_frames)} ({len(moving_frames)/len(far_player_movements)*100:.1f}%)")
                logger.info(f"Frames classified as ready stance: {len(ready_frames)} ({len(ready_frames)/len(far_player_movements)*100:.1f}%)")
                logger.info(f"Average movement: {np.mean(movements):.2f} pixels")
                logger.info(f"Max movement: {np.max(movements):.2f} pixels")
                logger.info(f"Min movement: {np.min(movements):.2f} pixels")
                logger.info(f"Movement threshold: {self.movement_threshold * 0.1:.1f} pixels")
                
                # Show frames with high movement but classified as ready stance
                high_movement_ready = [m for m in ready_frames if m['window_movement'] > 10]
                if high_movement_ready:
                    logger.info(f"\n⚠️  Frames with high movement (>10px) but classified as ready stance:")
                    for m in high_movement_ready[:10]:  # Show first 10
                        logger.info(f"  Frame {m['frame']}: {m['window_movement']:.1f}px movement")
                
                # Show frames with low movement but classified as moving
                low_movement_moving = [m for m in moving_frames if m['window_movement'] < 2]
                if low_movement_moving:
                    logger.info(f"\n⚠️  Frames with low movement (<2px) but classified as moving:")
                    for m in low_movement_moving[:10]:  # Show first 10
                        logger.info(f"  Frame {m['frame']}: {m['window_movement']:.1f}px movement")
            
        except Exception as e:
            logger.error(f"Error analyzing far player movement: {e}")
            import traceback
            traceback.print_exc()

def main():
    debugger = FarPlayerMovementDebugger()
    debugger.analyze_far_player_movement('tennis_analysis_data.csv')

if __name__ == "__main__":
    main()
