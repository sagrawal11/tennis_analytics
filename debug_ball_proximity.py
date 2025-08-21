#!/usr/bin/env python3
"""
Debug ball proximity and classification logic
"""

import pandas as pd
import numpy as np
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def debug_ball_proximity():
    """Debug ball proximity and classification"""
    
    # Load CSV data
    df = pd.read_csv('tennis_analysis_data.csv')
    logger.info(f"Loaded {len(df)} frames of data")
    
    # Check frames with different proximity thresholds
    proximity_thresholds = [100, 150, 200, 250, 300]
    
    for threshold in proximity_thresholds:
        logger.info(f"\n{'='*50}")
        logger.info(f"Checking proximity threshold: {threshold} pixels")
        logger.info(f"{'='*50}")
        
        near_count = 0
        total_frames = 0
        
        for frame_idx in range(min(20, len(df))):  # Check first 20 frames
            row = df.iloc[frame_idx]
            
            # Parse ball position
            ball_x = row.get('ball_x', '')
            ball_y = row.get('ball_y', '')
            
            if pd.isna(ball_x) or pd.isna(ball_y) or ball_x == '' or ball_y == '':
                continue
            
            ball_position = [int(float(ball_x)), int(float(ball_y))]
            
            # Parse player bboxes
            bboxes_str = row.get('player_bboxes', '')
            if pd.isna(bboxes_str) or bboxes_str == '':
                continue
            
            player_bboxes = []
            boxes = bboxes_str.split(';')
            for box in boxes:
                if box.strip():
                    try:
                        x1, y1, x2, y2 = map(int, box.split(','))
                        player_bboxes.append([x1, y1, x2, y2])
                    except (ValueError, IndexError):
                        continue
            
            total_frames += 1
            
            # Check ball proximity for each player
            frame_near = False
            for i, bbox in enumerate(player_bboxes):
                x1, y1, x2, y2 = bbox
                player_center_x = (x1 + x2) / 2
                player_center_y = (y1 + y2) / 2
                
                distance = np.sqrt((ball_position[0] - player_center_x)**2 + (ball_position[1] - player_center_y)**2)
                ball_near = distance < threshold
                
                if ball_near:
                    frame_near = True
                    logger.info(f"Frame {frame_idx}: Player {i+1} near ball (distance: {distance:.1f})")
            
            if frame_near:
                near_count += 1
        
        logger.info(f"Frames with ball near players: {near_count}/{total_frames} ({near_count/total_frames*100:.1f}%)")

if __name__ == "__main__":
    debug_ball_proximity()
