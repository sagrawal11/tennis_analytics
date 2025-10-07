#!/usr/bin/env python3
"""
Improved Annotation Strategy for Tennis Ball Bounces

The current issue: Single-frame annotations don't capture bounce dynamics.
Real bounces last 3-7 frames. We need multi-frame annotations.
"""

import pandas as pd
import numpy as np
import argparse
import logging
from typing import List, Tuple

logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')
logger = logging.getLogger(__name__)


def analyze_current_annotations(annotations_file: str):
    """Analyze current annotation patterns to identify issues"""
    
    df = pd.read_csv(annotations_file)
    logger.info(f"Analyzing {len(df)} annotations...")
    
    # Check bounce patterns
    bounce_frames = df[df['is_bounce'] == 1]['frame_number'].values
    non_bounce_frames = df[df['is_bounce'] == 0]['frame_number'].values
    
    if len(bounce_frames) > 1:
        gaps = np.diff(np.sort(bounce_frames))
        
        logger.info("=== CURRENT ANNOTATION ANALYSIS ===")
        logger.info(f"Total bounce annotations: {len(bounce_frames)}")
        logger.info(f"Frame gaps between bounces:")
        logger.info(f"  Mean: {np.mean(gaps):.1f} frames")
        logger.info(f"  Min: {np.min(gaps)} frames")
        logger.info(f"  Max: {np.max(gaps)} frames")
        
        # Check for consecutive bounces
        consecutive_count = np.sum(gaps == 1)
        logger.info(f"Consecutive bounce frames: {consecutive_count}")
        
        if consecutive_count == 0:
            logger.warning("🚨 PROBLEM: No consecutive bounce frames!")
            logger.warning("Real tennis bounces should last 3-7 frames")
            logger.warning("Single-frame annotations miss bounce dynamics")
    
    return df


def suggest_improved_annotations(current_df: pd.DataFrame, 
                                ball_data_file: str,
                                output_file: str):
    """Suggest improved annotations based on ball trajectory analysis"""
    
    logger.info("=== IMPROVED ANNOTATION STRATEGY ===")
    
    # Load ball data to analyze trajectories
    try:
        ball_df = pd.read_csv(ball_data_file)
        if 'ball_x' in ball_df.columns:
            x_col, y_col = 'ball_x', 'ball_y'
        else:
            x_col, y_col = 'x', 'y'
    except Exception as e:
        logger.error(f"Could not load ball data: {e}")
        return
    
    improved_annotations = []
    
    for video_name, video_df in current_df.groupby('video_name'):
        logger.info(f"Processing {video_name}...")
        
        # Get ball data for this video
        video_ball_data = ball_df[ball_df.get('video_name', '') == video_name] if 'video_name' in ball_df.columns else ball_df
        video_ball_data = video_ball_data.sort_values('frame')
        
        # Find potential bounce events based on Y-velocity reversals
        potential_bounces = find_potential_bounces(video_ball_data, x_col, y_col)
        
        # Create improved annotations
        for frame in video_df['frame_number']:
            is_bounce = video_df[video_df['frame_number'] == frame]['is_bounce'].iloc[0]
            
            if is_bounce == 1:
                # This is a current bounce annotation - expand it
                expanded_frames = expand_bounce_annotation(frame, potential_bounces, video_ball_data, x_col, y_col)
                
                for expanded_frame in expanded_frames:
                    improved_annotations.append({
                        'video_name': video_name,
                        'frame_number': expanded_frame,
                        'is_bounce': 1,
                        'confidence': 'high' if expanded_frame == frame else 'medium'
                    })
            else:
                # Non-bounce frame
                improved_annotations.append({
                    'video_name': video_name,
                    'frame_number': frame,
                    'is_bounce': 0,
                    'confidence': 'high'
                })
    
    # Save improved annotations
    improved_df = pd.DataFrame(improved_annotations)
    improved_df.to_csv(output_file, index=False)
    
    logger.info(f"Saved improved annotations to {output_file}")
    logger.info(f"Original annotations: {len(current_df)}")
    logger.info(f"Improved annotations: {len(improved_df)}")
    
    return improved_df


def find_potential_bounces(ball_df: pd.DataFrame, x_col: str, y_col: str) -> List[int]:
    """Find potential bounce frames based on Y-velocity analysis"""
    
    potential_bounces = []
    
    if len(ball_df) < 5:
        return potential_bounces
    
    # Calculate Y-velocities
    y_coords = ball_df[y_col].values
    y_velocities = np.diff(y_coords)
    
    # Find velocity reversals (downward to upward)
    for i in range(1, len(y_velocities)):
        if y_velocities[i-1] < -5 and y_velocities[i] > 5:  # Significant reversal
            frame_idx = i
            if frame_idx < len(ball_df):
                potential_bounces.append(ball_df.iloc[frame_idx]['frame'])
    
    return potential_bounces


def expand_bounce_annotation(center_frame: int, 
                           potential_bounces: List[int],
                           ball_df: pd.DataFrame,
                           x_col: str, y_col: str) -> List[int]:
    """Expand a single bounce annotation to multiple frames"""
    
    expanded_frames = [center_frame]
    
    # Look for consecutive frames with similar Y-coordinates (ball on ground)
    center_idx = ball_df[ball_df['frame'] == center_frame].index
    
    if len(center_idx) == 0:
        return expanded_frames
    
    center_idx = center_idx[0]
    center_y = ball_df.iloc[center_idx][y_col]
    
    # Look backward and forward for similar Y-coordinates
    tolerance = 20  # pixels
    
    # Backward
    for i in range(center_idx - 1, max(0, center_idx - 5), -1):
        if abs(ball_df.iloc[i][y_col] - center_y) < tolerance:
            expanded_frames.insert(0, ball_df.iloc[i]['frame'])
        else:
            break
    
    # Forward  
    for i in range(center_idx + 1, min(len(ball_df), center_idx + 5)):
        if abs(ball_df.iloc[i][y_col] - center_y) < tolerance:
            expanded_frames.append(ball_df.iloc[i]['frame'])
        else:
            break
    
    return expanded_frames


def main():
    parser = argparse.ArgumentParser(description='Analyze and improve bounce annotations')
    parser.add_argument('--annotations', default='all_bounce_annotations.csv', 
                       help='Current annotations file')
    parser.add_argument('--ball-data', default='all_ball_coordinates.csv',
                       help='Ball tracking data file')
    parser.add_argument('--output', default='improved_bounce_annotations.csv',
                       help='Output file for improved annotations')
    
    args = parser.parse_args()
    
    try:
        # Analyze current annotations
        current_df = analyze_current_annotations(args.annotations)
        
        # Create improved annotations
        improved_df = suggest_improved_annotations(current_df, args.ball_data, args.output)
        
        logger.info("✅ Annotation improvement completed!")
        logger.info("📝 Next steps:")
        logger.info("  1. Review the improved annotations")
        logger.info("  2. Manually verify/correct multi-frame bounces")
        logger.info("  3. Retrain models with improved data")
        
    except Exception as e:
        logger.error(f"Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
