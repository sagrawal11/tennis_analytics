#!/usr/bin/env python3
"""
Combine all analysis CSV files into a comprehensive dataset
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')
logger = logging.getLogger(__name__)

def load_ball_detection_data(csv_file: str) -> pd.DataFrame:
    """Load ball detection data"""
    try:
        df = pd.read_csv(csv_file)
        logger.info(f"Loaded ball detection data: {len(df)} records")
        return df
    except Exception as e:
        logger.error(f"Error loading ball detection data: {e}")
        return pd.DataFrame()

def load_court_keypoints_data(csv_file: str) -> pd.DataFrame:
    """Load court keypoints data"""
    try:
        df = pd.read_csv(csv_file)
        # Get only the averaged keypoints (last row)
        if len(df) > 0 and df.iloc[-1]['frame'] == 'AVERAGE':
            avg_row = df.iloc[-1].copy()
            avg_row['frame'] = 'AVERAGE'
            df_avg = pd.DataFrame([avg_row])
            logger.info(f"Loaded court keypoints data: {len(df)} frames, using averaged keypoints")
            return df_avg
        else:
            logger.warning("No averaged keypoints found in court data")
            return pd.DataFrame()
    except Exception as e:
        logger.error(f"Error loading court keypoints data: {e}")
        return pd.DataFrame()

def load_player_positioning_data(csv_file: str) -> pd.DataFrame:
    """Load player positioning data"""
    try:
        df = pd.read_csv(csv_file)
        logger.info(f"Loaded player positioning data: {len(df)} records")
        return df
    except Exception as e:
        logger.error(f"Error loading player positioning data: {e}")
        return pd.DataFrame()

def combine_analysis_data(ball_csv: str, court_csv: str, positioning_csv: str, output_csv: str):
    """Combine all analysis data into a comprehensive dataset"""
    
    # Load all data
    ball_data = load_ball_detection_data(ball_csv)
    court_data = load_court_keypoints_data(court_csv)
    positioning_data = load_player_positioning_data(positioning_csv)
    
    if ball_data.empty:
        logger.error("No ball detection data available")
        return
    
    # Start with ball detection data as the base
    combined_data = ball_data.copy()
    
    # Add court keypoints data (static for all frames)
    if not court_data.empty:
        # Extract keypoint columns
        keypoint_cols = [col for col in court_data.columns if col.startswith('keypoint_')]
        for col in keypoint_cols:
            combined_data[col] = court_data.iloc[0][col]
        logger.info(f"Added {len(keypoint_cols)} court keypoint columns")
    
    # Add player positioning data
    if not positioning_data.empty:
        # Pivot positioning data to have separate columns for each player
        positioning_pivot = positioning_data.pivot_table(
            index='frame', 
            columns='player_id', 
            values=['position', 'confidence', 'feet_x', 'feet_y', 'zone'],
            aggfunc='first'
        ).reset_index()
        
        # Flatten column names
        positioning_pivot.columns = [f"{col[0]}_player_{col[1]}" if col[1] != '' else col[0] 
                                   for col in positioning_pivot.columns]
        
        # Merge with combined data
        combined_data = combined_data.merge(
            positioning_pivot, 
            left_on='frame', 
            right_on='frame', 
            how='left'
        )
        logger.info("Added player positioning data")
    
    # Add some derived features
    combined_data['has_ball_detection'] = combined_data['x'].notna()
    combined_data['ball_detection_confidence'] = combined_data['confidence'].fillna(0.0)
    
    # Save combined data
    combined_data.to_csv(output_csv, index=False)
    logger.info(f"Combined analysis data saved to {output_csv}")
    logger.info(f"Total records: {len(combined_data)}")
    logger.info(f"Total columns: {len(combined_data.columns)}")
    
    # Print summary
    print("\n=== COMBINED DATA SUMMARY ===")
    print(f"Total frames: {len(combined_data)}")
    print(f"Frames with ball detection: {combined_data['has_ball_detection'].sum()}")
    print(f"Ball detection rate: {combined_data['has_ball_detection'].mean()*100:.1f}%")
    
    if not positioning_data.empty:
        print(f"Frames with player 0 positioning: {combined_data['position_player_0'].notna().sum()}")
        print(f"Frames with player 1 positioning: {combined_data['position_player_1'].notna().sum()}")
    
    print(f"Columns: {list(combined_data.columns)}")

def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Combine tennis analysis CSV files')
    parser.add_argument('--ball-csv', default='ball_detection.csv', help='Ball detection CSV file')
    parser.add_argument('--court-csv', default='court_keypoints.csv', help='Court keypoints CSV file')
    parser.add_argument('--positioning-csv', default='player_positioning.csv', help='Player positioning CSV file')
    parser.add_argument('--output', default='comprehensive_analysis_data.csv', help='Output combined CSV file')
    
    args = parser.parse_args()
    
    combine_analysis_data(
        args.ball_csv, 
        args.court_csv, 
        args.positioning_csv, 
        args.output
    )

if __name__ == "__main__":
    main()
