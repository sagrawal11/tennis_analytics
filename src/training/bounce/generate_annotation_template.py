#!/usr/bin/env python3
"""
Generate Frame-by-Frame Annotation Template

This script helps you create frame-by-frame annotation templates for your videos.
It generates a CSV with all frames set to is_bounce=0, and you can then edit
the specific frames where bounces occur.

Usage:
    python generate_annotation_template.py --video tennis_test1.mp4 --output annotations_template.csv
"""

import cv2
import pandas as pd
import argparse
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')
logger = logging.getLogger(__name__)


def get_video_frame_count(video_path: str) -> int:
    """Get total frame count from video"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return frame_count


def generate_annotation_template(video_path: str, output_path: str):
    """Generate frame-by-frame annotation template"""
    try:
        # Get video info
        video_name = Path(video_path).name
        frame_count = get_video_frame_count(video_path)
        
        logger.info(f"Video: {video_name}")
        logger.info(f"Total frames: {frame_count}")
        
        # Create template data
        template_data = []
        for frame_num in range(1, frame_count + 1):
            template_data.append({
                'video_name': video_name,
                'frame_number': frame_num,
                'is_bounce': 0  # Default to no bounce, you'll edit the bounce frames
            })
        
        # Create DataFrame and save
        df = pd.DataFrame(template_data)
        df.to_csv(output_path, index=False)
        
        logger.info(f"Generated annotation template: {output_path}")
        logger.info(f"Template contains {len(template_data)} frames")
        logger.info(f"All frames set to is_bounce=0")
        logger.info(f"Edit the CSV to set is_bounce=1 for frames where bounces occur")
        
    except Exception as e:
        logger.error(f"Error generating template: {e}")
        raise


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Generate Frame-by-Frame Annotation Template')
    parser.add_argument('--video', required=True, help='Input video file')
    parser.add_argument('--output', required=True, help='Output CSV file for annotations')
    
    args = parser.parse_args()
    
    try:
        generate_annotation_template(args.video, args.output)
        logger.info("Template generation completed successfully!")
        
    except Exception as e:
        logger.error(f"Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
