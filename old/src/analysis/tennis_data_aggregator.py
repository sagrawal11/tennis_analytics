#!/usr/bin/env python3
"""
Tennis Data Aggregator
Combines data from multiple tennis analysis scripts into a comprehensive CSV
"""

import cv2
import numpy as np
import pandas as pd
import argparse
import logging
import sys
import os
import time
import subprocess
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import yaml
import json

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TennisDataAggregator:
    """Aggregates data from multiple tennis analysis scripts"""
    
    def __init__(self, config_path: str = "config.yaml"):
        """Initialize the data aggregator"""
        self.config = self._load_config(config_path)
        self.temp_dir = Path("temp_analysis")
        self.temp_dir.mkdir(exist_ok=True)
        
        logger.info("✓ Tennis Data Aggregator initialized")
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            logger.info(f"Configuration loaded from {config_path}")
            return config
        except Exception as e:
            logger.error(f"Error loading config: {e}")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration"""
        return {
            'models': {
                'yolo_player_model': 'models/playersnball5.pt',
                'rfdetr_model': 'rf-detr-base.pth',
                'pose_model': 'models/yolov8n-pose.pt'
            },
            'player_detection': {
                'confidence_threshold': 0.5,
                'iou_threshold': 0.45,
                'max_players': 2
            },
            'pose_estimation': {
                'confidence_threshold': 0.5,
                'min_keypoints': 5
            }
        }
    
    def process_video(self, video_path: str, output_csv: str):
        """Process video and generate comprehensive CSV by running each component"""
        logger.info(f"Processing video: {video_path}")
        logger.info(f"Output CSV: {output_csv}")
        
        # Check if video exists
        if not Path(video_path).exists():
            logger.error(f"Video file not found: {video_path}")
            return False
        
        try:
            # 1. Run tennis_CV.py to get comprehensive analysis data
            logger.info("Step 1: Running comprehensive tennis analysis...")
            cv_csv = self.temp_dir / "tennis_analysis_data.csv"
            success = self._run_tennis_cv(video_path, cv_csv)
            if not success:
                logger.error("Failed to run tennis_CV.py")
                return False
            
            # 2. Run tennis_ball.py to get enhanced ball detection
            logger.info("Step 2: Running enhanced ball detection...")
            ball_csv = self.temp_dir / "tennis_ball_results.csv"
            success = self._run_tennis_ball(video_path, ball_csv)
            if not success:
                logger.warning("Failed to run tennis_ball.py, using CV ball data")
                ball_csv = None
            
            # 3. Run court_segmenter.py to get court segmentation
            logger.info("Step 3: Running court segmentation...")
            court_csv = self.temp_dir / "court_segmentation.csv"
            success = self._run_court_segmenter(video_path, court_csv)
            if not success:
                logger.warning("Failed to run court_segmenter.py")
                court_csv = None
            
            # 4. Run tennis_positioning.py to get player positioning
            logger.info("Step 4: Running player positioning...")
            positioning_csv = self.temp_dir / "player_positioning.csv"
            success = self._run_tennis_positioning(video_path, positioning_csv)
            if not success:
                logger.warning("Failed to run tennis_positioning.py")
                positioning_csv = None
            
            # 5. Combine all CSV data
            logger.info("Step 5: Combining all data...")
            success = self._combine_csvs(cv_csv, ball_csv, court_csv, positioning_csv, output_csv)
            if not success:
                logger.error("Failed to combine CSV data")
                return False
            
            logger.info("✓ Data aggregation completed successfully!")
            return True
            
        except Exception as e:
            logger.error(f"Error during processing: {e}")
            return False
        
        finally:
            # Clean up temp files
            self._cleanup_temp_files()
    
    def _run_tennis_cv(self, video_path: str, output_csv: str) -> bool:
        """Run tennis_CV.py to get comprehensive analysis data"""
        try:
            cmd = [
                "python", "tennis_CV.py",
                "--video", video_path,
                "--output", str(output_csv)
            ]
            
            logger.info(f"Running: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            
            if result.returncode == 0:
                logger.info("✓ tennis_CV.py completed successfully")
                return True
            else:
                logger.error(f"tennis_CV.py failed: {result.stderr}")
                return False
                
        except subprocess.TimeoutExpired:
            logger.error("tennis_CV.py timed out")
            return False
        except Exception as e:
            logger.error(f"Error running tennis_CV.py: {e}")
            return False
    
    def _run_tennis_ball(self, video_path: str, output_csv: str) -> bool:
        """Run tennis_ball.py to get enhanced ball detection"""
        try:
            cmd = [
                "python", "tennis_ball.py",
                "--video", video_path,
                "--results-csv", str(output_csv)
            ]
            
            # Set environment variable for MPS fallback to ensure RF-DETR works properly
            env = os.environ.copy()
            env['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
            
            logger.info(f"Running: {' '.join(cmd)} (with PYTORCH_ENABLE_MPS_FALLBACK=1)")
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300, env=env)
            
            if result.returncode == 0:
                logger.info("✓ tennis_ball.py completed successfully")
                return True
            else:
                logger.error(f"tennis_ball.py failed: {result.stderr}")
                return False
                
        except subprocess.TimeoutExpired:
            logger.error("tennis_ball.py timed out")
            return False
        except Exception as e:
            logger.error(f"Error running tennis_ball.py: {e}")
            return False
    
    def _run_court_segmenter(self, video_path: str, output_csv: str) -> bool:
        """Run court_segmenter.py to get court segmentation"""
        try:
            cmd = [
                "python", "court_segmenter.py",
                "--video", video_path,
                "--csv", str(output_csv)
            ]
            
            logger.info(f"Running: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            
            if result.returncode == 0:
                logger.info("✓ court_segmenter.py completed successfully")
                return True
            else:
                logger.error(f"court_segmenter.py failed: {result.stderr}")
                return False
                
        except subprocess.TimeoutExpired:
            logger.error("court_segmenter.py timed out")
            return False
        except Exception as e:
            logger.error(f"Error running court_segmenter.py: {e}")
            return False
    
    def _run_tennis_positioning(self, video_path: str, output_csv: str) -> bool:
        """Run tennis_positioning.py to get player positioning"""
        try:
            cmd = [
                "python", "tennis_positioning.py",
                "--video", video_path,
                "--csv", str(output_csv)
            ]
            
            logger.info(f"Running: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            
            if result.returncode == 0:
                logger.info("✓ tennis_positioning.py completed successfully")
                return True
            else:
                logger.error(f"tennis_positioning.py failed: {result.stderr}")
                return False
                
        except subprocess.TimeoutExpired:
            logger.error("tennis_positioning.py timed out")
            return False
        except Exception as e:
            logger.error(f"Error running tennis_positioning.py: {e}")
            return False
    
    def _combine_csvs(self, cv_csv: str, ball_csv: Optional[str], 
                     court_csv: Optional[str], positioning_csv: Optional[str], 
                     output_csv: str) -> bool:
        """Combine all CSV data into a comprehensive output"""
        try:
            # Load main CV data
            if not Path(cv_csv).exists():
                logger.error(f"Main CV CSV not found: {cv_csv}")
                return False
            
            df_main = pd.read_csv(cv_csv)
            logger.info(f"Loaded main CV data: {len(df_main)} frames")
            
            # Add enhanced ball data if available
            if ball_csv and Path(ball_csv).exists():
                df_ball = pd.read_csv(ball_csv)
                logger.info(f"Loaded ball data: {len(df_ball)} frames")
                
                # Merge ball data (assuming same frame structure)
                if len(df_ball) == len(df_main):
                    df_main['enhanced_ball_x'] = df_ball.get('ball_x', 0)
                    df_main['enhanced_ball_y'] = df_ball.get('ball_y', 0)
                    df_main['enhanced_ball_confidence'] = df_ball.get('ball_confidence', 0)
                    df_main['enhanced_ball_source'] = df_ball.get('ball_source', 'none')
                    logger.info("✓ Enhanced ball data merged")
                else:
                    logger.warning("Ball data frame count mismatch, skipping merge")
            
            # Add court segmentation data if available
            if court_csv and Path(court_csv).exists():
                df_court = pd.read_csv(court_csv)
                logger.info(f"Loaded court data: {len(df_court)} frames")
                
                # Merge court data
                if len(df_court) == len(df_main):
                    court_columns = [col for col in df_court.columns if col not in df_main.columns]
                    for col in court_columns:
                        df_main[col] = df_court[col]
                    logger.info("✓ Court segmentation data merged")
                else:
                    logger.warning("Court data frame count mismatch, skipping merge")
            
            # Add player positioning data if available
            if positioning_csv and Path(positioning_csv).exists():
                df_positioning = pd.read_csv(positioning_csv)
                logger.info(f"Loaded positioning data: {len(df_positioning)} frames")
                
                # Merge positioning data
                if len(df_positioning) == len(df_main):
                    positioning_columns = [col for col in df_positioning.columns if col not in df_main.columns]
                    for col in positioning_columns:
                        df_main[col] = df_positioning[col]
                    logger.info("✓ Player positioning data merged")
                else:
                    logger.warning("Positioning data frame count mismatch, skipping merge")
            
            # Save combined CSV
            df_main.to_csv(output_csv, index=False)
            logger.info(f"✓ Combined CSV saved to {output_csv}")
            
            # Show summary statistics
            logger.info(f"Combined CSV Summary:")
            logger.info(f"  - Total frames: {len(df_main)}")
            logger.info(f"  - Columns: {len(df_main.columns)}")
            logger.info(f"  - Average players per frame: {df_main.get('player_count', pd.Series([0])).mean():.2f}")
            logger.info(f"  - Ball detection rate: {(df_main.get('ball_confidence', pd.Series([0])) > 0).mean() * 100:.1f}%")
            
            return True
            
        except Exception as e:
            logger.error(f"Error combining CSVs: {e}")
            return False
    
    def _cleanup_temp_files(self):
        """Clean up temporary files"""
        try:
            if self.temp_dir.exists():
                import shutil
                shutil.rmtree(self.temp_dir)
                logger.info("✓ Temporary files cleaned up")
        except Exception as e:
            logger.warning(f"Error cleaning up temp files: {e}")

def main():
    parser = argparse.ArgumentParser(description='Tennis Data Aggregator - Generate comprehensive CSV')
    parser.add_argument('--video', required=True, help='Input video file')
    parser.add_argument('--output', required=True, help='Output CSV file')
    parser.add_argument('--config', default='config.yaml', help='Configuration file')
    
    args = parser.parse_args()
    
    # Check if video exists
    if not Path(args.video).exists():
        logger.error(f"Video file not found: {args.video}")
        return 1
    
    # Initialize aggregator
    aggregator = TennisDataAggregator(args.config)
    
    # Process video
    success = aggregator.process_video(args.video, args.output)
    
    if success:
        logger.info("✓ Data aggregation completed successfully!")
        return 0
    else:
        logger.error("✗ Data aggregation failed")
        return 1

if __name__ == "__main__":
    sys.exit(main())
