#!/usr/bin/env python3
"""
Download and prepare TrackNet dataset
Downloads the official TrackNet tennis ball tracking dataset
"""

import os
import requests
import zipfile
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def download_dataset():
    """Download the TrackNet dataset"""
    logger.info("🎾 Downloading TrackNet dataset...")
    
    # Create datasets directory
    datasets_dir = Path("datasets")
    datasets_dir.mkdir(exist_ok=True)
    
    # The dataset is hosted on Google Drive
    # We'll need to download it manually since it's a large file
    dataset_url = "https://drive.google.com/drive/folders/11r0RUaQHX7I3ANkaYG4jOxXK1OYo01Ut"
    
    logger.info("📥 TrackNet dataset download instructions:")
    logger.info("=" * 60)
    logger.info("1. Visit the dataset link:")
    logger.info(f"   {dataset_url}")
    logger.info("")
    logger.info("2. Download the dataset (it's a large file)")
    logger.info("")
    logger.info("3. Extract the downloaded file to:")
    logger.info("   datasets/trackNet_raw/")
    logger.info("")
    logger.info("4. The structure should look like:")
    logger.info("   datasets/trackNet_raw/")
    logger.info("   ├── game1/")
    logger.info("   │   ├── Clip1/")
    logger.info("   │   │   ├── 0000.jpg")
    logger.info("   │   │   ├── 0001.jpg")
    logger.info("   │   │   ├── ...")
    logger.info("   │   │   └── Label.csv")
    logger.info("   │   └── Clip2/")
    logger.info("   │       └── ...")
    logger.info("   ├── game2/")
    logger.info("   └── ...")
    logger.info("")
    logger.info("5. Run the training script:")
    logger.info("   python train_tracknet.py --data_path datasets/trackNet_raw")
    logger.info("=" * 60)
    
    return True

def verify_dataset_structure():
    """Verify that the dataset is properly structured"""
    dataset_path = Path("datasets/trackNet_raw")
    
    if not dataset_path.exists():
        logger.error(f"Dataset not found at {dataset_path}")
        logger.info("Please download and extract the dataset first")
        return False
    
    # Check for expected structure
    expected_games = [f"game{i}" for i in range(1, 11)]
    found_games = []
    
    for game_dir in dataset_path.iterdir():
        if game_dir.is_dir() and game_dir.name.startswith("game"):
            found_games.append(game_dir.name)
            
            # Check for clips in each game
            clips = [d for d in game_dir.iterdir() if d.is_dir()]
            if not clips:
                logger.warning(f"No clips found in {game_dir.name}")
            else:
                logger.info(f"Found {len(clips)} clips in {game_dir.name}")
                
                # Check for Label.csv files
                for clip in clips:
                    label_file = clip / "Label.csv"
                    if not label_file.exists():
                        logger.warning(f"No Label.csv found in {clip}")
    
    logger.info(f"Found {len(found_games)} games: {found_games}")
    
    if len(found_games) >= 8:  # At least 8 games for training
        logger.info("✅ Dataset structure looks good!")
        return True
    else:
        logger.error("❌ Dataset structure incomplete")
        return False

def main():
    """Main function"""
    logger.info("🚀 TrackNet Dataset Setup")
    logger.info("=" * 40)
    
    # Check if dataset already exists
    if Path("datasets/trackNet_raw").exists():
        logger.info("Dataset directory already exists. Verifying structure...")
        if verify_dataset_structure():
            logger.info("✅ Dataset is ready for training!")
            logger.info("")
            logger.info("To start training, run:")
            logger.info("python train_tracknet.py --data_path datasets/trackNet_raw")
            return
        else:
            logger.info("Dataset structure incomplete. Please re-download.")
    
    # Download instructions
    download_dataset()

if __name__ == "__main__":
    main()
