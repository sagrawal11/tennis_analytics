#!/usr/bin/env python3
"""
Image Resizing Script

This script resizes all images in the TrackNet dataset to 360x640 resolution
and saves them to the images_custom folder for faster training.
"""

import os
import cv2
import numpy as np
from pathlib import Path
import argparse
from tqdm import tqdm


def resize_images(input_dir: str, output_dir: str, target_width: int = 640, target_height: int = 360):
    """
    Resize all images in input_dir to target resolution and save to output_dir
    
    Args:
        input_dir: Directory containing original images
        output_dir: Directory to save resized images
        target_width: Target width (640)
        target_height: Target height (360)
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    # Create output directory if it doesn't exist
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Find all image files
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
    image_files = []
    
    for root, dirs, files in os.walk(input_path):
        for file in files:
            if Path(file).suffix.lower() in image_extensions:
                image_files.append(Path(root) / file)
    
    print(f"Found {len(image_files)} images to resize")
    print(f"Target resolution: {target_width}x{target_height}")
    print(f"Output directory: {output_path}")
    
    # Process images with progress bar
    for img_path in tqdm(image_files, desc="Resizing images"):
        try:
            # Read image
            img = cv2.imread(str(img_path))
            if img is None:
                print(f"Warning: Could not read {img_path}")
                continue
            
            # Get relative path from input directory
            rel_path = img_path.relative_to(input_path)
            
            # Create output subdirectory structure
            output_subdir = output_path / rel_path.parent
            output_subdir.mkdir(parents=True, exist_ok=True)
            
            # Resize image
            resized_img = cv2.resize(img, (target_width, target_height), interpolation=cv2.INTER_LINEAR)
            
            # Save resized image
            output_file = output_subdir / rel_path.name
            cv2.imwrite(str(output_file), resized_img)
            
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
    
    print(f"✅ Resized {len(image_files)} images to {target_width}x{target_height}")
    print(f"Images saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Resize TrackNet images to 360x640')
    parser.add_argument('--input-dir', type=str, default='../datasets/trackNet/images',
                       help='Input directory with original images')
    parser.add_argument('--output-dir', type=str, default='../datasets/trackNet/images_custom',
                       help='Output directory for resized images')
    parser.add_argument('--width', type=int, default=640, help='Target width')
    parser.add_argument('--height', type=int, default=360, help='Target height')
    
    args = parser.parse_args()
    
    print("🖼️  Image Resizing Tool")
    print("=" * 40)
    print(f"Input directory: {args.input_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Target resolution: {args.width}x{args.height}")
    print("=" * 40)
    
    # Validate input directory
    if not os.path.exists(args.input_dir):
        print(f"❌ Input directory not found: {args.input_dir}")
        return
    
    # Resize images
    resize_images(args.input_dir, args.output_dir, args.width, args.height)


if __name__ == "__main__":
    main()
