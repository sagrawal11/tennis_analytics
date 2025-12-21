#!/bin/bash
# Example usage script for hero video generation

# Activate virtual environment
source ../tennis_env/bin/activate

# Example 1: Basic usage with default settings
python process_hero_video.py \
    --input ../path/to/your/video.mp4 \
    --output hero_video_basic.mp4 \
    --ball-prompt "tennis ball"

# Example 2: Fast processing with keypoints-only mode
python process_hero_video.py \
    --input ../path/to/your/video.mp4 \
    --output hero_video_fast.mp4 \
    --ball-prompt "tennis ball" \
    --keypoints-only \
    --frame-skip 2

# Example 3: Custom colors matching CourtVision theme
python process_hero_video.py \
    --input ../path/to/your/video.mp4 \
    --output hero_video_custom.mp4 \
    --ball-prompt "tennis ball" \
    --player-color "#50C878" \
    --ball-color "#FFD700" \
    --trail-length 45 \
    --fps 30

# Example 4: High quality (slower, processes all frames)
python process_hero_video.py \
    --input ../path/to/your/video.mp4 \
    --output hero_video_hq.mp4 \
    --ball-prompt "tennis ball" \
    --frame-skip 1 \
    --fps 30
