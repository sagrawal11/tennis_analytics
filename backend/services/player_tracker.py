"""
Player Tracking Service.

This module handles player tracking based on identification clicks.
Uses color recognition to track the identified player throughout the video.
"""

import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional


def extract_color_from_coords(frame: np.ndarray, coords: Dict[str, float]) -> np.ndarray:
    """
    Extract color information from coordinates in a frame.
    
    Args:
        frame: Video frame (BGR format)
        coords: Coordinates dict with 'x' and 'y' as percentages (0-100)
        
    Returns:
        Color histogram or feature vector representing the player
    """
    h, w = frame.shape[:2]
    x = int((coords["x"] / 100) * w)
    y = int((coords["y"] / 100) * h)
    
    # Extract a small region around the click point
    region_size = 20
    x1 = max(0, x - region_size)
    y1 = max(0, y - region_size)
    x2 = min(w, x + region_size)
    y2 = min(h, y + region_size)
    
    region = frame[y1:y2, x1:x2]
    
    if region.size == 0:
        return np.array([])
    
    # Calculate color histogram
    hist_b = cv2.calcHist([region], [0], None, [32], [0, 256])
    hist_g = cv2.calcHist([region], [1], None, [32], [0, 256])
    hist_r = cv2.calcHist([region], [2], None, [32], [0, 256])
    
    # Normalize
    hist = np.concatenate([hist_b, hist_g, hist_r])
    hist = hist / (hist.sum() + 1e-7)
    
    return hist


def track_player_in_video(
    video_path: str,
    identification_coords: List[Dict[str, float]],
    identification_frames: List[np.ndarray]
) -> List[Dict]:
    """
    Track player throughout video using color recognition.
    
    Args:
        video_path: Path to video file
        identification_coords: List of coordinate dicts from identification clicks
        identification_frames: List of frames where player was identified
        
    Returns:
        List of player positions for each frame
    """
    # Build color model from identification frames
    color_models = []
    for frame, coords in zip(identification_frames, identification_coords):
        color_model = extract_color_from_coords(frame, coords)
        if color_model.size > 0:
            color_models.append(color_model)
    
    if not color_models:
        return []
    
    # Average color model
    avg_color_model = np.mean(color_models, axis=0)
    
    # Track through video
    cap = cv2.VideoCapture(video_path)
    positions = []
    frame_num = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Find best matching region using color similarity
        best_match = find_best_match(frame, avg_color_model)
        if best_match:
            positions.append({
                "frame": frame_num,
                "position": best_match,
            })
        
        frame_num += 1
    
    cap.release()
    return positions


def find_best_match(frame: np.ndarray, color_model: np.ndarray) -> Optional[Dict[str, float]]:
    """
    Find best matching region in frame using color model.
    
    Args:
        frame: Video frame
        color_model: Color histogram model
        
    Returns:
        Position dict with 'x' and 'y' percentages, or None
    """
    h, w = frame.shape[:2]
    
    # Search in a grid pattern
    step = 50
    best_score = 0
    best_pos = None
    
    for y in range(0, h, step):
        for x in range(0, w, step):
            # Extract region
            region_size = 40
            x1 = max(0, x - region_size)
            y1 = max(0, y - region_size)
            x2 = min(w, x + region_size)
            y2 = min(h, y + region_size)
            
            region = frame[y1:y2, x1:x2]
            if region.size == 0:
                continue
            
            # Calculate color histogram
            hist_b = cv2.calcHist([region], [0], None, [32], [0, 256])
            hist_g = cv2.calcHist([region], [1], None, [32], [0, 256])
            hist_r = cv2.calcHist([region], [2], None, [32], [0, 256])
            hist = np.concatenate([hist_b, hist_g, hist_r])
            hist = hist / (hist.sum() + 1e-7)
            
            # Compare with model (using correlation)
            score = cv2.compareHist(color_model.reshape(-1, 1).astype(np.float32),
                                   hist.reshape(-1, 1).astype(np.float32),
                                   cv2.HISTCMP_CORREL)
            
            if score > best_score:
                best_score = score
                best_pos = {
                    "x": (x / w) * 100,
                    "y": (y / h) * 100,
                }
    
    # Only return if score is above threshold
    if best_score > 0.5:
        return best_pos
    
    return None
