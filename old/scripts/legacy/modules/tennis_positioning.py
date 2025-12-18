#!/usr/bin/env python3
"""
Tennis Player Positioning System
Determines if players are at the front (service box) or back (baseline) of the court
"""

import cv2
import numpy as np
import pandas as pd
import argparse
import logging
from typing import List, Tuple, Optional, Dict
import os
import sys

# Add TennisCourtDetector to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'TennisCourtDetector'))

# Import YOLO for player detection
try:
    from ultralytics import YOLO
except ImportError:
    print("Warning: ultralytics not found. Player detection will be disabled.")
    YOLO = None

# Import YAML for config loading
try:
    import yaml
except ImportError:
    print("Warning: PyYAML not found. Using default configuration.")
    yaml = None

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')
logger = logging.getLogger(__name__)


class PlayerPositioning:
    """Tennis player positioning system for front/back court detection"""
    
    def __init__(self):
        """Initialize player positioning system"""
        self.court_segmenter = None
        self.front_zones = ['WIDE', 'BODY', 'TEE']  # Service box zones
        self.back_zones = ['A', 'B', 'C', 'D']      # Baseline zones
        self.doubles_zones = ['AA', 'DD']           # Doubles lane zones
        
        logger.info("Player positioning system initialized")
        logger.info(f"Front court zones: {self.front_zones}")
        logger.info(f"Back court zones: {self.back_zones}")
        logger.info(f"Doubles zones: {self.doubles_zones}")
    
    def set_court_segmenter(self, court_segmenter):
        """Set the court segmenter for zone detection"""
        self.court_segmenter = court_segmenter
        logger.info("Court segmenter set for positioning")
    
    def get_player_position(self, player_x: float, player_y: float) -> str:
        """
        Determine if a player is at the front or back of the court
        
        Args:
            player_x: Player's x-coordinate
            player_y: Player's y-coordinate
            
        Returns:
            'FRONT', 'BACK', 'DOUBLES', or 'OUTSIDE_COURT'
        """
        if not self.court_segmenter:
            return 'OUTSIDE_COURT'
        
        # Get the zone for this player position
        zone = self.court_segmenter.get_zone_for_point(player_x, player_y)
        
        if not zone:
            return 'OUTSIDE_COURT'
        
        # Extract zone name (remove region prefix)
        zone_name = zone.split('_')[-1] if '_' in zone else zone
        
        # Determine position based on zone
        if zone_name in self.front_zones:
            return 'FRONT'
        elif zone_name in self.back_zones:
            return 'BACK'
        elif zone_name in self.doubles_zones:
            return 'DOUBLES'
        else:
            return 'OUTSIDE_COURT'
    
    def get_position_confidence(self, player_x: float, player_y: float) -> float:
        """
        Get confidence score for position detection based on distance to zone boundaries
        
        Args:
            player_x: Player's x-coordinate
            player_y: Player's y-coordinate
            
        Returns:
            Confidence score between 0.0 and 1.0
        """
        if not self.court_segmenter or not self.court_segmenter.zone_definitions:
            return 0.0
        
        # Find the closest zone and calculate confidence based on distance to center
        min_distance = float('inf')
        closest_zone = None
        
        for zone_name, zone_data in self.court_segmenter.zone_definitions.items():
            if 'points' not in zone_data:
                continue
                
            # Calculate distance to zone center
            zone_center = np.mean(zone_data['points'], axis=0)
            distance = np.sqrt((player_x - zone_center[0])**2 + (player_y - zone_center[1])**2)
            
            if distance < min_distance:
                min_distance = distance
                closest_zone = zone_name
        
        if closest_zone is None:
            return 0.0
        
        # Convert distance to confidence (closer = higher confidence)
        # Assuming typical zone size of ~100 pixels, normalize distance
        max_expected_distance = 100.0
        confidence = max(0.0, 1.0 - (min_distance / max_expected_distance))
        
        return min(1.0, confidence)


class TennisPositioningProcessor:
    """Processes video directly and creates positioning analysis"""
    
    def __init__(self, court_segmenter):
        """Initialize processor with court segmenter"""
        self.positioning = PlayerPositioning()
        self.positioning.set_court_segmenter(court_segmenter)
        self.position_history = []  # Store positioning data
        
        # Load configuration
        self.config = self._load_config()
        
        # Initialize YOLO model for player detection
        self.yolo_model = None
        if YOLO is not None:
            try:
                model_path = self.config.get('models', {}).get('yolo_player', 'yolov8n.pt')
                self.yolo_model = YOLO(model_path)
                logger.info(f"YOLO model loaded for player detection: {model_path}")
            except Exception as e:
                logger.warning(f"Could not load YOLO model: {e}")
                self.yolo_model = None
    
    def _load_config(self) -> dict:
        """Load configuration from config.yaml"""
        config_path = 'config.yaml'
        if yaml is not None and os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    config = yaml.safe_load(f)
                logger.info(f"Loaded configuration from {config_path}")
                return config
            except Exception as e:
                logger.warning(f"Could not load config file: {e}")
        
        # Return default configuration
        return {
            'models': {
                'yolo_player': 'yolov8n.pt'
            },
            'yolo_player': {
                'conf_threshold': 0.5,
                'iou_threshold': 0.45,
                'max_det': 10
            }
        }
        
    def process_video(self, video_file: str, output_file: str = None, csv_output: str = None, show_viewer: bool = False):
        """Process video directly with player positioning analysis"""
        cap = cv2.VideoCapture(video_file)
        
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_file}")
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        logger.info(f"Processing video: {video_file}")
        logger.info(f"Video: {width}x{height} @ {fps}fps, {total_frames} frames")
        
        # Setup video writer if output specified
        out = None
        if output_file:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_file, fourcc, fps, (width, height))
        
        # Setup viewer
        if show_viewer:
            cv2.namedWindow('Tennis Player Positioning', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('Tennis Player Positioning', 1200, 800)
        
        # Collect positioning data for CSV output
        positioning_data = []
        frame_number = 0
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Process frame and get positioning data
                frame_with_overlays, frame_positions = self._process_frame(frame, frame_number)
                
                # Store positioning data
                if frame_positions:
                    positioning_data.extend(frame_positions)
                
                # Write frame
                if out:
                    out.write(frame_with_overlays)
                
                # Show in viewer
                if show_viewer:
                    cv2.imshow('Tennis Player Positioning', frame_with_overlays)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        break
                    elif key == ord(' '):  # Space to pause
                        cv2.waitKey(0)
                
                # Progress update
                if frame_number % 30 == 0:
                    logger.info(f"Processed {frame_number}/{total_frames} frames ({frame_number/total_frames*100:.1f}%)")
                
                frame_number += 1
        
        finally:
            # Cleanup
            cap.release()
            if out:
                out.release()
            if show_viewer:
                cv2.destroyAllWindows()
        
        # Save positioning data to CSV if specified
        if csv_output and positioning_data:
            df_positions = pd.DataFrame(positioning_data)
            df_positions.to_csv(csv_output, index=False)
            logger.info(f"Player positioning data saved to {csv_output}")
        
        # Print positioning summary
        self._print_summary()
    
    def _process_frame(self, frame: np.ndarray, frame_number: int) -> Tuple[np.ndarray, List[Dict]]:
        """Process a single frame for player positioning"""
        frame = frame.copy()
        
        # Add frame info
        cv2.putText(frame, f"Frame: {frame_number}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Detect players using YOLO
        player_bboxes = self._detect_players(frame)
        
        # Process each detected player
        player_positions = []
        
        for player_idx, (x1, y1, x2, y2, confidence) in enumerate(player_bboxes):
            # Use feet position (bottom center of bounding box) for court positioning
            feet_x = (x1 + x2) / 2
            feet_y = y2  # Bottom of bounding box
            
            # Get player position based on feet
            position = self.positioning.get_player_position(feet_x, feet_y)
            position_confidence = self.positioning.get_position_confidence(feet_x, feet_y)
            
            # Store position data
            player_positions.append({
                'player_id': player_idx,
                'position': position,
                'confidence': position_confidence,
                'feet_x': feet_x,
                'feet_y': feet_y
            })
            
            # Draw player bounding box
            color = self._get_position_color(position)
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            
            # Draw player feet position
            cv2.circle(frame, (int(feet_x), int(feet_y)), 5, color, -1)
            # Draw small "feet" label
            cv2.putText(frame, "feet", (int(feet_x) + 8, int(feet_y)), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
            
            # Draw position label
            label = f"P{player_idx}: {position}"
            if position_confidence > 0.5:  # Only show confidence if reasonable
                label += f" ({position_confidence:.2f})"
            
            # Position label above player
            label_y = max(20, int(y1) - 10)
            cv2.putText(frame, label, (int(x1), label_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        # Store frame data
        self.position_history.append({
            'frame': frame_number,
            'players': player_positions
        })
        
        # Create positioning data for CSV output
        frame_positions = []
        for pos_data in player_positions:
            frame_positions.append({
                'frame': frame_number,
                'player_id': pos_data['player_id'],
                'position': pos_data['position'],
                'confidence': pos_data['confidence'],
                'feet_x': pos_data['feet_x'],
                'feet_y': pos_data['feet_y'],
                'zone': pos_data.get('zone', '')
            })
        
        return frame, frame_positions
    
    def _detect_players(self, frame: np.ndarray) -> List[Tuple[float, float, float, float, float]]:
        """Detect players in frame using YOLO"""
        if self.yolo_model is None:
            return []
        
        try:
            # Get detection parameters from config
            yolo_config = self.config.get('yolo_player', {})
            conf_threshold = yolo_config.get('conf_threshold', 0.5)
            iou_threshold = yolo_config.get('iou_threshold', 0.45)
            max_det = yolo_config.get('max_det', 10)
            
            # Run YOLO detection with config parameters
            results = self.yolo_model(
                frame, 
                conf=conf_threshold,
                iou=iou_threshold,
                max_det=max_det,
                verbose=False
            )
            
            player_bboxes = []
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        # Check if it's a person (class 0 in COCO dataset)
                        # For custom models, we might need to check different class IDs
                        class_id = int(box.cls)
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        conf = float(box.conf[0].cpu().numpy())
                        
                        # For custom tennis models, class 0 is typically person/player
                        # But let's be more flexible and accept any detection above threshold
                        if conf > conf_threshold:
                            player_bboxes.append((x1, y1, x2, y2, conf))
            
            # Sort by confidence and limit to top 2 players
            player_bboxes.sort(key=lambda x: x[4], reverse=True)
            return player_bboxes[:2]  # Max 2 players
            
        except Exception as e:
            logger.warning(f"Error in player detection: {e}")
            return []
    
    
    def _get_position_color(self, position: str) -> Tuple[int, int, int]:
        """Get color for position type"""
        color_map = {
            'FRONT': (0, 255, 0),      # Green
            'BACK': (0, 0, 255),       # Red
            'DOUBLES': (255, 0, 255),  # Magenta
            'OUTSIDE_COURT': (128, 128, 128) # Gray
        }
        return color_map.get(position, (128, 128, 128))
    
    def _print_summary(self):
        """Print positioning analysis summary"""
        logger.info("=== PLAYER POSITIONING SUMMARY ===")
        
        if not self.position_history:
            logger.info("No positioning data available")
            return
        
        # Count positions for each player
        player_stats = {0: {}, 1: {}}
        
        for frame_data in self.position_history:
            for player_data in frame_data['players']:
                player_id = player_data['player_id']
                position = player_data['position']
                
                if position not in player_stats[player_id]:
                    player_stats[player_id][position] = 0
                player_stats[player_id][position] += 1
        
        # Print statistics
        for player_id, stats in player_stats.items():
            total_frames = sum(stats.values())
            logger.info(f"Player {player_id} positioning:")
            for position, count in stats.items():
                percentage = (count / total_frames) * 100 if total_frames > 0 else 0
                logger.info(f"  {position}: {count} frames ({percentage:.1f}%)")


def main():
    """Main function for standalone usage"""
    parser = argparse.ArgumentParser(description='Tennis Player Positioning System')
    parser.add_argument('--video', required=True, help='Input video file')
    parser.add_argument('--output', default='tennis_positioning_analysis.mp4', help='Output video file')
    parser.add_argument('--csv-output', default='player_positioning.csv', help='Output CSV file for positioning data')
    parser.add_argument('--viewer', action='store_true', help='Show real-time viewer')
    
    args = parser.parse_args()
    
    # Import court segmenter
    try:
        from court_segmenter import CourtSegmenter
        court_segmenter = CourtSegmenter()
        
        # Load court keypoints from the generated CSV
        court_keypoints_csv = 'court_keypoints.csv'
        if os.path.exists(court_keypoints_csv):
            court_segmenter.load_court_keypoints(court_keypoints_csv)
            logger.info(f"Loaded court keypoints from {court_keypoints_csv}")
        else:
            logger.error(f"Court keypoints CSV not found: {court_keypoints_csv}")
            logger.error("Please run court_segmenter.py first to generate court keypoints")
            return
        
        processor = TennisPositioningProcessor(court_segmenter)
        processor.process_video(args.video, args.output, args.csv_output, show_viewer=args.viewer)
        
    except ImportError as e:
        logger.error(f"Could not import court_segmenter: {e}")
        logger.error("Make sure court_segmenter.py is in the same directory")
    except Exception as e:
        logger.error(f"Error running positioning analysis: {e}")


if __name__ == "__main__":
    main()
