"""
Main Tennis Analyzer
Orchestrates player detection, pose estimation, and ball tracking for comprehensive tennis analytics
"""

import cv2
import numpy as np
import yaml
import logging
import json
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import time
from datetime import datetime

from player_detector import PlayerDetector
from pose_estimator import PoseEstimator
from ball_tracker_pytorch import TrackNetPyTorch, BallTrajectory

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TennisAnalyzer:
    """Main tennis analytics system that combines all components"""
    
    def __init__(self, config_path: str):
        """
        Initialize tennis analyzer
        
        Args:
            config_path: Path to configuration file
        """
        self.config = self._load_config(config_path)
        self.frame_count = 0
        self.start_time = time.time()
        
        # Initialize components
        self.player_detector = None
        self.pose_estimator = None
        self.ball_tracker = None
        
        # Analysis results storage
        self.analysis_results = {
            'session_info': {
                'start_time': datetime.now().isoformat(),
                'config': self.config
            },
            'frames': [],
            'trajectories': [],
            'swing_analysis': [],
            'summary_stats': {
                'total_frames': 0,
                'total_players_detected': 0,
                'total_poses_estimated': 0,
                'total_ball_detections': 0,
                'swing_phases': {},
                'average_processing_time': 0,
                'processing_times': []
            }
        }
        
        self._initialize_components()
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            logger.info(f"Configuration loaded from {config_path}")
            return config
        except Exception as e:
            logger.error(f"Error loading config: {e}")
            raise
    
    def _initialize_components(self):
        """Initialize all analysis components"""
        try:
            # Initialize player detector
            if Path(self.config['models']['yolo_player']).exists():
                self.player_detector = PlayerDetector(
                    self.config['models']['yolo_player'],
                    self.config['yolo_player']
                )
                logger.info("Player detector initialized successfully")
            else:
                logger.warning(f"Player detection model not found: {self.config['models']['yolo_player']}")
            
            # Initialize pose estimator
            if Path(self.config['models']['yolo_pose']).exists():
                self.pose_estimator = PoseEstimator(
                    self.config['models']['yolo_pose'],
                    self.config['yolo_pose']
                )
                logger.info("Pose estimator initialized successfully")
            else:
                logger.warning(f"Pose estimation model not found: {self.config['models']['yolo_pose']}")
            
            # Initialize ball tracker (PyTorch version)
            try:
                self.ball_tracker = TrackNetPyTorch(
                    self.config['models']['tracknet'],
                    self.config['tracknet']
                )
                logger.info("PyTorch TrackNet ball tracker initialized successfully")
            except Exception as e:
                logger.warning(f"TrackNet initialization failed: {e}")
                logger.info("Continuing without ball tracking...")
                
        except Exception as e:
            logger.error(f"Error initializing components: {e}")
            raise
    
    def analyze_frame(self, frame: np.ndarray) -> Dict[str, Any]:
        """
        Analyze a single frame for tennis insights
        
        Args:
            frame: Input frame (BGR format)
            
        Returns:
            Dictionary with frame analysis results
        """
        frame_start_time = time.time()
        frame_results = {
            'frame_number': self.frame_count,
            'timestamp': time.time() - self.start_time,
            'player_detections': [],
            'poses': [],
            'ball_position': None,
            'swing_analysis': [],
            'processing_time': 0
        }
        
        try:
            # 1. Player Detection
            if self.player_detector:
                player_detections = self.player_detector.detect_players(frame)
                frame_results['player_detections'] = player_detections
                
                # 2. Pose Estimation
                if self.pose_estimator and player_detections:
                    poses = self.pose_estimator.estimate_poses(frame, player_detections)
                    frame_results['poses'] = poses
                    
                    # Analyze swing mechanics for each pose
                    for pose in poses:
                        swing_analysis = self.pose_estimator.analyze_swing_mechanics(pose)
                        if swing_analysis:
                            swing_analysis['pose_id'] = len(frame_results['swing_analysis'])
                            frame_results['swing_analysis'].append(swing_analysis)
            
            # 3. Ball Tracking
            if self.ball_tracker:
                ball_position = self.ball_tracker.predict_ball_position(frame)
                if ball_position:
                    frame_results['ball_position'] = {
                        'x': ball_position.x,
                        'y': ball_position.y,
                        'confidence': ball_position.confidence
                    }
                    
                    # Update trajectory
                    self.ball_tracker.update_trajectory(ball_position)
            
            # Calculate processing time
            frame_results['processing_time'] = time.time() - frame_start_time
            
            # Store frame results
            self.analysis_results['frames'].append(frame_results)
            
            # Update summary statistics
            self._update_summary_stats(frame_results)
            
            self.frame_count += 1
            
        except Exception as e:
            logger.error(f"Error analyzing frame {self.frame_count}: {e}")
            frame_results['error'] = str(e)
        
        return frame_results
    
    def _update_summary_stats(self, frame_results: Dict[str, Any]):
        """Update summary statistics with frame results"""
        if 'summary_stats' not in self.analysis_results:
            self.analysis_results['summary_stats'] = {
                'total_frames': 0,
                'total_players_detected': 0,
                'total_poses_estimated': 0,
                'total_ball_detections': 0,
                'swing_phases': {},
                'average_processing_time': 0,
                'processing_times': []
            }
        
        stats = self.analysis_results['summary_stats']
        stats['total_frames'] += 1
        stats['total_players_detected'] += len(frame_results.get('player_detections', []))
        stats['total_poses_estimated'] += len(frame_results.get('poses', []))
        
        if frame_results.get('ball_position'):
            stats['total_ball_detections'] += 1
        
        # Track swing phases
        for swing in frame_results.get('swing_analysis', []):
            phase = swing.get('swing_phase', 'unknown')
            stats['swing_phases'][phase] = stats['swing_phases'].get(phase, 0) + 1
        
        # Track processing times
        if frame_results.get('processing_time'):
            stats['processing_times'].append(frame_results['processing_time'])
            stats['average_processing_time'] = np.mean(stats['processing_times'])
    
    def get_current_ball_trajectory(self) -> Optional[BallTrajectory]:
        """Get current ball trajectory"""
        if self.ball_tracker:
            return self.ball_tracker.get_current_trajectory()
        return None
    
    def analyze_current_trajectory(self) -> Dict[str, Any]:
        """Analyze current ball trajectory"""
        if self.ball_tracker:
            trajectory = self.ball_tracker.get_current_trajectory()
            if trajectory:
                analysis = self.ball_tracker.analyze_trajectory(trajectory)
                
                # Store trajectory analysis
                if trajectory.is_valid:
                    self.analysis_results['trajectories'].append(analysis)
                
                return analysis
        return {}
    
    def draw_analysis_on_frame(self, frame: np.ndarray, frame_results: Dict[str, Any]) -> np.ndarray:
        """
        Draw all analysis results on frame
        
        Args:
            frame: Input frame
            frame_results: Frame analysis results
            
        Returns:
            Frame with drawn analysis
        """
        frame_copy = frame.copy()
        
        try:
            # Draw player detections
            if self.player_detector and frame_results.get('player_detections'):
                frame_copy = self.player_detector.draw_detections(
                    frame_copy, frame_results['player_detections']
                )
            
            # Draw poses
            if self.pose_estimator and frame_results.get('poses'):
                frame_copy = self.pose_estimator.draw_poses(
                    frame_copy, frame_results['poses']
                )
            
            # Draw ball position and trajectory
            if self.ball_tracker:
                # Draw current ball position
                if frame_results.get('ball_position'):
                    x, y = frame_results['ball_position']['x'], frame_results['ball_position']['y']
                    cv2.circle(frame_copy, (int(x), int(y)), 8, (0, 255, 0), -1)  # Green circle for current ball
                    cv2.circle(frame_copy, (int(x), int(y)), 10, (255, 255, 255), 2)  # White outline
                
                # Draw ball trajectory
                trajectory = self.ball_tracker.get_current_trajectory()
                if trajectory and len(trajectory.positions) > 1:
                    frame_copy = self.ball_tracker.draw_trajectory(frame_copy, trajectory)
            
            # Draw frame info
            self._draw_frame_info(frame_copy, frame_results)
            
        except Exception as e:
            logger.error(f"Error drawing analysis on frame: {e}")
        
        return frame_copy
    
    def _draw_frame_info(self, frame: np.ndarray, frame_results: Dict[str, Any]):
        """Draw frame information overlay"""
        # Frame number and timestamp
        info_text = f"Frame: {frame_results['frame_number']}"
        cv2.putText(frame, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Processing time
        if frame_results.get('processing_time'):
            time_text = f"Time: {frame_results['processing_time']:.3f}s"
            cv2.putText(frame, time_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Player count
        player_count = len(frame_results.get('player_detections', []))
        player_text = f"Players: {player_count}"
        cv2.putText(frame, player_text, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Ball detection status
        if frame_results.get('ball_position'):
            ball_text = f"Ball: {frame_results['ball_position']['confidence']:.2f}"
            cv2.putText(frame, ball_text, (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            cv2.putText(frame, "Ball: Not detected", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    def save_analysis_results(self, output_path: str):
        """Save analysis results to JSON file"""
        try:
            # Finalize any ongoing trajectories
            if self.ball_tracker:
                self.ball_tracker._finalize_trajectory()
            
            # Add session end time
            self.analysis_results['session_info']['end_time'] = datetime.now().isoformat()
            self.analysis_results['session_info']['total_duration'] = time.time() - self.start_time
            
            # Save to file
            with open(output_path, 'w') as f:
                json.dump(self.analysis_results, f, indent=2, default=str)
            
            logger.info(f"Analysis results saved to {output_path}")
            
        except Exception as e:
            logger.error(f"Error saving analysis results: {e}")
    
    def get_summary_report(self) -> Dict[str, Any]:
        """Get a summary report of the analysis session"""
        if not self.analysis_results['summary_stats']:
            return {}
        
        stats = self.analysis_results['summary_stats']
        
        # Calculate additional metrics
        total_duration = time.time() - self.start_time
        fps = stats['total_frames'] / total_duration if total_duration > 0 else 0
        
        # Ball trajectory analysis
        trajectory_stats = {}
        if self.ball_tracker:
            trajectory_stats = {
                'total_trajectories': len(self.analysis_results['trajectories']),
                'valid_trajectories': sum(1 for t in self.analysis_results['trajectories'] if t.get('is_valid', False)),
                'average_trajectory_duration': np.mean([t.get('duration', 0) for t in self.analysis_results['trajectories']]) if self.analysis_results['trajectories'] else 0
            }
        
        report = {
            'session_duration': total_duration,
            'total_frames_processed': stats['total_frames'],
            'average_fps': fps,
            'total_players_detected': stats['total_players_detected'],
            'total_poses_estimated': stats['total_poses_estimated'],
            'total_ball_detections': stats['total_ball_detections'],
            'swing_phase_distribution': stats['swing_phases'],
            'average_processing_time_per_frame': stats['average_processing_time'],
            'ball_trajectory_stats': trajectory_stats,
            'detection_rate': {
                'players': stats['total_players_detected'] / max(stats['total_frames'], 1),
                'poses': stats['total_poses_estimated'] / max(stats['total_frames'], 1),
                'ball': stats['total_ball_detections'] / max(stats['total_frames'], 1)
            }
        }
        
        return report
    
    def print_summary_report(self):
        """Print a formatted summary report to console"""
        report = self.get_summary_report()
        
        if not report:
            logger.warning("No analysis data available for summary report")
            return
        
        print("\n" + "="*60)
        print("TENNIS ANALYSIS SESSION SUMMARY")
        print("="*60)
        print(f"Session Duration: {report['session_duration']:.2f} seconds")
        print(f"Frames Processed: {report['total_frames_processed']}")
        print(f"Average FPS: {report['average_fps']:.2f}")
        print(f"Players Detected: {report['total_players_detected']}")
        print(f"Poses Estimated: {report['total_poses_estimated']}")
        print(f"Ball Detections: {report['total_ball_detections']}")
        print(f"Avg Processing Time: {report['average_processing_time_per_frame']:.3f}s")
        
        if report['ball_trajectory_stats']:
            print(f"\nBall Trajectories:")
            print(f"  Total: {report['ball_trajectory_stats']['total_trajectories']}")
            print(f"  Valid: {report['ball_trajectory_stats']['valid_trajectories']}")
            print(f"  Avg Duration: {report['ball_trajectory_stats']['average_trajectory_duration']:.2f}s")
        
        if report['swing_phase_distribution']:
            print(f"\nSwing Phase Distribution:")
            for phase, count in report['swing_phase_distribution'].items():
                percentage = (count / max(report['total_frames'], 1)) * 100
                print(f"  {phase}: {count} ({percentage:.1f}%)")
        
        print(f"\nDetection Rates:")
        print(f"  Players: {report['detection_rate']['players']:.2f}")
        print(f"  Poses: {report['detection_rate']['poses']:.2f}")
        print(f"  Ball: {report['detection_rate']['ball']:.2f}")
        
        print("="*60)
