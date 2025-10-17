#!/usr/bin/env python3
"""
Tennis Ball Detection System
Self-contained script using hybrid approach combining our system and TRACE
"""

import cv2
import numpy as np
import torch
import torch.nn as nn
import pandas as pd
import logging
from typing import Optional, Tuple, List, Dict, Any
from pathlib import Path
import sys
import os
import argparse
import yaml
from PIL import Image

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add RF-DETR to path
sys.path.append('.')

# Import RF-DETR for enhanced ball detection
try:
    from rfdetr import RFDETRNano
    RFDETR_AVAILABLE = True
    logger.info("RF-DETR imports successful - Enhanced detection enabled")
except ImportError as e:
    RFDETR_AVAILABLE = False
    logger.warning(f"RF-DETR imports failed: {e} - Enhanced detection will be disabled")

class ConvBlock(nn.Module):
    """Convolutional block with optional batch normalization"""
    def __init__(self, in_channels, out_channels, kernel_size, pad, bias=True, bn=True):
        super().__init__()
        if bn:
            self.block = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size, padding=pad, bias=bias),
                nn.ReLU(),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.block = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size, padding=pad, bias=bias),
                nn.ReLU()
            )

    def forward(self, x):
        return self.block(x)

class BallTrackerNet(nn.Module):
    """TrackNet implementation based on TRACE"""
    def __init__(self, out_channels=256, bn=True):
        super().__init__()
        self.out_channels = out_channels

        # Encoder layers
        layer_1 = ConvBlock(in_channels=9, out_channels=64, kernel_size=3, pad=1, bias=True, bn=bn)
        layer_2 = ConvBlock(in_channels=64, out_channels=64, kernel_size=3, pad=1, bias=True, bn=bn)
        layer_3 = nn.MaxPool2d(kernel_size=2, stride=2)
        layer_4 = ConvBlock(in_channels=64, out_channels=128, kernel_size=3, pad=1, bias=True, bn=bn)
        layer_5 = ConvBlock(in_channels=128, out_channels=128, kernel_size=3, pad=1, bias=True, bn=bn)
        layer_6 = nn.MaxPool2d(kernel_size=2, stride=2)
        layer_7 = ConvBlock(in_channels=128, out_channels=256, kernel_size=3, pad=1, bias=True, bn=bn)
        layer_8 = ConvBlock(in_channels=256, out_channels=256, kernel_size=3, pad=1, bias=True, bn=bn)
        layer_9 = ConvBlock(in_channels=256, out_channels=256, kernel_size=3, pad=1, bias=True, bn=bn)
        layer_10 = nn.MaxPool2d(kernel_size=2, stride=2)
        layer_11 = ConvBlock(in_channels=256, out_channels=512, kernel_size=3, pad=1, bias=True, bn=bn)
        layer_12 = ConvBlock(in_channels=512, out_channels=512, kernel_size=3, pad=1, bias=True, bn=bn)
        layer_13 = ConvBlock(in_channels=512, out_channels=512, kernel_size=3, pad=1, bias=True, bn=bn)

        self.encoder = nn.Sequential(layer_1, layer_2, layer_3, layer_4, layer_5, layer_6, layer_7, layer_8, layer_9,
                                     layer_10, layer_11, layer_12, layer_13)

        # Decoder layers
        layer_14 = nn.Upsample(scale_factor=2)
        layer_15 = ConvBlock(in_channels=512, out_channels=256, kernel_size=3, pad=1, bias=True, bn=bn)
        layer_16 = ConvBlock(in_channels=256, out_channels=256, kernel_size=3, pad=1, bias=True, bn=bn)
        layer_17 = ConvBlock(in_channels=256, out_channels=256, kernel_size=3, pad=1, bias=True, bn=bn)
        layer_18 = nn.Upsample(scale_factor=2)
        layer_19 = ConvBlock(in_channels=256, out_channels=128, kernel_size=3, pad=1, bias=True, bn=bn)
        layer_20 = ConvBlock(in_channels=128, out_channels=128, kernel_size=3, pad=1, bias=True, bn=bn)
        layer_21 = nn.Upsample(scale_factor=2)
        layer_22 = ConvBlock(in_channels=128, out_channels=64, kernel_size=3, pad=1, bias=True, bn=bn)
        layer_23 = ConvBlock(in_channels=64, out_channels=64, kernel_size=3, pad=1, bias=True, bn=bn)
        layer_24 = ConvBlock(in_channels=64, out_channels=self.out_channels, kernel_size=3, pad=1, bias=True, bn=bn)

        self.decoder = nn.Sequential(layer_14, layer_15, layer_16, layer_17, layer_18, layer_19, layer_20, layer_21,
                                     layer_22, layer_23, layer_24)

        self.softmax = nn.Softmax(dim=1)
        self._init_weights()

    def forward(self, x, testing=False):
        batch_size = x.size(0)
        features = self.encoder(x)
        scores_map = self.decoder(features)
        output = scores_map.reshape(batch_size, self.out_channels, -1)
        if testing:
            output = self.softmax(output)
        return output

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.uniform_(module.weight, -0.05, 0.05)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)

    def inference(self, frames: torch.Tensor):
        self.eval()
        with torch.no_grad():
            if len(frames.shape) == 3:
                frames = frames.unsqueeze(0)
            if next(self.parameters()).is_cuda:
                frames.cuda()
            # Forward pass
            output = self(frames, True)
            output = output.argmax(dim=1).detach().cpu().numpy()
            if self.out_channels == 2:
                output *= 255
            x, y = self.get_center_ball(output)
        return x, y

    def get_center_ball(self, output):
        """Detect the center of the ball using Hough circle transform"""
        output = output.reshape((360, 640))
        output = output.astype(np.uint8)
        heatmap = cv2.resize(output, (640, 360))
        
        # Convert to binary image
        ret, heatmap = cv2.threshold(heatmap, 127, 255, cv2.THRESH_BINARY)
        
        # Find circles with radius 2-7 pixels
        circles = cv2.HoughCircles(heatmap, cv2.HOUGH_GRADIENT, dp=1, minDist=1, 
                                 param1=50, param2=8, minRadius=2, maxRadius=7)
        
        if circles is not None and len(circles) == 1:
            x = int(circles[0][0][0])
            y = int(circles[0][0][1])
            return x, y
        return None, None

def combine_three_frames(frame1, frame2, frame3, width, height):
    """Combine three consecutive frames for TrackNet input"""
    # Resize and convert to float
    img = cv2.resize(frame1, (width, height)).astype(np.float32)
    img1 = cv2.resize(frame2, (width, height)).astype(np.float32)
    img2 = cv2.resize(frame3, (width, height)).astype(np.float32)
    
    # Combine frames (width, height, rgb*3)
    imgs = np.concatenate((img, img1, img2), axis=2)
    
    # Change to channels_first for TrackNet
    imgs = np.rollaxis(imgs, 2, 0)
    return np.array(imgs)

class TraceBallDetector:
    """TRACE ball detector"""
    
    def __init__(self, model_path: str = "TRACE/TrackNet/Weights.pth", out_channels: int = 2):
        self.device = torch.device("cpu")
        
        # Load TrackNet model
        self.detector = BallTrackerNet(out_channels=out_channels)
        try:
            saved_state_dict = torch.load(model_path, map_location=torch.device("cpu"), weights_only=False)
            self.detector.load_state_dict(saved_state_dict['model_state'])
            logger.info(f"Loaded TrackNet model from {model_path}")
        except Exception as e:
            logger.error(f"Failed to load model from {model_path}: {e}")
            raise
        
        self.detector.eval().to(self.device)
        
        # Frame history for 3-frame input
        self.current_frame = None
        self.last_frame = None
        self.before_last_frame = None
        
        # Video dimensions
        self.video_width = None
        self.video_height = None
        self.model_input_width = 640
        self.model_input_height = 360
        
        # Detection parameters
        self.threshold_dist = 100  # Maximum distance between consecutive detections
        self.xy_coordinates = np.array([[None, None], [None, None]])
        
        # Detection history
        self.detection_history = []
        self.max_history = 50

    def detect_ball(self, frame: np.ndarray) -> Tuple[Optional[float], Optional[float], float]:
        """
        Detect ball in current frame using 3-frame TrackNet approach
        Returns: (x, y, confidence)
        """
        # Save frame dimensions
        if self.video_width is None:
            self.video_width = frame.shape[1]
            self.video_height = frame.shape[0]
        
        # Update frame history
        self.last_frame = self.before_last_frame
        self.before_last_frame = self.current_frame
        self.current_frame = frame.copy()
        
        # Only detect if we have 3 frames
        if self.last_frame is not None:
            # Combine frames for TrackNet input
            frames = combine_three_frames(self.current_frame, self.before_last_frame, self.last_frame,
                                        self.model_input_width, self.model_input_height)
            frames = (torch.from_numpy(frames) / 255).to(self.device)
            
            # Run inference
            x, y = self.detector.inference(frames)
            
            if x is not None and y is not None:
                # Scale coordinates back to video dimensions
                x = int(x * (self.video_width / self.model_input_width))
                y = int(y * (self.video_height / self.model_input_height))
                
                # Check distance from previous detection (outlier filtering)
                if self.xy_coordinates[-1][0] is not None:
                    prev_x, prev_y = self.xy_coordinates[-1]
                    distance = np.linalg.norm(np.array([x, y]) - np.array([prev_x, prev_y]))
                    
                    if distance > self.threshold_dist:
                        # Outlier detected, use previous position
                        x, y = prev_x, prev_y
                        confidence = 0.3  # Lower confidence for outlier
                    else:
                        confidence = 0.9  # High confidence for valid detection
                else:
                    confidence = 0.8  # Medium confidence for first detection
                
                # Update coordinate history
                self.xy_coordinates = np.append(self.xy_coordinates, np.array([[x, y]]), axis=0)
                
                # Keep only recent history
                if len(self.xy_coordinates) > self.max_history:
                    self.xy_coordinates = self.xy_coordinates[-self.max_history:]
                
                # Add to detection history
                self.detection_history.append({
                    'x': x, 'y': y, 'confidence': confidence
                })
                
                return float(x), float(y), confidence
            else:
                # No detection, return None with low confidence
                return None, None, 0.0
        else:
            return None, None, 0.0

class HybridBallDetector:
    """Hybrid ball detector combining TRACE and RF-DETR"""
    
    def __init__(self, trace_model_path: str = "TRACE/TrackNet/Weights.pth", 
                 rfdetr_model_path: str = "models/playersnball5.pt", 
                 config_path: str = "config.yaml"):
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Initialize TRACE detector
        self.trace_detector = TraceBallDetector(trace_model_path)
        
        # Initialize RF-DETR detector
        self.rfdetr_detector = RFDETRBallDetector(rfdetr_model_path, self.config.get('rfdetr', {}))
        
        # Detection history for quality assessment
        self.detection_history = []
        self.max_history = 20
        
        # Quality metrics
        self.trace_quality_score = 0.5
        self.rfdetr_quality_score = 0.5
        
        # Detection parameters
        self.min_confidence_threshold = 0.3
        self.max_jump_distance = 150  # Maximum reasonable jump between frames
        
        logger.info("Initialized HybridBallDetector with TRACE and RF-DETR")

    def get_rfdetr_ball_detection(self, frame: np.ndarray) -> Tuple[Optional[float], Optional[float], float]:
        """Get ball detection using RF-DETR"""
        return self.rfdetr_detector.detect_ball(frame)

    def assess_detection_quality(self, x: float, y: float, confidence: float, 
                               method: str) -> float:
        """Assess the quality of a ball detection with improved logic"""
        if x is None or y is None or confidence < self.min_confidence_threshold:
            return 0.0
        
        # Start with confidence as base quality
        quality = confidence
        
        # 1. Movement consistency analysis
        movement_penalty = 0.0
        if len(self.detection_history) > 0:
            last_detection = self.detection_history[-1]
            if last_detection['x'] is not None and last_detection['y'] is not None:
                distance = np.sqrt((x - last_detection['x'])**2 + (y - last_detection['y'])**2)
                
                # More aggressive penalties for large jumps
                if distance > self.max_jump_distance:
                    movement_penalty = 0.8  # Very heavy penalty
                elif distance > 100:
                    movement_penalty = 0.4  # Heavy penalty
                elif distance > 50:
                    movement_penalty = 0.1  # Light penalty
                else:
                    movement_penalty = -0.1  # Small bonus for reasonable movement
        
        # 2. Method-specific analysis
        method_bonus = 0.0
        if method == "trace":
            # TRACE-specific penalties
            if len(self.detection_history) >= 3:
                # Check if TRACE has been stuck recently
                recent_trace = [d for d in self.detection_history[-3:] if d['method'] == 'trace' and d['x'] is not None]
                if len(recent_trace) >= 2:
                    positions = [(d['x'], d['y']) for d in recent_trace]
                    variance = np.var(positions, axis=0).sum()
                    if variance < 50:  # Very low variance = stuck
                        method_bonus = -0.6  # Heavy penalty for stuck TRACE
                    elif variance < 100:
                        method_bonus = -0.3  # Moderate penalty
                
                # Check if TRACE has been consistently low quality
                recent_trace_quality = [d['quality'] for d in self.detection_history[-5:] if d['method'] == 'trace']
                if len(recent_trace_quality) >= 3:
                    avg_trace_quality = np.mean(recent_trace_quality)
                    if avg_trace_quality < 0.3:
                        method_bonus -= 0.4  # Penalize consistently poor TRACE
                        
        elif method == "rfdetr":
            # RF-DETR-specific bonuses
            method_bonus = 0.2  # Base bonus for RF-DETR
            
            # Additional bonus if RF-DETR has been performing well recently
            if len(self.detection_history) >= 3:
                recent_rfdetr_quality = [d['quality'] for d in self.detection_history[-5:] if d['method'] == 'rfdetr']
                if len(recent_rfdetr_quality) >= 2:
                    avg_rfdetr_quality = np.mean(recent_rfdetr_quality)
                    if avg_rfdetr_quality > 0.5:
                        method_bonus += 0.2  # Bonus for consistently good RF-DETR
        
        # 3. Confidence-based adjustments
        confidence_bonus = 0.0
        if confidence > 0.8:
            confidence_bonus = 0.2  # Bonus for high confidence
        elif confidence < 0.4:
            confidence_bonus = -0.3  # Penalty for low confidence
        
        # 4. Temporal consistency (check if this position makes sense given recent trajectory)
        trajectory_bonus = 0.0
        if len(self.detection_history) >= 2:
            recent_detections = [d for d in self.detection_history[-3:] if d['x'] is not None and d['y'] is not None]
            if len(recent_detections) >= 2:
                # Calculate expected position based on recent movement
                positions = [(d['x'], d['y']) for d in recent_detections]
                if len(positions) >= 2:
                    # Simple linear extrapolation
                    dx = positions[-1][0] - positions[-2][0]
                    dy = positions[-1][1] - positions[-2][1]
                    expected_x = positions[-1][0] + dx
                    expected_y = positions[-1][1] + dy
                    
                    # Check how close this detection is to expected position
                    expected_distance = np.sqrt((x - expected_x)**2 + (y - expected_y)**2)
                    if expected_distance < 30:
                        trajectory_bonus = 0.3  # Bonus for following expected trajectory
                    elif expected_distance > 100:
                        trajectory_bonus = -0.2  # Penalty for deviating from expected trajectory
        
        # Combine all factors
        final_quality = quality + method_bonus + confidence_bonus + trajectory_bonus - movement_penalty
        
        return min(1.0, max(0.0, final_quality))

    def detect_ball_hybrid(self, frame: np.ndarray) -> Tuple[Optional[float], Optional[float], float, str]:
        """
        Hybrid ball detection combining TRACE and RF-DETR
        Returns: (x, y, confidence, method_used)
        """
        # Get TRACE detection
        trace_x, trace_y, trace_conf = self.trace_detector.detect_ball(frame)
        
        # Get RF-DETR detection
        rfdetr_x, rfdetr_y, rfdetr_conf = self.get_rfdetr_ball_detection(frame)
        
        # Assess quality of both detections
        trace_quality = self.assess_detection_quality(trace_x, trace_y, trace_conf, "trace")
        rfdetr_quality = self.assess_detection_quality(rfdetr_x, rfdetr_y, rfdetr_conf, "rfdetr")
        
        # Debug logging
        if trace_x is not None or rfdetr_x is not None:
            logger.debug(f"TRACE: ({trace_x}, {trace_y}) conf={trace_conf:.3f} quality={trace_quality:.3f}")
            logger.debug(f"RF-DETR: ({rfdetr_x}, {rfdetr_y}) conf={rfdetr_conf:.3f} quality={rfdetr_quality:.3f}")
        
        # Choose the best detection with improved logic
        best_quality = 0.0
        chosen_x, chosen_y, chosen_conf = None, None, 0.0
        chosen_method = "none"
        
        # If both models detect something, be more selective
        if trace_x is not None and rfdetr_x is not None:
            # Both models detected something - choose based on quality difference
            quality_diff = abs(trace_quality - rfdetr_quality)
            
            if quality_diff > 0.3:  # Significant quality difference
                # Choose the clearly better one
                if trace_quality > rfdetr_quality:
                    chosen_x, chosen_y, chosen_conf = trace_x, trace_y, trace_conf
                    chosen_method = "trace"
                    best_quality = trace_quality
                else:
                    chosen_x, chosen_y, chosen_conf = rfdetr_x, rfdetr_y, rfdetr_conf
                    chosen_method = "rfdetr"
                    best_quality = rfdetr_quality
            else:
                # Quality is similar - prefer RF-DETR as it's generally more reliable
                if rfdetr_quality >= trace_quality * 0.8:  # RF-DETR is at least 80% as good
                    chosen_x, chosen_y, chosen_conf = rfdetr_x, rfdetr_y, rfdetr_conf
                    chosen_method = "rfdetr"
                    best_quality = rfdetr_quality
                elif trace_quality > rfdetr_quality * 1.2:  # TRACE is significantly better
                    chosen_x, chosen_y, chosen_conf = trace_x, trace_y, trace_conf
                    chosen_method = "trace"
                    best_quality = trace_quality
                else:
                    # Still prefer RF-DETR if quality is close
                    chosen_x, chosen_y, chosen_conf = rfdetr_x, rfdetr_y, rfdetr_conf
                    chosen_method = "rfdetr"
                    best_quality = rfdetr_quality
        else:
            # Only one model detected something - use it if quality is reasonable
            if trace_x is not None and trace_quality > 0.3:
                chosen_x, chosen_y, chosen_conf = trace_x, trace_y, trace_conf
                chosen_method = "trace"
                best_quality = trace_quality
            elif rfdetr_x is not None and rfdetr_quality > 0.2:
                chosen_x, chosen_y, chosen_conf = rfdetr_x, rfdetr_y, rfdetr_conf
                chosen_method = "rfdetr"
                best_quality = rfdetr_quality
        
        # Update quality scores for adaptive behavior
        if chosen_method == "trace":
            self.trace_quality_score = 0.7 * self.trace_quality_score + 0.3 * best_quality
        elif chosen_method == "rfdetr":
            self.rfdetr_quality_score = 0.7 * self.rfdetr_quality_score + 0.3 * best_quality
        
        # Debug logging for chosen method
        if chosen_method != "none":
            logger.debug(f"CHOSEN: {chosen_method} at ({chosen_x}, {chosen_y}) conf={chosen_conf:.3f} quality={best_quality:.3f}")
        
        # Store detection in history
        self.detection_history.append({
            'x': chosen_x,
            'y': chosen_y,
            'confidence': chosen_conf,
            'method': chosen_method,
            'quality': best_quality,
            'trace_x': trace_x,
            'trace_y': trace_y,
            'trace_conf': trace_conf,
            'rfdetr_x': rfdetr_x,
            'rfdetr_y': rfdetr_y,
            'rfdetr_conf': rfdetr_conf
        })
        
        # Keep only recent history
        if len(self.detection_history) > self.max_history:
            self.detection_history.pop(0)
        
        return chosen_x, chosen_y, chosen_conf, chosen_method

class RFDETRBallDetector:
    """RF-DETR-based ball detection for tennis"""
    
    def __init__(self, model_path: str, config: Dict[str, Any]):
        self.config = config
        self.model = None
        
        if not RFDETR_AVAILABLE:
            logger.warning("RF-DETR not available, ball detection will be disabled")
            return
            
        try:
            # Load checkpoint first to get configuration
            checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
            if 'args' in checkpoint and 'model' in checkpoint:
                args = checkpoint['args']
                logger.info(f"RF-DETR model with {args.num_classes} classes: {args.class_names}")
                
                # Create RF-DETR with custom classes
                self.model = RFDETRNano(
                    num_classes=len(args.class_names),  # 2 classes: ball + player
                    pretrain_weights=None  # Don't load default weights
                )
                
                # Load custom state dict
                missing_keys, unexpected_keys = self.model.model.model.load_state_dict(checkpoint['model'], strict=False)
                if missing_keys:
                    logger.warning(f"Missing keys when loading RF-DETR: {len(missing_keys)}")
                if unexpected_keys:
                    logger.warning(f"Unexpected keys when loading RF-DETR: {len(unexpected_keys)}")
                
                # Set class names
                try:
                    self.model.class_names = args.class_names
                except:
                    self.model.model.class_names = args.class_names
                
                self.args = args
                
                # Optimize model for inference
                try:
                    self.model.optimize_for_inference()
                    logger.info("RF-DETR model optimized for inference")
                except Exception as e:
                    logger.warning(f"Could not optimize RF-DETR model for inference: {e}")
                
                logger.info("RF-DETR ball detector initialized successfully")
            else:
                logger.error("Invalid RF-DETR checkpoint format")
                self.model = None
        except Exception as e:
            logger.error(f"Error loading RF-DETR model: {e}")
            self.model = None
    
    def detect_ball(self, frame: np.ndarray) -> Tuple[Optional[float], Optional[float], float]:
        """
        Detect ball using RF-DETR
        Returns: (x, y, confidence)
        """
        if not self.model:
            return None, None, 0.0
        
        try:
            # Convert BGR to RGB and to PIL
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(frame_rgb)
            
            # Run inference
            detections = self.model.predict(pil_image, threshold=self.config.get('ball_conf_threshold', 0.2))
            
            # Filter for ball only (class_id == 1 for 'ball')
            balls = []
            
            if hasattr(detections, 'xyxy') and len(detections.xyxy) > 0:
                for i in range(len(detections.xyxy)):
                    bbox = detections.xyxy[i]
                    confidence = detections.confidence[i]
                    class_id = detections.class_id[i]
                    
                    # Only keep balls (class_id == 1 for 'ball')
                    if class_id == 1 and confidence > self.config.get('ball_conf_threshold', 0.2):
                        x1, y1, x2, y2 = bbox
                        center_x = int((x1 + x2) / 2)
                        center_y = int((y1 + y2) / 2)
                        balls.append((center_x, center_y, confidence))
            
            # Return highest confidence ball
            if balls:
                balls.sort(key=lambda x: x[2], reverse=True)  # Sort by confidence
                x, y, conf = balls[0]
                return float(x), float(y), float(conf)
            else:
                return None, None, 0.0
            
        except Exception as e:
            logger.error(f"RF-DETR ball detection error: {e}")
            return None, None, 0.0

class TennisBallProcessor:
    """Main processor for tennis ball detection"""
    
    def __init__(self, video_path: str, output_path: str = "tennis_ball_detection.mp4"):
        self.video_path = video_path
        self.output_path = output_path
        
        # Initialize video capture
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Initialize hybrid detector
        self.hybrid_detector = HybridBallDetector()
        
        # Initialize video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.out = cv2.VideoWriter(output_path, fourcc, self.fps, (self.width, self.height))
        
        # Results storage
        self.detection_results = []
        
        logger.info(f"Initialized TennisBallProcessor for {video_path}")
        logger.info(f"Video: {self.width}x{self.height} @ {self.fps}fps")


    def process_video(self):
        """Process entire video with hybrid ball detection"""
        frame_count = 0
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            # Get hybrid detection using only models
            hybrid_x, hybrid_y, hybrid_conf, method = self.hybrid_detector.detect_ball_hybrid(frame)
            
            # Store results
            self.detection_results.append({
                'frame': frame_count,
                'x': hybrid_x,
                'y': hybrid_y,
                'confidence': hybrid_conf,
                'method': method
            })
            
            # Draw ball if detected
            if hybrid_x is not None and hybrid_y is not None:
                # Use consistent yellow color for ball
                ball_color = (0, 255, 255)  # Yellow for ball
                text_color = (255, 255, 255)  # White for text
                
                # Draw ball
                cv2.circle(frame, (int(hybrid_x), int(hybrid_y)), 8, ball_color, -1)
                cv2.putText(frame, f"{method.upper()}: {hybrid_conf:.2f}", 
                           (int(hybrid_x) + 10, int(hybrid_y) - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2)
            
            # Draw method comparison info
            cv2.putText(frame, f"TRACE Quality: {self.hybrid_detector.trace_quality_score:.2f}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"RF-DETR Quality: {self.hybrid_detector.rfdetr_quality_score:.2f}", 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            
            # Write frame
            self.out.write(frame)
            frame_count += 1
            
            if frame_count % 30 == 0:
                logger.info(f"Processed {frame_count} frames")
        
        # Cleanup
        self.cap.release()
        self.out.release()
        
        logger.info(f"Processing complete! Processed {frame_count} frames")
        logger.info(f"Output saved to: {self.output_path}")
        
        return self.detection_results

    def save_results_to_csv(self, output_csv: str = "tennis_ball_results.csv"):
        """Save detection results to CSV"""
        df = pd.DataFrame(self.detection_results)
        df.to_csv(output_csv, index=False)
        logger.info(f"Results saved to: {output_csv}")

    def print_summary(self):
        """Print detection summary"""
        total_frames = len(self.detection_results)
        detections = [r for r in self.detection_results if r['x'] is not None]
        trace_detections = [r for r in detections if r['method'] == 'trace']
        rfdetr_detections = [r for r in detections if r['method'] == 'rfdetr']
        
        print(f"\n=== TENNIS BALL DETECTION SUMMARY ===")
        print(f"Total frames: {total_frames}")
        print(f"Total detections: {len(detections)} ({len(detections)/total_frames*100:.1f}%)")
        print(f"TRACE detections: {len(trace_detections)} ({len(trace_detections)/len(detections)*100:.1f}%)")
        print(f"RF-DETR detections: {len(rfdetr_detections)} ({len(rfdetr_detections)/len(detections)*100:.1f}%)")
        print(f"Final TRACE quality score: {self.hybrid_detector.trace_quality_score:.3f}")
        print(f"Final RF-DETR quality score: {self.hybrid_detector.rfdetr_quality_score:.3f}")

def main():
    """Main function to run tennis ball detection"""
    parser = argparse.ArgumentParser(description="Tennis Ball Detection using hybrid approach")
    parser.add_argument("--video", required=True, help="Input video path")
    parser.add_argument("--output", default="tennis_ball_detection.mp4", help="Output video path")
    parser.add_argument("--results-csv", default="tennis_ball_results.csv", help="Output CSV path")
    
    args = parser.parse_args()
    
    try:
        # Initialize processor
        processor = TennisBallProcessor(args.video, args.output)
        
        # Process video
        results = processor.process_video()
        
        # Save results
        processor.save_results_to_csv(args.results_csv)
        
        # Print summary
        processor.print_summary()
        
    except Exception as e:
        logger.error(f"Error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
