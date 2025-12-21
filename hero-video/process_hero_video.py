#!/usr/bin/env python3
"""
Hero Video Generator - Combines SAM-3d-body player tracking with SAM3 ball detection
to create promotional videos for the CourtVision hero section.
"""

import sys
import os
import argparse
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any
import json

import cv2
import numpy as np
import torch
from tqdm import tqdm
from PIL import Image

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

# Add SAM-3d-body to path
SAM3D_BODY_PATH = PROJECT_ROOT / "SAM-3d-body" / "sam-3d-body"
if SAM3D_BODY_PATH.exists():
    sys.path.insert(0, str(SAM3D_BODY_PATH))

# Import SAM-3d-body utilities
try:
    from notebook.utils import setup_sam_3d_body, load_sam_3d_body_hf
    from sam_3d_body.sam_3d_body_estimator import SAM3DBodyEstimator
    from sam_3d_body.visualization.skeleton_visualizer import SkeletonVisualizer
    from sam_3d_body.metadata.mhr70 import pose_info as mhr70_pose_info
except ImportError as e:
    print(f"Error importing SAM-3d-body: {e}")
    print("Make sure SAM-3d-body is set up correctly.")
    sys.exit(1)

# Import YOLO human detector
try:
    from yolo_human_detector import YOLOHumanDetector
    YOLO_DETECTOR_AVAILABLE = True
except ImportError as e:
    print(f"Warning: YOLO human detector not available: {e}")
    YOLO_DETECTOR_AVAILABLE = False

# Import SAM3 ball detector (always needed as fallback)
try:
    from SAM3.test_sam3_ball_detection import SAM3BallDetector
    SAM3_DETECTOR_AVAILABLE = True
except ImportError as e:
    print(f"Error importing SAM3: {e}")
    print("Make sure SAM3 is set up correctly.")
    SAM3_DETECTOR_AVAILABLE = False
    sys.exit(1)

# Import ensemble ball detector (combines all available models)
try:
    from ensemble_ball_detector import EnsembleBallDetector
    ENSEMBLE_DETECTOR_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Ensemble detector not available, will use SAM3 only: {e}")
    ENSEMBLE_DETECTOR_AVAILABLE = False

# Import visualizer
from visualizer import HeroVideoVisualizer

# Import custom mesh visualizer for emerald green mesh (optional, for full mesh mode)
try:
    from mesh_visualizer import visualize_sample_together_emerald
    MESH_VISUALIZER_AVAILABLE = True
except ImportError:
    MESH_VISUALIZER_AVAILABLE = False
    # Fallback to original if custom not available
    try:
        from tools.vis_utils import visualize_sample_together
        visualize_sample_together_emerald = visualize_sample_together
        MESH_VISUALIZER_AVAILABLE = True
    except ImportError:
        MESH_VISUALIZER_AVAILABLE = False

# Try to import court detection (optional)
COURT_DETECTION_AVAILABLE = False
CourtDetector = None
try:
    # Try importing from old codebase
    sys.path.insert(0, str(PROJECT_ROOT / "old" / "scripts" / "legacy" / "demos"))
    from court_demo import CourtDetector
    COURT_DETECTION_AVAILABLE = True
except (ImportError, ModuleNotFoundError):
    try:
        # Try importing from core module
        sys.path.insert(0, str(PROJECT_ROOT / "old" / "src" / "core"))
        from tennis_CV import CourtDetector
        COURT_DETECTION_AVAILABLE = True
    except (ImportError, ModuleNotFoundError):
        # Court detection not available - that's okay, it's optional
        CourtDetector = None


def hex_to_bgr(hex_color: str) -> Tuple[int, int, int]:
    """Convert hex color to BGR tuple for OpenCV."""
    hex_color = hex_color.lstrip('#')
    r, g, b = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    return (b, g, r)  # BGR format


def process_video(
    input_path: Path,
    output_path: Path,
    ball_prompt: str = "tennis ball",
    frame_skip: int = 30,  # Default to every 30th frame for speed (full mesh is VERY slow!)
    fps: float = 30.0,
    player_color: str = "#50C878",
    ball_color: str = "#50C878",
    trail_length: int = 30,
    keypoints_only: bool = False,  # Default to full mesh (user preference)
    device: Optional[str] = None,
    enable_court_detection: bool = False,  # Disable by default for speed
    court_model_path: Optional[Path] = None,
    use_ensemble_ball: bool = False,  # Default to SAM3 only for speed (can enable ensemble)
    process_resolution: Optional[int] = 720,  # Default to 720px width for speed (full mesh is VERY slow!)
):
    """
    Process video with both player tracking and ball detection.
    
    Args:
        input_path: Path to input video
        output_path: Path to output video
        ball_prompt: Text prompt for SAM3 ball detection
        frame_skip: Process every Nth frame
        fps: Output video FPS
        player_color: Hex color for player skeletons
        ball_color: Hex color for ball and trajectory
        trail_length: Number of frames in ball trajectory trail
        keypoints_only: Use fast keypoints-only mode
        device: Device to use ('cuda', 'mps', 'cpu', or None for auto)
    """
    input_path = Path(input_path)
    output_path = Path(output_path)
    
    if not input_path.exists():
        print(f"Error: Input video not found at {input_path}")
        return
    
    # Auto-detect device
    if device is None or device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
            print("Using CUDA")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
            print("Using Apple Silicon MPS")
        else:
            device = "cpu"
            print("Using CPU")
    else:
        print(f"Using device: {device}")
    
    # Initialize models
    print("\n" + "="*60)
    print("Initializing Models")
    print("="*60)
    
    # Setup SAM-3d-body with YOLO human detector
    print("\n1. Setting up SAM-3d-body...")
    
    # Try to use YOLO human detector if available
    human_detector = None
    if YOLO_DETECTOR_AVAILABLE:
        yolo_model_path = PROJECT_ROOT / "old" / "models" / "player" / "playersnball5.pt"
        if yolo_model_path.exists():
            try:
                human_detector = YOLOHumanDetector(model_path=yolo_model_path, device=device)
                print("✓ YOLO human detector loaded (playersnball5.pt)")
            except Exception as e:
                print(f"⚠ Could not load YOLO detector: {e}")
                human_detector = None
    
    # Load SAM-3d-body model
    try:
        model, model_cfg = load_sam_3d_body_hf("facebook/sam-3d-body-dinov3", device=device)
        
        # Load FOV estimator
        try:
            from tools.build_fov_estimator import FOVEstimator
            fov_estimator = FOVEstimator(name="moge2", device=device)
        except Exception:
            fov_estimator = None
        
        # Create estimator with YOLO detector
        estimator = SAM3DBodyEstimator(
            sam_3d_body_model=model,
            model_cfg=model_cfg,
            human_detector=human_detector,
            human_segmentor=None,  # Not needed for keypoints-only or full mesh
            fov_estimator=fov_estimator,
        )
        
        if human_detector:
            print("✓ SAM-3d-body ready (with YOLO human detector)")
        else:
            print("✓ SAM-3d-body ready (without detector, will process full image)")
    except Exception as e:
        print(f"⚠ Error setting up SAM-3d-body: {e}")
        # Fallback to standard setup
        try:
            estimator = setup_sam_3d_body(
                hf_repo_id="facebook/sam-3d-body-dinov3",
                device=device,
                detector_name=None
            )
            print("✓ SAM-3d-body ready (fallback setup)")
        except Exception as e2:
            print(f"✗ Failed to setup SAM-3d-body: {e2}")
            return
    
    # Setup ball detector
    print("\n2. Setting up ball detector...")
    if use_ensemble_ball and ENSEMBLE_DETECTOR_AVAILABLE:
        ball_detector = EnsembleBallDetector(device=device)
        print("✓ Ensemble ball detector ready (combines multiple models)")
    elif SAM3_DETECTOR_AVAILABLE:
        # Use SAM3 only (faster)
        sam3_model_path = PROJECT_ROOT / "SAM3"
        if not sam3_model_path.exists():
            print(f"Error: SAM3 model not found at {sam3_model_path}")
            return
        
        ball_detector = SAM3BallDetector(
            model_path=sam3_model_path,
            device=device,
            use_transformers=True
        )
        print("✓ SAM3 ball detector ready (single model, faster)")
    else:
        print("Error: No ball detector available")
        return
    
    # Initialize court detector (optional)
    court_detector = None
    if enable_court_detection and COURT_DETECTION_AVAILABLE and CourtDetector:
        print("\n3. Setting up court detector...")
        # Try to find court model
        if court_model_path is None:
            # Try default locations
            possible_paths = [
                PROJECT_ROOT / "old" / "models" / "court" / "model_tennis_court_det.pt",
                PROJECT_ROOT / "models" / "court" / "model_tennis_court_det.pt",
            ]
            for path in possible_paths:
                if path.exists():
                    court_model_path = path
                    break
        
        if court_model_path and court_model_path.exists():
            try:
                # Create a minimal config for court detector
                court_config = {
                    'input_width': 640,
                    'input_height': 360,
                    'low_threshold': 170,
                    'min_radius': 10,
                    'max_radius': 25,
                    'use_refine_kps': True,
                    'use_homography': True
                }
                court_detector = CourtDetector(
                    model_path=str(court_model_path),
                    config=court_config
                )
                if court_detector.model is not None:
                    print("✓ Court detector ready")
                else:
                    print("⚠ Court detector model failed to load (will skip court detection)")
                    court_detector = None
            except Exception as e:
                print(f"⚠ Court detector initialization failed: {e}")
                print("  Continuing without court detection...")
                court_detector = None
        else:
            print("⚠ Court detection model not found (will skip court detection)")
            print(f"  Searched in: {possible_paths if court_model_path is None else [court_model_path]}")
    else:
        if not enable_court_detection:
            print("\n3. Court detection disabled by user")
        else:
            print("\n3. Court detection not available")
    
    # Initialize visualizer
    print("\n4. Setting up visualizer...")
    emerald_green_bgr = hex_to_bgr("#50C878")
    visualizer = HeroVideoVisualizer(
        player_color=hex_to_bgr(player_color),
        ball_color=hex_to_bgr(ball_color),
        court_color=emerald_green_bgr,  # Always use emerald green for court
        trail_length=trail_length
    )
    print("✓ Visualizer ready")
    
    # Open video
    print(f"\n5. Opening video: {input_path}")
    cap = cv2.VideoCapture(str(input_path))
    
    if not cap.isOpened():
        print(f"Error: Could not open video {input_path}")
        return
    
    # Get video properties
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Downscale if requested for speed
    scale_factor = 1.0
    if process_resolution and width > process_resolution:
        scale_factor = process_resolution / width
        process_width = process_resolution
        process_height = int(height * scale_factor)
        print(f"   ⚡ Processing at reduced resolution: {process_width}x{process_height} (scale: {scale_factor:.2f})")
    else:
        process_width = width
        process_height = height
    
    frames_to_process = (total_frames + frame_skip - 1) // frame_skip
    # Estimate time: keypoints-only ~0.5s/frame, full mesh ~60-300s/frame (VERY slow!)
    # Full mesh rendering is extremely slow, especially at higher resolutions
    # Based on actual observation: ~260s/frame at 1280px, so adjust estimates
    if keypoints_only:
        time_per_frame = 0.5
    else:
        # Full mesh: MUCH slower, especially at higher resolution
        # Actual observed: ~260s/frame at 1280px, so:
        # At 720px: ~120-180s/frame, at 960px: ~180-240s/frame, at 1280px: ~240-300s/frame, at full res: ~300-360s/frame
        if scale_factor < 0.5:  # Very downscaled (720px or less)
            time_per_frame = 120.0  # ~2 minutes per frame
        elif scale_factor < 0.6:  # Moderately downscaled (960px)
            time_per_frame = 180.0  # ~3 minutes per frame
        elif scale_factor < 0.8:  # Slightly downscaled (1280px)
            time_per_frame = 260.0  # ~4.3 minutes per frame (observed)
        else:
            time_per_frame = 300.0  # ~5 minutes per frame at full resolution
    
    if use_ensemble_ball:
        time_per_frame += 15.0  # Ensemble adds significant time per frame
    
    if enable_court_detection:
        time_per_frame += 10.0  # Court detection adds time (but we skip it most frames)
    
    estimated_time_minutes = (frames_to_process * time_per_frame) / 60
    estimated_time_hours = estimated_time_minutes / 60
    
    print(f"   Original resolution: {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")
    print(f"   Processing resolution: {process_width}x{process_height}")
    print(f"   Original FPS: {original_fps:.2f}")
    print(f"   Total frames: {total_frames}")
    print(f"   Processing every {frame_skip} frame(s) = {frames_to_process} frames")
    if estimated_time_hours >= 1.0:
        print(f"   Estimated time: ~{estimated_time_hours:.1f} hours ({estimated_time_minutes:.0f} minutes)")
    else:
        print(f"   Estimated time: ~{estimated_time_minutes:.1f} minutes")
    
    if not keypoints_only:
        print(f"   ✓ Using full 3D mesh rendering (slower but highest quality)")
    if frame_skip == 1:
        print(f"   ⚠ WARNING: Processing every frame. Use --frame-skip 5 or higher for much faster processing")
    if scale_factor >= 1.0 and not keypoints_only:
        print(f"   ⚠ TIP: Use --process-resolution 1280 to speed up mesh rendering significantly")
    
    # Setup output video writer (use original resolution for output)
    output_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    output_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (output_width, output_height))
    
    # Initialize skeleton visualizer for keypoints
    if keypoints_only:
        # Convert emerald green from BGR to RGB for skeleton visualizer
        emerald_green_rgb = tuple(reversed(hex_to_bgr("#50C878")))  # Always use emerald green
        skeleton_visualizer = SkeletonVisualizer(
            line_width=2,
            radius=4,
            kpt_color=emerald_green_rgb,
            link_color=emerald_green_rgb
        )
        skeleton_visualizer.set_pose_meta(mhr70_pose_info)
    else:
        skeleton_visualizer = None
    
    # Ball trajectory tracking
    ball_trajectory: List[Tuple[int, int]] = []
    
    # Process frames
    print("\n6. Processing frames...")
    print("="*60)
    
    frame_count = 0
    processed_count = 0
    
    with tqdm(total=frames_to_process, desc="Processing") as pbar:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Skip frames if needed
            if frame_count % frame_skip != 0:
                frame_count += 1
                pbar.update(1)
                continue
            
            # Downscale frame if requested
            if scale_factor < 1.0:
                frame = cv2.resize(frame, (process_width, process_height), interpolation=cv2.INTER_AREA)
            
            # Process with SAM-3d-body
            try:
                # Convert BGR to RGB for SAM-3d-body
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Run inference
                outputs = estimator.process_one_image(
                    frame_rgb,
                    inference_type="full" if not keypoints_only else "keypoints_only"
                )
            except Exception as e:
                print(f"\nWarning: SAM-3d-body processing failed on frame {frame_count}: {e}")
                outputs = []
            
            # Process with ball detector
            ball_detection = None
            try:
                if use_ensemble_ball and ENSEMBLE_DETECTOR_AVAILABLE:
                    ball_detection = ball_detector.detect_ball(frame, text_prompt=ball_prompt)
                elif hasattr(ball_detector, 'detect_ball'):
                    # SAM3 or other detector
                    if hasattr(ball_detector.detect_ball, '__code__'):
                        # Check if it accepts threshold parameter
                        import inspect
                        sig = inspect.signature(ball_detector.detect_ball)
                        if 'threshold' in sig.parameters:
                            ball_detection = ball_detector.detect_ball(
                                frame,
                                text_prompt=ball_prompt,
                                threshold=0.3
                            )
                        else:
                            ball_detection = ball_detector.detect_ball(frame, text_prompt=ball_prompt)
                    else:
                        ball_detection = ball_detector.detect_ball(frame, text_prompt=ball_prompt)
            except Exception as e:
                # Silently fail - don't spam warnings
                pass
            
            # Process with court detector (optional, skip every few frames for speed)
            court_keypoints = None
            if enable_court_detection and court_detector and court_detector.model is not None:
                # Only run court detection every 10th processed frame (saves significant time)
                if processed_count % 10 == 0:
                    try:
                        court_keypoints = court_detector.detect_court_in_frame(frame)
                        # Filter out None points
                        if court_keypoints:
                            court_keypoints = [
                                (kp[0], kp[1]) if kp and kp[0] is not None and kp[1] is not None else None
                                for kp in court_keypoints
                            ]
                    except Exception as e:
                        # Silently fail - court detection is optional
                        court_keypoints = None
                else:
                    # Reuse previous court detection (courts don't move much)
                    pass  # Will use None, which is fine
            
            # Update ball trajectory
            if ball_detection:
                center, confidence, mask = ball_detection
                ball_trajectory.append(center)
                # Keep only recent trajectory points
                if len(ball_trajectory) > trail_length:
                    ball_trajectory.pop(0)
            else:
                # Keep trajectory even if ball not detected (fade out)
                if len(ball_trajectory) > 0:
                    ball_trajectory.pop(0)
            
            # Create visualization
            vis_frame = visualizer.create_frame(
                frame=frame,
                player_outputs=outputs,
                ball_detection=ball_detection,
                ball_trajectory=ball_trajectory,
                skeleton_visualizer=skeleton_visualizer if keypoints_only else None,
                court_keypoints=court_keypoints
            )
            
            # Upscale back to original resolution if we downscaled
            if scale_factor < 1.0:
                vis_frame = cv2.resize(vis_frame, (output_width, output_height), interpolation=cv2.INTER_LINEAR)
            
            # Write frame
            out.write(vis_frame)
            
            processed_count += 1
            frame_count += 1
            pbar.update(1)
    
    # Cleanup
    cap.release()
    out.release()
    
    print("\n" + "="*60)
    print("Processing Complete!")
    print("="*60)
    print(f"Processed {processed_count} frames (every {frame_skip} frame(s))")
    print(f"Output saved to: {output_path}")
    print(f"Output resolution: {output_width}x{output_height}")
    print(f"Output FPS: {fps}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate hero video with player and ball tracking"
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to input video file"
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to output video file"
    )
    parser.add_argument(
        "--ball-prompt",
        type=str,
        default="tennis ball",
        help="Text prompt for SAM3 ball detection (default: 'tennis ball')"
    )
    parser.add_argument(
        "--frame-skip",
        type=int,
        default=5,
        help="Process every Nth frame (default: 5 = every 5th frame for speed)"
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=30.0,
        help="Output video FPS (default: 30.0)"
    )
    parser.add_argument(
        "--player-color",
        type=str,
        default="#50C878",
        help="Hex color for player skeletons (default: #50C878 - emerald green)"
    )
    parser.add_argument(
        "--ball-color",
        type=str,
        default="#50C878",
        help="Hex color for ball and trajectory (default: #50C878 - emerald green)"
    )
    parser.add_argument(
        "--trail-length",
        type=int,
        default=30,
        help="Number of frames in ball trajectory trail (default: 30)"
    )
    parser.add_argument(
        "--keypoints-only",
        action="store_true",
        help="Use fast keypoints-only mode (no 3D mesh rendering) - 10x+ faster but no mesh"
    )
    parser.add_argument(
        "--ensemble-ball",
        action="store_true",
        help="Use ensemble ball detector (all models) - slower but higher quality. Default: SAM3 only (faster)."
    )
    parser.add_argument(
        "--process-resolution",
        type=int,
        default=720,
        help="Downscale video to this width for processing (default: 720 for full mesh - it's VERY slow!). Higher = slower but better quality. Use 0 for original resolution."
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "mps", "cpu"],
        help="Device to use (default: auto-detect)"
    )
    parser.add_argument(
        "--no-court-detection",
        action="store_true",
        help="Disable court detection (skip if model/dependencies unavailable)"
    )
    parser.add_argument(
        "--court-model",
        type=str,
        default=None,
        help="Path to court detection model file (default: auto-detect)"
    )
    
    args = parser.parse_args()
    
    process_video(
        input_path=Path(args.input),
        output_path=Path(args.output),
        ball_prompt=args.ball_prompt,
        frame_skip=args.frame_skip,
        fps=args.fps,
        player_color=args.player_color,
        ball_color=args.ball_color,
        trail_length=args.trail_length,
        keypoints_only=args.keypoints_only,
        device=None if args.device == "auto" else args.device,
        enable_court_detection=not args.no_court_detection,
        court_model_path=Path(args.court_model) if args.court_model else None,
        use_ensemble_ball=args.ensemble_ball,
        process_resolution=args.process_resolution if args.process_resolution > 0 else None,
    )


if __name__ == "__main__":
    main()
