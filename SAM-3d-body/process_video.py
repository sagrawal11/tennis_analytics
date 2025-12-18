#!/usr/bin/env python3
"""
Video processing script for SAM-3d-body.
Extracts frames from a video file, processes each frame with SAM-3d-body,
and saves the results as images and optionally as a video.
"""

import sys
import os
import argparse
from pathlib import Path

# Don't set OpenGL platform - let pyrender use default OpenGL on macOS
# This works better than osmesa which requires special library setup
import platform
if platform.system() == 'Darwin':
    # Clear any osmesa setting - use default OpenGL
    if os.environ.get('PYOPENGL_PLATFORM') == 'osmesa':
        del os.environ['PYOPENGL_PLATFORM']

# Add the cloned sam-3d-body repository to Python path
repo_path = Path(__file__).parent / "sam-3d-body"
if repo_path.exists():
    sys.path.insert(0, str(repo_path))
else:
    print(f"Error: SAM_3D repository not found at {repo_path}")
    print("Please make sure the repository is cloned in the project directory.")
    sys.exit(1)

import cv2
import numpy as np
import torch
from tqdm import tqdm
import urllib.error
import ssl
import json
import gc
from typing import Dict, List, Any

# Try to import SAM_3D utilities
try:
    from notebook.utils import setup_sam_3d_body
    from tools.vis_utils import visualize_sample_together
    from sam_3d_body.visualization.skeleton_visualizer import SkeletonVisualizer
    from sam_3d_body.metadata.mhr70 import pose_info as mhr70_pose_info
except ImportError as e:
    print("Error: Could not import SAM_3D utilities.")
    print(f"Import error: {e}")
    print("Make sure the sam-3d-body repository is cloned and all dependencies are installed.")
    print("\nIf you see OpenGL/EGL errors on macOS, you may need to install osmesa:")
    print("  conda install -c conda-forge mesalib")
    raise

# Initialize skeleton visualizer for keypoints-only mode
_keypoints_visualizer = None
def get_keypoints_visualizer():
    """Get or create the skeleton visualizer for keypoints-only mode"""
    global _keypoints_visualizer
    if _keypoints_visualizer is None:
        _keypoints_visualizer = SkeletonVisualizer(line_width=2, radius=5)
        _keypoints_visualizer.set_pose_meta(mhr70_pose_info)
    return _keypoints_visualizer


def visualize_keypoints_only(img_cv2, outputs):
    """
    Create a 3-panel visualization for keypoints-only mode:
    1. Original frame
    2. Keypoints overlaid on original frame
    3. Keypoints on white background (skeleton only)
    
    This is much faster than full mesh rendering since it skips 3D rendering entirely.
    """
    visualizer = get_keypoints_visualizer()
    
    # Panel 1: Original frame
    img_original = img_cv2.copy()
    
    # Panel 2: Keypoints overlaid on original
    img_keypoints_overlay = img_cv2.copy()
    
    # Panel 3: Keypoints on white background
    img_keypoints_white = np.ones_like(img_cv2) * 255
    
    # Sort outputs by depth (furthest to closest) for proper layering
    if len(outputs) > 1:
        all_depths = np.stack([tmp['pred_cam_t'] for tmp in outputs], axis=0)[:, 2]
        outputs_sorted = [outputs[idx] for idx in np.argsort(-all_depths)]
    else:
        outputs_sorted = outputs
    
    # Draw all keypoints on both overlay and white background
    for person_output in outputs_sorted:
        keypoints_2d = person_output["pred_keypoints_2d"]
        # Add visibility column (all visible for now)
        keypoints_2d_with_vis = np.concatenate(
            [keypoints_2d, np.ones((keypoints_2d.shape[0], 1))], axis=-1
        )
        
        # Draw on overlay
        img_keypoints_overlay = visualizer.draw_skeleton(img_keypoints_overlay, keypoints_2d_with_vis)
        
        # Draw on white background
        img_keypoints_white = visualizer.draw_skeleton(img_keypoints_white, keypoints_2d_with_vis)
    
    # Concatenate the 3 panels horizontally
    combined_img = np.concatenate([img_original, img_keypoints_overlay, img_keypoints_white], axis=1)
    
    return combined_img


def numpy_to_list(obj):
    """Convert numpy arrays and scalars to Python native types for JSON serialization."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, dict):
        return {key: numpy_to_list(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [numpy_to_list(item) for item in obj]
    elif obj is None:
        return None
    else:
        return obj


def extract_skeleton_only(person_output: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract only skeleton/keypoint data for a single detected person.
    
    Returns a dictionary with:
    - skeleton: 3D and 2D keypoints, joint coordinates, rotations
    """
    data = {
        # Skeleton data (keypoints)
        "skeleton": {
            "keypoints_3d": numpy_to_list(person_output.get("pred_keypoints_3d")),  # [70, 3] - 3D keypoints in camera space
            "keypoints_2d": numpy_to_list(person_output.get("pred_keypoints_2d")),  # [70, 2] - 2D keypoints projected to image
            "joint_coords_3d": numpy_to_list(person_output.get("pred_joint_coords")),  # [127, 3] - 3D joint coordinates
            "joint_global_rotations": numpy_to_list(person_output.get("pred_global_rots")),  # [127, 3, 3] - Global joint rotation matrices
        },
    }
    
    return data


def extract_person_data(person_output: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract and structure all data for a single detected person.
    
    Returns a dictionary with:
    - skeleton: 3D and 2D keypoints
    - mesh: 3D vertices
    - pose: body, hand, and global pose parameters
    - shape: body shape parameters
    - camera: camera parameters
    - bbox: bounding boxes
    """
    data = {
        # Skeleton data (keypoints)
        "skeleton": {
            "keypoints_3d": numpy_to_list(person_output.get("pred_keypoints_3d")),  # [70, 3] - 3D keypoints in camera space
            "keypoints_2d": numpy_to_list(person_output.get("pred_keypoints_2d")),  # [70, 2] - 2D keypoints projected to image
            "joint_coords_3d": numpy_to_list(person_output.get("pred_joint_coords")),  # [127, 3] - 3D joint coordinates
            "joint_global_rotations": numpy_to_list(person_output.get("pred_global_rots")),  # [127, 3, 3] - Global joint rotation matrices
        },
        
        # Mesh data (3D vertices)
        "mesh": {
            "vertices": numpy_to_list(person_output.get("pred_vertices")),  # [N, 3] - 3D mesh vertices in camera space
            "num_vertices": len(person_output.get("pred_vertices", [])) if person_output.get("pred_vertices") is not None else 0,
        },
        
        # Pose parameters
        "pose": {
            "global_rotation": numpy_to_list(person_output.get("global_rot")),  # [3] - Global rotation (Euler angles)
            "pose_raw": numpy_to_list(person_output.get("pred_pose_raw")),  # Raw pose parameters
            "body_pose": numpy_to_list(person_output.get("body_pose_params")),  # Body pose parameters
            "hand_pose": numpy_to_list(person_output.get("hand_pose_params")),  # Hand pose parameters (left + right)
            "face_expression": numpy_to_list(person_output.get("expr_params")),  # Face expression parameters
        },
        
        # Shape and scale parameters
        "shape": {
            "shape_params": numpy_to_list(person_output.get("shape_params")),  # Body shape parameters
            "scale_params": numpy_to_list(person_output.get("scale_params")),  # Scale parameters
        },
        
        # Camera parameters
        "camera": {
            "translation": numpy_to_list(person_output.get("pred_cam_t")),  # [3] - Camera translation
            "focal_length": numpy_to_list(person_output.get("focal_length")),  # Estimated focal length
        },
        
        # Bounding boxes
        "bbox": {
            "body": numpy_to_list(person_output.get("bbox")),  # [4] - Body bounding box [x1, y1, x2, y2]
            "left_hand": numpy_to_list(person_output.get("lhand_bbox")) if "lhand_bbox" in person_output else None,
            "right_hand": numpy_to_list(person_output.get("rhand_bbox")) if "rhand_bbox" in person_output else None,
        },
        
        # Additional metadata
        "metadata": {
            "has_mask": person_output.get("mask") is not None,
        }
    }
    
    return data


def process_video(
    video_path,
    output_dir,
    hf_repo_id="facebook/sam-3d-body-dinov3",
    device=None,
    frame_skip=1,
    create_output_video=True,
    output_video_fps=30.0,
    save_json=True,
    skip_visualization=False,
    keypoints_only=False,
    max_output_width=1920
):
    """
    Process a video file with SAM-3d-body.
    
    Args:
        video_path: Path to input video file
        output_dir: Directory to save output images, video, and JSON data
        hf_repo_id: HuggingFace repository ID for the model
        device: Device to use ('mps', 'cuda', 'cpu', or None for auto-detect)
        frame_skip: Process every Nth frame (1 = all frames, 2 = every other frame, etc.)
        create_output_video: Whether to create an output video from processed frames
        output_video_fps: FPS for output video
        save_json: Whether to save skeleton and mesh data to JSON (default: True)
        skip_visualization: Skip visualization/rendering step for faster processing (default: False)
        keypoints_only: Extract only skeleton keypoints data (faster, saves to separate JSON) (default: False)
        max_output_width: Maximum width for output video to prevent memory issues (default: 1920)
    
    Outputs:
        - Processed video (if create_output_video=True and skip_visualization=False)
        - JSON data file (if save_json=True) containing:
          * Skeleton data (3D/2D keypoints, joint coordinates, rotations)
          * Mesh data (3D vertices + face indices for reconstruction)
          * Pose parameters (body, hand, face)
          * Shape parameters
          * Camera parameters
          * Bounding boxes
    """
    video_path = Path(video_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Check if video exists
    if not video_path.exists():
        print(f"Error: Video not found at {video_path}")
        return
    
    # Auto-detect device if not specified
    if device is None:
        if torch.backends.mps.is_available():
            device = "mps"
            print("Using Apple Silicon MPS acceleration")
        elif torch.cuda.is_available():
            device = "cuda"
            print("Using CUDA")
        else:
            device = "cpu"
            print("Using CPU")
    else:
        print(f"Using device: {device}")
    
    # Set up SAM-3d-body estimator
    print("\nSetting up SAM-3d-body estimator...")
    try:
        # Try with detector first, but fall back to no detector if SSL/certificate issues occur
        try:
            estimator = setup_sam_3d_body(hf_repo_id=hf_repo_id, device=device, detector_name="vitdet")
            print("✓ Estimator set up successfully with detector")
        except (urllib.error.URLError, ssl.SSLError, Exception) as e:
            if isinstance(e, (urllib.error.URLError, ssl.SSLError)):
                print(f"⚠ Warning: Could not download detector model (SSL/certificate issue)")
            else:
                print(f"⚠ Warning: Could not load detector ({type(e).__name__})")
            print("  Continuing without detector - will process full image frames")
            estimator = setup_sam_3d_body(hf_repo_id=hf_repo_id, device=device, detector_name=None)
            print("✓ Estimator set up successfully (without detector)")
    except Exception as e:
        print(f"✗ Error setting up estimator: {e}")
        raise
    
    # Open video file
    print(f"\nOpening video: {video_path}")
    cap = cv2.VideoCapture(str(video_path))
    
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"Video properties:")
    print(f"  Resolution: {width}x{height}")
    print(f"  FPS: {fps:.2f}")
    print(f"  Total frames: {total_frames}")
    print(f"  Frame skip: {frame_skip} (processing every {frame_skip} frame(s))")
    if keypoints_only:
        print(f"  Mode: Keypoints only (skeleton data only)")
    
    # Calculate number of frames to process
    frames_to_process = (total_frames + frame_skip - 1) // frame_skip
    print(f"  Frames to process: {frames_to_process}")
    
    # Prepare video writer if creating output video
    video_writer = None
    output_video_path = None
    if create_output_video:
        if keypoints_only:
            output_video_path = output_dir / f"{video_path.stem}_processed_keypoints.mp4"
        else:
            output_video_path = output_dir / f"{video_path.stem}_processed.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        # We'll set the width after we process the first frame (since visualization may change dimensions)
        video_writer = None  # Initialize after first frame
    
    # Prepare JSON data structure if saving JSON
    json_data = None
    output_json_path = None
    if save_json:
        if keypoints_only:
            output_json_path = output_dir / f"{video_path.stem}_keypoints.json"
        else:
            output_json_path = output_dir / f"{video_path.stem}_data.json"
        
        # Get mesh faces from estimator (only needed if not keypoints_only)
        mesh_faces = None
        num_faces = 0
        if not keypoints_only and hasattr(estimator, 'faces'):
            mesh_faces = numpy_to_list(estimator.faces)
            num_faces = len(estimator.faces)
        
        json_data = {
            "video_info": {
                "video_path": str(video_path),
                "resolution": {"width": width, "height": height},
                "fps": float(fps),
                "total_frames": int(total_frames),
                "frames_processed": 0,
                "frame_skip": frame_skip,
                "keypoints_only": keypoints_only,
            },
            "frames": []
        }
        
        # Only include mesh_info if not keypoints_only
        if not keypoints_only:
            json_data["mesh_info"] = {
                "faces": mesh_faces,  # Face indices for mesh reconstruction
                "num_faces": num_faces,
            }
    
    # Process frames
    frame_count = 0
    processed_count = 0
    
    print("\nProcessing frames...")
    with tqdm(total=frames_to_process, desc="Processing") as pbar:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Skip frames according to frame_skip
            if frame_count % frame_skip != 0:
                frame_count += 1
                continue
            
            # Convert BGR to RGB for processing
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process frame with SAM-3d-body
            try:
                outputs = estimator.process_one_image(frame_rgb)
                
                # Extract data for JSON if saving JSON
                if save_json and json_data is not None:
                    frame_data = {
                        "frame_number": int(frame_count),
                        "timestamp": float(frame_count / fps) if fps > 0 else 0.0,
                        "num_persons": len(outputs),
                        "persons": []
                    }
                    
                    for person_idx, person_output in enumerate(outputs):
                        if keypoints_only:
                            person_data = extract_skeleton_only(person_output)
                        else:
                            person_data = extract_person_data(person_output)
                        person_data["person_id"] = person_idx
                        frame_data["persons"].append(person_data)
                    
                    json_data["frames"].append(frame_data)
                
                # Skip visualization if requested (saves significant time)
                rend_img = None
                if not skip_visualization and create_output_video:
                    # For keypoints-only mode, use fast skeleton-only visualization (no 3D mesh rendering)
                    if keypoints_only:
                        try:
                            rend_img = visualize_keypoints_only(frame, outputs)
                        except Exception as e:
                            print(f"\nWarning: Keypoints visualization failed ({e})")
                            print("  Falling back to simple visualization...")
                            rend_img = frame.copy()
                            # Draw keypoints and bboxes
                            for person_output in outputs:
                                # Draw bbox
                                bbox = person_output.get("bbox", None)
                                if bbox is not None:
                                    cv2.rectangle(
                                        rend_img,
                                        (int(bbox[0]), int(bbox[1])),
                                        (int(bbox[2]), int(bbox[3])),
                                        (0, 255, 0),
                                        2,
                                    )
                                # Draw keypoints
                                keypoints_2d = person_output.get("pred_keypoints_2d", None)
                                if keypoints_2d is not None:
                                    for kp in keypoints_2d:
                                        if kp[2] > 0:  # Visibility check
                                            cv2.circle(rend_img, (int(kp[0]), int(kp[1])), 3, (0, 0, 255), -1)
                    else:
                        # Full mesh visualization (slower, uses 3D rendering)
                        vis_error = None
                        
                        # Try different OpenGL platforms if first attempt fails
                        # On macOS, just use default (don't try osmesa as it's not easily available)
                        platforms_to_try = [None]  # None means use default OpenGL
                        
                        original_platform = os.environ.get('PYOPENGL_PLATFORM')
                        for gl_platform in platforms_to_try:
                            try:
                                if gl_platform is not None:
                                    os.environ['PYOPENGL_PLATFORM'] = gl_platform
                                
                                rend_img = visualize_sample_together(frame, outputs, estimator.faces)
                                break  # Success!
                            except Exception as e:
                                vis_error = e
                                # Restore original platform setting before next try
                                if original_platform is not None:
                                    os.environ['PYOPENGL_PLATFORM'] = original_platform
                                elif 'PYOPENGL_PLATFORM' in os.environ:
                                    del os.environ['PYOPENGL_PLATFORM']
                                continue
                        
                        # If all OpenGL attempts failed, create a simple visualization
                        if rend_img is None:
                            print(f"\nWarning: OpenGL visualization failed ({vis_error})")
                            print("  Attempting simple keypoint visualization...")
                            print("  To enable full 3D rendering, you may need to install osmesa:")
                            print("    conda install -c conda-forge mesalib")
                            rend_img = frame.copy()
                            # Draw keypoints and bboxes
                            for person_output in outputs:
                                # Draw bbox
                                bbox = person_output.get("bbox", None)
                                if bbox is not None:
                                    cv2.rectangle(
                                        rend_img,
                                        (int(bbox[0]), int(bbox[1])),
                                        (int(bbox[2]), int(bbox[3])),
                                        (0, 255, 0),
                                        2,
                                    )
                                # Draw keypoints
                                keypoints_2d = person_output.get("pred_keypoints_2d", None)
                                if keypoints_2d is not None:
                                    for kp in keypoints_2d:
                                        if kp[2] > 0:  # Visibility check
                                            cv2.circle(rend_img, (int(kp[0]), int(kp[1])), 3, (0, 0, 255), -1)
                
                # Downscale if output is too wide to prevent memory crashes
                if rend_img is not None:
                    out_height, out_width = rend_img.shape[:2]
                    if out_width > max_output_width:
                        scale_factor = max_output_width / out_width
                        new_width = max_output_width
                        new_height = int(out_height * scale_factor)
                        rend_img = cv2.resize(rend_img, (new_width, new_height), interpolation=cv2.INTER_AREA)
                        if processed_count == 0:
                            print(f"\n⚠ Warning: Output resolution {out_width}x{out_height} exceeds max width {max_output_width}")
                            print(f"  Downscaling to {new_width}x{new_height} to prevent memory issues")
                
                # Initialize video writer on first processed frame (only if we have visualization)
                if create_output_video and video_writer is None and rend_img is not None:
                    out_height, out_width = rend_img.shape[:2]
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    video_writer = cv2.VideoWriter(
                        str(output_video_path),
                        fourcc,
                        output_video_fps,
                        (out_width, out_height)
                    )
                    print(f"\nOutput video will be saved to: {output_video_path}")
                    print(f"Output video resolution: {out_width}x{out_height}")
                
                # Add frame to video if creating output video and we have visualization
                if video_writer is not None and rend_img is not None:
                    video_writer.write(rend_img.astype(np.uint8))
                
                processed_count += 1
                
                # Print detection info for first few frames
                if processed_count <= 3:
                    if isinstance(outputs, list) and len(outputs) > 0:
                        print(f"\nFrame {frame_count}: Detected {len(outputs)} person(s)")
                    else:
                        print(f"\nFrame {frame_count}: No persons detected")
                
                # Memory cleanup after each frame (more aggressive for keypoints-only to prevent crashes)
                if rend_img is not None:
                    del rend_img
                del outputs
                del frame_rgb
                
                # Clear GPU/MPS cache more frequently to prevent memory buildup
                cleanup_interval = 3 if keypoints_only else 10  # More frequent cleanup for keypoints mode
                if processed_count % cleanup_interval == 0:
                    try:
                        if device == "cuda":
                            torch.cuda.empty_cache()
                        # MPS doesn't have empty_cache, but we can still force GC
                    except:
                        pass
                    gc.collect()  # Force Python garbage collection
                
            except Exception as e:
                print(f"\nError processing frame {frame_count}: {e}")
                # Add error to JSON if saving
                if save_json and json_data is not None:
                    frame_data = {
                        "frame_number": int(frame_count),
                        "timestamp": float(frame_count / fps) if fps > 0 else 0.0,
                        "error": str(e),
                        "num_persons": 0,
                        "persons": []
                    }
                    json_data["frames"].append(frame_data)
                # Continue with next frame
                pass
            
            # Clean up frame even if processing failed
            if 'frame_rgb' in locals():
                del frame_rgb
            if 'rend_img' in locals():
                del rend_img
            if 'outputs' in locals():
                del outputs
            
            frame_count += 1
            pbar.update(1)
    
    # Final memory cleanup
    gc.collect()
    try:
        if device == "cuda":
            torch.cuda.empty_cache()
    except:
        pass
    
    # Release resources
    cap.release()
    if video_writer is not None:
        video_writer.release()
        print(f"\n✓ Output video saved to: {output_video_path}")
    
    # Save JSON data if requested
    if save_json and json_data is not None:
        json_data["video_info"]["frames_processed"] = processed_count
        
        print(f"\nSaving data to JSON: {output_json_path}")
        with open(output_json_path, 'w') as f:
            json.dump(json_data, f, indent=2)
        
        # Create summary
        summary = {
            "total_frames": processed_count,
            "total_persons_detected": sum(len(frame["persons"]) for frame in json_data["frames"]),
            "frames_with_persons": sum(1 for frame in json_data["frames"] if frame["num_persons"] > 0),
            "frames_without_persons": sum(1 for frame in json_data["frames"] if frame["num_persons"] == 0),
        }
        
        summary_path = output_dir / f"{video_path.stem}_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"✓ JSON data saved to: {output_json_path}")
        print(f"✓ Summary saved to: {summary_path}")
    
    # Print summary
    print("\n" + "="*50)
    print("Processing Summary:")
    print("="*50)
    print(f"Total frames in video: {total_frames}")
    print(f"Frames processed: {processed_count}")
    if create_output_video and output_video_path:
        print(f"Output video saved to: {output_video_path}")
    if save_json and output_json_path:
        print(f"JSON data saved to: {output_json_path}")
        if json_data:
            total_persons = sum(len(frame["persons"]) for frame in json_data["frames"])
            print(f"Total persons detected: {total_persons}")
    print("\n✓ Processing complete!")


def main():
    parser = argparse.ArgumentParser(
        description='Process video with SAM-3d-body',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process all frames
  python process_video.py --video IMG_8169.MOV
  
  # Process every 5th frame (faster)
  python process_video.py --video IMG_8169.MOV --frame_skip 5
  
  # Process without creating output video
  python process_video.py --video IMG_8169.MOV --no-video
  
  # Skip visualization for faster processing (only saves JSON data)
  python process_video.py --video IMG_8169.MOV --skip-visualization
  
  # Fastest: Skip visualization and video, only JSON data
  python process_video.py --video IMG_8169.MOV --skip-visualization --no-video
  
  # Extract only skeleton keypoints (MUCH faster - no 3D mesh rendering, saves to _keypoints.json)
  # Creates 3-panel video: original | keypoints overlay | keypoints on white background
  python process_video.py --video IMG_8169.MOV --keypoints-only
        """
    )
    parser.add_argument(
        '--video',
        type=str,
        default='IMG_8169.MOV',
        help='Path to input video file (default: IMG_8169.MOV)'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='./outputs',
        help='Directory to save output images and video (default: ./outputs)'
    )
    parser.add_argument(
        '--hf_repo_id',
        type=str,
        default='facebook/sam-3d-body-dinov3',
        help='HuggingFace repository ID (default: facebook/sam-3d-body-dinov3)'
    )
    parser.add_argument(
        '--device',
        type=str,
        default=None,
        choices=['mps', 'cuda', 'cpu'],
        help='Device to use (mps, cuda, or cpu). Auto-detect if not specified.'
    )
    parser.add_argument(
        '--frame_skip',
        type=int,
        default=1,
        help='Process every Nth frame (1 = all frames, 2 = every other frame, etc.) (default: 1)'
    )
    parser.add_argument(
        '--no-video',
        action='store_true',
        help='Do not create output video, only save individual frame images'
    )
    parser.add_argument(
        '--output_fps',
        type=float,
        default=30.0,
        help='FPS for output video (default: 30.0)'
    )
    parser.add_argument(
        '--no-json',
        action='store_true',
        help='Do not save JSON data file (default: JSON is saved)'
    )
    parser.add_argument(
        '--skip-visualization',
        action='store_true',
        help='Skip visualization/rendering step (faster, only saves JSON data)'
    )
    parser.add_argument(
        '--keypoints-only',
        action='store_true',
        help='Extract only skeleton keypoints data (faster, saves to separate JSON file, uses fast 3-panel visualization)'
    )
    parser.add_argument(
        '--max-output-width',
        type=int,
        default=1920,
        help='Maximum width for output video to prevent memory crashes (default: 1920). Output will be downscaled if wider.'
    )
    
    args = parser.parse_args()
    
    process_video(
        video_path=args.video,
        output_dir=args.output_dir,
        hf_repo_id=args.hf_repo_id,
        device=args.device,
        frame_skip=args.frame_skip,
        create_output_video=not args.no_video,
        output_video_fps=args.output_fps,
        save_json=not args.no_json,
        skip_visualization=args.skip_visualization,
        keypoints_only=args.keypoints_only,
        max_output_width=args.max_output_width
    )


if __name__ == "__main__":
    main()

