#!/usr/bin/env python3
"""
Standalone test script for SAM 3 ball detection on tennis videos.

This script tests SAM 3 individually before integrating into the comparison framework.
It processes tennis videos and generates annotated output videos showing SAM 3 detections.

Usage:
    python SAM3/test_sam3_ball_detection.py --video path/to/video.mp4 --prompt "tennis ball"
"""

import argparse
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch
from PIL import Image

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

# Try to import SAM 3 from transformers
TRANSFORMERS_AVAILABLE = False
SAM3_PACKAGE_AVAILABLE = False

try:
    from transformers import Sam3Model, Sam3Processor
    TRANSFORMERS_AVAILABLE = True
except ImportError as e:
    TRANSFORMERS_AVAILABLE = False
    if "Sam3Model" in str(e) or "Sam3Processor" in str(e):
        print("⚠️  SAM 3 not available in current transformers version.")
        print("   Current transformers version may be too old (need 5.0+).")
        print("   To upgrade, run in your terminal:")
        print("   pip install --upgrade git+https://github.com/huggingface/transformers")
        print("   (This may take several minutes to clone and build)")
    else:
        print(f"⚠️  Transformers import error: {e}")

# Try sam3 package as fallback
if not TRANSFORMERS_AVAILABLE:
    try:
        from sam3.model_builder import build_sam3_image_model
        from sam3.model.sam3_image_processor import Sam3Processor as Sam3ProcessorOriginal
        SAM3_PACKAGE_AVAILABLE = True
    except ImportError as e:
        SAM3_PACKAGE_AVAILABLE = False
        if "sam3.sam" in str(e):
            print("⚠️  SAM3 package has missing dependencies.")
            print("   The sam3 package requires additional modules that aren't included.")
            print("   Recommendation: Use transformers API instead.")
        else:
            print(f"⚠️  SAM3 package import error: {e}")

# Check if we have at least one working option
if not TRANSFORMERS_AVAILABLE and not SAM3_PACKAGE_AVAILABLE:
    print("\n❌ Error: No working SAM 3 implementation found.")
    print("\nTo fix this, choose one of the following:")
    print("\nOption 1 (Recommended): Upgrade transformers")
    print("  Run in your terminal (may take 5-10 minutes):")
    print("  pip install --upgrade git+https://github.com/huggingface/transformers")
    print("\nOption 2: Use --use-sam3-package flag (if package issues are resolved)")
    print("\nExiting...")
    sys.exit(1)

SAM3_MODEL_PATH = PROJECT_ROOT / "SAM3"
OUTPUT_BASE = PROJECT_ROOT / "outputs" / "videos" / "sam3_ball_trials"


class SAM3BallDetector:
    """SAM 3 ball detector using text prompts."""

    def __init__(self, model_path: Path, device: Optional[str] = None, use_transformers: bool = True):
        """
        Initialize SAM 3 ball detector.

        Args:
            model_path: Path to SAM 3 model directory
            device: Device to use ('cuda', 'cpu', 'mps', or None for auto-detect)
            use_transformers: Whether to use transformers API (True) or original sam3 package (False)
        """
        self.model_path = model_path
        self.use_transformers = use_transformers

        # Auto-detect device
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        self.device = device

        print(f"Loading SAM 3 model from {model_path}...")
        print(f"Using device: {device}")

        if use_transformers and TRANSFORMERS_AVAILABLE:
            self._load_transformers_model()
        elif SAM3_PACKAGE_AVAILABLE:
            self._load_sam3_package_model()
        else:
            raise RuntimeError("No SAM 3 implementation available")

        print("✓ SAM 3 model loaded successfully")

    def _load_transformers_model(self):
        """Load SAM 3 using transformers library."""
        try:
            self.model = Sam3Model.from_pretrained(str(self.model_path)).to(self.device)
            self.processor = Sam3Processor.from_pretrained(str(self.model_path))
            self.model.eval()
            print("✓ Loaded using transformers API")
        except Exception as e:
            raise RuntimeError(f"Failed to load SAM 3 model with transformers: {e}") from e

    def _load_sam3_package_model(self):
        """Load SAM 3 using original sam3 package."""
        try:
            self.model = build_sam3_image_model()
            # Load checkpoint
            checkpoint = torch.load(self.model_path / "sam3.pt", map_location=self.device, weights_only=False)
            if "model" in checkpoint:
                self.model.load_state_dict(checkpoint["model"], strict=False)
            else:
                self.model.load_state_dict(checkpoint, strict=False)
            self.model.to(self.device)
            self.model.eval()
            self.processor = Sam3ProcessorOriginal(self.model)
            print("✓ Loaded using sam3 package API")
        except Exception as e:
            raise RuntimeError(f"Failed to load SAM 3 model with sam3 package: {e}") from e

    def detect_ball(
        self, frame: np.ndarray, text_prompt: str = "tennis ball", threshold: float = 0.5
    ) -> Optional[Tuple[Tuple[int, int], float, np.ndarray]]:
        """
        Detect tennis ball in a frame using SAM 3.

        Args:
            frame: Input frame as BGR numpy array
            text_prompt: Text prompt for detection (e.g., "tennis ball", "ball")
            threshold: Confidence threshold for detections

        Returns:
            Tuple of (center, confidence, mask) or None if no detection
            - center: (x, y) tuple of ball center
            - confidence: Detection confidence score
            - mask: Binary mask of detected ball
        """
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(frame_rgb)

        try:
            if self.use_transformers and TRANSFORMERS_AVAILABLE:
                return self._detect_with_transformers(pil_image, text_prompt, threshold)
            else:
                return self._detect_with_sam3_package(pil_image, text_prompt, threshold)
        except Exception as e:
            print(f"Error during detection: {e}")
            return None

    def _detect_with_transformers(
        self, image: Image.Image, text_prompt: str, threshold: float
    ) -> Optional[Tuple[Tuple[int, int], float, np.ndarray]]:
        """Detect using transformers API."""
        # Process image and text prompt
        inputs = self.processor(images=image, text=text_prompt, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)

        # Post-process results
        results = self.processor.post_process_instance_segmentation(
            outputs,
            threshold=threshold,
            mask_threshold=0.5,
            target_sizes=inputs.get("original_sizes").tolist(),
        )[0]

        if len(results["masks"]) == 0:
            return None

        # Get the best detection (highest score)
        best_idx = results["scores"].argmax().item()
        mask = results["masks"][best_idx].cpu().numpy().astype(np.uint8)
        score = float(results["scores"][best_idx].item())
        box = results["boxes"][best_idx].cpu().numpy()

        # Calculate center from bounding box
        x1, y1, x2, y2 = box
        center = (int((x1 + x2) / 2), int((y1 + y2) / 2))

        return (center, score, mask)

    def _detect_with_sam3_package(
        self, image: Image.Image, text_prompt: str, threshold: float
    ) -> Optional[Tuple[Tuple[int, int], float, np.ndarray]]:
        """Detect using original sam3 package API."""
        # Set image
        inference_state = self.processor.set_image(image)

        # Set text prompt
        output = self.processor.set_text_prompt(state=inference_state, prompt=text_prompt)

        # Get masks, boxes, and scores
        masks = output["masks"]
        boxes = output["boxes"]
        scores = output["scores"]

        if len(masks) == 0:
            return None

        # Get the best detection (highest score)
        best_idx = scores.argmax().item()
        mask = masks[best_idx].cpu().numpy().astype(np.uint8)
        score = float(scores[best_idx].item())
        box = boxes[best_idx].cpu().numpy()

        if score < threshold:
            return None

        # Calculate center from bounding box
        x1, y1, x2, y2 = box
        center = (int((x1 + x2) / 2), int((y1 + y2) / 2))

        return (center, score, mask)


def annotate_frame(
    frame: np.ndarray,
    detection: Optional[Tuple[Tuple[int, int], float, np.ndarray]],
    prompt: str,
    show_mask: bool = True,
) -> np.ndarray:
    """
    Annotate frame with SAM 3 detection results.

    Args:
        frame: Input frame
        detection: Detection result (center, confidence, mask) or None
        prompt: Text prompt used
        show_mask: Whether to overlay the mask

    Returns:
        Annotated frame
    """
    annotated = frame.copy()

    # Add prompt label
    cv2.putText(
        annotated,
        f"SAM3: {prompt}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )

    if detection is None:
        cv2.putText(
            annotated,
            "No detection",
            (20, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )
        return annotated

    center, confidence, mask = detection

    # Overlay mask if requested
    if show_mask:
        # Create colored mask overlay
        mask_colored = np.zeros_like(frame)
        mask_colored[:, :, 1] = mask * 255  # Green channel
        mask_colored[:, :, 2] = mask * 128  # Red channel for visibility
        annotated = cv2.addWeighted(annotated, 0.7, mask_colored, 0.3, 0)

    # Draw center point
    cx, cy = center
    cv2.circle(annotated, (cx, cy), radius=15, color=(0, 255, 0), thickness=3)
    cv2.circle(annotated, (cx, cy), radius=3, color=(255, 255, 255), thickness=-1)

    # Draw bounding box from mask
    if mask.sum() > 0:
        coords = np.column_stack(np.where(mask > 0))
        if len(coords) > 0:
            x, y, w, h = cv2.boundingRect(coords[:, ::-1])
            cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Add confidence label
    label = f"Conf: {confidence:.2f}"
    cv2.putText(
        annotated,
        label,
        (cx + 20, cy - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )

    return annotated


def process_video(
    video_path: Path,
    detector: SAM3BallDetector,
    text_prompt: str,
    output_path: Path,
    show_mask: bool = True,
    threshold: float = 0.5,
):
    """
    Process video with SAM 3 ball detection.

    Args:
        video_path: Path to input video
        detector: SAM3BallDetector instance
        text_prompt: Text prompt for detection
        output_path: Path to save output video
        show_mask: Whether to overlay masks
        threshold: Confidence threshold
    """
    print(f"Processing video: {video_path}")
    print(f"Text prompt: '{text_prompt}'")
    print(f"Confidence threshold: {threshold}")
    print(f"Show mask overlay: {show_mask}")

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Video properties: {width}x{height} @ {fps:.2f} fps, {total_frames} frames")

    # Create output directory
    output_path.parent.mkdir(parents=True, exist_ok=True)

    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )

    frame_count = 0
    detection_count = 0

    print("Processing frames...")
    print("=" * 60)
    while True:
        ret, frame = capture.read()
        if not ret:
            break

        frame_count += 1
        
        # Detailed progress output
        print(f"\n[Frame {frame_count}/{total_frames}] Starting frame processing...")
        
        # Detect ball
        print(f"  → Step 1/3: Running SAM 3 detection with prompt '{text_prompt}'...")
        detection = detector.detect_ball(frame, text_prompt=text_prompt, threshold=threshold)
        
        if detection is not None:
            detection_count += 1
            center, confidence, mask = detection
            print(f"  → Step 2/3: ✓ Ball detected! Center: {center}, Confidence: {confidence:.3f}")
        else:
            print(f"  → Step 2/3: ✗ No ball detected (below threshold {threshold})")
        
        # Annotate frame
        print(f"  → Step 3/3: Annotating frame with detection results...")
        annotated = annotate_frame(frame, detection, text_prompt, show_mask=show_mask)

        # Add frame counter
        cv2.putText(
            annotated,
            f"Frame: {frame_count}/{total_frames}",
            (20, height - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        writer.write(annotated)
        
        # Progress summary every 10 frames
        if frame_count % 10 == 0:
            progress_pct = (frame_count / total_frames) * 100
            detection_rate = (detection_count / frame_count) * 100
            print(f"  [Progress: {progress_pct:.1f}%] Detections so far: {detection_count}/{frame_count} ({detection_rate:.1f}%)")

    capture.release()
    writer.release()

    print(f"\n✓ Processing complete!")
    print(f"  Total frames: {frame_count}")
    print(f"  Detections: {detection_count} ({100*detection_count/frame_count:.1f}%)")
    print(f"  Output saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Test SAM 3 ball detection on tennis videos")
    parser.add_argument(
        "--video",
        required=True,
        type=Path,
        help="Path to input video file",
    )
    parser.add_argument(
        "--prompt",
        default="tennis ball",
        type=str,
        help='Text prompt for detection (default: "tennis ball")',
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Path to output video (default: outputs/videos/sam3_ball_trials/<video_name>_<prompt>.mp4)",
    )
    parser.add_argument(
        "--threshold",
        default=0.5,
        type=float,
        help="Confidence threshold (default: 0.5)",
    )
    parser.add_argument(
        "--no-mask",
        action="store_true",
        help="Don't overlay masks on output video",
    )
    parser.add_argument(
        "--use-sam3-package",
        action="store_true",
        help="Use original sam3 package instead of transformers",
    )

    args = parser.parse_args()

    # Validate video path
    if not args.video.exists():
        print(f"Error: Video file not found: {args.video}")
        sys.exit(1)

    # Validate model path
    if not SAM3_MODEL_PATH.exists():
        print(f"Error: SAM 3 model not found at {SAM3_MODEL_PATH}")
        print("Please ensure the SAM3 folder contains the model files.")
        sys.exit(1)

    # Determine output path
    if args.output:
        output_path = args.output
    else:
        OUTPUT_BASE.mkdir(parents=True, exist_ok=True)
        prompt_safe = args.prompt.replace(" ", "_").replace('"', "").replace("'", "")
        output_path = OUTPUT_BASE / f"{args.video.stem}_sam3_{prompt_safe}.mp4"

    # Initialize detector
    print("=" * 60)
    print("SAM 3 Ball Detection Test")
    print("=" * 60)
    try:
        detector = SAM3BallDetector(
            SAM3_MODEL_PATH,
            use_transformers=not args.use_sam3_package,
        )
    except Exception as e:
        print(f"Error initializing SAM 3 detector: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)

    # Process video
    try:
        process_video(
            args.video,
            detector,
            args.prompt,
            output_path,
            show_mask=not args.no_mask,
            threshold=args.threshold,
        )
    except Exception as e:
        print(f"Error processing video: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

