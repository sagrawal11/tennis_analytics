#!/usr/bin/env python3
"""
Test SAM 3 with multiple text prompt variations on tennis videos.

This script runs the SAM 3 ball detection test with different prompts
to compare which works best for tennis ball detection.

Usage:
    python SAM3/test_multiple_prompts.py --video path/to/video.mp4
"""

import argparse
import subprocess
import sys
from pathlib import Path

PROMPTS = [
    "tennis ball",
    "ball",
    "yellow ball",
    "small yellow ball",
    "tennis",
]


def main():
    parser = argparse.ArgumentParser(
        description="Test SAM 3 with multiple prompt variations"
    )
    parser.add_argument(
        "--video",
        required=True,
        type=Path,
        help="Path to input video file",
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

    args = parser.parse_args()

    if not args.video.exists():
        print(f"Error: Video file not found: {args.video}")
        sys.exit(1)

    script_path = Path(__file__).parent / "test_sam3_ball_detection.py"

    print("=" * 60)
    print("Testing SAM 3 with Multiple Prompts")
    print("=" * 60)
    print(f"Video: {args.video}")
    print(f"Prompts to test: {len(PROMPTS)}")
    print()

    for i, prompt in enumerate(PROMPTS, 1):
        print(f"[{i}/{len(PROMPTS)}] Testing prompt: '{prompt}'")
        print("-" * 60)

        cmd = [
            sys.executable,
            str(script_path),
            "--video",
            str(args.video),
            "--prompt",
            prompt,
            "--threshold",
            str(args.threshold),
        ]

        if args.no_mask:
            cmd.append("--no-mask")

        try:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            print(result.stdout)
            if result.stderr:
                print(result.stderr, file=sys.stderr)
        except subprocess.CalledProcessError as e:
            print(f"Error testing prompt '{prompt}': {e}")
            print(e.stdout)
            print(e.stderr, file=sys.stderr)
            continue

        print()

    print("=" * 60)
    print("✓ All prompts tested!")
    print(f"Check outputs in: outputs/videos/sam3_ball_trials/")


if __name__ == "__main__":
    main()

