#!/usr/bin/env python3
"""
Test script for the updated shot classification with overhead smash and serve
"""

import subprocess
import sys
import os

def main():
    print("Testing updated tennis shot classification with new shot types...")
    print("Shot types now include: forehand, backhand, overhead_smash, serve")
    print()
    
    # Test the updated tennis_shot2.py
    cmd = [
        "python", "tennis_shot2.py",
        "--csv", "tennis_analysis_data.csv",
        "--video", "tennis_test5.mp4",
        "--viewer"
    ]
    
    print(f"Running: {' '.join(cmd)}")
    print("Look for the new shot types in the viewer:")
    print("- Green: Forehand")
    print("- Red: Backhand") 
    print("- Magenta: Overhead Smash")
    print("- Orange: Serve")
    print()
    
    try:
        result = subprocess.run(cmd, check=True)
        print("Test completed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"Error running test: {e}")
        return 1
    except KeyboardInterrupt:
        print("\nTest interrupted by user")
        return 0
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
