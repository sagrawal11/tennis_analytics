# Hero Video Generator

This folder contains scripts to create promotional videos for the CourtVision hero section by combining player tracking (SAM-3d-body) and ball detection (SAM3) visualizations.

## Overview

The hero video showcases the computer vision capabilities by:
- Tracking players using SAM-3d-body (skeleton/keypoints overlay)
- Detecting tennis balls using SAM3
- Combining both visualizations into a polished promotional video

## Files

- `process_hero_video.py` - Main script to process videos and combine visualizations
- `visualizer.py` - Utilities for creating clean, professional visualizations
- `requirements.txt` - Python dependencies (if needed)

## Usage

### Basic Usage

```bash
# Activate your virtual environment
source ../tennis_env/bin/activate  # or your venv path

# Process a video
python process_hero_video.py \
    --input path/to/your/video.mp4 \
    --output hero_video_output.mp4 \
    --ball-prompt "tennis ball"
```

### Options

```bash
python process_hero_video.py \
    --input video.mp4 \
    --output output.mp4 \
    --ball-prompt "tennis ball" \
    --frame-skip 1 \
    --fps 30 \
    --player-color "#50C878" \
    --ball-color "#50C878" \
    --trail-length 30 \
    --keypoints-only
```

### Arguments

- `--input`: Path to input video file (required)
- `--output`: Path to output video file (required)
- `--ball-prompt`: Text prompt for SAM3 ball detection (default: "tennis ball")
- `--frame-skip`: Process every Nth frame (1 = all frames, default: 1)
- `--fps`: Output video FPS (default: 30)
- `--player-color`: Hex color for player skeletons (default: "#50C878" - emerald green)
- `--ball-color`: Hex color for ball and trajectory (default: "#50C878" - emerald green)
- `--trail-length`: Number of frames to show in ball trajectory trail (default: 30)
- `--keypoints-only`: Use fast keypoints-only mode (no 3D mesh rendering, faster)
- `--device`: Device to use ('cuda', 'mps', 'cpu', or 'auto' for auto-detect)
- `--no-court-detection`: Disable court detection (optional feature)
- `--court-model`: Path to court detection model file (default: auto-detect from `old/models/court/`)

## Workflow

1. **Prepare your video clips**: Gather pro match clips that showcase:
   - Clear player movements (serves, volleys, rallies)
   - Visible ball trajectory
   - Good camera angles (aerial or side view work well)

2. **Process each clip**: Run the script on each clip to generate annotated versions

3. **Edit and combine**: Use video editing software to:
   - Select the best segments
   - Create smooth transitions
   - Add any text overlays or branding
   - Export as a looped video

4. **Add to frontend**: Place the final video in `frontend/public/` and update `hero-section.tsx`

## Visual Style

The script creates clean, professional visualizations:
- **Player skeletons**: Emerald green (#50C878) keypoints and connections
- **Player mesh**: Emerald green 3D mesh (when using full mesh mode, not keypoints-only)
- **Ball tracking**: Emerald green ball with smooth trajectory trail
- **Court lines**: Emerald green court boundaries (if court detection model is available)
- **Minimal UI**: No debug text or cluttered overlays

**Color Scheme**: Everything uses emerald green (#50C878) for a cohesive, branded look.

**Court Detection**: Court detection is an optional feature that will be automatically enabled if:
- The court detection model file exists (`old/models/court/model_tennis_court_det.pt`)
- Required dependencies are available
- If unavailable, the script will continue without court detection

## Tips

- **Performance**: Use `--keypoints-only` for faster processing (skips 3D mesh rendering)
- **Quality**: Process at original resolution, then downscale if needed for web
- **Length**: Keep hero video short (10-30 seconds) and loopable
- **Selection**: Choose clips with dynamic action and clear visibility

## Troubleshooting

### OpenGL/Visualization Errors
If you get OpenGL errors, use `--keypoints-only` flag to skip 3D rendering.

### Memory Issues
- Reduce `--frame-skip` to process fewer frames
- Process shorter clips and combine later
- Use lower resolution input videos

### Ball Detection Issues
- Try different prompts: "tennis ball", "ball", "yellow ball"
- Adjust confidence threshold in the script if needed

### Court Detection Not Working
- Court detection is optional and requires the model file at `old/models/court/model_tennis_court_det.pt`
- If the model isn't found, the script will continue without court detection (you'll see a warning)
- Use `--no-court-detection` to explicitly disable it
- Court detection requires additional dependencies (TennisCourtDetector library)

## Example Usage

See `example_usage.sh` for example commands. Basic usage:

```bash
# Activate virtual environment
source ../tennis_env/bin/activate

# Process video
python process_hero_video.py \
    --input path/to/video.mp4 \
    --output hero_output.mp4 \
    --ball-prompt "tennis ball" \
    --keypoints-only
```

## Next Steps

After generating the video:
1. Review the output and select best segments
2. Edit in video software (Final Cut, Premiere, etc.) to:
   - Select the most impressive clips
   - Add smooth transitions
   - Create a seamless loop
   - Add any text overlays or branding
3. Export as optimized MP4 for web (H.264, reasonable bitrate)
4. Place final video in `frontend/public/hero-video.mp4` (or your preferred name)
5. Update `frontend/components/landing/hero-section.tsx` to use the video:
   ```tsx
   <video
     autoPlay
     loop
     muted
     playsInline
     className="w-full h-full object-cover"
   >
     <source src="/hero-video.mp4" type="video/mp4" />
   </video>
   ```
