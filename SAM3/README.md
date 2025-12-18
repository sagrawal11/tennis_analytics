# SAM 3 Ball Detection

This directory contains the SAM 3 model files and test scripts for ball detection in tennis videos.

## Model Files

The SAM 3 model is stored in this directory with the following structure:
- `model.safetensors` - Model weights
- `sam3.pt` - Alternative model format
- `config.json` - Model configuration
- `processor_config.json` - Processor configuration
- `tokenizer.json`, `vocab.json` - Tokenizer files

## Installation

### For Apple Silicon (M1/M2/M3)

```bash
# Install transformers from GitHub for compatibility
pip install git+https://github.com/huggingface/transformers torchvision

# Install other dependencies
pip install -r requirements.txt
```

### For Other Platforms

```bash
pip install -r requirements.txt
```

## Usage

### Standalone Ball Detection Test

Test SAM 3 ball detection on a tennis video:

```bash
python SAM3/test_sam3_ball_detection.py \
    --video path/to/tennis_video.mp4 \
    --prompt "tennis ball"
```

**Options:**
- `--video`: Path to input video (required)
- `--prompt`: Text prompt for detection (default: "tennis ball")
  - Try variations: "ball", "yellow ball", "tennis ball"
- `--output`: Path to output video (optional, auto-generated if not specified)
- `--threshold`: Confidence threshold (default: 0.5)
- `--no-mask`: Don't overlay masks on output
- `--use-sam3-package`: Use original sam3 package instead of transformers

**Examples:**

```bash
# Basic usage
python SAM3/test_sam3_ball_detection.py --video old/data/raw/tennis_test5.mp4

# Try different prompts
python SAM3/test_sam3_ball_detection.py \
    --video old/data/raw/tennis_test5.mp4 \
    --prompt "ball"

python SAM3/test_sam3_ball_detection.py \
    --video old/data/raw/tennis_test5.mp4 \
    --prompt "yellow ball"

# Adjust confidence threshold
python SAM3/test_sam3_ball_detection.py \
    --video old/data/raw/tennis_test5.mp4 \
    --threshold 0.3
```

## Output

Output videos are saved to `outputs/videos/sam3_ball_trials/` with the naming pattern:
`<video_name>_sam3_<prompt>.mp4`

The output videos show:
- SAM 3 detection masks overlaid on frames
- Ball center point (green circle)
- Bounding boxes
- Confidence scores
- Frame counter

## API Options

SAM 3 can be used with two different APIs:

1. **Transformers API** (recommended, default)
   - Uses `Sam3Model` and `Sam3Processor` from transformers
   - Simpler and more standard
   - Better compatibility

2. **Original sam3 package**
   - Uses `build_sam3_image_model()` from sam3 package
   - Requires: `pip install git+https://github.com/facebookresearch/sam3`
   - Use with `--use-sam3-package` flag

## Next Steps

After testing SAM 3 standalone, it will be integrated into the main ball detection comparison framework at `tests/ball_model_comparison.py`.
