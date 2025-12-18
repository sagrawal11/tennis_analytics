# SAM 3D Body - Complete Setup Guide for macOS (Apple Silicon)

This guide provides comprehensive step-by-step instructions to set up and use SAM 3D Body on macOS with Apple Silicon (M1, M2, M3, or later), including all fixes and optimizations for MPS (Metal Performance Shaders) compatibility.

## What is SAM 3D Body?

SAM 3D Body is Meta's model for robust full-body human mesh recovery from a single image. It estimates 3D human pose, body shape, and hand pose based on the Momentum Human Rig (MHR) representation.

**Official Repository**: https://github.com/facebookresearch/sam-3d-body  
**HuggingFace Models**: 
- `facebook/sam-3d-body-dinov3` (DINOv3-H+ backbone, 840M params) - Recommended
- `facebook/sam-3d-body-vith` (ViT-H backbone, 631M params)

## Prerequisites for macOS

1. **macOS 12.3+** (Monterey or later) - Required for MPS support
2. **Python 3.11** (recommended, works best with Apple Silicon)
3. **Homebrew** (for installing system libraries)
4. **HuggingFace Account** with access to SAM 3D Body models
5. **Git** (usually pre-installed on macOS)
6. **Xcode Command Line Tools** (for building some dependencies)

### Install Prerequisites

   ```bash
# Install Xcode Command Line Tools (if not already installed)
   xcode-select --install

# Install Homebrew (if not already installed)
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install Mesa library for OpenGL support (required for visualization)
brew install mesa
   ```

## Step-by-Step Setup

### Step 1: Create Project Directory

```bash
# Create your project directory
mkdir SAM_3d_test
cd SAM_3d_test
```

### Step 2: Clone the SAM 3D Body Repository

```bash
# Clone the official repository
git clone https://github.com/facebookresearch/sam-3d-body.git
```

### Step 3: Create Python Virtual Environment

**Important**: Use Python 3.11 (required for transformers compatibility)

```bash
# Create virtual environment with Python 3.11
python3.11 -m venv SAM_body_venv

# Activate the virtual environment
source SAM_body_venv/bin/activate
```

**Verify Python version:**
```bash
python --version
# Should output: Python 3.11.x
```

### Step 4: Install PyTorch for Apple Silicon (MPS)

**Critical**: Install PyTorch with MPS support for Apple Silicon acceleration.

```bash
# Install PyTorch with MPS support (Apple Silicon optimized)
pip install torch torchvision torchaudio
```

**Verify MPS is available:**
```python
python -c "import torch; print(f'MPS available: {torch.backends.mps.is_available()}'); print(f'MPS built: {torch.backends.mps.is_built()}')"
```

Expected output:
```
MPS available: True
MPS built: True
```

If MPS is not available, update PyTorch:
```bash
pip install --upgrade torch torchvision torchaudio
```

### Step 5: Install Core Python Dependencies

Install all required Python packages in this order:

```bash
# Core dependencies
pip install pytorch-lightning pyrender opencv-python yacs scikit-image einops timm dill pandas rich hydra-core hydra-submitit-launcher hydra-colorlog pyrootutils webdataset chump networkx==3.2.1 roma joblib seaborn wandb appdirs appnope ffmpeg cython jsonlines pytest xtcocotools loguru optree fvcore black pycocotools tensorboard huggingface_hub

# Install transformers from GitHub (required for Apple Silicon compatibility)
# Standard pip install may have compatibility issues
pip install git+https://github.com/huggingface/transformers torchvision
```

**Note**: Installing transformers from GitHub ensures compatibility with Apple Silicon and latest features.

### Step 6: Install Detectron2 (Optional, for Human Detection)

```bash
pip install 'git+https://github.com/facebookresearch/detectron2.git@a1ce2f9' --no-build-isolation --no-deps
```

**Note**: Detectron2 is optional. If installation fails or SSL certificate issues occur, the video processing script will work without it (processing full image frames instead of detected bounding boxes).

### Step 7: Install MoGe (Required for FOV Estimation)

```bash
pip install git+https://github.com/microsoft/MoGe.git
```

### Step 8: Request Access to HuggingFace Models

1. Go to https://huggingface.co/facebook/sam-3d-body-dinov3
2. Click "Request Access" and fill out the form
3. Wait for approval (usually quick)
4. Repeat for `facebook/sam-3d-body-vith` if you want that variant

### Step 9: Login to HuggingFace

```bash
huggingface-cli login
```

Enter your HuggingFace token when prompted. Get your token from: https://huggingface.co/settings/tokens

**Verify login:**
```bash
huggingface-cli whoami
```

### Step 10: Download Model Checkpoints

The models will be automatically downloaded on first use, but you can also download them manually:

**Option A: Using HuggingFace CLI**

```bash
# Download the DINOv3 model (recommended)
huggingface-cli download facebook/sam-3d-body-dinov3 --local-dir checkpoints/sam-3d-body-dinov3

# Or download the ViT-H model
huggingface-cli download facebook/sam-3d-body-vith --local-dir checkpoints/sam-3d-body-vith
```

**Option B: Automatic Download**

Models will be automatically downloaded when you first run the processing script.

## macOS-Specific Fixes Applied

This setup includes several critical fixes for macOS/Apple Silicon compatibility:

### 1. MPS Device Compatibility Fixes

The following files have been modified to ensure proper MPS support:

- **`sam-3d-body/sam_3d_body/build_models.py`**: Fixed device argument passing
- **`sam-3d-body/sam_3d_body/sam_3d_body_estimator.py`**: Fixed hardcoded CUDA references
- **`sam-3d-body/sam_3d_body/models/meta_arch/sam3d_body.py`**: Replaced `.cuda()` calls with device-agnostic `.to(device)`
- **`sam-3d-body/sam_3d_body/utils/dist.py`**: Added float64 to float32 conversion for MPS
- **`sam-3d-body/sam_3d_body/data/utils/prepare_batch.py`**: Fixed camera intrinsics dtype
- **`sam-3d-body/sam_3d_body/data/transforms/common.py`**: Fixed numpy array dtypes
- **`sam-3d-body/sam_3d_body/data/transforms/bbox_utils.py`**: Fixed numpy array dtypes
- **`sam-3d-body/sam_3d_body/visualization/renderer.py`**: Fixed tensor dtypes and macOS OpenGL platform
- **`sam-3d-body/tools/build_fov_estimator.py`**: Fixed FOV estimator to return proper types
- **`sam-3d-body/sam_3d_body/models/heads/mhr_head.py`**: Fixed MHR model to work with MPS (keeps MHR on CPU)

### 2. Float64 to Float32 Conversion

MPS (Metal Performance Shaders) doesn't support float64. All tensor operations have been updated to use float32.

### 3. MHR Model CPU Fallback

The MHR (Momentum Human Rig) TorchScript model uses double precision internally, which MPS doesn't support. The code automatically keeps the MHR model on CPU while the rest of the model runs on MPS.

### 4. OpenGL Visualization Fixes

- Removed osmesa requirement (not easily available on macOS)
- Uses default OpenGL platform on macOS
- Added fallback visualization if OpenGL fails

## Complete Dependency List

Here are all the Python packages required:

### Core Dependencies
```
pytorch-lightning
pyrender
opencv-python
yacs
scikit-image
einops
timm
dill
pandas
rich
hydra-core
hydra-submitit-launcher
hydra-colorlog
pyrootutils
webdataset
chump
networkx==3.2.1
roma
joblib
seaborn
wandb
appdirs
appnope
ffmpeg
cython
jsonlines
pytest
xtcocotools
loguru
optree
fvcore
black
pycocotools
tensorboard
huggingface_hub
```

### From GitHub
```
git+https://github.com/huggingface/transformers
git+https://github.com/facebookresearch/detectron2.git@a1ce2f9 (optional)
git+https://github.com/microsoft/MoGe.git
```

### Optional: 3D Visualization Libraries
```
pyvista (recommended for interactive 3D rendering)
# or
open3d (alternative for interactive 3D rendering)
```

### System Libraries (via Homebrew)
```
mesa (for OpenGL support)
```

## Usage: Video Processing

### Basic Video Processing

The main script for processing videos is `process_video.py`. It processes each frame of a video and generates 3D mesh visualizations.

**Command:**
```bash
# Activate virtual environment
source SAM_body_venv/bin/activate

# Process all frames
python process_video.py --video IMG_8169.MOV --output_dir ./outputs

# Process every 5th frame (faster)
python process_video.py --video IMG_8169.MOV --output_dir ./outputs --frame_skip 5

# Process without creating output video (only saves individual frames)
python process_video.py --video IMG_8169.MOV --output_dir ./outputs --no-video
```

### Command-Line Arguments

```bash
python process_video.py --help
```

**Available options:**
- `--video`: Path to input video file (default: IMG_8169.MOV)
- `--output_dir`: Directory to save output images and video (default: ./outputs)
- `--hf_repo_id`: HuggingFace repository ID (default: facebook/sam-3d-body-dinov3)
- `--device`: Device to use - mps, cuda, or cpu (default: auto-detect)
- `--frame_skip`: Process every Nth frame (default: 1 = all frames)
- `--no-video`: Don't create output video, only save individual frame images
- `--output_fps`: FPS for output video (default: 30.0)

### Example Commands

```bash
# Process all frames with default settings
python process_video.py --video my_video.mov

# Process every 10th frame (10x faster)
python process_video.py --video my_video.mov --frame_skip 10

# Process with custom output directory
python process_video.py --video my_video.mov --output_dir ./my_outputs

# Process without creating video file
python process_video.py --video my_video.mov --no-video
```

### Output Files

After processing, you'll find:
- **Individual frame images**: `outputs/frame_000000.jpg`, `frame_000001.jpg`, etc.
- **Processed video**: `outputs/IMG_8169_processed.mp4` (if `--no-video` not used)

Each frame image contains:
- Original image
- 2D keypoints visualization
- 3D mesh rendering (front view)
- 3D mesh rendering (side view)

## Usage: Single Image Processing

You can also process single images using the same estimator:

```python
import cv2
import numpy as np
import sys
from pathlib import Path

# Add sam-3d-body to path
sys.path.insert(0, str(Path(__file__).parent / "sam-3d-body"))

from notebook.utils import setup_sam_3d_body
from tools.vis_utils import visualize_sample_together

# Auto-detect device (MPS for Apple Silicon)
import torch
device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using device: {device}")

# Set up the estimator
estimator = setup_sam_3d_body(
    hf_repo_id="facebook/sam-3d-body-dinov3",
    device=device,
    detector_name=None,  # Optional: set to "vitdet" if detector is available
    fov_name="moge2"  # FOV estimator
)

# Load and process image
image_path = "path/to/your/image.jpg"
img_bgr = cv2.imread(image_path)
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

# Process image
outputs = estimator.process_one_image(img_rgb)

# Visualize results
rend_img = visualize_sample_together(img_bgr, outputs, estimator.faces)
cv2.imwrite("output.jpg", rend_img.astype(np.uint8))

print(f"Detected {len(outputs)} person(s)")
```

## Usage: Interactive 3D Renderer

After processing videos with `process_video.py`, you can use the interactive 3D renderer to view the results in an interactive 3D viewer that supports mouse controls for rotation, zooming, and panning.

### Installation

Install one of the visualization libraries (PyVista is recommended for better interaction):

```bash
# Activate virtual environment
source SAM_body_venv/bin/activate

# Option 1: PyVista (recommended - smoother interaction)
pip install pyvista

# Option 2: Open3D (alternative)
pip install open3d
```

### Basic Usage

The renderer script (`render_3d.py`) loads JSON data files created by `process_video.py` and displays interactive 3D meshes.

**Command:**
```bash
# View first frame, first person (default)
python render_3d.py --json outputs/video_data.json

# View specific frame and person
python render_3d.py --json outputs/video_data.json --frame 10 --person 0

# View all persons in a frame
python render_3d.py --json outputs/video_data.json --frame 5 --all-persons

# Hide skeleton, only show mesh
python render_3d.py --json outputs/video_data.json --no-skeleton
```

### Command-Line Arguments

```bash
python render_3d.py --help
```

**Available options:**
- `--json`: Path to JSON data file (required) - output from `process_video.py`
- `--frame`: Frame index to visualize (default: 0)
- `--person`: Person index to visualize (default: 0, ignored if `--all-persons` is used)
- `--all-persons`: Show all persons in the frame (default: show only one person)
- `--no-skeleton`: Hide skeleton keypoints (default: show skeleton)

### Interactive Controls

Once the 3D viewer opens, you can interact with it:

- **Left Click + Drag**: Rotate the 3D model
- **Right Click + Drag**: Pan/move the view
- **Scroll Wheel**: Zoom in/out
- **Close Window or 'Q'**: Exit the viewer

### Example Commands

```bash
# View frame 0, person 0 (default)
python render_3d.py --json outputs/IMG_8169_data.json

# View frame 20, person 1
python render_3d.py --json outputs/IMG_8169_data.json --frame 20 --person 1

# View all persons in frame 10
python render_3d.py --json outputs/IMG_8169_data.json --frame 10 --all-persons

# View mesh only (no skeleton keypoints)
python render_3d.py --json outputs/IMG_8169_data.json --no-skeleton

# Fast processing workflow: Skip visualization during processing, then view later
python process_video.py --video my_video.mov --skip-visualization
python render_3d.py --json outputs/my_video_data.json --frame 0
```

### Workflow: Fast Processing + Interactive Viewing

For faster video processing, you can skip visualization during processing and view results interactively later:

```bash
# Step 1: Process video without visualization (much faster)
python process_video.py --video my_video.mov --skip-visualization --frame_skip 5

# Step 2: View results interactively in 3D
python render_3d.py --json outputs/my_video_data.json --frame 0
```

This workflow separates processing from visualization, allowing you to:
- Process videos quickly without rendering overhead
- View and explore results interactively at your own pace
- Navigate through different frames and persons easily

## Project Structure

After setup, your project should look like:

```
SAM_3d_test/
├── sam-3d-body/              # Cloned repository
│   ├── sam_3d_body/          # Main package (with MPS fixes)
│   ├── notebook/             # Example notebooks and utilities
│   ├── tools/                # Helper tools
│   └── ...
├── checkpoints/              # Model checkpoints (auto-downloaded)
│   └── sam-3d-body-dinov3/
│       ├── model.ckpt
│       ├── assets/
│       │   └── mhr_model.pt
│       └── model_config.yaml
├── SAM_body_venv/            # Python virtual environment
├── outputs/                  # Output images/videos
│   ├── frame_000000.jpg
│   ├── frame_000001.jpg
│   ├── IMG_8169_processed.mp4
│   └── IMG_8169_data.json    # JSON data file (for renderer)
├── process_video.py          # Main video processing script
├── render_3d.py              # Interactive 3D renderer script
├── SETUP_GUIDE.md            # This file
└── IMG_8169.MOV              # Input video (example)
```

## Output Format

Each detected person returns a dictionary with:

```python
{
    'pred_vertices': np.ndarray,      # 3D mesh vertices [N, 3] in camera coordinates
    'pred_keypoints_3d': np.ndarray, # 3D pose keypoints [70, 3]
    'pred_keypoints_2d': np.ndarray, # 2D keypoints projected to image [70, 2]
    'pred_cam_t': np.ndarray,        # Camera translation [3]
    'focal_length': float,           # Estimated focal length
    'body_pose_params': np.ndarray,  # Body pose parameters
    'hand_pose_params': np.ndarray,  # Hand pose parameters
    'shape_params': np.ndarray,      # Body shape parameters
    'bbox': np.ndarray,              # Bounding box [x1, y1, x2, y2]
}
```

## Troubleshooting

### MPS (Metal Performance Shaders) Issues

**Problem**: MPS not available or not working

**Solutions**:
1. **Check macOS version**: MPS requires macOS 12.3+ (Monterey or later)
   ```bash
   sw_vers
   ```

2. **Verify PyTorch MPS support**:
   ```python
   import torch
   print(f"MPS available: {torch.backends.mps.is_available()}")
   print(f"MPS built: {torch.backends.mps.is_built()}")
   ```

3. **If MPS is False**, reinstall PyTorch:
   ```bash
   pip uninstall torch torchvision torchaudio
   pip install torch torchvision torchaudio
   ```

4. **Force CPU if MPS causes issues**:
   ```bash
   python process_video.py --video IMG_8169.MOV --device cpu
   ```

### Float64 Error on MPS

**Problem**: `Cannot convert a MPS Tensor to float64 dtype`

**Solution**: This has been fixed in the code. If you still see this error, ensure you're using the modified files from this setup.

### OpenGL Visualization Issues

**Problem**: OpenGL errors or visualization fails

**Solutions**:
1. **Install Mesa library** (already done in setup):
   ```bash
   brew install mesa
   ```

2. **The script automatically falls back** to simple keypoint visualization if OpenGL fails

3. **Check OpenGL platform**:
   ```python
   import os
   print(os.environ.get('PYOPENGL_PLATFORM'))
   # Should be None or not set on macOS (uses default)
   ```

### SSL Certificate Errors

**Problem**: SSL certificate verification fails when downloading detector model

**Solution**: The script automatically handles this by continuing without the detector. The model will process full image frames instead of detected bounding boxes.

### Import Errors

**Problem**: `ModuleNotFoundError` or import errors

**Solutions**:
1. **Verify virtual environment is activated**:
   ```bash
   which python
   # Should point to SAM_body_venv/bin/python
   ```

2. **Reinstall missing packages**:
   ```bash
   pip install <package-name>
   ```

3. **Check Python path**:
   ```python
   import sys
   print(sys.path)
   # Should include path to sam-3d-body directory
   ```

### Model Download Fails

**Problem**: Cannot download model from HuggingFace

**Solutions**:
1. **Verify you're logged in**:
   ```bash
   huggingface-cli whoami
   ```

2. **Check you have access** to the model on HuggingFace website

3. **Try downloading manually**:
   ```bash
   huggingface-cli download facebook/sam-3d-body-dinov3 --local-dir checkpoints/sam-3d-body-dinov3
   ```

### Memory Issues

**Problem**: Out of memory errors

**Solutions**:
1. **Close other applications** to free up RAM
2. **Process fewer frames** using `--frame_skip`:
   ```bash
   python process_video.py --video IMG_8169.MOV --frame_skip 10
   ```
3. **Use CPU instead of MPS** (slower but uses less memory):
   ```bash
   python process_video.py --video IMG_8169.MOV --device cpu
   ```

### MHR Model Errors

**Problem**: MHR model errors or device mismatch

**Solution**: This has been fixed. The MHR model automatically runs on CPU while the rest uses MPS. If you see errors, ensure you're using the modified `mhr_head.py` file.

## Performance Notes for Apple Silicon

- **Model Size**: ~840M parameters (DINOv3) or ~631M (ViT-H)
- **Inference Time on Apple Silicon**:
  - **MPS (M1/M2/M3)**: ~2-3 seconds per frame
  - **CPU only**: ~5-8 seconds per frame
- **Memory**: 
  - Uses unified memory (RAM + GPU memory combined)
  - ~4-8 GB recommended for smooth operation
  - M1: 8GB minimum, 16GB recommended
  - M2/M3: 16GB+ recommended for best performance
- **Device**: 
  - **MPS (recommended)**: Automatic GPU acceleration via Metal
  - **CPU**: Fallback option, slower but more stable
- **Optimization Tips**:
  - Keep other applications closed to maximize available memory
  - Use `--frame_skip` to process fewer frames if needed
  - Resize very large videos before processing

## Complete Installation Command Summary

Here's a complete command sequence to set up everything from scratch:

```bash
# 1. Install system prerequisites
xcode-select --install
brew install mesa

# 2. Create project directory
mkdir SAM_3d_test
cd SAM_3d_test

# 3. Clone repository
git clone https://github.com/facebookresearch/sam-3d-body.git

# 4. Create virtual environment
python3.11 -m venv SAM_body_venv
source SAM_body_venv/bin/activate

# 5. Install PyTorch
pip install torch torchvision torchaudio

# 6. Install core dependencies
pip install pytorch-lightning pyrender opencv-python yacs scikit-image einops timm dill pandas rich hydra-core hydra-submitit-launcher hydra-colorlog pyrootutils webdataset chump networkx==3.2.1 roma joblib seaborn wandb appdirs appnope ffmpeg cython jsonlines pytest xtcocotools loguru optree fvcore black pycocotools tensorboard huggingface_hub

# 7. Install transformers from GitHub
pip install git+https://github.com/huggingface/transformers torchvision

# 8. Install Detectron2 (optional)
pip install 'git+https://github.com/facebookresearch/detectron2.git@a1ce2f9' --no-build-isolation --no-deps

# 9. Install MoGe
pip install git+https://github.com/microsoft/MoGe.git

# 10. Login to HuggingFace
huggingface-cli login

# 11. Install 3D visualization library (optional, for interactive renderer)
pip install pyvista
# or
# pip install open3d

# 12. Test installation
python -c "import torch; print(f'MPS: {torch.backends.mps.is_available()}')"
```

## Quick Start Checklist

- [ ] Install Xcode Command Line Tools
- [ ] Install Homebrew and Mesa
- [ ] Clone sam-3d-body repository
- [ ] Create Python 3.11 virtual environment
- [ ] Install PyTorch with MPS support
- [ ] Install all Python dependencies
- [ ] Install transformers from GitHub
- [ ] Install Detectron2 (optional)
- [ ] Install MoGe
- [ ] Request HuggingFace access
- [ ] Login to HuggingFace CLI
- [ ] Install 3D visualization library (PyVista or Open3D)
- [ ] Test MPS availability
- [ ] Run video processing script
- [ ] Test interactive 3D renderer

## Additional Resources

- **Official Repository**: https://github.com/facebookresearch/sam-3d-body
- **Paper**: https://ai.meta.com/research/publications/sam-3d-body-robust-full-body-human-mesh-recovery/
- **Blog Post**: https://ai.meta.com/blog/sam-3d/
- **Live Demo**: https://www.aidemos.meta.com/segment-anything/editor/convert-body-to-3d
- **HuggingFace**: https://huggingface.co/facebook/sam-3d-body-dinov3

## License

SAM 3D Body is licensed under the SAM License. See the LICENSE file in the repository for details.

---

**Last Updated**: Based on setup completed with all macOS/Apple Silicon compatibility fixes applied.
