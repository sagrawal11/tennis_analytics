# SAM_3D Body Model Test Project

This project contains scripts to download and test the SAM_3D Body model from HuggingFace.

## Prerequisites

1. **Conda Environment**: You should have the `sam_3d_body` conda environment set up and activated
2. **HuggingFace Access**: You must be approved for access to the `facebook/sam-3d-body-dinov3` model
3. **HuggingFace Login**: You must be logged in to HuggingFace CLI

## Setup

### 1. Activate your conda environment

```bash
conda activate sam_3d_body
```

### 2. Install required packages

```bash
pip install -r requirements.txt
```

### 3. Login to HuggingFace (if not already done)

```bash
huggingface-cli login
```

Enter your HuggingFace token when prompted.

### 4. Download the model

Run the download script to fetch the model from HuggingFace:

```bash
python download_model.py
```

This will download the model to `./checkpoints/sam-3d-body-dinov3/`

## Usage

### Option 1: Simple Test Script

Edit `test_single_image.py` and update the `image_path` variable with your image path, then run:

```bash
python test_single_image.py
```

### Option 2: Command-line Demo Script

Process a single image:

```bash
python demo.py \
    --image_path path/to/your/image.jpg \
    --output_folder ./outputs
```

Process a folder of images:

```bash
python demo.py \
    --image_folder path/to/your/images \
    --output_folder ./outputs
```

### Option 3: Using the HuggingFace Example Code

You can also use the code directly in a Python script or notebook:

```python
import cv2
import numpy as np
from notebook.utils import setup_sam_3d_body
from tools.vis_utils import visualize_sample_together

# Set up the estimator
estimator = setup_sam_3d_body(hf_repo_id="facebook/sam-3d-body-dinov3")

# Load and process image
img_bgr = cv2.imread("path/to/image.jpg")
outputs = estimator.process_one_image(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))

# Visualize and save results
rend_img = visualize_sample_together(img_bgr, outputs, estimator.faces)
cv2.imwrite("output.jpg", rend_img.astype(np.uint8))
```

## Output Format

Each detected person returns a dictionary containing:

- `pred_vertices`: 3D mesh vertices in camera coordinates
- `pred_keypoints_3d`: 3D pose keypoints
- `pred_keypoints_2d`: 2D pose keypoints projected to image
- `pred_cam_t`: Camera translation parameters
- `focal_length`: Estimated focal length
- `body_pose_params`: Body pose parameters
- `hand_pose_params`: Hand pose parameters
- `shape_params`: Body shape parameters

## Project Structure

```
SAM_3d_test/
├── checkpoints/              # Model checkpoints (created after download)
│   └── sam-3d-body-dinov3/
├── outputs/                  # Output images (created when running scripts)
├── download_model.py         # Script to download model from HuggingFace
├── demo.py                   # Full-featured demo script
├── test_single_image.py      # Simple test script
├── requirements.txt          # Python dependencies
└── README.md                 # This file
```

## Troubleshooting

1. **Import errors**: Make sure you're in the correct conda environment and that the `sam_3d_body` package is properly installed
2. **Model download fails**: Verify you're logged in to HuggingFace and have access to the model
3. **CUDA errors**: If you have GPU issues, the model should fall back to CPU automatically

## Notes

- The model files are large and may take some time to download
- Processing time depends on image size and hardware
- Make sure you have sufficient disk space for the model checkpoints


