# SAM-3d-body Data Output Format

This document describes the JSON data structure output by `extract_data_to_json.py`.

## Overview

SAM-3d-body provides comprehensive 3D human body reconstruction data including:
- **Skeleton**: 3D and 2D keypoints, joint coordinates, and rotations
- **Mesh**: 3D mesh vertices for full body reconstruction
- **Pose**: Body, hand, and global pose parameters
- **Shape**: Body shape and scale parameters
- **Camera**: Camera translation and focal length

## JSON Structure

The output JSON file has the following structure:

```json
{
  "video_info": {
    "video_path": "path/to/video.mov",
    "resolution": {"width": 1920, "height": 1080},
    "fps": 30.0,
    "total_frames": 77,
    "frames_processed": 77,
    "frame_skip": 1
  },
  "frames": [
    {
      "frame_number": 0,
      "timestamp": 0.0,
      "num_persons": 1,
      "persons": [
        {
          "person_id": 0,
          "skeleton": { ... },
          "mesh": { ... },
          "pose": { ... },
          "shape": { ... },
          "camera": { ... },
          "bbox": { ... },
          "metadata": { ... }
        }
      ]
    },
    ...
  ]
}
```

## Data Fields Explained

### Skeleton Data

```json
"skeleton": {
  "keypoints_3d": [[x, y, z], ...],      // [70, 3] - 3D keypoints in camera space (meters)
  "keypoints_2d": [[x, y], ...],         // [70, 2] - 2D keypoints projected to image (pixels)
  "joint_coords_3d": [[x, y, z], ...],  // [127, 3] - 3D joint coordinates (meters)
  "joint_global_rotations": [[[r11, r12, r13], [r21, r22, r23], [r31, r32, r33]], ...]
                                         // [127, 3, 3] - Global joint rotation matrices
}
```

**Keypoints (70 keypoints)**:
- Full body pose keypoints including head, torso, arms, legs, hands, and feet
- 3D coordinates are in camera space (meters)
- 2D coordinates are in image space (pixels)

**Joint Coordinates (127 joints)**:
- Detailed skeleton joint positions
- Includes all body joints plus hand joints

**Joint Rotations**:
- 3x3 rotation matrices for each joint
- Represents the global orientation of each joint in 3D space

### Mesh Data

```json
"mesh": {
  "vertices": [[x, y, z], ...],  // [N, 3] - 3D mesh vertices in camera space (meters)
  "num_vertices": 18439           // Number of vertices in the mesh
}
```

**Vertices**:
- Full 3D mesh vertices for the human body
- Typically ~18,439 vertices for full body mesh
- Coordinates are in camera space (meters)
- Can be used to reconstruct the full 3D body surface

### Pose Parameters

```json
"pose": {
  "global_rotation": [rx, ry, rz],           // [3] - Global body rotation (Euler angles in radians)
  "pose_raw": [p1, p2, ..., p266],           // Raw pose parameters (266 values)
  "body_pose": [b1, b2, ..., b130],          // Body pose parameters (130 values)
  "hand_pose": [h1, h2, ..., h108],         // Hand pose parameters (54 per hand × 2)
  "face_expression": [f1, f2, ..., f72]     // Face expression parameters (72 values)
}
```

**Global Rotation**:
- Euler angles (ZYX order) representing the global body orientation
- Units: radians

**Body Pose**:
- 130 parameters controlling body joint rotations
- Based on the Momentum Human Rig (MHR) representation

**Hand Pose**:
- 108 parameters total (54 per hand)
- Controls finger and hand joint rotations

**Face Expression**:
- 72 parameters controlling facial expressions
- Includes mouth, eyes, eyebrows, etc.

### Shape Parameters

```json
"shape": {
  "shape_params": [s1, s2, ..., s45],   // [45] - Body shape parameters
  "scale_params": [sc1, sc2, ..., sc28] // [28] - Scale parameters
}
```

**Shape Parameters**:
- 45 parameters controlling body shape (height, weight, proportions)
- Based on PCA decomposition of body shapes

**Scale Parameters**:
- 28 parameters controlling body part scales
- Allows fine-grained control over body proportions

### Camera Parameters

```json
"camera": {
  "translation": [tx, ty, tz],  // [3] - Camera translation in 3D space (meters)
  "focal_length": 1234.56       // Estimated focal length (pixels)
}
```

**Translation**:
- 3D position of the camera relative to the person
- Units: meters

**Focal Length**:
- Estimated camera focal length
- Units: pixels
- Used for 3D to 2D projection

### Bounding Boxes

```json
"bbox": {
  "body": [x1, y1, x2, y2],           // [4] - Body bounding box (pixels)
  "left_hand": [x1, y1, x2, y2],     // [4] - Left hand bbox (if available)
  "right_hand": [x1, y1, x2, y2]     // [4] - Right hand bbox (if available)
}
```

**Bounding Boxes**:
- Format: [x_min, y_min, x_max, y_max] in image coordinates (pixels)
- Body bbox is always present
- Hand bboxes are only present if full inference is used

## Usage

### Extract Data from Video

```bash
# Activate virtual environment
source SAM_body_venv/bin/activate

# Extract all frames
python extract_data_to_json.py --video IMG_8169.MOV --output data.json

# Extract every 5th frame (faster)
python extract_data_to_json.py --video IMG_8169.MOV --output data.json --frame_skip 5
```

### Accessing the Data in Python

```python
import json

# Load the JSON data
with open('data.json', 'r') as f:
    data = json.load(f)

# Access video info
video_info = data['video_info']
print(f"Processed {video_info['frames_processed']} frames")

# Access frame data
for frame in data['frames']:
    frame_num = frame['frame_number']
    timestamp = frame['timestamp']
    num_persons = frame['num_persons']
    
    for person in frame['persons']:
        # Get 3D keypoints
        keypoints_3d = person['skeleton']['keypoints_3d']  # [70, 3]
        
        # Get mesh vertices
        vertices = person['mesh']['vertices']  # [N, 3]
        
        # Get pose parameters
        body_pose = person['pose']['body_pose']  # [130]
        hand_pose = person['pose']['hand_pose']  # [108]
        
        # Get shape parameters
        shape = person['shape']['shape_params']  # [45]
        
        # Get camera translation
        cam_t = person['camera']['translation']  # [3]
```

## Data Dimensions Reference

| Data Type | Shape | Description |
|-----------|-------|-------------|
| `keypoints_3d` | [70, 3] | 3D keypoints in camera space (meters) |
| `keypoints_2d` | [70, 2] | 2D keypoints in image space (pixels) |
| `joint_coords_3d` | [127, 3] | 3D joint coordinates (meters) |
| `joint_global_rotations` | [127, 3, 3] | Joint rotation matrices |
| `vertices` | [~18439, 3] | Mesh vertices (meters) |
| `global_rotation` | [3] | Euler angles (radians) |
| `body_pose` | [130] | Body pose parameters |
| `hand_pose` | [108] | Hand pose parameters (54 per hand) |
| `face_expression` | [72] | Face expression parameters |
| `shape_params` | [45] | Body shape parameters |
| `scale_params` | [28] | Scale parameters |
| `camera.translation` | [3] | Camera translation (meters) |

## Coordinate Systems

### 3D Coordinates (Camera Space)
- Origin: Camera center
- X-axis: Right
- Y-axis: Down
- Z-axis: Forward (into the scene)
- Units: Meters

### 2D Coordinates (Image Space)
- Origin: Top-left corner of image
- X-axis: Right (increasing)
- Y-axis: Down (increasing)
- Units: Pixels

## Notes

1. **Multiple Persons**: Each frame can contain multiple detected persons. Each person has their own complete data structure.

2. **Missing Data**: If a frame fails to process, it will have an `"error"` field instead of person data.

3. **Hand Data**: Hand bounding boxes and detailed hand pose are only available if full inference is used (default).

4. **Mesh Vertices**: The mesh vertices can be used with the face indices from `estimator.faces` to reconstruct the full 3D mesh.

5. **Coordinate Units**: 
   - 3D coordinates: meters
   - 2D coordinates: pixels
   - Angles: radians

6. **Frame Skipping**: Use `--frame_skip` to process fewer frames for faster processing or lower file sizes.

## Example: Reconstructing 3D Mesh

```python
import json
import numpy as np
import trimesh

# Load data
with open('data.json', 'r') as f:
    data = json.load(f)

# Get first person from first frame
frame = data['frames'][0]
person = frame['persons'][0]

# Get vertices
vertices = np.array(person['mesh']['vertices'])

# Get faces (from estimator, or use default MHR faces)
# Note: You'll need to get faces from the estimator or use a standard mesh
faces = estimator.faces  # This would be from the estimator object

# Create mesh
mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
mesh.export('body_mesh.obj')
```

## Example: Visualizing Skeleton

```python
import json
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Load data
with open('data.json', 'r') as f:
    data = json.load(f)

# Get first person from first frame
person = data['frames'][0]['persons'][0]

# Get 3D keypoints
keypoints_3d = np.array(person['skeleton']['keypoints_3d'])

# Plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(keypoints_3d[:, 0], keypoints_3d[:, 1], keypoints_3d[:, 2])
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.show()
```

