# Core Classes for FIFA Skeletal Tracking Dataset

This module provides an object-oriented interface to work with the FIFA Skeletal Tracking Challenge dataset. The classes abstract away the complexity of loading and manipulating NPZ files, providing intuitive methods for data access and visualization.

## Overview

The module consists of two main categories:

1. **Data Classes**: Handle specific data types (poses, cameras, bboxes, skeletons)
2. **Metadata Classes**: Aggregate multiple data types for convenient access

## Installation & Setup

To use these classes in your scripts:

```python
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path.cwd() / 'src'))

from classes import (
    PosesData, CamerasData, BBoxesData,
    Skeleton2DData, Skeleton3DData,
    ImageMetadata, VideoMetadata
)
```

## Data Classes

### PosesData

Handles SMPL pose parameters from the WorldPose dataset.

**Data Structure:**
- `global_orient`: (num_subjects, num_frames, 3) - Global orientation (axis-angle)
- `body_pose`: (num_subjects, num_frames, 69) - Body pose (23 joints × 3)
- `transl`: (num_subjects, num_frames, 3) - Translation in world coordinates
- `betas`: (num_subjects, num_frames, 10) - Shape parameters

**Usage:**

```python
from pathlib import Path
from classes import PosesData

# Load a single sequence
poses = PosesData.load(Path('data/poses'), 'ARG_FRA_183303')
print(poses)  # PosesData(sequence='ARG_FRA_183303', frames=1944, subjects=22)

# Access frame data
frame_data = poses.get_frame_data(frame_idx=100)
print(frame_data.keys())  # ['global_orient', 'body_pose', 'transl', 'betas']

# Get specific subject's data
subject_data = poses.get_frame_data(frame_idx=100, subject_idx=0)

# Get trajectory (translation over time) for a subject
trajectory = poses.get_subject_trajectory(subject_idx=0)  # (num_frames, 3)

# Load all sequences
all_poses = PosesData.load_all(Path('data/poses'))
print(f"Loaded {len(all_poses)} sequences")
```

**Attributes:**
- `sequence_name`: Name of the sequence
- `global_orient`, `body_pose`, `transl`, `betas`: SMPL parameters
- `num_subjects`: Number of subjects
- `num_frames`: Number of frames

---

### CamerasData

Handles camera calibration parameters (intrinsics and extrinsics).

**Data Structure:**
- `K`: (num_frames, 3, 3) - Intrinsic matrices per frame
- `k`: (num_frames, 5) - Distortion coefficients (only k[0:2] valid)
- `R`: (1, 3, 3) - Rotation matrix (first frame only)
- `t`: (1, 3) - Translation vector (first frame only)
- `Rt`: (1, 3, 4) - Combined [R|t] (first frame only)

**Usage:**

```python
from classes import CamerasData

# Load camera data
cameras = CamerasData.load(Path('data/cameras'), 'ARG_FRA_183303')

# Get intrinsics for a specific frame
K, k_dist = cameras.get_intrinsics(frame_idx=100)

# Get extrinsics for first frame
R, t = cameras.get_extrinsics_first_frame()

# Get full projection matrix for first frame
P = cameras.get_projection_matrix_first_frame()  # (3, 4)

# Project 3D points to 2D
points_3d = np.array([[0, 0, 5], [1, 1, 5]])  # (N, 3)
points_2d = cameras.project_3d_to_2d(
    points_3d,
    frame_idx=100,
    R=R,
    t=t,
    apply_distortion=True
)
```

**Attributes:**
- `sequence_name`: Name of the sequence
- `K`, `k`, `R`, `t`, `Rt`: Camera parameters
- `num_frames`: Number of frames

---

### BBoxesData

Handles bounding box annotations in XYXY format.

**Data Structure:**
- `boxes`: (num_frames, num_subjects, 4) - XYXY format [x_min, y_min, x_max, y_max]
- NaN values indicate subject not present in frame

**Usage:**

```python
from classes import BBoxesData
import cv2

# Load all bounding boxes
boxes_dict = BBoxesData.load_all(Path('data/boxes.npz'))

# Load specific sequence
bboxes = BBoxesData.load(Path('data/boxes.npz'), 'ARG_FRA_183303')

# Get boxes for a specific frame
frame_boxes = bboxes.get_frame_boxes(frame_idx=100)  # Valid boxes only
all_boxes = bboxes.get_frame_boxes(frame_idx=100, valid_only=False)

# Get boxes for a specific subject across all frames
subject_boxes = bboxes.get_subject_boxes(subject_idx=0)

# Count valid subjects in a frame
num_subjects = bboxes.count_valid_subjects(frame_idx=100)

# Visualize bounding boxes on image
image = cv2.imread('data/images/ARG_FRA_183303/00100.jpg')
img_with_boxes = bboxes.visualize_frame(image, frame_idx=100)

# Convert to XYWH format
xywh_boxes = bboxes.to_xywh(frame_idx=100)
```

**Attributes:**
- `sequence_name`: Name of the sequence
- `boxes`: (num_frames, num_subjects, 4) bounding boxes
- `num_frames`: Number of frames
- `num_subjects`: Maximum number of subjects

---

### Skeleton2DData

Handles 2D skeletal keypoint data (25 joints: SMPL 24 + nose).

**Data Structure:**
- `keypoints`: (num_frames, num_subjects, 25, 2) - 2D pixel coordinates

**Usage:**

```python
from classes import Skeleton2DData

# Load 2D skeleton data
skel_2d = Skeleton2DData.load(Path('data/skel_2d.npz'), 'ARG_FRA_183303')

# Get keypoints for a frame
frame_keypoints = skel_2d.get_frame_keypoints(frame_idx=100)  # All subjects
subject_keypoints = skel_2d.get_frame_keypoints(frame_idx=100, subject_idx=0)

# Visualize on image
image = cv2.imread('data/images/ARG_FRA_183303/00100.jpg')
img_with_skeleton = skel_2d.visualize_frame(
    image,
    frame_idx=100,
    show_skeleton=True,
    show_joints=True
)
```

**Joint Structure:**
See `SMPL_JOINT_NAMES` in `skeleton.py` for the complete list of 25 joints.

---

### Skeleton3DData

Handles 3D skeletal keypoint data (25 joints: SMPL 24 + nose).

**Data Structure:**
- `keypoints`: (num_frames, num_subjects, 25, 3) - 3D world coordinates (meters)

**Usage:**

```python
from classes import Skeleton3DData
import matplotlib.pyplot as plt

# Load 3D skeleton data
skel_3d = Skeleton3DData.load(Path('data/skel_3d.npz'), 'ARG_FRA_183303')

# Get keypoints for a frame
frame_keypoints = skel_3d.get_frame_keypoints(frame_idx=100)

# Convert to submission format (15 joints)
submission_kpts = skel_3d.to_submission_format(frame_idx=100)  # (num_subjects, 15, 3)

# Visualize in 3D space
fig = skel_3d.visualize_3d(frame_idx=100, elev=20, azim=-60)
plt.show()
```

**Submission Joints:**
The 15 joints for submission are: nose, shoulders, elbows, wrists, hips, knees, ankles, feet.
See `SUBMISSION_JOINT_INDICES` in `skeleton.py`.

---

## Metadata Classes

### ImageMetadata

Aggregates all data related to a single frame/image.

**Usage:**

```python
from classes import ImageMetadata, BBoxesData

# Create ImageMetadata with specific data components
bboxes = BBoxesData.load(Path('data/boxes.npz'), 'ARG_FRA_183303')

frame_meta = ImageMetadata(
    sequence_name='ARG_FRA_183303',
    frame_idx=100,
    bboxes=bboxes
)

# Load image
frame_meta.load_image(Path('data/images'))

# Access data through unified interface
boxes = frame_meta.get_bboxes()
K, k = frame_meta.get_camera_intrinsics()  # If cameras loaded
skel_2d = frame_meta.get_skeleton_2d()     # If skel_2d loaded
skel_3d = frame_meta.get_skeleton_3d()     # If skel_3d loaded

# Visualize
img_with_boxes = frame_meta.visualize_bboxes()
img_with_skel = frame_meta.visualize_skeleton_2d()
fig = frame_meta.visualize_skeleton_3d()
```

**Key Methods:**
- `load_image(images_dir)`: Load image from standard directory
- `get_poses_data(subject_idx)`: Get SMPL parameters
- `get_bboxes(valid_only)`: Get bounding boxes
- `get_skeleton_2d(subject_idx)`: Get 2D keypoints
- `get_skeleton_3d(subject_idx)`: Get 3D keypoints
- `get_camera_intrinsics()`: Get camera intrinsics
- `visualize_bboxes(image)`: Visualize bounding boxes
- `visualize_skeleton_2d(image)`: Visualize 2D skeleton
- `visualize_skeleton_3d(**kwargs)`: Visualize 3D skeleton

---

### VideoMetadata

Aggregates all data related to an entire video sequence.

**Usage:**

```python
from classes import VideoMetadata

# Load all data for a sequence
video_meta = VideoMetadata.load(
    Path('data'),
    'ARG_FRA_183303',
    load_poses=True,
    load_cameras=True,
    load_bboxes=True,
    load_skel_2d=True,
    load_skel_3d=True
)

print(video_meta)
# VideoMetadata(sequence='ARG_FRA_183303', frames=1944,
#               components=[poses, cameras, bboxes, skel_2d, skel_3d])

# Access data components directly
poses = video_meta.poses
cameras = video_meta.cameras
bboxes = video_meta.bboxes
skel_2d = video_meta.skel_2d
skel_3d = video_meta.skel_3d

# Extract specific frame as ImageMetadata
frame_meta = video_meta.get_frame(
    frame_idx=100,
    load_image=True,
    images_dir=Path('data/images')
)

# Load all sequences
all_videos = VideoMetadata.load_all(Path('data'))
print(f"Loaded {len(all_videos)} sequences")
```

**Key Attributes:**
- `sequence_name`: Name of the sequence
- `num_frames`: Number of frames
- `poses`, `cameras`, `bboxes`, `skel_2d`, `skel_3d`: Data components

**Key Methods:**
- `load(data_dir, sequence_name, ...)`: Load single sequence
- `load_all(data_dir, ...)`: Load all sequences
- `get_frame(frame_idx, load_image, images_dir)`: Extract ImageMetadata

---

## Complete Example Workflow

Here's a complete example showing how to use the classes together:

```python
import sys
from pathlib import Path
import cv2
import matplotlib.pyplot as plt

# Add src to path
sys.path.insert(0, str(Path.cwd() / 'src'))

from classes import VideoMetadata

# 1. Load complete video metadata
data_dir = Path('data')
video = VideoMetadata.load(
    data_dir,
    'ARG_FRA_183303',
    load_poses=True,
    load_cameras=True,
    load_bboxes=True,
    load_skel_2d=True,
    load_skel_3d=True
)

print(f"Loaded sequence: {video}")
print(f"Number of frames: {video.num_frames}")

# 2. Extract a specific frame
frame = video.get_frame(
    frame_idx=100,
    load_image=True,
    images_dir=data_dir / 'images'
)

# 3. Work with the frame data
# Access bounding boxes
boxes = frame.get_bboxes()
print(f"Number of subjects: {len(boxes)}")

# Access 3D skeleton
skel_3d = frame.get_skeleton_3d()
print(f"3D skeleton shape: {skel_3d.shape}")

# Access SMPL poses
poses = frame.get_poses_data()
print(f"Body pose shape: {poses['body_pose'].shape}")

# 4. Visualize
# Visualize bounding boxes
img_boxes = frame.visualize_bboxes()
cv2.imwrite('output_boxes.jpg', img_boxes)

# Visualize 2D skeleton
img_skel_2d = frame.visualize_skeleton_2d()
cv2.imwrite('output_skel_2d.jpg', img_skel_2d)

# Visualize 3D skeleton
fig = frame.visualize_skeleton_3d(elev=20, azim=-60)
fig.savefig('output_skel_3d.png')

# 5. Process all frames
for frame_idx in range(0, video.num_frames, 100):
    frame = video.get_frame(frame_idx)
    boxes = frame.get_bboxes()
    print(f"Frame {frame_idx}: {len(boxes)} subjects")
```

## Benefits of Using These Classes

1. **Simplified API**: No need to remember NPZ file structure and keys
2. **Type Safety**: Clear method signatures and return types
3. **Encapsulation**: Data loading logic is encapsulated in the classes
4. **Reusability**: Visualization methods are built-in and reusable
5. **Consistency**: Uniform interface across all data types
6. **Extensibility**: Easy to add new methods and functionality

## Migration from Old Code

**Before (raw NPZ access):**
```python
# Load data manually
skel_3d = np.load('data/skel_3d.npz', allow_pickle=True)
boxes = np.load('data/boxes.npz', allow_pickle=True)

# Access with dictionary keys
sequence_skel = skel_3d['ARG_FRA_183303']
sequence_boxes = boxes['ARG_FRA_183303']

# Extract frame data manually
frame_skel = sequence_skel[frame_idx]
frame_boxes = sequence_boxes[frame_idx]

# Filter valid boxes manually
valid_mask = ~np.isnan(frame_boxes[:, 0])
valid_boxes = frame_boxes[valid_mask]
```

**After (using classes):**
```python
# Load data with classes
from classes import VideoMetadata

video = VideoMetadata.load(Path('data'), 'ARG_FRA_183303')
frame = video.get_frame(frame_idx)

# Access data through clean API
frame_skel = frame.get_skeleton_3d()
valid_boxes = frame.get_bboxes(valid_only=True)
```

## Directory Structure

```
src/classes/
├── __init__.py         # Package initialization
├── poses.py            # PosesData class
├── cameras.py          # CamerasData class
├── bboxes.py           # BBoxesData class
├── skeleton.py         # Skeleton2DData, Skeleton3DData classes
├── metadata.py         # ImageMetadata, VideoMetadata classes
└── README.md           # This file
```

## Additional Resources

- Main project README: `../../README.md`
- Visualization scripts: `../../scripts/visualization/`
- Evaluation pipeline: `../../scripts/evaluation/`

## Support

For questions or issues with these classes, please open an issue in the repository.
