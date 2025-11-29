# SMPL Poses Data Structure Documentation

## Overview

The SMPL poses data contains raw pose parameters from the WorldPose dataset for the FIFA Skeletal Tracking Challenge. This data represents human body poses using the SMPL (Skinned Multi-Person Linear) model parameters.

## File Format

**Location**: `data/poses/<sequence_name>.npz`  
**Format**: NumPy compressed archive (`.npz`)  
**Availability**: Training sequences only (89 files)

## Data Structure

Each pose file contains the following arrays:

```python
{
    "global_orient": np.ndarray,  # (num_subjects, num_frames, 3)
    "body_pose": np.ndarray,      # (num_subjects, num_frames, 69)
    "transl": np.ndarray,         # (num_subjects, num_frames, 3)
    "betas": np.ndarray           # (num_subjects, 10)
}
```

### Dimensions

- **num_subjects**: Number of players in the sequence (~20-25 players)
- **num_frames**: Number of frames in the sequence (matches video frame count)

### Data Arrays

#### 1. `global_orient` - Global Orientation
- **Shape**: `(num_subjects, num_frames, 3)`
- **Type**: Axis-angle representation
- **Units**: Radians
- **Description**: Global rotation of the root joint (pelvis) in world coordinates
- **Range**: [-π, π] for each axis

#### 2. `body_pose` - Body Pose Parameters
- **Shape**: `(num_subjects, num_frames, 69)`
- **Type**: Axis-angle representation
- **Units**: Radians
- **Description**: Pose parameters for 23 body joints (23 × 3 = 69 parameters)
- **Joint Order**: SMPL standard joint hierarchy
- **Range**: [-π, π] for each parameter

#### 3. `transl` - Translation Vectors
- **Shape**: `(num_subjects, num_frames, 3)`
- **Type**: 3D world coordinates
- **Units**: Meters
- **Description**: Position of the root joint (pelvis) in world coordinates
- **Coordinate System**: 
  - X: Horizontal (pitch length direction)
  - Y: Horizontal (pitch width direction)  
  - Z: Vertical (height above ground)
  - (0,0,0): represents the center of the football pitch, ground floor. 
    - (10,0,0): represents a shift of 10m towards right direction, along the length of the pitch.
    - (0,10,0): represents a shift of 10m towards top direction, along the width of the pitch.
    - (0,0,10): represents a shift of 10m towards height direction, along the vertical axis.

#### 4. `betas` - Shape Parameters
- **Shape**: `(num_subjects, 10)`
- **Type**: Shape coefficients
- **Units**: Dimensionless
- **Description**: SMPL shape parameters defining body shape variations
- **Note**: Constant per subject (not time-varying)

## SMPL Joint Hierarchy

The SMPL model uses 24 joints in a hierarchical structure:

```
0:  pelvis (root)
├── 1:  left_hip
│   ├── 4:  left_knee
│   │   ├── 7:  left_ankle
│   │   │   └── 10: left_foot
├── 2:  right_hip
│   ├── 5:  right_knee
│   │   ├── 8:  right_ankle
│   │   │   └── 11: right_foot
└── 3:  spine1
    └── 6:  spine2
        └── 9:  spine3
            ├── 12: neck
            │   └── 15: head
            ├── 13: left_collar
            │   └── 16: left_shoulder
            │       └── 18: left_elbow
            │           └── 20: left_wrist
            │               └── 22: left_hand
            └── 14: right_collar
                └── 17: right_shoulder
                    └── 19: right_elbow
                        └── 21: right_wrist
                            └── 23: right_hand
```

## Coordinate System

### World Coordinates
- **Origin**: Football pitch center (approximately)
- **X-axis**: Along pitch length (goal to goal)
- **Y-axis**: Along pitch width (sideline to sideline)
- **Z-axis**: Vertical (ground to sky)
- **Units**: Meters

### Typical Value Ranges
- **X**: -52.5 to +52.5 meters (pitch length: 105m)
- **Y**: -34.0 to +34.0 meters (pitch width: 68m)
- **Z**: 0.0 to +2.0 meters (ground to head height)

## Data Transformations

### Forward Kinematics
To compute 3D joint positions from SMPL parameters:

1. **Apply global orientation** to root joint
2. **Apply body pose rotations** using Rodrigues formula
3. **Compute forward kinematics** through joint hierarchy
4. **Add translation** to get world coordinates

### Coordinate Centering

The visualization system applies coordinate transformation to center the pitch at (0,0) using the **data mean center** approach.

**Method: Data Mean Center**

This method is more robust than using a specific player reference (e.g., S8), as player positions vary significantly between sequences. The data mean approach:

1. **Samples positions** across the sequence (up to 200 frames, evenly distributed)
2. **Calculates the mean** of all valid player positions
3. **Applies the offset** to center coordinates at (0, 0)

```python
# Calculate data mean center
all_coords = []
sample_frames = range(0, min(num_frames, 200), max(1, num_frames // 20))

for frame_idx in sample_frames:
    for subj_idx in range(num_subjects):
        coord = transl[subj_idx, frame_idx, :2]  # X, Y only
        if not (np.isnan(coord).any() or np.isinf(coord).any()):
            all_coords.append(coord)

# Calculate offsets
offset_x = np.mean(all_coords[:, 0])
offset_y = np.mean(all_coords[:, 1])

# Apply transformation
pitch_coords_x = raw_coords_x - offset_x
pitch_coords_y = raw_coords_y - offset_y
```

**Why Data Mean Center?**

- **Universal**: Works consistently across all sequences
- **Robust**: Not dependent on specific player positions
- **Representative**: Samples across entire sequence for accuracy
- **Validated**: Analysis shows average 7.46m from origin vs 13.25m for player-based references

**Implementation Note:**

The `PosesData` class caches the calculated offset in `_pitch_offset_x` and `_pitch_offset_y` for efficiency. The offset is computed once on first use and reused for all subsequent coordinate transformations.

## Usage Examples

### Loading Pose Data
```python
import numpy as np
from pathlib import Path
from src.classes import PosesData

# Load single sequence
poses = PosesData.load(Path('data/poses'), 'ARG_CRO_220001')

# Load all sequences
poses_dict = PosesData.load_all(Path('data/poses'))
```

### Accessing Data
```python
# Get frame data for all subjects
frame_data = poses.get_frame_data(frame_idx=100)

# Get specific subject data
subject_data = poses.get_frame_data(frame_idx=100, subject_idx=0)

# Get trajectory for subject
trajectory = poses.get_subject_trajectory(subject_idx=0)
```

### Computing Joint Positions
```python
# Get 3D joint positions using forward kinematics
joints_3d = poses.get_smpl_joints(frame_idx=100)  # (num_subjects, 24, 3)

# Get pitch coordinates (X, Y only)
pitch_coords = poses.get_pitch_coordinates(frame_idx=100)  # (num_subjects, 2)
```

## Data Quality Notes

### Valid Data
- All training sequences have complete pose data
- Frame counts match video frame counts exactly
- Data is synchronized at 50 FPS

### Missing Data
- Test and challenge sequences do NOT have pose data
- Some subjects may have NaN values for certain frames
- Shape parameters (betas) are constant per subject

### Data Validation
```python
# Check for valid data
valid_mask = ~(np.isnan(poses.transl).any(axis=2) | 
               np.isinf(poses.transl).any(axis=2))

# Filter valid subjects per frame
valid_subjects = np.where(valid_mask[:, frame_idx])[0]
```

## Visualization

### 3D Pose Visualization
```python
# Static visualization
fig = poses.visualize_3d_poses(frame_idx=100, elev=45, azim=45)

# Animated visualization
fig = poses.animate_3d_poses(start_frame=0, end_frame=100, fps=50)
```

### Pitch Tracking
```python
# Static pitch tracking
fig = poses.visualize_pitch_tracking(start_frame=0, end_frame=100)

# Animated pitch tracking
fig = poses.animate_pitch_tracking(start_frame=0, end_frame=100, fps=50)
```

## Technical Specifications

### File Sizes
- Typical file size: 5-15 MB per sequence
- Depends on number of subjects and frames
- Compressed format for efficient storage

### Performance
- Loading time: ~100-500ms per sequence
- Memory usage: ~50-200MB per sequence in memory
- Forward kinematics: ~1-5ms per frame

### Dependencies
- NumPy for array operations
- Matplotlib for visualization
- SciPy for rotation operations (Rodrigues formula)

## Related Documentation

- [CAMERAS.md](CAMERAS.md) - Camera calibration data structure
- [src/classes/README.md](../src/classes/README.md) - Object-oriented interface
- [scripts/README.md](../scripts/README.md) - Visualization scripts
- [README.md](../README.md) - Main project documentation