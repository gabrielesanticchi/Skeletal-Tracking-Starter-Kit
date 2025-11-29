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
2. **Apply body pose rotations** using Rodrigues formula (axis-angle to rotation matrix)
3. **Compute forward kinematics** through joint hierarchy
4. **Add translation** to get world coordinates

The forward kinematics is implemented in `src/visualization/poses_viz.py` with the `compute_smpl_joints()` function, which properly propagates rotations through the SMPL joint hierarchy for realistic pose animations.

### Coordinate System Conversions

#### Center-Origin Coordinate System (Default)

The raw `transl` data uses a **center-origin coordinate system** where (0,0,0) is at the center of the football pitch:

- **Origin**: Center of the pitch, ground level
- **Range**:
  - X: approximately -52.5 to +52.5 meters (pitch length: 105m)
  - Y: approximately -34.0 to +34.0 meters (pitch width: 68m)
  - Z: 0.0 to +2.0 meters (ground to head height)

This is the **native coordinate system** of the data and no transformation is needed to use it.

#### Bottom-Left Corner Origin (For Plotting)

For visualization purposes, you may want coordinates with origin at the bottom-left corner of the pitch, resulting in all-positive values. Use the `convert_coords_from_center_to_bl_corner()` method:

```python
# Get pitch coordinates in center-origin system
coords_center = poses.get_pitch_coordinates(frame_idx=100)

# Convert to bottom-left corner origin
coords_bl = poses.convert_coords_from_center_to_bl_corner(coords_center)

# Now coords_bl has:
# - X range: 0 to 105 meters
# - Y range: 0 to 68 meters
```

**Transformation Formula:**
```
X_bl = X_center + 52.5  (half pitch length)
Y_bl = Y_center + 34.0  (half pitch width)
Z_bl = Z_center         (unchanged)
```

**Important Notes:**

- The visualization module (`src/visualization/poses_viz.py`) works with the **center-origin system** directly
- Pitch outline is drawn centered at (0,0) matching the data's native coordinate system
- No offset calculations or caching are needed - the data is already correctly referenced to the pitch center
- Use `convert_coords_from_center_to_bl_corner()` only when you specifically need bottom-left origin coordinates

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

# Get pitch coordinates (X, Y only) in center-origin system
pitch_coords = poses.get_pitch_coordinates(frame_idx=100)  # (num_subjects, 2)

# Convert to bottom-left corner origin if needed
pitch_coords_bl = poses.convert_coords_from_center_to_bl_corner(pitch_coords)
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

**Note:** Visualization functionality is implemented in the separate `src/visualization` module following OOP principles (separation of data manipulation and visualization). The `PosesData` class provides thin wrapper methods that delegate to the visualization module.

### 3D Pose Visualization
```python
# Static visualization
fig = poses.visualize_3d_poses(frame_idx=100, elev=45, azim=45)

# Animated visualization with improved joint movements
fig = poses.animate_3d_poses(start_frame=0, end_frame=100, fps=50)

# Direct use of visualization module
from src.visualization import poses_viz
fig = poses_viz.visualize_3d_poses(poses, frame_idx=100)
```

### Pitch Tracking
```python
# Static pitch tracking (uses center-origin coordinates)
fig = poses.visualize_pitch_tracking(start_frame=0, end_frame=100)

# Animated pitch tracking (uses center-origin coordinates)
fig = poses.animate_pitch_tracking(start_frame=0, end_frame=100, fps=50)

# Direct use of visualization module
from src.visualization import poses_viz
fig = poses_viz.animate_pitch_tracking(poses, start_frame=0, end_frame=100)
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
- [src/visualization/](../src/visualization/) - Visualization module for poses data
- [scripts/README.md](../scripts/README.md) - Visualization scripts
- [README.md](../README.md) - Main project documentation

## Architecture Notes

### Separation of Concerns

The codebase follows OOP principles by separating data manipulation from visualization:

- **`src/classes/poses.py`**: Data loading, access, and manipulation
  - SMPL parameter storage and access
  - Coordinate system conversions
  - Trajectory extraction
  - Thin wrapper methods that delegate to visualization module

- **`src/visualization/poses_viz.py`**: Visualization and animation
  - Forward kinematics implementation
  - 3D pose visualization
  - Pitch tracking visualization
  - Animation generation
  - Pitch outline drawing

This separation makes the code more maintainable, testable, and follows the single responsibility principle.