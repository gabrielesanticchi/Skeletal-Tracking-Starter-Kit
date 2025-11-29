# Camera Calibration Data Structure Documentation

## Overview

The camera calibration data contains intrinsic and extrinsic camera parameters for the FIFA Skeletal Tracking Challenge. This data enables projection of 3D world coordinates to 2D image coordinates and camera pose estimation.

## File Format

**Location**: `data/cameras/<sequence_name>.npz`  
**Format**: NumPy compressed archive (`.npz`)  
**Availability**: 102 files (89 training + 7 test + 6 challenge sequences)

## Data Structure

Each camera file contains the following arrays:

```python
{
    "K": np.ndarray,   # (num_frames, 3, 3) - Intrinsic matrix per frame
    "k": np.ndarray,   # (num_frames, 5) - Distortion coefficients per frame
    "R": np.ndarray,   # (1, 3, 3) - Rotation matrix for FIRST FRAME ONLY
    "t": np.ndarray,   # (1, 3) - Translation vector for FIRST FRAME ONLY
    "Rt": np.ndarray   # (1, 3, 4) - Combined [R|t] for first frame
}
```

### Dimensions

- **num_frames**: Number of frames in the sequence (matches video frame count)

## Data Arrays

#### 1. `K` - Intrinsic Matrix (Per Frame)
- **Shape**: `(num_frames, 3, 3)`
- **Type**: Camera intrinsic parameters
- **Units**: Pixels
- **Description**: Camera calibration matrix for each frame

```python
K = [[fx,  s, cx],
     [ 0, fy, cy],
     [ 0,  0,  1]]
```

**Parameters:**
- `fx`, `fy`: Focal lengths in pixels
- `cx`, `cy`: Principal point coordinates (image center)
- `s`: Skew parameter (usually 0)

**Typical Values:**
- `fx`, `fy`: 1000-2000 pixels
- `cx`: ~960 pixels (half of 1920px width)
- `cy`: ~540 pixels (half of 1080px height)

#### 2. `k` - Distortion Coefficients (Per Frame)
- **Shape**: `(num_frames, 5)`
- **Type**: Radial and tangential distortion parameters
- **Units**: Dimensionless
- **Description**: Lens distortion correction parameters

```python
k = [k1, k2, p1, p2, k3]
```

**Parameters:**
- `k1`, `k2`, `k3`: Radial distortion coefficients
- `p1`, `p2`: Tangential distortion coefficients

**Note**: Only `k1` and `k2` are typically valid; `p1`, `p2`, `k3` are often zero.

#### 3. `R` - Rotation Matrix (First Frame Only)
- **Shape**: `(1, 3, 3)`
- **Type**: 3D rotation matrix
- **Units**: Dimensionless
- **Description**: Camera orientation for the first frame only

```python
R = [[r11, r12, r13],
     [r21, r22, r23],
     [r31, r32, r33]]
```

**Properties:**
- Orthogonal matrix: `R @ R.T = I`
- Determinant: `det(R) = 1`
- Converts world coordinates to camera coordinates

#### 4. `t` - Translation Vector (First Frame Only)
- **Shape**: `(1, 3)`
- **Type**: 3D translation vector
- **Units**: Meters
- **Description**: Camera position for the first frame only

```python
t = [tx, ty, tz]
```

**Interpretation:**
- Camera position in world coordinates: `-R.T @ t`
- Translation from world origin to camera center

#### 5. `Rt` - Combined Transformation (First Frame Only)
- **Shape**: `(1, 3, 4)`
- **Type**: Combined rotation and translation matrix
- **Units**: Mixed (rotation: dimensionless, translation: meters)
- **Description**: Combined `[R|t]` transformation matrix

```python
Rt = [R | t] = [[r11, r12, r13, tx],
                [r21, r22, r23, ty],
                [r31, r32, r33, tz]]
```

## Camera Model

### Projection Pipeline

1. **World to Camera Coordinates**:
   ```python
   X_cam = R @ X_world + t
   ```

2. **Camera to Normalized Coordinates**:
   ```python
   x_norm = X_cam[0] / X_cam[2]
   y_norm = X_cam[1] / X_cam[2]
   ```

3. **Apply Distortion**:
   ```python
   r2 = x_norm**2 + y_norm**2
   radial = 1 + k1*r2 + k2*r2**2 + k3*r2**3
   tangential_x = 2*p1*x_norm*y_norm + p2*(r2 + 2*x_norm**2)
   tangential_y = p1*(r2 + 2*y_norm**2) + 2*p2*x_norm*y_norm
   
   x_dist = x_norm * radial + tangential_x
   y_dist = y_norm * radial + tangential_y
   ```

4. **Normalized to Pixel Coordinates**:
   ```python
   u = fx * x_dist + s * y_dist + cx
   v = fy * y_dist + cy
   ```

### Coordinate Systems

#### World Coordinate System
- **Origin**: Football pitch center (approximately)
- **X-axis**: Along pitch length
- **Y-axis**: Along pitch width
- **Z-axis**: Vertical (upward)
- **Units**: Meters

#### Camera Coordinate System
- **Origin**: Camera optical center
- **X-axis**: Right (in image plane)
- **Y-axis**: Down (in image plane)
- **Z-axis**: Forward (optical axis)
- **Units**: Meters

#### Image Coordinate System
- **Origin**: Top-left corner
- **X-axis**: Right (columns)
- **Y-axis**: Down (rows)
- **Units**: Pixels
- **Range**: [0, 1920] × [0, 1080]

## Data Availability

### Training Sequences (89 files)
- **Complete camera data**: All parameters available
- **All frames**: Intrinsics and distortion for every frame
- **First frame extrinsics**: R, t, Rt for frame 0 only
- **Missing**: Extrinsics for frames 1+

### Test Sequences (7 files)
- **Available**: `ARG_CRO_000737`, `ARG_FRA_183303`, `BRA_KOR_230503`, `CRO_MOR_190500`, `ENG_FRA_223104`, `FRA_MOR_220726`, `MOR_POR_180940`
- **Missing**: `NET_ARG_003203`

### Challenge Sequences (6 files)
- **Available**: `ARG_CRO_225412`, `ARG_FRA_184210`, `ENG_FRA_231427`, `MOR_POR_184642`, `MOR_POR_193202`, `NET_ARG_004041`
- **Missing**: `CRO_MOR_182145`

## Usage Examples

### Loading Camera Data
```python
import numpy as np
from pathlib import Path
from src.classes import CamerasData

# Load single sequence
cameras = CamerasData.load(Path('data/cameras'), 'ARG_CRO_220001')

# Load all available sequences
cameras_dict = CamerasData.load_all(Path('data/cameras'))
```

### Accessing Camera Parameters
```python
# Get intrinsics for specific frame
K_frame = cameras.get_intrinsics(frame_idx=100)

# Get distortion coefficients
k_frame = cameras.get_distortion(frame_idx=100)

# Get first frame extrinsics
R, t = cameras.get_extrinsics(frame_idx=0)  # Only available for frame 0
```

### 3D to 2D Projection
```python
# Project 3D world point to 2D image coordinates
world_point = np.array([10.0, 5.0, 1.8])  # X, Y, Z in meters
image_point = cameras.project_3d_to_2d(world_point, frame_idx=0)
```

### Camera Pose Estimation
```python
# Estimate camera pose for frames 1+ (not provided)
# This is part of the challenge - participants must implement this
def estimate_camera_pose(frame_idx):
    # Your implementation here
    # Use feature tracking, SLAM, or other methods
    pass
```

## Camera Tracking Challenge

### Problem Statement
- **Given**: Intrinsics and distortion for all frames
- **Given**: Extrinsics (R, t) for first frame only
- **Task**: Estimate camera pose (R, t) for frames 1 through N

### Approaches
1. **Feature Tracking**: Track visual features across frames
2. **SLAM**: Simultaneous Localization and Mapping
3. **Bundle Adjustment**: Optimize camera poses and 3D points
4. **Deep Learning**: Neural network-based pose estimation

### Evaluation
- Compare estimated poses with ground truth (if available)
- Measure reprojection error for 3D points
- Assess temporal consistency of camera motion

## Technical Specifications

### Image Resolution
- **Width**: 1920 pixels
- **Height**: 1080 pixels
- **Aspect Ratio**: 16:9
- **Format**: Full HD

### Camera Properties
- **Type**: Broadcast camera (professional sports coverage)
- **Lens**: Variable focal length (zoom)
- **Distortion**: Moderate radial distortion
- **Frame Rate**: 50 FPS

### File Sizes
- Typical file size: 1-5 MB per sequence
- Depends on number of frames
- Compressed format for efficient storage

### Performance
- Loading time: ~10-50ms per sequence
- Memory usage: ~1-10MB per sequence in memory
- Projection speed: ~0.1ms per point

## Coordinate Transformations

### World to Image Pipeline
```python
def world_to_image(X_world, K, k, R, t):
    """Transform 3D world coordinates to 2D image coordinates."""
    
    # 1. World to camera coordinates
    X_cam = R @ X_world + t
    
    # 2. Perspective projection
    x_norm = X_cam[0] / X_cam[2]
    y_norm = X_cam[1] / X_cam[2]
    
    # 3. Apply distortion
    r2 = x_norm**2 + y_norm**2
    radial = 1 + k[0]*r2 + k[1]*r2**2 + k[4]*r2**3
    tangential_x = 2*k[2]*x_norm*y_norm + k[3]*(r2 + 2*x_norm**2)
    tangential_y = k[2]*(r2 + 2*y_norm**2) + 2*k[3]*x_norm*y_norm
    
    x_dist = x_norm * radial + tangential_x
    y_dist = y_norm * radial + tangential_y
    
    # 4. Apply intrinsics
    u = K[0,0] * x_dist + K[0,1] * y_dist + K[0,2]
    v = K[1,1] * y_dist + K[1,2]
    
    return np.array([u, v])
```

### Image to World (Inverse)
```python
def image_to_world_ray(u, v, K, k, R, t):
    """Get 3D ray from image coordinates (requires depth for full 3D point)."""
    
    # 1. Pixel to normalized coordinates
    x_norm = (u - K[0,2]) / K[0,0]
    y_norm = (v - K[1,2]) / K[1,1]
    
    # 2. Undistort (iterative process)
    x_undist, y_undist = undistort_point(x_norm, y_norm, k)
    
    # 3. Camera to world ray direction
    ray_cam = np.array([x_undist, y_undist, 1.0])
    ray_world = R.T @ ray_cam
    
    # 4. Camera position in world coordinates
    camera_pos = -R.T @ t
    
    return camera_pos, ray_world
```

## Related Documentation

- [POSES.md](POSES.md) - SMPL pose data structure
- [src/classes/README.md](../src/classes/README.md) - Object-oriented interface
- [scripts/README.md](../scripts/README.md) - Visualization scripts
- [README.md](../README.md) - Main project documentation

## References

- [OpenCV Camera Calibration](https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html)
- [Multiple View Geometry](http://www.robots.ox.ac.uk/~vgg/hzbook/) - Hartley & Zisserman
- [Computer Vision: Algorithms and Applications](http://szeliski.org/Book/) - Szeliski