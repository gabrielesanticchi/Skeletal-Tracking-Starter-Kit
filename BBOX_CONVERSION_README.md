# Bounding Box Generation from 2D Skeleton Keypoints

This document explains the approach used to generate accurate bounding boxes from 2D skeleton keypoints for the FIFA Skeletal Tracking Challenge dataset.

## Overview

The conversion from poses to 2D bounding boxes uses the existing 2D skeleton keypoints to create tight bounding boxes around detected subjects. This approach is more reliable than SMPL mesh projection and provides high-quality results.

## Method: 2D Skeleton Keypoints to Bounding Boxes

### Input Data
- **2D Skeleton Keypoints**: `data/skel_2d.npz` containing (num_frames, num_subjects, 25, 2) arrays
- **25 Keypoints**: SMPL skeleton with 24 joints + nose (from 4D-Humans)
- **Image Dimensions**: 1920x1080 pixels (standard for FIFA dataset)

### Conversion Process

#### 1. **Keypoint Validation**
```python
# Filter valid keypoints (non-zero, finite values)
valid_mask = (
    np.isfinite(keypoints).all(axis=1) & 
    ~(keypoints == 0).all(axis=1)
)
```

#### 2. **Bounds Checking**
```python
# Ensure keypoints are within image boundaries
in_bounds_mask = (
    (valid_keypoints[:, 0] >= 0) & 
    (valid_keypoints[:, 0] < image_width) &
    (valid_keypoints[:, 1] >= 0) & 
    (valid_keypoints[:, 1] < image_height)
)
```

#### 3. **Bounding Box Calculation**
```python
# Compute tight bounding box from valid keypoints
x_min, y_min = bounded_keypoints.min(axis=0)
x_max, y_max = bounded_keypoints.max(axis=0)
```

#### 4. **Size Validation**
```python
# Ensure minimum bounding box size (30 pixels)
width = x_max - x_min
height = y_max - y_min
if width < min_bbox_size or height < min_bbox_size:
    return invalid_bbox
```

#### 5. **Margin Addition**
```python
# Add 10% margin around the bounding box
if margin > 0:
    x_margin = width * margin  # 10% of width
    y_margin = height * margin  # 10% of height
    
    x_min -= x_margin
    y_min -= y_margin
    x_max += x_margin
    y_max += y_margin
```

#### 6. **Final Clamping**
```python
# Clamp to image boundaries
x_min = max(0, x_min)
y_min = max(0, y_min)
x_max = min(image_width, x_max)
y_max = min(image_height, y_max)
```

### Quality Metrics

Based on comparison with original manual annotations:

- **Mean IoU**: 0.818 (82% overlap with ground truth)
- **High Quality Ratio**: 99.2% of boxes have IoU > 0.5
- **Coverage**: 99.8% of valid subjects get bounding boxes
- **Valid Box Rate**: ~46% average across all sequences

### Advantages of This Approach

1. **High Accuracy**: 82% IoU with manual annotations
2. **Reliable**: No dependency on SMPL models or complex 3D projections
3. **Fast**: Direct computation from existing 2D keypoints
4. **Robust**: Handles invalid/missing keypoints gracefully
5. **Consistent**: Works across all sequences with 2D skeleton data

### Limitations

- **Coverage**: Only works for sequences with 2D skeleton data (13 out of 89 sequences)
- **Dependency**: Requires high-quality 2D pose estimation
- **Tight Boxes**: May be tighter than manual annotations (which sometimes include more context)

## Usage

### Generate Bounding Boxes
```bash
# Generate for all available sequences
python scripts/preprocessing/generate_boxes_from_2d_skeleton.py --output data/boxes_accurate.npz

# Generate for specific sequences with debug info
python scripts/preprocessing/generate_boxes_from_2d_skeleton.py --sequences ARG_FRA_183303 --debug

# Adjust margin (default 0.1 = 10%)
python scripts/preprocessing/generate_boxes_from_2d_skeleton.py --margin 0.15
```

### Visualize Results
```bash
# Create animated video with bounding boxes
python scripts/visualization/animate_bboxes.py --sequence ARG_FRA_183303 --boxes-file boxes_accurate.npz --output result.mp4

# Compare with original annotations
python scripts/visualization/compare_bboxes.py --original-boxes data/boxes.npz --generated-boxes data/boxes_accurate.npz
```

## File Structure

```
scripts/
├── preprocessing/
│   └── generate_boxes_from_2d_skeleton.py    # Main conversion script
└── visualization/
    ├── animate_bboxes.py                      # Create MP4 videos with bboxes
    ├── visualize_bboxes.py                    # Single frame visualization
    └── compare_bboxes.py                      # Quality analysis tool
```

## Technical Details

### Coordinate System
- **Origin**: Top-left corner (0, 0)
- **X-axis**: Left to right (0 to 1920)
- **Y-axis**: Top to bottom (0 to 1080)
- **Format**: [x_min, y_min, x_max, y_max]

### Data Format
- **Input**: 2D keypoints (num_frames, num_subjects, 25, 2)
- **Output**: Bounding boxes (num_frames, num_subjects, 4)
- **Invalid boxes**: Filled with NaN values

### Performance
- **Processing Speed**: ~1000 frames/second
- **Memory Usage**: Minimal (processes frame by frame)
- **Output Size**: ~3.6MB for 13 sequences

## Future Improvements

1. **Extend Coverage**: Generate 2D skeletons for remaining 76 sequences
2. **Adaptive Margins**: Use different margins based on pose confidence
3. **Temporal Smoothing**: Apply smoothing across frames for stability
4. **Multi-person Handling**: Better handling of crowded scenes

This approach provides a reliable, accurate method for generating bounding boxes from pose data without the complexity and issues of 3D mesh projection approaches.