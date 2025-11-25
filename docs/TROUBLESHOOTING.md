# Troubleshooting Guide

## Current Status

This document tracks the refactoring progress, known issues, and debugging procedures for the FIFA Skeletal Tracking Starter Kit.

---

## ✅ Completed Refactoring (2024-11-03)

### Major Restructuring

1. **Created OOP Architecture** (`src/classes/`)
   - ✅ `PosesData`: SMPL pose parameters handler
   - ✅ `CamerasData`: Camera calibration handler
   - ✅ `BBoxesData`: Bounding box handler with visualization
   - ✅ `Skeleton2DData`: 2D skeletal keypoints handler
   - ✅ `Skeleton3DData`: 3D skeletal keypoints handler
   - ✅ `ImageMetadata`: Unified frame-level interface
   - ✅ `VideoMetadata`: Unified sequence-level interface

2. **Created Utils Module** (`src/utils/`)
   - ✅ `ArgsParser`: Common CLI argument parsing (eliminates code duplication)
   - ✅ `SkeletonColorMapper`: Anatomically meaningful joint coloring
     - Blue tones → Torso/Spine
     - Green tones → Left side
     - Red tones → Right side

3. **Reorganized Evaluation Code**
   - ✅ Moved all evaluation code to `src/evaluation/`
   - ✅ Removed duplicate code from `scripts/evaluation/`
   - ✅ Fixed import paths (detectors, trackers)

4. **Updated Visualization Scripts**
   - ✅ `visualize_bboxes.py`: Simplified with ArgsParser
   - ✅ `visualize_2d_pose.py`: NEW - 2D skeleton overlay with color coding
   - ✅ `visualize_3d_pose.py`: Enhanced with color-coded joints and legend

5. **Code Quality Improvements**
   - ✅ Removed ~1,700 lines of duplicate code
   - ✅ Applied DRY principle throughout
   - ✅ One-line patterns instead of verbose if-else blocks
   - ✅ Fixed OpenCV color type issues (int conversion)
   - ✅ Added NaN value handling for invalid keypoints

---

## ⚠️ Known Issues

### 1. Joint Order Mapping Issue (RESOLVED ✅)

**Status**: RESOLVED (2024-11-03)

**Problem**: The initial SMPL joint name mapping did not match the actual joint positions in the output data.

**Evidence**:
- Joint 24 (currently labeled "nose") appears at ankle position in images
- Joint coordinates show inconsistent anatomical structure
- Example from debug output:
  ```
  24: nose at (187.3, 1014.7)  # Bottom of image - should be at top near head!
  15: head at (225.2, 931.3)   # Correct position at top
  ```

**Data Structure**:
- Input data: `skel_2d.npz` and `skel_3d.npz` contain 25 joints per subject
- Shape: `(num_frames, num_subjects, 25, 2)` for 2D, `(num_frames, num_subjects, 25, 3)` for 3D
- These are OUTPUT joints from 4D-Humans, not SMPL body_pose parameters
- body_pose parameters are 69-dimensional (23 joints × 3 axis-angle)

**Correct Mapping** (OpenPose BODY_25):
```python
JOINT_NAMES = [
    'nose',            # 0
    'neck',            # 1
    'right_shoulder',  # 2
    'right_elbow',     # 3
    'right_wrist',     # 4
    'left_shoulder',   # 5
    'left_elbow',      # 6
    'left_wrist',      # 7
    'mid_hip',         # 8
    'right_hip',       # 9
    'right_knee',      # 10
    'right_ankle',     # 11
    'left_hip',        # 12
    'left_knee',       # 13
    'left_ankle',      # 14
    'right_eye',       # 15
    'left_eye',        # 16
    'right_ear',       # 17
    'left_ear',        # 18
    'left_big_toe',    # 19
    'left_small_toe',  # 20
    'left_heel',       # 21
    'right_big_toe',   # 22
    'right_small_toe', # 23
    'right_heel'       # 24
]
```

**Source of Confusion**:
1. SMPL model has 24 body joints (indices 0-23)
2. 4D-Humans adds extra joints, outputting 25 total
3. PHALP has a `smpl_to_openpose` mapping: `[24, 12, 17, 19, 21, 16, 18, 20, 0, 2, 5, 8, 1, 4, 7, 25, 26, ...]`
4. This suggests output might be in OpenPose format or a custom format

**Solution**:
1. ✅ Identified that output format is **OpenPose BODY_25**, not raw SMPL
2. ✅ Found `smpl_to_openpose` mapping in PHALP: `[24, 12, 17, 19, 21, 16, 18, 20, 0, 2, 5, 8, 1, 4, 7, 25, ...]`
3. ✅ Updated `JOINT_NAMES` in `src/utils/skeleton_viz.py` with OpenPose BODY_25 format
4. ✅ Updated `SKELETON_CONNECTIONS` for OpenPose topology
5. ✅ Updated `JOINT_COLORS` with anatomical color scheme (Purple=face, Blue=torso, Green=left, Red=right)
6. ✅ Verified 2D and 3D skeleton visualizations are anatomically correct

**Key Insight**: The 4D-Humans/PHALP pipeline converts SMPL joints to OpenPose BODY_25 format before outputting to skel_2d.npz and skel_3d.npz. This is NOT the raw SMPL joint order!

**Debug Tools**:
```bash
# Visualize joints with numbers and names on white background
python scripts/visualization/debug_joint_mapping.py
# Output: /tmp/joint_mapping_debug.jpg

# Visualize joints overlaid on actual image
python scripts/visualization/debug_joint_order.py
# Output: /tmp/joint_numbers.jpg
```

---

## 🔍 Investigation Notes

### SMPL Joint Structure

**SMPL Body Model**:
- 24 joints total (index 0-23)
- Joint 0 (pelvis) is the root
- Body pose parameters: 23 joints × 3 × 3 = 207 params (rotation matrices)
- Or 23 joints × 3 = 69 params (axis-angle representation)

**Official SMPL Joint Order** (from smplx library):
```python
JOINT_NAMES = [
    'pelvis',           # 0
    'left_hip',         # 1
    'right_hip',        # 2
    'spine1',           # 3
    'left_knee',        # 4
    'right_knee',       # 5
    'spine2',           # 6
    'left_ankle',       # 7
    'right_ankle',      # 8
    'spine3',           # 9
    'left_foot',        # 10
    'right_foot',       # 11
    'neck',             # 12
    'left_collar',      # 13
    'right_collar',     # 14
    'head',             # 15
    'left_shoulder',    # 16
    'right_shoulder',   # 17
    'left_elbow',       # 18
    'right_elbow',      # 19
    'left_wrist',       # 20
    'right_wrist',      # 21
    'jaw',              # 22 (SMPL-X)
    'left_eye_smplhf',  # 23 (SMPL-X)
    'right_eye_smplhf', # 24 (SMPL-X)
    # ... more joints for hands and face in SMPL-X/H
]
```

**4D-Humans Output**:
- Outputs 25 joints (24 SMPL + 1 extra)
- Extra joint is likely nose or similar facial landmark
- Need to check 4D-Humans code for exact output format

**PHALP Mapping**:
- File: `/Users/soccerment/Desktop/VISIO/PHALP/phalp/utils/smpl_utils.py`
- Contains: `smpl_to_openpose = [24, 12, 17, 19, 21, 16, 18, 20, 0, 2, 5, 8, 1, 4, 7, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34]`
- This maps SMPL joint indices → OpenPose format
- But our data only has 25 joints, not 35

### Data Files Structure

**Poses Data** (`data/poses/<sequence>.npz`):
```python
Keys: ['global_orient', 'body_pose', 'transl', 'betas']
Shapes:
  - global_orient: (num_subjects, num_frames, 3)   # Root orientation
  - body_pose:     (num_subjects, num_frames, 69)  # 23 joints × 3 (axis-angle)
  - transl:        (num_subjects, num_frames, 3)   # Translation
  - betas:         (num_subjects, num_frames, 10)  # Shape parameters
```

**Skeleton Data** (`data/skel_2d.npz`, `data/skel_3d.npz`):
```python
Keys: [<sequence_names>]
Shapes (per sequence):
  - skel_2d: (num_frames, num_subjects, 25, 2)  # 2D pixel coordinates
  - skel_3d: (num_frames, num_subjects, 25, 3)  # 3D world coordinates
```

**Key Insight**: The skeleton data (skel_2d/skel_3d) are the OUTPUT joints after SMPL forward pass, not the input body_pose parameters!

---

## 🛠️ How to Fix Joint Mapping

### Step 1: Identify Correct Mapping

Run the debug script and examine the output:
```bash
python scripts/visualization/debug_joint_mapping.py
```

Check `/tmp/joint_mapping_debug.jpg` and identify which joint number corresponds to which body part.

### Step 2: Create Mapping

Based on visual inspection, create a mapping dictionary. For example:
```python
# If we find that output joints are in this order:
CORRECT_JOINT_NAMES = [
    'nose',            # 0 - if joint 0 is at head
    'neck',            # 1 - if joint 1 is below head
    'right_shoulder',  # 2 - and so on...
    # ... complete the mapping
]
```

### Step 3: Update Code

Update the following files:
1. `src/utils/skeleton_viz.py` - Update `JOINT_NAMES` list
2. `src/utils/skeleton_viz.py` - Update `SKELETON_CONNECTIONS` accordingly
3. `src/classes/skeleton.py` - Import updated names

### Step 4: Verify

Re-run visualizations:
```bash
# Check 2D skeleton on image
python scripts/visualization/visualize_2d_pose.py \
    --sequence ARG_CRO_225412 --frame 100 --show-labels

# Check 3D skeleton
python scripts/visualization/visualize_3d_pose.py \
    --sequence ARG_CRO_225412 --frame 100 --show-labels
```

Verify that:
- Joints appear at correct anatomical positions
- Connections form a proper skeleton
- Colors match body parts (blue=spine, green=left, red=right)

---

## 📚 Reference Resources

### Repositories
- **4D-Humans**: `/Users/soccerment/Desktop/VISIO/4D-Humans`
  - Check: `hmr2/models/`, `hmr2/datasets/` for output format
  - Look for joint regressor or output mapping code

- **PHALP**: `/Users/soccerment/Desktop/VISIO/PHALP`
  - File: `phalp/utils/smpl_utils.py` (contains smpl_to_openpose mapping)
  - File: `phalp/models/heads/smpl_head.py` (SMPL output generation)

### SMPL Resources
- **smplx library**: `.venv/lib/python3.12/site-packages/smplx/`
  - `joint_names.py`: Official SMPL joint names
  - `body_models.py`: SMPL forward pass implementation

- **Official SMPL**: https://smpl.is.tue.mpg.de/
- **SMPL-X**: https://smpl-x.is.tue.mpg.de/

### OpenPose Format
- 25 body keypoints in specific order
- Standard format: Nose, Neck, Shoulders, Elbows, Wrists, Hips, Knees, Ankles, Eyes, Ears, Feet

---

## 🔧 Debugging Commands

### Check Data Structure
```bash
# Check pose data shape
python -c "import numpy as np; d=np.load('data/poses/ARG_CRO_225412.npz'); print(d.files); print('body_pose:', d['body_pose'].shape)"

# Check skeleton data shape
python -c "import numpy as np; d=np.load('data/skel_3d.npz'); print(d.files); print('First seq:', d['ARG_CRO_225412'].shape)"

# List available sequences
python -c "import numpy as np; d=np.load('data/skel_2d.npz'); print('Sequences:', list(d.files))"
```

### Test Visualizations
```bash
# Test bounding boxes
python scripts/visualization/visualize_bboxes.py --sequence ARG_CRO_225412 --frame 100

# Test 2D skeleton
python scripts/visualization/visualize_2d_pose.py --sequence ARG_CRO_225412 --frame 100

# Test 3D skeleton
python scripts/visualization/visualize_3d_pose.py --sequence ARG_CRO_225412 --frame 100
```

### Debug Joint Order
```bash
# Detailed joint mapping on white background
python scripts/visualization/debug_joint_mapping.py

# Joint numbers overlaid on actual image
python scripts/visualization/debug_joint_order.py
```

---

## 📝 TODO

### High Priority (COMPLETED ✅)
- [x] Investigate 4D-Humans output joint ordering
- [x] Determine correct joint mapping for 25-joint output (OpenPose BODY_25)
- [x] Update `JOINT_NAMES` in `src/utils/skeleton_viz.py`
- [x] Update `SKELETON_CONNECTIONS` for correct topology
- [x] Verify all visualizations show correct anatomy

### Medium Priority
- [ ] Add unit tests for data classes
- [ ] Add integration tests for visualization scripts
- [ ] Create example notebooks demonstrating usage
- [ ] Add more color schemes for visualization

### Low Priority
- [ ] Optimize data loading performance
- [ ] Add caching for frequently accessed data
- [ ] Create video generation from frame sequences
- [ ] Add support for multiple color mapping schemes

---

## 💡 Tips

### Working with the Codebase

1. **Always activate virtual environment**:
   ```bash
   source .venv/bin/activate
   ```

2. **Import path setup** (for scripts):
   ```python
   import sys
   from pathlib import Path
   sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))
   ```

3. **Quick data exploration**:
   ```python
   from classes import VideoMetadata
   video = VideoMetadata.load(Path('data'), 'ARG_CRO_225412')
   frame = video.get_frame(100, load_image=True, images_dir=Path('data/images'))
   ```

4. **Color mapping**:
   ```python
   from utils import SkeletonColorMapper
   mapper = SkeletonColorMapper()
   color_bgr = mapper.get_joint_color(0, format='bgr')  # For OpenCV
   color_rgb = mapper.get_joint_color_normalized(0, format='rgb')  # For matplotlib
   ```

### Common Pitfalls

1. **NaN values**: Always check for NaN in keypoint data before processing
2. **Color types**: OpenCV requires integer tuples, use `tuple(int(c) for c in color)`
3. **Coordinate systems**: 2D is in pixels, 3D is in meters
4. **Frame indexing**: Zero-based, check `num_frames` before accessing

---

## 📞 Support

For questions or issues:
1. Check this TROUBLESHOOTING.md file
2. Review `src/classes/README.md` for API documentation
3. Check `src/README.md` for architecture overview
4. Open an issue on GitHub with:
   - Description of the problem
   - Steps to reproduce
   - Expected vs actual behavior
   - Debug output/screenshots

---

## 📅 Change Log

### 2024-11-03
- Created OOP architecture with data classes
- Added utils module (ArgsParser, SkeletonColorMapper)
- Reorganized evaluation pipeline
- Created visualization scripts (bboxes, 2D pose, 3D pose)
- Fixed OpenCV color type issues
- Added NaN handling
- **Identified and RESOLVED joint mapping issue**:
  - Discovered output format is OpenPose BODY_25, not raw SMPL
  - Updated JOINT_NAMES to match OpenPose BODY_25 format (25 joints)
  - Updated SKELETON_CONNECTIONS for OpenPose topology
  - Updated color scheme: Purple=face, Blue=torso, Green=left, Red=right
  - Verified all visualizations show correct anatomy

### 2024-11-02
- Initial repository structure
- Added preprocessing scripts
- Created baseline implementation

---

*Last updated: 2024-11-03*
