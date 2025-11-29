# FIFA Skeletal Tracking Starter Kit

## 🚀 Getting Started

### 📦 Installation
Make sure you have the required dependencies installed:

```bash
pip install numpy torch opencv-python tqdm scipy matplotlib
```

### 🏗️ Project Structure

```
Skeletal-Tracking-Starter-Kit/
├── data/                           # Dataset files
│   ├── videos/                     # Video files
│   ├── cameras/                    # Camera parameters
│   ├── poses/                      # SMPL pose parameters
│   ├── images/                     # Extracted frames
│   ├── boxes.npz                   # Bounding boxes
├── gui/                            # PyQt GUI application
│   ├── main.py                     # Main GUI application
│   ├── widgets/                    # GUI widgets
│   ├── utils/                      # GUI utilities
│   └── README.md                   # GUI documentation
├── src/                            # Source code
│   ├── classes/                    # Core OOP classes
│   │   ├── poses.py                # PosesData class
│   │   ├── cameras.py              # CamerasData class
│   │   ├── bboxes.py               # BBoxesData class
│   │   ├── skeleton.py             # Skeleton2DData, Skeleton3DData
│   │   ├── metadata.py             # ImageMetadata, VideoMetadata
│   │   └── README.md               # Classes documentation
│   └── evaluation/                 # Evaluation pipeline
│       ├── base_detector.py        # Detector base class
│       ├── base_tracker.py         # Tracker base class
│       ├── evaluation_pipeline.py  # Main evaluation pipeline
│       ├── detectors/              # Detector implementations
│       └── trackers/               # Tracker implementations
├── scripts/                        # Utility scripts
│   ├── preprocessing/              # Data preprocessing
│   ├── visualization/              # Visualization tools
│   └── evaluation/                 # Evaluation scripts
├── results/                        # Generated results
│   └── SMPL/                       # Animation files
├── baseline.py                     # Baseline implementation
├── GUI_QUICK_START.md              # GUI quick start guide
└── README.md                       # This file
```

## 🖥️ PyQt GUI Viewer

A synchronized video viewer for quality checking preprocessing scripts:

```bash
# Activate GUI environment
source .venv_gui/bin/activate

# Run viewer
python gui/main.py --sequence ARG_CRO_220001
```

Features:
- **Synchronized Display**: Original video, 3D poses, and pitch tracking in sync
- **Full Sequence Support**: Visualize all 22 players across complete clips
- **Playback Controls**: Play/pause, seek, step-by-step navigation, speed control
- **Keyboard Shortcuts**: Space (play/pause), arrows (navigate), F (fullscreen)

For details, see [GUI_QUICK_START.md](GUI_QUICK_START.md) and [gui/README.md](gui/README.md).

## 📂 Dataset Overview

### Understanding the Dataset Structure

The FIFA Skeletal Tracking Challenge dataset is divided into **three subsets** with different data availability:

| Subset | Videos | Cameras | Bounding Boxes | SMPL Poses (Raw) | 2D/3D Poses (Processed) | Purpose |
|--------|--------|---------|----------------|------------------|------------------------|---------|
| **TRAIN_DATA** | 89 | ✓ (89) | ✗ | ✓ (89) | ✗ | Training - requires preprocessing |
| **TEST_DATA** | 8 | ✓ (7) | ✓ (7) | ✗ | ✗ | Validation |
| **CHALLENGE_DATA** | 7 | ✓ (6) | ✓ (6) | ✗ | ✗ | Final evaluation |
| **TOTAL** | **104** | **102** | **13** | **89** | **13** | |

### Data Availability Details

#### 1. **TRAIN_DATA (89 sequences)**
**What you have:**
- ✓ Videos (`.mp4` files)
- ✓ Camera parameters (intrinsics `K`, distortion `k`, first frame `R` and `t`)
- ✓ Raw images (after running `extract_frames.py`)
- ✓ SMPL poses (raw SMPL parameters in `data/poses/<sequence>.npz`)

**What you DON'T have (need to generate):**
- ✗ Bounding boxes (can be generated from SMPL mesh projections)
- ✗ 2D skeletal keypoints (processed)
- ✗ 3D skeletal keypoints (processed)

**How to use:**
- Train your models on these sequences
- Generate bounding boxes from SMPL mesh projections (see `scripts/preprocessing/generate_boxes_from_smpl.py`)
- Generate processed 2D/3D poses using `preprocess.py` (requires 4D-Humans)
- Estimate camera poses for frames 2+ (only frame 1 R,t provided)

**Example sequences:** `ARG_CRO_220001`, `ARG_FRA_182345`, `BRA_CRO_210113`, etc.

#### 2. **TEST_DATA (8 sequences, 7 with full annotations)**
**What you have:**
- ✓ Videos (`.mp4` files)
- ✓ Camera parameters (7 sequences - missing [`NET_ARG_003203`](data/cameras/NET_ARG_003203.npz))
- ✓ Bounding boxes (7 sequences in [`boxes.npz`](results/boxes.npz) - missing [`NET_ARG_003203`](results/boxes.npz))

**What you DON'T have:**
- ✗ SMPL poses (NOT included in training data)
- ✗ 2D/3D skeletal keypoints (processed)

**Available sequences:**
- ✓ `ARG_CRO_000737` - 1042 frames, 21 subjects
- ✓ `ARG_FRA_183303` - 1944 frames, 22 subjects
- ✓ `BRA_KOR_230503` - 1812 frames, 22 subjects
- ✓ `CRO_MOR_190500` - 2022 frames, 23 subjects
- ✓ `ENG_FRA_223104` - 1878 frames, 23 subjects
- ✓ `FRA_MOR_220726` - 2149 frames, 23 subjects
- ✓ `MOR_POR_180940` - 1800 frames, 23 subjects
- ✗ `NET_ARG_003203` - No camera parameters or bounding boxes

**How to use:**
- Validate your approach on these sequences
- Compare your predictions against ground truth
- Develop camera tracking algorithms

#### 3. **CHALLENGE_DATA (7 sequences, 6 with full annotations)**
**What you have:**
- ✓ Videos (`.mp4` files)
- ✓ Camera parameters (6 sequences - missing [`CRO_MOR_182145`](data/cameras/CRO_MOR_182145.npz))
- ✓ Bounding boxes (6 sequences in [`boxes.npz`](results/boxes.npz) - missing [`CRO_MOR_182145`](results/boxes.npz))
- ✓ 2D skeletal keypoints (6 sequences, pre-generated in [`skel_2d.npz`](results/skel_2d.npz))
- ✓ 3D skeletal keypoints (6 sequences, pre-generated in [`skel_3d.npz`](results/skel_3d.npz))

**What you DON'T have:**
- ✗ SMPL poses (NOT included in training data)
- ✗ Camera parameters for [`CRO_MOR_182145`](data/cameras/CRO_MOR_182145.npz) (must be estimated)
- ✗ Bounding boxes for [`CRO_MOR_182145`](results/boxes.npz) (can be generated from SMPL if poses were available)

**Available sequences:**
- ✓ `ARG_CRO_225412` - 569 frames, 21 subjects
- ✓ `ARG_FRA_184210` - 987 frames, 23 subjects
- ✗ `CRO_MOR_182145` - No camera parameters or bounding boxes
- ✓ `ENG_FRA_231427` - 1060 frames, 22 subjects
- ✓ `MOR_POR_184642` - 968 frames, 23 subjects
- ✓ `MOR_POR_193202` - 685 frames, 19 subjects
- ✓ `NET_ARG_004041` - 904 frames, 23 subjects

**How to use:**
- Final evaluation sequences
- Submit predictions for these sequences

### Video Specifications

All videos in the FIFA Skeletal Tracking dataset have consistent technical specifications:

- **Frame Rate**: 50 FPS (frames per second)
- **Resolution**: 1920x1080 (Full HD)
- **Format**: MP4
- **Duration Range**: 11-60 seconds (varies by sequence)
- **Frame Count Range**: 569-3000+ frames (varies by sequence)

**Important for Data Understanding:**
- SMPL pose data has **frame-by-frame** information matching video frame count
- Data structure: `(num_subjects, num_frames, dimensions)` where subjects = ~20-25 players
- Pose animations automatically sync at 50 FPS to match video duration
- Example: 949 pose frames over 18.98s video = 50 FPS (perfect sync)

Use [`scripts/preprocessing/analyze_video_fps.py`](scripts/preprocessing/analyze_video_fps.py) to analyze specific sequences:

```bash
# Check video/pose consistency for a sequence
python scripts/preprocessing/analyze_video_fps.py --check-sequence ARG_FRA_180702

# Analyze all videos (sample)
python scripts/preprocessing/analyze_video_fps.py --sample-size 5
```

### Data Files Explained

#### Camera Parameters (`data/cameras/<sequence>.npz`)
Each file contains per-frame camera intrinsics and first-frame extrinsics:

```python
{
    "K": (num_frames, 3, 3),    # Intrinsic matrix per frame
    "k": (num_frames, 5),        # Distortion coefficients (only k1, k2 valid)
    "R": (1, 3, 3),              # Rotation matrix for FIRST FRAME ONLY
    "t": (1, 3),                 # Translation vector for FIRST FRAME ONLY
    "Rt": (1, 3, 4)              # Combined [R|t] for first frame
}
```

**Important:** You must estimate camera poses (R, t) for frames 2 onwards.

#### SMPL Poses (`data/poses/<sequence>.npz`)
Contains raw SMPL parameters for **89 training sequences only** (NOT available for test/challenge):

```python
{
    "global_orient": (num_frames, num_subjects, 3),  # Global orientation (axis-angle)
    "body_pose": (num_frames, num_subjects, 69),     # Body pose parameters (23 joints × 3)
    "transl": (num_frames, num_subjects, 3),         # Translation in world coordinates
    "betas": (num_frames, num_subjects, 10)          # Shape parameters
}
```

**Note:** These are SMPL model parameters from WorldPose dataset for training sequences only. You can:
- Project SMPL meshes to generate bounding boxes for training sequences
- Use SMPL parameters for pose estimation and tracking on training data
- Generate 2D/3D keypoints using SMPL joint locations for training
- **Test and challenge sequences do NOT have SMPL poses available**

#### Bounding Boxes (`data/boxes.npz`)
Contains bounding boxes for **13 sequences** (7 test + 6 challenge sequences):

```python
{
    "<sequence_name>": (num_frames, num_subjects, 4)
    # Format: XYXY (x_min, y_min, x_max, y_max)
    # np.nan indicates subject not present in frame
}
```

**Note:** Bounding boxes can be generated for training sequences by projecting SMPL meshes (see [`scripts/preprocessing/generate_boxes_from_smpl.py`](scripts/preprocessing/generate_boxes_from_smpl.py)). Missing sequences: [`NET_ARG_003203`](results/boxes.npz) and [`CRO_MOR_182145`](results/boxes.npz).

#### 2D Poses (`data/skel_2d.npz`)
Contains 2D skeletal keypoints for **13 sequences** (25 joints from 4D-Humans = SMPL 24 + nose):

```python
{
    "<sequence_name>": (num_frames, num_subjects, 25, 2)
    # 25 joints with (x, y) pixel coordinates
}
```

#### 3D Poses (`data/skel_3d.npz`)
Contains 3D skeletal keypoints for **13 sequences** (25 joints from 4D-Humans):

```python
{
    "<sequence_name>": (num_frames, num_subjects, 25, 3)
    # 25 joints with (x, y, z) world coordinates in meters
}
```

**Joint structure:** See [SMPL joint mapping](#smpl-joint-mapping) below.

### SMPL Joint Mapping

**4D-Humans Output (25 joints):** Used in `skel_2d.npz` and `skel_3d.npz`
- Joints 0-23: Standard SMPL joints
- Joint 24: Extra nose joint (added by 4D-Humans)

**Submission Format (15 joints):** Select from SMPL indices `[24, 17, 16, 19, 18, 21, 20, 2, 1, 5, 4, 8, 7, 11, 10]`
- Nose, shoulders, elbows, wrists, hips, knees, ankles, feet

See `scripts/README.md` for complete joint structure reference.

## 📂 Data Preparation

### Expected Directory Structure

```
data/
├── videos/
│   ├── train_data/          # 89 training videos
│   ├── test_data/           # 8 validation videos
│   └── challenge_data/      # 7 challenge videos
├── cameras/                 # 102 camera parameter files (89 train + 7 test + 6 challenge)
│   ├── ARG_CRO_220001.npz
│   ├── ARG_FRA_182345.npz
│   └── ...
├── images/                  # Extracted frames (generate with extract_frames.py)
│   ├── <sequence_name>/
│   │   ├── 00000.jpg
│   │   ├── 00001.jpg
│   │   └── ...
├── boxes.npz               # Bounding boxes (13 test+challenge sequences)
├── skel_2d.npz             # 2D poses (13 sequences, 25 joints)
├── skel_3d.npz             # 3D poses (13 sequences, 25 joints)
└── pitch_points.txt        # Field marking points for refinement
```

### Data Sources

- **Videos, Cameras, Bounding Boxes:** Download from [Kaggle Competition](https://www.kaggle.com/competitions/fifa-skeletal-light)
- **Preprocessed 2D/3D Poses:** Download from [Google Drive](https://drive.google.com/drive/folders/12bu0Xmp3-euajRxIxYO92HswWWUtH-u1?usp=sharing) OR generate using `python preprocess.py`
- **Images:** Generate locally using `python scripts/preprocessing/extract_frames.py`

### Quick Start Data Preparation

```bash
# 1. Extract frames from videos (creates images/ directory)
python scripts/preprocessing/extract_frames.py

# 2. Generate bounding boxes from SMPL poses (for training data)
python scripts/preprocessing/generate_boxes_simple.py --merge --limit 10

# 3. Validate data consistency (optional)
python scripts/preprocessing/validate_data.py

# 4. Visualize bounding boxes
python scripts/visualization/visualize_bboxes.py --sequence ARG_FRA_183303

# 5. Visualize 3D poses (for sequences with processed poses)
python scripts/visualization/visualize_3d_pose.py --sequence ARG_FRA_183303 --frame 100
```

## 🎯 Working with the Dataset (OOP Interface)

This repository provides an object-oriented interface for working with the dataset. Instead of manually loading NPZ files, you can use the provided classes for clean, reusable code.

### Quick Example

```python
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path.cwd() / 'src'))

from classes import VideoMetadata

# Load all data for a sequence
data_dir = Path('data')
video = VideoMetadata.load(data_dir, 'ARG_FRA_183303')

print(f"Sequence: {video.sequence_name}")
print(f"Frames: {video.num_frames}")
print(f"Components: {video}")

# Get a specific frame
frame = video.get_frame(
    frame_idx=100,
    load_image=True,
    images_dir=data_dir / 'images'
)

# Access frame data
bboxes = frame.get_bboxes()
skel_3d = frame.get_skeleton_3d()
poses = frame.get_poses_data()

# Visualize
img_with_boxes = frame.visualize_bboxes()
fig_3d = frame.visualize_skeleton_3d()
```

### Available Classes

- **PosesData**: SMPL pose parameters (global_orient, body_pose, transl, betas)
- **CamerasData**: Camera calibration (intrinsics K, k and extrinsics R, t)
- **BBoxesData**: Bounding boxes in XYXY format
- **Skeleton2DData**: 2D skeletal keypoints (25 joints)
- **Skeleton3DData**: 3D skeletal keypoints (25 joints)
- **ImageMetadata**: Aggregates all data for a single frame
- **VideoMetadata**: Aggregates all data for an entire sequence

For detailed documentation and examples, see [`src/classes/README.md`](src/classes/README.md).

### 📺 Sample Visualization
To help you visualize the results, we provide a short sample sequence in `media/sample.mp4`.

## 🤖 Object Detection Evaluation Pipeline

A flexible, plug-in based pipeline for evaluating object detection and tracking algorithms.

### Features

- **Commercial-Friendly Detectors**:
  - RT-DETR (BSD-3-Clause): High-performance transformer-based detector
  - YOLO ONNX (MIT): YOLO models via ONNX Runtime
- **ByteTrack Tracker** (MIT): State-of-the-art multi-object tracking
- **YAML Configuration**: Easy configuration and experimentation
- **Comprehensive Output**: Video visualization and prediction files

### Quick Start

```bash
# Install dependencies
source .venv/bin/activate
uv pip install torch torchvision scipy pyyaml

# Run RT-DETR evaluation
python scripts/evaluation/run_evaluation.py \
    --config scripts/evaluation/configs/rtdetr_example.yaml \
    --sequence ARG_FRA_183303 \
    --output-dir results/ARG_FRA_183303
```

For detailed documentation, see [`scripts/evaluation/README.md`](scripts/evaluation/README.md).

## 🔧 Running the Baseline
To run the baseline model on the dataset, simply execute:

```bash
python baseline.py
```

By default, the script reads from the data/ directory and generates a `.npz` file (`dummy-solution.npz`) in the root folder:

You can then use the `prepare-submission.py` to create a submission file:

```bash
python prepare-submission.py -i dummy-solution.npz
```

## 📌 Notes
- This is a **naïve baseline** — you are encouraged to improve the accuracy by refining camera estimation, leveraging better keypoint tracking, or integrating deep learning approaches.

## 🤝 Contributing
If you find a bug or have suggestions for improvements, feel free to submit a pull request or open an issue.

## Acknowledgement
We use [4DHuman](https://github.com/shubham-goel/4D-Humans/tree/main) in the `preprocess.py` for estimating both 2D and 3D skeletons from bounding boxes. We appreciate the contributions of the developers and the broader research community in advancing human pose estimation.

## 📜 License
This project is licensed under the MIT License.


# Further info
from: https://www.kaggle.com/competitions/fifa-skeletal-light/data


Dataset Description
We provide camera and bounding box data for both validation (val) and test sets.

Due to .npz format limitations with nested dictionaries, camera data is stored separately per sequence, while bounding boxes are merged into a single file.

Video Access
The video footage is owned by FIFA and requires an additional agreement for access. To request permission, please complete this form. After reviewing your application, we will send you a separate license agreement along with further access details.

If you have already requested video footage from the WorldPose Dataset, you do not need to apply again, as the validation and test videos were included in that distribution.

Camera
Each camera file is stored separately per sequence in .npz format with the following structure.

{
    # Intrinsic Matrix per frame
    "K": np.array of shape (number_frames, 3, 3),  
    # Distortion coefficients per frame (k1, k2, p1, p2, k3) here only k1, k2 are valid),
    "k": np.array of shape (number_frames, 5),  
    # Rotation matrix for the first frame,
    "R": np.array of shape (1, 3, 3),  
    # Translation vector for the first frame,
    "t": np.array of shape (1, 3), 
}
To simulate a realistic setting, we provide intrinsic parameters and distortion coefficients, as modern cameras (e.g., your iPhones) often support exporting them directly. However, we only provide rotation and translation parameters for the first frame to help define the coordinate system. Participants will need to track subsequent camera poses.

Boxes
Bounding boxes are stored in a single `.npz file structured as:

{
    "<sequence_name>": np.array of shape (number_frames, Num_subjects, 4)
    # Each entry represents a bounding box per frame and subject,
    # stored in XYXY format: (x_min, y_min, x_max, y_max),
    # where (x_min, y_min) is the top-left corner
    # and (x_max, y_max) is the bottom-right corner.
    # If a subject is not present in a given frame, its bounding box is set to np.nan.
} 
Submission
For submission, keypoints should be provided in a merged file, similar to bounding boxes. Since Kaggle does not support direct submission of .npz files, we provide a conversion script to help you to convert them to the .parquet format.

{
    "<sequence_name>": np.array of shape (number_frames, Num_subjects, 15, 3), 
    # Each entry represents 3D keypoints per frame and subject,
    # stored in a (15, 3) matrix with (x, y, z) coordinates
    # for 15 selected keypoints.

    # For keypoints, we select 15 joints from **SMPL's** joint set:
    # [24, 17, 16, 19, 18, 21, 20, 2, 1, 5, 4, 8, 7, 11, 10]
    # These joints, in order, correspond to:
    # - "nose"
    # - "right_shoulder", "left_shoulder"
    # - "right_elbow", "left_elbow"
    # - "right_wrist", "left_wrist"
    # - "right_hip", "left_hip"
    # - "right_knee", "left_knee"
    # - "right_ankle", "left_ankle"
    # - "right_foot", "left_foot"

    # Please ensure you use the **SMPL** model for conversion,
    # as SMPL-H and SMPL-X have different joint orders.
}
Please ensure that your submission follows the specified format for compatibility with the evaluation system. We also provide sample submission files for your reference.