# FIFA Skeletal Tracking Starter Kit

This repository provides a **naïve baseline** for the **FIFA Skeletal Tracking Challenge**. It includes a simple, fully documented implementation to help participants get started with 3D pose estimation using bounding boxes, skeletal data, and camera parameters.

## 📌 Features
- **Baseline Implementation**: A simple approach for 3D skeletal tracking.
- **Camera Pose Estimation**: Computes camera transformations from bounding box correspondences.
- **Field Markings Refinement**: Improves camera rotation using detected Field Markings.
- **Pose Projection & Optimization**: Projects 3D skeletons onto 2D images and refines translation via optimization.

## 🚀 Getting Started

### 📦 Installation
Make sure you have the required dependencies installed:

```bash
pip install numpy torch opencv-python tqdm scipy
```

## 📂 Dataset Overview

### Understanding the Dataset Structure

The FIFA Skeletal Tracking Challenge dataset is divided into **three subsets** with different data availability:

| Subset | Videos | Cameras | Bounding Boxes | 2D Poses | 3D Poses | Purpose |
|--------|--------|---------|----------------|----------|----------|---------|
| **TRAIN_DATA** | 89 | ✓ | ✗ | ✗ | ✗ | Training - requires preprocessing |
| **TEST_DATA** | 8 | ✗ | ✓ (7/8) | ✓ (7/8) | ✓ (7/8) | Validation with ground truth |
| **CHALLENGE_DATA** | 7 | ✗ | ✓ (6/7) | ✓ (6/7) | ✓ (6/7) | Final evaluation |
| **TOTAL** | **104** | **89** | **13** | **13** | **13** | |

### Data Availability Details

#### 1. **TRAIN_DATA (89 sequences)**
**What you have:**
- ✓ Videos (`.mp4` files)
- ✓ Camera parameters (intrinsics `K`, distortion `k`, first frame `R` and `t`)
- ✓ Raw images (after running `extract_frames.py`)

**What you DON'T have (need to generate):**
- ✗ Bounding boxes
- ✗ 2D skeletal poses
- ✗ 3D skeletal poses

**How to use:**
- Train your models on these sequences
- Generate poses using `preprocess.py` (requires 4D-Humans)
- Estimate camera poses for frames 2+ (only frame 1 R,t provided)

**Example sequences:** `ARG_CRO_220001`, `ARG_FRA_182345`, `BRA_CRO_210113`, etc.

#### 2. **TEST_DATA (8 sequences, 7 with annotations)**
**What you have:**
- ✓ Videos (`.mp4` files)
- ✓ Bounding boxes (7 sequences)
- ✓ 2D skeletal poses (7 sequences, pre-generated)
- ✓ 3D skeletal poses (7 sequences, pre-generated)

**What you DON'T have:**
- ✗ Camera parameters (except first frame - you must track camera pose)

**Available sequences:**
- ✓ `ARG_CRO_000737` - 1042 frames, 21 subjects
- ✓ `ARG_FRA_183303` - 1944 frames, 22 subjects
- ✓ `BRA_KOR_230503` - 1812 frames, 22 subjects
- ✓ `CRO_MOR_190500` - 2022 frames, 23 subjects
- ✓ `ENG_FRA_223104` - 1878 frames, 23 subjects
- ✓ `FRA_MOR_220726` - 2149 frames, 23 subjects
- ✓ `MOR_POR_180940` - 1800 frames, 23 subjects
- ✗ `NET_ARG_003203` - No annotations

**How to use:**
- Validate your approach on these sequences
- Compare your predictions against ground truth
- Develop camera tracking algorithms

#### 3. **CHALLENGE_DATA (7 sequences, 6 with annotations)**
**What you have:**
- ✓ Videos (`.mp4` files)
- ✓ Bounding boxes (6 sequences)
- ✓ 2D skeletal poses (6 sequences, pre-generated)
- ✓ 3D skeletal poses (6 sequences, pre-generated)

**What you DON'T have:**
- ✗ Camera parameters (must be estimated)

**Available sequences:**
- ✓ `ARG_CRO_225412` - 569 frames, 21 subjects
- ✓ `ARG_FRA_184210` - 987 frames, 23 subjects
- ✗ `CRO_MOR_182145` - No annotations
- ✓ `ENG_FRA_231427` - 1060 frames, 22 subjects
- ✓ `MOR_POR_184642` - 968 frames, 23 subjects
- ✓ `MOR_POR_193202` - 685 frames, 19 subjects
- ✓ `NET_ARG_004041` - 904 frames, 23 subjects

**How to use:**
- Final evaluation sequences
- Submit predictions for these sequences

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

#### Bounding Boxes (`data/boxes.npz`)
Contains bounding boxes for 13 sequences (test + challenge data):

```python
{
    "<sequence_name>": (num_frames, num_subjects, 4)
    # Format: XYXY (x_min, y_min, x_max, y_max)
    # np.nan indicates subject not present in frame
}
```

#### 2D Poses (`data/skel_2d.npz`)
Contains 2D skeletal keypoints (25 joints from 4D-Humans = SMPL 24 + nose):

```python
{
    "<sequence_name>": (num_frames, num_subjects, 25, 2)
    # 25 joints with (x, y) pixel coordinates
}
```

#### 3D Poses (`data/skel_3d.npz`)
Contains 3D skeletal keypoints (25 joints from 4D-Humans):

```python
{
    "<sequence_name>": (num_frames, num_subjects, 25, 3)
    # 25 joints with (x, y, z) world coordinates in meters
}
```

**Joint structure:** See [SMPL joint mapping](#smpl-joint-mapping) below.

### What Can You Do With Each Subset?

#### Training Workflow (TRAIN_DATA):
1. Extract frames: `python scripts/preprocessing/extract_frames.py`
2. Generate poses: `python preprocess.py` (requires 4D-Humans setup)
3. Train your camera tracking models
4. Train your pose estimation models

#### Validation Workflow (TEST_DATA):
1. Visualize bounding boxes: `python scripts/visualization/visualize_bboxes.py --sequence ARG_FRA_183303`
2. Visualize 3D poses: `python scripts/visualization/visualize_3d_pose.py --sequence ARG_FRA_183303`
3. Run baseline: `python baseline.py`
4. Validate your approach against provided ground truth

#### Submission Workflow (CHALLENGE_DATA):
1. Run your model on challenge sequences
2. Generate predictions in submission format (15 joints)
3. Prepare submission: `python prepare-submission.py -i your-solution.npz`
4. Submit to Kaggle

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
├── cameras/                 # 89 camera parameter files (train only)
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

# 2. Validate data consistency (optional)
python scripts/preprocessing/validate_data.py

# 3. Visualize bounding boxes (test subset only)
python scripts/visualization/visualize_bboxes.py --sequence ARG_FRA_183303

# 4. Visualize 3D poses (test subset only)
python scripts/visualization/visualize_3d_pose.py --sequence ARG_FRA_183303 --frame 100
```

### 📺 Sample Visualization
To help you visualize the results, we provide a short sample sequence in `media/sample.mp4`. 

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