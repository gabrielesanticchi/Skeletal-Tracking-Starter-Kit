# Scripts Directory

This directory contains utility scripts for preprocessing and visualizing data from the FIFA Skeletal Tracking Challenge.

## Directory Structure

```
scripts/
├── preprocessing/     # Data preprocessing and validation scripts
│   ├── validate_data.py
│   └── extract_frames.py
└── visualization/     # Visualization tools
    ├── visualize_bboxes.py
    └── visualize_3d_pose.py
```

## Preprocessing Scripts

### validate_data.py

Validates consistency between video files and bounding box data.

**Usage:**
```bash
# From project root
python scripts/preprocessing/validate_data.py

# From preprocessing directory
cd scripts/preprocessing
python validate_data.py
```

**What it does:**
- Checks that video frame counts match bounding box frame counts
- Validates all sequences across train_data, test_data, and challenge_data
- Reports any mismatches or missing data

### extract_frames.py

Extracts frames from video files and saves them as sequential JPEG images.

**Usage:**
```bash
# Extract all sequences
python scripts/preprocessing/extract_frames.py

# Extract specific sequences
python scripts/preprocessing/extract_frames.py --sequences ARG_FRA_183303 BRA_KOR_230503

# Extract with FPS limiting (e.g., 10 FPS)
python scripts/preprocessing/extract_frames.py --fps-limit 10
```

**Arguments:**
- `--sequences`: Specific sequence names to process
- `--fps-limit`: Limit extraction to specified FPS (reduces dataset size)
- `--data-dir`: Custom data directory path (optional)

**Output:**
- Creates `data/images/<sequence_name>/` directories
- Saves frames as `00000.jpg`, `00001.jpg`, etc.

## Visualization Scripts

### visualize_bboxes.py

Visualizes bounding boxes overlaid on images.

**Usage:**
```bash
# Random sequence and frame
python scripts/visualization/visualize_bboxes.py

# Specific sequence, random frame
python scripts/visualization/visualize_bboxes.py --sequence ARG_FRA_183303

# Specific sequence and frame
python scripts/visualization/visualize_bboxes.py --sequence ARG_FRA_183303 --frame 100

# Save to file instead of displaying
python scripts/visualization/visualize_bboxes.py --output bbox_vis.jpg
```

**Arguments:**
- `--sequence`: Sequence name (random if not specified)
- `--frame`: Frame index (random if not specified)
- `--output`: Output file path (displays in window if not specified)
- `--data-dir`: Custom data directory path (optional)

**Features:**
- Color-coded bounding boxes for different subjects
- Subject ID labels
- Frame and sequence information overlay
- Handles missing subjects (NaN boxes)

### visualize_3d_pose.py

Visualizes 3D skeletal poses in 3D space or projected onto 2D images.

**Usage:**
```bash
# 3D visualization - random sequence and frame
python scripts/visualization/visualize_3d_pose.py

# 3D visualization - specific sequence and frame
python scripts/visualization/visualize_3d_pose.py --sequence ARG_FRA_183303 --frame 100

# Limit number of subjects displayed
python scripts/visualization/visualize_3d_pose.py --sequence ARG_FRA_183303 --frame 100 --max-subjects 4

# 2D projection on image
python scripts/visualization/visualize_3d_pose.py --sequence ARG_FRA_183303 --frame 100 --project

# Save to file
python scripts/visualization/visualize_3d_pose.py --output pose_3d.png
python scripts/visualization/visualize_3d_pose.py --project --output pose_2d.jpg
```

**Arguments:**
- `--sequence`: Sequence name (random if not specified)
- `--frame`: Frame index (random if not specified)
- `--project`: Project 3D poses to 2D and overlay on image
- `--output`: Output file path (displays in window if not specified)
- `--max-subjects`: Maximum number of subjects to display (default: 12)
- `--data-dir`: Custom data directory path (optional)

**Features:**
- **3D Mode**: Separate subplot for each subject showing clear skeletal structure
  - Each subject in its own 3D subplot with proper scaling
  - Color-coded body parts:
    - Blue: Spine/Head
    - Purple: Left Arm
    - Orange: Right Arm
    - Red: Left Leg
    - Green: Right Leg
  - Anatomically correct connections (25 SMPL joints)
  - Adaptive grid layout (1x1, 2x2, 3x3, 3x4, etc.)
  - Legend showing body part colors
  - Clear viewing angle for standing poses

- **2D Projection Mode**: Overlay 2D poses on original images
  - Projects 3D poses using camera parameters
  - Visualizes skeletal connections
  - Color-coded by subject

**Joint Structure (4D-Humans 25 joints = SMPL 24 + Nose):**
```
0: Pelvis (root)
1: L_Hip, 2: R_Hip
3: Spine1
4: L_Knee, 5: R_Knee
6: Spine2
7: L_Ankle, 8: R_Ankle
9: Spine3
10: L_Foot, 11: R_Foot
12: Neck
13: L_Collar, 14: R_Collar
15: Head
16: L_Shoulder, 17: R_Shoulder
18: L_Elbow, 19: R_Elbow
20: L_Wrist, 21: R_Wrist
22: L_Hand, 23: R_Hand
24: Nose (extra joint)
```

**Skeleton Structure:**
- **Spine:** Pelvis(0) → Spine1(3) → Spine2(6) → Spine3(9) → Neck(12) → Head(15) → Nose(24)
- **Left Leg:** Pelvis(0) → L_Hip(1) → L_Knee(4) → L_Ankle(7) → L_Foot(10)
- **Right Leg:** Pelvis(0) → R_Hip(2) → R_Knee(5) → R_Ankle(8) → R_Foot(11)
- **Left Arm:** Spine3(9) → L_Collar(13) → L_Shoulder(16) → L_Elbow(18) → L_Wrist(20) → L_Hand(22)
- **Right Arm:** Spine3(9) → R_Collar(14) → R_Shoulder(17) → R_Elbow(19) → R_Wrist(21) → R_Hand(23)

**Note:** This is the output format from 4D-Humans preprocessing. For competition submission, only 15 joints are used (see submission format in main README).

## Requirements

All scripts automatically detect the project root and work from any location:
- From project root: `python scripts/<category>/<script>.py`
- From script directory: `python <script>.py`

**Dependencies:**
- numpy
- opencv-python (cv2)
- matplotlib (for 3D visualization)
- tqdm (for progress bars)

Install with:
```bash
source .venv/bin/activate
uv pip install numpy opencv-python matplotlib tqdm
```

## Data Structure Expected

```
data/
├── boxes.npz           # Bounding boxes (13 sequences)
├── skel_2d.npz         # 2D skeletal keypoints
├── skel_3d.npz         # 3D skeletal keypoints
├── cameras/            # Camera parameters per sequence
│   └── <sequence>.npz
├── images/             # Extracted frames (created by extract_frames.py)
│   └── <sequence>/
│       ├── 00000.jpg
│       ├── 00001.jpg
│       └── ...
└── videos/             # Video files
    ├── train_data/
    ├── test_data/
    └── challenge_data/
```

## Notes

- All scripts use **absolute paths** internally for robustness
- **Random selection**: When sequence/frame not specified, random selection ensures good coverage
- **Error handling**: Scripts validate inputs and provide clear error messages
- **Progress tracking**: Long operations (frame extraction) show progress bars
- **Flexible backends**: Visualization works both interactively and for saving to files
