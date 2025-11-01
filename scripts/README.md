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
- `--data-dir`: Custom data directory path (optional)

**Features:**
- **3D Mode**: Interactive 3D plot showing skeletal structure
  - Color-coded skeletons for different subjects
  - Proper anatomical connections (15 SMPL joints)
  - Equal aspect ratio and grid

- **2D Projection Mode**: Overlay 2D poses on original images
  - Projects 3D poses using camera parameters
  - Visualizes skeletal connections
  - Color-coded by subject

**Joint Structure (15 SMPL joints):**
```
0: nose
1-2: right_shoulder, left_shoulder
3-4: right_elbow, left_elbow
5-6: right_wrist, left_wrist
7-8: right_hip, left_hip
9-10: right_knee, left_knee
11-12: right_ankle, left_ankle
13-14: right_foot, left_foot
```

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
