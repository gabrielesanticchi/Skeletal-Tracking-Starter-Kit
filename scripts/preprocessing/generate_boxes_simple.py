"""
Generate bounding boxes from SMPL pose data using forward kinematics.

This script generates bounding boxes without requiring the full SMPL mesh model.
Instead, it uses SMPL forward kinematics to compute joint locations and derives
bounding boxes from the projected joint positions.

Usage:
    # Generate boxes for all sequences with camera parameters
    python scripts/preprocessing/generate_boxes_simple.py

    # Generate boxes for specific sequences
    python scripts/preprocessing/generate_boxes_simple.py --sequences ARG_CRO_220001 ARG_FRA_182345

    # Specify output file
    python scripts/preprocessing/generate_boxes_simple.py --output data/boxes_all.npz

    # Merge with existing boxes
    python scripts/preprocessing/generate_boxes_simple.py --merge
"""

import numpy as np
import cv2
from pathlib import Path
import argparse
from tqdm import tqdm
import sys
from typing import Dict, Optional, Tuple

# SMPL joint tree structure (parent indices)
SMPL_PARENTS = np.array([
    -1,  # 0: Pelvis (root)
    0, 0, 0,  # 1-3: L_Hip, R_Hip, Spine1
    1, 2, 3,  # 4-6: L_Knee, R_Knee, Spine2
    4, 5, 6,  # 7-9: L_Ankle, R_Ankle, Spine3
    7, 8, 9,  # 10-12: L_Foot, R_Foot, Neck
    9, 9,     # 13-14: L_Collar, R_Collar
    12,       # 15: Head
    13, 14,   # 16-17: L_Shoulder, R_Shoulder
    16, 17,   # 18-19: L_Elbow, R_Elbow
    18, 19,   # 20-21: L_Wrist, R_Wrist
    20, 21    # 22-23: L_Hand, R_Hand
], dtype=np.int32)


def rodrigues_rotation_matrix(axis_angle: np.ndarray) -> np.ndarray:
    """
    Convert axis-angle representation to rotation matrix.

    Args:
        axis_angle: (3,) axis-angle vector

    Returns:
        (3, 3) rotation matrix
    """
    angle = np.linalg.norm(axis_angle)

    if angle < 1e-8:
        return np.eye(3)

    axis = axis_angle / angle
    K = np.array([
        [0, -axis[2], axis[1]],
        [axis[2], 0, -axis[0]],
        [-axis[1], axis[0], 0]
    ])

    R = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)
    return R


def forward_kinematics_smpl(
    global_orient: np.ndarray,
    body_pose: np.ndarray,
    transl: np.ndarray
) -> np.ndarray:
    """
    Compute SMPL joint positions using forward kinematics.

    Args:
        global_orient: (3,) global orientation in axis-angle
        body_pose: (69,) body pose parameters (23 joints × 3)
        transl: (3,) global translation

    Returns:
        (24, 3) array of joint positions in world coordinates
    """
    # Reshape body pose to (23, 3)
    body_pose = body_pose.reshape(23, 3)

    # Combine global orientation with body pose
    all_poses = np.vstack([global_orient.reshape(1, 3), body_pose])  # (24, 3)

    # Convert all axis-angle to rotation matrices
    rotations = np.array([rodrigues_rotation_matrix(aa) for aa in all_poses])  # (24, 3, 3)

    # SMPL template joint positions (approximate, in meters)
    # These are rough estimates - in reality, they depend on shape parameters (betas)
    # But for bounding box purposes, this approximation is sufficient
    template_joints = np.array([
        [0.0, 0.0, 0.0],      # 0: Pelvis
        [0.1, -0.05, 0.0],    # 1: L_Hip
        [-0.1, -0.05, 0.0],   # 2: R_Hip
        [0.0, 0.1, 0.0],      # 3: Spine1
        [0.1, -0.45, 0.0],    # 4: L_Knee
        [-0.1, -0.45, 0.0],   # 5: R_Knee
        [0.0, 0.2, 0.0],      # 6: Spine2
        [0.1, -0.85, 0.0],    # 7: L_Ankle
        [-0.1, -0.85, 0.0],   # 8: R_Ankle
        [0.0, 0.35, 0.0],     # 9: Spine3
        [0.1, -0.95, 0.05],   # 10: L_Foot
        [-0.1, -0.95, 0.05],  # 11: R_Foot
        [0.0, 0.5, 0.0],      # 12: Neck
        [0.05, 0.4, 0.0],     # 13: L_Collar
        [-0.05, 0.4, 0.0],    # 14: R_Collar
        [0.0, 0.65, 0.0],     # 15: Head
        [0.15, 0.4, 0.0],     # 16: L_Shoulder
        [-0.15, 0.4, 0.0],    # 17: R_Shoulder
        [0.35, 0.3, 0.0],     # 18: L_Elbow
        [-0.35, 0.3, 0.0],    # 19: R_Elbow
        [0.55, 0.2, 0.0],     # 20: L_Wrist
        [-0.55, 0.2, 0.0],    # 21: R_Wrist
        [0.65, 0.15, 0.0],    # 22: L_Hand
        [-0.65, 0.15, 0.0],   # 23: R_Hand
    ], dtype=np.float32)

    # Compute global joint transformations
    global_transforms = np.zeros((24, 4, 4))
    joint_positions = np.zeros((24, 3))

    for i in range(24):
        # Local transformation
        local_transform = np.eye(4)
        local_transform[:3, :3] = rotations[i]
        local_transform[:3, 3] = template_joints[i]

        # Global transformation
        if SMPL_PARENTS[i] == -1:
            # Root joint
            global_transforms[i] = local_transform
        else:
            parent_idx = SMPL_PARENTS[i]
            global_transforms[i] = global_transforms[parent_idx] @ local_transform

        # Extract position
        joint_positions[i] = global_transforms[i][:3, 3]

    # Apply global translation
    joint_positions += transl

    return joint_positions


def project_points_to_2d(
    points_3d: np.ndarray,
    K: np.ndarray,
    R: np.ndarray,
    t: np.ndarray,
    dist_coeffs: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Project 3D points onto 2D image plane.

    Args:
        points_3d: (N, 3) array of 3D coordinates
        K: (3, 3) camera intrinsic matrix
        R: (3, 3) rotation matrix
        t: (3,) translation vector
        dist_coeffs: (5,) distortion coefficients (optional)

    Returns:
        (N, 2) array of 2D projected coordinates
    """
    if dist_coeffs is None:
        dist_coeffs = np.zeros(5, dtype=np.float32)

    # Use OpenCV's projectPoints for accurate projection with distortion
    rvec = cv2.Rodrigues(R)[0]
    pts_2d, _ = cv2.projectPoints(
        points_3d.astype(np.float32),
        rvec,
        t.astype(np.float32),
        K.astype(np.float32),
        dist_coeffs.astype(np.float32)
    )

    return pts_2d.squeeze(1)


def joints_to_bbox(joints_2d: np.ndarray, margin: float = 0.1) -> np.ndarray:
    """
    Compute bounding box from 2D joint positions.

    Args:
        joints_2d: (N, 2) array of 2D joint coordinates
        margin: Additional margin to add to bbox (as fraction of bbox size)

    Returns:
        (4,) array [x_min, y_min, x_max, y_max] or [nan, nan, nan, nan] if invalid
    """
    # Filter out invalid joints
    valid_mask = np.isfinite(joints_2d).all(axis=1)

    if not valid_mask.any():
        return np.array([np.nan, np.nan, np.nan, np.nan])

    valid_joints = joints_2d[valid_mask]

    x_min, y_min = valid_joints.min(axis=0)
    x_max, y_max = valid_joints.max(axis=0)

    # Add margin
    if margin > 0:
        width = x_max - x_min
        height = y_max - y_min
        x_margin = width * margin
        y_margin = height * margin

        x_min -= x_margin
        y_min -= y_margin
        x_max += x_margin
        y_max += y_margin

    return np.array([x_min, y_min, x_max, y_max])


def get_project_root() -> Path:
    """Get the project root directory."""
    current = Path(__file__).resolve()
    return current.parent.parent.parent


def generate_boxes_for_sequence(
    sequence_name: str,
    data_dir: Path,
    margin: float = 0.1
) -> Optional[np.ndarray]:
    """
    Generate bounding boxes for a sequence using SMPL joint projection.

    Args:
        sequence_name: Name of the sequence
        data_dir: Path to data directory
        margin: Margin to add to bounding boxes (fraction of bbox size)

    Returns:
        (num_frames, num_subjects, 4) array of bounding boxes or None if failed
    """
    # Load SMPL pose data
    pose_path = data_dir / "poses" / f"{sequence_name}.npz"
    if not pose_path.exists():
        print(f"  ⚠️  No pose data found for {sequence_name}")
        return None

    # Load camera data
    camera_path = data_dir / "cameras" / f"{sequence_name}.npz"
    if not camera_path.exists():
        print(f"  ⚠️  No camera data found for {sequence_name}")
        return None

    try:
        pose_data = np.load(pose_path, allow_pickle=True)
        camera_data = np.load(camera_path, allow_pickle=True)

        global_orient = pose_data['global_orient']  # (num_frames, num_subjects, 3)
        body_pose = pose_data['body_pose']          # (num_frames, num_subjects, 69)
        transl = pose_data['transl']                # (num_frames, num_subjects, 3)

        K = camera_data['K']          # (num_frames, 3, 3)
        k_dist = camera_data['k']     # (num_frames, 5)
        R = camera_data['R']          # (1, 3, 3) - only first frame
        t = camera_data['t']          # (1, 3) - only first frame

        num_frames, num_subjects = global_orient.shape[:2]

        # Initialize output array
        boxes = np.zeros((num_frames, num_subjects, 4))
        boxes.fill(np.nan)

        # Process each frame
        for frame_idx in tqdm(range(num_frames), desc=f"  Processing {sequence_name}", leave=False):
            # Get camera parameters for this frame
            K_frame = K[frame_idx] if len(K.shape) > 2 else K[0]
            k_frame = k_dist[frame_idx] if len(k_dist.shape) > 1 else k_dist[0]

            # Use first frame R, t (note: this is a simplification)
            R_frame = R[0] if len(R.shape) > 2 else R
            t_frame = t[0] if len(t.shape) > 1 else t

            # Process each subject
            for subj_idx in range(num_subjects):
                # Check if subject is present (valid pose data)
                if np.isnan(global_orient[frame_idx, subj_idx]).any():
                    continue

                # Compute joint positions using forward kinematics
                joints_3d = forward_kinematics_smpl(
                    global_orient[frame_idx, subj_idx],
                    body_pose[frame_idx, subj_idx],
                    transl[frame_idx, subj_idx]
                )

                # Project joints to 2D
                joints_2d = project_points_to_2d(
                    joints_3d,
                    K_frame,
                    R_frame,
                    t_frame,
                    k_frame
                )

                # Compute bounding box
                bbox = joints_to_bbox(joints_2d, margin=margin)
                boxes[frame_idx, subj_idx] = bbox

        return boxes

    except Exception as e:
        print(f"  ❌ Error processing {sequence_name}: {e}")
        import traceback
        traceback.print_exc()
        return None


def merge_with_existing_boxes(
    generated_boxes: Dict[str, np.ndarray],
    existing_boxes_path: Path
) -> Dict[str, np.ndarray]:
    """
    Merge generated boxes with existing boxes.npz file.

    Args:
        generated_boxes: Dictionary of generated boxes
        existing_boxes_path: Path to existing boxes.npz file

    Returns:
        Merged dictionary of all boxes
    """
    all_boxes = {}

    # Load existing boxes if available
    if existing_boxes_path.exists():
        print(f"\n📦 Loading existing boxes from {existing_boxes_path}")
        existing = np.load(existing_boxes_path, allow_pickle=True)
        for key in existing.files:
            all_boxes[key] = existing[key]
        print(f"   Loaded {len(all_boxes)} sequences from existing file")

    # Add/update with generated boxes
    print(f"\n📦 Merging {len(generated_boxes)} generated sequences")
    for key, value in generated_boxes.items():
        if key in all_boxes:
            print(f"   Updating {key} (using existing ground truth)")
            # Keep existing ground truth boxes, don't override
            continue
        else:
            print(f"   Adding {key}")
            all_boxes[key] = value

    return all_boxes


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Generate bounding boxes from SMPL pose data (no mesh model required)"
    )
    parser.add_argument(
        "--sequences",
        nargs='+',
        help="Specific sequences to process (default: all with camera data)"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output file path (default: data/boxes_all.npz)"
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Base data directory"
    )
    parser.add_argument(
        "--margin",
        type=float,
        default=0.1,
        help="Margin to add to bboxes as fraction of bbox size (default: 0.1)"
    )
    parser.add_argument(
        "--merge",
        action='store_true',
        help="Merge with existing boxes.npz file (preserves ground truth)"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of sequences to process (for testing)"
    )

    args = parser.parse_args()

    # Setup paths
    if args.data_dir:
        data_dir = args.data_dir
    else:
        project_root = get_project_root()
        data_dir = project_root / "data"

    if args.output:
        output_path = args.output
    else:
        output_path = data_dir / "boxes_all.npz"

    print("\n" + "="*80)
    print("GENERATE BOUNDING BOXES FROM SMPL POSE DATA")
    print("="*80 + "\n")

    # Get sequences to process
    if args.sequences:
        sequences = args.sequences
    else:
        # Get all sequences with camera data
        camera_dir = data_dir / "cameras"
        sequences = [p.stem for p in sorted(camera_dir.glob("*.npz"))]

        # Filter to only sequences that need boxes (don't have ground truth)
        if args.merge:
            existing_boxes_path = data_dir / "boxes.npz"
            if existing_boxes_path.exists():
                existing = np.load(existing_boxes_path, allow_pickle=True)
                existing_sequences = set(existing.files)
                sequences_to_generate = [s for s in sequences if s not in existing_sequences]
                print(f"📋 Found {len(existing_sequences)} sequences with ground truth boxes")
                print(f"📋 Will generate boxes for {len(sequences_to_generate)} remaining sequences")
                sequences = sequences_to_generate

    if args.limit:
        sequences = sequences[:args.limit]

    print(f"📋 Processing {len(sequences)} sequences")
    print(f"📂 Data directory: {data_dir}")
    print(f"💾 Output file: {output_path}")
    print(f"🎨 Margin: {args.margin}")

    # Process sequences
    print(f"\n🔄 Processing sequences...\n")
    generated_boxes = {}

    for seq in sequences:
        print(f"📹 {seq}")
        boxes = generate_boxes_for_sequence(
            seq,
            data_dir,
            margin=args.margin
        )

        if boxes is not None:
            generated_boxes[seq] = boxes
            print(f"   ✓ Generated {boxes.shape[0]} frames, {boxes.shape[1]} subjects")
        else:
            print(f"   ⚠️  Skipped")

    # Merge with existing if requested
    if args.merge:
        existing_path = data_dir / "boxes.npz"
        all_boxes = merge_with_existing_boxes(generated_boxes, existing_path)
    else:
        all_boxes = generated_boxes

    # Save results
    print(f"\n💾 Saving {len(all_boxes)} sequences to {output_path}")
    np.savez_compressed(output_path, **all_boxes)
    print(f"✓ Saved successfully")

    # Print summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Total sequences: {len(all_boxes)}")
    print(f"Output file: {output_path}")
    print(f"File size: {output_path.stat().st_size / 1024 / 1024:.2f} MB")
    print("\nSequences included:")
    for seq in sorted(all_boxes.keys()):
        frames, subjects = all_boxes[seq].shape[:2]
        print(f"  - {seq}: {frames} frames, {subjects} subjects")
    print("\n" + "="*80 + "\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
