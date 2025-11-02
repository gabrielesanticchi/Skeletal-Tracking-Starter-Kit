"""
Generate bounding boxes from SMPL mesh projections for all sequences.

This script projects SMPL meshes onto the image plane using camera parameters
and derives bounding boxes from the projected vertices. This allows generating
bounding boxes for all sequences (including those without manual annotations).

Usage:
    # Generate boxes for all sequences with camera parameters
    python scripts/preprocessing/generate_boxes_from_smpl.py

    # Generate boxes for specific sequences
    python scripts/preprocessing/generate_boxes_from_smpl.py --sequences ARG_CRO_220001 ARG_FRA_182345

    # Specify output file
    python scripts/preprocessing/generate_boxes_from_smpl.py --output data/boxes_all.npz
"""

import numpy as np
import cv2
from pathlib import Path
import argparse
from tqdm import tqdm
import sys
from typing import Dict, Optional, Tuple

try:
    import torch
    from smplx import SMPL
    HAS_SMPLX = True
except ImportError:
    HAS_SMPLX = False
    print("Warning: smplx not found. Install with: uv pip install smplx")


def get_project_root() -> Path:
    """Get the project root directory."""
    current = Path(__file__).resolve()
    return current.parent.parent.parent


def project_vertices_to_2d(
    vertices: np.ndarray,
    K: np.ndarray,
    R: np.ndarray,
    t: np.ndarray,
    dist_coeffs: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Project 3D vertices onto 2D image plane.

    Args:
        vertices: (N, 3) array of 3D vertex coordinates
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
        vertices.astype(np.float32),
        rvec,
        t.astype(np.float32),
        K.astype(np.float32),
        dist_coeffs.astype(np.float32)
    )

    return pts_2d.squeeze(1)


def vertices_to_bbox(vertices_2d: np.ndarray, margin: float = 0.0) -> np.ndarray:
    """
    Compute bounding box from 2D vertices.

    Args:
        vertices_2d: (N, 2) array of 2D coordinates
        margin: Additional margin to add to bbox (as fraction of bbox size)

    Returns:
        (4,) array [x_min, y_min, x_max, y_max] or [nan, nan, nan, nan] if invalid
    """
    # Filter out invalid vertices
    valid_mask = np.isfinite(vertices_2d).all(axis=1)

    if not valid_mask.any():
        return np.array([np.nan, np.nan, np.nan, np.nan])

    valid_vertices = vertices_2d[valid_mask]

    x_min, y_min = valid_vertices.min(axis=0)
    x_max, y_max = valid_vertices.max(axis=0)

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


def load_smpl_model(gender: str = 'neutral', model_path: Optional[Path] = None) -> SMPL:
    """
    Load SMPL model.

    Args:
        gender: 'neutral', 'male', or 'female'
        model_path: Path to SMPL model files

    Returns:
        SMPL model instance
    """
    if not HAS_SMPLX:
        raise ImportError("smplx is required. Install with: uv pip install smplx")

    if model_path is None:
        # Default path - user should download SMPL models
        model_path = get_project_root() / "models" / "smpl"

    return SMPL(
        model_path=str(model_path),
        gender=gender,
        batch_size=1,
        create_transl=False
    )


def generate_boxes_for_sequence(
    sequence_name: str,
    data_dir: Path,
    smpl_model: SMPL,
    margin: float = 0.05,
    device: str = 'cpu'
) -> Optional[np.ndarray]:
    """
    Generate bounding boxes for a sequence using SMPL mesh projection.

    Args:
        sequence_name: Name of the sequence
        data_dir: Path to data directory
        smpl_model: SMPL model instance
        margin: Margin to add to bounding boxes (fraction of bbox size)
        device: Device to use for SMPL model ('cpu' or 'cuda')

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
        betas = pose_data['betas']                  # (num_frames, num_subjects, 10)

        K = camera_data['K']          # (num_frames, 3, 3)
        k_dist = camera_data['k']     # (num_frames, 5)
        R = camera_data['R']          # (1, 3, 3) - only first frame
        t = camera_data['t']          # (1, 3) - only first frame

        num_frames, num_subjects = global_orient.shape[:2]

        # Initialize output array
        boxes = np.zeros((num_frames, num_subjects, 4))
        boxes.fill(np.nan)

        # Move model to device
        smpl_model = smpl_model.to(device)

        # Process each frame
        for frame_idx in tqdm(range(num_frames), desc=f"  Processing {sequence_name}", leave=False):
            # Get camera parameters for this frame
            K_frame = K[frame_idx] if len(K.shape) > 2 else K[0]
            k_frame = k_dist[frame_idx] if len(k_dist.shape) > 1 else k_dist[0]

            # Use first frame R, t (or estimate for subsequent frames if available)
            R_frame = R[0] if len(R.shape) > 2 else R
            t_frame = t[0] if len(t.shape) > 1 else t

            # Process each subject
            for subj_idx in range(num_subjects):
                # Check if subject is present (valid pose data)
                if np.isnan(global_orient[frame_idx, subj_idx]).any():
                    continue

                # Prepare SMPL parameters
                global_orient_subj = torch.tensor(
                    global_orient[frame_idx, subj_idx:subj_idx+1],
                    dtype=torch.float32,
                    device=device
                )
                body_pose_subj = torch.tensor(
                    body_pose[frame_idx, subj_idx:subj_idx+1],
                    dtype=torch.float32,
                    device=device
                )
                betas_subj = torch.tensor(
                    betas[frame_idx, subj_idx:subj_idx+1],
                    dtype=torch.float32,
                    device=device
                )
                transl_subj = torch.tensor(
                    transl[frame_idx, subj_idx:subj_idx+1],
                    dtype=torch.float32,
                    device=device
                )

                # Generate SMPL mesh
                with torch.no_grad():
                    output = smpl_model(
                        global_orient=global_orient_subj,
                        body_pose=body_pose_subj,
                        betas=betas_subj,
                        transl=transl_subj
                    )
                    vertices = output.vertices[0].cpu().numpy()  # (6890, 3)

                # Project vertices to 2D
                vertices_2d = project_vertices_to_2d(
                    vertices,
                    K_frame,
                    R_frame,
                    t_frame,
                    k_frame
                )

                # Compute bounding box
                bbox = vertices_to_bbox(vertices_2d, margin=margin)
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
            print(f"   Updating {key}")
        else:
            print(f"   Adding {key}")
        all_boxes[key] = value

    return all_boxes


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Generate bounding boxes from SMPL mesh projections"
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
        default=0.05,
        help="Margin to add to bboxes as fraction of bbox size (default: 0.05)"
    )
    parser.add_argument(
        "--smpl-model-path",
        type=Path,
        default=None,
        help="Path to SMPL model files"
    )
    parser.add_argument(
        "--device",
        type=str,
        default='cpu',
        choices=['cpu', 'cuda'],
        help="Device to use for SMPL model (default: cpu)"
    )
    parser.add_argument(
        "--merge",
        action='store_true',
        help="Merge with existing boxes.npz file"
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
    print("GENERATE BOUNDING BOXES FROM SMPL MESH PROJECTIONS")
    print("="*80 + "\n")

    # Check for smplx
    if not HAS_SMPLX:
        print("❌ Error: smplx package is required")
        print("   Install with: uv pip install smplx")
        print("\n   You also need to download SMPL models from:")
        print("   https://smpl.is.tue.mpg.de/")
        sys.exit(1)

    # Get sequences to process
    if args.sequences:
        sequences = args.sequences
    else:
        # Get all sequences with camera data
        camera_dir = data_dir / "cameras"
        sequences = [p.stem for p in sorted(camera_dir.glob("*.npz"))]

    print(f"📋 Processing {len(sequences)} sequences")
    print(f"📂 Data directory: {data_dir}")
    print(f"💾 Output file: {output_path}")
    print(f"🎨 Margin: {args.margin}")
    print(f"🖥️  Device: {args.device}")

    # Load SMPL model
    print(f"\n🔄 Loading SMPL model...")
    try:
        smpl_model = load_smpl_model(model_path=args.smpl_model_path)
        print(f"✓ SMPL model loaded")
    except Exception as e:
        print(f"❌ Error loading SMPL model: {e}")
        print("\n   Download SMPL models from: https://smpl.is.tue.mpg.de/")
        print(f"   Place models in: {args.smpl_model_path or get_project_root() / 'models' / 'smpl'}")
        sys.exit(1)

    # Process sequences
    print(f"\n🔄 Processing sequences...\n")
    generated_boxes = {}

    for seq in sequences:
        print(f"📹 {seq}")
        boxes = generate_boxes_for_sequence(
            seq,
            data_dir,
            smpl_model,
            margin=args.margin,
            device=args.device
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
