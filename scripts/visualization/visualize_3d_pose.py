"""
Visualize 3D skeletal poses from the FIFA Skeletal Tracking Challenge dataset.

This script allows you to visualize 3D poses with bounding boxes overlaid on images.
If no arguments are provided, a random sequence and frame are selected.

Usage:
    # Random sequence and frame
    python scripts/visualization/visualize_3d_pose.py

    # Specific sequence, random frame
    python scripts/visualization/visualize_3d_pose.py --sequence ARG_FRA_183303

    # Specific sequence and frame
    python scripts/visualization/visualize_3d_pose.py --sequence ARG_FRA_183303 --frame 100

    # Project 3D to 2D and overlay on image
    python scripts/visualization/visualize_3d_pose.py --sequence ARG_FRA_183303 --frame 100 --project

    # Save output
    python scripts/visualization/visualize_3d_pose.py --output pose_3d.png
"""

import numpy as np
import cv2
import matplotlib
# Use Agg backend for non-interactive plotting (when saving to file)
# Will switch to interactive backend when needed
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path
import sys
import argparse
import random
from typing import Optional, Tuple, Dict


def get_project_root() -> Path:
    """Get the project root directory."""
    current = Path(__file__).resolve()
    return current.parent.parent.parent


def load_data(data_dir: Path) -> Tuple[Dict, Dict, Dict, list]:
    """
    Load 3D pose, 2D pose, bounding box, and camera data.

    Args:
        data_dir: Path to data directory

    Returns:
        Tuple of (skel_3d, skel_2d, boxes, cameras, sequences)
    """
    skel_3d_path = data_dir / "skel_3d.npz"
    skel_2d_path = data_dir / "skel_2d.npz"
    boxes_path = data_dir / "boxes.npz"

    if not skel_3d_path.exists():
        raise FileNotFoundError(f"skel_3d.npz not found at {skel_3d_path}")
    if not skel_2d_path.exists():
        raise FileNotFoundError(f"skel_2d.npz not found at {skel_2d_path}")
    if not boxes_path.exists():
        raise FileNotFoundError(f"boxes.npz not found at {boxes_path}")

    skel_3d = np.load(skel_3d_path, allow_pickle=True)
    skel_2d = np.load(skel_2d_path, allow_pickle=True)
    boxes = np.load(boxes_path, allow_pickle=True)

    skel_3d_data = {key: skel_3d[key] for key in skel_3d.files}
    skel_2d_data = {key: skel_2d[key] for key in skel_2d.files}
    boxes_data = {key: boxes[key] for key in boxes.files}

    sequences = list(boxes_data.keys())

    # Load camera data for the sequences
    cameras_data = {}
    cameras_dir = data_dir / "cameras"
    for seq in sequences:
        camera_path = cameras_dir / f"{seq}.npz"
        if camera_path.exists():
            cameras_data[seq] = np.load(camera_path, allow_pickle=True)

    return skel_3d_data, skel_2d_data, boxes_data, cameras_data, sequences


def select_random_sequence_and_frame(
    boxes_data: dict,
    sequences: list,
    sequence: Optional[str] = None,
    frame_idx: Optional[int] = None
) -> Tuple[str, int]:
    """
    Select a random or specified sequence and frame.

    Args:
        boxes_data: Dictionary of bounding box data
        sequences: List of available sequences
        sequence: Optional specific sequence name
        frame_idx: Optional specific frame index

    Returns:
        Tuple of (sequence_name, frame_index)
    """
    if sequence is None:
        sequence = random.choice(sequences)
        print(f"📌 Randomly selected sequence: {sequence}")
    else:
        if sequence not in sequences:
            raise ValueError(f"Sequence '{sequence}' not found. Available: {sequences}")
        print(f"📌 Using specified sequence: {sequence}")

    num_frames = boxes_data[sequence].shape[0]
    if frame_idx is None:
        frame_idx = random.randint(0, num_frames - 1)
        print(f"📌 Randomly selected frame: {frame_idx} (out of {num_frames})")
    else:
        if frame_idx < 0 or frame_idx >= num_frames:
            raise ValueError(f"Frame index {frame_idx} out of range [0, {num_frames-1}]")
        print(f"📌 Using specified frame: {frame_idx} (out of {num_frames})")

    return sequence, frame_idx


def get_skeleton_connections():
    """
    Define skeleton connections for visualization.
    Returns list of tuples (joint1_idx, joint2_idx) representing bones.

    Joint order (15 joints from SMPL):
    0: nose
    1-2: right_shoulder, left_shoulder
    3-4: right_elbow, left_elbow
    5-6: right_wrist, left_wrist
    7-8: right_hip, left_hip
    9-10: right_knee, left_knee
    11-12: right_ankle, left_ankle
    13-14: right_foot, left_foot
    """
    connections = [
        # Spine
        (0, 1), (0, 2),  # Nose to shoulders
        (1, 7), (2, 8),  # Shoulders to hips
        (7, 8),          # Hip connection

        # Right arm
        (1, 3), (3, 5),  # Shoulder -> elbow -> wrist

        # Left arm
        (2, 4), (4, 6),  # Shoulder -> elbow -> wrist

        # Right leg
        (7, 9), (9, 11), (11, 13),  # Hip -> knee -> ankle -> foot

        # Left leg
        (8, 10), (10, 12), (12, 14),  # Hip -> knee -> ankle -> foot
    ]

    return connections


def visualize_3d_pose_matplotlib(
    poses_3d: np.ndarray,
    sequence: str,
    frame_idx: int,
    output_path: Optional[str] = None
):
    """
    Visualize 3D poses using matplotlib.

    Args:
        poses_3d: 3D poses array of shape (num_subjects, 15, 3)
        sequence: Sequence name
        frame_idx: Frame index
        output_path: Optional output file path
    """
    # Set backend based on whether we're saving or displaying
    if output_path:
        matplotlib.use('Agg')  # Non-interactive backend for saving
    else:
        # Try to use interactive backend, fall back to Agg if not available
        try:
            matplotlib.use('TkAgg')
        except:
            matplotlib.use('Agg')
            print("Warning: Interactive backend not available. Saving to 'pose_3d_temp.png' instead.")
            output_path = 'pose_3d_temp.png'

    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    connections = get_skeleton_connections()

    # Define colors for different subjects
    colors = plt.cm.tab10(np.linspace(0, 1, 10))

    num_subjects = 0

    # Plot each subject's skeleton
    for subject_idx, pose in enumerate(poses_3d):
        # Skip if pose contains NaN (subject not present)
        if np.any(np.isnan(pose)):
            continue

        num_subjects += 1
        color = colors[subject_idx % len(colors)]

        # Extract x, y, z coordinates
        xs = pose[:, 0]
        ys = pose[:, 1]
        zs = pose[:, 2]

        # Plot joints
        ax.scatter(xs, ys, zs, c=[color], marker='o', s=50, label=f'Subject {subject_idx}')

        # Plot skeleton connections
        for joint1, joint2 in connections:
            if joint1 < len(pose) and joint2 < len(pose):
                ax.plot(
                    [xs[joint1], xs[joint2]],
                    [ys[joint1], ys[joint2]],
                    [zs[joint1], zs[joint2]],
                    c=color,
                    linewidth=2
                )

    # Set labels and title
    ax.set_xlabel('X (m)', fontsize=12)
    ax.set_ylabel('Y (m)', fontsize=12)
    ax.set_zlabel('Z (m)', fontsize=12)
    ax.set_title(f'3D Poses - {sequence} - Frame {frame_idx}\nSubjects: {num_subjects}',
                 fontsize=14, fontweight='bold')

    # Set equal aspect ratio
    max_range = np.array([
        poses_3d[~np.isnan(poses_3d[:, :, 0])].max() - poses_3d[~np.isnan(poses_3d[:, :, 0])].min(),
        poses_3d[~np.isnan(poses_3d[:, :, 1])].max() - poses_3d[~np.isnan(poses_3d[:, :, 1])].min(),
        poses_3d[~np.isnan(poses_3d[:, :, 2])].max() - poses_3d[~np.isnan(poses_3d[:, :, 2])].min()
    ]).max() / 2.0

    mid_x = (poses_3d[~np.isnan(poses_3d[:, :, 0])].max() + poses_3d[~np.isnan(poses_3d[:, :, 0])].min()) * 0.5
    mid_y = (poses_3d[~np.isnan(poses_3d[:, :, 1])].max() + poses_3d[~np.isnan(poses_3d[:, :, 1])].min()) * 0.5
    mid_z = (poses_3d[~np.isnan(poses_3d[:, :, 2])].max() + poses_3d[~np.isnan(poses_3d[:, :, 2])].min()) * 0.5

    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    # Add grid
    ax.grid(True, alpha=0.3)

    # Add legend if multiple subjects
    if num_subjects > 1:
        ax.legend(loc='upper right')

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"✓ 3D visualization saved to: {output_path}")
    else:
        plt.show()


def project_and_visualize_2d(
    poses_2d: np.ndarray,
    image: np.ndarray,
    sequence: str,
    frame_idx: int,
    output_path: Optional[str] = None
):
    """
    Project 3D poses to 2D and overlay on image.

    Args:
        poses_2d: 2D poses array of shape (num_subjects, 15, 2)
        image: Input image (BGR format)
        sequence: Sequence name
        frame_idx: Frame index
        output_path: Optional output file path
    """
    img_display = image.copy()
    connections = get_skeleton_connections()

    # Define colors for different subjects (BGR format)
    colors = [
        (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
        (255, 0, 255), (0, 255, 255), (128, 0, 128), (255, 165, 0),
        (0, 128, 128), (128, 128, 0)
    ]

    num_subjects = 0

    # Draw each subject's skeleton
    for subject_idx, pose in enumerate(poses_2d):
        # Skip if pose contains NaN
        if np.any(np.isnan(pose)):
            continue

        num_subjects += 1
        color = colors[subject_idx % len(colors)]

        # Draw skeleton connections
        for joint1, joint2 in connections:
            if joint1 < len(pose) and joint2 < len(pose):
                pt1 = tuple(pose[joint1].astype(int))
                pt2 = tuple(pose[joint2].astype(int))
                cv2.line(img_display, pt1, pt2, color, 2)

        # Draw joints
        for joint in pose:
            pt = tuple(joint.astype(int))
            cv2.circle(img_display, pt, 4, color, -1)
            cv2.circle(img_display, pt, 5, (255, 255, 255), 1)

    # Add info text
    info_text = f"Sequence: {sequence} | Frame: {frame_idx} | Subjects: {num_subjects}"
    cv2.putText(img_display, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (0, 255, 0), 2, cv2.LINE_AA)

    if output_path:
        cv2.imwrite(output_path, img_display)
        print(f"✓ 2D projection saved to: {output_path}")
    else:
        cv2.imshow(f"2D Pose - {sequence} - Frame {frame_idx}", img_display)
        print("Press any key to close the window...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def main():
    """Main visualization function."""
    parser = argparse.ArgumentParser(
        description="Visualize 3D poses from FIFA Skeletal Tracking Challenge"
    )
    parser.add_argument("--sequence", type=str, help="Sequence name")
    parser.add_argument("--frame", type=int, help="Frame index")
    parser.add_argument("--project", action="store_true",
                       help="Project 3D to 2D and overlay on image")
    parser.add_argument("--output", type=str, help="Output file path")
    parser.add_argument("--data-dir", type=Path, default=None,
                       help="Base data directory")

    args = parser.parse_args()

    # Setup paths
    if args.data_dir:
        data_dir = args.data_dir
    else:
        project_root = get_project_root()
        data_dir = project_root / "data"

    print("\n" + "="*80)
    print("3D POSE VISUALIZATION")
    print("="*80 + "\n")

    try:
        # Load data
        print("Loading data...")
        skel_3d, skel_2d, boxes, cameras, sequences = load_data(data_dir)
        print(f"✓ Loaded {len(sequences)} sequences\n")

        # Select sequence and frame
        sequence, frame_idx = select_random_sequence_and_frame(
            boxes, sequences, args.sequence, args.frame
        )

        # Get 3D poses for this frame
        poses_3d = skel_3d[sequence][frame_idx]
        print(f"\n✓ Loaded 3D poses: {poses_3d.shape}")

        if args.project:
            # Project to 2D and overlay on image
            print("\nProjecting 3D poses to 2D and overlaying on image...")

            # Load image
            image_path = data_dir / "images" / sequence / f"{frame_idx:05d}.jpg"
            if not image_path.exists():
                raise FileNotFoundError(f"Image not found at {image_path}")

            image = cv2.imread(str(image_path))
            poses_2d = skel_2d[sequence][frame_idx]

            project_and_visualize_2d(poses_2d, image, sequence, frame_idx, args.output)
        else:
            # Visualize in 3D
            print("\nVisualizing 3D poses...")
            visualize_3d_pose_matplotlib(poses_3d, sequence, frame_idx, args.output)

        print("\n" + "="*80)
        print("VISUALIZATION COMPLETE")
        print("="*80 + "\n")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    return 0


if __name__ == "__main__":
    sys.exit(main())
