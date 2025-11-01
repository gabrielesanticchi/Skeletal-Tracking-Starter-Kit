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
    Define skeleton connections for SMPL 25-joint visualization.
    Returns list of tuples (joint1_idx, joint2_idx) representing bones.

    SMPL 25 joint order (standard SMPL-X body model):
    0: pelvis, 1: left_hip, 2: right_hip, 3: spine1, 4: left_knee, 5: right_knee,
    6: spine2, 7: left_ankle, 8: right_ankle, 9: spine3, 10: left_foot, 11: right_foot,
    12: neck, 13: left_collar, 14: right_collar, 15: head, 16: left_shoulder, 17: right_shoulder,
    18: left_elbow, 19: right_elbow, 20: left_wrist, 21: right_wrist, 22: left_hand, 23: right_hand
    """
    connections = [
        # Spine chain (central axis)
        (0, 3),   # pelvis -> spine1
        (3, 6),   # spine1 -> spine2
        (6, 9),   # spine2 -> spine3
        (9, 12),  # spine3 -> neck
        (12, 15), # neck -> head

        # Left leg
        (0, 1),   # pelvis -> left_hip
        (1, 4),   # left_hip -> left_knee
        (4, 7),   # left_knee -> left_ankle
        (7, 10),  # left_ankle -> left_foot

        # Right leg
        (0, 2),   # pelvis -> right_hip
        (2, 5),   # right_hip -> right_knee
        (5, 8),   # right_knee -> right_ankle
        (8, 11),  # right_ankle -> right_foot

        # Left arm
        (9, 13),  # spine3 -> left_collar
        (13, 16), # left_collar -> left_shoulder
        (16, 18), # left_shoulder -> left_elbow
        (18, 20), # left_elbow -> left_wrist
        (20, 22), # left_wrist -> left_hand

        # Right arm
        (9, 14),  # spine3 -> right_collar
        (14, 17), # right_collar -> right_shoulder
        (17, 19), # right_shoulder -> right_elbow
        (19, 21), # right_elbow -> right_wrist
        (21, 23), # right_wrist -> right_hand
    ]

    return connections


def visualize_3d_pose_matplotlib(
    poses_3d: np.ndarray,
    sequence: str,
    frame_idx: int,
    output_path: Optional[str] = None,
    max_subjects: int = 12
):
    """
    Visualize 3D poses using matplotlib with separate subplot for each subject.

    Args:
        poses_3d: 3D poses array of shape (num_subjects, 25, 3)
        sequence: Sequence name
        frame_idx: Frame index
        output_path: Optional output file path
        max_subjects: Maximum number of subjects to display (default: 12)
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

    connections = get_skeleton_connections()

    # Filter out NaN poses and get valid subjects
    valid_subjects = []
    valid_poses = []
    for subject_idx, pose in enumerate(poses_3d):
        if not np.any(np.isnan(pose)):
            valid_subjects.append(subject_idx)
            valid_poses.append(pose)
            if len(valid_subjects) >= max_subjects:
                break

    num_subjects = len(valid_subjects)

    if num_subjects == 0:
        print("❌ No valid subjects found in this frame!")
        return

    print(f"Visualizing {num_subjects} subjects (IDs: {valid_subjects})")

    # Calculate grid layout for subplots
    if num_subjects == 1:
        rows, cols = 1, 1
    elif num_subjects == 2:
        rows, cols = 1, 2
    elif num_subjects <= 4:
        rows, cols = 2, 2
    elif num_subjects <= 6:
        rows, cols = 2, 3
    elif num_subjects <= 9:
        rows, cols = 3, 3
    elif num_subjects <= 12:
        rows, cols = 3, 4
    else:
        rows, cols = 4, 4

    # Create figure with subplots
    fig = plt.figure(figsize=(5 * cols, 4.5 * rows))
    fig.suptitle(f'3D Poses - {sequence} - Frame {frame_idx} ({num_subjects} subjects)',
                 fontsize=16, fontweight='bold', y=0.995)

    # Define colors for skeleton parts
    color_spine = '#2E86AB'     # Blue
    color_left_arm = '#A23B72'  # Purple
    color_right_arm = '#F18F01' # Orange
    color_left_leg = '#C73E1D'  # Red
    color_right_leg = '#6A994E' # Green

    # Plot each subject in a separate subplot
    for plot_idx, (subject_idx, pose) in enumerate(zip(valid_subjects, valid_poses)):
        ax = fig.add_subplot(rows, cols, plot_idx + 1, projection='3d')

        # Extract x, y, z coordinates
        xs = pose[:, 0]
        ys = pose[:, 1]
        zs = pose[:, 2]

        # Plot joints (smaller markers)
        ax.scatter(xs, ys, zs, c='black', marker='o', s=20, alpha=0.6)

        # Plot skeleton connections with different colors for body parts
        for joint1, joint2 in connections:
            if joint1 < len(pose) and joint2 < len(pose):
                # Determine color based on body part
                if joint1 == 0 or joint2 == 0 or (joint1 in [3, 6, 9, 12, 15] and joint2 in [3, 6, 9, 12, 15]):
                    color = color_spine
                    linewidth = 3
                elif joint1 in [1, 4, 7, 10] or joint2 in [1, 4, 7, 10]:
                    color = color_left_leg
                    linewidth = 2.5
                elif joint1 in [2, 5, 8, 11] or joint2 in [2, 5, 8, 11]:
                    color = color_right_leg
                    linewidth = 2.5
                elif joint1 in [13, 16, 18, 20, 22] or joint2 in [13, 16, 18, 20, 22]:
                    color = color_left_arm
                    linewidth = 2.5
                elif joint1 in [14, 17, 19, 21, 23] or joint2 in [14, 17, 19, 21, 23]:
                    color = color_right_arm
                    linewidth = 2.5
                else:
                    color = 'gray'
                    linewidth = 2

                ax.plot(
                    [xs[joint1], xs[joint2]],
                    [ys[joint1], ys[joint2]],
                    [zs[joint1], zs[joint2]],
                    c=color,
                    linewidth=linewidth,
                    alpha=0.8
                )

        # Set labels
        ax.set_xlabel('X (m)', fontsize=9)
        ax.set_ylabel('Y (m)', fontsize=9)
        ax.set_zlabel('Z (m)', fontsize=9)
        ax.set_title(f'Subject {subject_idx}', fontsize=11, fontweight='bold')

        # Set equal aspect ratio for this subject
        pose_range = np.ptp(pose, axis=0).max() / 2.0
        pose_center = pose.mean(axis=0)

        ax.set_xlim(pose_center[0] - pose_range, pose_center[0] + pose_range)
        ax.set_ylim(pose_center[1] - pose_range, pose_center[1] + pose_range)
        ax.set_zlim(pose_center[2] - pose_range, pose_center[2] + pose_range)

        # Add grid and set viewing angle
        ax.grid(True, alpha=0.3)
        ax.view_init(elev=10, azim=45)

        # Smaller tick labels
        ax.tick_params(labelsize=8)

    # Add legend in the last subplot or in empty space
    if num_subjects < rows * cols:
        # Use empty subplot for legend
        legend_ax = fig.add_subplot(rows, cols, num_subjects + 1)
        legend_ax.axis('off')

        legend_elements = [
            plt.Line2D([0], [0], color=color_spine, linewidth=3, label='Spine/Head'),
            plt.Line2D([0], [0], color=color_left_arm, linewidth=2.5, label='Left Arm'),
            plt.Line2D([0], [0], color=color_right_arm, linewidth=2.5, label='Right Arm'),
            plt.Line2D([0], [0], color=color_left_leg, linewidth=2.5, label='Left Leg'),
            plt.Line2D([0], [0], color=color_right_leg, linewidth=2.5, label='Right Leg'),
        ]
        legend_ax.legend(handles=legend_elements, loc='center', fontsize=10, frameon=True)

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
    parser.add_argument("--max-subjects", type=int, default=12,
                       help="Maximum number of subjects to visualize (default: 12)")

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
            visualize_3d_pose_matplotlib(poses_3d, sequence, frame_idx, args.output, args.max_subjects)

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
