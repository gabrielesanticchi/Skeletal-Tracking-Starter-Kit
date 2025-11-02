"""
Skeletal keypoint data handling (2D and 3D).

Provides object-oriented interface to 2D and 3D skeletal keypoint data.
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import sys

# Import color mapper
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.skeleton_viz import SkeletonColorMapper, SKELETON_CONNECTIONS


# SMPL skeleton structure (24 joints + nose)
SMPL_JOINT_NAMES = [
    'pelvis', 'left_hip', 'right_hip', 'spine1', 'left_knee', 'right_knee',
    'spine2', 'left_ankle', 'right_ankle', 'spine3', 'left_foot', 'right_foot',
    'neck', 'left_collar', 'right_collar', 'head', 'left_shoulder', 'right_shoulder',
    'left_elbow', 'right_elbow', 'left_wrist', 'right_wrist', 'left_hand', 'right_hand',
    'nose'  # Joint 24: extra nose joint added by 4D-Humans
]

# Use connections from utils
SMPL_SKELETON_CONNECTIONS = SKELETON_CONNECTIONS

# Submission format: 15 joints from SMPL
SUBMISSION_JOINT_INDICES = [24, 17, 16, 19, 18, 21, 20, 2, 1, 5, 4, 8, 7, 11, 10]
SUBMISSION_JOINT_NAMES = [
    'nose', 'right_shoulder', 'left_shoulder', 'right_elbow', 'left_elbow',
    'right_wrist', 'left_wrist', 'right_hip', 'left_hip', 'right_knee',
    'left_knee', 'right_ankle', 'left_ankle', 'right_foot', 'left_foot'
]


class Skeleton2DData:
    """
    Handler for 2D skeletal keypoint data.

    Contains 2D keypoints from 4D-Humans (25 joints: SMPL 24 + nose).

    Attributes:
        sequence_name: Name of the sequence
        keypoints: (num_frames, num_subjects, 25, 2) array of 2D keypoints
        num_frames: Number of frames
        num_subjects: Maximum number of subjects
        num_joints: Number of joints (25)
    """

    def __init__(self, sequence_name: str, keypoints: np.ndarray):
        """
        Initialize Skeleton2DData.

        Args:
            sequence_name: Name of the sequence
            keypoints: (num_frames, num_subjects, 25, 2) array
        """
        self.sequence_name = sequence_name
        self.keypoints = keypoints
        self.num_frames = keypoints.shape[0]
        self.num_subjects = keypoints.shape[1]
        self.num_joints = keypoints.shape[2]

    @classmethod
    def load(cls, skel_2d_path: Path, sequence_name: str) -> 'Skeleton2DData':
        """
        Load 2D skeleton data for a specific sequence.

        Args:
            skel_2d_path: Path to skel_2d.npz file
            sequence_name: Name of the sequence

        Returns:
            Skeleton2DData instance
        """
        if not skel_2d_path.exists():
            raise FileNotFoundError(f"Skeleton 2D file not found: {skel_2d_path}")

        npz_data = np.load(skel_2d_path, allow_pickle=True)

        if sequence_name not in npz_data.files:
            raise KeyError(f"Sequence '{sequence_name}' not found in skel_2d file")

        return cls(sequence_name, npz_data[sequence_name])

    @classmethod
    def load_all(cls, skel_2d_path: Path) -> Dict[str, 'Skeleton2DData']:
        """
        Load all sequences from skel_2d file.

        Args:
            skel_2d_path: Path to skel_2d.npz file

        Returns:
            Dictionary mapping sequence names to Skeleton2DData instances
        """
        if not skel_2d_path.exists():
            raise FileNotFoundError(f"Skeleton 2D file not found: {skel_2d_path}")

        npz_data = np.load(skel_2d_path, allow_pickle=True)
        skel_dict = {}

        for sequence_name in npz_data.files:
            skel_dict[sequence_name] = cls(sequence_name, npz_data[sequence_name])

        return skel_dict

    def get_frame_keypoints(self, frame_idx: int, subject_idx: Optional[int] = None) -> np.ndarray:
        """
        Get 2D keypoints for a specific frame.

        Args:
            frame_idx: Frame index
            subject_idx: Optional subject index (if None, returns all subjects)

        Returns:
            (num_subjects, 25, 2) or (25, 2) array of keypoints
        """
        if frame_idx < 0 or frame_idx >= self.num_frames:
            raise ValueError(f"Frame index {frame_idx} out of range [0, {self.num_frames})")

        if subject_idx is None:
            return self.keypoints[frame_idx]
        else:
            if subject_idx < 0 or subject_idx >= self.num_subjects:
                raise ValueError(f"Subject index {subject_idx} out of range [0, {self.num_subjects})")
            return self.keypoints[frame_idx, subject_idx]

    def visualize_frame(
        self,
        image: np.ndarray,
        frame_idx: int,
        show_skeleton: bool = True,
        show_joints: bool = True,
        show_labels: bool = False
    ) -> np.ndarray:
        """
        Visualize 2D skeleton on an image with color-coded joints.

        Args:
            image: Input image (BGR format)
            frame_idx: Frame index
            show_skeleton: Whether to draw skeleton connections
            show_joints: Whether to draw joint points
            show_labels: Whether to show joint labels

        Returns:
            Image with skeleton drawn
        """
        color_mapper = SkeletonColorMapper()
        img_display = image.copy()
        keypoints = self.keypoints[frame_idx]

        # Draw skeleton for each subject
        for subject_idx in range(self.num_subjects):
            kpts = keypoints[subject_idx]

            # Skip if all keypoints are zero (invalid subject)
            if np.all(kpts == 0):
                continue

            # Draw skeleton connections first (under joints)
            if show_skeleton:
                for joint1_idx, joint2_idx in SMPL_SKELETON_CONNECTIONS:
                    if joint1_idx < len(kpts) and joint2_idx < len(kpts):
                        pt1_coords = kpts[joint1_idx]
                        pt2_coords = kpts[joint2_idx]

                        # Skip if either point is invalid (NaN or zero)
                        if np.any(np.isnan(pt1_coords)) or np.any(np.isnan(pt2_coords)):
                            continue
                        if np.all(pt1_coords == 0) or np.all(pt2_coords == 0):
                            continue

                        pt1 = tuple(pt1_coords.astype(int))
                        pt2 = tuple(pt2_coords.astype(int))

                        # Use connection color (average of endpoints)
                        conn_color = color_mapper.get_connection_color((joint1_idx, joint2_idx), format='bgr')
                        conn_color = tuple(int(c) for c in conn_color)  # Ensure integers
                        cv2.line(img_display, pt1, pt2, conn_color, 2)

            # Draw joint points with individual colors
            if show_joints:
                for joint_idx in range(len(kpts)):
                    pt_coords = kpts[joint_idx]

                    # Skip invalid points (NaN or zero)
                    if np.any(np.isnan(pt_coords)) or np.all(pt_coords == 0):
                        continue

                    pt = tuple(pt_coords.astype(int))

                    # Get joint-specific color
                    joint_color = color_mapper.get_joint_color(joint_idx, format='bgr')
                    joint_color = tuple(int(c) for c in joint_color)  # Ensure integers
                    cv2.circle(img_display, pt, 5, joint_color, -1)
                    cv2.circle(img_display, pt, 5, (255, 255, 255), 1)  # White border

                    # Draw label if requested
                    if show_labels:
                        label = color_mapper.get_joint_name(joint_idx)
                        cv2.putText(
                            img_display,
                            label,
                            (pt[0] + 7, pt[1] - 7),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.3,
                            (255, 255, 255),
                            1,
                            cv2.LINE_AA
                        )

        # Add info text and color legend
        info_text = f"2D Skeleton - {self.sequence_name} - Frame {frame_idx}"
        cv2.putText(img_display, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)

        # Add color legend
        legend_y = 60
        cv2.putText(img_display, "Legend: ", (10, legend_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(img_display, "Blue=Spine", (10, legend_y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 150, 50), 1, cv2.LINE_AA)
        cv2.putText(img_display, "Green=Left", (10, legend_y + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (50, 200, 50), 1, cv2.LINE_AA)
        cv2.putText(img_display, "Red=Right", (10, legend_y + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (50, 50, 200), 1, cv2.LINE_AA)

        return img_display

    def __repr__(self) -> str:
        """String representation."""
        return (f"Skeleton2DData(sequence='{self.sequence_name}', "
                f"frames={self.num_frames}, subjects={self.num_subjects}, joints={self.num_joints})")


class Skeleton3DData:
    """
    Handler for 3D skeletal keypoint data.

    Contains 3D keypoints from 4D-Humans (25 joints: SMPL 24 + nose).

    Attributes:
        sequence_name: Name of the sequence
        keypoints: (num_frames, num_subjects, 25, 3) array of 3D keypoints
        num_frames: Number of frames
        num_subjects: Maximum number of subjects
        num_joints: Number of joints (25)
    """

    def __init__(self, sequence_name: str, keypoints: np.ndarray):
        """
        Initialize Skeleton3DData.

        Args:
            sequence_name: Name of the sequence
            keypoints: (num_frames, num_subjects, 25, 3) array
        """
        self.sequence_name = sequence_name
        self.keypoints = keypoints
        self.num_frames = keypoints.shape[0]
        self.num_subjects = keypoints.shape[1]
        self.num_joints = keypoints.shape[2]

    @classmethod
    def load(cls, skel_3d_path: Path, sequence_name: str) -> 'Skeleton3DData':
        """
        Load 3D skeleton data for a specific sequence.

        Args:
            skel_3d_path: Path to skel_3d.npz file
            sequence_name: Name of the sequence

        Returns:
            Skeleton3DData instance
        """
        if not skel_3d_path.exists():
            raise FileNotFoundError(f"Skeleton 3D file not found: {skel_3d_path}")

        npz_data = np.load(skel_3d_path, allow_pickle=True)

        if sequence_name not in npz_data.files:
            raise KeyError(f"Sequence '{sequence_name}' not found in skel_3d file")

        return cls(sequence_name, npz_data[sequence_name])

    @classmethod
    def load_all(cls, skel_3d_path: Path) -> Dict[str, 'Skeleton3DData']:
        """
        Load all sequences from skel_3d file.

        Args:
            skel_3d_path: Path to skel_3d.npz file

        Returns:
            Dictionary mapping sequence names to Skeleton3DData instances
        """
        if not skel_3d_path.exists():
            raise FileNotFoundError(f"Skeleton 3D file not found: {skel_3d_path}")

        npz_data = np.load(skel_3d_path, allow_pickle=True)
        skel_dict = {}

        for sequence_name in npz_data.files:
            skel_dict[sequence_name] = cls(sequence_name, npz_data[sequence_name])

        return skel_dict

    def get_frame_keypoints(self, frame_idx: int, subject_idx: Optional[int] = None) -> np.ndarray:
        """
        Get 3D keypoints for a specific frame.

        Args:
            frame_idx: Frame index
            subject_idx: Optional subject index (if None, returns all subjects)

        Returns:
            (num_subjects, 25, 3) or (25, 3) array of keypoints
        """
        if frame_idx < 0 or frame_idx >= self.num_frames:
            raise ValueError(f"Frame index {frame_idx} out of range [0, {self.num_frames})")

        if subject_idx is None:
            return self.keypoints[frame_idx]
        else:
            if subject_idx < 0 or subject_idx >= self.num_subjects:
                raise ValueError(f"Subject index {subject_idx} out of range [0, {self.num_subjects})")
            return self.keypoints[frame_idx, subject_idx]

    def to_submission_format(self, frame_idx: Optional[int] = None) -> np.ndarray:
        """
        Convert to submission format (15 joints).

        Args:
            frame_idx: Optional frame index (if None, converts all frames)

        Returns:
            (num_frames, num_subjects, 15, 3) or (num_subjects, 15, 3) array
        """
        if frame_idx is not None:
            keypoints = self.keypoints[frame_idx:frame_idx+1]
        else:
            keypoints = self.keypoints

        # Select submission joints
        submission_keypoints = keypoints[:, :, SUBMISSION_JOINT_INDICES, :]

        return submission_keypoints[0] if frame_idx is not None else submission_keypoints

    def visualize_3d(
        self,
        frame_idx: int,
        figsize: Tuple[int, int] = (15, 5),
        elev: int = 20,
        azim: int = -60,
        show_labels: bool = False
    ) -> plt.Figure:
        """
        Visualize 3D skeleton in 3D space with color-coded joints.

        Args:
            frame_idx: Frame index
            figsize: Figure size
            elev: Elevation angle for 3D plot
            azim: Azimuth angle for 3D plot
            show_labels: Whether to show joint labels

        Returns:
            Matplotlib figure
        """
        color_mapper = SkeletonColorMapper()
        keypoints = self.keypoints[frame_idx]

        # Count valid subjects
        valid_subjects = []
        for subject_idx in range(self.num_subjects):
            if not np.all(keypoints[subject_idx] == 0):
                valid_subjects.append(subject_idx)

        if not valid_subjects:
            raise ValueError(f"No valid subjects in frame {frame_idx}")

        # Create subplots
        num_subjects = len(valid_subjects)
        fig = plt.figure(figsize=figsize)

        for plot_idx, subject_idx in enumerate(valid_subjects):
            ax = fig.add_subplot(1, num_subjects, plot_idx + 1, projection='3d')

            kpts = keypoints[subject_idx]

            # Draw skeleton connections first (under joints)
            for joint1_idx, joint2_idx in SMPL_SKELETON_CONNECTIONS:
                if joint1_idx < len(kpts) and joint2_idx < len(kpts):
                    pt1 = kpts[joint1_idx]
                    pt2 = kpts[joint2_idx]

                    # Skip if either point is zero
                    if np.all(pt1 == 0) or np.all(pt2 == 0):
                        continue

                    # Get connection color (normalized for matplotlib)
                    conn_color = color_mapper.get_joint_color_normalized(joint1_idx, format='rgb')

                    ax.plot(
                        [pt1[0], pt2[0]],
                        [pt1[1], pt2[1]],
                        [pt1[2], pt2[2]],
                        color=conn_color,
                        linewidth=2,
                        alpha=0.7
                    )

            # Draw joint points with individual colors
            for joint_idx in range(len(kpts)):
                pt = kpts[joint_idx]

                # Skip if point is zero
                if np.all(pt == 0):
                    continue

                # Get joint-specific color
                joint_color = color_mapper.get_joint_color_normalized(joint_idx, format='rgb')

                ax.scatter(
                    pt[0], pt[1], pt[2],
                    c=[joint_color],
                    s=50,
                    edgecolors='white',
                    linewidths=1
                )

                # Add label if requested
                if show_labels:
                    label = color_mapper.get_joint_name(joint_idx)
                    ax.text(pt[0], pt[1], pt[2], f'  {label}', fontsize=6)

            # Set labels and title
            ax.set_xlabel('X (m)')
            ax.set_ylabel('Y (m)')
            ax.set_zlabel('Z (m)')
            ax.set_title(f'Subject {subject_idx}')

            # Set view angle
            ax.view_init(elev=elev, azim=azim)

            # Set aspect ratio
            ax.set_box_aspect([1, 1, 1])

            # Add grid
            ax.grid(True, alpha=0.3)

        fig.suptitle(f'{self.sequence_name} - Frame {frame_idx}')

        # Add color legend
        legend_labels = color_mapper.get_legend_labels()
        legend_elements = [plt.Line2D([0], [0], marker='o', color='w',
                                      markerfacecolor=color, markersize=8, label=label)
                          for label, color in legend_labels.items()]
        fig.legend(handles=legend_elements, loc='upper right', framealpha=0.9)

        plt.tight_layout()

        return fig

    def __repr__(self) -> str:
        """String representation."""
        return (f"Skeleton3DData(sequence='{self.sequence_name}', "
                f"frames={self.num_frames}, subjects={self.num_subjects}, joints={self.num_joints})")
