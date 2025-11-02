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


# SMPL skeleton structure (24 joints + nose)
SMPL_JOINT_NAMES = [
    'pelvis', 'left_hip', 'right_hip', 'spine1', 'left_knee', 'right_knee',
    'spine2', 'left_ankle', 'right_ankle', 'spine3', 'left_foot', 'right_foot',
    'neck', 'left_collar', 'right_collar', 'head', 'left_shoulder', 'right_shoulder',
    'left_elbow', 'right_elbow', 'left_wrist', 'right_wrist', 'left_hand', 'right_hand',
    'nose'  # Joint 24: extra nose joint added by 4D-Humans
]

# Skeleton connections for visualization
SMPL_SKELETON_CONNECTIONS = [
    # Spine
    (0, 1), (0, 2), (0, 3), (3, 6), (6, 9), (9, 12), (12, 15),
    # Left leg
    (1, 4), (4, 7), (7, 10),
    # Right leg
    (2, 5), (5, 8), (8, 11),
    # Left arm
    (9, 13), (13, 16), (16, 18), (18, 20), (20, 22),
    # Right arm
    (9, 14), (14, 17), (17, 19), (19, 21), (21, 23),
    # Head
    (12, 24),  # neck to nose
]

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
        color_palette: Optional[List[Tuple[int, int, int]]] = None
    ) -> np.ndarray:
        """
        Visualize 2D skeleton on an image.

        Args:
            image: Input image (BGR format)
            frame_idx: Frame index
            show_skeleton: Whether to draw skeleton connections
            show_joints: Whether to draw joint points
            color_palette: Optional list of BGR colors for different subjects

        Returns:
            Image with skeleton drawn
        """
        # Default color palette
        if color_palette is None:
            color_palette = [
                (255, 0, 0),    # Blue
                (0, 255, 0),    # Green
                (0, 0, 255),    # Red
                (255, 255, 0),  # Cyan
                (255, 0, 255),  # Magenta
                (0, 255, 255),  # Yellow
            ]

        img_display = image.copy()
        keypoints = self.keypoints[frame_idx]

        # Draw skeleton for each subject
        for subject_idx in range(self.num_subjects):
            kpts = keypoints[subject_idx]

            # Skip if all keypoints are zero (invalid subject)
            if np.all(kpts == 0):
                continue

            color = color_palette[subject_idx % len(color_palette)]

            # Draw skeleton connections
            if show_skeleton:
                for joint1_idx, joint2_idx in SMPL_SKELETON_CONNECTIONS:
                    if joint1_idx < len(kpts) and joint2_idx < len(kpts):
                        pt1 = tuple(kpts[joint1_idx].astype(int))
                        pt2 = tuple(kpts[joint2_idx].astype(int))

                        # Skip if either point is invalid
                        if pt1 == (0, 0) or pt2 == (0, 0):
                            continue

                        cv2.line(img_display, pt1, pt2, color, 2)

            # Draw joint points
            if show_joints:
                for joint_idx in range(len(kpts)):
                    pt = tuple(kpts[joint_idx].astype(int))

                    # Skip invalid points
                    if pt == (0, 0):
                        continue

                    cv2.circle(img_display, pt, 3, color, -1)

        # Add info text
        info_text = f"2D Skeleton - {self.sequence_name} - Frame {frame_idx}"
        cv2.putText(
            img_display,
            info_text,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
            cv2.LINE_AA
        )

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
        azim: int = -60
    ) -> plt.Figure:
        """
        Visualize 3D skeleton in 3D space.

        Args:
            frame_idx: Frame index
            figsize: Figure size
            elev: Elevation angle for 3D plot
            azim: Azimuth angle for 3D plot

        Returns:
            Matplotlib figure
        """
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

            # Draw skeleton connections
            for joint1_idx, joint2_idx in SMPL_SKELETON_CONNECTIONS:
                if joint1_idx < len(kpts) and joint2_idx < len(kpts):
                    pt1 = kpts[joint1_idx]
                    pt2 = kpts[joint2_idx]

                    # Skip if either point is zero
                    if np.all(pt1 == 0) or np.all(pt2 == 0):
                        continue

                    ax.plot(
                        [pt1[0], pt2[0]],
                        [pt1[1], pt2[1]],
                        [pt1[2], pt2[2]],
                        'b-', linewidth=2
                    )

            # Draw joint points
            valid_kpts = kpts[~np.all(kpts == 0, axis=1)]
            if len(valid_kpts) > 0:
                ax.scatter(
                    valid_kpts[:, 0],
                    valid_kpts[:, 1],
                    valid_kpts[:, 2],
                    c='r', s=30
                )

            # Set labels and title
            ax.set_xlabel('X (m)')
            ax.set_ylabel('Y (m)')
            ax.set_zlabel('Z (m)')
            ax.set_title(f'Subject {subject_idx}')

            # Set view angle
            ax.view_init(elev=elev, azim=azim)

            # Set aspect ratio
            ax.set_box_aspect([1, 1, 1])

        fig.suptitle(f'{self.sequence_name} - Frame {frame_idx}')
        plt.tight_layout()

        return fig

    def __repr__(self) -> str:
        """String representation."""
        return (f"Skeleton3DData(sequence='{self.sequence_name}', "
                f"frames={self.num_frames}, subjects={self.num_subjects}, joints={self.num_joints})")
