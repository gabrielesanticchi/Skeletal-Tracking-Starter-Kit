"""
SMPL Poses data handling.

Provides object-oriented interface to SMPL pose parameters from the WorldPose dataset.
"""

import numpy as np
from pathlib import Path
from typing import Dict, Optional, Tuple, Any
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


class PosesData:
    """
    Handler for SMPL pose data.

    SMPL poses contain raw parameters from WorldPose dataset:
    - global_orient: Global orientation (axis-angle)
    - body_pose: Body pose parameters (23 joints × 3)
    - transl: Translation in world coordinates
    - betas: Shape parameters

    Attributes:
        sequence_name: Name of the sequence
        global_orient: (num_subjects, num_frames, 3) global orientation
        body_pose: (num_subjects, num_frames, 69) body pose parameters
        transl: (num_subjects, num_frames, 3) translation vectors
        betas: (num_subjects, num_frames, 10) shape parameters
        num_subjects: Number of subjects in sequence
        num_frames: Number of frames in sequence
    """

    def __init__(self, sequence_name: str, npz_data: Any):
        """
        Initialize PosesData from NPZ file.

        Args:
            sequence_name: Name of the sequence
            npz_data: Loaded NPZ file containing SMPL parameters
        """
        self.sequence_name = sequence_name

        # Load SMPL parameters
        # Original shape: (num_subjects, num_frames, dim)
        self.global_orient = npz_data['global_orient']  # (num_subjects, num_frames, 3)
        self.body_pose = npz_data['body_pose']          # (num_subjects, num_frames, 69)
        self.transl = npz_data['transl']                # (num_subjects, num_frames, 3)
        self.betas = npz_data['betas']                  # (num_subjects, num_frames, 10)

        # Store dimensions
        self.num_subjects = self.body_pose.shape[0]
        self.num_frames = self.body_pose.shape[1]

    @classmethod
    def load(cls, poses_dir: Path, sequence_name: str) -> 'PosesData':
        """
        Load poses data from directory.

        Args:
            poses_dir: Directory containing pose files
            sequence_name: Name of the sequence

        Returns:
            PosesData instance
        """
        pose_path = poses_dir / f"{sequence_name}.npz"
        if not pose_path.exists():
            raise FileNotFoundError(f"Pose file not found: {pose_path}")

        npz_data = np.load(pose_path, allow_pickle=True)
        return cls(sequence_name, npz_data)

    @classmethod
    def load_all(cls, poses_dir: Path) -> Dict[str, 'PosesData']:
        """
        Load all pose sequences from directory.

        Args:
            poses_dir: Directory containing pose files

        Returns:
            Dictionary mapping sequence names to PosesData instances
        """
        poses_dict = {}
        for pose_path in sorted(poses_dir.glob("*.npz")):
            sequence_name = pose_path.stem
            poses_dict[sequence_name] = cls.load(poses_dir, sequence_name)
        return poses_dict

    def get_frame_data(self, frame_idx: int, subject_idx: Optional[int] = None) -> Dict[str, np.ndarray]:
        """
        Get SMPL parameters for a specific frame.

        Args:
            frame_idx: Frame index
            subject_idx: Optional subject index (if None, returns all subjects)

        Returns:
            Dictionary with SMPL parameters
        """
        if frame_idx < 0 or frame_idx >= self.num_frames:
            raise ValueError(f"Frame index {frame_idx} out of range [0, {self.num_frames})")

        if subject_idx is None:
            return {
                'global_orient': self.global_orient[:, frame_idx, :],
                'body_pose': self.body_pose[:, frame_idx, :],
                'transl': self.transl[:, frame_idx, :],
                'betas': self.betas[:, frame_idx, :]
            }
        else:
            if subject_idx < 0 or subject_idx >= self.num_subjects:
                raise ValueError(f"Subject index {subject_idx} out of range [0, {self.num_subjects})")

            return {
                'global_orient': self.global_orient[subject_idx, frame_idx, :],
                'body_pose': self.body_pose[subject_idx, frame_idx, :],
                'transl': self.transl[subject_idx, frame_idx, :],
                'betas': self.betas[subject_idx, frame_idx, :]
            }

    def get_subject_trajectory(self, subject_idx: int) -> np.ndarray:
        """
        Get trajectory (translation over time) for a specific subject.

        Args:
            subject_idx: Subject index

        Returns:
            (num_frames, 3) array of positions
        """
        if subject_idx < 0 or subject_idx >= self.num_subjects:
            raise ValueError(f"Subject index {subject_idx} out of range [0, {self.num_subjects})")

        return self.transl[subject_idx, :, :]

    def __repr__(self) -> str:
        """String representation."""
        return (f"PosesData(sequence='{self.sequence_name}', "
                f"frames={self.num_frames}, subjects={self.num_subjects})")

    def __str__(self) -> str:
        """Detailed string representation."""
        return (f"PosesData for sequence '{self.sequence_name}':\n"
                f"  Frames: {self.num_frames}\n"
                f"  Subjects: {self.num_subjects}\n"
                f"  Shape: global_orient{self.global_orient.shape}, "
                f"body_pose{self.body_pose.shape}, "
                f"transl{self.transl.shape}, "
                f"betas{self.betas.shape}")
