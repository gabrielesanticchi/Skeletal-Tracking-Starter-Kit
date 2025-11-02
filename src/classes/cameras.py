"""
Camera parameters handling.

Provides object-oriented interface to camera intrinsics and extrinsics.
"""

import numpy as np
from pathlib import Path
from typing import Dict, Optional, Tuple, Any


class CamerasData:
    """
    Handler for camera calibration data.

    Contains per-frame intrinsics and first-frame extrinsics:
    - K: Intrinsic matrix per frame
    - k: Distortion coefficients (only k1, k2 are valid)
    - R: Rotation matrix (first frame only)
    - t: Translation vector (first frame only)
    - Rt: Combined [R|t] matrix (first frame only)

    Attributes:
        sequence_name: Name of the sequence
        K: (num_frames, 3, 3) intrinsic matrices
        k: (num_frames, 5) distortion coefficients
        R: (1, 3, 3) rotation matrix for first frame
        t: (1, 3) translation vector for first frame
        Rt: (1, 3, 4) combined [R|t] for first frame
        num_frames: Number of frames
    """

    def __init__(self, sequence_name: str, npz_data: Any):
        """
        Initialize CamerasData from NPZ file.

        Args:
            sequence_name: Name of the sequence
            npz_data: Loaded NPZ file containing camera parameters
        """
        self.sequence_name = sequence_name

        # Load camera parameters
        self.K = npz_data['K']    # (num_frames, 3, 3)
        self.k = npz_data['k']    # (num_frames, 5) - only k[0:2] are valid
        self.R = npz_data['R']    # (1, 3, 3) - first frame only
        self.t = npz_data['t']    # (1, 3) - first frame only
        self.Rt = npz_data['Rt']  # (1, 3, 4) - first frame only

        self.num_frames = self.K.shape[0]

    @classmethod
    def load(cls, cameras_dir: Path, sequence_name: str) -> 'CamerasData':
        """
        Load camera data from directory.

        Args:
            cameras_dir: Directory containing camera files
            sequence_name: Name of the sequence

        Returns:
            CamerasData instance
        """
        camera_path = cameras_dir / f"{sequence_name}.npz"
        if not camera_path.exists():
            raise FileNotFoundError(f"Camera file not found: {camera_path}")

        npz_data = np.load(camera_path, allow_pickle=True)
        return cls(sequence_name, npz_data)

    @classmethod
    def load_all(cls, cameras_dir: Path) -> Dict[str, 'CamerasData']:
        """
        Load all camera sequences from directory.

        Args:
            cameras_dir: Directory containing camera files

        Returns:
            Dictionary mapping sequence names to CamerasData instances
        """
        cameras_dict = {}
        for camera_path in sorted(cameras_dir.glob("*.npz")):
            sequence_name = camera_path.stem
            cameras_dict[sequence_name] = cls.load(cameras_dir, sequence_name)
        return cameras_dict

    def get_intrinsics(self, frame_idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get intrinsic parameters for a specific frame.

        Args:
            frame_idx: Frame index

        Returns:
            Tuple of (K, k) where:
                K: (3, 3) intrinsic matrix
                k: (5,) distortion coefficients (only first 2 are valid)
        """
        if frame_idx < 0 or frame_idx >= self.num_frames:
            raise ValueError(f"Frame index {frame_idx} out of range [0, {self.num_frames})")

        return self.K[frame_idx], self.k[frame_idx]

    def get_extrinsics_first_frame(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get extrinsic parameters for the first frame.

        Note: Only first frame extrinsics are provided. Camera pose tracking
        is required for subsequent frames.

        Returns:
            Tuple of (R, t) where:
                R: (3, 3) rotation matrix
                t: (3,) translation vector
        """
        return self.R[0], self.t[0]

    def get_projection_matrix_first_frame(self) -> np.ndarray:
        """
        Get full projection matrix P = K[R|t] for first frame.

        Returns:
            (3, 4) projection matrix
        """
        K, _ = self.get_intrinsics(0)
        R, t = self.get_extrinsics_first_frame()

        # Construct [R|t]
        Rt = np.hstack([R, t.reshape(3, 1)])

        # P = K[R|t]
        P = K @ Rt
        return P

    def project_3d_to_2d(
        self,
        points_3d: np.ndarray,
        frame_idx: int,
        R: Optional[np.ndarray] = None,
        t: Optional[np.ndarray] = None,
        apply_distortion: bool = False
    ) -> np.ndarray:
        """
        Project 3D points to 2D image coordinates.

        Args:
            points_3d: (N, 3) array of 3D points in world coordinates
            frame_idx: Frame index (for intrinsics)
            R: (3, 3) rotation matrix (if None, uses first frame R)
            t: (3,) translation vector (if None, uses first frame t)
            apply_distortion: Whether to apply distortion correction

        Returns:
            (N, 2) array of 2D pixel coordinates
        """
        K, k_dist = self.get_intrinsics(frame_idx)

        # Use first frame extrinsics if not provided
        if R is None or t is None:
            R, t = self.get_extrinsics_first_frame()

        # Transform to camera coordinates
        points_cam = (R @ points_3d.T).T + t

        # Project to normalized image coordinates
        points_normalized = points_cam[:, :2] / points_cam[:, 2:3]

        # Apply distortion if requested
        if apply_distortion:
            r2 = np.sum(points_normalized**2, axis=1)
            radial = 1 + k_dist[0] * r2 + k_dist[1] * r2**2
            points_normalized = points_normalized * radial[:, np.newaxis]

        # Apply intrinsics
        points_2d = (K[:2, :2] @ points_normalized.T).T + K[:2, 2]

        return points_2d

    def __repr__(self) -> str:
        """String representation."""
        return (f"CamerasData(sequence='{self.sequence_name}', "
                f"frames={self.num_frames})")

    def __str__(self) -> str:
        """Detailed string representation."""
        K_first = self.K[0]
        R_first = self.R[0]
        t_first = self.t[0]

        return (f"CamerasData for sequence '{self.sequence_name}':\n"
                f"  Frames: {self.num_frames}\n"
                f"  Intrinsics K (first frame):\n{K_first}\n"
                f"  Rotation R (first frame):\n{R_first}\n"
                f"  Translation t (first frame): {t_first}")
