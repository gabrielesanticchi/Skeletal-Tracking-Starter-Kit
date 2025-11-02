"""
Metadata classes for organizing sequence data.

Provides unified interfaces (ImageMetadata, VideoMetadata) that aggregate
all related data (poses, cameras, bboxes, skeletons) for a sequence.
"""

import numpy as np
import cv2
from pathlib import Path
from typing import Optional, Dict, Tuple
import matplotlib.pyplot as plt

from .poses import PosesData
from .cameras import CamerasData
from .bboxes import BBoxesData
from .skeleton import Skeleton2DData, Skeleton3DData


class ImageMetadata:
    """
    Metadata container for a single frame/image.

    Aggregates all data related to a specific frame in a sequence:
    - Poses (SMPL parameters)
    - Camera parameters
    - Bounding boxes
    - 2D and 3D skeletal keypoints
    - Image data

    This class provides a unified interface to access and visualize
    all frame-related information.
    """

    def __init__(
        self,
        sequence_name: str,
        frame_idx: int,
        poses: Optional[PosesData] = None,
        cameras: Optional[CamerasData] = None,
        bboxes: Optional[BBoxesData] = None,
        skel_2d: Optional[Skeleton2DData] = None,
        skel_3d: Optional[Skeleton3DData] = None,
        image: Optional[np.ndarray] = None,
        image_path: Optional[Path] = None
    ):
        """
        Initialize ImageMetadata.

        Args:
            sequence_name: Name of the sequence
            frame_idx: Frame index
            poses: Optional PosesData instance
            cameras: Optional CamerasData instance
            bboxes: Optional BBoxesData instance
            skel_2d: Optional Skeleton2DData instance
            skel_3d: Optional Skeleton3DData instance
            image: Optional pre-loaded image
            image_path: Optional path to image file
        """
        self.sequence_name = sequence_name
        self.frame_idx = frame_idx

        # Data components
        self.poses = poses
        self.cameras = cameras
        self.bboxes = bboxes
        self.skel_2d = skel_2d
        self.skel_3d = skel_3d

        # Image data
        self._image = image
        self.image_path = image_path

    @property
    def image(self) -> Optional[np.ndarray]:
        """
        Get image, loading from path if necessary.

        Returns:
            Image array (BGR format) or None if not available
        """
        if self._image is not None:
            return self._image

        if self.image_path is not None and self.image_path.exists():
            self._image = cv2.imread(str(self.image_path))
            return self._image

        return None

    @image.setter
    def image(self, value: np.ndarray):
        """Set image data."""
        self._image = value

    def load_image(self, images_dir: Path) -> np.ndarray:
        """
        Load image from standard directory structure.

        Args:
            images_dir: Base images directory

        Returns:
            Loaded image (BGR format)
        """
        image_path = images_dir / self.sequence_name / f"{self.frame_idx:05d}.jpg"
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        self.image_path = image_path
        self._image = cv2.imread(str(image_path))
        return self._image

    def get_poses_data(self, subject_idx: Optional[int] = None) -> Dict[str, np.ndarray]:
        """
        Get SMPL pose parameters for this frame.

        Args:
            subject_idx: Optional subject index

        Returns:
            Dictionary with SMPL parameters
        """
        if self.poses is None:
            raise ValueError("Poses data not available")

        return self.poses.get_frame_data(self.frame_idx, subject_idx)

    def get_bboxes(self, valid_only: bool = True) -> np.ndarray:
        """
        Get bounding boxes for this frame.

        Args:
            valid_only: If True, filter out NaN boxes

        Returns:
            (N, 4) array of bounding boxes
        """
        if self.bboxes is None:
            raise ValueError("Bounding boxes data not available")

        return self.bboxes.get_frame_boxes(self.frame_idx, valid_only)

    def get_skeleton_2d(self, subject_idx: Optional[int] = None) -> np.ndarray:
        """
        Get 2D skeletal keypoints for this frame.

        Args:
            subject_idx: Optional subject index

        Returns:
            2D keypoints array
        """
        if self.skel_2d is None:
            raise ValueError("2D skeleton data not available")

        return self.skel_2d.get_frame_keypoints(self.frame_idx, subject_idx)

    def get_skeleton_3d(self, subject_idx: Optional[int] = None) -> np.ndarray:
        """
        Get 3D skeletal keypoints for this frame.

        Args:
            subject_idx: Optional subject index

        Returns:
            3D keypoints array
        """
        if self.skel_3d is None:
            raise ValueError("3D skeleton data not available")

        return self.skel_3d.get_frame_keypoints(self.frame_idx, subject_idx)

    def get_camera_intrinsics(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get camera intrinsics for this frame.

        Returns:
            Tuple of (K, k) intrinsic parameters
        """
        if self.cameras is None:
            raise ValueError("Camera data not available")

        return self.cameras.get_intrinsics(self.frame_idx)

    def visualize_bboxes(self, image: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Visualize bounding boxes on image.

        Args:
            image: Optional image (uses self.image if None)

        Returns:
            Image with bounding boxes drawn
        """
        if self.bboxes is None:
            raise ValueError("Bounding boxes data not available")

        img = image if image is not None else self.image
        if img is None:
            raise ValueError("Image not available")

        return self.bboxes.visualize_frame(img, self.frame_idx)

    def visualize_skeleton_2d(self, image: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Visualize 2D skeleton on image.

        Args:
            image: Optional image (uses self.image if None)

        Returns:
            Image with 2D skeleton drawn
        """
        if self.skel_2d is None:
            raise ValueError("2D skeleton data not available")

        img = image if image is not None else self.image
        if img is None:
            raise ValueError("Image not available")

        return self.skel_2d.visualize_frame(img, self.frame_idx)

    def visualize_skeleton_3d(self, **kwargs) -> plt.Figure:
        """
        Visualize 3D skeleton in 3D space.

        Args:
            **kwargs: Additional arguments passed to Skeleton3DData.visualize_3d()

        Returns:
            Matplotlib figure
        """
        if self.skel_3d is None:
            raise ValueError("3D skeleton data not available")

        return self.skel_3d.visualize_3d(self.frame_idx, **kwargs)

    def __repr__(self) -> str:
        """String representation."""
        components = []
        if self.poses: components.append("poses")
        if self.cameras: components.append("cameras")
        if self.bboxes: components.append("bboxes")
        if self.skel_2d: components.append("skel_2d")
        if self.skel_3d: components.append("skel_3d")
        if self.image is not None: components.append("image")

        return (f"ImageMetadata(sequence='{self.sequence_name}', "
                f"frame={self.frame_idx}, "
                f"components=[{', '.join(components)}])")


class VideoMetadata:
    """
    Metadata container for a video sequence.

    Aggregates all data related to an entire sequence:
    - Poses (SMPL parameters)
    - Camera parameters
    - Bounding boxes
    - 2D and 3D skeletal keypoints

    Provides convenient methods to extract ImageMetadata for individual frames.
    """

    def __init__(
        self,
        sequence_name: str,
        poses: Optional[PosesData] = None,
        cameras: Optional[CamerasData] = None,
        bboxes: Optional[BBoxesData] = None,
        skel_2d: Optional[Skeleton2DData] = None,
        skel_3d: Optional[Skeleton3DData] = None
    ):
        """
        Initialize VideoMetadata.

        Args:
            sequence_name: Name of the sequence
            poses: Optional PosesData instance
            cameras: Optional CamerasData instance
            bboxes: Optional BBoxesData instance
            skel_2d: Optional Skeleton2DData instance
            skel_3d: Optional Skeleton3DData instance
        """
        self.sequence_name = sequence_name

        # Data components
        self.poses = poses
        self.cameras = cameras
        self.bboxes = bboxes
        self.skel_2d = skel_2d
        self.skel_3d = skel_3d

        # Determine number of frames from available data
        self.num_frames = self._infer_num_frames()

    def _infer_num_frames(self) -> int:
        """Infer number of frames from available data."""
        for data in [self.poses, self.cameras, self.bboxes, self.skel_2d, self.skel_3d]:
            if data is not None:
                return data.num_frames
        return 0

    @classmethod
    def load(
        cls,
        data_dir: Path,
        sequence_name: str,
        load_poses: bool = True,
        load_cameras: bool = True,
        load_bboxes: bool = True,
        load_skel_2d: bool = True,
        load_skel_3d: bool = True
    ) -> 'VideoMetadata':
        """
        Load video metadata from data directory.

        Args:
            data_dir: Base data directory
            sequence_name: Name of the sequence
            load_poses: Whether to load poses
            load_cameras: Whether to load cameras
            load_bboxes: Whether to load bounding boxes
            load_skel_2d: Whether to load 2D skeleton
            load_skel_3d: Whether to load 3D skeleton

        Returns:
            VideoMetadata instance
        """
        poses = None
        cameras = None
        bboxes = None
        skel_2d = None
        skel_3d = None

        # Load poses
        if load_poses:
            poses_dir = data_dir / "poses"
            if poses_dir.exists():
                try:
                    poses = PosesData.load(poses_dir, sequence_name)
                except FileNotFoundError:
                    pass

        # Load cameras
        if load_cameras:
            cameras_dir = data_dir / "cameras"
            if cameras_dir.exists():
                try:
                    cameras = CamerasData.load(cameras_dir, sequence_name)
                except FileNotFoundError:
                    pass

        # Load bounding boxes
        if load_bboxes:
            boxes_path = data_dir / "boxes.npz"
            if boxes_path.exists():
                try:
                    bboxes = BBoxesData.load(boxes_path, sequence_name)
                except (FileNotFoundError, KeyError):
                    pass

        # Load 2D skeleton
        if load_skel_2d:
            skel_2d_path = data_dir / "skel_2d.npz"
            if skel_2d_path.exists():
                try:
                    skel_2d = Skeleton2DData.load(skel_2d_path, sequence_name)
                except (FileNotFoundError, KeyError):
                    pass

        # Load 3D skeleton
        if load_skel_3d:
            skel_3d_path = data_dir / "skel_3d.npz"
            if skel_3d_path.exists():
                try:
                    skel_3d = Skeleton3DData.load(skel_3d_path, sequence_name)
                except (FileNotFoundError, KeyError):
                    pass

        return cls(sequence_name, poses, cameras, bboxes, skel_2d, skel_3d)

    @classmethod
    def load_all(
        cls,
        data_dir: Path,
        load_poses: bool = True,
        load_cameras: bool = True,
        load_bboxes: bool = True,
        load_skel_2d: bool = True,
        load_skel_3d: bool = True
    ) -> Dict[str, 'VideoMetadata']:
        """
        Load all video sequences from data directory.

        Args:
            data_dir: Base data directory
            load_poses: Whether to load poses
            load_cameras: Whether to load cameras
            load_bboxes: Whether to load bounding boxes
            load_skel_2d: Whether to load 2D skeleton
            load_skel_3d: Whether to load 3D skeleton

        Returns:
            Dictionary mapping sequence names to VideoMetadata instances
        """
        # Find all available sequences
        sequence_names = set()

        # Check poses directory
        if load_poses:
            poses_dir = data_dir / "poses"
            if poses_dir.exists():
                sequence_names.update(f.stem for f in poses_dir.glob("*.npz"))

        # Check cameras directory
        if load_cameras:
            cameras_dir = data_dir / "cameras"
            if cameras_dir.exists():
                sequence_names.update(f.stem for f in cameras_dir.glob("*.npz"))

        # Check boxes file
        if load_bboxes:
            boxes_path = data_dir / "boxes.npz"
            if boxes_path.exists():
                npz_data = np.load(boxes_path, allow_pickle=True)
                sequence_names.update(npz_data.files)

        # Load each sequence
        videos_dict = {}
        for sequence_name in sorted(sequence_names):
            videos_dict[sequence_name] = cls.load(
                data_dir, sequence_name,
                load_poses, load_cameras, load_bboxes,
                load_skel_2d, load_skel_3d
            )

        return videos_dict

    def get_frame(self, frame_idx: int, load_image: bool = False, images_dir: Optional[Path] = None) -> ImageMetadata:
        """
        Extract ImageMetadata for a specific frame.

        Args:
            frame_idx: Frame index
            load_image: Whether to load the image
            images_dir: Optional images directory (required if load_image=True)

        Returns:
            ImageMetadata instance for the frame
        """
        if frame_idx < 0 or frame_idx >= self.num_frames:
            raise ValueError(f"Frame index {frame_idx} out of range [0, {self.num_frames})")

        # Create ImageMetadata
        frame_meta = ImageMetadata(
            self.sequence_name,
            frame_idx,
            self.poses,
            self.cameras,
            self.bboxes,
            self.skel_2d,
            self.skel_3d
        )

        # Load image if requested
        if load_image:
            if images_dir is None:
                raise ValueError("images_dir must be provided when load_image=True")
            frame_meta.load_image(images_dir)

        return frame_meta

    def __repr__(self) -> str:
        """String representation."""
        components = []
        if self.poses: components.append("poses")
        if self.cameras: components.append("cameras")
        if self.bboxes: components.append("bboxes")
        if self.skel_2d: components.append("skel_2d")
        if self.skel_3d: components.append("skel_3d")

        return (f"VideoMetadata(sequence='{self.sequence_name}', "
                f"frames={self.num_frames}, "
                f"components=[{', '.join(components)}])")

    def __len__(self) -> int:
        """Return number of frames."""
        return self.num_frames
