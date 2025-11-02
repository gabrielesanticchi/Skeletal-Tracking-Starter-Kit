"""
Bounding boxes data handling.

Provides object-oriented interface to bounding box annotations.
"""

import numpy as np
import cv2
from pathlib import Path
from typing import Dict, List, Optional, Tuple


class BBoxesData:
    """
    Handler for bounding box data.

    Bounding boxes are stored in XYXY format (x_min, y_min, x_max, y_max).
    NaN values indicate subject not present in frame.

    Attributes:
        sequence_name: Name of the sequence
        boxes: (num_frames, num_subjects, 4) array of bounding boxes
        num_frames: Number of frames
        num_subjects: Maximum number of subjects
    """

    def __init__(self, sequence_name: str, boxes: np.ndarray):
        """
        Initialize BBoxesData.

        Args:
            sequence_name: Name of the sequence
            boxes: (num_frames, num_subjects, 4) array in XYXY format
        """
        self.sequence_name = sequence_name
        self.boxes = boxes
        self.num_frames = boxes.shape[0]
        self.num_subjects = boxes.shape[1]

    @classmethod
    def load(cls, boxes_path: Path, sequence_name: str) -> 'BBoxesData':
        """
        Load bounding boxes for a specific sequence.

        Args:
            boxes_path: Path to boxes.npz file
            sequence_name: Name of the sequence

        Returns:
            BBoxesData instance
        """
        if not boxes_path.exists():
            raise FileNotFoundError(f"Boxes file not found: {boxes_path}")

        npz_data = np.load(boxes_path, allow_pickle=True)

        if sequence_name not in npz_data.files:
            raise KeyError(f"Sequence '{sequence_name}' not found in boxes file")

        return cls(sequence_name, npz_data[sequence_name])

    @classmethod
    def load_all(cls, boxes_path: Path) -> Dict[str, 'BBoxesData']:
        """
        Load all sequences from boxes file.

        Args:
            boxes_path: Path to boxes.npz file

        Returns:
            Dictionary mapping sequence names to BBoxesData instances
        """
        if not boxes_path.exists():
            raise FileNotFoundError(f"Boxes file not found: {boxes_path}")

        npz_data = np.load(boxes_path, allow_pickle=True)
        boxes_dict = {}

        for sequence_name in npz_data.files:
            boxes_dict[sequence_name] = cls(sequence_name, npz_data[sequence_name])

        return boxes_dict

    def get_frame_boxes(self, frame_idx: int, valid_only: bool = True) -> np.ndarray:
        """
        Get bounding boxes for a specific frame.

        Args:
            frame_idx: Frame index
            valid_only: If True, filter out NaN boxes

        Returns:
            (N, 4) array of bounding boxes (N <= num_subjects if valid_only)
        """
        if frame_idx < 0 or frame_idx >= self.num_frames:
            raise ValueError(f"Frame index {frame_idx} out of range [0, {self.num_frames})")

        boxes = self.boxes[frame_idx]

        if valid_only:
            # Filter out NaN boxes
            valid_mask = ~np.isnan(boxes[:, 0])
            return boxes[valid_mask]

        return boxes

    def get_subject_boxes(self, subject_idx: int, valid_only: bool = True) -> np.ndarray:
        """
        Get bounding boxes for a specific subject across all frames.

        Args:
            subject_idx: Subject index
            valid_only: If True, filter out NaN boxes

        Returns:
            (M, 4) array of bounding boxes (M <= num_frames if valid_only)
        """
        if subject_idx < 0 or subject_idx >= self.num_subjects:
            raise ValueError(f"Subject index {subject_idx} out of range [0, {self.num_subjects})")

        boxes = self.boxes[:, subject_idx, :]

        if valid_only:
            # Filter out NaN boxes
            valid_mask = ~np.isnan(boxes[:, 0])
            return boxes[valid_mask]

        return boxes

    def count_valid_subjects(self, frame_idx: int) -> int:
        """
        Count number of valid subjects in a frame.

        Args:
            frame_idx: Frame index

        Returns:
            Number of valid (non-NaN) subjects
        """
        boxes = self.get_frame_boxes(frame_idx, valid_only=False)
        return np.sum(~np.isnan(boxes[:, 0]))

    def visualize_frame(
        self,
        image: np.ndarray,
        frame_idx: int,
        show_labels: bool = True,
        color_palette: Optional[List[Tuple[int, int, int]]] = None
    ) -> np.ndarray:
        """
        Visualize bounding boxes on an image.

        Args:
            image: Input image (BGR format)
            frame_idx: Frame index
            show_labels: Whether to show subject ID labels
            color_palette: Optional list of BGR colors for different subjects

        Returns:
            Image with bounding boxes drawn
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
                (128, 0, 128),  # Purple
                (255, 165, 0),  # Orange
                (0, 128, 128),  # Teal
                (128, 128, 0),  # Olive
            ]

        img_display = image.copy()
        boxes = self.boxes[frame_idx]
        num_valid = 0

        # Draw each bounding box
        for subject_idx, bbox in enumerate(boxes):
            # Skip if bbox is NaN
            if np.any(np.isnan(bbox)):
                continue

            num_valid += 1

            # Extract coordinates
            x_min, y_min, x_max, y_max = bbox.astype(int)

            # Select color
            color = color_palette[subject_idx % len(color_palette)]

            # Draw rectangle
            cv2.rectangle(img_display, (x_min, y_min), (x_max, y_max), color, 2)

            # Draw label if requested
            if show_labels:
                label = f"ID:{subject_idx}"
                label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                label_y = max(y_min - 5, label_size[1] + 5)

                # Draw label background
                cv2.rectangle(
                    img_display,
                    (x_min, label_y - label_size[1] - 4),
                    (x_min + label_size[0] + 4, label_y + 2),
                    color,
                    -1
                )

                # Draw label text
                cv2.putText(
                    img_display,
                    label,
                    (x_min + 2, label_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA
                )

        # Add info text
        info_text = f"Sequence: {self.sequence_name} | Frame: {frame_idx} | Subjects: {num_valid}"
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

    def to_xywh(self, frame_idx: Optional[int] = None) -> np.ndarray:
        """
        Convert boxes to XYWH format (x_center, y_center, width, height).

        Args:
            frame_idx: Optional frame index (if None, converts all frames)

        Returns:
            Boxes in XYWH format
        """
        if frame_idx is not None:
            boxes = self.boxes[frame_idx:frame_idx+1]
        else:
            boxes = self.boxes

        # Convert XYXY to XYWH
        x_min, y_min, x_max, y_max = boxes[..., 0], boxes[..., 1], boxes[..., 2], boxes[..., 3]
        x_center = (x_min + x_max) / 2
        y_center = (y_min + y_max) / 2
        width = x_max - x_min
        height = y_max - y_min

        xywh = np.stack([x_center, y_center, width, height], axis=-1)

        return xywh[0] if frame_idx is not None else xywh

    def __repr__(self) -> str:
        """String representation."""
        return (f"BBoxesData(sequence='{self.sequence_name}', "
                f"frames={self.num_frames}, subjects={self.num_subjects})")

    def __str__(self) -> str:
        """Detailed string representation."""
        total_valid = np.sum(~np.isnan(self.boxes[:, :, 0]))
        return (f"BBoxesData for sequence '{self.sequence_name}':\n"
                f"  Frames: {self.num_frames}\n"
                f"  Max subjects: {self.num_subjects}\n"
                f"  Total valid boxes: {total_valid}\n"
                f"  Shape: {self.boxes.shape}")
