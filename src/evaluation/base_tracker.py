"""
Base tracker interface for object tracking evaluation pipeline.

All tracker implementations must inherit from this base class.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Tuple, Optional
import numpy as np


class BaseTracker(ABC):
    """
    Base class for all object trackers.

    All tracker implementations must implement the update() method.
    """

    def __init__(self, config: Dict):
        """
        Initialize the tracker with configuration.

        Args:
            config: Dictionary containing tracker-specific configuration
        """
        self.config = config
        self.tracks = []
        self.next_id = 0
        self.max_age = config.get('max_age', 30)
        self.min_hits = config.get('min_hits', 3)
        self.iou_threshold = config.get('iou_threshold', 0.3)

    @abstractmethod
    def update(
        self,
        boxes: np.ndarray,
        scores: np.ndarray,
        classes: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Update tracker with new detections.

        Args:
            boxes: (N, 4) array of bounding boxes in XYXY format [x1, y1, x2, y2]
            scores: (N,) array of confidence scores
            classes: (N,) array of class IDs (optional)

        Returns:
            tracks: (M, 5) array of tracked boxes with IDs [x1, y1, x2, y2, track_id]
                   or (M, 6) if classes included [x1, y1, x2, y2, track_id, class_id]
        """
        pass

    def reset(self) -> None:
        """Reset tracker state."""
        self.tracks = []
        self.next_id = 0

    def get_name(self) -> str:
        """Get tracker name."""
        return self.__class__.__name__

    def get_config(self) -> Dict:
        """Get tracker configuration."""
        return self.config

    @staticmethod
    def compute_iou(box1: np.ndarray, box2: np.ndarray) -> float:
        """
        Compute IoU between two boxes.

        Args:
            box1: (4,) array [x1, y1, x2, y2]
            box2: (4,) array [x1, y1, x2, y2]

        Returns:
            IoU value between 0 and 1
        """
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        intersection = max(0, x2 - x1) * max(0, y2 - y1)

        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

        union = area1 + area2 - intersection

        return intersection / union if union > 0 else 0.0

    @staticmethod
    def compute_iou_batch(boxes1: np.ndarray, boxes2: np.ndarray) -> np.ndarray:
        """
        Compute IoU matrix between two sets of boxes.

        Args:
            boxes1: (N, 4) array of boxes
            boxes2: (M, 4) array of boxes

        Returns:
            (N, M) array of IoU values
        """
        # Expand dimensions for broadcasting
        boxes1 = boxes1[:, None, :]  # (N, 1, 4)
        boxes2 = boxes2[None, :, :]  # (1, M, 4)

        # Compute intersection
        x1 = np.maximum(boxes1[..., 0], boxes2[..., 0])
        y1 = np.maximum(boxes1[..., 1], boxes2[..., 1])
        x2 = np.minimum(boxes1[..., 2], boxes2[..., 2])
        y2 = np.minimum(boxes1[..., 3], boxes2[..., 3])

        intersection = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)

        # Compute union
        area1 = (boxes1[..., 2] - boxes1[..., 0]) * (boxes1[..., 3] - boxes1[..., 1])
        area2 = (boxes2[..., 2] - boxes2[..., 0]) * (boxes2[..., 3] - boxes2[..., 1])

        union = area1 + area2 - intersection

        return intersection / np.maximum(union, 1e-6)
