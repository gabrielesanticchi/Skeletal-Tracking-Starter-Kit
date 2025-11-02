"""
Base detector interface for object detection evaluation pipeline.

All detector implementations must inherit from this base class.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Tuple, Optional
import numpy as np
from pathlib import Path


class BaseDetector(ABC):
    """
    Base class for all object detectors.

    All detector implementations must implement the detect() method.
    """

    def __init__(self, config: Dict):
        """
        Initialize the detector with configuration.

        Args:
            config: Dictionary containing detector-specific configuration
        """
        self.config = config
        self.model = None
        self.device = config.get('device', 'cpu')
        self.confidence_threshold = config.get('confidence_threshold', 0.5)
        self.nms_threshold = config.get('nms_threshold', 0.45)
        self.input_size = config.get('input_size', 640)

    @abstractmethod
    def load_model(self, model_path: Optional[Path] = None) -> None:
        """
        Load the detection model.

        Args:
            model_path: Path to model weights (if None, load pretrained)
        """
        pass

    @abstractmethod
    def detect(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Perform object detection on a single image.

        Args:
            image: Input image (H, W, 3) in BGR format

        Returns:
            boxes: (N, 4) array of bounding boxes in XYXY format [x1, y1, x2, y2]
            scores: (N,) array of confidence scores
            classes: (N,) array of class IDs
        """
        pass

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess image for detection (can be overridden by subclasses).

        Args:
            image: Input image (H, W, 3) in BGR format

        Returns:
            Preprocessed image
        """
        return image

    def postprocess(
        self,
        boxes: np.ndarray,
        scores: np.ndarray,
        classes: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Postprocess detection results (can be overridden by subclasses).

        Args:
            boxes: (N, 4) array of bounding boxes
            scores: (N,) array of scores
            classes: (N,) array of class IDs

        Returns:
            Filtered boxes, scores, and classes
        """
        # Filter by confidence threshold
        mask = scores >= self.confidence_threshold

        return boxes[mask], scores[mask], classes[mask]

    def get_name(self) -> str:
        """Get detector name."""
        return self.__class__.__name__

    def get_config(self) -> Dict:
        """Get detector configuration."""
        return self.config
