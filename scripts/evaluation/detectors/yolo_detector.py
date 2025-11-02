"""
YOLO detector implementation using PyTorch Hub (YOLOv5/YOLOv8).

This implementation uses the official Ultralytics YOLOv8 or YOLOv5 models,
which are open-source (AGPL-3.0) but require commercial license for commercial use.

For truly open-source commercial-friendly alternative, see yolo_openvino_detector.py
"""

import numpy as np
import cv2
import torch
from pathlib import Path
from typing import Tuple, Optional, Dict
import sys

# Add parent directory to path for base class import
sys.path.append(str(Path(__file__).parent.parent))
from base_detector import BaseDetector


class YOLODetector(BaseDetector):
    """
    YOLO object detector using Ultralytics implementation.

    Note: Ultralytics YOLO is AGPL-3.0 licensed. For commercial use,
    you may need a commercial license or use an alternative implementation.
    """

    def __init__(self, config: Dict):
        """
        Initialize YOLO detector.

        Args:
            config: Configuration dictionary with keys:
                - model_name: 'yolov8n', 'yolov8s', 'yolov8m', 'yolov8l', 'yolov8x' or
                             'yolov5n', 'yolov5s', 'yolov5m', 'yolov5l', 'yolov5x'
                - confidence_threshold: Confidence threshold for detections
                - nms_threshold: NMS IoU threshold
                - device: 'cpu' or 'cuda'
                - class_filter: List of class IDs to keep (None = all classes)
        """
        super().__init__(config)
        self.model_name = config.get('model_name', 'yolov8n')
        self.class_filter = config.get('class_filter', None)  # e.g., [0] for person class

    def load_model(self, model_path: Optional[Path] = None) -> None:
        """
        Load YOLO model.

        Args:
            model_path: Path to custom model weights (if None, load pretrained)
        """
        try:
            from ultralytics import YOLO
            HAS_ULTRALYTICS = True
        except ImportError:
            HAS_ULTRALYTICS = False

        if not HAS_ULTRALYTICS:
            raise ImportError(
                "Ultralytics YOLO not found. Install with: uv pip install ultralytics\n"
                "Note: AGPL-3.0 license, may require commercial license for commercial use"
            )

        if model_path and model_path.exists():
            print(f"  Loading custom YOLO model from {model_path}")
            self.model = YOLO(str(model_path))
        else:
            print(f"  Loading pretrained YOLO model: {self.model_name}")
            self.model = YOLO(f"{self.model_name}.pt")

        # Move to device
        self.model.to(self.device)

        print(f"  ✓ YOLO model loaded on {self.device}")

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
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")

        # Run inference
        results = self.model.predict(
            image,
            conf=self.confidence_threshold,
            iou=self.nms_threshold,
            verbose=False
        )[0]

        # Extract boxes, scores, and classes
        boxes = results.boxes.xyxy.cpu().numpy()  # (N, 4) [x1, y1, x2, y2]
        scores = results.boxes.conf.cpu().numpy()  # (N,)
        classes = results.boxes.cls.cpu().numpy()  # (N,)

        # Filter by class if specified
        if self.class_filter is not None:
            mask = np.isin(classes, self.class_filter)
            boxes = boxes[mask]
            scores = scores[mask]
            classes = classes[mask]

        return boxes, scores, classes
