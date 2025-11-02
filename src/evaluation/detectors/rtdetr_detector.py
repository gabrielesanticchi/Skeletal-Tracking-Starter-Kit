"""
RT-DETR detector implementation using torchvision (BSD-3-Clause - commercial-friendly).

RT-DETR (Real-Time Detection Transformer) is a high-performance real-time object detector
developed by Baidu. This implementation uses the torchvision version which is
BSD-3-Clause licensed and fully commercial-friendly.
"""

import numpy as np
import torch
import cv2
from pathlib import Path
from typing import Tuple, Optional, Dict, List
import sys

# Add parent directory to path for base class import
sys.path.append(str(Path(__file__).parent.parent))
from base_detector import BaseDetector


class RTDETRDetector(BaseDetector):
    """
    RT-DETR object detector using torchvision (BSD-3-Clause License - commercial-friendly).

    RT-DETR combines the efficiency of YOLO-style detectors with the accuracy of DETR.
    """

    def __init__(self, config: Dict):
        """
        Initialize RT-DETR detector.

        Args:
            config: Configuration dictionary with keys:
                - model_name: 'rtdetr_r18', 'rtdetr_r34', 'rtdetr_r50', 'rtdetr_r101'
                - confidence_threshold: Confidence threshold
                - device: 'cpu' or 'cuda'
                - class_filter: List of class IDs to keep (None = all)
        """
        super().__init__(config)
        self.model_name = config.get('model_name', 'rtdetr_r50')
        self.class_filter = config.get('class_filter', None)

    def load_model(self, model_path: Optional[Path] = None) -> None:
        """
        Load RT-DETR model.

        Args:
            model_path: Path to custom model weights (if None, load pretrained)
        """
        try:
            import torchvision
            from torchvision.models.detection import rtdetr_r50_fpn, rtdetr_r18_fpn, rtdetr_r34_fpn, rtdetr_r101_fpn
            from torchvision.models.detection import RTDETR_R50_FPN_Weights, RTDETR_R18_FPN_Weights, RTDETR_R34_FPN_Weights, RTDETR_R101_FPN_Weights
            HAS_RTDETR = True
        except ImportError:
            HAS_RTDETR = False

        if not HAS_RTDETR:
            raise ImportError(
                "RT-DETR not found in torchvision. Install with: uv pip install 'torch torchvision'\n"
                "Note: Requires torchvision >= 0.18 for RT-DETR support\n"
                "(BSD-3-Clause License - commercial-friendly)"
            )

        # Select model and weights based on configuration
        model_map = {
            'rtdetr_r18': (rtdetr_r18_fpn, RTDETR_R18_FPN_Weights.DEFAULT),
            'rtdetr_r34': (rtdetr_r34_fpn, RTDETR_R34_FPN_Weights.DEFAULT),
            'rtdetr_r50': (rtdetr_r50_fpn, RTDETR_R50_FPN_Weights.DEFAULT),
            'rtdetr_r101': (rtdetr_r101_fpn, RTDETR_R101_FPN_Weights.DEFAULT),
        }

        if self.model_name not in model_map:
            raise ValueError(
                f"Unknown model name: {self.model_name}. "
                f"Choose from: {list(model_map.keys())}"
            )

        model_fn, weights = model_map[self.model_name]

        if model_path and model_path.exists():
            print(f"  Loading custom RT-DETR model from {model_path}")
            self.model = model_fn(weights=None)
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        else:
            print(f"  Loading pretrained RT-DETR model: {self.model_name}")
            self.model = model_fn(weights=weights)

        self.model.to(self.device)
        self.model.eval()

        # Get transforms from weights
        self.transforms = weights.transforms()

        print(f"  ✓ RT-DETR model loaded on {self.device}")

    def preprocess(self, image: np.ndarray) -> torch.Tensor:
        """
        Preprocess image for RT-DETR model.

        Args:
            image: Input image (H, W, 3) in BGR format

        Returns:
            Preprocessed image tensor
        """
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Convert to tensor and apply transforms
        image_tensor = torch.from_numpy(image_rgb).permute(2, 0, 1).float() / 255.0

        # Apply model-specific transforms
        if self.transforms is not None:
            image_tensor = self.transforms(image_tensor)

        return image_tensor

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

        # Preprocess
        image_tensor = self.preprocess(image)
        image_tensor = image_tensor.to(self.device)

        # Run inference
        with torch.no_grad():
            predictions = self.model([image_tensor])[0]

        # Extract results
        boxes = predictions['boxes'].cpu().numpy()  # (N, 4) [x1, y1, x2, y2]
        scores = predictions['scores'].cpu().numpy()  # (N,)
        classes = predictions['labels'].cpu().numpy()  # (N,)

        # Filter by confidence threshold
        mask = scores >= self.confidence_threshold
        boxes = boxes[mask]
        scores = scores[mask]
        classes = classes[mask]

        # Filter by class if specified (COCO classes are 1-indexed, so subtract 1 for filter)
        if self.class_filter is not None:
            # Note: torchvision uses 1-indexed classes, but we use 0-indexed in config
            mask = np.isin(classes - 1, self.class_filter)
            boxes = boxes[mask]
            scores = scores[mask]
            classes = classes[mask] - 1  # Convert to 0-indexed

        return boxes, scores, classes
