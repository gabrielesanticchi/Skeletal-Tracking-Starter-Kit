"""
YOLO detector using ONNX Runtime - fully open-source and commercial-friendly.

This implementation uses ONNX Runtime to run YOLO models exported to ONNX format.
ONNX Runtime is MIT licensed and can be used commercially without restrictions.

You can export models from various sources (PyTorch, TensorFlow, etc.) to ONNX format.
"""

import numpy as np
import cv2
from pathlib import Path
from typing import Tuple, Optional, Dict, List
import sys

# Add parent directory to path for base class import
sys.path.append(str(Path(__file__).parent.parent))
from base_detector import BaseDetector


class YOLOONNXDetector(BaseDetector):
    """
    YOLO object detector using ONNX Runtime (MIT License - commercial-friendly).

    Supports any YOLO model exported to ONNX format.
    """

    def __init__(self, config: Dict):
        """
        Initialize YOLO ONNX detector.

        Args:
            config: Configuration dictionary with keys:
                - model_path: Path to ONNX model file
                - input_size: Model input size (default: 640)
                - confidence_threshold: Confidence threshold
                - nms_threshold: NMS IoU threshold
                - class_names: List of class names (optional)
                - class_filter: List of class IDs to keep (None = all)
        """
        super().__init__(config)
        self.model_path = config.get('model_path', None)
        self.class_names = config.get('class_names', self._get_coco_names())
        self.class_filter = config.get('class_filter', None)

    def load_model(self, model_path: Optional[Path] = None) -> None:
        """
        Load ONNX model.

        Args:
            model_path: Path to ONNX model file
        """
        try:
            import onnxruntime as ort
            self.HAS_ONNX = True
        except ImportError:
            raise ImportError(
                "ONNX Runtime not found. Install with: uv pip install onnxruntime\n"
                "(MIT License - commercial-friendly)"
            )

        if model_path is None:
            model_path = self.model_path

        if model_path is None or not Path(model_path).exists():
            raise ValueError(
                f"ONNX model not found at {model_path}\n"
                "Please provide a valid ONNX model file.\n"
                "You can export YOLO models to ONNX using various tools."
            )

        # Create ONNX Runtime session
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if self.device == 'cuda' else ['CPUExecutionProvider']

        self.session = ort.InferenceSession(str(model_path), providers=providers)

        # Get model input/output names and shapes
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [output.name for output in self.session.get_outputs()]

        input_shape = self.session.get_inputs()[0].shape
        print(f"  Model input shape: {input_shape}")
        print(f"  Model output names: {self.output_names}")
        print(f"  ✓ ONNX model loaded")

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess image for YOLO model.

        Args:
            image: Input image (H, W, 3) in BGR format

        Returns:
            Preprocessed image tensor (1, 3, H, W) normalized to [0, 1]
        """
        # Resize to input size while maintaining aspect ratio
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Letterbox resize
        h, w = img_rgb.shape[:2]
        scale = min(self.input_size / h, self.input_size / w)
        new_h, new_w = int(h * scale), int(w * scale)

        img_resized = cv2.resize(img_rgb, (new_w, new_h))

        # Pad to square
        img_padded = np.full((self.input_size, self.input_size, 3), 114, dtype=np.uint8)
        top = (self.input_size - new_h) // 2
        left = (self.input_size - new_w) // 2
        img_padded[top:top+new_h, left:left+new_w] = img_resized

        # Normalize and transpose to (C, H, W)
        img_normalized = img_padded.astype(np.float32) / 255.0
        img_transposed = np.transpose(img_normalized, (2, 0, 1))

        # Add batch dimension
        img_batch = np.expand_dims(img_transposed, axis=0)

        # Store scale and padding for postprocessing
        self._scale = scale
        self._pad_top = top
        self._pad_left = left

        return img_batch

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
        if not hasattr(self, 'session'):
            raise ValueError("Model not loaded. Call load_model() first.")

        # Preprocess
        input_tensor = self.preprocess(image)

        # Run inference
        outputs = self.session.run(self.output_names, {self.input_name: input_tensor})

        # Postprocess outputs
        boxes, scores, classes = self._postprocess_outputs(outputs, image.shape[:2])

        return boxes, scores, classes

    def _postprocess_outputs(
        self,
        outputs: List[np.ndarray],
        original_shape: Tuple[int, int]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Postprocess ONNX model outputs.

        Args:
            outputs: List of output arrays from ONNX model
            original_shape: (H, W) of original image

        Returns:
            boxes, scores, classes arrays
        """
        # YOLO output format: (1, num_boxes, 85) for COCO (80 classes + 4 bbox + 1 obj)
        # or (1, 25200, 85) for YOLOv5/v8
        predictions = outputs[0]  # (1, num_boxes, num_attrs)

        # Remove batch dimension
        predictions = predictions[0]  # (num_boxes, num_attrs)

        # Extract boxes, objectness, and class scores
        # Format: [x_center, y_center, width, height, objectness, class1_score, ..., classN_score]
        boxes_xywh = predictions[:, :4]
        objectness = predictions[:, 4]
        class_scores = predictions[:, 5:]

        # Get class with highest score and its confidence
        class_ids = np.argmax(class_scores, axis=1)
        class_confidences = np.max(class_scores, axis=1)

        # Final confidence = objectness * class_confidence
        confidences = objectness * class_confidences

        # Filter by confidence threshold
        mask = confidences >= self.confidence_threshold
        boxes_xywh = boxes_xywh[mask]
        confidences = confidences[mask]
        class_ids = class_ids[mask]

        if len(boxes_xywh) == 0:
            return np.array([]), np.array([]), np.array([])

        # Convert from xywh to xyxy
        boxes_xyxy = np.copy(boxes_xywh)
        boxes_xyxy[:, 0] = boxes_xywh[:, 0] - boxes_xywh[:, 2] / 2  # x1
        boxes_xyxy[:, 1] = boxes_xywh[:, 1] - boxes_xywh[:, 3] / 2  # y1
        boxes_xyxy[:, 2] = boxes_xywh[:, 0] + boxes_xywh[:, 2] / 2  # x2
        boxes_xyxy[:, 3] = boxes_xywh[:, 1] + boxes_xywh[:, 3] / 2  # y2

        # Rescale boxes to original image coordinates
        boxes_xyxy[:, [0, 2]] -= self._pad_left
        boxes_xyxy[:, [1, 3]] -= self._pad_top
        boxes_xyxy /= self._scale

        # Clip to image boundaries
        h_orig, w_orig = original_shape
        boxes_xyxy[:, [0, 2]] = np.clip(boxes_xyxy[:, [0, 2]], 0, w_orig)
        boxes_xyxy[:, [1, 3]] = np.clip(boxes_xyxy[:, [1, 3]], 0, h_orig)

        # Apply NMS
        indices = self._nms(boxes_xyxy, confidences, self.nms_threshold)
        boxes_xyxy = boxes_xyxy[indices]
        confidences = confidences[indices]
        class_ids = class_ids[indices]

        # Filter by class if specified
        if self.class_filter is not None:
            mask = np.isin(class_ids, self.class_filter)
            boxes_xyxy = boxes_xyxy[mask]
            confidences = confidences[mask]
            class_ids = class_ids[mask]

        return boxes_xyxy, confidences, class_ids

    @staticmethod
    def _nms(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float) -> np.ndarray:
        """
        Non-Maximum Suppression.

        Args:
            boxes: (N, 4) array of boxes in XYXY format
            scores: (N,) array of scores
            iou_threshold: IoU threshold for NMS

        Returns:
            Indices of boxes to keep
        """
        if len(boxes) == 0:
            return np.array([], dtype=int)

        # Sort by score
        sorted_indices = np.argsort(scores)[::-1]

        keep = []
        while len(sorted_indices) > 0:
            # Pick box with highest score
            current = sorted_indices[0]
            keep.append(current)

            if len(sorted_indices) == 1:
                break

            # Compute IoU with remaining boxes
            current_box = boxes[current]
            remaining_boxes = boxes[sorted_indices[1:]]

            ious = YOLOONNXDetector._compute_iou_vectorized(current_box, remaining_boxes)

            # Keep boxes with IoU below threshold
            sorted_indices = sorted_indices[1:][ious < iou_threshold]

        return np.array(keep, dtype=int)

    @staticmethod
    def _compute_iou_vectorized(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
        """
        Compute IoU between one box and multiple boxes.

        Args:
            box: (4,) array [x1, y1, x2, y2]
            boxes: (N, 4) array of boxes

        Returns:
            (N,) array of IoU values
        """
        # Compute intersection
        x1 = np.maximum(box[0], boxes[:, 0])
        y1 = np.maximum(box[1], boxes[:, 1])
        x2 = np.minimum(box[2], boxes[:, 2])
        y2 = np.minimum(box[3], boxes[:, 3])

        intersection = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)

        # Compute union
        box_area = (box[2] - box[0]) * (box[3] - box[1])
        boxes_area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        union = box_area + boxes_area - intersection

        return intersection / np.maximum(union, 1e-6)

    @staticmethod
    def _get_coco_names() -> List[str]:
        """Get COCO class names."""
        return [
            'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck',
            'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench',
            'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
            'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
            'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
            'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
            'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
            'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
            'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
            'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
            'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
            'toothbrush'
        ]
