# Object Detection and Tracking Evaluation Pipeline

A flexible, plug-in based pipeline for evaluating object detection and tracking algorithms on video sequences.

## 🌟 Features

- **Plug-in Architecture**: Easy to add new detectors and trackers
- **Commercial-Friendly**: Uses only MIT/BSD licensed components
- **YAML Configuration**: Easy configuration through YAML files
- **Multiple Detectors**:
  - **RT-DETR** (BSD-3-Clause): High-performance real-time detector from torchvision
  - **YOLO ONNX** (MIT): YOLO models via ONNX Runtime (commercial-friendly)
- **ByteTrack Tracker** (MIT): State-of-the-art multi-object tracking
- **Comprehensive Output**: Video visualization and prediction files

## 📦 Installation

### Required Dependencies

```bash
# Activate virtual environment
source .venv/bin/activate

# Core dependencies (required)
uv pip install numpy opencv-python pyyaml scipy tqdm

# For RT-DETR detector
uv pip install torch torchvision

# For YOLO ONNX detector
uv pip install onnxruntime

# Optional: For GPU support
uv pip install onnxruntime-gpu  # For ONNX with GPU
```

### Optional: SMPLX for Bbox Generation

```bash
# For generating bounding boxes from SMPL poses
uv pip install smplx
```

## 🚀 Quick Start

### 1. Basic Usage with RT-DETR

```bash
# Run evaluation on a sequence
python scripts/evaluation/run_evaluation.py \
    --config scripts/evaluation/configs/rtdetr_example.yaml \
    --sequence ARG_FRA_183303 \
    --output-dir results/ARG_FRA_183303
```

### 2. Custom Configuration

Create a custom YAML configuration file:

```yaml
# my_config.yaml
detector:
  type: rtdetr
  module: detectors.rtdetr_detector
  class: RTDETRDetector
  model_name: rtdetr_r50  # or rtdetr_r18, rtdetr_r34, rtdetr_r101
  confidence_threshold: 0.6
  device: cuda  # or 'cpu'
  class_filter: [0]  # Person class only

tracker:
  enabled: true
  type: bytetrack
  module: trackers.bytetrack_tracker
  class: ByteTrackTracker
  track_thresh: 0.5
  track_buffer: 30
  match_thresh: 0.8
  low_thresh: 0.1
  min_hits: 3

output:
  save_video: true
  save_predictions: true
```

### 3. Using YOLO with ONNX

First, you need an ONNX model. You can export one using:

```python
# Example: Export YOLOv8 to ONNX (requires ultralytics for export only)
from ultralytics import YOLO
model = YOLO('yolov8n.pt')
model.export(format='onnx')
```

Then run:

```bash
python scripts/evaluation/run_evaluation.py \
    --config scripts/evaluation/configs/yolo_onnx_example.yaml \
    --sequence ARG_FRA_183303 \
    --output-dir results/ARG_FRA_183303
```

## 📂 Pipeline Architecture

### Directory Structure

```
scripts/evaluation/
├── __init__.py
├── README.md                      # This file
├── evaluation_pipeline.py         # Main pipeline class
├── base_detector.py              # Base detector interface
├── base_tracker.py               # Base tracker interface
├── run_evaluation.py             # CLI script
├── detectors/
│   ├── __init__.py
│   ├── rtdetr_detector.py        # RT-DETR implementation
│   ├── yolo_onnx_detector.py     # YOLO ONNX implementation
│   └── yolo_detector.py          # Ultralytics YOLO (AGPL, optional)
├── trackers/
│   ├── __init__.py
│   └── bytetrack_tracker.py      # ByteTrack implementation
└── configs/
    ├── rtdetr_example.yaml       # RT-DETR config example
    └── yolo_onnx_example.yaml    # YOLO ONNX config example
```

### Component Overview

#### 1. **Detectors**

All detectors inherit from `BaseDetector` and implement:
- `load_model()`: Load detection model
- `detect()`: Perform detection on a single image
- `preprocess()`: Optional preprocessing
- `postprocess()`: Optional postprocessing

**Available Detectors:**

| Detector | License | Description | Commercial Use |
|----------|---------|-------------|----------------|
| `RTDETRDetector` | BSD-3-Clause | TorchVision RT-DETR | ✅ Yes |
| `YOLOONNXDetector` | MIT | YOLO via ONNX Runtime | ✅ Yes |
| `YOLODetector` | AGPL-3.0 | Ultralytics YOLO | ⚠️ Requires license |

#### 2. **Trackers**

All trackers inherit from `BaseTracker` and implement:
- `update()`: Update tracker with new detections
- `reset()`: Reset tracker state

**Available Trackers:**

| Tracker | License | Description | Features |
|---------|---------|-------------|----------|
| `ByteTrackTracker` | MIT | ByteTrack algorithm | Kalman filter, Hungarian matching, two-stage association |

#### 3. **Evaluation Pipeline**

Main orchestrator that:
1. Loads detector and tracker from configuration
2. Processes images through detection
3. Optionally applies tracking
4. Generates visualizations and predictions

## 🔧 Configuration Reference

### Detector Configuration

```yaml
detector:
  type: <detector_type>           # Detector type identifier
  module: <module_path>           # Python module path
  class: <class_name>             # Detector class name
  model_name: <model_name>        # Model variant
  model_path: <path_or_null>      # Custom weights path (null = pretrained)
  confidence_threshold: 0.5       # Detection confidence threshold
  nms_threshold: 0.45            # NMS IoU threshold (for ONNX)
  device: cpu                     # 'cpu' or 'cuda'
  input_size: 640                 # Input image size (for ONNX)
  class_filter: [0]               # Classes to keep (null = all)
```

### Tracker Configuration

```yaml
tracker:
  enabled: true                   # Enable/disable tracking
  type: bytetrack                 # Tracker type
  module: <module_path>           # Python module path
  class: <class_name>             # Tracker class name
  track_thresh: 0.5               # High confidence threshold
  track_buffer: 30                # Frames to keep lost tracks
  match_thresh: 0.8               # IoU threshold for matching
  low_thresh: 0.1                 # Low confidence threshold
  min_hits: 3                     # Minimum hits to confirm track
```

### Output Configuration

```yaml
output:
  save_video: true                # Save annotated video
  save_predictions: true          # Save predictions as .npz file

visualization:
  show_detections: true           # Draw detection boxes
  show_tracks: true               # Draw track IDs
  fps: 25                         # Output video FPS
```

## 📊 Output Format

### 1. Predictions File

Saved as `<sequence>_predictions.npz` with format matching `boxes.npz`:

```python
{
    "<sequence_name>": np.ndarray of shape (num_frames, num_detections, 4)
    # Format: XYXY (x_min, y_min, x_max, y_max)
    # np.nan indicates no detection in that slot
}
```

### 2. Visualization Video

Saved as `<sequence>_results.mp4` with:
- Green boxes: Detections with confidence scores
- Colored boxes: Tracks with track IDs (if tracking enabled)

## 🔌 Adding Custom Detectors

### 1. Create Detector Class

```python
# detectors/my_detector.py
from base_detector import BaseDetector
import numpy as np

class MyDetector(BaseDetector):
    def load_model(self, model_path=None):
        # Load your model
        self.model = ...

    def detect(self, image):
        # Run detection
        boxes = ...  # (N, 4) XYXY format
        scores = ...  # (N,)
        classes = ...  # (N,)
        return boxes, scores, classes
```

### 2. Create Configuration

```yaml
# configs/my_detector.yaml
detector:
  type: my_detector
  module: detectors.my_detector
  class: MyDetector
  # ... custom parameters
```

### 3. Run Evaluation

```bash
python scripts/evaluation/run_evaluation.py \
    --config configs/my_detector.yaml \
    --sequence <sequence_name>
```

## 🔌 Adding Custom Trackers

Similar to detectors, inherit from `BaseTracker`:

```python
# trackers/my_tracker.py
from base_tracker import BaseTracker
import numpy as np

class MyTracker(BaseTracker):
    def update(self, boxes, scores, classes=None):
        # Update tracking
        tracks = ...  # (M, 5) [x1, y1, x2, y2, track_id]
        return tracks
```

## 🎯 Use Cases

### 1. Evaluate Detection Quality

Compare generated bounding boxes against ground truth:

```bash
# Generate predictions
python scripts/evaluation/run_evaluation.py \
    --config configs/rtdetr_example.yaml \
    --sequence ARG_FRA_183303 \
    --output-dir results/ARG_FRA_183303

# Compare with ground truth using visualize_bboxes.py
python scripts/visualization/visualize_bboxes.py \
    --sequence ARG_FRA_183303 \
    --boxes-file results/ARG_FRA_183303/ARG_FRA_183303_predictions.npz
```

### 2. Test Different Detectors

```bash
# RT-DETR
python run_evaluation.py --config configs/rtdetr_example.yaml ...

# YOLO ONNX
python run_evaluation.py --config configs/yolo_onnx_example.yaml ...
```

### 3. Hyperparameter Tuning

Modify YAML configs to test different thresholds:
- Confidence threshold
- NMS threshold (YOLO)
- Tracking thresholds (ByteTrack)

### 4. Generate Training Data

Use high-quality detections as pseudo-labels for training:

```bash
# Generate boxes for all training sequences
for seq in data/images/*/; do
    seq_name=$(basename $seq)
    python run_evaluation.py \
        --config configs/rtdetr_example.yaml \
        --sequence $seq_name \
        --output-dir results/$seq_name
done
```

## 📈 Performance Tips

### 1. GPU Acceleration

```yaml
detector:
  device: cuda  # Use GPU
```

### 2. Batch Processing

For processing multiple sequences, create a shell script:

```bash
#!/bin/bash
for seq in ARG_FRA_183303 BRA_KOR_230503 CRO_MOR_190500; do
    python scripts/evaluation/run_evaluation.py \
        --config scripts/evaluation/configs/rtdetr_example.yaml \
        --sequence $seq \
        --output-dir results/$seq
done
```

### 3. Model Selection

Choose model based on speed/accuracy tradeoff:

| Model | Speed | Accuracy | Use Case |
|-------|-------|----------|----------|
| `rtdetr_r18` | ⚡⚡⚡ | ⭐⭐ | Real-time, low-resource |
| `rtdetr_r34` | ⚡⚡ | ⭐⭐⭐ | Balanced |
| `rtdetr_r50` | ⚡ | ⭐⭐⭐⭐ | High accuracy (default) |
| `rtdetr_r101` | 🐌 | ⭐⭐⭐⭐⭐ | Best accuracy |

## 🐛 Troubleshooting

### Issue: "Module not found"

**Solution**: Make sure you're running from the correct directory:

```bash
# Run from project root
cd /path/to/Skeletal-Tracking-Starter-Kit
python scripts/evaluation/run_evaluation.py ...
```

### Issue: "ONNX model not found"

**Solution**: Provide valid ONNX model path in config:

```yaml
detector:
  model_path: path/to/your/model.onnx
```

### Issue: "Out of memory"

**Solution**: Reduce batch size or use smaller model:

```yaml
detector:
  model_name: rtdetr_r18  # Smaller model
  device: cpu             # Use CPU instead of GPU
```

### Issue: "No detections"

**Solution**: Lower confidence threshold:

```yaml
detector:
  confidence_threshold: 0.3  # Lower threshold
```

## 📝 License

This evaluation pipeline uses only commercial-friendly open-source components:

- **RT-DETR**: BSD-3-Clause License ✅
- **ONNX Runtime**: MIT License ✅
- **ByteTrack**: MIT License ✅
- **Core Libraries**: BSD/MIT Licensed ✅

Note: The optional `YOLODetector` (Ultralytics) is AGPL-3.0 licensed and may require a commercial license for commercial use.

## 🔗 References

- **RT-DETR**: [Paper](https://arxiv.org/abs/2304.08069) | [TorchVision Docs](https://pytorch.org/vision/stable/models/rtdetr.html)
- **ByteTrack**: [Paper](https://arxiv.org/abs/2110.06864) | [GitHub](https://github.com/ifzhang/ByteTrack)
- **ONNX Runtime**: [Docs](https://onnxruntime.ai/) | [GitHub](https://github.com/microsoft/onnxruntime)

## 🤝 Contributing

To add new detectors or trackers:

1. Inherit from `BaseDetector` or `BaseTracker`
2. Implement required methods
3. Create example config YAML
4. Update this README
5. Test on sample sequences

## 📧 Support

For issues or questions:
- Check this README first
- Review example configs in `configs/`
- Consult the base classes: `base_detector.py`, `base_tracker.py`
