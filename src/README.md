# Source Code Directory

This directory contains the core source code for the FIFA Skeletal Tracking Challenge toolkit, organized into modular components following OOP principles.

## Structure

```
src/
├── classes/                # Core data classes (OOP interface)
│   ├── __init__.py        # Package exports
│   ├── poses.py           # PosesData - SMPL pose parameters
│   ├── cameras.py         # CamerasData - Camera calibration
│   ├── bboxes.py          # BBoxesData - Bounding boxes
│   ├── skeleton.py        # Skeleton2DData, Skeleton3DData
│   ├── metadata.py        # ImageMetadata, VideoMetadata
│   └── README.md          # Detailed documentation
├── evaluation/            # Detection and tracking evaluation pipeline
│   ├── __init__.py
│   ├── base_detector.py   # Abstract detector interface
│   ├── base_tracker.py    # Abstract tracker interface
│   ├── evaluation_pipeline.py  # Main evaluation pipeline
│   ├── detectors/         # Detector implementations
│   │   ├── rtdetr_detector.py
│   │   ├── yolo_detector.py
│   │   └── yolo_onnx_detector.py
│   └── trackers/          # Tracker implementations
│       └── bytetrack_tracker.py
└── README.md             # This file
```

## Components

### 1. Classes Module (`classes/`)

Provides object-oriented interfaces to work with the dataset. Instead of manually managing NPZ files, use these classes for clean, maintainable code.

**Key Features:**
- Simplified data loading with `load()` and `load_all()` methods
- Built-in visualization methods
- Type-safe data access
- Unified interface across all data types

**Usage Example:**
```python
from classes import VideoMetadata

# Load complete sequence metadata
video = VideoMetadata.load(Path('data'), 'ARG_FRA_183303')

# Access specific frame
frame = video.get_frame(100, load_image=True, images_dir=Path('data/images'))

# Visualize
img_with_boxes = frame.visualize_bboxes()
```

**See:** [`classes/README.md`](classes/README.md) for detailed documentation.

---

### 2. Evaluation Module (`evaluation/`)

Plug-in based pipeline for evaluating object detection and tracking algorithms on video sequences.

**Key Features:**
- Abstract base classes for detectors and trackers
- Easy integration of new detectors/trackers
- YAML-based configuration
- Comprehensive output (video + predictions)

**Supported Detectors:**
- RT-DETR (transformer-based, BSD-3-Clause license)
- YOLO ONNX (MIT license)

**Supported Trackers:**
- ByteTrack (state-of-the-art MOT, MIT license)

**Usage Example:**
```bash
python scripts/evaluation/run_evaluation.py \
    --config scripts/evaluation/configs/rtdetr_example.yaml \
    --sequence ARG_FRA_183303 \
    --output-dir results/
```

**See:** `scripts/evaluation/README.md` for detailed documentation.

---

## Adding New Functionality

### Adding a New Data Class

1. Create a new file in `classes/`
2. Implement following the pattern:
   ```python
   class MyDataClass:
       def __init__(self, sequence_name: str, data: Any):
           self.sequence_name = sequence_name
           # ... store data

       @classmethod
       def load(cls, path: Path, sequence_name: str):
           # Load from file
           return cls(sequence_name, data)

       @classmethod
       def load_all(cls, path: Path):
           # Load all sequences
           return {seq: cls.load(path, seq) for seq in sequences}
   ```
3. Export in `classes/__init__.py`
4. Update `metadata.py` to integrate if needed

### Adding a New Detector

1. Create `detectors/my_detector.py`
2. Inherit from `BaseDetector`
3. Implement required methods:
   ```python
   from evaluation.base_detector import BaseDetector

   class MyDetector(BaseDetector):
       def load_model(self, model_path=None):
           # Load your model
           pass

       def detect(self, image):
           # Return boxes, scores, classes
           return boxes, scores, classes
   ```
4. Create a YAML config file
5. Run with `run_evaluation.py`

### Adding a New Tracker

Similar to detectors, inherit from `BaseTracker` and implement `update()` method.

---

## Design Principles

1. **Separation of Concerns**: Data loading, processing, and visualization are separated
2. **Single Responsibility**: Each class has a clear, focused purpose
3. **Open/Closed**: Easy to extend (new detectors/trackers) without modifying existing code
4. **DRY**: Reusable components reduce code duplication
5. **Type Safety**: Clear method signatures and return types

---

## Dependencies

Core dependencies:
```bash
pip install numpy opencv-python matplotlib scipy
```

Evaluation dependencies:
```bash
pip install torch torchvision pyyaml
```

---

## Integration with Scripts

The `scripts/` directory contains user-facing scripts that use these modules:

- **Visualization scripts** (`scripts/visualization/`): Use `classes` module
- **Evaluation scripts** (`scripts/evaluation/`): Use `evaluation` module
- **Preprocessing scripts** (`scripts/preprocessing/`): Work with raw data

All scripts add `src/` to the Python path:
```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))
```

---

## Testing

To verify the installation:
```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd() / 'src'))

from classes import (
    PosesData, CamerasData, BBoxesData,
    Skeleton2DData, Skeleton3DData,
    ImageMetadata, VideoMetadata
)

print("✓ All classes imported successfully")
```

---

## Future Enhancements

Potential additions:
- [ ] `utils/` module for common utilities (coordinate transformations, etc.)
- [ ] `models/` module for ML models (pose estimation, tracking)
- [ ] `metrics/` module for evaluation metrics (MPJPE, PCK, mAP)
- [ ] Unit tests for all classes
- [ ] Type stubs for better IDE support

---

## Support

For questions or issues:
1. Check the class-specific README: [`classes/README.md`](classes/README.md)
2. Check the main project README: [`../README.md`](../README.md)
3. Open an issue on GitHub

---

## License

This code follows the same MIT license as the main project.
