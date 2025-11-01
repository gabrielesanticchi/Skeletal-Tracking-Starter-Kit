"""
Visualize bounding boxes on images from the FIFA Skeletal Tracking Challenge dataset.

This script allows you to visualize bounding boxes overlaid on images.
If no arguments are provided, a random sequence and frame are selected.

Usage:
    # Random sequence and frame
    python scripts/visualization/visualize_bboxes.py

    # Specific sequence, random frame
    python scripts/visualization/visualize_bboxes.py --sequence ARG_FRA_183303

    # Specific sequence and frame
    python scripts/visualization/visualize_bboxes.py --sequence ARG_FRA_183303 --frame 100

    # Save output instead of displaying
    python scripts/visualization/visualize_bboxes.py --output visualization.jpg
"""

import numpy as np
import cv2
from pathlib import Path
import sys
import argparse
import random
from typing import Optional, Tuple


def get_project_root() -> Path:
    """Get the project root directory."""
    current = Path(__file__).resolve()
    return current.parent.parent.parent


def load_data(data_dir: Path) -> Tuple[dict, dict]:
    """
    Load bounding box data.

    Args:
        data_dir: Path to data directory

    Returns:
        Tuple of (boxes_data, sequences)
    """
    boxes_path = data_dir / "boxes.npz"
    if not boxes_path.exists():
        raise FileNotFoundError(f"boxes.npz not found at {boxes_path}")

    boxes = np.load(boxes_path, allow_pickle=True)
    boxes_data = {key: boxes[key] for key in boxes.files}

    return boxes_data, list(boxes_data.keys())


def select_random_sequence_and_frame(
    boxes_data: dict,
    sequences: list,
    sequence: Optional[str] = None,
    frame_idx: Optional[int] = None
) -> Tuple[str, int]:
    """
    Select a random or specified sequence and frame.

    Args:
        boxes_data: Dictionary of bounding box data
        sequences: List of available sequences
        sequence: Optional specific sequence name
        frame_idx: Optional specific frame index

    Returns:
        Tuple of (sequence_name, frame_index)
    """
    # Select sequence
    if sequence is None:
        sequence = random.choice(sequences)
        print(f"📌 Randomly selected sequence: {sequence}")
    else:
        if sequence not in sequences:
            raise ValueError(f"Sequence '{sequence}' not found in dataset. Available: {sequences}")
        print(f"📌 Using specified sequence: {sequence}")

    # Select frame
    num_frames = boxes_data[sequence].shape[0]
    if frame_idx is None:
        frame_idx = random.randint(0, num_frames - 1)
        print(f"📌 Randomly selected frame: {frame_idx} (out of {num_frames})")
    else:
        if frame_idx < 0 or frame_idx >= num_frames:
            raise ValueError(f"Frame index {frame_idx} out of range [0, {num_frames-1}]")
        print(f"📌 Using specified frame: {frame_idx} (out of {num_frames})")

    return sequence, frame_idx


def load_image(data_dir: Path, sequence: str, frame_idx: int) -> np.ndarray:
    """
    Load an image from the dataset.

    Args:
        data_dir: Path to data directory
        sequence: Sequence name
        frame_idx: Frame index

    Returns:
        Image array (BGR format)
    """
    image_path = data_dir / "images" / sequence / f"{frame_idx:05d}.jpg"
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found at {image_path}")

    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Failed to load image from {image_path}")

    return image


def draw_bboxes(
    image: np.ndarray,
    bboxes: np.ndarray,
    sequence: str,
    frame_idx: int
) -> np.ndarray:
    """
    Draw bounding boxes on the image.

    Args:
        image: Input image (BGR format)
        bboxes: Bounding boxes array of shape (num_subjects, 4) in XYXY format
        sequence: Sequence name (for display)
        frame_idx: Frame index (for display)

    Returns:
        Image with bounding boxes drawn
    """
    # Create a copy to avoid modifying original
    img_display = image.copy()

    # Define colors for different subjects (BGR format)
    colors = [
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

    num_subjects = 0

    # Draw each bounding box
    for subject_idx, bbox in enumerate(bboxes):
        # Skip if bbox is NaN (subject not present in frame)
        if np.any(np.isnan(bbox)):
            continue

        num_subjects += 1

        # Extract bbox coordinates (XYXY format)
        x_min, y_min, x_max, y_max = bbox.astype(int)

        # Select color (cycle through colors if more subjects than colors)
        color = colors[subject_idx % len(colors)]

        # Draw bounding box rectangle
        cv2.rectangle(img_display, (x_min, y_min), (x_max, y_max), color, 2)

        # Add subject ID label
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

    # Add info text at the top
    info_text = f"Sequence: {sequence} | Frame: {frame_idx} | Subjects: {num_subjects}"
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


def main():
    """Main visualization function."""
    parser = argparse.ArgumentParser(
        description="Visualize bounding boxes on images from FIFA Skeletal Tracking Challenge"
    )
    parser.add_argument(
        "--sequence",
        type=str,
        help="Sequence name (e.g., ARG_FRA_183303). If not specified, random sequence is selected."
    )
    parser.add_argument(
        "--frame",
        type=int,
        help="Frame index. If not specified, random frame is selected."
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Output file path to save visualization. If not specified, displays in window."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Base data directory (default: auto-detect from project root)"
    )

    args = parser.parse_args()

    # Setup paths
    if args.data_dir:
        data_dir = args.data_dir
    else:
        project_root = get_project_root()
        data_dir = project_root / "data"

    print("\n" + "="*80)
    print("BOUNDING BOX VISUALIZATION")
    print("="*80 + "\n")

    try:
        # Load data
        print("Loading bounding box data...")
        boxes_data, sequences = load_data(data_dir)
        print(f"✓ Loaded {len(sequences)} sequences\n")

        # Select sequence and frame
        sequence, frame_idx = select_random_sequence_and_frame(
            boxes_data,
            sequences,
            args.sequence,
            args.frame
        )

        # Load image
        print(f"\nLoading image...")
        image = load_image(data_dir, sequence, frame_idx)
        print(f"✓ Image loaded: {image.shape[1]}x{image.shape[0]} pixels")

        # Get bounding boxes for this frame
        bboxes = boxes_data[sequence][frame_idx]

        # Draw bounding boxes
        print(f"\nDrawing bounding boxes...")
        img_with_bboxes = draw_bboxes(image, bboxes, sequence, frame_idx)

        # Save or display
        if args.output:
            output_path = Path(args.output)
            cv2.imwrite(str(output_path), img_with_bboxes)
            print(f"✓ Visualization saved to: {output_path}")
        else:
            # Display in window
            window_name = f"Bounding Boxes - {sequence} - Frame {frame_idx}"
            cv2.imshow(window_name, img_with_bboxes)
            print(f"\n✓ Displaying visualization")
            print("Press any key to close the window...")
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        print("\n" + "="*80)
        print("VISUALIZATION COMPLETE")
        print("="*80 + "\n")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)

    return 0


if __name__ == "__main__":
    sys.exit(main())
