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

import sys
import cv2
import random
import argparse
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

from classes import BBoxesData, ImageMetadata


def get_project_root() -> Path:
    """Get the project root directory."""
    current = Path(__file__).resolve()
    return current.parent.parent.parent


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
        # Load bounding box data
        print("Loading bounding box data...")
        boxes_path = data_dir / "boxes.npz"
        boxes_dict = BBoxesData.load_all(boxes_path)
        sequences = list(boxes_dict.keys())
        print(f"✓ Loaded {len(sequences)} sequences with bounding boxes")
        print(f"  (Note: Only sequences in boxes.npz can be visualized)\n")

        # Select sequence
        if args.sequence is None:
            sequence_name = random.choice(sequences)
            print(f"📌 Randomly selected sequence: {sequence_name}")
        else:
            sequence_name = args.sequence
            if sequence_name not in sequences:
                print(f"\n❌ Error: Sequence '{sequence_name}' not found in boxes.npz")
                print(f"Available sequences: {', '.join(sequences)}")
                sys.exit(1)
            print(f"📌 Using specified sequence: {sequence_name}")

        bboxes = boxes_dict[sequence_name]

        # Select frame
        if args.frame is None:
            frame_idx = random.randint(0, bboxes.num_frames - 1)
            print(f"📌 Randomly selected frame: {frame_idx} (out of {bboxes.num_frames})")
        else:
            frame_idx = args.frame
            if frame_idx < 0 or frame_idx >= bboxes.num_frames:
                print(f"\n❌ Error: Frame index {frame_idx} out of range [0, {bboxes.num_frames-1}]")
                sys.exit(1)
            print(f"📌 Using specified frame: {frame_idx} (out of {bboxes.num_frames})")

        # Create ImageMetadata
        print(f"\nLoading image...")
        frame_meta = ImageMetadata(
            sequence_name=sequence_name,
            frame_idx=frame_idx,
            bboxes=bboxes
        )
        image = frame_meta.load_image(data_dir / "images")
        print(f"✓ Image loaded: {image.shape[1]}x{image.shape[0]} pixels")

        # Visualize bounding boxes
        print(f"\nDrawing bounding boxes...")
        img_with_bboxes = frame_meta.visualize_bboxes()

        # Save or display
        if args.output:
            output_path = Path(args.output)
            cv2.imwrite(str(output_path), img_with_bboxes)
            print(f"✓ Visualization saved to: {output_path}")
        else:
            # Display in window
            window_name = f"Bounding Boxes - {sequence_name} - Frame {frame_idx}"
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
        import traceback
        traceback.print_exc()
        sys.exit(1)

    return 0


if __name__ == "__main__":
    sys.exit(main())
