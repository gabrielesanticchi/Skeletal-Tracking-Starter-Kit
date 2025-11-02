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
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

from classes import BBoxesData, ImageMetadata
from utils import ArgsParser


def main():
    """Main visualization function."""
    parser = ArgsParser.create_base_parser(
        "Visualize bounding boxes on images from FIFA Skeletal Tracking Challenge"
    )
    args = parser.parse_args()

    # Get data directory
    data_dir = ArgsParser.get_data_dir(args)

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
        sequence_name = args.sequence or random.choice(sequences)
        if sequence_name not in sequences:
            print(f"\n❌ Error: Sequence '{sequence_name}' not found in boxes.npz")
            print(f"Available sequences: {', '.join(sequences)}")
            sys.exit(1)
        print(f"📌 {'Randomly selected' if not args.sequence else 'Using'} sequence: {sequence_name}")

        bboxes = boxes_dict[sequence_name]

        # Select frame
        frame_idx = args.frame if args.frame is not None else random.randint(0, bboxes.num_frames - 1)
        if frame_idx < 0 or frame_idx >= bboxes.num_frames:
            print(f"\n❌ Error: Frame index {frame_idx} out of range [0, {bboxes.num_frames-1}]")
            sys.exit(1)
        print(f"📌 {'Randomly selected' if args.frame is None else 'Using'} frame: {frame_idx} (out of {bboxes.num_frames})")

        # Create ImageMetadata
        print(f"\nLoading image...")
        frame_meta = ImageMetadata(sequence_name=sequence_name, frame_idx=frame_idx, bboxes=bboxes)
        image = frame_meta.load_image(data_dir / "images")
        print(f"✓ Image loaded: {image.shape[1]}x{image.shape[0]} pixels")

        # Visualize bounding boxes
        print(f"\nDrawing bounding boxes...")
        img_with_bboxes = frame_meta.visualize_bboxes()

        # Save or display
        output_path = args.output
        if output_path:
            cv2.imwrite(output_path, img_with_bboxes)
            print(f"✓ Visualization saved to: {output_path}")
        else:
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
