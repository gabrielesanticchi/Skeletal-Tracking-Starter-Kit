"""
Visualize 2D skeletal poses overlaid on images.

This script visualizes 2D skeletons with color-coded joints overlaid on images.
If no arguments are provided, a random sequence and frame are selected.

Usage:
    # Random sequence and frame
    python scripts/visualization/visualize_2d_pose.py

    # Specific sequence, random frame
    python scripts/visualization/visualize_2d_pose.py --sequence ARG_FRA_183303

    # Specific sequence and frame
    python scripts/visualization/visualize_2d_pose.py --sequence ARG_FRA_183303 --frame 100

    # Save output
    python scripts/visualization/visualize_2d_pose.py --output pose_2d.jpg

    # Show joint labels
    python scripts/visualization/visualize_2d_pose.py --sequence ARG_FRA_183303 --frame 100 --show-labels

    # Limit to first 2 subjects
    python scripts/visualization/visualize_2d_pose.py --sequence ARG_FRA_183303 --frame 100 --num-subjects 2
"""

import sys
import random
import cv2
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

from classes import Skeleton2DData, ImageMetadata
from utils import ArgsParser


def main():
    """Main visualization function."""
    # Create parser with base and 2D viz arguments
    parser = ArgsParser.create_base_parser(
        "Visualize 2D skeletal poses overlaid on images from FIFA Skeletal Tracking Challenge"
    )
    parser = ArgsParser.add_2d_viz_args(parser)
    args = parser.parse_args()

    # Get data directory
    data_dir = ArgsParser.get_data_dir(args)

    print("\n" + "="*80)
    print("2D SKELETON VISUALIZATION")
    print("="*80 + "\n")

    try:
        # Load 2D skeleton data
        print("Loading 2D skeleton data...")
        skel_2d_path = data_dir / "skel_2d.npz"
        skel_2d_dict = Skeleton2DData.load_all(skel_2d_path)
        sequences = list(skel_2d_dict.keys())
        print(f"✓ Loaded {len(sequences)} sequences with 2D poses\n")

        # Select sequence
        sequence_name = args.sequence or random.choice(sequences)
        if sequence_name not in sequences:
            print(f"\n❌ Error: Sequence '{sequence_name}' not found in skel_2d.npz")
            print(f"Available sequences: {', '.join(sequences)}")
            sys.exit(1)
        print(f"📌 {'Randomly selected' if not args.sequence else 'Using'} sequence: {sequence_name}")

        skel_2d = skel_2d_dict[sequence_name]

        # Select frame
        frame_idx = args.frame if args.frame is not None else random.randint(0, skel_2d.num_frames - 1)
        if frame_idx < 0 or frame_idx >= skel_2d.num_frames:
            print(f"\n❌ Error: Frame index {frame_idx} out of range [0, {skel_2d.num_frames-1}]")
            sys.exit(1)
        print(f"📌 {'Randomly selected' if args.frame is None else 'Using'} frame: {frame_idx} (out of {skel_2d.num_frames})")

        # Load image
        print(f"\nLoading image...")
        frame_meta = ImageMetadata(sequence_name=sequence_name, frame_idx=frame_idx, skel_2d=skel_2d)
        image = frame_meta.load_image(data_dir / "images")
        print(f"✓ Image loaded: {image.shape[1]}x{image.shape[0]} pixels")

        # Visualize 2D skeleton
        print(f"\nGenerating 2D visualization...")
        if args.num_subjects is not None:
            print(f"📌 Limiting to {args.num_subjects} subjects")
        img_with_skeleton = skel_2d.visualize_frame(
            image,
            frame_idx,
            show_skeleton=args.show_skeleton,
            show_joints=args.show_joints,
            show_labels=args.show_labels,
            num_subjects=args.num_subjects
        )

        # Save or display
        output_path = args.output
        if output_path:
            cv2.imwrite(output_path, img_with_skeleton)
            print(f"✓ Visualization saved to: {output_path}")
        else:
            window_name = f"2D Skeleton - {sequence_name} - Frame {frame_idx}"
            cv2.imshow(window_name, img_with_skeleton)
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
