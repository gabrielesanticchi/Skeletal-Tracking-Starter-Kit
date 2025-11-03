"""
Visualize 3D skeletal poses from the FIFA Skeletal Tracking Challenge dataset.

This script visualizes 3D skeletons with color-coded joints in 3D space.
If no arguments are provided, a random sequence and frame are selected.

Usage:
    # Random sequence and frame
    python scripts/visualization/visualize_3d_pose.py

    # Specific sequence, random frame
    python scripts/visualization/visualize_3d_pose.py --sequence ARG_FRA_183303

    # Specific sequence and frame
    python scripts/visualization/visualize_3d_pose.py --sequence ARG_FRA_183303 --frame 100

    # Save output
    python scripts/visualization/visualize_3d_pose.py --output pose_3d.png

    # Show joint labels
    python scripts/visualization/visualize_3d_pose.py --sequence ARG_FRA_183303 --frame 100 --show-labels

    # Custom view angles
    python scripts/visualization/visualize_3d_pose.py --elev 30 --azim -45

    # Limit to first 2 subjects
    python scripts/visualization/visualize_3d_pose.py --sequence ARG_FRA_183303 --frame 100 --num-subjects 2
"""

import sys
import random
import matplotlib.pyplot as plt
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

from classes import Skeleton3DData
from utils import ArgsParser


def main():
    """Main visualization function."""
    # Create parser with base and 3D viz arguments
    parser = ArgsParser.create_base_parser(
        "Visualize 3D skeletal poses from FIFA Skeletal Tracking Challenge"
    )
    parser = ArgsParser.add_3d_viz_args(parser)
    parser.add_argument(
        '--show-labels',
        action='store_true',
        default=False,
        help='Show joint labels (default: False)'
    )
    args = parser.parse_args()

    # Get data directory
    data_dir = ArgsParser.get_data_dir(args)

    print("\n" + "="*80)
    print("3D POSE VISUALIZATION")
    print("="*80 + "\n")

    try:
        # Load 3D skeleton data
        print("Loading 3D skeleton data...")
        skel_3d_path = data_dir / "skel_3d.npz"
        skel_3d_dict = Skeleton3DData.load_all(skel_3d_path)
        sequences = list(skel_3d_dict.keys())
        print(f"✓ Loaded {len(sequences)} sequences with 3D poses\n")

        # Select sequence
        sequence_name = args.sequence or random.choice(sequences)
        if sequence_name not in sequences:
            print(f"\n❌ Error: Sequence '{sequence_name}' not found in skel_3d.npz")
            print(f"Available sequences: {', '.join(sequences)}")
            sys.exit(1)
        print(f"📌 {'Randomly selected' if not args.sequence else 'Using'} sequence: {sequence_name}")

        skel_3d = skel_3d_dict[sequence_name]

        # Select frame
        frame_idx = args.frame if args.frame is not None else random.randint(0, skel_3d.num_frames - 1)
        if frame_idx < 0 or frame_idx >= skel_3d.num_frames:
            print(f"\n❌ Error: Frame index {frame_idx} out of range [0, {skel_3d.num_frames-1}]")
            sys.exit(1)
        print(f"📌 {'Randomly selected' if args.frame is None else 'Using'} frame: {frame_idx} (out of {skel_3d.num_frames})")

        # Visualize 3D pose
        print(f"\nGenerating 3D visualization...")
        if args.num_subjects is not None:
            print(f"📌 Limiting to {args.num_subjects} subjects")
        fig = skel_3d.visualize_3d(
            frame_idx,
            figsize=tuple(args.figsize),
            elev=args.elev,
            azim=args.azim,
            show_labels=args.show_labels,
            num_subjects=args.num_subjects
        )

        # Save or display
        output_path = args.output
        if output_path:
            fig.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"✓ Visualization saved to: {output_path}")
            plt.close(fig)
        else:
            print(f"\n✓ Displaying visualization")
            print("Close the window to exit...")
            plt.show()

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
