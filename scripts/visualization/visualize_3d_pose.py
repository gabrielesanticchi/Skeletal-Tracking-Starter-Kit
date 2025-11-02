"""
Visualize 3D skeletal poses from the FIFA Skeletal Tracking Challenge dataset.

This script allows you to visualize 3D poses in 3D space.
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
"""

import sys
import random
import argparse
import matplotlib.pyplot as plt
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

from classes import Skeleton3DData, VideoMetadata


def get_project_root() -> Path:
    """Get the project root directory."""
    current = Path(__file__).resolve()
    return current.parent.parent.parent


def main():
    """Main visualization function."""
    parser = argparse.ArgumentParser(
        description="Visualize 3D skeletal poses from FIFA Skeletal Tracking Challenge"
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
        help="Output file path to save visualization. If not specified, displays interactively."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Base data directory (default: auto-detect from project root)"
    )
    parser.add_argument(
        "--elev",
        type=int,
        default=20,
        help="Elevation angle for 3D plot (default: 20)"
    )
    parser.add_argument(
        "--azim",
        type=int,
        default=-60,
        help="Azimuth angle for 3D plot (default: -60)"
    )

    args = parser.parse_args()

    # Setup paths
    if args.data_dir:
        data_dir = args.data_dir
    else:
        project_root = get_project_root()
        data_dir = project_root / "data"

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
        if args.sequence is None:
            sequence_name = random.choice(sequences)
            print(f"📌 Randomly selected sequence: {sequence_name}")
        else:
            sequence_name = args.sequence
            if sequence_name not in sequences:
                print(f"\n❌ Error: Sequence '{sequence_name}' not found in skel_3d.npz")
                print(f"Available sequences: {', '.join(sequences)}")
                sys.exit(1)
            print(f"📌 Using specified sequence: {sequence_name}")

        skel_3d = skel_3d_dict[sequence_name]

        # Select frame
        if args.frame is None:
            frame_idx = random.randint(0, skel_3d.num_frames - 1)
            print(f"📌 Randomly selected frame: {frame_idx} (out of {skel_3d.num_frames})")
        else:
            frame_idx = args.frame
            if frame_idx < 0 or frame_idx >= skel_3d.num_frames:
                print(f"\n❌ Error: Frame index {frame_idx} out of range [0, {skel_3d.num_frames-1}]")
                sys.exit(1)
            print(f"📌 Using specified frame: {frame_idx} (out of {skel_3d.num_frames})")

        # Visualize 3D pose
        print(f"\nGenerating 3D visualization...")
        fig = skel_3d.visualize_3d(frame_idx, elev=args.elev, azim=args.azim)

        # Save or display
        if args.output:
            output_path = Path(args.output)
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
