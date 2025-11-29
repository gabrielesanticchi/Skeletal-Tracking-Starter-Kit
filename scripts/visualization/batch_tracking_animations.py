"""
Batch generate pitch tracking animations for all training sequences.

This script processes all available SMPL pose sequences and generates
pitch tracking animations (top-down view) to validate the coordinate
transformation across the entire dataset.

Output is saved to: data/tracking_animations/

Usage:
    # Generate all tracking animations (300 frames max per sequence)
    python scripts/visualization/batch_tracking_animations.py

    # Custom max frames and subjects
    python scripts/visualization/batch_tracking_animations.py --max-frames 500 --num-subjects 15

    # Test with first 5 sequences
    python scripts/visualization/batch_tracking_animations.py --limit 5

    # Skip existing files
    python scripts/visualization/batch_tracking_animations.py --skip-existing
"""

import sys
import time
import json
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

from classes import PosesData


def format_size(size_bytes: int) -> str:
    """Format file size in human-readable format."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f}{unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.1f}TB"


def format_time(seconds: float) -> str:
    """Format duration in human-readable format."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        secs = seconds % 60
        return f"{minutes}m {secs:.0f}s"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        return f"{hours}h {minutes}m"


def main():
    """Main batch processing function."""
    import argparse
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt

    parser = argparse.ArgumentParser(
        description="Batch generate pitch tracking animations for all training sequences"
    )
    parser.add_argument(
        '--max-frames',
        type=int,
        default=300,
        help='Maximum frames per animation (default: 300, ~12s at 25fps)'
    )
    parser.add_argument(
        '--num-subjects',
        type=int,
        default=None,
        help='Number of subjects to visualize (default: None = all)'
    )
    parser.add_argument(
        '--fps',
        type=float,
        default=25.0,
        help='Animation FPS (default: 25.0)'
    )
    parser.add_argument(
        '--trail-length',
        type=int,
        default=50,
        help='Trail length in frames (default: 50)'
    )
    parser.add_argument(
        '--limit',
        type=int,
        default=None,
        help='Limit to first N sequences (for testing)'
    )
    parser.add_argument(
        '--skip-existing',
        action='store_true',
        help='Skip sequences that already have animations'
    )

    args = parser.parse_args()

    # Setup paths
    data_dir = Path('data')
    poses_dir = data_dir / 'poses'
    output_dir = data_dir / 'tracking_animations'

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "="*80)
    print("BATCH PITCH TRACKING ANIMATION GENERATION")
    print("="*80 + "\n")

    print(f"Configuration:")
    print(f"  Max frames: {args.max_frames}")
    print(f"  Subjects: {args.num_subjects if args.num_subjects else 'all'}")
    print(f"  FPS: {args.fps}")
    print(f"  Trail length: {args.trail_length}")
    print(f"  Skip existing: {args.skip_existing}")
    if args.limit:
        print(f"  Limit: First {args.limit} sequences")
    print(f"  Output: {output_dir}")
    print()

    # Load all sequences
    print("Loading SMPL poses data...")
    if not poses_dir.exists():
        print(f"❌ Error: Poses directory not found: {poses_dir}")
        return 1

    try:
        poses_dict = PosesData.load_all(poses_dir)
        sequences = sorted(poses_dict.keys())

        # Apply limit if specified
        if args.limit:
            sequences = sequences[:args.limit]

        print(f"✓ Loaded {len(sequences)} sequences\n")
    except Exception as e:
        print(f"❌ Error loading poses: {e}")
        import traceback
        traceback.print_exc()
        return 1

    # Processing statistics
    stats = {
        'total': len(sequences),
        'success': 0,
        'failed': 0,
        'skipped': 0,
        'total_time': 0,
        'total_size': 0,
        'results': {}
    }

    start_time = time.time()

    # Process each sequence
    for idx, seq_name in enumerate(sequences, 1):
        print(f"[{idx}/{len(sequences)}] {seq_name}", end=' ... ')
        sys.stdout.flush()

        poses = poses_dict[seq_name]
        output_path = output_dir / f"{seq_name}_tracking.mp4"

        seq_stats = {
            'frames': poses.num_frames,
            'subjects': poses.num_subjects,
        }

        # Check if already exists
        if args.skip_existing and output_path.exists():
            print(f"⏭️  SKIPPED (exists)")
            stats['skipped'] += 1
            seq_stats['skipped'] = True
            stats['results'][seq_name] = seq_stats
            continue

        # Generate animation
        gen_start = time.time()
        try:
            # Determine frame range
            end_frame = min(args.max_frames - 1, poses.num_frames - 1) if args.max_frames else poses.num_frames - 1

            # Generate animation
            fig = poses.animate_pitch_tracking(
                start_frame=0,
                end_frame=end_frame,
                frame_step=1,
                figsize=(14, 10),
                num_subjects=args.num_subjects,
                fps=args.fps,
                trail_length=args.trail_length,
                show_pitch=True
            )

            # Save animation
            fig._animation.save(
                str(output_path),
                writer='ffmpeg',
                fps=args.fps,
                extra_args=['-r', str(args.fps), '-vcodec', 'libx264', '-pix_fmt', 'yuv420p']
            )

            plt.close(fig)

            # Get file size
            file_size = output_path.stat().st_size if output_path.exists() else 0
            duration = time.time() - gen_start

            print(f"✓ {end_frame + 1}f, {format_size(file_size)}, {format_time(duration)}")

            stats['success'] += 1
            stats['total_size'] += file_size
            seq_stats['success'] = True
            seq_stats['animated_frames'] = end_frame + 1
            seq_stats['file_size'] = file_size
            seq_stats['duration'] = duration

        except Exception as e:
            duration = time.time() - gen_start
            print(f"❌ FAILED: {str(e)[:50]}")
            stats['failed'] += 1
            seq_stats['success'] = False
            seq_stats['error'] = str(e)
            seq_stats['duration'] = duration

        stats['results'][seq_name] = seq_stats

    # Final statistics
    total_time = time.time() - start_time
    stats['total_time'] = total_time

    print(f"\n{'='*80}")
    print("BATCH PROCESSING COMPLETE")
    print(f"{'='*80}\n")

    print(f"Summary:")
    print(f"  Total sequences: {stats['total']}")
    print(f"  ✓ Success: {stats['success']}")
    print(f"  ❌ Failed: {stats['failed']}")
    print(f"  ⏭️  Skipped: {stats['skipped']}")
    print(f"  Total time: {format_time(total_time)}")
    print(f"  Total size: {format_size(stats['total_size'])}")
    if stats['success'] > 0:
        print(f"  Avg time/seq: {format_time(total_time / stats['success'])}")
    print()

    # Save report
    report_path = output_dir / f"generation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(report_path, 'w') as f:
        json.dump(stats, f, indent=2, default=str)

    print(f"📊 Detailed report saved to: {report_path}")

    # List failed sequences if any
    if stats['failed'] > 0:
        print(f"\n⚠️  Failed sequences ({stats['failed']}):")
        for seq_name, seq_stats in stats['results'].items():
            if not seq_stats.get('success', True) and not seq_stats.get('skipped', False):
                error = seq_stats.get('error', 'Unknown error')
                print(f"  - {seq_name}: {error[:70]}")

    print(f"\n{'='*80}\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
