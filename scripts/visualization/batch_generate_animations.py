"""
Batch generate tracking and pose animations for all training sequences.

This script processes all available SMPL pose sequences and generates:
1. Pitch tracking animations (top-down view)
2. 3D pose animations (skeletal view)

Output is saved to:
- data/tracking_animations/ - Pitch tracking MP4 files
- data/poses_animations/ - 3D pose MP4 files

Usage:
    # Generate all animations
    python scripts/visualization/batch_generate_animations.py

    # Generate with custom settings
    python scripts/visualization/batch_generate_animations.py --max-frames 200 --num-subjects 10

    # Generate only tracking animations
    python scripts/visualization/batch_generate_animations.py --tracking-only

    # Generate only pose animations
    python scripts/visualization/batch_generate_animations.py --poses-only

    # Test with first N sequences
    python scripts/visualization/batch_generate_animations.py --limit 5
"""

import sys
import time
import json
from pathlib import Path
from datetime import datetime
import traceback

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

from classes import PosesData


def generate_tracking_animation(poses: PosesData, output_path: Path,
                                max_frames: int = None, num_subjects: int = None,
                                fps: float = 25.0) -> dict:
    """
    Generate pitch tracking animation for a sequence.

    Args:
        poses: PosesData instance
        output_path: Output file path
        max_frames: Maximum frames to animate (None = all)
        num_subjects: Number of subjects to show (None = all)
        fps: Animation FPS

    Returns:
        Dictionary with generation statistics
    """
    import matplotlib.pyplot as plt

    start_time = time.time()

    try:
        # Determine frame range
        end_frame = min(max_frames - 1, poses.num_frames - 1) if max_frames else poses.num_frames - 1

        # Generate animation
        fig = poses.animate_pitch_tracking(
            start_frame=0,
            end_frame=end_frame,
            frame_step=1,
            figsize=(14, 10),
            num_subjects=num_subjects,
            fps=fps,
            trail_length=50,
            show_pitch=True
        )

        # Save animation
        fig._animation.save(
            str(output_path),
            writer='ffmpeg',
            fps=fps,
            extra_args=['-r', str(fps), '-vcodec', 'libx264', '-pix_fmt', 'yuv420p']
        )

        plt.close(fig)

        duration = time.time() - start_time

        return {
            'success': True,
            'frames': end_frame + 1,
            'duration': duration,
            'file_size': output_path.stat().st_size if output_path.exists() else 0
        }

    except Exception as e:
        duration = time.time() - start_time
        return {
            'success': False,
            'error': str(e),
            'duration': duration
        }


def generate_pose_animation(poses: PosesData, output_path: Path,
                            max_frames: int = None, num_subjects: int = None,
                            fps: float = 25.0) -> dict:
    """
    Generate 3D pose animation for a sequence.

    Args:
        poses: PosesData instance
        output_path: Output file path
        max_frames: Maximum frames to animate (None = all)
        num_subjects: Number of subjects to show (None = all)
        fps: Animation FPS

    Returns:
        Dictionary with generation statistics
    """
    import matplotlib.pyplot as plt

    start_time = time.time()

    try:
        # Determine frame range
        end_frame = min(max_frames - 1, poses.num_frames - 1) if max_frames else poses.num_frames - 1

        # Generate animation
        fig = poses.animate_3d_poses(
            start_frame=0,
            end_frame=end_frame,
            frame_step=1,
            figsize=(12, 8),
            elev=45,
            azim=45,
            num_subjects=num_subjects,
            fps=fps
        )

        # Save animation
        fig._animation.save(
            str(output_path),
            writer='ffmpeg',
            fps=fps,
            extra_args=['-r', str(fps), '-vcodec', 'libx264', '-pix_fmt', 'yuv420p']
        )

        plt.close(fig)

        duration = time.time() - start_time

        return {
            'success': True,
            'frames': end_frame + 1,
            'duration': duration,
            'file_size': output_path.stat().st_size if output_path.exists() else 0
        }

    except Exception as e:
        duration = time.time() - start_time
        return {
            'success': False,
            'error': str(e),
            'duration': duration
        }


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

    parser = argparse.ArgumentParser(
        description="Batch generate tracking and pose animations for all training sequences"
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
        default=10,
        help='Number of subjects to visualize (default: 10, None = all)'
    )
    parser.add_argument(
        '--fps',
        type=float,
        default=25.0,
        help='Animation FPS (default: 25.0)'
    )
    parser.add_argument(
        '--tracking-only',
        action='store_true',
        help='Generate only tracking animations'
    )
    parser.add_argument(
        '--poses-only',
        action='store_true',
        help='Generate only pose animations'
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
    tracking_dir = data_dir / 'tracking_animations'
    poses_anim_dir = data_dir / 'poses_animations'

    # Create output directories
    tracking_dir.mkdir(parents=True, exist_ok=True)
    poses_anim_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "="*80)
    print("BATCH ANIMATION GENERATION")
    print("="*80 + "\n")

    print(f"Configuration:")
    print(f"  Max frames: {args.max_frames}")
    print(f"  Subjects: {args.num_subjects if args.num_subjects else 'all'}")
    print(f"  FPS: {args.fps}")
    print(f"  Generate tracking: {not args.poses_only}")
    print(f"  Generate poses: {not args.tracking_only}")
    print(f"  Skip existing: {args.skip_existing}")
    if args.limit:
        print(f"  Limit: First {args.limit} sequences")
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
        return 1

    # Processing statistics
    stats = {
        'total': len(sequences),
        'tracking_success': 0,
        'tracking_failed': 0,
        'tracking_skipped': 0,
        'poses_success': 0,
        'poses_failed': 0,
        'poses_skipped': 0,
        'total_time': 0,
        'total_size': 0,
        'results': {}
    }

    start_time = time.time()

    # Process each sequence
    for idx, seq_name in enumerate(sequences, 1):
        print(f"\n{'='*80}")
        print(f"[{idx}/{len(sequences)}] Processing: {seq_name}")
        print(f"{'='*80}")

        poses = poses_dict[seq_name]
        print(f"  Frames: {poses.num_frames}, Subjects: {poses.num_subjects}")

        seq_stats = {
            'frames': poses.num_frames,
            'subjects': poses.num_subjects,
            'tracking': None,
            'poses': None
        }

        # Generate tracking animation
        if not args.poses_only:
            tracking_path = tracking_dir / f"{seq_name}_tracking.mp4"

            if args.skip_existing and tracking_path.exists():
                print(f"  ⏭️  Tracking: Skipped (already exists)")
                stats['tracking_skipped'] += 1
                seq_stats['tracking'] = {'skipped': True}
            else:
                print(f"  🎬 Generating tracking animation...")
                result = generate_tracking_animation(
                    poses, tracking_path,
                    max_frames=args.max_frames,
                    num_subjects=args.num_subjects,
                    fps=args.fps
                )

                if result['success']:
                    print(f"     ✓ Success: {result['frames']} frames, "
                          f"{format_size(result['file_size'])}, "
                          f"{format_time(result['duration'])}")
                    stats['tracking_success'] += 1
                    stats['total_size'] += result['file_size']
                else:
                    print(f"     ❌ Failed: {result['error']}")
                    stats['tracking_failed'] += 1

                seq_stats['tracking'] = result

        # Generate pose animation
        if not args.tracking_only:
            pose_path = poses_anim_dir / f"{seq_name}_poses.mp4"

            if args.skip_existing and pose_path.exists():
                print(f"  ⏭️  Poses: Skipped (already exists)")
                stats['poses_skipped'] += 1
                seq_stats['poses'] = {'skipped': True}
            else:
                print(f"  🎬 Generating pose animation...")
                result = generate_pose_animation(
                    poses, pose_path,
                    max_frames=args.max_frames,
                    num_subjects=args.num_subjects,
                    fps=args.fps
                )

                if result['success']:
                    print(f"     ✓ Success: {result['frames']} frames, "
                          f"{format_size(result['file_size'])}, "
                          f"{format_time(result['duration'])}")
                    stats['poses_success'] += 1
                    stats['total_size'] += result['file_size']
                else:
                    print(f"     ❌ Failed: {result['error']}")
                    stats['poses_failed'] += 1

                seq_stats['poses'] = result

        stats['results'][seq_name] = seq_stats

        # Show progress
        elapsed = time.time() - start_time
        avg_time = elapsed / idx
        remaining = avg_time * (len(sequences) - idx)
        print(f"\n  Progress: {idx}/{len(sequences)} ({idx/len(sequences)*100:.1f}%)")
        print(f"  Elapsed: {format_time(elapsed)}, Est. remaining: {format_time(remaining)}")

    # Final statistics
    total_time = time.time() - start_time
    stats['total_time'] = total_time

    print(f"\n{'='*80}")
    print("BATCH PROCESSING COMPLETE")
    print(f"{'='*80}\n")

    print(f"Summary:")
    print(f"  Total sequences: {stats['total']}")
    print(f"  Total time: {format_time(total_time)}")
    print(f"  Total size: {format_size(stats['total_size'])}")
    print()

    if not args.poses_only:
        print(f"Tracking animations:")
        print(f"  ✓ Success: {stats['tracking_success']}")
        print(f"  ❌ Failed: {stats['tracking_failed']}")
        print(f"  ⏭️  Skipped: {stats['tracking_skipped']}")
        print()

    if not args.tracking_only:
        print(f"Pose animations:")
        print(f"  ✓ Success: {stats['poses_success']}")
        print(f"  ❌ Failed: {stats['poses_failed']}")
        print(f"  ⏭️  Skipped: {stats['poses_skipped']}")
        print()

    # Save report
    report_path = data_dir / f"animation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(report_path, 'w') as f:
        json.dump(stats, f, indent=2, default=str)

    print(f"📊 Detailed report saved to: {report_path}")

    # List failed sequences if any
    failed_sequences = []
    for seq_name, seq_stats in stats['results'].items():
        if seq_stats.get('tracking') and not seq_stats['tracking'].get('success', True):
            failed_sequences.append((seq_name, 'tracking', seq_stats['tracking'].get('error')))
        if seq_stats.get('poses') and not seq_stats['poses'].get('success', True):
            failed_sequences.append((seq_name, 'poses', seq_stats['poses'].get('error')))

    if failed_sequences:
        print(f"\n⚠️  Failed sequences ({len(failed_sequences)}):")
        for seq_name, anim_type, error in failed_sequences:
            print(f"  - {seq_name} ({anim_type}): {error}")

    print(f"\n{'='*80}\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
