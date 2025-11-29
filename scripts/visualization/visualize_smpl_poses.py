"""
Visualize SMPL poses in 3D world coordinates from training data.

This script visualizes SMPL poses from the training sequences, showing the 3D skeletal
structure in world coordinates. Unlike the processed 2D/3D poses, this uses the raw
SMPL parameters to show poses in their original world coordinate system.

Usage:
    # Random sequence and frame
    python scripts/visualization/visualize_smpl_poses.py

    # Specific sequence, random frame
    python scripts/visualization/visualize_smpl_poses.py --sequence ARG_CRO_220001

    # Specific sequence and frame
    python scripts/visualization/visualize_smpl_poses.py --sequence ARG_CRO_220001 --frame 100

    # Save output
    python scripts/visualization/visualize_smpl_poses.py --output smpl_poses_3d.png

    # Show joint labels and limit subjects
    python scripts/visualization/visualize_smpl_poses.py --sequence ARG_CRO_220001 --frame 100 --show-labels --num-subjects 3

    # Custom view angles
    python scripts/visualization/visualize_smpl_poses.py --elev 30 --azim -45

    # Pitch tracking view (top-down)
    python scripts/visualization/visualize_smpl_poses.py --sequence ARG_CRO_220001 --pitch-view --start-frame 0 --end-frame 100

    # Enhanced Pitch Tracking with football pitch outline (default)
    python scripts/visualization/visualize_smpl_poses.py --sequence ARG_CRO_220001 --pitch-view

    # Animated pitch tracking with full sequence (default)
    python scripts/visualization/visualize_smpl_poses.py --sequence ARG_CRO_220001 --pitch-view --animate-pitch

    # Animated pitch tracking with custom settings
    python scripts/visualization/visualize_smpl_poses.py --sequence ARG_CRO_220001 --pitch-view --animate-pitch --start-frame 0 --end-frame 200 --fps 25 --trail-length 40

    # Disable pitch outline
    python scripts/visualization/visualize_smpl_poses.py --sequence ARG_CRO_220001 --pitch-view --no-pitch

    # Animated tracking without pitch outline
    python scripts/visualization/visualize_smpl_poses.py --sequence ARG_CRO_220001 --pitch-view --animate-pitch --no-pitch
"""

import sys
import random
import matplotlib.pyplot as plt
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

from classes import PosesData
from utils import ArgsParser


def main():
    """Main visualization function."""
    # Create parser with base and 3D viz arguments
    parser = ArgsParser.create_base_parser(
        "Visualize SMPL poses in 3D world coordinates from FIFA training data"
    )
    parser = ArgsParser.add_3d_viz_args(parser)
    
    # Add SMPL-specific arguments
    parser.add_argument(
        '--show-labels',
        action='store_true',
        default=False,
        help='Show subject labels (default: False)'
    )
    
    parser.add_argument(
        '--pitch-view',
        action='store_true',
        default=False,
        help='Show pitch tracking view (top-down) instead of 3D poses'
    )
    
    parser.add_argument(
        '--animate-pitch',
        action='store_true',
        default=False,
        help='Create animated pitch tracking instead of static view'
    )
    
    parser.add_argument(
        '--trail-length',
        type=int,
        default=50,
        help='Length of movement trail for animated pitch tracking (default: 50)'
    )
    
    parser.add_argument(
        '--fps',
        type=float,
        default=50.0,
        help='Frames per second for animated pitch tracking (default: 50.0 for video sync)'
    )
    
    parser.add_argument(
        '--duration',
        type=float,
        default=None,
        help='Total duration in seconds for animated pitch tracking (overrides fps)'
    )
    
    parser.add_argument(
        '--start-frame',
        type=int,
        default=0,
        help='Start frame for pitch tracking view (default: 0)'
    )
    
    parser.add_argument(
        '--end-frame',
        type=int,
        default=None,
        help='End frame for pitch tracking view (default: last frame - full sequence)'
    )
    
    parser.add_argument(
        '--show-pitch',
        action='store_true',
        default=True,
        help='Show football pitch outline (default: True)'
    )
    
    parser.add_argument(
        '--no-pitch',
        action='store_true',
        default=False,
        help='Hide football pitch outline'
    )
    
    parser.add_argument(
        '--frame-step',
        type=int,
        default=1,
        help='Frame step for pitch tracking view (default: 1)'
    )
    
    args = parser.parse_args()

    # Get data directory
    data_dir = ArgsParser.get_data_dir(args)

    print("\n" + "="*80)
    print("SMPL POSES VISUALIZATION")
    print("="*80 + "\n")

    try:
        # Load SMPL poses data
        print("Loading SMPL poses data...")
        poses_dir = data_dir / "poses"
        if not poses_dir.exists():
            print(f"❌ Error: Poses directory not found: {poses_dir}")
            print("💡 SMPL poses are only available for training sequences")
            sys.exit(1)
        
        poses_dict = PosesData.load_all(poses_dir)
        sequences = list(poses_dict.keys())
        print(f"✓ Loaded {len(sequences)} training sequences with SMPL poses\n")

        # Select sequence
        sequence_name = args.sequence or random.choice(sequences)
        if sequence_name not in sequences:
            print(f"\n❌ Error: Sequence '{sequence_name}' not found in poses directory")
            print(f"Available sequences: {', '.join(sequences[:10])}{'...' if len(sequences) > 10 else ''}")
            sys.exit(1)
        print(f"📌 {'Randomly selected' if not args.sequence else 'Using'} sequence: {sequence_name}")

        poses = poses_dict[sequence_name]
        print(f"📊 Sequence info: {poses.num_frames} frames, {poses.num_subjects} subjects")

        if args.pitch_view:
            # Pitch tracking visualization
            start_frame = args.start_frame
            end_frame = args.end_frame if args.end_frame is not None else poses.num_frames - 1
            
            if start_frame < 0 or start_frame >= poses.num_frames:
                print(f"\n❌ Error: Start frame {start_frame} out of range [0, {poses.num_frames-1}]")
                sys.exit(1)
            
            if end_frame < 0 or end_frame >= poses.num_frames:
                print(f"\n❌ Error: End frame {end_frame} out of range [0, {poses.num_frames-1}]")
                sys.exit(1)
                
            if start_frame >= end_frame:
                print(f"\n❌ Error: Start frame ({start_frame}) must be less than end frame ({end_frame})")
                sys.exit(1)

            print(f"📌 Pitch tracking range: frames {start_frame} to {end_frame} (step: {args.frame_step})")
            
            if args.num_subjects is not None:
                print(f"📌 Limiting to {args.num_subjects} subjects")

            if args.animate_pitch:
                # Generate animated pitch tracking
                print(f"\nGenerating animated pitch tracking...")
                if args.duration:
                    print(f"📌 Animation duration: {args.duration:.1f}s")
                else:
                    total_frames = len(range(start_frame, end_frame + 1, args.frame_step))
                    duration = total_frames / args.fps
                    print(f"📌 Animation settings: {args.fps} fps, ~{duration:.1f}s duration")
                
                print(f"📌 Trail length: {args.trail_length} positions")
                
                fig = poses.animate_pitch_tracking(
                    start_frame=start_frame,
                    end_frame=end_frame,
                    frame_step=args.frame_step,
                    figsize=tuple(args.figsize),
                    num_subjects=args.num_subjects,
                    fps=args.fps,
                    duration=args.duration,
                    trail_length=args.trail_length,
                    show_pitch=args.show_pitch and not args.no_pitch
                )
            else:
                # Generate static pitch tracking visualization
                print(f"\nGenerating static pitch tracking visualization...")
                fig = poses.visualize_pitch_tracking(
                    start_frame=start_frame,
                    end_frame=end_frame,
                    frame_step=args.frame_step,
                    figsize=tuple(args.figsize),
                    num_subjects=args.num_subjects,
                    show_pitch=args.show_pitch and not args.no_pitch
                )
        else:
            # 3D poses visualization
            frame_idx = args.frame if args.frame is not None else random.randint(0, poses.num_frames - 1)
            if frame_idx < 0 or frame_idx >= poses.num_frames:
                print(f"\n❌ Error: Frame index {frame_idx} out of range [0, {poses.num_frames-1}]")
                sys.exit(1)
            print(f"📌 {'Randomly selected' if args.frame is None else 'Using'} frame: {frame_idx} (out of {poses.num_frames})")

            if args.num_subjects is not None:
                print(f"📌 Limiting to {args.num_subjects} subjects")

            # Generate 3D visualization
            print(f"\nGenerating 3D poses visualization...")
            fig = poses.visualize_3d_poses(
                frame_idx=frame_idx,
                figsize=tuple(args.figsize),
                elev=args.elev,
                azim=args.azim,
                num_subjects=args.num_subjects,
                show_labels=args.show_labels
            )

        # Save or display - default to MP4 output for animations
        output_path = args.output
        if output_path is None and args.animate_pitch:
            # Default MP4 output for animations
            output_path = f"{sequence_name}_pitch_tracking.mp4"
        
        if output_path:
            if args.animate_pitch:
                # Handle animation saving
                print(f"\n💾 Saving animation to: {output_path}")
                print("⏳ This may take a while...")
                
                # Determine writer based on file extension
                if output_path.lower().endswith('.gif'):
                    writer = 'pillow'
                    writer_fps = args.fps
                    print(f"📝 Using GIF format at {writer_fps} fps")
                elif output_path.lower().endswith('.mp4'):
                    writer = 'ffmpeg'
                    writer_fps = args.fps
                    print(f"📝 Using MP4 format at {writer_fps} fps (requires ffmpeg)")
                else:
                    writer = 'ffmpeg'
                    writer_fps = args.fps
                    output_path = output_path + '.mp4' if not output_path.lower().endswith(('.mp4', '.gif')) else output_path
                    print(f"📝 Using default MP4 format at {writer_fps} fps")
                
                try:
                    # For MP4, use additional parameters to ensure precise FPS
                    if writer == 'ffmpeg':
                        extra_args = ['-r', str(writer_fps), '-vcodec', 'libx264', '-pix_fmt', 'yuv420p']
                        fig._animation.save(output_path, writer=writer, fps=writer_fps, extra_args=extra_args)
                    else:
                        fig._animation.save(output_path, writer=writer, fps=writer_fps)
                    
                    print(f"✓ Animation saved successfully at {writer_fps} fps!")
                    print(f"📁 File: {output_path}")
                except Exception as e:
                    print(f"❌ Error saving animation: {e}")
                    print("💡 Try installing ffmpeg for MP4 or use .gif extension")
                    sys.exit(1)
                
                plt.close(fig)
            else:
                # Static image saving
                fig.savefig(output_path, dpi=150, bbox_inches='tight')
                print(f"✓ Visualization saved to: {output_path}")
                plt.close(fig)
        else:
            print(f"\n✓ Displaying visualization")
            if args.animate_pitch:
                print("⚠️  Note: Interactive display FPS may not be exact")
                print("💡 For precise FPS control, save as MP4: --output animation.mp4")
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