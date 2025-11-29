"""
Animate SMPL poses in 3D world coordinates from training data.

This script creates animated 3D visualizations of SMPL poses from training sequences,
showing the skeletal movement in world coordinates over time. Unlike the processed
2D/3D poses, this uses the raw SMPL parameters to show poses in their original
world coordinate system.

Usage:
    # Random sequence, full animation
    python scripts/visualization/animate_smpl_poses.py

    # Specific sequence, full animation
    python scripts/visualization/animate_smpl_poses.py --sequence ARG_CRO_220001

    # Specific frame range
    python scripts/visualization/animate_smpl_poses.py --sequence ARG_CRO_220001 --start-frame 50 --end-frame 150

    # Save animation as MP4 with precise FPS (default: 50 fps for video sync)
    python scripts/visualization/animate_smpl_poses.py --output smpl_animation.mp4 --num-subjects 3

    # Custom view angles (default: elev=45, azim=45 for full pose visibility)
    python scripts/visualization/animate_smpl_poses.py --elev 30 --azim -45

    # Limit to first 2 subjects with custom duration
    python scripts/visualization/animate_smpl_poses.py --sequence ARG_CRO_220001 --num-subjects 2 --duration 5.0

    # Skip frames for faster animation
    python scripts/visualization/animate_smpl_poses.py --sequence ARG_CRO_220001 --frame-step 2 --fps 20

    # Default MP4 output with 50 fps for video synchronization
    python scripts/visualization/animate_smpl_poses.py --sequence ARG_CRO_220001

    # Enhanced joint animation with actual pose changes
    python scripts/visualization/animate_smpl_poses.py --sequence ARG_CRO_220001 --start-frame 50 --end-frame 150

    # Full sequence animation (uses all frames by default)
    python scripts/visualization/animate_smpl_poses.py --sequence ARG_CRO_220001 --frame-step 10
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
    """Main animation function."""
    # Create parser with base, 3D viz, and animation arguments
    parser = ArgsParser.create_base_parser(
        "Animate SMPL poses in 3D world coordinates from FIFA training data"
    )
    parser = ArgsParser.add_3d_viz_args(parser)
    parser = ArgsParser.add_animation_args(parser)
    
    # Add SMPL-specific arguments
    parser.add_argument(
        '--show-labels',
        action='store_true',
        default=False,
        help='Show subject labels (default: False)'
    )
    
    args = parser.parse_args()

    # Get data directory
    data_dir = ArgsParser.get_data_dir(args)

    print("\n" + "="*80)
    print("SMPL POSES ANIMATION")
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

        # Validate frame range
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

        print(f"📌 Animation range: frames {start_frame} to {end_frame} (step: {args.frame_step})")
        
        # Calculate animation info
        total_frames = len(range(start_frame, end_frame + 1, args.frame_step))
        if args.duration:
            actual_fps = total_frames / args.duration
            print(f"📌 Animation duration: {args.duration:.1f}s ({actual_fps:.1f} fps)")
        else:
            duration = total_frames / args.fps
            print(f"📌 Animation settings: {args.fps} fps, ~{duration:.1f}s duration")
        
        if args.num_subjects is not None:
            print(f"📌 Limiting to {args.num_subjects} subjects")

        # Create animated 3D visualization
        print(f"\nGenerating 3D animation...")
        fig = poses.animate_3d_poses(
            start_frame=start_frame,
            end_frame=end_frame,
            frame_step=args.frame_step,
            figsize=tuple(args.figsize),
            elev=args.elev,
            azim=args.azim,
            num_subjects=args.num_subjects,
            fps=args.fps,
            duration=args.duration
        )

        # Save or display - default to MP4 output
        output_path = args.output
        if output_path is None:
            # Default MP4 output with sequence name
            output_path = f"{sequence_name}_poses_animation.mp4"
        
        if output_path:
            print(f"\n💾 Saving animation to: {output_path}")
            print("⏳ This may take a while...")
            
            # Determine writer based on file extension - default to MP4
            if output_path.lower().endswith('.gif'):
                writer = 'pillow'
                writer_fps = args.fps
                print(f"📝 Using GIF format at {writer_fps} fps")
            elif output_path.lower().endswith('.mp4'):
                writer = 'ffmpeg'
                writer_fps = args.fps
                print(f"📝 Using MP4 format at {writer_fps} fps (requires ffmpeg)")
            else:
                # Default to MP4 if no extension specified
                writer = 'ffmpeg'
                writer_fps = args.fps
                if not output_path.lower().endswith(('.mp4', '.gif')):
                    output_path = output_path + '.mp4'
                print(f"📝 Using default MP4 format at {writer_fps} fps")
            
            try:
                # For MP4, use additional parameters to ensure precise FPS
                if writer == 'ffmpeg':
                    # Use extra_args to ensure precise frame rate
                    extra_args = ['-r', str(writer_fps), '-vcodec', 'libx264', '-pix_fmt', 'yuv420p']
                    fig._animation.save(output_path, writer=writer, fps=writer_fps, extra_args=extra_args)
                else:
                    fig._animation.save(output_path, writer=writer, fps=writer_fps)
                
                print(f"✓ Animation saved successfully at {writer_fps} fps!")
                print(f"📁 File: {output_path}")
            except Exception as e:
                print(f"❌ Error saving animation: {e}")
                print("💡 Try installing ffmpeg for MP4 or use .gif extension")
                print("💡 For Ubuntu/Debian: sudo apt install ffmpeg")
                print("💡 For macOS: brew install ffmpeg")
                sys.exit(1)
            
            plt.close(fig)
        else:
            print(f"\n✓ Displaying animation")
            print("⚠️  Note: Interactive display FPS may not be exact")
            print("💡 For precise FPS control, save as MP4: --output animation.mp4")
            print("Close the window to exit...")
            plt.show()

        print("\n" + "="*80)
        print("ANIMATION COMPLETE")
        print("="*80 + "\n")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    return 0


if __name__ == "__main__":
    sys.exit(main())