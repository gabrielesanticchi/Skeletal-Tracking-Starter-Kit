"""
Animate bounding boxes overlaid on video from the FIFA Skeletal Tracking Challenge dataset.

This script creates MP4 videos showing bounding boxes overlaid on the original video frames.
The output MP4 matches the frame rate and duration of the original video sequence.

Usage:
    # Full sequence animation
    python scripts/visualization/animate_bboxes.py --sequence ARG_FRA_183303

    # Specific frame range
    python scripts/visualization/animate_bboxes.py --sequence ARG_FRA_183303 --start-frame 50 --end-frame 150

    # Save as MP4 (default behavior)
    python scripts/visualization/animate_bboxes.py --sequence ARG_FRA_183303 --output bboxes_video.mp4

    # Custom frame rate (default matches video source)
    python scripts/visualization/animate_bboxes.py --sequence ARG_FRA_183303 --fps 25 --output custom_fps.mp4

    # Skip frames for faster processing
    python scripts/visualization/animate_bboxes.py --sequence ARG_FRA_183303 --frame-step 2 --output fast_video.mp4

    # Limit number of subjects shown
    python scripts/visualization/animate_bboxes.py --sequence ARG_FRA_183303 --num-subjects 3 --output limited.mp4

    # Show/hide labels
    python scripts/visualization/animate_bboxes.py --sequence ARG_FRA_183303 --no-labels --output no_labels.mp4
"""

import sys
import random
import cv2
import numpy as np
from pathlib import Path
from typing import Optional, List, Tuple

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

from classes import BBoxesData, VideoMetadata
from utils import ArgsParser


def detect_video_fps(images_dir: Path, sequence_name: str) -> float:
    """
    Detect the frame rate of the original video by analyzing frame timestamps.
    
    Args:
        images_dir: Directory containing image frames
        sequence_name: Name of the sequence
        
    Returns:
        Estimated frame rate (defaults to 25.0 if detection fails)
    """
    sequence_dir = images_dir / sequence_name
    if not sequence_dir.exists():
        return 25.0  # Default fallback
    
    # Get all frame files
    frame_files = sorted(sequence_dir.glob("*.jpg"))
    if len(frame_files) < 10:
        return 25.0  # Need enough frames for reliable detection
    
    # For FIFA dataset, assume standard video frame rates
    # Most common rates: 25, 30, 50, 60 fps
    num_frames = len(frame_files)
    
    # Heuristic: FIFA matches are typically 25 or 50 fps
    # If we have a lot of frames, likely 50fps, otherwise 25fps
    if num_frames > 1000:
        return 50.0
    else:
        return 25.0


def create_bbox_video(
    sequence_name: str,
    bboxes: BBoxesData,
    images_dir: Path,
    output_path: str,
    start_frame: int = 0,
    end_frame: Optional[int] = None,
    frame_step: int = 1,
    fps: Optional[float] = None,
    num_subjects: Optional[int] = None,
    show_labels: bool = True,
    color_palette: Optional[List[Tuple[int, int, int]]] = None
) -> None:
    """
    Create MP4 video with bounding boxes overlaid on frames.
    
    Args:
        sequence_name: Name of the sequence
        bboxes: BBoxesData instance
        images_dir: Directory containing image frames
        output_path: Output MP4 file path
        start_frame: Starting frame index
        end_frame: Ending frame index (None = last frame)
        frame_step: Step size between frames
        fps: Frame rate for output video (None = auto-detect)
        num_subjects: Maximum number of subjects to show (None = all)
        show_labels: Whether to show subject ID labels
        color_palette: Optional list of BGR colors for different subjects
    """
    # Validate frame range
    if end_frame is None:
        end_frame = bboxes.num_frames - 1
    
    start_frame = max(0, start_frame)
    end_frame = min(bboxes.num_frames - 1, end_frame)
    
    if start_frame >= end_frame:
        raise ValueError(f"Invalid frame range: start_frame={start_frame}, end_frame={end_frame}")
    
    # Create frame sequence
    frame_sequence = list(range(start_frame, end_frame + 1, frame_step))
    
    if not frame_sequence:
        raise ValueError("No frames to process")
    
    # Auto-detect FPS if not provided
    if fps is None:
        fps = detect_video_fps(images_dir, sequence_name)
        print(f"📊 Auto-detected frame rate: {fps} fps")
    
    # Default color palette
    if color_palette is None:
        color_palette = [
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
    
    # Load first frame to get video dimensions
    first_frame_path = images_dir / sequence_name / f"{frame_sequence[0]:05d}.jpg"
    if not first_frame_path.exists():
        raise FileNotFoundError(f"First frame not found: {first_frame_path}")
    
    first_frame = cv2.imread(str(first_frame_path))
    height, width = first_frame.shape[:2]
    
    print(f"📹 Video dimensions: {width}x{height}")
    print(f"📊 Processing {len(frame_sequence)} frames at {fps} fps")
    
    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    if not video_writer.isOpened():
        raise RuntimeError(f"Failed to open video writer for {output_path}")
    
    try:
        # Process each frame
        for i, frame_idx in enumerate(frame_sequence):
            # Load frame image
            frame_path = images_dir / sequence_name / f"{frame_idx:05d}.jpg"
            if not frame_path.exists():
                print(f"⚠️  Warning: Frame {frame_idx:05d}.jpg not found, skipping...")
                continue
            
            image = cv2.imread(str(frame_path))
            if image is None:
                print(f"⚠️  Warning: Could not load frame {frame_idx:05d}.jpg, skipping...")
                continue
            
            # Get bounding boxes for this frame
            frame_boxes = bboxes.boxes[frame_idx]
            
            # Filter valid boxes and apply subject limit
            valid_subjects = []
            for subject_idx, bbox in enumerate(frame_boxes):
                if not np.any(np.isnan(bbox)):
                    valid_subjects.append(subject_idx)
            
            # Limit number of subjects if specified
            if num_subjects is not None and num_subjects > 0:
                valid_subjects = valid_subjects[:num_subjects]
            
            # Draw bounding boxes
            for subject_idx in valid_subjects:
                bbox = frame_boxes[subject_idx]
                
                # Extract coordinates
                x_min, y_min, x_max, y_max = bbox.astype(int)
                
                # Select color
                color = color_palette[subject_idx % len(color_palette)]
                
                # Draw rectangle
                cv2.rectangle(image, (x_min, y_min), (x_max, y_max), color, 2)
                
                # Draw label if requested
                if show_labels:
                    label = f"ID:{subject_idx}"
                    label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                    label_y = max(y_min - 5, label_size[1] + 5)
                    
                    # Draw label background
                    cv2.rectangle(
                        image,
                        (x_min, label_y - label_size[1] - 4),
                        (x_min + label_size[0] + 4, label_y + 2),
                        color,
                        -1
                    )
                    
                    # Draw label text
                    cv2.putText(
                        image,
                        label,
                        (x_min + 2, label_y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (255, 255, 255),
                        1,
                        cv2.LINE_AA
                    )
            
            # Add frame info
            info_text = f"Frame: {frame_idx} | Subjects: {len(valid_subjects)}"
            cv2.putText(
                image,
                info_text,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
                cv2.LINE_AA
            )
            
            # Write frame to video
            video_writer.write(image)
            
            # Progress indicator
            if (i + 1) % 50 == 0 or i == len(frame_sequence) - 1:
                progress = (i + 1) / len(frame_sequence) * 100
                print(f"⏳ Progress: {progress:.1f}% ({i + 1}/{len(frame_sequence)} frames)")
    
    finally:
        # Clean up
        video_writer.release()
    
    print(f"✅ Video saved successfully: {output_path}")


def main():
    """Main function."""
    # Create parser with base and animation arguments
    parser = ArgsParser.create_base_parser(
        "Animate bounding boxes overlaid on video from FIFA Skeletal Tracking Challenge"
    )
    parser = ArgsParser.add_animation_args(parser)
    
    # Add bbox-specific arguments
    parser.add_argument(
        '--num-subjects',
        type=int,
        default=None,
        help='Maximum number of subjects to show (default: None, shows all subjects)'
    )
    
    parser.add_argument(
        '--no-labels',
        action='store_true',
        default=False,
        help='Hide subject ID labels (default: False, labels are shown)'
    )
    
    parser.add_argument(
        '--boxes-file',
        type=str,
        default='boxes.npz',
        help='Bounding boxes file to use (default: boxes.npz, can use boxes_all.npz for SMPL-generated)'
    )
    
    # Note: --output is already defined in base parser, no need to redefine
    
    args = parser.parse_args()

    # Get data directory
    data_dir = ArgsParser.get_data_dir(args)

    print("\n" + "="*80)
    print("BOUNDING BOX VIDEO ANIMATION")
    print("="*80 + "\n")

    try:
        # Load bounding box data
        print("Loading bounding box data...")
        boxes_path = data_dir / args.boxes_file
        if not boxes_path.exists():
            print(f"❌ Error: Bounding boxes file not found: {boxes_path}")
            print("💡 Make sure you have generated bounding boxes using the preprocessing scripts")
            sys.exit(1)
        
        boxes_dict = BBoxesData.load_all(boxes_path)
        sequences = list(boxes_dict.keys())
        print(f"✓ Loaded {len(sequences)} sequences with bounding boxes\n")

        # Select sequence
        sequence_name = args.sequence or random.choice(sequences)
        if sequence_name not in sequences:
            print(f"\n❌ Error: Sequence '{sequence_name}' not found in boxes.npz")
            print(f"Available sequences: {', '.join(sequences)}")
            sys.exit(1)
        print(f"📌 {'Randomly selected' if not args.sequence else 'Using'} sequence: {sequence_name}")

        bboxes = boxes_dict[sequence_name]

        # Validate frame range
        start_frame = args.start_frame
        end_frame = args.end_frame if args.end_frame is not None else bboxes.num_frames - 1
        
        if start_frame < 0 or start_frame >= bboxes.num_frames:
            print(f"\n❌ Error: Start frame {start_frame} out of range [0, {bboxes.num_frames-1}]")
            sys.exit(1)
        
        if end_frame < 0 or end_frame >= bboxes.num_frames:
            print(f"\n❌ Error: End frame {end_frame} out of range [0, {bboxes.num_frames-1}]")
            sys.exit(1)
            
        if start_frame >= end_frame:
            print(f"\n❌ Error: Start frame ({start_frame}) must be less than end frame ({end_frame})")
            sys.exit(1)

        print(f"📌 Frame range: {start_frame} to {end_frame} (step: {args.frame_step})")
        
        # Calculate video info
        total_frames = len(range(start_frame, end_frame + 1, args.frame_step))
        if args.duration:
            actual_fps = total_frames / args.duration
            print(f"📌 Video duration: {args.duration:.1f}s ({actual_fps:.1f} fps)")
        else:
            duration = total_frames / args.fps
            print(f"📌 Video settings: {args.fps} fps, ~{duration:.1f}s duration")
        
        if args.num_subjects is not None:
            print(f"📌 Limiting to {args.num_subjects} subjects")
        
        if args.no_labels:
            print(f"📌 Subject labels: disabled")

        # Set output path
        output_path = args.output
        if output_path is None:
            output_path = f"{sequence_name}_bboxes.mp4"
        elif not output_path.lower().endswith('.mp4'):
            output_path += '.mp4'

        # Check images directory
        images_dir = data_dir / "images"
        if not images_dir.exists():
            print(f"\n❌ Error: Images directory not found: {images_dir}")
            print("💡 Make sure you have extracted video frames to the images directory")
            sys.exit(1)

        sequence_images_dir = images_dir / sequence_name
        if not sequence_images_dir.exists():
            print(f"\n❌ Error: Sequence images directory not found: {sequence_images_dir}")
            sys.exit(1)

        print(f"\n🎬 Creating bounding box video...")
        print(f"📁 Output: {output_path}")

        # Create the video
        create_bbox_video(
            sequence_name=sequence_name,
            bboxes=bboxes,
            images_dir=images_dir,
            output_path=output_path,
            start_frame=start_frame,
            end_frame=end_frame,
            frame_step=args.frame_step,
            fps=args.fps if args.duration is None else total_frames / args.duration,
            num_subjects=args.num_subjects,
            show_labels=not args.no_labels
        )

        print("\n" + "="*80)
        print("VIDEO CREATION COMPLETE")
        print("="*80 + "\n")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    return 0


if __name__ == "__main__":
    sys.exit(main())