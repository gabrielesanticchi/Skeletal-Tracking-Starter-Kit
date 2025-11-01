"""
Extract frames from video files and save them as sequential JPEG images.
This script creates the images/ folder structure required by the baseline model.

Usage:
    From project root: python scripts/preprocessing/extract_frames.py
    From this directory: python extract_frames.py [--sequences SEQ1 SEQ2 ...] [--fps-limit N]

If no sequences are specified, all sequences found in boxes.npz will be processed.
"""

import numpy as np
import cv2
from pathlib import Path
from typing import List, Optional
import sys
import argparse
from tqdm import tqdm


def get_project_root() -> Path:
    """
    Get the project root directory.

    Returns:
        Path to project root
    """
    current = Path(__file__).resolve()
    # Go up from scripts/preprocessing/ to project root
    return current.parent.parent.parent


def load_camera_sequences(cameras_dir: Path) -> List[str]:
    """
    Load sequence names from cameras directory.

    Args:
        cameras_dir: Path to cameras directory

    Returns:
        List of sequence names (without .npz extension)
    """
    if not cameras_dir.exists():
        return []
    
    # Get all .npz files in cameras directory
    camera_files = sorted(cameras_dir.glob("*.npz"))
    # Extract sequence names (remove .npz extension)
    sequences = [f.stem for f in camera_files]
    return sequences


def find_video_file(sequence_name: str, video_base_dir: Path) -> Optional[Path]:
    """
    Find the video file for a given sequence in the video directory structure.

    Args:
        sequence_name: Name of the sequence
        video_base_dir: Base directory containing video subdirectories

    Returns:
        Path to video file if found, None otherwise
    """
    video_subdirs = ["train_data", "test_data", "challenge_data"]

    for subdir in video_subdirs:
        video_path = video_base_dir / subdir / f"{sequence_name}.mp4"
        if video_path.exists():
            return video_path

    return None


def check_if_images_exist(output_dir: Path) -> bool:
    """
    Check if images already exist for a sequence.

    Args:
        output_dir: Output directory to check

    Returns:
        True if images exist, False otherwise
    """
    if not output_dir.exists():
        return False
    
    # Check if directory has any .jpg files
    jpg_files = list(output_dir.glob("*.jpg"))
    return len(jpg_files) > 0


def extract_frames_from_video(
    video_path: Path,
    output_dir: Path,
    fps_limit: Optional[int] = None
) -> int:
    """
    Extract frames from a video file and save as sequential JPEG images.

    Args:
        video_path: Path to the video file
        output_dir: Output directory for extracted frames
        fps_limit: If specified, only extract frames up to this FPS

    Returns:
        Number of frames extracted
    """
    # Open video
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    # Get video properties
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Calculate frame skip if fps_limit is specified
    frame_skip = 1
    if fps_limit and fps > fps_limit:
        frame_skip = int(fps / fps_limit)

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Extract frames
    frame_idx = 0
    saved_count = 0

    with tqdm(total=total_frames, desc=f"Extracting {video_path.stem}") as pbar:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Save frame if we're not skipping it
            if frame_idx % frame_skip == 0:
                # Format filename with 5-digit padding: 00000.jpg, 00001.jpg, etc.
                frame_filename = f"{saved_count:05d}.jpg"
                frame_path = output_dir / frame_filename

                # Save frame as JPEG
                cv2.imwrite(str(frame_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
                saved_count += 1

            frame_idx += 1
            pbar.update(1)

    cap.release()
    return saved_count


def main():
    """Main extraction function."""
    parser = argparse.ArgumentParser(
        description="Extract frames from videos for FIFA Skeletal Tracking Challenge"
    )
    parser.add_argument(
        "--sequences",
        nargs="+",
        help="Specific sequences to process (e.g., ARG_FRA_182345). If not specified, all sequences in cameras/ will be processed."
    )
    parser.add_argument(
        "--fps-limit",
        type=int,
        help="Limit extraction to specified FPS (useful for reducing dataset size)"
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Base data directory (default: auto-detect from project root)"
    )

    args = parser.parse_args()

    # Setup paths relative to project root
    if args.data_dir:
        data_dir = args.data_dir
    else:
        project_root = get_project_root()
        data_dir = project_root / "data"

    cameras_dir = data_dir / "cameras"
    video_base_dir = data_dir / "videos"
    images_base_dir = data_dir / "images"

    # Check if required paths exist
    if not cameras_dir.exists():
        print(f"❌ Error: cameras directory not found at {cameras_dir}")
        sys.exit(1)

    if not video_base_dir.exists():
        print(f"❌ Error: videos directory not found at {video_base_dir}")
        sys.exit(1)

    # Determine which sequences to process
    if args.sequences:
        sequences_to_process = args.sequences
        print(f"Processing {len(sequences_to_process)} specified sequence(s)")
    else:
        sequences_to_process = load_camera_sequences(cameras_dir)
        print(f"Processing all {len(sequences_to_process)} sequences from cameras/")

    # Process each sequence
    print("\n" + "="*80)
    print("FRAME EXTRACTION")
    print("="*80 + "\n")

    successful = []
    failed = []

    skipped = []

    for sequence_name in sequences_to_process:
        print(f"\n📹 Processing: {sequence_name}")

        # Setup output directory
        output_dir = images_base_dir / sequence_name

        # Check if images already exist
        if check_if_images_exist(output_dir):
            print(f"   ⏭️  Images already exist, skipping...")
            skipped.append(sequence_name)
            continue

        # Find video file
        video_path = find_video_file(sequence_name, video_base_dir)
        if video_path is None:
            print(f"   ❌ Video file not found for {sequence_name}")
            failed.append(sequence_name)
            continue

        print(f"   Found video: {video_path.relative_to(data_dir)}")

        # Extract frames
        try:
            num_frames = extract_frames_from_video(
                video_path,
                output_dir,
                fps_limit=args.fps_limit
            )
            print(f"   ✅ Extracted {num_frames} frames to {output_dir.relative_to(data_dir)}")
            successful.append(sequence_name)
        except Exception as e:
            print(f"   ❌ Error: {e}")
            failed.append(sequence_name)

    # Summary
    print("\n" + "="*80)
    print("EXTRACTION SUMMARY")
    print("="*80)
    print(f"✅ Successful: {len(successful)}/{len(sequences_to_process)}")
    if skipped:
        print(f"⏭️  Skipped (already exist): {len(skipped)}")
    if failed:
        print(f"❌ Failed: {len(failed)}")
        print("   Failed sequences:")
        for seq in failed:
            print(f"   - {seq}")
    
    if len(failed) == 0 and len(successful) > 0:
        print("\n🎉 All new sequences processed successfully!")
        print(f"\nFrames saved to: {images_base_dir}/")
        print("\nNext steps:")
        print("1. Verify the extracted images match your expectations")
        print("2. Run the baseline model: python baseline.py")
    elif len(skipped) > 0 and len(successful) == 0 and len(failed) == 0:
        print("\n✓ All sequences already have extracted images.")

    return 0 if len(failed) == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
