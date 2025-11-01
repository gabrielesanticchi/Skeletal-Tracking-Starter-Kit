"""
Validation script to check consistency between video frames and bounding box data.
This script verifies that the number of frames in each video matches the corresponding
bounding box entries in boxes.npz.

Usage:
    From project root: python scripts/preprocessing/validate_data.py
    From this directory: python validate_data.py
"""

import numpy as np
import cv2
from pathlib import Path
from typing import Dict, Tuple
import sys


def get_project_root() -> Path:
    """
    Get the project root directory.

    Returns:
        Path to project root
    """
    current = Path(__file__).resolve()
    # Go up from scripts/preprocessing/ to project root
    return current.parent.parent.parent


def get_video_frame_count(video_path: Path) -> int:
    """
    Get the number of frames in a video file.

    Args:
        video_path: Path to the video file

    Returns:
        Number of frames in the video
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return frame_count


def load_boxes_data(boxes_path: Path) -> Dict[str, np.ndarray]:
    """
    Load bounding box data from .npz file.

    Args:
        boxes_path: Path to boxes.npz file

    Returns:
        Dictionary with sequence names as keys and bbox arrays as values
    """
    data = np.load(boxes_path, allow_pickle=True)
    return {key: data[key] for key in data.files}


def validate_consistency(
    video_dir: Path,
    boxes_data: Dict[str, np.ndarray]
) -> Tuple[bool, list]:
    """
    Validate consistency between video frames and bounding box data.

    Args:
        video_dir: Directory containing video files
        boxes_data: Dictionary of bounding box data

    Returns:
        Tuple of (is_consistent, issues_list)
    """
    issues = []
    all_consistent = True

    print("\n" + "="*80)
    print("VALIDATION REPORT: Video Frames vs Bounding Boxes")
    print("="*80 + "\n")

    # Get all video files
    video_files = sorted(video_dir.glob("*.mp4"))

    print(f"Found {len(video_files)} video files")
    print(f"Found {len(boxes_data)} sequences in boxes.npz\n")

    # Check each video
    for video_path in video_files:
        sequence_name = video_path.stem  # Filename without extension

        # Check if sequence exists in boxes data
        if sequence_name not in boxes_data:
            issue = f"❌ {sequence_name}: No corresponding entry in boxes.npz"
            issues.append(issue)
            print(issue)
            all_consistent = False
            continue

        # Get frame counts
        video_frames = get_video_frame_count(video_path)
        bbox_array = boxes_data[sequence_name]
        bbox_frames = bbox_array.shape[0]  # First dimension is frames
        num_subjects = bbox_array.shape[1] if len(bbox_array.shape) > 1 else 0

        # Check consistency
        if video_frames == bbox_frames:
            print(f"✓ {sequence_name}: {video_frames} frames, {num_subjects} subjects - CONSISTENT")
        else:
            issue = (f"❌ {sequence_name}: Video has {video_frames} frames, "
                    f"but boxes.npz has {bbox_frames} frames - MISMATCH")
            issues.append(issue)
            print(issue)
            all_consistent = False

    # Check for sequences in boxes.npz without corresponding videos
    video_names = {v.stem for v in video_files}
    for seq_name in boxes_data.keys():
        if seq_name not in video_names:
            issue = f"⚠️  {seq_name}: Entry in boxes.npz but no corresponding video file"
            issues.append(issue)
            print(issue)

    print("\n" + "="*80)
    if all_consistent and len(issues) == 0:
        print("✅ VALIDATION PASSED: All data is consistent!")
        print("You can proceed with image extraction.")
    else:
        print(f"❌ VALIDATION FAILED: Found {len(issues)} issue(s)")
        print("Please review the issues before proceeding.")
    print("="*80 + "\n")

    return all_consistent, issues


def main():
    """Main validation function."""
    # Setup paths relative to project root
    project_root = get_project_root()
    data_dir = project_root / "data"
    boxes_path = data_dir / "boxes.npz"
    video_base_dir = data_dir / "videos"

    # Check if required files exist
    if not boxes_path.exists():
        print(f"❌ Error: boxes.npz not found at {boxes_path}")
        sys.exit(1)

    if not video_base_dir.exists():
        print(f"❌ Error: videos directory not found at {video_base_dir}")
        sys.exit(1)

    # Load bounding box data
    print("Loading bounding box data...")
    boxes_data = load_boxes_data(boxes_path)

    # Validate each video subdirectory
    video_subdirs = ["train_data", "test_data", "challenge_data"]
    all_valid = True

    for subdir in video_subdirs:
        video_dir = video_base_dir / subdir
        if not video_dir.exists():
            print(f"⚠️  Warning: {subdir} directory not found, skipping...")
            continue

        print(f"\n{'='*80}")
        print(f"Checking {subdir.upper()}")
        print(f"{'='*80}")

        is_valid, _ = validate_consistency(video_dir, boxes_data)
        all_valid = all_valid and is_valid

    # Summary
    print("\n" + "="*80)
    print("OVERALL SUMMARY")
    print("="*80)
    if all_valid:
        print("✅ All video directories passed validation!")
        print("\nNext steps:")
        print("1. Run the image extraction script to generate images/ folder")
        print("2. Images should be named sequentially: 00000.jpg, 00001.jpg, etc.")
        return 0
    else:
        print("❌ Some validation checks failed.")
        print("Please review the issues above before proceeding with image extraction.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
