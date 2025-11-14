"""
Generate bounding boxes from 2D skeleton keypoints for all sequences.

This script generates bounding boxes by using the existing 2D skeleton keypoints
and creating tight bounding boxes around them. This is more reliable than SMPL
mesh projection and doesn't require SMPL models.

Usage:
    # Generate boxes for all sequences with 2D skeleton data
    python scripts/preprocessing/generate_boxes_from_2d_skeleton.py

    # Generate boxes for specific sequences
    python scripts/preprocessing/generate_boxes_from_2d_skeleton.py --sequences ARG_CRO_220001 ARG_FRA_182345

    # Test on single sequence with debug output
    python scripts/preprocessing/generate_boxes_from_2d_skeleton.py --sequences ARG_CRO_220001 --debug
"""

import numpy as np
from pathlib import Path
import argparse
from tqdm import tqdm
import sys
from typing import Dict, Optional, Tuple


def get_project_root() -> Path:
    """Get the project root directory."""
    current = Path(__file__).resolve()
    return current.parent.parent.parent


def keypoints_to_bbox(
    keypoints: np.ndarray,
    margin: float = 0.1,
    min_bbox_size: int = 30,
    image_width: int = 1920,
    image_height: int = 1080
) -> np.ndarray:
    """
    Compute bounding box from 2D keypoints.
    
    Args:
        keypoints: (25, 2) array of 2D keypoints
        margin: Additional margin to add to bbox (as fraction of bbox size)
        min_bbox_size: Minimum bounding box size in pixels
        image_width: Image width for bounds checking
        image_height: Image height for bounds checking
        
    Returns:
        (4,) array [x_min, y_min, x_max, y_max] or [nan, nan, nan, nan] if invalid
    """
    # Filter out invalid keypoints (zero or NaN)
    valid_mask = (
        np.isfinite(keypoints).all(axis=1) & 
        ~(keypoints == 0).all(axis=1)
    )
    
    if not valid_mask.any():
        return np.array([np.nan, np.nan, np.nan, np.nan])
    
    valid_keypoints = keypoints[valid_mask]
    
    # Filter keypoints that are within image bounds
    in_bounds_mask = (
        (valid_keypoints[:, 0] >= 0) & 
        (valid_keypoints[:, 0] < image_width) &
        (valid_keypoints[:, 1] >= 0) & 
        (valid_keypoints[:, 1] < image_height)
    )
    
    if not in_bounds_mask.any():
        return np.array([np.nan, np.nan, np.nan, np.nan])
    
    bounded_keypoints = valid_keypoints[in_bounds_mask]
    
    # Compute bounding box
    x_min, y_min = bounded_keypoints.min(axis=0)
    x_max, y_max = bounded_keypoints.max(axis=0)
    
    # Check minimum size
    width = x_max - x_min
    height = y_max - y_min
    
    if width < min_bbox_size or height < min_bbox_size:
        return np.array([np.nan, np.nan, np.nan, np.nan])
    
    # Add margin
    if margin > 0:
        x_margin = width * margin
        y_margin = height * margin
        
        x_min -= x_margin
        y_min -= y_margin
        x_max += x_margin
        y_max += y_margin
    
    # Clamp to image bounds
    x_min = max(0, x_min)
    y_min = max(0, y_min)
    x_max = min(image_width, x_max)
    y_max = min(image_height, y_max)
    
    return np.array([x_min, y_min, x_max, y_max])


def generate_boxes_for_sequence(
    sequence_name: str,
    data_dir: Path,
    margin: float = 0.1,
    debug: bool = False
) -> Optional[np.ndarray]:
    """
    Generate bounding boxes for a sequence using 2D skeleton keypoints.
    
    Args:
        sequence_name: Name of the sequence
        data_dir: Path to data directory
        margin: Margin to add to bounding boxes (fraction of bbox size)
        debug: Enable debug output
        
    Returns:
        (num_frames, num_subjects, 4) array of bounding boxes or None if failed
    """
    # Load 2D skeleton data
    skel_2d_path = data_dir / "skel_2d.npz"
    if not skel_2d_path.exists():
        print(f"  ⚠️  No 2D skeleton data found: {skel_2d_path}")
        return None
    
    try:
        skel_2d_data = np.load(skel_2d_path, allow_pickle=True)
        
        if sequence_name not in skel_2d_data.files:
            print(f"  ⚠️  Sequence {sequence_name} not found in 2D skeleton data")
            return None
        
        keypoints = skel_2d_data[sequence_name]  # (num_frames, num_subjects, 25, 2)
        num_frames, num_subjects = keypoints.shape[:2]
        
        if debug:
            print(f"    Keypoints shape: {keypoints.shape}")
            print(f"    Frames: {num_frames}, subjects: {num_subjects}")
        
        # Initialize output array
        boxes = np.zeros((num_frames, num_subjects, 4))
        boxes.fill(np.nan)
        
        # Process each frame
        for frame_idx in tqdm(range(num_frames), desc=f"  Processing {sequence_name}", leave=False):
            frame_keypoints = keypoints[frame_idx]  # (num_subjects, 25, 2)
            
            # Process each subject
            for subj_idx in range(num_subjects):
                subject_keypoints = frame_keypoints[subj_idx]  # (25, 2)
                
                # Generate bounding box from keypoints
                bbox = keypoints_to_bbox(subject_keypoints, margin=margin)
                boxes[frame_idx, subj_idx] = bbox
                
                if debug and frame_idx < 3 and subj_idx < 2:
                    valid_kpts = subject_keypoints[
                        (subject_keypoints != 0).any(axis=1) & 
                        np.isfinite(subject_keypoints).all(axis=1)
                    ]
                    if len(valid_kpts) > 0:
                        print(f"    Frame {frame_idx}, Subject {subj_idx}: {len(valid_kpts)} valid keypoints")
                        print(f"    Keypoint range: X[{valid_kpts[:, 0].min():.1f}, {valid_kpts[:, 0].max():.1f}] Y[{valid_kpts[:, 1].min():.1f}, {valid_kpts[:, 1].max():.1f}]")
                        if not np.isnan(bbox).any():
                            print(f"    Bbox: [{bbox[0]:.1f}, {bbox[1]:.1f}, {bbox[2]:.1f}, {bbox[3]:.1f}]")
                        else:
                            print(f"    Bbox: Invalid")
                    else:
                        print(f"    Frame {frame_idx}, Subject {subj_idx}: No valid keypoints")
        
        return boxes
    
    except Exception as e:
        print(f"  ❌ Error processing {sequence_name}: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Generate bounding boxes from 2D skeleton keypoints"
    )
    parser.add_argument(
        "--sequences",
        nargs='+',
        help="Specific sequences to process (default: all with 2D skeleton data)"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output file path (default: data/boxes_from_2d.npz)"
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Base data directory"
    )
    parser.add_argument(
        "--margin",
        type=float,
        default=0.1,
        help="Margin to add to bboxes as fraction of bbox size (default: 0.1)"
    )
    parser.add_argument(
        "--debug",
        action='store_true',
        help="Enable debug output for first few frames"
    )
    
    args = parser.parse_args()
    
    # Setup paths
    if args.data_dir:
        data_dir = args.data_dir
    else:
        project_root = get_project_root()
        data_dir = project_root / "data"
    
    if args.output:
        output_path = args.output
    else:
        output_path = data_dir / "boxes_from_2d.npz"
    
    print("\n" + "="*80)
    print("GENERATE BOUNDING BOXES FROM 2D SKELETON KEYPOINTS")
    print("="*80 + "\n")
    
    # Check for 2D skeleton data
    skel_2d_path = data_dir / "skel_2d.npz"
    if not skel_2d_path.exists():
        print(f"❌ Error: 2D skeleton data not found: {skel_2d_path}")
        sys.exit(1)
    
    # Get sequences to process
    if args.sequences:
        sequences = args.sequences
    else:
        # Get all sequences from 2D skeleton data
        skel_2d_data = np.load(skel_2d_path, allow_pickle=True)
        sequences = sorted(list(skel_2d_data.files))
    
    print(f"📋 Processing {len(sequences)} sequences")
    print(f"📂 Data directory: {data_dir}")
    print(f"💾 Output file: {output_path}")
    print(f"🎨 Margin: {args.margin}")
    if args.debug:
        print(f"🐛 Debug mode: ON")
    
    # Process sequences
    print(f"\n🔄 Processing sequences...\n")
    generated_boxes = {}
    
    for seq in sequences:
        print(f"📹 {seq}")
        boxes = generate_boxes_for_sequence(
            seq,
            data_dir,
            margin=args.margin,
            debug=args.debug
        )
        
        if boxes is not None:
            generated_boxes[seq] = boxes
            valid_boxes = np.sum(~np.isnan(boxes[:, :, 0]))
            total_boxes = boxes.shape[0] * boxes.shape[1]
            print(f"   ✓ Generated {boxes.shape[0]} frames, {boxes.shape[1]} subjects")
            print(f"   ✓ Valid boxes: {valid_boxes}/{total_boxes} ({valid_boxes/total_boxes*100:.1f}%)")
        else:
            print(f"   ⚠️  Skipped")
    
    # Save results
    print(f"\n💾 Saving {len(generated_boxes)} sequences to {output_path}")
    np.savez_compressed(output_path, **generated_boxes)
    print(f"✓ Saved successfully")
    
    # Print summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Total sequences: {len(generated_boxes)}")
    print(f"Output file: {output_path}")
    if output_path.exists():
        print(f"File size: {output_path.stat().st_size / 1024 / 1024:.2f} MB")
    print("\nSequences included:")
    for seq in sorted(generated_boxes.keys()):
        frames, subjects = generated_boxes[seq].shape[:2]
        valid_boxes = np.sum(~np.isnan(generated_boxes[seq][:, :, 0]))
        total_boxes = frames * subjects
        print(f"  - {seq}: {frames} frames, {subjects} subjects, {valid_boxes}/{total_boxes} valid boxes")
    print("\n" + "="*80 + "\n")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())