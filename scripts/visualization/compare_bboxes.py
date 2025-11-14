"""
Compare bounding boxes from different sources (original vs SMPL-generated).

This script compares bounding boxes from boxes.npz (original/manual annotations) 
with boxes_all.npz (generated from SMPL parameters) to assess accuracy.

Usage:
    # Compare all overlapping sequences
    python scripts/visualization/compare_bboxes.py

    # Compare specific sequence
    python scripts/visualization/compare_bboxes.py --sequence ARG_FRA_183303

    # Generate comparison visualization
    python scripts/visualization/compare_bboxes.py --sequence ARG_FRA_183303 --frame 100 --output comparison.jpg

    # Show detailed statistics
    python scripts/visualization/compare_bboxes.py --sequence ARG_FRA_183303 --stats
"""

import sys
import numpy as np
import cv2
from pathlib import Path
from typing import Dict, Tuple, Optional, List
import matplotlib.pyplot as plt

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

from classes import BBoxesData, ImageMetadata
from utils import ArgsParser


def calculate_iou(box1: np.ndarray, box2: np.ndarray) -> float:
    """
    Calculate Intersection over Union (IoU) between two bounding boxes.
    
    Args:
        box1: [x_min, y_min, x_max, y_max]
        box2: [x_min, y_min, x_max, y_max]
        
    Returns:
        IoU score (0.0 to 1.0)
    """
    # Skip if either box is invalid (contains NaN)
    if np.any(np.isnan(box1)) or np.any(np.isnan(box2)):
        return 0.0
    
    # Calculate intersection
    x_min = max(box1[0], box2[0])
    y_min = max(box1[1], box2[1])
    x_max = min(box1[2], box2[2])
    y_max = min(box1[3], box2[3])
    
    if x_max <= x_min or y_max <= y_min:
        return 0.0
    
    intersection = (x_max - x_min) * (y_max - y_min)
    
    # Calculate union
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection
    
    if union <= 0:
        return 0.0
    
    return intersection / union


def calculate_bbox_metrics(original_boxes: np.ndarray, generated_boxes: np.ndarray) -> Dict[str, float]:
    """
    Calculate various metrics comparing original and generated bounding boxes.
    
    Args:
        original_boxes: (num_frames, num_subjects, 4) original boxes
        generated_boxes: (num_frames, num_subjects, 4) generated boxes
        
    Returns:
        Dictionary with metrics
    """
    metrics = {
        'mean_iou': 0.0,
        'median_iou': 0.0,
        'iou_std': 0.0,
        'high_iou_ratio': 0.0,  # Ratio of boxes with IoU > 0.5
        'valid_pairs': 0,
        'total_pairs': 0,
        'coverage_original': 0.0,  # How many original boxes have valid generated counterparts
        'coverage_generated': 0.0,  # How many generated boxes have valid original counterparts
    }
    
    num_frames, num_subjects = original_boxes.shape[:2]
    ious = []
    valid_pairs = 0
    total_pairs = 0
    
    original_valid = 0
    generated_valid = 0
    matched_original = 0
    matched_generated = 0
    
    for frame_idx in range(num_frames):
        for subj_idx in range(num_subjects):
            total_pairs += 1
            
            orig_box = original_boxes[frame_idx, subj_idx]
            gen_box = generated_boxes[frame_idx, subj_idx]
            
            orig_valid = not np.any(np.isnan(orig_box))
            gen_valid = not np.any(np.isnan(gen_box))
            
            if orig_valid:
                original_valid += 1
            if gen_valid:
                generated_valid += 1
            
            if orig_valid and gen_valid:
                iou = calculate_iou(orig_box, gen_box)
                ious.append(iou)
                valid_pairs += 1
                
                if iou > 0.1:  # Consider matched if IoU > 0.1
                    matched_original += 1
                    matched_generated += 1
    
    if ious:
        ious = np.array(ious)
        metrics['mean_iou'] = float(np.mean(ious))
        metrics['median_iou'] = float(np.median(ious))
        metrics['iou_std'] = float(np.std(ious))
        metrics['high_iou_ratio'] = float(np.sum(ious > 0.5) / len(ious))
    
    metrics['valid_pairs'] = valid_pairs
    metrics['total_pairs'] = total_pairs
    
    if original_valid > 0:
        metrics['coverage_original'] = matched_original / original_valid
    if generated_valid > 0:
        metrics['coverage_generated'] = matched_generated / generated_valid
    
    return metrics


def visualize_bbox_comparison(
    image: np.ndarray,
    original_boxes: np.ndarray,
    generated_boxes: np.ndarray,
    frame_idx: int,
    sequence_name: str
) -> np.ndarray:
    """
    Create visualization comparing original and generated bounding boxes.
    
    Args:
        image: Input image
        original_boxes: Original bounding boxes for the frame
        generated_boxes: Generated bounding boxes for the frame
        frame_idx: Frame index
        sequence_name: Sequence name
        
    Returns:
        Image with comparison visualization
    """
    img_display = image.copy()
    
    # Colors: Blue for original, Red for generated, Green for good matches
    original_color = (255, 0, 0)  # Blue
    generated_color = (0, 0, 255)  # Red
    match_color = (0, 255, 0)     # Green
    
    num_subjects = min(len(original_boxes), len(generated_boxes))
    
    for subj_idx in range(num_subjects):
        orig_box = original_boxes[subj_idx]
        gen_box = generated_boxes[subj_idx]
        
        orig_valid = not np.any(np.isnan(orig_box))
        gen_valid = not np.any(np.isnan(gen_box))
        
        # Calculate IoU if both are valid
        iou = 0.0
        if orig_valid and gen_valid:
            iou = calculate_iou(orig_box, gen_box)
        
        # Draw original box (blue)
        if orig_valid:
            x_min, y_min, x_max, y_max = orig_box.astype(int)
            color = match_color if iou > 0.5 else original_color
            cv2.rectangle(img_display, (x_min, y_min), (x_max, y_max), color, 2)
            
            # Label
            label = f"Orig-{subj_idx}"
            if iou > 0:
                label += f" (IoU:{iou:.2f})"
            cv2.putText(img_display, label, (x_min, y_min - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        # Draw generated box (red, dashed-like effect with thinner line)
        if gen_valid:
            x_min, y_min, x_max, y_max = gen_box.astype(int)
            color = match_color if iou > 0.5 else generated_color
            
            # Draw dashed rectangle effect
            thickness = 1 if iou > 0.5 else 2
            cv2.rectangle(img_display, (x_min + 2, y_min + 2), (x_max - 2, y_max - 2), color, thickness)
            
            # Label
            label = f"SMPL-{subj_idx}"
            cv2.putText(img_display, label, (x_min + 2, y_max + 15), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    
    # Add legend and info
    info_text = f"Comparison - {sequence_name} - Frame {frame_idx}"
    cv2.putText(img_display, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Legend
    legend_y = 60
    cv2.putText(img_display, "Legend:", (10, legend_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(img_display, "Blue: Original", (10, legend_y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, original_color, 1)
    cv2.putText(img_display, "Red: SMPL Generated", (10, legend_y + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.4, generated_color, 1)
    cv2.putText(img_display, "Green: Good Match (IoU>0.5)", (10, legend_y + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.4, match_color, 1)
    
    return img_display


def main():
    """Main function."""
    parser = ArgsParser.create_base_parser(
        "Compare bounding boxes from different sources (original vs SMPL-generated)"
    )
    
    parser.add_argument(
        '--stats',
        action='store_true',
        help='Show detailed statistics for the comparison'
    )
    
    parser.add_argument(
        '--original-boxes',
        type=Path,
        default=None,
        help='Path to original boxes file (default: data/boxes.npz)'
    )
    
    parser.add_argument(
        '--generated-boxes',
        type=Path,
        default=None,
        help='Path to generated boxes file (default: data/boxes_all.npz)'
    )
    
    args = parser.parse_args()

    # Get data directory
    data_dir = ArgsParser.get_data_dir(args)

    print("\n" + "="*80)
    print("BOUNDING BOX COMPARISON ANALYSIS")
    print("="*80 + "\n")

    try:
        # Set default paths
        original_path = args.original_boxes or data_dir / "boxes.npz"
        generated_path = args.generated_boxes or data_dir / "boxes_all.npz"
        
        # Load bounding box data
        print("Loading bounding box data...")
        
        if not original_path.exists():
            print(f"❌ Error: Original boxes file not found: {original_path}")
            sys.exit(1)
            
        if not generated_path.exists():
            print(f"❌ Error: Generated boxes file not found: {generated_path}")
            sys.exit(1)
        
        original_dict = BBoxesData.load_all(original_path)
        generated_dict = BBoxesData.load_all(generated_path)
        
        print(f"✓ Original boxes: {len(original_dict)} sequences")
        print(f"✓ Generated boxes: {len(generated_dict)} sequences")
        
        # Find overlapping sequences
        original_sequences = set(original_dict.keys())
        generated_sequences = set(generated_dict.keys())
        common_sequences = original_sequences & generated_sequences
        
        print(f"✓ Common sequences: {len(common_sequences)}")
        print(f"  {', '.join(sorted(common_sequences))}\n")
        
        if not common_sequences:
            print("❌ No common sequences found for comparison")
            sys.exit(1)
        
        # Select sequence for analysis
        if args.sequence:
            if args.sequence not in common_sequences:
                print(f"❌ Sequence '{args.sequence}' not found in both files")
                print(f"Available sequences: {', '.join(sorted(common_sequences))}")
                sys.exit(1)
            sequences_to_analyze = [args.sequence]
        else:
            sequences_to_analyze = sorted(common_sequences)
        
        # Analyze sequences
        for sequence_name in sequences_to_analyze:
            print(f"📊 Analyzing sequence: {sequence_name}")
            
            original_boxes = original_dict[sequence_name]
            generated_boxes = generated_dict[sequence_name]
            
            print(f"  Original shape: {original_boxes.boxes.shape}")
            print(f"  Generated shape: {generated_boxes.boxes.shape}")
            
            # Calculate metrics
            metrics = calculate_bbox_metrics(original_boxes.boxes, generated_boxes.boxes)
            
            print(f"  📈 Metrics:")
            print(f"    Mean IoU: {metrics['mean_iou']:.3f}")
            print(f"    Median IoU: {metrics['median_iou']:.3f}")
            print(f"    IoU Std: {metrics['iou_std']:.3f}")
            print(f"    High IoU ratio (>0.5): {metrics['high_iou_ratio']:.3f}")
            print(f"    Valid pairs: {metrics['valid_pairs']}/{metrics['total_pairs']}")
            print(f"    Coverage (original): {metrics['coverage_original']:.3f}")
            print(f"    Coverage (generated): {metrics['coverage_generated']:.3f}")
            
            # Generate visualization if requested
            if args.frame is not None and args.output:
                print(f"  🎨 Generating comparison visualization...")
                
                # Load image
                frame_meta = ImageMetadata(sequence_name=sequence_name, frame_idx=args.frame, 
                                         bboxes=original_boxes)
                image = frame_meta.load_image(data_dir / "images")
                
                # Get boxes for the specific frame
                orig_frame_boxes = original_boxes.get_frame_boxes(args.frame, valid_only=False)
                gen_frame_boxes = generated_boxes.get_frame_boxes(args.frame, valid_only=False)
                
                # Create comparison visualization
                comparison_img = visualize_bbox_comparison(
                    image, orig_frame_boxes, gen_frame_boxes, args.frame, sequence_name
                )
                
                # Save
                cv2.imwrite(args.output, comparison_img)
                print(f"  ✓ Comparison saved to: {args.output}")
            
            print()

        print("="*80)
        print("COMPARISON ANALYSIS COMPLETE")
        print("="*80 + "\n")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    return 0


if __name__ == "__main__":
    sys.exit(main())