"""
Debug script to visualize joint positions on white background with scaling.

This plots joints at their relative positions but scaled up to see them clearly.
Each joint shows its index number and current name mapping.
"""

import sys
import cv2
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

from classes import Skeleton2DData
from utils.skeleton_viz import JOINT_NAMES

# Load data
data_dir = Path('data')
skel_2d_dict = Skeleton2DData.load_all(data_dir / 'skel_2d.npz')
sequence_name = list(skel_2d_dict.keys())[0]
skel_2d = skel_2d_dict[sequence_name]

frame_idx = 100
subject_idx = 0

# Get keypoints
kpts = skel_2d.get_frame_keypoints(frame_idx, subject_idx)

print(f"Sequence: {sequence_name}, Frame: {frame_idx}, Subject: {subject_idx}")
print(f"Keypoints shape: {kpts.shape}\n")

# Filter out invalid keypoints
valid_joints = []
valid_positions = []

for joint_idx in range(len(kpts)):
    pt = kpts[joint_idx]
    if not np.any(np.isnan(pt)) and not np.all(pt == 0):
        valid_joints.append(joint_idx)
        valid_positions.append(pt)

valid_positions = np.array(valid_positions)

print(f"Found {len(valid_joints)} valid joints\n")

# Find bounding box of all joints
min_x, min_y = valid_positions.min(axis=0)
max_x, max_y = valid_positions.max(axis=0)

print(f"Original bounding box: ({min_x:.1f}, {min_y:.1f}) to ({max_x:.1f}, {max_y:.1f})")
print(f"Original size: {max_x - min_x:.1f} x {max_y - min_y:.1f}\n")

# Create large canvas (3000x3000 pixels)
canvas_size = 3000
img_debug = np.ones((canvas_size, canvas_size, 3), dtype=np.uint8) * 255

# Scale factor to use most of the canvas
margin = 200
available_size = canvas_size - 2 * margin
scale_x = available_size / (max_x - min_x) if (max_x - min_x) > 0 else 1
scale_y = available_size / (max_y - min_y) if (max_y - min_y) > 0 else 1
scale = min(scale_x, scale_y) * 0.8  # Use 80% to have some margin

print(f"Scale factor: {scale:.2f}")
print(f"Canvas size: {canvas_size}x{canvas_size}\n")

# Transform and draw joints
print("Joint mapping (index: name -> scaled position):")
print("=" * 70)

for idx, joint_idx in enumerate(valid_joints):
    # Original position
    orig_pt = valid_positions[idx]

    # Normalize to 0-1, then scale to canvas
    norm_x = (orig_pt[0] - min_x) / (max_x - min_x) if (max_x - min_x) > 0 else 0.5
    norm_y = (orig_pt[1] - min_y) / (max_y - min_y) if (max_y - min_y) > 0 else 0.5

    # Scale to canvas with margin
    scaled_x = int(margin + norm_x * available_size)
    scaled_y = int(margin + norm_y * available_size)

    pt = (scaled_x, scaled_y)

    # Get joint name
    joint_name = JOINT_NAMES[joint_idx] if joint_idx < len(JOINT_NAMES) else f"joint{joint_idx}"

    print(f"{joint_idx:2d}: {joint_name:20s} -> ({scaled_x:4d}, {scaled_y:4d})")

    # Draw large circle
    cv2.circle(img_debug, pt, 40, (100, 100, 255), -1)  # Filled red circle
    cv2.circle(img_debug, pt, 40, (0, 0, 0), 3)  # Black border

    # Draw joint number (white, bold)
    cv2.putText(
        img_debug,
        str(joint_idx),
        (pt[0] - 15, pt[1] + 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (255, 255, 255),
        3
    )

    # Draw joint name below the circle (black)
    text_size = cv2.getTextSize(joint_name, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
    text_x = pt[0] - text_size[0] // 2
    text_y = pt[1] + 60

    cv2.putText(
        img_debug,
        joint_name,
        (text_x, text_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 0, 0),
        2
    )

# Add title
title = f"{sequence_name} - Frame {frame_idx} - Subject {subject_idx}"
cv2.putText(
    img_debug,
    title,
    (50, 80),
    cv2.FONT_HERSHEY_SIMPLEX,
    1.5,
    (0, 0, 0),
    3
)

# Add instructions
instructions = [
    "Each circle shows: [Joint Index] and joint name",
    "Verify the joint names match the skeleton structure",
    "Report any mismatches to correct the mapping"
]

for i, instruction in enumerate(instructions):
    cv2.putText(
        img_debug,
        instruction,
        (50, 130 + i * 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 0, 255),
        2
    )

# Save image
output_path = './joint_mapping_debug.jpg'
cv2.imwrite(output_path, img_debug)

print("\n" + "=" * 70)
print(f"✓ Saved visualization to {output_path}")
print("\nNow check the image and tell me which joints are incorrectly named!")
print("\nFor example, if circle '24' (currently labeled 'nose') is actually")
print("at the ankle position, report: 'Joint 24 should be right_ankle, not nose'")
