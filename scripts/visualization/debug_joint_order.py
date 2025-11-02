"""
Debug script to verify joint ordering by visualizing each joint individually.
"""

import sys
import cv2
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

from classes import Skeleton2DData, ImageMetadata

# Load data
data_dir = Path('data')
skel_2d_dict = Skeleton2DData.load_all(data_dir / 'skel_2d.npz')
sequence_name = list(skel_2d_dict.keys())[0]
skel_2d = skel_2d_dict[sequence_name]

frame_idx = 100
frame_meta = ImageMetadata(sequence_name=sequence_name, frame_idx=frame_idx, skel_2d=skel_2d)
image = frame_meta.load_image(data_dir / 'images')

# Get keypoints for first subject
kpts = skel_2d.get_frame_keypoints(frame_idx, subject_idx=0)

print(f"Sequence: {sequence_name}, Frame: {frame_idx}")
print(f"Keypoints shape: {kpts.shape}")
print("\nJoint positions:")

from utils.skeleton_viz import JOINT_NAMES

for joint_idx in range(len(kpts)):
    pt = kpts[joint_idx]
    if not np.any(np.isnan(pt)) and not np.all(pt == 0):
        joint_name = JOINT_NAMES[joint_idx] if joint_idx < len(JOINT_NAMES) else f"joint{joint_idx}"
        print(f"{joint_idx:2d}: {joint_name:20s} at ({pt[0]:7.1f}, {pt[1]:7.1f})")

# Create visualization with numbers
img_debug = image.copy()
for joint_idx in range(len(kpts)):
    pt_coords = kpts[joint_idx]
    if np.any(np.isnan(pt_coords)) or np.all(pt_coords == 0):
        continue

    pt = tuple(pt_coords.astype(int))

    # Draw large circle with joint number
    cv2.circle(img_debug, pt, 20, (0, 0, 255), -1)
    cv2.circle(img_debug, pt, 20, (255, 255, 255), 2)

    # Draw joint number
    cv2.putText(
        img_debug,
        str(joint_idx),
        (pt[0] - 10, pt[1] + 5),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2
    )

cv2.imwrite('/tmp/joint_numbers.jpg', img_debug)
print("\n✓ Saved visualization with joint numbers to /tmp/joint_numbers.jpg")
print("Check this image to verify which joint is which!")
