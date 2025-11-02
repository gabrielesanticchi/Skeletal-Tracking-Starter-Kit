import sys
from pathlib import Path
import cv2
import matplotlib.pyplot as plt

# Add src to path
sys.path.insert(0, str(Path.cwd() / 'src'))

from classes import VideoMetadata

# 1. Load complete video metadata
data_dir = Path('data')
video = VideoMetadata.load(
    data_dir,
    'MOR_POR_184724',
    load_poses=True,
    load_cameras=True,
    # load_bboxes=True,
    load_skel_2d=True,
    load_skel_3d=True
)

print(f"Loaded sequence: {video}")
print(f"Number of frames: {video.num_frames}")

# 2. Extract a specific frame
frame = video.get_frame(
    frame_idx=100,
    load_image=True,
    images_dir=data_dir / 'images'
)

# 3. Work with the frame data
# Access SMPL poses
poses = frame.get_poses_data()
print(f"Body pose shape: {poses['body_pose'].shape}")

# Access bounding boxes
if frame.bboxes is not None:
    boxes = frame.get_bboxes()
    print(f"Number of subjects: {len(boxes)}")

# Access 2D skeleton
if frame.skel_2d is not None:
    skel_2d = frame.get_skeleton_2d()
    print(f"2D skeleton shape: {skel_2d.shape}")

# Access 3D skeleton
if frame.skel_3d is not None:
    skel_3d = frame.get_skeleton_3d()
    print(f"3D skeleton shape: {skel_3d.shape}")

# 4. Visualize
# Visualize bounding boxes
if frame.bboxes is not None:
    img_boxes = frame.visualize_bboxes()
    cv2.imwrite('output_boxes.jpg', img_boxes)

# Visualize 2D skeleton
if frame.skel_2d is not None:
    img_skel_2d = frame.visualize_skeleton_2d()
    cv2.imwrite('output_skel_2d.jpg', img_skel_2d)

# Visualize 3D skeleton
if frame.skel_3d is not None:
    fig = frame.visualize_skeleton_3d(elev=20, azim=-60)
    fig.savefig('output_skel_3d.png')

# 5. Process all frames
for frame_idx in range(0, video.num_frames, 100):
    if frame.bboxes is not None:
        frame = video.get_frame(frame_idx)
        boxes = frame.get_bboxes()
        print(f"Frame {frame_idx}: {len(boxes)} subjects")
