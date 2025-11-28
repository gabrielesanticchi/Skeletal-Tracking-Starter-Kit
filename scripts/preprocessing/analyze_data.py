#!/usr/bin/env python3
import numpy as np
from pathlib import Path

def analyze_boxes():
    """Analyze the boxes.npz file to see which sequences have bounding boxes."""
    boxes_path = Path("results/boxes.npz")
    if not boxes_path.exists():
        # Try data/boxes.npz
        boxes_path = Path("data/boxes.npz")
    
    if not boxes_path.exists():
        print("No boxes.npz file found!")
        return
    
    print(f"Loading boxes from: {boxes_path}")
    boxes_data = np.load(boxes_path)
    
    print(f"\nSequences with bounding boxes ({len(boxes_data.files)}):")
    for i, seq_name in enumerate(sorted(boxes_data.files), 1):
        shape = boxes_data[seq_name].shape
        print(f"{i:2d}. {seq_name} - Shape: {shape}")
    
    return sorted(boxes_data.files)

def analyze_poses():
    """Analyze poses directory."""
    poses_dir = Path("data/poses")
    if not poses_dir.exists():
        print("No poses directory found!")
        return []
    
    pose_files = list(poses_dir.glob("*.npz"))
    print(f"\nPose files ({len(pose_files)}):")
    pose_names = []
    for i, pose_file in enumerate(sorted(pose_files), 1):
        seq_name = pose_file.stem
        pose_names.append(seq_name)
        print(f"{i:2d}. {seq_name}")
    
    return sorted(pose_names)

def analyze_cameras():
    """Analyze cameras directory."""
    cameras_dir = Path("data/cameras")
    if not cameras_dir.exists():
        print("No cameras directory found!")
        return []
    
    camera_files = list(cameras_dir.glob("*.npz"))
    print(f"\nCamera files ({len(camera_files)}):")
    camera_names = []
    for i, camera_file in enumerate(sorted(camera_files), 1):
        seq_name = camera_file.stem
        camera_names.append(seq_name)
        print(f"{i:2d}. {seq_name}")
    
    return sorted(camera_names)

def analyze_videos():
    """Analyze video directories."""
    videos_dir = Path("data/videos")
    
    train_videos = []
    test_videos = []
    challenge_videos = []
    
    # Train data
    train_dir = videos_dir / "train_data"
    if train_dir.exists():
        train_files = list(train_dir.glob("*.mp4"))
        train_videos = [f.stem for f in train_files]
        print(f"\nTrain videos ({len(train_videos)}):")
        for i, name in enumerate(sorted(train_videos), 1):
            print(f"{i:2d}. {name}")
    
    # Test data
    test_dir = videos_dir / "test_data"
    if test_dir.exists():
        test_files = list(test_dir.glob("*.mp4"))
        test_videos = [f.stem for f in test_files]
        print(f"\nTest videos ({len(test_videos)}):")
        for i, name in enumerate(sorted(test_videos), 1):
            print(f"{i:2d}. {name}")
    
    # Challenge data
    challenge_dir = videos_dir / "challenge_data"
    if challenge_dir.exists():
        challenge_files = list(challenge_dir.glob("*.mp4"))
        challenge_videos = [f.stem for f in challenge_files]
        print(f"\nChallenge videos ({len(challenge_videos)}):")
        for i, name in enumerate(sorted(challenge_videos), 1):
            print(f"{i:2d}. {name}")
    
    return train_videos, test_videos, challenge_videos

if __name__ == "__main__":
    print("=== DATASET ANALYSIS ===")
    
    # Analyze all components
    train_videos, test_videos, challenge_videos = analyze_videos()
    camera_names = analyze_cameras()
    pose_names = analyze_poses()
    box_names = analyze_boxes()
    
    print("\n=== SUMMARY ===")
    print(f"Total videos: {len(train_videos) + len(test_videos) + len(challenge_videos)}")
    print(f"  - Train: {len(train_videos)}")
    print(f"  - Test: {len(test_videos)}")
    print(f"  - Challenge: {len(challenge_videos)}")
    print(f"Total cameras: {len(camera_names)}")
    print(f"Total poses: {len(pose_names)}")
    print(f"Total boxes: {len(box_names)}")
    
    # Find missing camera files
    all_videos = set(train_videos + test_videos + challenge_videos)
    camera_set = set(camera_names)
    missing_cameras = all_videos - camera_set
    
    print(f"\nMissing camera files ({len(missing_cameras)}):")
    for name in sorted(missing_cameras):
        if name in test_videos:
            print(f"  - {name} (test_data)")
        elif name in challenge_videos:
            print(f"  - {name} (challenge_data)")
        else:
            print(f"  - {name} (unknown)")
    
    # Check which sequences have boxes
    test_with_boxes = [name for name in test_videos if name in box_names]
    challenge_with_boxes = [name for name in challenge_videos if name in box_names]
    
    print(f"\nTest sequences with boxes ({len(test_with_boxes)}):")
    for name in sorted(test_with_boxes):
        print(f"  - {name}")
    
    print(f"\nChallenge sequences with boxes ({len(challenge_with_boxes)}):")
    for name in sorted(challenge_with_boxes):
        print(f"  - {name}")
    
    print(f"\nTest sequences WITHOUT boxes ({len(test_videos) - len(test_with_boxes)}):")
    for name in sorted(set(test_videos) - set(test_with_boxes)):
        print(f"  - {name}")
    
    print(f"\nChallenge sequences WITHOUT boxes ({len(challenge_videos) - len(challenge_with_boxes)}):")
    for name in sorted(set(challenge_videos) - set(challenge_with_boxes)):
        print(f"  - {name}")