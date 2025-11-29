#!/usr/bin/env python3
"""
Analyze video FPS and duration for FIFA Skeletal Tracking dataset.

This script examines video files to determine their actual frame rate,
duration, and frame count to help fix synchronization issues with
pose animations.
"""

import cv2
import numpy as np
from pathlib import Path
import argparse
import sys

def analyze_video(video_path):
    """Analyze a single video file for FPS, duration, and frame count."""
    cap = cv2.VideoCapture(str(video_path))
    
    if not cap.isOpened():
        return None
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Calculate duration
    duration = frame_count / fps if fps > 0 else 0
    
    cap.release()
    
    return {
        'fps': fps,
        'frame_count': frame_count,
        'duration': duration,
        'width': width,
        'height': height
    }

def analyze_dataset_videos(data_dir, subset=None, sample_size=5):
    """Analyze videos from the dataset."""
    data_path = Path(data_dir)
    videos_dir = data_path / 'videos'
    
    results = {}
    
    # Define subsets to analyze
    subsets = ['train_data', 'test_data', 'challenge_data']
    if subset:
        subsets = [subset]
    
    for subset_name in subsets:
        subset_dir = videos_dir / subset_name
        if not subset_dir.exists():
            print(f"Warning: {subset_dir} does not exist")
            continue
            
        video_files = list(subset_dir.glob('*.mp4'))
        if not video_files:
            print(f"Warning: No MP4 files found in {subset_dir}")
            continue
        
        print(f"\n=== Analyzing {subset_name.upper()} ({len(video_files)} videos) ===")
        
        # Sample videos for analysis
        if sample_size and len(video_files) > sample_size:
            video_files = video_files[:sample_size]
            print(f"Sampling first {sample_size} videos for analysis")
        
        subset_results = []
        
        for video_file in video_files:
            print(f"Analyzing: {video_file.name}")
            
            video_info = analyze_video(video_file)
            if video_info:
                subset_results.append({
                    'filename': video_file.name,
                    'sequence': video_file.stem,
                    **video_info
                })
                
                print(f"  FPS: {video_info['fps']:.2f}")
                print(f"  Frames: {video_info['frame_count']}")
                print(f"  Duration: {video_info['duration']:.2f}s")
                print(f"  Resolution: {video_info['width']}x{video_info['height']}")
            else:
                print(f"  ERROR: Could not analyze {video_file.name}")
        
        results[subset_name] = subset_results
    
    return results

def print_summary(results):
    """Print summary statistics."""
    print("\n" + "="*60)
    print("SUMMARY STATISTICS")
    print("="*60)
    
    all_fps = []
    all_durations = []
    all_frame_counts = []
    
    for subset_name, subset_results in results.items():
        if not subset_results:
            continue
            
        fps_values = [r['fps'] for r in subset_results]
        durations = [r['duration'] for r in subset_results]
        frame_counts = [r['frame_count'] for r in subset_results]
        
        print(f"\n{subset_name.upper()}:")
        print(f"  Videos analyzed: {len(subset_results)}")
        print(f"  FPS - Min: {min(fps_values):.2f}, Max: {max(fps_values):.2f}, Mean: {np.mean(fps_values):.2f}")
        print(f"  Duration - Min: {min(durations):.1f}s, Max: {max(durations):.1f}s, Mean: {np.mean(durations):.1f}s")
        print(f"  Frames - Min: {min(frame_counts)}, Max: {max(frame_counts)}, Mean: {np.mean(frame_counts):.0f}")
        
        all_fps.extend(fps_values)
        all_durations.extend(durations)
        all_frame_counts.extend(frame_counts)
    
    if all_fps:
        print(f"\nOVERALL DATASET:")
        print(f"  Total videos analyzed: {len(all_fps)}")
        print(f"  FPS - Min: {min(all_fps):.2f}, Max: {max(all_fps):.2f}, Mean: {np.mean(all_fps):.2f}")
        print(f"  Duration - Min: {min(all_durations):.1f}s, Max: {max(all_durations):.1f}s, Mean: {np.mean(all_durations):.1f}s")
        print(f"  Frames - Min: {min(all_frame_counts)}, Max: {max(all_frame_counts)}, Mean: {np.mean(all_frame_counts):.0f}")
        
        # Check for FPS consistency
        unique_fps = set(round(fps, 1) for fps in all_fps)
        print(f"\nFPS CONSISTENCY:")
        print(f"  Unique FPS values: {sorted(unique_fps)}")
        if len(unique_fps) == 1:
            print(f"  ✓ All videos have consistent FPS: {list(unique_fps)[0]}")
        else:
            print(f"  ⚠ Multiple FPS values detected - may cause synchronization issues")

def check_pose_data_consistency(data_dir, sequence_name):
    """Check if pose data frame count matches video frame count."""
    data_path = Path(data_dir)
    
    # Find video file
    video_file = None
    for subset in ['train_data', 'test_data', 'challenge_data']:
        potential_path = data_path / 'videos' / subset / f'{sequence_name}.mp4'
        if potential_path.exists():
            video_file = potential_path
            break
    
    if not video_file:
        print(f"Video file not found for sequence: {sequence_name}")
        return
    
    # Analyze video
    video_info = analyze_video(video_file)
    if not video_info:
        print(f"Could not analyze video: {video_file}")
        return
    
    # Check pose data
    poses_file = data_path / 'poses' / f'{sequence_name}.npz'
    if poses_file.exists():
        poses_data = np.load(poses_file)
        # Correct: shape is (num_subjects, num_frames, dim)
        num_subjects = poses_data['global_orient'].shape[0]
        pose_frames = poses_data['global_orient'].shape[1]
        
        print(f"\nCONSISTENCY CHECK for {sequence_name}:")
        print(f"  Video frames: {video_info['frame_count']}")
        print(f"  Video FPS: {video_info['fps']:.2f}")
        print(f"  Video duration: {video_info['duration']:.2f}s")
        print(f"  Pose subjects: {num_subjects}")
        print(f"  Pose frames: {pose_frames}")
        print(f"  Frame difference: {abs(video_info['frame_count'] - pose_frames)}")
        
        if video_info['frame_count'] == pose_frames:
            print(f"  ✓ Frame counts match perfectly")
        else:
            print(f"  ⚠ Frame count mismatch detected")
            
        # Calculate what FPS would be needed for pose animation
        if pose_frames > 0:
            required_fps = pose_frames / video_info['duration']
            print(f"  Required animation FPS for sync: {required_fps:.2f}")
    else:
        print(f"Pose data not found: {poses_file}")

def main():
    parser = argparse.ArgumentParser(description='Analyze video FPS and duration')
    parser.add_argument('--data-dir', default='data', help='Data directory path')
    parser.add_argument('--subset', choices=['train_data', 'test_data', 'challenge_data'], 
                       help='Analyze specific subset only')
    parser.add_argument('--sample-size', type=int, default=5, 
                       help='Number of videos to sample per subset (0 for all)')
    parser.add_argument('--check-sequence', type=str,
                       help='Check specific sequence for video/pose consistency')
    parser.add_argument('--full-analysis', action='store_true',
                       help='Analyze all videos (overrides sample-size)')
    
    args = parser.parse_args()
    
    if args.full_analysis:
        args.sample_size = 0
    
    print("FIFA Skeletal Tracking Dataset - Video Analysis")
    print("=" * 50)
    
    if args.check_sequence:
        check_pose_data_consistency(args.data_dir, args.check_sequence)
    else:
        results = analyze_dataset_videos(args.data_dir, args.subset, args.sample_size)
        print_summary(results)
        
        # Provide recommendations
        print("\n" + "="*60)
        print("RECOMMENDATIONS")
        print("="*60)
        
        all_fps = []
        for subset_results in results.values():
            all_fps.extend([r['fps'] for r in subset_results])
        
        if all_fps:
            mean_fps = np.mean(all_fps)
            print(f"\nFor pose animation synchronization:")
            print(f"  Recommended default FPS: {mean_fps:.0f}")
            print(f"  Current visualization default: 50 fps")
            
            if abs(mean_fps - 50) > 1:
                print(f"  ⚠ FPS mismatch detected! Update visualization scripts.")
            else:
                print(f"  ✓ Current default FPS is appropriate.")

if __name__ == '__main__':
    main()