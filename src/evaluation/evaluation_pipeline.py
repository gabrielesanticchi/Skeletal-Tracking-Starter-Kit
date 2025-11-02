"""
Evaluation pipeline for object detection and tracking.

This module provides a flexible, plug-in based pipeline for evaluating
object detection and tracking algorithms on video sequences.
"""

import numpy as np
import cv2
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from tqdm import tqdm
import yaml
import importlib
import sys

from evaluation.base_detector import BaseDetector
from evaluation.base_tracker import BaseTracker


class EvaluationPipeline:
    """
    Main evaluation pipeline for object detection and tracking.

    Supports plug-in architecture for different detectors and trackers.
    """

    def __init__(self, config_path: Optional[Path] = None, config_dict: Optional[Dict] = None):
        """
        Initialize evaluation pipeline.

        Args:
            config_path: Path to YAML configuration file
            config_dict: Configuration dictionary (if not using file)
        """
        if config_path:
            with open(config_path, 'r') as f:
                self.config = yaml.safe_load(f)
        elif config_dict:
            self.config = config_dict
        else:
            raise ValueError("Must provide either config_path or config_dict")

        self.detector = None
        self.tracker = None
        self.results = {}

    def load_detector(self) -> None:
        """Load detector based on configuration."""
        detector_config = self.config.get('detector', {})
        detector_type = detector_config.get('type', 'yolo')
        detector_module_path = detector_config.get('module', f'detectors.{detector_type}_detector')

        # Dynamically import detector module
        try:
            module = importlib.import_module(detector_module_path)
            detector_class_name = detector_config.get('class', f'{detector_type.upper()}Detector')
            detector_class = getattr(module, detector_class_name)

            # Initialize detector
            self.detector = detector_class(detector_config)
            self.detector.load_model(
                detector_config.get('model_path', None)
            )

            print(f"✓ Loaded detector: {self.detector.get_name()}")

        except Exception as e:
            raise ImportError(f"Failed to load detector '{detector_type}': {e}")

    def load_tracker(self) -> None:
        """Load tracker based on configuration (optional)."""
        tracker_config = self.config.get('tracker', None)

        if tracker_config is None or not tracker_config.get('enabled', False):
            print("✓ No tracker enabled")
            return

        tracker_type = tracker_config.get('type', 'bytetrack')
        tracker_module_path = tracker_config.get('module', f'trackers.{tracker_type}_tracker')

        # Dynamically import tracker module
        try:
            module = importlib.import_module(tracker_module_path)
            tracker_class_name = tracker_config.get('class', f'{tracker_type.capitalize()}Tracker')
            tracker_class = getattr(module, tracker_class_name)

            # Initialize tracker
            self.tracker = tracker_class(tracker_config)

            print(f"✓ Loaded tracker: {self.tracker.get_name()}")

        except Exception as e:
            raise ImportError(f"Failed to load tracker '{tracker_type}': {e}")

    def load_images(self, image_dir: Path) -> List[np.ndarray]:
        """
        Load images from directory.

        Args:
            image_dir: Path to directory containing images

        Returns:
            List of images (BGR format)
        """
        image_paths = sorted(image_dir.glob("*.jpg")) + sorted(image_dir.glob("*.png"))

        if not image_paths:
            raise FileNotFoundError(f"No images found in {image_dir}")

        images = []
        for img_path in tqdm(image_paths, desc="Loading images"):
            img = cv2.imread(str(img_path))
            if img is not None:
                images.append(img)

        return images

    def run_detection(
        self,
        images: List[np.ndarray],
        sequence_name: str
    ) -> Dict[str, np.ndarray]:
        """
        Run detection on all images.

        Args:
            images: List of images
            sequence_name: Name of the sequence

        Returns:
            Dictionary containing:
                - 'boxes': (num_frames, max_detections, 4) array
                - 'scores': (num_frames, max_detections) array
                - 'classes': (num_frames, max_detections) array
        """
        if self.detector is None:
            raise ValueError("Detector not loaded. Call load_detector() first.")

        all_boxes = []
        all_scores = []
        all_classes = []

        for img in tqdm(images, desc=f"Detecting {sequence_name}"):
            boxes, scores, classes = self.detector.detect(img)
            all_boxes.append(boxes)
            all_scores.append(scores)
            all_classes.append(classes)

        # Convert to fixed-size arrays (pad with NaN)
        max_detections = max(len(b) for b in all_boxes) if all_boxes else 0
        num_frames = len(images)

        boxes_array = np.full((num_frames, max_detections, 4), np.nan, dtype=np.float32)
        scores_array = np.full((num_frames, max_detections), np.nan, dtype=np.float32)
        classes_array = np.full((num_frames, max_detections), np.nan, dtype=np.float32)

        for i, (boxes, scores, classes) in enumerate(zip(all_boxes, all_scores, all_classes)):
            n = len(boxes)
            if n > 0:
                boxes_array[i, :n] = boxes
                scores_array[i, :n] = scores
                classes_array[i, :n] = classes

        return {
            'boxes': boxes_array,
            'scores': scores_array,
            'classes': classes_array
        }

    def run_tracking(
        self,
        detection_results: Dict[str, np.ndarray],
        sequence_name: str
    ) -> Dict[str, np.ndarray]:
        """
        Run tracking on detection results.

        Args:
            detection_results: Detection results from run_detection()
            sequence_name: Name of the sequence

        Returns:
            Dictionary containing:
                - 'tracks': (num_frames, max_tracks, 5) array [x1, y1, x2, y2, track_id]
        """
        if self.tracker is None:
            print("⚠️  No tracker enabled, skipping tracking")
            return {}

        self.tracker.reset()

        all_tracks = []
        boxes_array = detection_results['boxes']
        scores_array = detection_results['scores']
        classes_array = detection_results['classes']

        for frame_idx in tqdm(range(len(boxes_array)), desc=f"Tracking {sequence_name}"):
            # Get valid detections for this frame
            valid_mask = ~np.isnan(boxes_array[frame_idx, :, 0])
            boxes = boxes_array[frame_idx][valid_mask]
            scores = scores_array[frame_idx][valid_mask]
            classes = classes_array[frame_idx][valid_mask]

            # Update tracker
            tracks = self.tracker.update(boxes, scores, classes)
            all_tracks.append(tracks)

        # Convert to fixed-size arrays
        max_tracks = max(len(t) for t in all_tracks) if all_tracks else 0
        num_frames = len(boxes_array)

        tracks_array = np.full((num_frames, max_tracks, 5), np.nan, dtype=np.float32)

        for i, tracks in enumerate(all_tracks):
            n = len(tracks)
            if n > 0:
                tracks_array[i, :n] = tracks

        return {'tracks': tracks_array}

    def visualize_results(
        self,
        images: List[np.ndarray],
        detection_results: Dict[str, np.ndarray],
        tracking_results: Optional[Dict[str, np.ndarray]] = None,
        output_path: Optional[Path] = None,
        show_detections: bool = True,
        show_tracks: bool = True
    ) -> Optional[List[np.ndarray]]:
        """
        Visualize detection and tracking results.

        Args:
            images: List of input images
            detection_results: Detection results
            tracking_results: Tracking results (optional)
            output_path: Path to save output video (optional)
            show_detections: Whether to show detections
            show_tracks: Whether to show tracks

        Returns:
            List of annotated images (if output_path is None)
        """
        annotated_images = []
        boxes_array = detection_results['boxes']
        scores_array = detection_results['scores']

        tracks_array = tracking_results.get('tracks', None) if tracking_results else None

        for frame_idx, img in enumerate(tqdm(images, desc="Visualizing")):
            img_vis = img.copy()

            # Draw detections
            if show_detections:
                valid_mask = ~np.isnan(boxes_array[frame_idx, :, 0])
                boxes = boxes_array[frame_idx][valid_mask]
                scores = scores_array[frame_idx][valid_mask]

                for box, score in zip(boxes, scores):
                    x1, y1, x2, y2 = box.astype(int)
                    cv2.rectangle(img_vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(
                        img_vis,
                        f"{score:.2f}",
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 0),
                        2
                    )

            # Draw tracks
            if show_tracks and tracks_array is not None:
                valid_mask = ~np.isnan(tracks_array[frame_idx, :, 0])
                tracks = tracks_array[frame_idx][valid_mask]

                for track in tracks:
                    x1, y1, x2, y2, track_id = track.astype(int)
                    color = tuple(int(c) for c in plt.cm.tab10(track_id % 10)[:3] * 255)
                    cv2.rectangle(img_vis, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(
                        img_vis,
                        f"ID: {track_id}",
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        color,
                        2
                    )

            annotated_images.append(img_vis)

        # Save video if output path provided
        if output_path:
            self.save_video(annotated_images, output_path)
            return None
        else:
            return annotated_images

    def save_video(self, images: List[np.ndarray], output_path: Path, fps: int = 25) -> None:
        """
        Save images as video.

        Args:
            images: List of images
            output_path: Path to save video
            fps: Frames per second
        """
        if not images:
            return

        height, width = images[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

        for img in tqdm(images, desc="Saving video"):
            out.write(img)

        out.release()
        print(f"✓ Saved video to {output_path}")

    def save_predictions(
        self,
        detection_results: Dict[str, np.ndarray],
        sequence_name: str,
        output_path: Path
    ) -> None:
        """
        Save detection predictions in boxes.npz format.

        Args:
            detection_results: Detection results
            sequence_name: Name of the sequence
            output_path: Path to save predictions
        """
        # Convert to boxes format (num_frames, num_subjects, 4)
        boxes_array = detection_results['boxes']

        predictions = {sequence_name: boxes_array}

        np.savez_compressed(output_path, **predictions)
        print(f"✓ Saved predictions to {output_path}")

    def run(
        self,
        image_dir: Path,
        sequence_name: str,
        output_dir: Optional[Path] = None,
        save_video: bool = True,
        save_predictions: bool = True
    ) -> Dict:
        """
        Run complete evaluation pipeline.

        Args:
            image_dir: Directory containing input images
            sequence_name: Name of the sequence
            output_dir: Directory to save outputs
            save_video: Whether to save annotated video
            save_predictions: Whether to save prediction file

        Returns:
            Dictionary containing detection and tracking results
        """
        print("\n" + "="*80)
        print(f"EVALUATION PIPELINE - {sequence_name}")
        print("="*80 + "\n")

        # Load components
        self.load_detector()
        self.load_tracker()

        # Load images
        print(f"\n📂 Loading images from {image_dir}")
        images = self.load_images(image_dir)
        print(f"✓ Loaded {len(images)} images")

        # Run detection
        print(f"\n🔍 Running detection...")
        detection_results = self.run_detection(images, sequence_name)
        print(f"✓ Detection complete")

        # Run tracking (if enabled)
        tracking_results = {}
        if self.tracker:
            print(f"\n🎯 Running tracking...")
            tracking_results = self.run_tracking(detection_results, sequence_name)
            print(f"✓ Tracking complete")

        # Save results
        if output_dir:
            output_dir.mkdir(parents=True, exist_ok=True)

            if save_predictions:
                pred_path = output_dir / f"{sequence_name}_predictions.npz"
                self.save_predictions(detection_results, sequence_name, pred_path)

            if save_video:
                video_path = output_dir / f"{sequence_name}_results.mp4"
                self.visualize_results(
                    images,
                    detection_results,
                    tracking_results,
                    output_path=video_path
                )

        print("\n" + "="*80)
        print("PIPELINE COMPLETE")
        print("="*80 + "\n")

        return {
            'detection': detection_results,
            'tracking': tracking_results
        }
