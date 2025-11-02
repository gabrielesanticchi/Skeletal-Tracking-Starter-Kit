"""
ByteTrack tracker implementation.

ByteTrack is a simple, fast and strong multi-object tracker.
Paper: https://arxiv.org/abs/2110.06864
Code: https://github.com/ifzhang/ByteTrack (MIT License - commercial-friendly)

This is a simplified implementation based on the ByteTrack paper.
"""

import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import sys
from scipy.optimize import linear_sum_assignment

# Add parent directory to path for base class import
sys.path.append(str(Path(__file__).parent.parent))
from base_tracker import BaseTracker


class KalmanFilter:
    """
    Simple Kalman Filter for bounding box tracking.
    State: [x_center, y_center, area, aspect_ratio, vx, vy, va, var]
    """

    def __init__(self):
        """Initialize Kalman Filter."""
        # State dimension: 8 (4 positions + 4 velocities)
        # Measurement dimension: 4 (x, y, area, aspect_ratio)
        self.dt = 1.0  # Time step

        # State transition matrix
        self.F = np.eye(8, dtype=np.float32)
        for i in range(4):
            self.F[i, i+4] = self.dt

        # Measurement matrix
        self.H = np.eye(4, 8, dtype=np.float32)

        # Process noise covariance
        self.Q = np.eye(8, dtype=np.float32)
        self.Q[4:, 4:] *= 0.01  # Lower noise for velocities

        # Measurement noise covariance
        self.R = np.eye(4, dtype=np.float32) * 10

        # State covariance
        self.P = np.eye(8, dtype=np.float32) * 1000

        # State
        self.x = np.zeros(8, dtype=np.float32)

    def predict(self):
        """Predict next state."""
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.x[:4]

    def update(self, measurement: np.ndarray):
        """Update state with measurement."""
        # Kalman gain
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)

        # Update state
        y = measurement - self.H @ self.x  # Innovation
        self.x = self.x + K @ y

        # Update covariance
        I = np.eye(8)
        self.P = (I - K @ self.H) @ self.P

    def init_from_box(self, box: np.ndarray):
        """Initialize filter from bounding box [x1, y1, x2, y2]."""
        x1, y1, x2, y2 = box
        w = x2 - x1
        h = y2 - y1

        x_center = (x1 + x2) / 2
        y_center = (y1 + y2) / 2
        area = w * h
        aspect_ratio = w / max(h, 1e-6)

        self.x[:4] = [x_center, y_center, area, aspect_ratio]
        self.x[4:] = 0  # Zero velocity

    def get_box(self) -> np.ndarray:
        """Get current bounding box [x1, y1, x2, y2]."""
        x_center, y_center, area, aspect_ratio = self.x[:4]

        h = np.sqrt(area / max(aspect_ratio, 1e-6))
        w = aspect_ratio * h

        x1 = x_center - w / 2
        y1 = y_center - h / 2
        x2 = x_center + w / 2
        y2 = y_center + h / 2

        return np.array([x1, y1, x2, y2])


class Track:
    """Single object track."""

    def __init__(self, box: np.ndarray, score: float, track_id: int, class_id: Optional[int] = None):
        """
        Initialize track.

        Args:
            box: Bounding box [x1, y1, x2, y2]
            score: Detection confidence score
            track_id: Unique track ID
            class_id: Object class ID (optional)
        """
        self.track_id = track_id
        self.class_id = class_id
        self.score = score
        self.box = box
        self.age = 0
        self.hits = 1
        self.time_since_update = 0

        # Kalman filter
        self.kf = KalmanFilter()
        self.kf.init_from_box(box)

    def predict(self):
        """Predict next position."""
        self.age += 1
        self.time_since_update += 1
        predicted_state = self.kf.predict()
        self.box = self.kf.get_box()
        return self.box

    def update(self, box: np.ndarray, score: float):
        """Update track with new detection."""
        self.box = box
        self.score = score
        self.hits += 1
        self.time_since_update = 0

        # Convert box to measurement format
        x1, y1, x2, y2 = box
        w = x2 - x1
        h = y2 - y1
        x_center = (x1 + x2) / 2
        y_center = (y1 + y2) / 2
        area = w * h
        aspect_ratio = w / max(h, 1e-6)

        measurement = np.array([x_center, y_center, area, aspect_ratio])
        self.kf.update(measurement)


class ByteTrackTracker(BaseTracker):
    """
    ByteTrack: Simple, Fast and Strong Multi-Object Tracker.

    ByteTrack uses a combination of high and low confidence detections
    to achieve robust tracking.
    """

    def __init__(self, config: Dict):
        """
        Initialize ByteTrack tracker.

        Args:
            config: Configuration dictionary with keys:
                - track_thresh: High confidence threshold for track initialization (default: 0.5)
                - track_buffer: Number of frames to keep lost tracks (default: 30)
                - match_thresh: IoU threshold for matching (default: 0.8)
                - low_thresh: Low confidence threshold for second association (default: 0.1)
        """
        super().__init__(config)
        self.track_thresh = config.get('track_thresh', 0.5)
        self.track_buffer = config.get('track_buffer', 30)
        self.match_thresh = config.get('match_thresh', 0.8)
        self.low_thresh = config.get('low_thresh', 0.1)

        self.tracked_tracks = []  # Active tracks
        self.lost_tracks = []     # Lost tracks (can be recovered)
        self.removed_tracks = []  # Removed tracks
        self.frame_id = 0

    def update(
        self,
        boxes: np.ndarray,
        scores: np.ndarray,
        classes: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Update tracker with new detections.

        Args:
            boxes: (N, 4) array of bounding boxes in XYXY format
            scores: (N,) array of confidence scores
            classes: (N,) array of class IDs (optional)

        Returns:
            tracks: (M, 5) array of tracked boxes [x1, y1, x2, y2, track_id]
        """
        self.frame_id += 1

        # Separate high and low confidence detections
        high_mask = scores >= self.track_thresh
        low_mask = (scores >= self.low_thresh) & (scores < self.track_thresh)

        boxes_high = boxes[high_mask]
        scores_high = scores[high_mask]
        classes_high = classes[high_mask] if classes is not None else None

        boxes_low = boxes[low_mask]
        scores_low = scores[low_mask]
        classes_low = classes[low_mask] if classes is not None else None

        # Predict all tracks
        for track in self.tracked_tracks + self.lost_tracks:
            track.predict()

        # First association with high confidence detections
        unmatched_tracks, unmatched_dets = self._associate(
            self.tracked_tracks,
            boxes_high,
            scores_high,
            classes_high,
            self.match_thresh
        )

        # Second association with low confidence detections
        if len(boxes_low) > 0:
            lost_tracks_for_second = [self.tracked_tracks[i] for i in unmatched_tracks]
            unmatched_tracks_second, _ = self._associate(
                lost_tracks_for_second,
                boxes_low,
                scores_low,
                classes_low,
                self.match_thresh
            )

            # Update lost tracks based on second association
            for idx in unmatched_tracks_second:
                track = lost_tracks_for_second[idx]
                if track.time_since_update > self.track_buffer:
                    self.lost_tracks.append(track)
                    self.tracked_tracks.remove(track)

        # Handle unmatched tracked detections from first association
        for idx in unmatched_tracks:
            track = self.tracked_tracks[idx]
            if track.time_since_update > self.track_buffer:
                self.lost_tracks.append(track)
                self.tracked_tracks.remove(track)

        # Initialize new tracks with unmatched high confidence detections
        for idx in unmatched_dets:
            track = Track(
                boxes_high[idx],
                scores_high[idx],
                self.next_id,
                classes_high[idx] if classes_high is not None else None
            )
            self.tracked_tracks.append(track)
            self.next_id += 1

        # Remove old lost tracks
        self.lost_tracks = [
            t for t in self.lost_tracks
            if t.time_since_update <= self.track_buffer
        ]

        # Return active tracks
        output = []
        for track in self.tracked_tracks:
            if track.hits >= self.min_hits:
                output.append(np.concatenate([track.box, [track.track_id]]))

        return np.array(output) if output else np.array([]).reshape(0, 5)

    def _associate(
        self,
        tracks: List[Track],
        boxes: np.ndarray,
        scores: np.ndarray,
        classes: Optional[np.ndarray],
        threshold: float
    ) -> Tuple[List[int], List[int]]:
        """
        Associate detections to tracks using IoU.

        Returns:
            unmatched_tracks: List of track indices that were not matched
            unmatched_dets: List of detection indices that were not matched
        """
        if len(tracks) == 0 or len(boxes) == 0:
            return list(range(len(tracks))), list(range(len(boxes)))

        # Compute IoU cost matrix
        track_boxes = np.array([t.box for t in tracks])
        iou_matrix = self.compute_iou_batch(track_boxes, boxes)

        # Use Hungarian algorithm for optimal assignment
        row_ind, col_ind = linear_sum_assignment(-iou_matrix)

        # Filter matches by IoU threshold
        matches = []
        unmatched_tracks = list(range(len(tracks)))
        unmatched_dets = list(range(len(boxes)))

        for r, c in zip(row_ind, col_ind):
            if iou_matrix[r, c] >= threshold:
                matches.append((r, c))
                unmatched_tracks.remove(r)
                unmatched_dets.remove(c)

        # Update matched tracks
        for track_idx, det_idx in matches:
            tracks[track_idx].update(boxes[det_idx], scores[det_idx])

        return unmatched_tracks, unmatched_dets

    def reset(self):
        """Reset tracker state."""
        super().reset()
        self.tracked_tracks = []
        self.lost_tracks = []
        self.removed_tracks = []
        self.frame_id = 0
