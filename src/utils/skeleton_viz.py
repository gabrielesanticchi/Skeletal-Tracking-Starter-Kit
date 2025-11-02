"""
Skeleton visualization utilities.

Provides color mapping and visualization helpers for skeletal keypoints.
"""

import numpy as np
from typing import Dict, Tuple, List


# SMPL Joint names (25 joints: 24 SMPL + nose)
JOINT_NAMES = [
    'pelvis',          # 0
    'left_hip',        # 1
    'right_hip',       # 2
    'spine1',          # 3
    'left_knee',       # 4
    'right_knee',      # 5
    'spine2',          # 6
    'left_ankle',      # 7
    'right_ankle',     # 8
    'spine3',          # 9
    'left_foot',       # 10
    'right_foot',      # 11
    'neck',            # 12
    'left_collar',     # 13
    'right_collar',    # 14
    'head',            # 15
    'left_shoulder',   # 16
    'right_shoulder',  # 17
    'left_elbow',      # 18
    'right_elbow',     # 19
    'left_wrist',      # 20
    'right_wrist',     # 21
    'left_hand',       # 22
    'right_hand',      # 23
    'nose'             # 24 (extra joint from 4D-Humans)
]

# Color mapping for joints (BGR format for OpenCV, RGB for matplotlib)
# Organized by body part for better visualization
JOINT_COLORS = {
    # Torso/Spine (Blue tones)
    'pelvis': (255, 200, 100),      # Light blue
    'spine1': (255, 150, 50),       # Medium blue
    'spine2': (255, 100, 0),        # Dark blue
    'spine3': (200, 100, 0),        # Darker blue
    'neck': (150, 50, 0),           # Navy
    'head': (100, 0, 0),            # Dark navy
    'nose': (150, 100, 100),        # Purple-ish

    # Left side (Green tones)
    'left_hip': (100, 255, 100),    # Light green
    'left_knee': (50, 200, 50),     # Medium green
    'left_ankle': (0, 150, 0),      # Dark green
    'left_foot': (0, 100, 0),       # Darker green
    'left_collar': (150, 255, 150), # Very light green
    'left_shoulder': (100, 255, 100), # Light green
    'left_elbow': (50, 200, 50),    # Medium green
    'left_wrist': (0, 150, 0),      # Dark green
    'left_hand': (0, 100, 0),       # Darker green

    # Right side (Red tones)
    'right_hip': (100, 100, 255),   # Light red
    'right_knee': (50, 50, 200),    # Medium red
    'right_ankle': (0, 0, 150),     # Dark red
    'right_foot': (0, 0, 100),      # Darker red
    'right_collar': (150, 150, 255), # Very light red
    'right_shoulder': (100, 100, 255), # Light red
    'right_elbow': (50, 50, 200),   # Medium red
    'right_wrist': (0, 0, 150),     # Dark red
    'right_hand': (0, 0, 100),      # Darker red
}

# Skeleton connections
SKELETON_CONNECTIONS = [
    # Spine
    (0, 1), (0, 2), (0, 3),         # pelvis to hips and spine1
    (3, 6), (6, 9), (9, 12),        # spine chain
    (12, 15),                       # neck to head
    (12, 24),                       # neck to nose

    # Left leg
    (1, 4), (4, 7), (7, 10),        # hip -> knee -> ankle -> foot

    # Right leg
    (2, 5), (5, 8), (8, 11),        # hip -> knee -> ankle -> foot

    # Left arm
    (9, 13), (13, 16),              # spine3 -> collar -> shoulder
    (16, 18), (18, 20), (20, 22),   # shoulder -> elbow -> wrist -> hand

    # Right arm
    (9, 14), (14, 17),              # spine3 -> collar -> shoulder
    (17, 19), (19, 21), (21, 23),   # shoulder -> elbow -> wrist -> hand
]


class SkeletonColorMapper:
    """
    Utility class for mapping skeleton joints to colors.

    Provides consistent color mapping across 2D and 3D visualizations.
    """

    def __init__(self):
        """Initialize the color mapper."""
        self.joint_names = JOINT_NAMES
        self.joint_colors = JOINT_COLORS
        self.connections = SKELETON_CONNECTIONS

    def get_joint_color(self, joint_idx: int, format: str = 'bgr') -> Tuple[int, int, int]:
        """
        Get color for a specific joint index.

        Args:
            joint_idx: Joint index (0-24)
            format: Color format - 'bgr' for OpenCV, 'rgb' for matplotlib

        Returns:
            Color tuple (B, G, R) or (R, G, B)
        """
        if joint_idx < 0 or joint_idx >= len(self.joint_names):
            return (128, 128, 128)  # Gray for invalid

        joint_name = self.joint_names[joint_idx]
        color = self.joint_colors.get(joint_name, (128, 128, 128))

        if format == 'rgb':
            # Convert BGR to RGB
            return (color[2], color[1], color[0])
        return color

    def get_joint_color_normalized(self, joint_idx: int, format: str = 'rgb') -> Tuple[float, float, float]:
        """
        Get normalized color (0-1) for matplotlib.

        Args:
            joint_idx: Joint index (0-24)
            format: Color format - 'rgb' or 'bgr'

        Returns:
            Normalized color tuple
        """
        color = self.get_joint_color(joint_idx, format)
        return tuple(c / 255.0 for c in color)

    def get_all_colors(self, format: str = 'bgr') -> List[Tuple[int, int, int]]:
        """
        Get colors for all joints.

        Args:
            format: Color format - 'bgr' or 'rgb'

        Returns:
            List of color tuples
        """
        return [self.get_joint_color(i, format) for i in range(len(self.joint_names))]

    def get_all_colors_normalized(self, format: str = 'rgb') -> List[Tuple[float, float, float]]:
        """
        Get normalized colors for all joints.

        Args:
            format: Color format - 'rgb' or 'bgr'

        Returns:
            List of normalized color tuples
        """
        return [self.get_joint_color_normalized(i, format) for i in range(len(self.joint_names))]

    def get_connection_color(
        self,
        connection: Tuple[int, int],
        format: str = 'bgr'
    ) -> Tuple[int, int, int]:
        """
        Get color for a skeleton connection (average of endpoint colors).

        Args:
            connection: Tuple of (joint1_idx, joint2_idx)
            format: Color format - 'bgr' or 'rgb'

        Returns:
            Color tuple
        """
        color1 = np.array(self.get_joint_color(connection[0], format))
        color2 = np.array(self.get_joint_color(connection[1], format))
        avg_color = ((color1 + color2) / 2).astype(int)
        return tuple(avg_color)

    def get_joint_name(self, joint_idx: int) -> str:
        """Get joint name for a given index."""
        if joint_idx < 0 or joint_idx >= len(self.joint_names):
            return f"unknown_{joint_idx}"
        return self.joint_names[joint_idx]

    def get_legend_labels(self) -> Dict[str, Tuple[float, float, float]]:
        """
        Get legend labels with colors for visualization.

        Returns:
            Dictionary mapping body parts to RGB colors (normalized)
        """
        return {
            'Torso/Spine': (0.5, 0.5, 1.0),      # Blue
            'Left Side': (0.0, 1.0, 0.0),        # Green
            'Right Side': (1.0, 0.0, 0.0),       # Red
            'Head': (0.6, 0.4, 0.6),             # Purple
        }
