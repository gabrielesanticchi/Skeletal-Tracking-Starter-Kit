"""
Skeleton visualization utilities.

Provides color mapping and visualization helpers for skeletal keypoints.
"""

import numpy as np
from typing import Dict, Tuple, List


# OpenPose BODY_25 Joint names (25 joints)
# This is the output format from 4D-Humans/PHALP after SMPL to OpenPose conversion
JOINT_NAMES = [
    'nose',            # 0
    'neck',            # 1
    'right_shoulder',  # 2
    'right_elbow',     # 3
    'right_wrist',     # 4
    'left_shoulder',   # 5
    'left_elbow',      # 6
    'left_wrist',      # 7
    'mid_hip',         # 8
    'right_hip',       # 9
    'right_knee',      # 10
    'right_ankle',     # 11
    'left_hip',        # 12
    'left_knee',       # 13
    'left_ankle',      # 14
    'right_eye',       # 15
    'left_eye',        # 16
    'right_ear',       # 17
    'left_ear',        # 18
    'left_big_toe',    # 19
    'left_small_toe',  # 20
    'left_heel',       # 21
    'right_big_toe',   # 22
    'right_small_toe', # 23
    'right_heel'       # 24
]

# Color mapping for joints (BGR format for OpenCV, RGB for matplotlib)
# Organized by body part for better visualization
JOINT_COLORS = {
    # Head/Face (Purple tones)
    'nose': (200, 150, 200),        # Light purple
    'right_eye': (180, 120, 180),   # Medium purple
    'left_eye': (180, 120, 180),    # Medium purple
    'right_ear': (150, 100, 150),   # Dark purple
    'left_ear': (150, 100, 150),    # Dark purple

    # Torso/Core (Blue tones)
    'neck': (255, 200, 100),        # Light blue
    'mid_hip': (200, 150, 50),      # Medium blue

    # Left side (Green tones)
    'left_shoulder': (100, 255, 100),   # Light green
    'left_elbow': (50, 200, 50),        # Medium green
    'left_wrist': (0, 150, 0),          # Dark green
    'left_hip': (100, 255, 100),        # Light green
    'left_knee': (50, 200, 50),         # Medium green
    'left_ankle': (0, 150, 0),          # Dark green
    'left_big_toe': (0, 100, 0),        # Darker green
    'left_small_toe': (0, 100, 0),      # Darker green
    'left_heel': (0, 120, 0),           # Dark green

    # Right side (Red tones)
    'right_shoulder': (100, 100, 255),  # Light red
    'right_elbow': (50, 50, 200),       # Medium red
    'right_wrist': (0, 0, 150),         # Dark red
    'right_hip': (100, 100, 255),       # Light red
    'right_knee': (50, 50, 200),        # Medium red
    'right_ankle': (0, 0, 150),         # Dark red
    'right_big_toe': (0, 0, 100),       # Darker red
    'right_small_toe': (0, 0, 100),     # Darker red
    'right_heel': (0, 0, 120),          # Dark red
}

# Skeleton connections (OpenPose BODY_25 topology)
SKELETON_CONNECTIONS = [
    # Head/Face
    (0, 1),                         # nose to neck
    (0, 15), (0, 16),              # nose to eyes
    (15, 17), (16, 18),            # eyes to ears

    # Torso
    (1, 8),                        # neck to mid_hip

    # Left arm
    (1, 5),                        # neck to left_shoulder
    (5, 6), (6, 7),                # left_shoulder -> left_elbow -> left_wrist

    # Right arm
    (1, 2),                        # neck to right_shoulder
    (2, 3), (3, 4),                # right_shoulder -> right_elbow -> right_wrist

    # Left leg
    (8, 12),                       # mid_hip to left_hip
    (12, 13), (13, 14),            # left_hip -> left_knee -> left_ankle

    # Right leg
    (8, 9),                        # mid_hip to right_hip
    (9, 10), (10, 11),             # right_hip -> right_knee -> right_ankle

    # Left foot
    (14, 19), (14, 21),            # left_ankle to left_big_toe and left_heel
    (19, 20),                      # left_big_toe to left_small_toe

    # Right foot
    (11, 22), (11, 24),            # right_ankle to right_big_toe and right_heel
    (22, 23),                      # right_big_toe to right_small_toe
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
            'Head/Face': (0.78, 0.59, 0.78),     # Purple
            'Torso/Core': (0.39, 0.59, 1.0),     # Blue
            'Left Side': (0.0, 1.0, 0.0),        # Green
            'Right Side': (1.0, 0.0, 0.0),       # Red
        }
