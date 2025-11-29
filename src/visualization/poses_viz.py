"""
Visualization for SMPL poses data.

Provides class-based visualization interface for 3D pose visualization,
pitch tracking, and animation with proper forward kinematics.
"""

import numpy as np
from typing import Optional, Tuple
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation


class PosesVisualizer:
    """
    Visualizer for SMPL poses data.

    Provides methods for 3D pose visualization, pitch tracking, and animations.
    Shared properties (pitch dimensions, skeleton, colors) are stored as instance attributes.
    """

    def __init__(self):
        """Initialize visualizer with shared properties."""
        # FIFA pitch dimensions (meters)
        self.pitch_length = 105.0
        self.pitch_width = 68.0
        self.half_length = 52.5
        self.half_width = 34.0
        self.margin = 10.0

        # Penalty area dimensions
        self.penalty_length = 16.5
        self.penalty_width = 40.3
        self.center_circle_radius = 9.15

        # SMPL skeleton connections (24 joints)
        self.skeleton_connections = [
            (0, 1), (0, 2), (0, 3),  # pelvis connections
            (1, 4), (4, 7), (7, 10),  # left leg
            (2, 5), (5, 8), (8, 11),  # right leg
            (3, 6), (6, 9), (9, 12), (12, 15),  # spine to head
            (9, 13), (9, 14),  # collar bones
            (13, 16), (16, 18), (18, 20), (20, 22),  # left arm
            (14, 17), (17, 19), (19, 21), (21, 23),  # right arm
        ]

        # SMPL joint hierarchy
        self.joint_parents = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19, 20, 21]

        # SMPL base joint positions in T-pose (simplified template)
        self.base_joints = np.array([
            [0.0, 0.0, 0.0],      # 0: pelvis (root)
            [-0.1, -0.05, 0.0],   # 1: left_hip
            [0.1, -0.05, 0.0],    # 2: right_hip
            [0.0, 0.1, 0.0],      # 3: spine1
            [-0.1, -0.4, 0.0],    # 4: left_knee
            [0.1, -0.4, 0.0],     # 5: right_knee
            [0.0, 0.2, 0.0],      # 6: spine2
            [-0.1, -0.75, 0.0],   # 7: left_ankle
            [0.1, -0.75, 0.0],    # 8: right_ankle
            [0.0, 0.3, 0.0],      # 9: spine3
            [-0.1, -0.85, 0.1],   # 10: left_foot
            [0.1, -0.85, 0.1],    # 11: right_foot
            [0.0, 0.45, 0.0],     # 12: neck
            [-0.15, 0.35, 0.0],   # 13: left_collar
            [0.15, 0.35, 0.0],    # 14: right_collar
            [0.0, 0.55, 0.0],     # 15: head
            [-0.3, 0.3, 0.0],     # 16: left_shoulder
            [0.3, 0.3, 0.0],      # 17: right_shoulder
            [-0.5, 0.2, 0.0],     # 18: left_elbow
            [0.5, 0.2, 0.0],      # 19: right_elbow
            [-0.7, 0.15, 0.0],    # 20: left_wrist
            [0.7, 0.15, 0.0],     # 21: right_wrist
            [-0.8, 0.1, 0.0],     # 22: left_hand
            [0.8, 0.1, 0.0],      # 23: right_hand
        ])

        # Color palette for subjects
        self.colors = plt.cm.tab10(np.linspace(0, 1, 10))

    @staticmethod
    def rodrigues_rotation(axis_angle: np.ndarray) -> np.ndarray:
        """
        Convert axis-angle representation to rotation matrix using Rodrigues' formula.

        Args:
            axis_angle: Axis-angle vector (3,)

        Returns:
            Rotation matrix (3, 3)
        """
        if np.allclose(axis_angle, 0):
            return np.eye(3)

        angle = np.linalg.norm(axis_angle)
        if angle == 0:
            return np.eye(3)

        axis = axis_angle / angle
        cos_angle = np.cos(angle)
        sin_angle = np.sin(angle)

        # Rodrigues' rotation formula
        K = np.array([[0, -axis[2], axis[1]],
                     [axis[2], 0, -axis[0]],
                     [-axis[1], axis[0], 0]])

        R = np.eye(3) + sin_angle * K + (1 - cos_angle) * np.dot(K, K)
        return R

    def compute_smpl_joints(self, global_orient: np.ndarray, body_pose: np.ndarray,
                           transl: np.ndarray) -> np.ndarray:
        """
        Compute SMPL joint positions from pose parameters using forward kinematics.

        Properly propagates rotations through the joint hierarchy for realistic animations.

        Args:
            global_orient: Global orientation (3,) - axis-angle
            body_pose: Body pose parameters (69,) - 23 joints × 3
            transl: Translation vector (3,)

        Returns:
            Joint positions (24, 3) in world coordinates
        """
        # Handle NaN/Inf values
        if np.any(np.isnan(transl)) or np.any(np.isinf(transl)):
            transl = np.array([0.0, 0.0, 0.0])
        if np.any(np.isnan(global_orient)) or np.any(np.isinf(global_orient)):
            global_orient = np.array([0.0, 0.0, 0.0])
        if np.any(np.isnan(body_pose)) or np.any(np.isinf(body_pose)):
            body_pose = np.zeros(69)

        # Initialize joint transformations
        local_rotations = [np.eye(3) for _ in range(24)]

        # Apply global orientation to root
        local_rotations[0] = self.rodrigues_rotation(global_orient)

        # Apply body pose rotations (23 joints, 3 parameters each)
        for i in range(1, 24):
            pose_idx = (i - 1) * 3
            if pose_idx + 2 < len(body_pose):
                axis_angle = body_pose[pose_idx:pose_idx + 3]
                local_rotations[i] = self.rodrigues_rotation(axis_angle)

        # Forward kinematics: compute world positions
        world_joints = np.zeros((24, 3))
        world_rotations = [np.eye(3) for _ in range(24)]

        for i in range(24):
            parent = self.joint_parents[i]
            if parent == -1:  # Root joint (pelvis)
                world_rotations[i] = local_rotations[i]
                world_joints[i] = transl + np.dot(world_rotations[i], self.base_joints[i])
            else:
                # Accumulate rotation from parent
                world_rotations[i] = np.dot(world_rotations[parent], local_rotations[i])

                # Transform position: rotate base offset by parent's rotation and add to parent position
                local_offset = self.base_joints[i] - self.base_joints[parent]
                transformed_offset = np.dot(world_rotations[parent], local_offset)
                world_joints[i] = world_joints[parent] + transformed_offset

        return world_joints

    def draw_football_pitch(self, ax, show_3d: bool = False, margin: Optional[float] = None):
        """
        Draw a football pitch outline centered at (0,0).

        Args:
            ax: Matplotlib axes object (2D or 3D)
            show_3d: Whether to draw on 3D axes (as ground plane at Z=0)
            margin: Custom margin (uses default if None)
        """
        margin = margin if margin is not None else self.margin

        if show_3d:
            # Draw pitch as ground plane on 3D axes
            # Pitch outline
            pitch_x = [-self.half_length, self.half_length, self.half_length, -self.half_length, -self.half_length]
            pitch_y = [-self.half_width, -self.half_width, self.half_width, self.half_width, -self.half_width]
            pitch_z = [0, 0, 0, 0, 0]
            ax.plot(pitch_x, pitch_y, pitch_z, 'k-', linewidth=2, alpha=0.8)

            # Center line
            ax.plot([0, 0], [-self.half_width, self.half_width], [0, 0], 'k-', linewidth=1, alpha=0.6)

            # Center circle (approximate as polygon for 3D)
            theta = np.linspace(0, 2*np.pi, 50)
            circle_x = self.center_circle_radius * np.cos(theta)
            circle_y = self.center_circle_radius * np.sin(theta)
            circle_z = np.zeros_like(theta)
            ax.plot(circle_x, circle_y, circle_z, 'k-', linewidth=1, alpha=0.6)

            # Penalty areas
            # Left
            left_penalty_x = [-self.half_length, -self.half_length + self.penalty_length,
                             -self.half_length + self.penalty_length, -self.half_length,
                             -self.half_length]
            left_penalty_y = [-self.penalty_width/2, -self.penalty_width/2,
                             self.penalty_width/2, self.penalty_width/2,
                             -self.penalty_width/2]
            left_penalty_z = [0, 0, 0, 0, 0]
            ax.plot(left_penalty_x, left_penalty_y, left_penalty_z, 'k-', linewidth=1, alpha=0.6)

            # Right
            right_penalty_x = [self.half_length, self.half_length - self.penalty_length,
                              self.half_length - self.penalty_length, self.half_length,
                              self.half_length]
            right_penalty_y = [-self.penalty_width/2, -self.penalty_width/2,
                               self.penalty_width/2, self.penalty_width/2,
                               -self.penalty_width/2]
            right_penalty_z = [0, 0, 0, 0, 0]
            ax.plot(right_penalty_x, right_penalty_y, right_penalty_z, 'k-', linewidth=1, alpha=0.6)

            # Draw ground plane mesh
            x_min, x_max = -self.half_length - margin, self.half_length + margin
            y_min, y_max = -self.half_width - margin, self.half_width + margin
            xx, yy = np.meshgrid(np.linspace(x_min, x_max, 10),
                               np.linspace(y_min, y_max, 10))
            zz = np.zeros_like(xx)
            ax.plot_surface(xx, yy, zz, alpha=0.1, color='lightgreen',
                          linewidth=0, antialiased=True)
        else:
            # Draw pitch on 2D axes
            # Pitch outline
            pitch_x = [-self.half_length, self.half_length, self.half_length, -self.half_length, -self.half_length]
            pitch_y = [-self.half_width, -self.half_width, self.half_width, self.half_width, -self.half_width]
            ax.plot(pitch_x, pitch_y, 'k-', linewidth=2, alpha=0.8, label='Pitch Outline')

            # Center line
            ax.plot([0, 0], [-self.half_width, self.half_width], 'k-', linewidth=1, alpha=0.6)

            # Center circle
            center_circle = plt.Circle((0, 0), self.center_circle_radius,
                                     fill=False, color='black', linewidth=1, alpha=0.6)
            ax.add_patch(center_circle)

            # Penalty areas
            # Left
            left_penalty_x = [-self.half_length, -self.half_length + self.penalty_length,
                             -self.half_length + self.penalty_length, -self.half_length,
                             -self.half_length]
            left_penalty_y = [-self.penalty_width/2, -self.penalty_width/2,
                             self.penalty_width/2, self.penalty_width/2,
                             -self.penalty_width/2]
            ax.plot(left_penalty_x, left_penalty_y, 'k-', linewidth=1, alpha=0.6)

            # Right
            right_penalty_x = [self.half_length, self.half_length - self.penalty_length,
                              self.half_length - self.penalty_length, self.half_length,
                              self.half_length]
            right_penalty_y = [-self.penalty_width/2, -self.penalty_width/2,
                               self.penalty_width/2, self.penalty_width/2,
                               -self.penalty_width/2]
            ax.plot(right_penalty_x, right_penalty_y, 'k-', linewidth=1, alpha=0.6)

        # Set axis limits with margin, centered on (0,0)
        ax.set_xlim(-self.half_length - margin, self.half_length + margin)
        ax.set_ylim(-self.half_width - margin, self.half_width + margin)

    def visualize_3d_poses(self, poses_data, frame_idx: int, figsize: Tuple[int, int] = (12, 8),
                          elev: float = 20, azim: float = -60,
                          num_subjects: Optional[int] = None,
                          show_labels: bool = False,
                          show_pitch: bool = True) -> plt.Figure:
        """Visualize 3D poses for a specific frame."""
        if frame_idx < 0 or frame_idx >= poses_data.num_frames:
            raise ValueError(f"Frame index {frame_idx} out of range [0, {poses_data.num_frames})")

        # Compute joint positions
        joints = np.zeros((poses_data.num_subjects, 24, 3))
        for subj in range(poses_data.num_subjects):
            global_orient = poses_data.global_orient[subj, frame_idx, :]
            body_pose = poses_data.body_pose[subj, frame_idx, :]
            transl = poses_data.transl[subj, frame_idx, :]
            joints[subj] = self.compute_smpl_joints(global_orient, body_pose, transl)

        # Limit subjects
        if num_subjects is not None and num_subjects > 0:
            joints = joints[:num_subjects]
            actual_subjects = min(num_subjects, poses_data.num_subjects)
        else:
            actual_subjects = poses_data.num_subjects

        # Create figure
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')

        # Draw pitch if requested
        if show_pitch:
            self.draw_football_pitch(ax, show_3d=True)

        # Plot each subject
        for subj_idx in range(actual_subjects):
            color = self.colors[subj_idx % len(self.colors)]
            subject_joints = joints[subj_idx]

            ax.scatter(subject_joints[:, 0], subject_joints[:, 1], subject_joints[:, 2],
                      c=[color], s=50, alpha=0.8)

            for start_joint, end_joint in self.skeleton_connections:
                start_pos = subject_joints[start_joint]
                end_pos = subject_joints[end_joint]
                ax.plot([start_pos[0], end_pos[0]],
                       [start_pos[1], end_pos[1]],
                       [start_pos[2], end_pos[2]],
                       c=color, alpha=0.6, linewidth=2)

            if show_labels:
                root_pos = subject_joints[0]
                ax.text(root_pos[0], root_pos[1], root_pos[2] + 0.2,
                       f'S{subj_idx}', fontsize=10, color=color)

        ax.set_xlabel('X (meters)')
        ax.set_ylabel('Y (meters)')
        ax.set_zlabel('Z (meters)')
        ax.set_title(f'SMPL Poses - {poses_data.sequence_name} - Frame {frame_idx}')
        ax.view_init(elev=elev, azim=azim)
        ax.set_zlim(-1.0, 4.0)
        plt.tight_layout()
        return fig

    def animate_3d_poses(self, poses_data, start_frame: int = 0, end_frame: Optional[int] = None,
                        frame_step: int = 1, figsize: Tuple[int, int] = (12, 8),
                        elev: float = 20, azim: float = -60,
                        num_subjects: Optional[int] = None,
                        fps: float = 25.0, duration: Optional[float] = None,
                        show_pitch: bool = True, zoom: int = 100) -> plt.Figure:
        """Create animated 3D visualization with zoom support."""
        if end_frame is None:
            end_frame = poses_data.num_frames - 1

        if start_frame < 0 or start_frame >= poses_data.num_frames:
            raise ValueError(f"Start frame {start_frame} out of range")
        if end_frame < 0 or end_frame >= poses_data.num_frames:
            raise ValueError(f"End frame {end_frame} out of range")
        if start_frame >= end_frame:
            raise ValueError(f"Start frame must be less than end frame")

        frame_sequence = list(range(start_frame, end_frame + 1, frame_step))

        if duration is not None:
            interval = (duration * 1000) / len(frame_sequence)
        else:
            interval = 1000 / fps

        actual_subjects = min(num_subjects, poses_data.num_subjects) if num_subjects else poses_data.num_subjects

        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')

        # Calculate zoom factor (100 = normal, 200 = 2x zoom, 50 = 0.5x zoom)
        zoom_factor = 100.0 / zoom
        visible_range_x = (self.half_length + self.margin) * zoom_factor
        visible_range_y = (self.half_width + self.margin) * zoom_factor

        ax.set_xlim(-visible_range_x, visible_range_x)
        ax.set_ylim(-visible_range_y, visible_range_y)
        ax.set_zlim(-1.0, 4.0)

        ax.set_xlabel('X (meters)')
        ax.set_ylabel('Y (meters)')
        ax.set_zlabel('Z (meters)')
        ax.view_init(elev=elev, azim=azim)

        if show_pitch:
            self.draw_football_pitch(ax, show_3d=True, margin=self.margin * zoom_factor)

        # Initialize plots
        joint_plots = []
        skeleton_plots = []
        text_plots = []

        for subj_idx in range(actual_subjects):
            color = self.colors[subj_idx % len(self.colors)]
            joint_plot = ax.scatter([], [], [], c=[color], s=50, alpha=0.8)
            joint_plots.append(joint_plot)

            subj_skeleton_plots = []
            for _ in self.skeleton_connections:
                line, = ax.plot([], [], [], c=color, alpha=0.6, linewidth=2)
                subj_skeleton_plots.append(line)
            skeleton_plots.append(subj_skeleton_plots)

            text = ax.text(0, 0, 0, f'S{subj_idx}', fontsize=10, color=color)
            text_plots.append(text)

        def animate(frame_num):
            frame_idx = frame_sequence[frame_num]
            ax.set_title(f'SMPL Poses - {poses_data.sequence_name} - Frame {frame_idx} (Zoom: {zoom}%)')

            for subj_idx in range(actual_subjects):
                global_orient = poses_data.global_orient[subj_idx, frame_idx, :]
                body_pose = poses_data.body_pose[subj_idx, frame_idx, :]
                transl = poses_data.transl[subj_idx, frame_idx, :]
                subject_joints = self.compute_smpl_joints(global_orient, body_pose, transl)

                joint_plots[subj_idx]._offsets3d = (subject_joints[:, 0],
                                                   subject_joints[:, 1],
                                                   subject_joints[:, 2])

                for conn_idx, (start_joint, end_joint) in enumerate(self.skeleton_connections):
                    start_pos = subject_joints[start_joint]
                    end_pos = subject_joints[end_joint]
                    skeleton_plots[subj_idx][conn_idx].set_data([start_pos[0], end_pos[0]],
                                                               [start_pos[1], end_pos[1]])
                    skeleton_plots[subj_idx][conn_idx].set_3d_properties([start_pos[2], end_pos[2]])

                root_pos = subject_joints[0]
                text_plots[subj_idx].set_position((root_pos[0], root_pos[1]))
                text_plots[subj_idx].set_3d_properties(root_pos[2] + 0.2)

            return joint_plots + [line for subj_lines in skeleton_plots for line in subj_lines] + text_plots

        anim = animation.FuncAnimation(fig, animate, frames=len(frame_sequence),
                                     interval=interval, blit=False, repeat=True)
        fig._animation = anim
        plt.tight_layout()
        return fig

    def visualize_pitch_tracking(self, poses_data, start_frame: int = 0, end_frame: Optional[int] = None,
                                frame_step: int = 1, figsize: Tuple[int, int] = (14, 10),
                                num_subjects: Optional[int] = None, show_pitch: bool = True) -> plt.Figure:
        """Visualize player tracking on pitch."""
        if end_frame is None:
            end_frame = poses_data.num_frames - 1

        frame_sequence = list(range(start_frame, end_frame + 1, frame_step))
        actual_subjects = min(num_subjects, poses_data.num_subjects) if num_subjects else poses_data.num_subjects

        fig, ax = plt.subplots(figsize=figsize)

        if show_pitch:
            self.draw_football_pitch(ax, show_3d=False)

        for subj_idx in range(actual_subjects):
            color = self.colors[subj_idx % len(self.colors)]
            trajectory = []
            for frame_idx in frame_sequence:
                pos = poses_data.get_pitch_coordinates(frame_idx, subj_idx)
                if not (np.isnan(pos).any() or np.isinf(pos).any()):
                    trajectory.append(pos)

            if len(trajectory) > 0:
                trajectory = np.array(trajectory)
                ax.plot(trajectory[:, 0], trajectory[:, 1],
                       color=color, alpha=0.7, linewidth=2, label=f'Subject {subj_idx}')
                ax.scatter(trajectory[0, 0], trajectory[0, 1],
                          color=color, s=100, marker='o', alpha=0.8, edgecolors='white', linewidth=1)
                ax.scatter(trajectory[-1, 0], trajectory[-1, 1],
                          color=color, s=100, marker='s', alpha=0.8, edgecolors='white', linewidth=1)

        ax.set_xlabel('X (meters)', fontsize=12)
        ax.set_ylabel('Y (meters)', fontsize=12)
        ax.set_title(f'Pitch Tracking - {poses_data.sequence_name} - Frames {start_frame}-{end_frame}', fontsize=14)
        ax.grid(True, alpha=0.3)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.set_aspect('equal')
        plt.tight_layout()
        return fig

    def animate_pitch_tracking(self, poses_data, start_frame: int = 0, end_frame: Optional[int] = None,
                              frame_step: int = 1, figsize: Tuple[int, int] = (14, 10),
                              num_subjects: Optional[int] = None,
                              fps: float = 25.0, duration: Optional[float] = None,
                              trail_length: int = 50, show_pitch: bool = True) -> plt.Figure:
        """Create animated pitch tracking."""
        if end_frame is None:
            end_frame = poses_data.num_frames - 1

        if start_frame < 0 or start_frame >= poses_data.num_frames:
            raise ValueError(f"Start frame {start_frame} out of range")
        if end_frame < 0 or end_frame >= poses_data.num_frames:
            raise ValueError(f"End frame {end_frame} out of range")
        if start_frame >= end_frame:
            raise ValueError(f"Start frame must be less than end frame")

        frame_sequence = list(range(start_frame, end_frame + 1, frame_step))

        if duration is not None:
            interval = (duration * 1000) / len(frame_sequence)
        else:
            interval = 1000 / fps

        actual_subjects = min(num_subjects, poses_data.num_subjects) if num_subjects else poses_data.num_subjects

        fig, ax = plt.subplots(figsize=figsize)

        if show_pitch:
            self.draw_football_pitch(ax, show_3d=False)

        ax.set_xlabel('X (meters)', fontsize=12)
        ax.set_ylabel('Y (meters)', fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')

        subject_plots = []
        trail_plots = []
        text_plots = []

        for subj_idx in range(actual_subjects):
            color = self.colors[subj_idx % len(self.colors)]
            current_plot, = ax.plot([], [], 'o', color=color, markersize=10, alpha=0.8)
            subject_plots.append(current_plot)

            trail_plot, = ax.plot([], [], '-', color=color, alpha=0.5, linewidth=2)
            trail_plots.append(trail_plot)

            text = ax.text(0, 0, f'S{subj_idx}', fontsize=10, color=color,
                          bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.7))
            text_plots.append(text)

        trail_data = [[] for _ in range(actual_subjects)]

        def animate(frame_num):
            frame_idx = frame_sequence[frame_num]
            ax.set_title(f'Pitch Tracking - {poses_data.sequence_name} - Frame {frame_idx}')

            all_coords = poses_data.get_pitch_coordinates(frame_idx)

            for subj_idx in range(actual_subjects):
                coord = all_coords[subj_idx]

                if not (np.isnan(coord).any() or np.isinf(coord).any()):
                    trail_data[subj_idx].append(coord.copy())

                    if len(trail_data[subj_idx]) > trail_length:
                        trail_data[subj_idx].pop(0)

                    subject_plots[subj_idx].set_data([coord[0]], [coord[1]])

                    if len(trail_data[subj_idx]) > 1:
                        trail_coords = np.array(trail_data[subj_idx])
                        trail_plots[subj_idx].set_data(trail_coords[:, 0], trail_coords[:, 1])
                    else:
                        trail_plots[subj_idx].set_data([], [])

                    text_plots[subj_idx].set_position((coord[0] + 0.5, coord[1] + 0.5))
                    text_plots[subj_idx].set_visible(True)
                else:
                    subject_plots[subj_idx].set_data([], [])
                    text_plots[subj_idx].set_visible(False)

            return subject_plots + trail_plots + text_plots

        anim = animation.FuncAnimation(fig, animate, frames=len(frame_sequence),
                                     interval=interval, blit=False, repeat=True)
        fig._animation = anim
        plt.tight_layout()
        return fig


# Create default visualizer instance for convenience
_default_visualizer = PosesVisualizer()

# Expose convenience functions that use the default visualizer
def visualize_3d_poses(poses_data, **kwargs):
    """Convenience function using default visualizer."""
    return _default_visualizer.visualize_3d_poses(poses_data, **kwargs)

def animate_3d_poses(poses_data, **kwargs):
    """Convenience function using default visualizer."""
    return _default_visualizer.animate_3d_poses(poses_data, **kwargs)

def visualize_pitch_tracking(poses_data, **kwargs):
    """Convenience function using default visualizer."""
    return _default_visualizer.visualize_pitch_tracking(poses_data, **kwargs)

def animate_pitch_tracking(poses_data, **kwargs):
    """Convenience function using default visualizer."""
    return _default_visualizer.animate_pitch_tracking(poses_data, **kwargs)

def compute_smpl_joints(global_orient, body_pose, transl):
    """Convenience function using default visualizer."""
    return _default_visualizer.compute_smpl_joints(global_orient, body_pose, transl)

def draw_football_pitch(ax, **kwargs):
    """Convenience function using default visualizer."""
    return _default_visualizer.draw_football_pitch(ax, **kwargs)
