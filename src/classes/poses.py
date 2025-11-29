"""
SMPL Poses data handling.

Provides object-oriented interface to SMPL pose parameters from the WorldPose dataset.
"""

import numpy as np
from pathlib import Path
from typing import Dict, Optional, Tuple, Any, List
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation


class PosesData:
    """
    Handler for SMPL pose data.

    SMPL poses contain raw parameters from WorldPose dataset:
    - global_orient: Global orientation (axis-angle)
    - body_pose: Body pose parametFrs (23 joints × 3)
    - transl: Translation in world coordinates
    - betas: Shape parameters

    Attributes:
        sequence_name: Name of the sequence
        global_orient: (num_subjects, num_frames, 3) global orientation
        body_pose: (num_subjects, num_frames, 69) body pose parameters
        transl: (num_subjects, num_frames, 3) translation vectors
        betas: (num_subjects, num_frames, 10) shape parameters
        num_subjects: Number of subjects in sequence
        num_frames: Number of frames in sequence
    """

    def __init__(self, sequence_name: str, npz_data: Any):
        """
        Initialize PosesData from NPZ file.

        Args:
            sequence_name: Name of the sequence
            npz_data: Loaded NPZ file containing SMPL parameters
        """
        self.sequence_name = sequence_name

        # Load SMPL parameters
        # Original shape: (num_subjects, num_frames, dim)
        self.global_orient = npz_data['global_orient']  # (num_subjects, num_frames, 3)
        self.body_pose = npz_data['body_pose']          # (num_subjects, num_frames, 69)
        self.transl = npz_data['transl']                # (num_subjects, num_frames, 3)
        self.betas = npz_data['betas']                  # (num_subjects, num_frames, 10)

        # Store dimensions
        self.num_subjects = self.body_pose.shape[0]
        self.num_frames = self.body_pose.shape[1]

    @classmethod
    def load(cls, poses_dir: Path, sequence_name: str) -> 'PosesData':
        """
        Load poses data from directory.

        Args:
            poses_dir: Directory containing pose files
            sequence_name: Name of the sequence

        Returns:
            PosesData instance
        """
        pose_path = poses_dir / f"{sequence_name}.npz"
        if not pose_path.exists():
            raise FileNotFoundError(f"Pose file not found: {pose_path}")

        npz_data = np.load(pose_path, allow_pickle=True)
        return cls(sequence_name, npz_data)

    @classmethod
    def load_all(cls, poses_dir: Path) -> Dict[str, 'PosesData']:
        """
        Load all pose sequences from directory.

        Args:
            poses_dir: Directory containing pose files

        Returns:
            Dictionary mapping sequence names to PosesData instances
        """
        poses_dict = {}
        for pose_path in sorted(poses_dir.glob("*.npz")):
            sequence_name = pose_path.stem
            poses_dict[sequence_name] = cls.load(poses_dir, sequence_name)
        return poses_dict

    def get_frame_data(self, frame_idx: int, subject_idx: Optional[int] = None) -> Dict[str, np.ndarray]:
        """
        Get SMPL parameters for a specific frame.

        Args:
            frame_idx: Frame index
            subject_idx: Optional subject index (if None, returns all subjects)

        Returns:
            Dictionary with SMPL parameters
        """
        if frame_idx < 0 or frame_idx >= self.num_frames:
            raise ValueError(f"Frame index {frame_idx} out of range [0, {self.num_frames})")

        if subject_idx is None:
            return {
                'global_orient': self.global_orient[:, frame_idx, :],
                'body_pose': self.body_pose[:, frame_idx, :],
                'transl': self.transl[:, frame_idx, :],
                'betas': self.betas[:, :]
            }
        else:
            if subject_idx < 0 or subject_idx >= self.num_subjects:
                raise ValueError(f"Subject index {subject_idx} out of range [0, {self.num_subjects})")

            return {
                'global_orient': self.global_orient[subject_idx, frame_idx, :],
                'body_pose': self.body_pose[subject_idx, frame_idx, :],
                'transl': self.transl[subject_idx, frame_idx, :],
                'betas': self.betas[subject_idx, frame_idx, :]
            }

    def get_subject_trajectory(self, subject_idx: int) -> np.ndarray:
        """
        Get trajectory (translation over time) for a specific subject.

        Args:
            subject_idx: Subject index

        Returns:
            (num_frames, 3) array of positions
        """
        if subject_idx < 0 or subject_idx >= self.num_subjects:
            raise ValueError(f"Subject index {subject_idx} out of range [0, {self.num_subjects})")

        return self.transl[subject_idx, :, :]

    def get_smpl_joints(self, frame_idx: int, subject_idx: Optional[int] = None) -> np.ndarray:
        """
        Get SMPL joint positions for a specific frame using pose parameters.
        
        This uses the SMPL pose parameters (global_orient, body_pose) to compute
        joint positions that reflect the actual pose, not just translation.
        
        Args:
            frame_idx: Frame index
            subject_idx: Optional subject index (if None, returns all subjects)
            
        Returns:
            Joint positions array: (num_subjects, 24, 3) or (24, 3) if subject_idx specified
        """
        if frame_idx < 0 or frame_idx >= self.num_frames:
            raise ValueError(f"Frame index {frame_idx} out of range [0, {self.num_frames})")
        
        def rodrigues_rotation(axis_angle):
            """Convert axis-angle to rotation matrix using Rodrigues' formula."""
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
        
        def compute_pose_joints(global_orient, body_pose, transl):
            """Compute joint positions from SMPL parameters."""
            # SMPL joint hierarchy and base positions (T-pose)
            joint_parents = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19, 20, 21]
            
            # Base joint positions in T-pose (simplified SMPL template)
            base_joints = np.array([
                [0.0, 0.0, 0.0],      # 0: pelvis (root)
                [-0.1, -0.05, 0.0],   # 1: left_hip
                [0.1, -0.05, 0.0],    # 2: right_hip
                [0.0, 0.1, 0.0],      # 3: spine1
                [-0.1, -0.4, 0.0],    # 4: left_knee
                [0.1, -0.4, 0.0],     # 5: right_knee
                [0.0, 0.2, 0.0],      # 6: spine2
                [-0.1, -0.7, 0.0],    # 7: left_ankle
                [0.1, -0.7, 0.0],     # 8: right_ankle
                [0.0, 0.3, 0.0],      # 9: spine3
                [-0.1, -0.8, 0.1],    # 10: left_foot
                [0.1, -0.8, 0.1],     # 11: right_foot
                [0.0, 0.4, 0.0],      # 12: neck
                [-0.15, 0.35, 0.0],   # 13: left_collar
                [0.15, 0.35, 0.0],    # 14: right_collar
                [0.0, 0.5, 0.0],      # 15: head
                [-0.25, 0.3, 0.0],    # 16: left_shoulder
                [0.25, 0.3, 0.0],     # 17: right_shoulder
                [-0.45, 0.25, 0.0],   # 18: left_elbow
                [0.45, 0.25, 0.0],    # 19: right_elbow
                [-0.65, 0.2, 0.0],    # 20: left_wrist
                [0.65, 0.2, 0.0],     # 21: right_wrist
                [-0.75, 0.15, 0.0],   # 22: left_hand
                [0.75, 0.15, 0.0],    # 23: right_hand
            ])
            
            # Handle NaN/Inf values
            if np.any(np.isnan(transl)) or np.any(np.isinf(transl)):
                transl = np.array([0.0, 0.0, 0.0])
            if np.any(np.isnan(global_orient)) or np.any(np.isinf(global_orient)):
                global_orient = np.array([0.0, 0.0, 0.0])
            if np.any(np.isnan(body_pose)) or np.any(np.isinf(body_pose)):
                body_pose = np.zeros(69)  # 23 joints * 3
            
            # Initialize joint transformations
            joint_rotations = [np.eye(3) for _ in range(24)]
            joint_positions = base_joints.copy()
            
            # Apply global orientation to root
            joint_rotations[0] = rodrigues_rotation(global_orient)
            
            # Apply body pose rotations (23 joints, 3 parameters each)
            for i in range(1, 24):
                if i <= 23:  # We have 23 body pose parameters
                    pose_idx = (i - 1) * 3
                    if pose_idx + 2 < len(body_pose):
                        axis_angle = body_pose[pose_idx:pose_idx + 3]
                        # Add some variation based on pose parameters
                        rotation = rodrigues_rotation(axis_angle * 0.3)  # Scale down for stability
                        joint_rotations[i] = rotation
                        
                        # Apply pose-based deformation to joint positions
                        pose_magnitude = np.linalg.norm(axis_angle)
                        if pose_magnitude > 0:
                            # Modify joint position based on pose
                            deformation = axis_angle * 0.05  # Small deformation
                            joint_positions[i] += deformation
            
            # Forward kinematics: compute world positions
            world_joints = np.zeros((24, 3))
            world_rotations = [np.eye(3) for _ in range(24)]
            
            for i in range(24):
                parent = joint_parents[i]
                if parent == -1:  # Root joint
                    world_rotations[i] = joint_rotations[i]
                    world_joints[i] = joint_positions[i] + transl
                else:
                    # Accumulate rotation from parent
                    world_rotations[i] = np.dot(world_rotations[parent], joint_rotations[i])
                    # Transform position by parent's rotation and add to parent position
                    local_pos = joint_positions[i] - joint_positions[parent]
                    transformed_pos = np.dot(world_rotations[parent], local_pos)
                    world_joints[i] = world_joints[parent] + transformed_pos
            
            return world_joints
        
        if subject_idx is None:
            # Return for all subjects
            joints = np.zeros((self.num_subjects, 24, 3))
            for subj in range(self.num_subjects):
                global_orient = self.global_orient[subj, frame_idx, :]
                body_pose = self.body_pose[subj, frame_idx, :]
                transl = self.transl[subj, frame_idx, :]
                
                joints[subj] = compute_pose_joints(global_orient, body_pose, transl)
            return joints
        else:
            if subject_idx < 0 or subject_idx >= self.num_subjects:
                raise ValueError(f"Subject index {subject_idx} out of range [0, {self.num_subjects})")
            
            global_orient = self.global_orient[subject_idx, frame_idx, :]
            body_pose = self.body_pose[subject_idx, frame_idx, :]
            transl = self.transl[subject_idx, frame_idx, :]
            
            return compute_pose_joints(global_orient, body_pose, transl)

    def visualize_3d_poses(self, frame_idx: int, figsize: Tuple[int, int] = (12, 8),
                          elev: float = 20, azim: float = -60,
                          num_subjects: Optional[int] = None,
                          show_labels: bool = False) -> plt.Figure:
        """
        Visualize 3D poses for a specific frame.
        
        Args:
            frame_idx: Frame index to visualize
            figsize: Figure size
            elev: Elevation angle for 3D view
            azim: Azimuth angle for 3D view
            num_subjects: Maximum number of subjects to show (None = all)
            show_labels: Whether to show subject labels
            
        Returns:
            Matplotlib figure
        """
        if frame_idx < 0 or frame_idx >= self.num_frames:
            raise ValueError(f"Frame index {frame_idx} out of range [0, {self.num_frames})")
        
        # Get joint positions
        joints = self.get_smpl_joints(frame_idx)
        
        # Limit subjects if specified
        if num_subjects is not None and num_subjects > 0:
            joints = joints[:num_subjects]
            actual_subjects = min(num_subjects, self.num_subjects)
        else:
            actual_subjects = self.num_subjects
        
        # Create figure
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')
        
        # Color palette for different subjects
        colors = plt.cm.tab10(np.linspace(0, 1, 10))
        
        # SMPL skeleton connections
        skeleton_connections = [
            (0, 1), (0, 2), (0, 3),  # pelvis connections
            (1, 4), (4, 7), (7, 10),  # left leg
            (2, 5), (5, 8), (8, 11),  # right leg
            (3, 6), (6, 9), (9, 12), (12, 15),  # spine to head
            (9, 13), (9, 14),  # collar bones
            (13, 16), (16, 18), (18, 20), (20, 22),  # left arm
            (14, 17), (17, 19), (19, 21), (21, 23),  # right arm
        ]
        
        # Plot each subject
        for subj_idx in range(actual_subjects):
            color = colors[subj_idx % len(colors)]
            subject_joints = joints[subj_idx]
            
            # Plot joints
            ax.scatter(subject_joints[:, 0], subject_joints[:, 1], subject_joints[:, 2],
                      c=[color], s=50, alpha=0.8)
            
            # Plot skeleton connections
            for start_joint, end_joint in skeleton_connections:
                start_pos = subject_joints[start_joint]
                end_pos = subject_joints[end_joint]
                ax.plot([start_pos[0], end_pos[0]],
                       [start_pos[1], end_pos[1]],
                       [start_pos[2], end_pos[2]],
                       c=color, alpha=0.6, linewidth=2)
            
            # Add subject label
            if show_labels:
                root_pos = subject_joints[0]  # pelvis position
                ax.text(root_pos[0], root_pos[1], root_pos[2] + 0.2,
                       f'S{subj_idx}', fontsize=10, color=color)
        
        # Add Z=0 ground plane with light green opacity
        all_joints = joints.reshape(-1, 3)
        valid_joints = all_joints[~(np.isnan(all_joints).any(axis=1) | np.isinf(all_joints).any(axis=1))]
        
        if len(valid_joints) > 0:
            # Calculate X-Y range for ground plane
            x_min, x_max = valid_joints[:, 0].min(), valid_joints[:, 0].max()
            y_min, y_max = valid_joints[:, 1].min(), valid_joints[:, 1].max()
            
            # Add margin for ground plane
            x_margin = (x_max - x_min) * 0.2
            y_margin = (y_max - y_min) * 0.2
            x_min -= x_margin
            x_max += x_margin
            y_min -= y_margin
            y_max += y_margin
            
            # Create ground plane mesh at Z=0
            xx, yy = np.meshgrid(np.linspace(x_min, x_max, 10),
                               np.linspace(y_min, y_max, 10))
            zz = np.zeros_like(xx)
            
            # Draw ground plane
            ax.plot_surface(xx, yy, zz, alpha=0.1, color='lightgreen',
                          linewidth=0, antialiased=True)
        
        # Set labels and title
        ax.set_xlabel('X (meters)')
        ax.set_ylabel('Y (meters)')
        ax.set_zlabel('Z (meters)')
        ax.set_title(f'SMPL Poses - {self.sequence_name} - Frame {frame_idx}')
        
        # Set view angle
        ax.view_init(elev=elev, azim=azim)
        
        # Set axis limits with proper Z-axis scaling
        if len(valid_joints) == 0:
            # If no valid joints, use default range
            max_range = 1.0
            mid_x, mid_y = 0.0, 0.0
        else:
            # Calculate X-Y range
            xy_range = np.array([valid_joints[:, 0].max() - valid_joints[:, 0].min(),
                               valid_joints[:, 1].max() - valid_joints[:, 1].min()]).max() / 2.0
            
            # Ensure max_range is not zero or NaN
            if xy_range == 0 or np.isnan(xy_range) or np.isinf(xy_range):
                xy_range = 1.0
            
            mid_x = (valid_joints[:, 0].max() + valid_joints[:, 0].min()) * 0.5
            mid_y = (valid_joints[:, 1].max() + valid_joints[:, 1].min()) * 0.5
            max_range = xy_range
        
        # Set X-Y limits based on data
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        
        # Set Z-axis to fixed range: -1 to +3 meters (feet to head height)
        ax.set_zlim(-1.0, 3.0)
        
        plt.tight_layout()
        return fig

    def animate_3d_poses(self, start_frame: int = 0, end_frame: Optional[int] = None,
                        frame_step: int = 1, figsize: Tuple[int, int] = (12, 8),
                        elev: float = 20, azim: float = -60,
                        num_subjects: Optional[int] = None,
                        fps: float = 25.0, duration: Optional[float] = None) -> plt.Figure:
        """
        Create animated 3D visualization of poses.
        
        Args:
            start_frame: Starting frame index
            end_frame: Ending frame index (None = last frame)
            frame_step: Step size between frames
            figsize: Figure size
            elev: Elevation angle for 3D view
            azim: Azimuth angle for 3D view
            num_subjects: Maximum number of subjects to show (None = all)
            fps: Frames per second for animation
            duration: Total duration in seconds (overrides fps)
            
        Returns:
            Matplotlib figure with animation
        """
        if end_frame is None:
            end_frame = self.num_frames - 1
        
        if start_frame < 0 or start_frame >= self.num_frames:
            raise ValueError(f"Start frame {start_frame} out of range [0, {self.num_frames})")
        if end_frame < 0 or end_frame >= self.num_frames:
            raise ValueError(f"End frame {end_frame} out of range [0, {self.num_frames})")
        if start_frame >= end_frame:
            raise ValueError(f"Start frame ({start_frame}) must be less than end frame ({end_frame})")
        
        # Create frame sequence
        frame_sequence = list(range(start_frame, end_frame + 1, frame_step))
        
        # Calculate animation parameters
        if duration is not None:
            interval = (duration * 1000) / len(frame_sequence)  # milliseconds per frame
        else:
            interval = 1000 / fps  # milliseconds per frame
        
        # Limit subjects if specified
        actual_subjects = min(num_subjects, self.num_subjects) if num_subjects else self.num_subjects
        
        # Create figure and axis
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')
        
        # Color palette
        colors = plt.cm.tab10(np.linspace(0, 1, 10))
        
        # SMPL skeleton connections
        skeleton_connections = [
            (0, 1), (0, 2), (0, 3),  # pelvis connections
            (1, 4), (4, 7), (7, 10),  # left leg
            (2, 5), (5, 8), (8, 11),  # right leg
            (3, 6), (6, 9), (9, 12), (12, 15),  # spine to head
            (9, 13), (9, 14),  # collar bones
            (13, 16), (16, 18), (18, 20), (20, 22),  # left arm
            (14, 17), (17, 19), (19, 21), (21, 23),  # right arm
        ]
        
        # Get all joints for axis limits
        all_joints_list = []
        for frame_idx in frame_sequence:
            joints = self.get_smpl_joints(frame_idx)[:actual_subjects]
            all_joints_list.append(joints.reshape(-1, 3))
        all_joints = np.vstack(all_joints_list)
        
        # Set axis limits - handle NaN/Inf values
        # Filter out NaN/Inf values for axis calculation
        valid_joints = all_joints[~(np.isnan(all_joints).any(axis=1) | np.isinf(all_joints).any(axis=1))]
        
        if len(valid_joints) == 0:
            # If no valid joints, use default range
            max_range = 1.0
            mid_x, mid_y = 0.0, 0.0
        else:
            # Calculate X-Y range only
            xy_range = np.array([valid_joints[:, 0].max() - valid_joints[:, 0].min(),
                               valid_joints[:, 1].max() - valid_joints[:, 1].min()]).max() / 2.0
            
            # Ensure max_range is not zero or NaN
            if xy_range == 0 or np.isnan(xy_range) or np.isinf(xy_range):
                xy_range = 1.0
            
            mid_x = (valid_joints[:, 0].max() + valid_joints[:, 0].min()) * 0.5
            mid_y = (valid_joints[:, 1].max() + valid_joints[:, 1].min()) * 0.5
            max_range = xy_range
            
            # Add Z=0 ground plane for animation
            x_min, x_max = mid_x - max_range, mid_x + max_range
            y_min, y_max = mid_y - max_range, mid_y + max_range
            
            # Create ground plane mesh at Z=0
            xx, yy = np.meshgrid(np.linspace(x_min, x_max, 10),
                               np.linspace(y_min, y_max, 10))
            zz = np.zeros_like(xx)
            
            # Draw ground plane
            ax.plot_surface(xx, yy, zz, alpha=0.1, color='lightgreen',
                          linewidth=0, antialiased=True)
        
        # Set X-Y limits based on data
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        
        # Set Z-axis to fixed range: -1 to +3 meters (feet to head height)
        ax.set_zlim(-1.0, 3.0)
        
        # Set labels and view
        ax.set_xlabel('X (meters)')
        ax.set_ylabel('Y (meters)')
        ax.set_zlabel('Z (meters)')
        ax.view_init(elev=elev, azim=azim)
        
        # Initialize empty plots
        joint_plots = []
        skeleton_plots = []
        text_plots = []
        
        for subj_idx in range(actual_subjects):
            color = colors[subj_idx % len(colors)]
            
            # Joint scatter plot
            joint_plot = ax.scatter([], [], [], c=[color], s=50, alpha=0.8)
            joint_plots.append(joint_plot)
            
            # Skeleton lines
            subj_skeleton_plots = []
            for _ in skeleton_connections:
                line, = ax.plot([], [], [], c=color, alpha=0.6, linewidth=2)
                subj_skeleton_plots.append(line)
            skeleton_plots.append(subj_skeleton_plots)
            
            # Text label
            text = ax.text(0, 0, 0, f'S{subj_idx}', fontsize=10, color=color)
            text_plots.append(text)
        
        # Animation function
        def animate(frame_num):
            frame_idx = frame_sequence[frame_num]
            joints = self.get_smpl_joints(frame_idx)[:actual_subjects]
            
            # Update title
            ax.set_title(f'SMPL Poses - {self.sequence_name} - Frame {frame_idx}')
            
            # Update each subject
            for subj_idx in range(actual_subjects):
                subject_joints = joints[subj_idx]
                
                # Update joint positions
                joint_plots[subj_idx]._offsets3d = (subject_joints[:, 0],
                                                   subject_joints[:, 1],
                                                   subject_joints[:, 2])
                
                # Update skeleton lines
                for conn_idx, (start_joint, end_joint) in enumerate(skeleton_connections):
                    start_pos = subject_joints[start_joint]
                    end_pos = subject_joints[end_joint]
                    skeleton_plots[subj_idx][conn_idx].set_data([start_pos[0], end_pos[0]],
                                                               [start_pos[1], end_pos[1]])
                    skeleton_plots[subj_idx][conn_idx].set_3d_properties([start_pos[2], end_pos[2]])
                
                # Update text position
                root_pos = subject_joints[0]  # pelvis position
                text_plots[subj_idx].set_position((root_pos[0], root_pos[1]))
                text_plots[subj_idx].set_3d_properties(root_pos[2] + 0.2)
            
            return joint_plots + [line for subj_lines in skeleton_plots for line in subj_lines] + text_plots
        
        # Create animation
        anim = animation.FuncAnimation(fig, animate, frames=len(frame_sequence),
                                     interval=interval, blit=False, repeat=True)
        
        # Store animation reference in figure
        fig._animation = anim
        
        plt.tight_layout()
        return fig

    def animate_pitch_tracking(self, start_frame: int = 0, end_frame: Optional[int] = None,
                              frame_step: int = 1, figsize: Tuple[int, int] = (14, 10),
                              num_subjects: Optional[int] = None,
                              fps: float = 25.0, duration: Optional[float] = None,
                              trail_length: int = 50, show_pitch: bool = True) -> plt.Figure:
        """
        Create animated pitch tracking visualization showing player movement over time.
        
        Args:
            start_frame: Starting frame index
            end_frame: Ending frame index (None = last frame)
            frame_step: Step size between frames
            figsize: Figure size
            num_subjects: Maximum number of subjects to show (None = all)
            fps: Frames per second for animation
            duration: Total duration in seconds (overrides fps)
            trail_length: Number of previous positions to show as trail
            show_pitch: Whether to show football pitch outline
            
        Returns:
            Matplotlib figure with animation
        """
        if end_frame is None:
            end_frame = self.num_frames - 1
        
        if start_frame < 0 or start_frame >= self.num_frames:
            raise ValueError(f"Start frame {start_frame} out of range [0, {self.num_frames})")
        if end_frame < 0 or end_frame >= self.num_frames:
            raise ValueError(f"End frame {end_frame} out of range [0, {self.num_frames})")
        if start_frame >= end_frame:
            raise ValueError(f"Start frame ({start_frame}) must be less than end frame ({end_frame})")
        
        # Create frame sequence
        frame_sequence = list(range(start_frame, end_frame + 1, frame_step))
        
        # Calculate animation parameters
        if duration is not None:
            interval = (duration * 1000) / len(frame_sequence)  # milliseconds per frame
        else:
            interval = 1000 / fps  # milliseconds per frame
        
        # Limit subjects if specified
        actual_subjects = min(num_subjects, self.num_subjects) if num_subjects else self.num_subjects
        
        # Create figure and axis
        fig, ax = plt.subplots(figsize=figsize)
        
        # Draw football pitch outline if requested
        if show_pitch:
            self._draw_football_pitch(ax)
        else:
            # Set axis limits based on data if no pitch outline
            all_coords = []
            for frame_idx in frame_sequence:
                coords = self.get_pitch_coordinates(frame_idx)[:actual_subjects]
                # Filter valid coordinates
                valid_coords = coords[~(np.isnan(coords).any(axis=1) | np.isinf(coords).any(axis=1))]
                if len(valid_coords) > 0:
                    all_coords.append(valid_coords)
            
            if all_coords:
                all_coords = np.vstack(all_coords)
                x_min, x_max = all_coords[:, 0].min(), all_coords[:, 0].max()
                y_min, y_max = all_coords[:, 1].min(), all_coords[:, 1].max()
                
                # Add margin
                x_margin = (x_max - x_min) * 0.1
                y_margin = (y_max - y_min) * 0.1
                ax.set_xlim(x_min - x_margin, x_max + x_margin)
                ax.set_ylim(y_min - y_margin, y_max + y_margin)
            else:
                # Default limits if no valid data
                ax.set_xlim(-10, 10)
                ax.set_ylim(-10, 10)
        
        # Color palette
        colors = plt.cm.tab10(np.linspace(0, 1, 10))
        
        # Set labels and grid
        ax.set_xlabel('X (meters)', fontsize=12)
        ax.set_ylabel('Y (meters)', fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
        
        # Initialize plots for each subject
        subject_plots = []
        trail_plots = []
        text_plots = []
        
        for subj_idx in range(actual_subjects):
            color = colors[subj_idx % len(colors)]
            
            # Current position plot
            current_plot, = ax.plot([], [], 'o', color=color, markersize=10, alpha=0.8)
            subject_plots.append(current_plot)
            
            # Trail plot
            trail_plot, = ax.plot([], [], '-', color=color, alpha=0.5, linewidth=2)
            trail_plots.append(trail_plot)
            
            # Text label
            text = ax.text(0, 0, f'S{subj_idx}', fontsize=10, color=color,
                          bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.7))
            text_plots.append(text)
        
        # Store trail data
        trail_data = [[] for _ in range(actual_subjects)]
        
        # Animation function
        def animate(frame_num):
            frame_idx = frame_sequence[frame_num]
            
            # Update title
            ax.set_title(f'Pitch Tracking - {self.sequence_name} - Frame {frame_idx}')
            
            # Get current coordinates
            current_coords = self.get_pitch_coordinates(frame_idx)[:actual_subjects]
            
            # Update each subject
            for subj_idx in range(actual_subjects):
                coord = current_coords[subj_idx]
                
                # Check if coordinate is valid
                if not (np.isnan(coord).any() or np.isinf(coord).any()):
                    # Add to trail
                    trail_data[subj_idx].append(coord.copy())
                    
                    # Limit trail length
                    if len(trail_data[subj_idx]) > trail_length:
                        trail_data[subj_idx].pop(0)
                    
                    # Update current position
                    subject_plots[subj_idx].set_data([coord[0]], [coord[1]])
                    
                    # Update trail
                    if len(trail_data[subj_idx]) > 1:
                        trail_coords = np.array(trail_data[subj_idx])
                        trail_plots[subj_idx].set_data(trail_coords[:, 0], trail_coords[:, 1])
                    
                    # Update text position
                    text_plots[subj_idx].set_position((coord[0] + 0.5, coord[1] + 0.5))
                    text_plots[subj_idx].set_visible(True)
                else:
                    # Hide if invalid
                    subject_plots[subj_idx].set_data([], [])
                    trail_plots[subj_idx].set_data([], [])
                    text_plots[subj_idx].set_visible(False)
            
            return subject_plots + trail_plots + text_plots
        
        # Create animation
        anim = animation.FuncAnimation(fig, animate, frames=len(frame_sequence),
                                     interval=interval, blit=False, repeat=True)
        
        # Store animation reference in figure
        fig._animation = anim
        
        plt.tight_layout()
        return fig

    def get_pitch_coordinates(self, frame_idx: int, subject_idx: Optional[int] = None) -> np.ndarray:
        """
        Get pitch coordinates (X, Y) for tracking purposes, centered on (0,0).
        
        This extracts the X and Y coordinates from the translation vector,
        applies coordinate transformation to center the pitch at (0,0).
        
        Args:
            frame_idx: Frame index
            subject_idx: Optional subject index (if None, returns all subjects)
            
        Returns:
            Pitch coordinates: (num_subjects, 2) or (2,) if subject_idx specified
        """
        if frame_idx < 0 or frame_idx >= self.num_frames:
            raise ValueError(f"Frame index {frame_idx} out of range [0, {self.num_frames})")
        
        if subject_idx is None:
            # Return X, Y coordinates for all subjects
            coords = self.transl[:, frame_idx, :2]  # Only X and Y
            # Replace NaN/Inf with zeros
            coords = np.where(np.isnan(coords) | np.isinf(coords), 0.0, coords)
            
            # Apply coordinate transformation to center pitch at (0,0)
            if hasattr(self, '_pitch_offset_x') and hasattr(self, '_pitch_offset_y'):
                coords[:, 0] -= self._pitch_offset_x
                coords[:, 1] -= self._pitch_offset_y
            
            return coords
        else:
            if subject_idx < 0 or subject_idx >= self.num_subjects:
                raise ValueError(f"Subject index {subject_idx} out of range [0, {self.num_subjects})")
            
            coords = self.transl[subject_idx, frame_idx, :2]  # Only X and Y
            # Replace NaN/Inf with zeros
            coords = np.where(np.isnan(coords) | np.isinf(coords), 0.0, coords)
            
            # Apply coordinate transformation to center pitch at (0,0)
            if hasattr(self, '_pitch_offset_x') and hasattr(self, '_pitch_offset_y'):
                coords[0] -= self._pitch_offset_x
                coords[1] -= self._pitch_offset_y
            
            return coords

    def visualize_pitch_tracking(self, start_frame: int = 0, end_frame: Optional[int] = None,
                                frame_step: int = 1, figsize: Tuple[int, int] = (14, 10),
                                num_subjects: Optional[int] = None, show_pitch: bool = True) -> plt.Figure:
        """
        Visualize player tracking on the pitch (top-down view).
        
        Args:
            start_frame: Starting frame index
            end_frame: Ending frame index (None = last frame)
            frame_step: Step size between frames
            figsize: Figure size
            num_subjects: Maximum number of subjects to show (None = all)
            show_pitch: Whether to show football pitch outline
            
        Returns:
            Matplotlib figure
        """
        if end_frame is None:
            end_frame = self.num_frames - 1
        
        # Create frame sequence
        frame_sequence = list(range(start_frame, end_frame + 1, frame_step))
        
        # Limit subjects if specified
        actual_subjects = min(num_subjects, self.num_subjects) if num_subjects else self.num_subjects
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Draw football pitch outline if requested
        if show_pitch:
            self._draw_football_pitch(ax)
        
        # Color palette
        colors = plt.cm.tab10(np.linspace(0, 1, 10))
        
        # Plot trajectories for each subject
        for subj_idx in range(actual_subjects):
            color = colors[subj_idx % len(colors)]
            
            # Get trajectory
            trajectory = []
            for frame_idx in frame_sequence:
                pos = self.get_pitch_coordinates(frame_idx, subj_idx)
                if not (np.isnan(pos).any() or np.isinf(pos).any()):
                    trajectory.append(pos)
            
            if len(trajectory) > 0:
                trajectory = np.array(trajectory)
                
                # Plot trajectory
                ax.plot(trajectory[:, 0], trajectory[:, 1],
                       color=color, alpha=0.7, linewidth=2, label=f'Subject {subj_idx}')
                
                # Mark start and end points
                ax.scatter(trajectory[0, 0], trajectory[0, 1],
                          color=color, s=100, marker='o', alpha=0.8, edgecolors='white', linewidth=1)  # Start
                ax.scatter(trajectory[-1, 0], trajectory[-1, 1],
                          color=color, s=100, marker='s', alpha=0.8, edgecolors='white', linewidth=1)  # End
        
        # Set labels and title
        ax.set_xlabel('X (meters)', fontsize=12)
        ax.set_ylabel('Y (meters)', fontsize=12)
        ax.set_title(f'Pitch Tracking - {self.sequence_name} - Frames {start_frame}-{end_frame}', fontsize=14)
        ax.grid(True, alpha=0.3)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.set_aspect('equal')
        
        plt.tight_layout()
        return fig

    def _draw_football_pitch(self, ax):
        """Draw a football pitch outline centered on (0,0)."""
        # Analyze actual data range to understand coordinate system
        all_coords = []
        sample_frames = range(0, min(self.num_frames, 500), 50)  # Sample frames
        
        for frame_idx in sample_frames:
            for subj_idx in range(min(self.num_subjects, 10)):  # Sample subjects
                coord = self.get_pitch_coordinates(frame_idx, subj_idx)
                if not (np.isnan(coord).any() or np.isinf(coord).any()):
                    all_coords.append(coord)
        
        if len(all_coords) > 0:
            all_coords = np.array(all_coords)
            data_x_min, data_x_max = all_coords[:, 0].min(), all_coords[:, 0].max()
            data_y_min, data_y_max = all_coords[:, 1].min(), all_coords[:, 1].max()
            data_center_x = (data_x_min + data_x_max) / 2
            data_center_y = (data_y_min + data_y_max) / 2
            
            print(f"📊 Pitch coordinate analysis:")
            print(f"   Data range: X=[{data_x_min:.1f}, {data_x_max:.1f}], Y=[{data_y_min:.1f}, {data_y_max:.1f}]")
            print(f"   Data center: ({data_center_x:.1f}, {data_center_y:.1f})")
        else:
            # Fallback if no valid data
            data_center_x, data_center_y = 0, 0
            data_x_min, data_x_max = -35, 35
            data_y_min, data_y_max = -25, 25
        
        # Standard FIFA pitch dimensions
        pitch_length = 105  # meters
        pitch_width = 68   # meters
        
        # Determine orientation based on data range
        data_x_range = data_x_max - data_x_min
        data_y_range = data_y_max - data_y_min
        
        # If X range is larger, pitch is oriented along X axis
        if data_x_range > data_y_range:
            # Pitch along X axis
            half_length = pitch_length / 2
            half_width = pitch_width / 2
        else:
            # Pitch along Y axis (rotated)
            half_length = pitch_width / 2
            half_width = pitch_length / 2
        
        # Center the pitch on (0,0) instead of data center
        pitch_center_x, pitch_center_y = 0.0, 0.0
        
        # Calculate offset to transform data coordinates to pitch-centered coordinates
        self._pitch_offset_x = data_center_x - pitch_center_x
        self._pitch_offset_y = data_center_y - pitch_center_y
        
        print(f"   Pitch offset: ({self._pitch_offset_x:.1f}, {self._pitch_offset_y:.1f})")
        print(f"   Pitch will be centered at (0, 0)")
        
        # Draw pitch outline centered on (0,0)
        pitch_x = [-half_length, half_length, half_length, -half_length, -half_length]
        pitch_y = [-half_width, -half_width, half_width, half_width, -half_width]
        
        ax.plot(pitch_x, pitch_y, 'k-', linewidth=2, alpha=0.8, label='Pitch Outline')
        
        # Draw center line
        center_line_x = [0, 0]
        center_line_y = [-half_width, half_width]
        ax.plot(center_line_x, center_line_y, 'k-', linewidth=1, alpha=0.6)
        
        # Draw center circle
        center_circle = plt.Circle((0, 0), 9.15,
                                 fill=False, color='black', linewidth=1, alpha=0.6)
        ax.add_patch(center_circle)
        
        # Draw penalty areas
        penalty_length = 16.5
        penalty_width = 40.3
        
        # Left penalty area
        left_penalty_x = [-half_length, -half_length + penalty_length,
                         -half_length + penalty_length, -half_length,
                         -half_length]
        left_penalty_y = [-penalty_width/2, -penalty_width/2,
                         penalty_width/2, penalty_width/2,
                         -penalty_width/2]
        ax.plot(left_penalty_x, left_penalty_y, 'k-', linewidth=1, alpha=0.6)
        
        # Right penalty area
        right_penalty_x = [half_length, half_length - penalty_length,
                          half_length - penalty_length, half_length,
                          half_length]
        right_penalty_y = [-penalty_width/2, -penalty_width/2,
                           penalty_width/2, penalty_width/2,
                           -penalty_width/2]
        ax.plot(right_penalty_x, right_penalty_y, 'k-', linewidth=1, alpha=0.6)
        
        # Set axis limits with some margin, centered on (0,0)
        margin = 10
        ax.set_xlim(-half_length - margin, half_length + margin)
        ax.set_ylim(-half_width - margin, half_width + margin)

    def __repr__(self) -> str:
        """String representation."""
        return (f"PosesData(sequence='{self.sequence_name}', "
                f"frames={self.num_frames}, subjects={self.num_subjects})")

    def __str__(self) -> str:
        """Detailed string representation."""
        return (f"PosesData for sequence '{self.sequence_name}':\n"
                f"  Frames: {self.num_frames}\n"
                f"  Subjects: {self.num_subjects}\n"
                f"  Shape: global_orient{self.global_orient.shape}, "
                f"body_pose{self.body_pose.shape}, "
                f"transl{self.transl.shape}, "
                f"betas{self.betas.shape}")
