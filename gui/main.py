"""
FIFA Skeletal Tracking Synchronized Viewer - Fast Native Playback

This GUI uses PyQt6's native video players for real-time performance.

Usage:
    source .venv_gui/bin/activate
    python gui/main.py --sequence ARG_CRO_220001
"""

import sys
import argparse
from pathlib import Path

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QGridLayout,
    QMessageBox, QFileDialog
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QAction

from widgets import VideoPanel, ControlPanel
from utils import SyncManager


class SynchronizedViewer(QMainWindow):
    """Main window for synchronized video viewing using native Qt players."""

    def __init__(self, args):
        """Initialize the synchronized viewer."""
        super().__init__()
        self.args = args
        self.sync_manager = SyncManager()

        # Paths
        self.project_root = Path(__file__).resolve().parent.parent
        self.data_dir = self.project_root / "data"
        self.results_dir = self.project_root / "results" / "SMPL"

        self.current_sequence = None

        self._init_ui()

        # Load initial sequence if provided
        if args.sequence:
            self._load_sequence(args.sequence)
        else:
            self._populate_sequences()

    def _init_ui(self):
        """Initialize the user interface."""
        self.setWindowTitle("FIFA Skeletal Tracking - Synchronized Viewer")
        self.setMinimumSize(1400, 900)

        # Create central widget and layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # Create 2x2 grid layout
        layout = QGridLayout()
        layout.setSpacing(5)
        layout.setContentsMargins(5, 5, 5, 5)

        # Video panels
        self.original_panel = VideoPanel("Original Video")
        self.poses_panel = VideoPanel("3D Poses")
        self.tracking_panel = VideoPanel("Pitch Tracking")

        # Control panel
        self.control_panel = ControlPanel()
        self.control_panel.play_pause_clicked.connect(self._toggle_playback)
        self.control_panel.seek_requested.connect(self._seek_to_position)
        self.control_panel.speed_changed.connect(self._set_playback_speed)
        self.control_panel.step_forward_clicked.connect(self._step_forward)
        self.control_panel.step_backward_clicked.connect(self._step_backward)
        self.control_panel.reset_clicked.connect(self._reset)
        self.control_panel.sequence_selected.connect(self._on_sequence_selected)
        self.control_panel.load_poses_clicked.connect(self._load_poses_file)
        self.control_panel.load_tracking_clicked.connect(self._load_tracking_file)

        # Connect sync manager signals
        self.sync_manager.position_changed.connect(self._update_position)

        # Add widgets to grid
        layout.addWidget(self.original_panel, 0, 0, 1, 1)
        layout.addWidget(self.poses_panel, 0, 1, 1, 1)
        layout.addWidget(self.control_panel, 1, 0, 1, 1)
        layout.addWidget(self.tracking_panel, 1, 1, 1, 1)

        # Set column and row stretches
        layout.setColumnStretch(0, 6)
        layout.setColumnStretch(1, 4)
        layout.setRowStretch(0, 7)
        layout.setRowStretch(1, 3)

        central_widget.setLayout(layout)

        # Create menu bar
        self._create_menu_bar()

    def _create_menu_bar(self):
        """Create the application menu bar."""
        menubar = self.menuBar()

        # File menu
        file_menu = menubar.addMenu("File")
        exit_action = QAction("Exit", self)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        # View menu
        view_menu = menubar.addMenu("View")
        fullscreen_action = QAction("Toggle Fullscreen", self)
        fullscreen_action.setShortcut("F")
        fullscreen_action.triggered.connect(self._toggle_fullscreen)
        view_menu.addAction(fullscreen_action)

    def _populate_sequences(self):
        """Populate the sequence dropdown."""
        videos_dir = self.data_dir / "videos" / "train_data"
        if videos_dir.exists():
            sequences = sorted([f.stem for f in videos_dir.glob("*.mp4")])
            for seq in sequences[:20]:
                self.control_panel.add_sequence(seq)

            if sequences:
                self.control_panel.set_sequence_info(
                    f"Found {len(sequences)} sequences. Select one from dropdown."
                )

    def _on_sequence_selected(self, sequence_name: str):
        """Handle sequence selection from dropdown."""
        if sequence_name:
            self._load_sequence(sequence_name)

    def _load_sequence(self, sequence_name: str):
        """Load a sequence with its video and animations."""
        print(f"\nLoading: {sequence_name}")
        self.current_sequence = sequence_name

        # Clean up previous videos
        self.sync_manager.cleanup()

        # Find video file
        video_path = None
        for subdir in ["train_data", "test_data", "challenge_data"]:
            potential_path = self.data_dir / "videos" / subdir / f"{sequence_name}.mp4"
            if potential_path.exists():
                video_path = potential_path
                break

        if not video_path:
            QMessageBox.warning(self, "Not Found", f"Video not found: {sequence_name}")
            return

        # Load original video
        if not self.original_panel.load_video(video_path):
            QMessageBox.critical(self, "Error", "Failed to load original video")
            return

        self.sync_manager.add_player(
            'original',
            self.original_panel.get_player(),
            is_master=True
        )
        print(f"✓ Loaded original video")

        # Try to load animations (optional)
        poses_path = self.results_dir / f"{sequence_name}_poses_animation.mp4"
        tracking_path = self.results_dir / f"{sequence_name}_pitch_tracking.mp4"

        if poses_path.exists() and self.poses_panel.load_video(poses_path):
            self.sync_manager.add_player('poses', self.poses_panel.get_player())
            print(f"✓ Loaded poses")
        else:
            self.poses_panel.clear()
            self.poses_panel.set_title("3D Poses - Click Load button →")
            print(f"- Poses not found")

        if tracking_path.exists() and self.tracking_panel.load_video(tracking_path):
            self.sync_manager.add_player('tracking', self.tracking_panel.get_player())
            print(f"✓ Loaded tracking")
        else:
            self.tracking_panel.clear()
            self.tracking_panel.set_title("Pitch Tracking - Click Load button →")
            print(f"- Tracking not found")

        # Update controls when duration is available
        master_player = self.original_panel.get_player()
        master_player.durationChanged.connect(self._on_duration_changed)

        # Update info
        self._update_info()

    def _on_duration_changed(self, duration: int):
        """Handle duration changed event."""
        self.control_panel.set_total_frames(duration)  # Using ms as "frames"
        self._update_info()

    def _update_info(self):
        """Update sequence info display."""
        duration = self.sync_manager.get_duration()
        loaded_count = len(self.sync_manager.players)

        info_text = (
            f"Sequence: {self.current_sequence}\n"
            f"Duration: {duration/1000:.1f}s\n"
            f"Videos loaded: {loaded_count}/3\n"
            f"Playback: Native Qt (hardware accelerated)"
        )
        self.control_panel.set_sequence_info(info_text)

    def _update_position(self, position: int):
        """Update position display."""
        duration = self.sync_manager.get_duration()
        self.control_panel.update_frame_info(position, duration)

    def _toggle_playback(self):
        """Toggle between play and pause."""
        if self.sync_manager.is_playing():
            self.sync_manager.pause()
            self.control_panel.set_play_state(False)
        else:
            self.sync_manager.play()
            self.control_panel.set_play_state(True)

    def _seek_to_position(self, position: int):
        """Seek to a specific position (slider value is in ms)."""
        self.sync_manager.seek(position)

    def _set_playback_speed(self, speed: float):
        """Set the playback speed."""
        self.sync_manager.set_playback_rate(speed)

    def _step_forward(self):
        """Step forward 1 second."""
        current = self.sync_manager.get_position()
        self.sync_manager.seek(current + 1000)

    def _step_backward(self):
        """Step backward 1 second."""
        current = self.sync_manager.get_position()
        self.sync_manager.seek(max(0, current - 1000))

    def _reset(self):
        """Reset to the beginning."""
        self.sync_manager.seek(0)

    def _toggle_fullscreen(self):
        """Toggle fullscreen mode."""
        if self.isFullScreen():
            self.showNormal()
        else:
            self.showFullScreen()

    def _load_poses_file(self):
        """Load 3D poses animation from file dialog."""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Load 3D Poses Animation",
            str(self.results_dir),
            "Video Files (*.mp4 *.avi);;All Files (*)"
        )

        if file_path:
            path = Path(file_path)
            if self.poses_panel.load_video(path):
                # Remove old player if exists
                if 'poses' in self.sync_manager.players:
                    del self.sync_manager.players['poses']

                self.sync_manager.add_player('poses', self.poses_panel.get_player())
                self.poses_panel.set_title(f"3D Poses - {path.name}")
                self._update_info()
                print(f"✓ Loaded poses: {path.name}")

    def _load_tracking_file(self):
        """Load pitch tracking animation from file dialog."""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Load Pitch Tracking Animation",
            str(self.results_dir),
            "Video Files (*.mp4 *.avi);;All Files (*)"
        )

        if file_path:
            path = Path(file_path)
            if self.tracking_panel.load_video(path):
                # Remove old player if exists
                if 'tracking' in self.sync_manager.players:
                    del self.sync_manager.players['tracking']

                self.sync_manager.add_player('tracking', self.tracking_panel.get_player())
                self.tracking_panel.set_title(f"Pitch Tracking - {path.name}")
                self._update_info()
                print(f"✓ Loaded tracking: {path.name}")

    def keyPressEvent(self, event):
        """Handle keyboard events."""
        key = event.key()

        if key == Qt.Key.Key_Space:
            self._toggle_playback()
        elif key == Qt.Key.Key_Left:
            self._step_backward()
        elif key == Qt.Key.Key_Right:
            self._step_forward()
        elif key == Qt.Key.Key_Home:
            self._reset()
        elif key == Qt.Key.Key_End:
            self.sync_manager.seek(self.sync_manager.get_duration())
        elif key == Qt.Key.Key_Up:
            current_speed = self.control_panel.speed_spinbox.value()
            self.control_panel.speed_spinbox.setValue(min(3.0, current_speed + 0.1))
        elif key == Qt.Key.Key_Down:
            current_speed = self.control_panel.speed_spinbox.value()
            self.control_panel.speed_spinbox.setValue(max(0.1, current_speed - 0.1))
        elif key == Qt.Key.Key_F:
            self._toggle_fullscreen()
        else:
            super().keyPressEvent(event)

    def closeEvent(self, event):
        """Handle window close event."""
        self.sync_manager.cleanup()
        self.original_panel.cleanup()
        self.poses_panel.cleanup()
        self.tracking_panel.cleanup()
        event.accept()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="FIFA Skeletal Tracking Synchronized Viewer"
    )
    parser.add_argument(
        '--sequence',
        type=str,
        help='Sequence name to load (e.g., ARG_CRO_220001)'
    )

    args = parser.parse_args()

    app = QApplication(sys.argv)
    app.setApplicationName("FIFA Skeletal Tracking Viewer")

    viewer = SynchronizedViewer(args)
    viewer.show()

    sys.exit(app.exec())


if __name__ == '__main__':
    main()
