"""Control panel widget for video playback."""

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
    QSlider, QLabel, QDoubleSpinBox, QFileDialog, QComboBox
)
from PyQt6.QtCore import Qt, pyqtSignal


class ControlPanel(QWidget):
    """Widget for video playback controls."""

    # Signals
    play_pause_clicked = pyqtSignal()
    seek_requested = pyqtSignal(int)  # position in ms
    speed_changed = pyqtSignal(float)  # speed multiplier
    step_forward_clicked = pyqtSignal()
    step_backward_clicked = pyqtSignal()
    reset_clicked = pyqtSignal()
    sequence_selected = pyqtSignal(str)  # sequence name
    load_poses_clicked = pyqtSignal()
    load_tracking_clicked = pyqtSignal()

    def __init__(self, parent=None):
        """
        Initialize the control panel.

        Args:
            parent: Parent widget
        """
        super().__init__(parent)
        self.duration_ms = 0  # Duration in milliseconds
        self._init_ui()

    def _init_ui(self):
        """Initialize the UI components."""
        layout = QVBoxLayout()
        layout.setContentsMargins(5, 5, 5, 5)
        layout.setSpacing(8)

        # Sequence selection
        seq_layout = QHBoxLayout()
        seq_label = QLabel("Sequence:")
        self.sequence_combo = QComboBox()
        self.sequence_combo.setMinimumWidth(200)
        self.sequence_combo.currentTextChanged.connect(
            lambda text: self.sequence_selected.emit(text) if text else None
        )
        seq_layout.addWidget(seq_label)
        seq_layout.addWidget(self.sequence_combo)
        seq_layout.addStretch()
        layout.addLayout(seq_layout)

        # Timeline slider
        slider_layout = QVBoxLayout()
        slider_layout.setSpacing(2)

        slider_label = QLabel("Timeline:")
        slider_layout.addWidget(slider_label)

        self.timeline_slider = QSlider(Qt.Orientation.Horizontal)
        self.timeline_slider.setMinimum(0)
        self.timeline_slider.setMaximum(100)
        self.timeline_slider.valueChanged.connect(
            lambda value: self.seek_requested.emit(value)
        )
        slider_layout.addWidget(self.timeline_slider)

        layout.addLayout(slider_layout)

        # Frame info
        info_layout = QHBoxLayout()
        self.frame_label = QLabel("Frame: 0 / 0")
        self.duration_label = QLabel("Time: 0.0s / 0.0s")
        info_layout.addWidget(self.frame_label)
        info_layout.addStretch()
        info_layout.addWidget(self.duration_label)
        layout.addLayout(info_layout)

        # Playback controls
        controls_layout = QHBoxLayout()

        # Reset button
        self.reset_btn = QPushButton("⏮ Reset")
        self.reset_btn.clicked.connect(self.reset_clicked.emit)
        controls_layout.addWidget(self.reset_btn)

        # Step backward button
        self.step_back_btn = QPushButton("◀ Step Back")
        self.step_back_btn.clicked.connect(self.step_backward_clicked.emit)
        controls_layout.addWidget(self.step_back_btn)

        # Play/Pause button
        self.play_pause_btn = QPushButton("▶ Play")
        self.play_pause_btn.setMinimumWidth(100)
        self.play_pause_btn.clicked.connect(self.play_pause_clicked.emit)
        controls_layout.addWidget(self.play_pause_btn)

        # Step forward button
        self.step_fwd_btn = QPushButton("Step Fwd ▶")
        self.step_fwd_btn.clicked.connect(self.step_forward_clicked.emit)
        controls_layout.addWidget(self.step_fwd_btn)

        controls_layout.addStretch()
        layout.addLayout(controls_layout)

        # Speed and scale controls
        controls_layout2 = QHBoxLayout()

        # Speed control
        speed_label = QLabel("Speed:")
        self.speed_spinbox = QDoubleSpinBox()
        self.speed_spinbox.setMinimum(0.1)
        self.speed_spinbox.setMaximum(3.0)
        self.speed_spinbox.setSingleStep(0.1)
        self.speed_spinbox.setValue(1.0)
        self.speed_spinbox.setSuffix("x")
        self.speed_spinbox.setMaximumWidth(80)
        self.speed_spinbox.valueChanged.connect(
            lambda value: self.speed_changed.emit(value)
        )
        controls_layout2.addWidget(speed_label)
        controls_layout2.addWidget(self.speed_spinbox)
        controls_layout2.addStretch()
        layout.addLayout(controls_layout2)

        # Load buttons
        load_layout = QHBoxLayout()

        self.load_poses_btn = QPushButton("📂 Load 3D Poses")
        self.load_poses_btn.clicked.connect(self.load_poses_clicked.emit)
        load_layout.addWidget(self.load_poses_btn)

        self.load_tracking_btn = QPushButton("📂 Load Pitch Tracking")
        self.load_tracking_btn.clicked.connect(self.load_tracking_clicked.emit)
        load_layout.addWidget(self.load_tracking_btn)

        load_layout.addStretch()
        layout.addLayout(load_layout)

        # Sequence info
        self.sequence_info_label = QLabel("")
        self.sequence_info_label.setWordWrap(True)
        self.sequence_info_label.setStyleSheet(
            "font-size: 10px; color: #666; padding: 5px;"
        )
        layout.addWidget(self.sequence_info_label)

        layout.addStretch()
        self.setLayout(layout)

    def set_play_state(self, is_playing: bool):
        """
        Update the play/pause button state.

        Args:
            is_playing: Whether video is currently playing
        """
        if is_playing:
            self.play_pause_btn.setText("⏸ Pause")
        else:
            self.play_pause_btn.setText("▶ Play")

    def set_total_frames(self, duration_ms: int):
        """
        Set the total duration.

        Args:
            duration_ms: Total duration in milliseconds
        """
        self.duration_ms = duration_ms
        self.timeline_slider.setMaximum(max(0, duration_ms))
        self.update_frame_info(0, duration_ms)

    def update_frame_info(self, position_ms: int, duration_ms: int = None):
        """
        Update the position information.

        Args:
            position_ms: Current position in milliseconds
            duration_ms: Total duration in milliseconds (optional)
        """
        if duration_ms is not None:
            self.duration_ms = duration_ms

        current_sec = position_ms / 1000.0
        total_sec = self.duration_ms / 1000.0

        self.frame_label.setText(
            f"Time: {current_sec:.1f}s / {total_sec:.1f}s"
        )

        # Update slider without triggering signal
        self.timeline_slider.blockSignals(True)
        self.timeline_slider.setValue(position_ms)
        self.timeline_slider.blockSignals(False)

    def set_sequence_info(self, info: str):
        """
        Set the sequence information text.

        Args:
            info: Information text to display
        """
        self.sequence_info_label.setText(info)

    def add_sequence(self, sequence_name: str):
        """
        Add a sequence to the dropdown.

        Args:
            sequence_name: Name of the sequence to add
        """
        if self.sequence_combo.findText(sequence_name) == -1:
            self.sequence_combo.addItem(sequence_name)

    def set_current_sequence(self, sequence_name: str):
        """
        Set the current sequence in the dropdown.

        Args:
            sequence_name: Name of the sequence to select
        """
        index = self.sequence_combo.findText(sequence_name)
        if index >= 0:
            self.sequence_combo.setCurrentIndex(index)

    def clear_sequences(self):
        """Clear all sequences from the dropdown."""
        self.sequence_combo.clear()
