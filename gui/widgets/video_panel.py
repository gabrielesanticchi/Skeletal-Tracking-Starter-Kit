"""Video display panel widget using native Qt video playback."""

from pathlib import Path
from typing import Optional

from PyQt6.QtWidgets import QVBoxLayout, QWidget, QLabel
from PyQt6.QtCore import Qt, QUrl
from PyQt6.QtMultimedia import QMediaPlayer, QAudioOutput
from PyQt6.QtMultimediaWidgets import QVideoWidget


class VideoPanel(QWidget):
    """Widget for displaying video using native Qt video playback."""

    def __init__(self, title: str = "", parent=None):
        """
        Initialize the video panel.

        Args:
            title: Title for the panel
            parent: Parent widget
        """
        super().__init__(parent)
        self.title = title
        self.player = QMediaPlayer()
        self.audio_output = QAudioOutput()
        self.audio_output.setMuted(True)  # Mute to avoid multiple audio streams
        self.player.setAudioOutput(self.audio_output)
        self._init_ui()

    def _init_ui(self):
        """Initialize the UI components."""
        layout = QVBoxLayout()
        layout.setContentsMargins(2, 2, 2, 2)
        layout.setSpacing(2)

        # Title label
        if self.title:
            self.title_label = QLabel(self.title)
            self.title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self.title_label.setStyleSheet(
                "font-weight: bold; font-size: 11px; padding: 3px;"
            )
            layout.addWidget(self.title_label)

        # Video widget
        self.video_widget = QVideoWidget()
        self.video_widget.setStyleSheet("background-color: black;")
        self.video_widget.setMinimumSize(320, 240)
        self.player.setVideoOutput(self.video_widget)
        layout.addWidget(self.video_widget, stretch=1)

        self.setLayout(layout)

    def load_video(self, video_path: Path) -> bool:
        """
        Load a video file.

        Args:
            video_path: Path to the video file

        Returns:
            True if successful, False otherwise
        """
        if not video_path.exists():
            return False

        url = QUrl.fromLocalFile(str(video_path.absolute()))
        self.player.setSource(url)
        return True

    def get_player(self) -> QMediaPlayer:
        """Get the media player."""
        return self.player

    def clear(self):
        """Clear the video display."""
        self.player.stop()
        self.player.setSource(QUrl())

    def set_title(self, title: str):
        """Update the panel title."""
        self.title = title
        if hasattr(self, 'title_label'):
            self.title_label.setText(title)

    def cleanup(self):
        """Cleanup resources."""
        self.player.stop()
        self.player.setSource(QUrl())
