"""Video synchronization manager using Qt media players."""

from typing import Dict
from PyQt6.QtMultimedia import QMediaPlayer
from PyQt6.QtCore import QObject, pyqtSignal


class SyncManager(QObject):
    """Manages synchronized playback of multiple videos using Qt media players."""

    position_changed = pyqtSignal(int)  # Emits position in milliseconds

    def __init__(self):
        """Initialize the synchronization manager."""
        super().__init__()
        self.players: Dict[str, QMediaPlayer] = {}
        self.master_player: QMediaPlayer = None
        self.is_syncing = False

    def add_player(self, name: str, player: QMediaPlayer, is_master: bool = False):
        """
        Add a media player to synchronization.

        Args:
            name: Identifier for the player
            player: QMediaPlayer instance
            is_master: Whether this is the master player
        """
        self.players[name] = player

        if is_master or self.master_player is None:
            self.master_player = player
            # Connect master player position changes
            self.master_player.positionChanged.connect(self._on_position_changed)

    def _on_position_changed(self, position: int):
        """
        Handle master player position changes.

        Args:
            position: Position in milliseconds
        """
        if not self.is_syncing:
            self.position_changed.emit(position)
            self._sync_all(position)

    def _sync_all(self, position: int):
        """
        Sync all players to the given position.

        Args:
            position: Position in milliseconds
        """
        self.is_syncing = True
        for name, player in self.players.items():
            if player != self.master_player:
                # Only sync if difference is significant (>100ms)
                if abs(player.position() - position) > 100:
                    player.setPosition(position)
        self.is_syncing = False

    def play(self):
        """Start playback of all players."""
        for player in self.players.values():
            if player.playbackState() != QMediaPlayer.PlaybackState.PlayingState:
                player.play()

    def pause(self):
        """Pause all players."""
        for player in self.players.values():
            if player.playbackState() == QMediaPlayer.PlaybackState.PlayingState:
                player.pause()

    def stop(self):
        """Stop all players."""
        for player in self.players.values():
            player.stop()

    def seek(self, position: int):
        """
        Seek all players to position.

        Args:
            position: Position in milliseconds
        """
        self.is_syncing = True
        for player in self.players.values():
            player.setPosition(position)
        self.is_syncing = False

    def set_playback_rate(self, rate: float):
        """
        Set playback rate for all players.

        Args:
            rate: Playback rate (0.1 to 3.0)
        """
        for player in self.players.values():
            player.setPlaybackRate(rate)

    def get_duration(self) -> int:
        """
        Get duration from master player.

        Returns:
            Duration in milliseconds
        """
        if self.master_player:
            return self.master_player.duration()
        return 0

    def get_position(self) -> int:
        """
        Get current position from master player.

        Returns:
            Position in milliseconds
        """
        if self.master_player:
            return self.master_player.position()
        return 0

    def is_playing(self) -> bool:
        """Check if master player is playing."""
        if self.master_player:
            return self.master_player.playbackState() == QMediaPlayer.PlaybackState.PlayingState
        return False

    def cleanup(self):
        """Cleanup all players."""
        for player in self.players.values():
            player.stop()
            player.setSource(None)
        self.players.clear()
        self.master_player = None
