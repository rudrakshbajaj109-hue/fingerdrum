"""Low-latency polyphonic playback for FingerDrum hit events."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Final

import pygame

from drum.hit_detector import HitEvent


pygame.mixer.pre_init(
    frequency=44100,
    size=-16,
    channels=2,
    buffer=256,
)


@dataclass(frozen=True)
class AudioEngineConfig:
    """Velocity-to-volume conversion parameters for sample playback."""

    minimum_volume: float = 0.20
    maximum_volume: float = 1.00
    velocity_for_maximum_volume: float = 2_500.0

    def __post_init__(self) -> None:
        if not 0.0 <= self.minimum_volume <= 1.0:
            raise ValueError("Minimum volume must be between 0.0 and 1.0.")
        if not self.minimum_volume <= self.maximum_volume <= 1.0:
            raise ValueError("Maximum volume must be between minimum volume and 1.0.")
        if self.velocity_for_maximum_volume <= 0:
            raise ValueError("Velocity for maximum volume must be positive.")


class AudioEngine:
    """Load and play drum samples without affecting active sounds."""

    DEFAULT_KIT_PATH: Final[Path] = Path(__file__).resolve().parents[2] / "assets" / "kits" / "Acoustic"
    _PAD_SOUNDS: Final[dict[str, str]] = {
        "KICK": "kick.wav",
        "SNARE": "snare.wav",
        "HI-HAT": "hihat.wav",
        "CLAP": "splash.wav",
        "CRASH": "crash.wav",
        "RIDE": "ride.wav",
        "TOM 1": "tom1.wav",
        "TOM 2": "tom2.wav",
    }

    def __init__(
        self,
        kit_path: str | Path | None = None,
        config: AudioEngineConfig | None = None,
    ) -> None:
        self._config = config or AudioEngineConfig()
        self._sounds: dict[str, pygame.mixer.Sound] = {}
        self._mixer_ready = self._initialize_mixer()
        self.load_kit(kit_path or self.DEFAULT_KIT_PATH)

    @property
    def mixer_ready(self) -> bool:
        """Return whether the mixer was initialized and can play samples."""
        return self._mixer_ready

    def load_kit(self, path: str | Path) -> None:
        """Load available mapped samples from a kit directory.

        Missing or unreadable files are ignored so the visual instrument stays
        usable when a kit is incomplete.
        """
        self._sounds.clear()
        if not self._mixer_ready:
            return

        kit_path = Path(path)
        for pad_id, filename in self._PAD_SOUNDS.items():
            sample_path = kit_path / filename
            if not sample_path.is_file():
                continue
            try:
                self._sounds[pad_id] = pygame.mixer.Sound(str(sample_path))
            except pygame.error:
                continue

    def play_hit(self, hit_event: HitEvent) -> None:
        """Play a mapped sample for ``hit_event`` at velocity-derived volume."""
        sound = self._sounds.get(hit_event.pad_id.upper())
        if sound is None:
            return

        try:
            channel = sound.play()
            if channel is not None:
                channel.set_volume(self._volume_for_velocity(hit_event.downward_velocity))
        except pygame.error:
            return

    def stop_all(self) -> None:
        """Stop every active mixer channel."""
        if self._mixer_ready:
            pygame.mixer.stop()

    def _volume_for_velocity(self, downward_velocity: float) -> float:
        """Linearly interpolate and clamp a downward velocity to playback volume."""
        progress = min(max(downward_velocity, 0.0) / self._config.velocity_for_maximum_volume, 1.0)
        return self._config.minimum_volume + progress * (
            self._config.maximum_volume - self._config.minimum_volume
        )

    @staticmethod
    def _initialize_mixer() -> bool:
        """Initialize pygame's mixer without failing when no audio device exists."""
        try:
            if not pygame.mixer.get_init():
                pygame.mixer.init()
            pygame.mixer.set_num_channels(32)
            return True
        except pygame.error:
            return False
