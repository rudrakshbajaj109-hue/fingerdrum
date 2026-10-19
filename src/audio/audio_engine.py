"""Low-latency polyphonic playback for FingerDrum hit events."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Final

import numpy as np
import pygame
import pygame.sndarray

from drum.hit_detector import HitEvent


pygame.mixer.pre_init(
    frequency=44100,
    size=-16,
    channels=2,
    buffer=256,
)

_ROOM_ECHOES: Final[tuple[tuple[int, float], ...]] = ((35, 0.12), (75, 0.05))


@dataclass(frozen=True)
class AudioEngineConfig:
    """Audio playback configuration."""

    master_volume: float = 1.0

    def __post_init__(self) -> None:
        if not 0.0 <= self.master_volume <= 1.0:
            raise ValueError("Master volume must be between 0.0 and 1.0.")


class AudioEngine:
    """Load and play drum samples."""

    DEFAULT_KIT_PATH: Final[Path] = (
        Path(__file__).resolve().parents[2]
        / "assets"
        / "kits"
        / "Acoustic"
    )

    _PAD_SOUNDS: Final[dict[str, str]] = {
        "KICK": "Kick.wav",
        "SNARE": "Snare.wav",
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
        return self._mixer_ready

    def load_kit(self, path: str | Path) -> None:

        self._sounds.clear()

        if not self._mixer_ready:
            return

        kit_path = Path(path)

        for pad_id, filename in self._PAD_SOUNDS.items():

            sample_path = kit_path / filename

            if not sample_path.is_file():
                continue

            try:
                sound = self._with_short_room_tail(
                    pygame.mixer.Sound(str(sample_path))
                )
                sound.set_volume(self._config.master_volume)
                self._sounds[pad_id] = sound

            except pygame.error:
                continue

    @staticmethod
    def _with_short_room_tail(sound: pygame.mixer.Sound) -> pygame.mixer.Sound:
        """Add two quiet reflections for a slightly fuller, longer drum decay."""
        samples = pygame.sndarray.array(sound).astype(np.float32)
        sample_rate = pygame.mixer.get_init()[0]
        tail_frames = int(sample_rate * _ROOM_ECHOES[-1][0] / 1000)
        extended = np.zeros((len(samples) + tail_frames, *samples.shape[1:]), dtype=np.float32)
        extended[: len(samples)] = samples

        for delay_ms, volume in _ROOM_ECHOES:
            offset = int(sample_rate * delay_ms / 1000)
            extended[offset : offset + len(samples)] += samples * volume

        pcm = np.clip(extended, -32768, 32767).astype(np.int16)
        return pygame.mixer.Sound(buffer=pcm.tobytes())

    def play_hit(self, hit_event: HitEvent) -> None:

        sound = self._sounds.get(hit_event.pad_id.upper())

        if sound is None:
            return

        try:
            sound.play()

        except pygame.error:
            pass

    def stop_all(self) -> None:

        if self._mixer_ready:
            pygame.mixer.stop()

    @staticmethod
    def _initialize_mixer() -> bool:

        try:

            if not pygame.mixer.get_init():
                pygame.mixer.init()

            pygame.mixer.set_num_channels(32)

            return True

        except pygame.error:
            return False
