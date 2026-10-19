"""Time-based flash state for confirmed drum hits."""

from __future__ import annotations

from dataclasses import dataclass

from drum.hit_detector import HitEvent


@dataclass
class _PadFlash:
    age: float = 0.0


class HitAnimationManager:
    """Track short, non-blocking pad flashes created by real hit events."""

    FLASH_DURATION = 0.28

    def __init__(self) -> None:
        self._flashes: dict[str, _PadFlash] = {}

    def register_hit(self, hit_event: HitEvent) -> None:
        """Start or restart the flash for the hit pad."""
        self._flashes[hit_event.pad_id.upper()] = _PadFlash()

    def update(self, delta_time: float) -> None:
        """Advance flashes using elapsed wall-clock time rather than frame count."""
        elapsed = max(delta_time, 0.0)
        for pad_id, flash in tuple(self._flashes.items()):
            flash.age += elapsed
            if flash.age >= self.FLASH_DURATION:
                del self._flashes[pad_id]

    def intensity_for(self, pad_id: str) -> float:
        """Return a smooth 0-to-1 flash intensity for the requested pad."""
        flash = self._flashes.get(pad_id.upper())
        if flash is None:
            return 0.0
        progress = min(flash.age / self.FLASH_DURATION, 1.0)
        return (1.0 - progress) ** 2
