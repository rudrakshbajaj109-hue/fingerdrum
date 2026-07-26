"""Low-overhead glow and ripple effects rendered over the drum interface."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import cv2
import numpy as np

from drum.hit_detector import HitEvent


Color = tuple[int, int, int]
PadRegion = tuple[tuple[int, int], int]


@dataclass
class _Glow:
    """A short-lived highlight for one pad."""

    center: tuple[int, int]
    radius: int
    color: Color
    intensity: float
    age: float = 0.0


@dataclass
class _Ripple:
    """A single expanding ring emitted by a drum strike."""

    center: tuple[int, int]
    base_radius: int
    color: Color
    intensity: float
    age: float = 0.0


class VisualEffectsManager:
    """Maintain and render non-blocking pad glow and ripple animations."""

    _GLOW_DURATION = 0.18
    _RIPPLE_DURATION = 0.42
    _VELOCITY_FOR_MAX_INTENSITY = 2_500.0
    _INSTRUMENT_COLORS: Mapping[str, Color] = {
        "KICK": (0, 145, 255),       # Orange
        "SNARE": (255, 150, 70),     # Blue
        "HI-HAT": (0, 235, 255),     # Yellow
        "CRASH": (255, 255, 0),      # Cyan
        "RIDE": (215, 105, 190),     # Purple
        "SPLASH": (255, 255, 255),   # White
        "CLAP": (255, 255, 255),     # The current layout uses CLAP for splash.
        "TOM 1": (85, 220, 90),      # Green
        "TOM 2": (70, 70, 245),      # Red
    }
    _DEFAULT_COLOR: Color = (235, 235, 235)

    def __init__(self, pad_regions: Mapping[str, PadRegion] | None = None) -> None:
        self._pad_regions: dict[str, PadRegion] = dict(pad_regions or {})
        self._glows: dict[str, _Glow] = {}
        self._ripples: list[_Ripple] = []

    def set_pad_regions(self, pad_regions: Mapping[str, PadRegion]) -> None:
        """Set the current on-screen pad geometry without creating effects."""
        self._pad_regions = dict(pad_regions)

    def register_hit(self, hit_event: HitEvent) -> None:
        """Start glow and ripple feedback for a recorded pad hit."""
        pad_id = hit_event.pad_id.upper()
        region = self._pad_regions.get(pad_id)
        if region is None:
            return

        center, radius = region
        intensity = self._intensity_from_velocity(hit_event.downward_velocity)
        color = self._INSTRUMENT_COLORS.get(pad_id, self._DEFAULT_COLOR)
        self._glows[pad_id] = _Glow(center, radius, color, intensity)
        self._ripples.append(_Ripple(center, radius, color, intensity))

    def update(self, delta_time: float) -> None:
        """Advance effects by elapsed time; this is independent of render FPS."""
        elapsed = max(delta_time, 0.0)
        for pad_id, glow in tuple(self._glows.items()):
            glow.age += elapsed
            if glow.age >= self._GLOW_DURATION:
                del self._glows[pad_id]

        active_ripples: list[_Ripple] = []
        for ripple in self._ripples:
            ripple.age += elapsed
            if ripple.age < self._RIPPLE_DURATION:
                active_ripples.append(ripple)
        self._ripples = active_ripples

    def draw(self, frame: np.ndarray) -> None:
        """Draw the active glow layer followed by all active ripple layers."""
        for glow in self._glows.values():
            progress = 1.0 - glow.age / self._GLOW_DURATION
            self._draw_glow(frame, glow, max(progress, 0.0))

        for ripple in self._ripples:
            progress = ripple.age / self._RIPPLE_DURATION
            self._draw_ripple(frame, ripple, min(max(progress, 0.0), 1.0))

    def _draw_glow(self, frame: np.ndarray, glow: _Glow, fade: float) -> None:
        """Blend a soft coloured pad highlight using small local regions only."""
        strength = glow.intensity * fade
        self._blend_circle(frame, glow.center, glow.radius, glow.color, 0.13 * strength, cv2.FILLED)
        self._blend_circle(frame, glow.center, glow.radius + 6, glow.color, 0.48 * strength, 3)
        self._blend_circle(frame, glow.center, glow.radius + 12, glow.color, 0.16 * strength, 2)

    def _draw_ripple(self, frame: np.ndarray, ripple: _Ripple, progress: float) -> None:
        """Blend an outward-moving, fading ring."""
        radius = int(6 + progress * (ripple.base_radius * 1.45))
        alpha = 0.70 * ripple.intensity * (1.0 - progress) ** 2
        self._blend_circle(frame, ripple.center, radius, ripple.color, alpha, 2)

    @staticmethod
    def _blend_circle(
        frame: np.ndarray,
        center: tuple[int, int],
        radius: int,
        color: Color,
        alpha: float,
        thickness: int,
    ) -> None:
        """Alpha-blend one circle into a bounded ROI to limit CPU work."""
        if alpha <= 0.0 or radius <= 0:
            return

        height, width = frame.shape[:2]
        x_start = max(center[0] - radius - abs(thickness), 0)
        x_end = min(center[0] + radius + abs(thickness) + 1, width)
        y_start = max(center[1] - radius - abs(thickness), 0)
        y_end = min(center[1] + radius + abs(thickness) + 1, height)
        if x_start >= x_end or y_start >= y_end:
            return

        roi = frame[y_start:y_end, x_start:x_end]
        overlay = roi.copy()
        local_center = center[0] - x_start, center[1] - y_start
        cv2.circle(overlay, local_center, radius, color, thickness, cv2.LINE_AA)
        cv2.addWeighted(overlay, min(alpha, 1.0), roi, 1.0 - min(alpha, 1.0), 0.0, dst=roi)

    @classmethod
    def _intensity_from_velocity(cls, downward_velocity: float) -> float:
        """Map hit speed smoothly to a visible intensity range."""
        normalized = min(max(downward_velocity, 0.0) / cls._VELOCITY_FOR_MAX_INTENSITY, 1.0)
        return 0.35 + normalized * 0.65
