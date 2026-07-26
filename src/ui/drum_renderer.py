"""Semi-transparent drum pad rendering for the live webcam view."""

from __future__ import annotations

from typing import Iterable

import cv2
import numpy as np

from .drum_layout import DrumLayout
from .hit_animation_manager import HitAnimationManager


class DrumRenderer:
    """Render on-camera pads with hover readiness and confirmed-hit flashes."""

    _HOVER_SCALE = 1.08
    _FLASH_SCALE = 1.12

    def __init__(self, layout: DrumLayout, animations: HitAnimationManager) -> None:
        self._layout = layout
        self._animations = animations

    def draw(self, frame: np.ndarray, fingertip_positions: Iterable[tuple[float, float]]) -> None:
        """Draw all pads over ``frame`` and mark pads containing a fingertip."""
        frame_height, frame_width = frame.shape[:2]
        regions = self._layout.pad_regions((frame_width, frame_height))
        fingertips = tuple(fingertip_positions)

        for pad in self._layout.pads:
            center, radius = regions[pad.label]
            hovered = any(self._contains(center, radius, point) for point in fingertips)
            flash = self._animations.intensity_for(pad.label)
            scale = self._FLASH_SCALE if flash > 0.0 else self._HOVER_SCALE if hovered else 1.0
            draw_radius = int(radius * scale)
            self._draw_pad(frame, pad.label, pad.color, center, draw_radius, hovered, flash)

    def _draw_pad(
        self,
        frame: np.ndarray,
        label: str,
        color: tuple[int, int, int],
        center: tuple[int, int],
        radius: int,
        hovered: bool,
        flash: float,
    ) -> None:
        """Blend one labelled pad, favouring hit feedback over hover state."""
        fill_alpha = 0.24 + (0.16 if hovered else 0.0) + 0.48 * flash
        outline_alpha = 0.68 + 0.30 * max(flash, float(hovered))
        self._blend_circle(frame, center, radius, color, min(fill_alpha, 0.92), cv2.FILLED)
        self._blend_circle(frame, center, radius, color, min(outline_alpha, 1.0), 2)
        if hovered and flash == 0.0:
            self._blend_circle(frame, center, radius + 6, color, 0.45, 2)
        if flash > 0.0:
            self._blend_circle(frame, center, radius + 8, (255, 255, 255), 0.55 * flash, 3)

        text_scale = 0.52 if radius < 60 else 0.62
        text_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_DUPLEX, text_scale, 1)
        text_origin = center[0] - text_size[0] // 2, center[1] + text_size[1] // 2
        cv2.putText(frame, label, text_origin, cv2.FONT_HERSHEY_DUPLEX, text_scale, (250, 250, 250), 1, cv2.LINE_AA)

    @staticmethod
    def _contains(center: tuple[int, int], radius: int, point: tuple[float, float]) -> bool:
        x_offset = point[0] - center[0]
        y_offset = point[1] - center[1]
        return x_offset * x_offset + y_offset * y_offset <= radius * radius

    @staticmethod
    def _blend_circle(
        frame: np.ndarray,
        center: tuple[int, int],
        radius: int,
        color: tuple[int, int, int],
        alpha: float,
        thickness: int,
    ) -> None:
        """Alpha-blend one bounded circle ROI to retain a responsive frame rate."""
        margin = abs(thickness) + 2
        frame_height, frame_width = frame.shape[:2]
        x_start, x_end = max(center[0] - radius - margin, 0), min(center[0] + radius + margin + 1, frame_width)
        y_start, y_end = max(center[1] - radius - margin, 0), min(center[1] + radius + margin + 1, frame_height)
        if x_start >= x_end or y_start >= y_end:
            return

        roi = frame[y_start:y_end, x_start:x_end]
        overlay = roi.copy()
        local_center = center[0] - x_start, center[1] - y_start
        cv2.circle(overlay, local_center, radius, color, thickness, cv2.LINE_AA)
        cv2.addWeighted(overlay, alpha, roi, 1.0 - alpha, 0.0, dst=roi)
