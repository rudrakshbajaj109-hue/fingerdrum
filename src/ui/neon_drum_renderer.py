"""Cyberpunk neon drum pad rendering — optimised for smooth frame rates.

Uses a single overlay per pad instead of 6-8 separate ROI copies.
"""

from __future__ import annotations

import math
import time
from typing import Iterable, Mapping

import cv2
import numpy as np

from .drum_layout import DrumLayout
from .hit_animation_manager import HitAnimationManager


# Vivid neon colors per instrument (BGR)
_NEON_COLORS: Mapping[str, tuple[int, int, int]] = {
    "KICK":    (255, 100, 40),     # Electric blue
    "SNARE":   (220, 120, 255),    # Hot magenta/purple
    "HI-HAT":  (0, 255, 220),      # Neon cyan
    "CLAP":    (80, 200, 255),     # Warm amber
    "TOM 1":   (80, 255, 140),     # Neon green
    "TOM 2":   (255, 160, 60),     # Deep blue
    "CRASH":   (0, 220, 255),      # Neon yellow
    "RIDE":    (230, 130, 230),    # Neon pink
}
_FALLBACK_COLOR = (200, 200, 200)


class NeonDrumRenderer:
    """Render on-camera pads with neon glow aesthetic and hit feedback.

    Each pad is drawn with a single overlay → blend pass (instead of
    multiple per-layer ROI copies) for smooth performance.
    """

    _HOVER_SCALE = 1.05
    _FLASH_SCALE = 1.08

    def __init__(self, layout: DrumLayout, animations: HitAnimationManager) -> None:
        self._layout = layout
        self._animations = animations
        self._boot_time = time.perf_counter()

    def draw(
        self,
        frame: np.ndarray,
        fingertip_positions: Iterable[tuple[float, float]],
    ) -> None:
        """Draw all neon pads over *frame* with hover/hit feedback."""
        frame_height, frame_width = frame.shape[:2]
        regions = self._layout.pad_regions((frame_width, frame_height))
        fingertips = tuple(fingertip_positions)
        elapsed = time.perf_counter() - self._boot_time

        for pad in self._layout.pads:
            center, radius = regions[pad.label]
            hovered = any(
                self._contains(center, radius, pt)
                for pt in fingertips
            )
            flash = self._animations.intensity_for(pad.label)
            scale = (
                self._FLASH_SCALE if flash > 0.0
                else self._HOVER_SCALE if hovered
                else 1.0
            )
            draw_radius = int(radius * scale)
            neon = _NEON_COLORS.get(pad.label, _FALLBACK_COLOR)

            self._draw_neon_pad(
                frame, pad.label, neon, center, draw_radius,
                hovered, flash, elapsed,
            )

    # ------------------------------------------------------------------ #
    #  Single pad rendering (single-overlay approach)
    # ------------------------------------------------------------------ #

    def _draw_neon_pad(
        self,
        frame: np.ndarray,
        label: str,
        color: tuple[int, int, int],
        center: tuple[int, int],
        radius: int,
        hovered: bool,
        flash: float,
        elapsed: float,
    ) -> None:
        """Render one pad using a single local overlay for all layers."""
        response = max(float(hovered), flash)

        # Subtle idle pulse
        pulse = 0.5 + 0.5 * math.sin(elapsed * 2.0 + hash(label) % 10)

        # Work in a local ROI for performance
        bloom_r = radius + int(10 + 6 * response)
        margin = bloom_r + 6
        fh, fw = frame.shape[:2]
        x0 = max(center[0] - margin, 0)
        x1 = min(center[0] + margin + 1, fw)
        y0 = max(center[1] - margin, 0)
        y1 = min(center[1] + margin + 1, fh)
        if x0 >= x1 or y0 >= y1:
            return

        roi = frame[y0:y1, x0:x1]
        overlay = roi.copy()
        lc = (center[0] - x0, center[1] - y0)

        # --- Dark backdrop disc ---
        cv2.circle(overlay, lc, radius + 4, (8, 10, 18), cv2.FILLED, cv2.LINE_AA)
        cv2.circle(overlay, lc, radius, (14, 18, 26), cv2.FILLED, cv2.LINE_AA)

        # --- Neon fill ---
        fill_alpha = 0.06 + 0.04 * pulse + 0.18 * response
        neon_fill = roi.copy()
        cv2.circle(neon_fill, lc, radius, color, cv2.FILLED, cv2.LINE_AA)
        cv2.addWeighted(neon_fill, fill_alpha, overlay, 1.0 - fill_alpha, 0, dst=overlay)

        # --- Outer neon border ---
        border_alpha = 0.40 + 0.40 * response + 0.08 * pulse
        cv2.circle(overlay, lc, radius, color, 2, cv2.LINE_AA)

        # --- Outer bloom ring ---
        bloom_local_r = bloom_r
        cv2.circle(overlay, lc, bloom_local_r, color, 1, cv2.LINE_AA)

        # --- Flash highlight on hit ---
        if flash > 0.0:
            white = (
                min(255, color[0] + 100),
                min(255, color[1] + 100),
                min(255, color[2] + 100),
            )
            flash_overlay = overlay.copy()
            cv2.circle(flash_overlay, lc, radius - 2, white, cv2.FILLED, cv2.LINE_AA)
            cv2.addWeighted(flash_overlay, 0.22 * flash, overlay, 1.0 - 0.22 * flash, 0, dst=overlay)

        # --- Blend the single overlay back ---
        blend = 0.70 + 0.12 * response + 0.05 * pulse
        cv2.addWeighted(overlay, min(blend, 0.92), roi, 1.0 - min(blend, 0.92), 0.0, dst=roi)

        # --- Label (direct draw, no overlay needed) ---
        self._draw_label(frame, label, color, center, radius, response)

    @staticmethod
    def _draw_label(
        frame: np.ndarray,
        label: str,
        color: tuple[int, int, int],
        center: tuple[int, int],
        radius: int,
        response: float,
    ) -> None:
        """Draw a label centred inside the pad with neon colouring."""
        text_scale = max(0.40, min(0.58, radius / 110))
        thickness = 1
        font = cv2.FONT_HERSHEY_DUPLEX
        text_size, _ = cv2.getTextSize(label, font, text_scale, thickness)
        origin = (
            center[0] - text_size[0] // 2,
            center[1] + text_size[1] // 2,
        )

        text_color = (
            min(255, int(color[0] * 0.35 + 155 + 50 * response)),
            min(255, int(color[1] * 0.35 + 155 + 50 * response)),
            min(255, int(color[2] * 0.35 + 155 + 50 * response)),
        )
        cv2.putText(frame, label, origin, font, text_scale, text_color, thickness, cv2.LINE_AA)

    # ------------------------------------------------------------------ #
    #  Helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def _contains(
        center: tuple[int, int],
        radius: int,
        point: tuple[float, float],
    ) -> bool:
        dx = point[0] - center[0]
        dy = point[1] - center[1]
        return dx * dx + dy * dy <= radius * radius
