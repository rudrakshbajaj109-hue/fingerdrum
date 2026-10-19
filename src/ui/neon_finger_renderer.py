"""Glowing neon finger cursor — lightweight, smooth rendering."""

from __future__ import annotations

import cv2
import numpy as np


class NeonFingerRenderer:
    """Draw vivid, glowing cursors for each detected index finger.

    Uses direct drawing (no per-cursor overlay copies) for smooth performance.
    """

    _CURSOR_COLOR = (0, 255, 220)       # Neon cyan (BGR)
    _CORE_COLOR = (255, 255, 255)       # White hot center

    def draw(
        self,
        frame: np.ndarray,
        fingertip_positions: tuple[tuple[float, float], ...],
    ) -> None:

        for x, y in fingertip_positions:
            center = (int(x), int(y))
            self._draw_neon_cursor(frame, center)

    def _draw_neon_cursor(
        self,
        frame: np.ndarray,
        center: tuple[int, int],
    ) -> None:
        """Lightweight neon cursor — colored ring → white-hot core → highlight."""
        fh, fw = frame.shape[:2]
        cx, cy = center

        if cx < 0 or cy < 0 or cx >= fw or cy >= fh:
            return

        # Soft outer glow (single thin ring, drawn directly)
        cv2.circle(frame, center, 16, self._CURSOR_COLOR, 1, cv2.LINE_AA)

        # Neon ring
        cv2.circle(frame, center, 10, self._CURSOR_COLOR, 2, cv2.LINE_AA)

        # Bright core
        cv2.circle(frame, center, 5, self._CORE_COLOR, -1, cv2.LINE_AA)

        # Tiny specular highlight
        highlight = (cx - 2, cy - 2)
        cv2.circle(frame, highlight, 2, (255, 255, 255), -1, cv2.LINE_AA)
