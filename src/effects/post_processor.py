"""Full-frame post-processing: vignette, colour grading, and scan lines."""

from __future__ import annotations

import cv2
import numpy as np


class PostProcessor:
    """Apply cinematic post-processing to make the neon aesthetic pop."""

    def __init__(self) -> None:
        self._vignette_cache: dict[tuple[int, int], np.ndarray] = {}

    def process(self, frame: np.ndarray) -> np.ndarray:
        """Apply vignette + subtle blue colour grade in-place and return *frame*."""
        # Slight overall darken to make neons pop
        frame = self._apply_vignette(frame)
        frame = self._color_grade(frame)
        return frame

    # ------------------------------------------------------------------ #
    #  Vignette
    # ------------------------------------------------------------------ #

    def _apply_vignette(self, frame: np.ndarray) -> np.ndarray:
        """Darken edges with a smooth radial gradient (cached per resolution)."""
        h, w = frame.shape[:2]
        key = (w, h)

        if key not in self._vignette_cache:
            self._vignette_cache[key] = self._build_vignette(w, h)

        mask = self._vignette_cache[key]

        # OpenCV applies the cached mask in one pass, avoiding multiple
        # full-frame float conversions and temporary arrays each frame.
        cv2.multiply(frame, mask, dst=frame, dtype=cv2.CV_8U)
        return frame

    @staticmethod
    def _build_vignette(w: int, h: int) -> np.ndarray:
        """Pre-compute a vignette mask — 1.0 at centre, ~0.35 at extreme corners."""
        y = np.linspace(-1, 1, h).astype(np.float32)
        x = np.linspace(-1, 1, w).astype(np.float32)
        xv, yv = np.meshgrid(x, y)
        dist = np.sqrt(xv ** 2 + yv ** 2)
        # Smooth falloff: centre=1.0, edge≈0.55, corner≈0.35
        mask = 1.0 - 0.55 * np.clip(dist / 1.42, 0.0, 1.0) ** 1.8
        return np.repeat(mask[:, :, np.newaxis].astype(np.float32), 3, axis=2)

    # ------------------------------------------------------------------ #
    #  Color grading
    # ------------------------------------------------------------------ #

    @staticmethod
    def _color_grade(frame: np.ndarray) -> np.ndarray:
        """Shift shadows toward cool blue and slightly desaturate mid-tones."""
        # Very gentle blue push in shadows
        cv2.add(frame, (6, 0, 0), dst=frame)
        cv2.subtract(frame, (0, 0, 3), dst=frame)
        return frame
