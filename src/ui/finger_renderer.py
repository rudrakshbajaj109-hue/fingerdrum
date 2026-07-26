"""Render a premium glowing fingertip cursor."""

from __future__ import annotations

import cv2
import numpy as np


class FingerRenderer:
    """Draw a glowing cursor that follows the index fingertip."""

    def __init__(self) -> None:
        pass

    def draw(
        self,
        frame: np.ndarray,
        fingertip_positions: tuple[tuple[float, float], ...],
    ) -> None:
        """Draw the cursor."""

        if not fingertip_positions:
            return

        x, y = fingertip_positions[0]
        center = (int(x), int(y))

        overlay = frame.copy()

        # Outer glow
        cv2.circle(
            overlay,
            center,
            24,
            (0, 255, 255),
            -1,
            cv2.LINE_AA,
        )

        cv2.addWeighted(
            overlay,
            0.15,
            frame,
            0.85,
            0,
            frame,
        )

        # White outer ring
        cv2.circle(
            frame,
            center,
            12,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        # Golden center
        cv2.circle(
            frame,
            center,
            8,
            (0, 220, 255),
            -1,
            cv2.LINE_AA,
        )

        # White highlight
        cv2.circle(
            frame,
            (center[0] - 3, center[1] - 3),
            2,
            (255, 255, 255),
            -1,
            cv2.LINE_AA,
        )