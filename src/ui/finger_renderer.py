"""Render glowing cursors for all detected index fingers."""

from __future__ import annotations

import cv2
import numpy as np


class FingerRenderer:
    """Draw one glowing cursor for each detected index finger."""

    def draw(
        self,
        frame: np.ndarray,
        fingertip_positions: tuple[tuple[float, float], ...],
    ) -> None:

        for x, y in fingertip_positions:

            center = (int(x), int(y))

            overlay = frame.copy()

            # Outer glow
            cv2.circle(
                overlay,
                center,
                20,
                (112, 207, 241),
                -1,
                cv2.LINE_AA,
            )

            cv2.addWeighted(
                overlay,
                0.12,
                frame,
                0.85,
                0,
                frame,
            )

            # White ring
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
                (120, 197, 242),
                -1,
                cv2.LINE_AA,
            )

            # Highlight
            cv2.circle(
                frame,
                (center[0] - 3, center[1] - 3),
                2,
                (255, 255, 255),
                -1,
                cv2.LINE_AA,
            )
