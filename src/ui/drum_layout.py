"""Dark themed visual drum interface rendered with OpenCV."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import cv2
import numpy as np


Color = tuple[int, int, int]


@dataclass(frozen=True)
class DrumPad:
    """A labelled circular pad in the drum interface.

    This class is intentionally visual-only. It has no sound playback or hit
    detection responsibilities.
    """

    label: str
    color: Color

    def draw(self, canvas: np.ndarray, center: tuple[int, int], radius: int) -> None:
        """Draw the pad and its centred label onto ``canvas``."""
        cv2.circle(canvas, center, radius + 5, (20, 24, 34), cv2.FILLED, cv2.LINE_AA)
        cv2.circle(canvas, center, radius, self.color, cv2.FILLED, cv2.LINE_AA)
        cv2.circle(canvas, center, radius, (225, 230, 240), 2, cv2.LINE_AA)
        cv2.circle(canvas, (center[0] - radius // 3, center[1] - radius // 3), radius // 4, (255, 255, 255), 1, cv2.LINE_AA)

        font = cv2.FONT_HERSHEY_DUPLEX
        text_size, _ = cv2.getTextSize(self.label, font, 0.55, 1)
        text_origin = (center[0] - text_size[0] // 2, center[1] + text_size[1] // 2)
        cv2.putText(canvas, self.label, text_origin, font, 0.55, (248, 249, 252), 1, cv2.LINE_AA)


class DrumLayout:
    """Compose a webcam preview with a modern, eight-pad drum surface."""

    _PANEL_WIDTH: Final[int] = 520
    _BACKGROUND: Final[Color] = (12, 15, 23)
    _PANEL_BACKGROUND: Final[Color] = (24, 29, 42)
    _PADS: Final[tuple[DrumPad, ...]] = (
        DrumPad("KICK", (85, 91, 221)),
        DrumPad("SNARE", (81, 159, 246)),
        DrumPad("HI-HAT", (65, 198, 177)),
        DrumPad("CLAP", (102, 187, 106)),
        DrumPad("TOM 1", (88, 139, 222)),
        DrumPad("TOM 2", (159, 122, 234)),
        DrumPad("CRASH", (81, 170, 212)),
        DrumPad("RIDE", (78, 153, 191)),
    )

    def render(self, webcam_frame: np.ndarray) -> np.ndarray:
        """Return a dashboard containing the visible webcam feed and drum pads."""
        frame_height, frame_width = webcam_frame.shape[:2]
        canvas = np.full((frame_height, frame_width + self._PANEL_WIDTH, 3), self._BACKGROUND, dtype=np.uint8)
        canvas[:, :frame_width] = webcam_frame

        panel_left = frame_width
        cv2.rectangle(canvas, (panel_left, 0), (canvas.shape[1], frame_height), self._PANEL_BACKGROUND, cv2.FILLED)
        cv2.line(canvas, (panel_left, 0), (panel_left, frame_height), (62, 71, 91), 1, cv2.LINE_AA)
        self._draw_header(canvas, panel_left)
        self._draw_pads(canvas, panel_left, frame_height)
        return canvas

    @staticmethod
    def _draw_header(canvas: np.ndarray, panel_left: int) -> None:
        """Draw the interface title and visual-only status message."""
        cv2.putText(canvas, "FINGERDRUM", (panel_left + 32, 47), cv2.FONT_HERSHEY_DUPLEX, 0.8, (239, 243, 250), 1, cv2.LINE_AA)
        cv2.putText(canvas, "DRUM INTERFACE", (panel_left + 34, 73), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (146, 157, 181), 1, cv2.LINE_AA)
        cv2.line(canvas, (panel_left + 32, 92), (canvas.shape[1] - 32, 92), (62, 71, 91), 1, cv2.LINE_AA)

    def _draw_pads(self, canvas: np.ndarray, panel_left: int, frame_height: int) -> None:
        """Lay out eight circular pads in a responsive four-by-two grid."""
        columns, rows = 4, 2
        content_top, content_bottom = 116, 26
        panel_width = canvas.shape[1] - panel_left
        available_height = max(frame_height - content_top - content_bottom, 1)
        cell_width = panel_width // columns
        cell_height = available_height // rows
        radius = max(12, min(cell_width, cell_height) // 3)

        for index, pad in enumerate(self._PADS):
            column = index % columns
            row = index // columns
            center = (
                panel_left + cell_width * column + cell_width // 2,
                content_top + cell_height * row + cell_height // 2,
            )
            pad.draw(canvas, center, radius)
