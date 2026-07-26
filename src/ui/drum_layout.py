"""Pad definitions and responsive on-camera geometry for FingerDrum."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final


Color = tuple[int, int, int]
PadRegion = tuple[tuple[int, int], int]


@dataclass(frozen=True)
class DrumPad:
    """Static visual identity for one circular drum pad."""

    label: str
    color: Color


class DrumLayout:
    """Provide the eight drum pad definitions and their live-camera positions."""

    _PADS: Final[tuple[DrumPad, ...]] = (
        DrumPad("KICK", (0, 145, 255)),
        DrumPad("SNARE", (255, 150, 70)),
        DrumPad("HI-HAT", (0, 235, 255)),
        DrumPad("CLAP", (255, 255, 255)),
        DrumPad("TOM 1", (85, 220, 90)),
        DrumPad("TOM 2", (70, 70, 245)),
        DrumPad("CRASH", (255, 255, 0)),
        DrumPad("RIDE", (215, 105, 190)),
    )

    @property
    def pads(self) -> tuple[DrumPad, ...]:
        """Return the immutable pad definitions in display order."""
        return self._PADS

    def pad_regions(self, frame_size: tuple[int, int]) -> dict[str, PadRegion]:
        """Return pad centers and radii directly in webcam pixel coordinates."""
        frame_width, frame_height = frame_size
        columns, rows = 4, 2
        content_top, content_bottom = 140, 26
        cell_width = frame_width // columns
        cell_height = max((frame_height - content_top - content_bottom) // rows, 1)
        radius = max(12, int(min(cell_width, cell_height) * 0.24))

        return {
            pad.label: (
                (
                    cell_width * (index % columns) + cell_width // 2,
                    content_top + cell_height * (index // columns) + cell_height // 2 + 12,
                ),
                radius,
            )
            for index, pad in enumerate(self._PADS)
        }
