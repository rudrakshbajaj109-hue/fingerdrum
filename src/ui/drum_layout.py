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
        DrumPad("KICK", (72, 145, 232)),
        DrumPad("SNARE", (194, 126, 222)),
        DrumPad("HI-HAT", (105, 204, 218)),
        DrumPad("CLAP", (198, 210, 224)),
        DrumPad("TOM 1", (105, 190, 157)),
        DrumPad("TOM 2", (115, 157, 224)),
        DrumPad("CRASH", (105, 188, 236)),
        DrumPad("RIDE", (176, 148, 215)),
    )

    @property
    def pads(self) -> tuple[DrumPad, ...]:
        """Return the immutable pad definitions in display order."""
        return self._PADS

    def pad_regions(self, frame_size: tuple[int, int]) -> dict[str, PadRegion]:
      """Compact drum kit layout — tighter grouping, lower on screen."""

      frame_width, frame_height = frame_size

      base_radius = int(min(frame_width, frame_height) * 0.068)

      return {
          "CRASH": (
              (int(frame_width * 0.40), int(frame_height * 0.34)),
              int(base_radius * 0.90),
          ),

          "RIDE": (
              (int(frame_width * 0.60), int(frame_height * 0.34)),
              int(base_radius * 0.90),
          ),

          "HI-HAT": (
              (int(frame_width * 0.31), int(frame_height * 0.50)),
              int(base_radius * 0.95),
          ),

          "TOM 1": (
              (int(frame_width * 0.45), int(frame_height * 0.49)),
              base_radius,
          ),

          "TOM 2": (
              (int(frame_width * 0.55), int(frame_height * 0.49)),
              base_radius,
          ),

          "SNARE": (
              (int(frame_width * 0.40), int(frame_height * 0.65)),
              int(base_radius * 1.10),
          ),

          "CLAP": (
              (int(frame_width * 0.58), int(frame_height * 0.64)),
              int(base_radius * 0.90),
          ),

          "KICK": (
              (int(frame_width * 0.50), int(frame_height * 0.82)),
              int(base_radius * 1.35),
          ),
      }
