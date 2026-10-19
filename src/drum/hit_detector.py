from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable, Sequence

from .circular_pad import CircularPad


PixelPoint = tuple[float, float]
NormalizedLandmark = tuple[float, float, float]
FingerKey = tuple[int, int]


@dataclass(frozen=True)
class HitEvent:
    hand_id: int
    finger_id: int
    pad_id: str
    timestamp: float
    fingertip_position: PixelPoint


@dataclass
class HitDetectorConfig:
    cooldown_seconds: float = 0.03


class HitDetector:

    def __init__(
        self,
        pads: Iterable[CircularPad],
        config: HitDetectorConfig | None = None,
    ):
        self._pads = tuple(pads)
        self._config = config or HitDetectorConfig()

        self._last_hit_time = defaultdict(float)

        # Stores whether a finger is currently inside each pad
        self._inside = defaultdict(bool)

    def update(
        self,
        fingertips: Sequence[Sequence[NormalizedLandmark]],
        frame_size: tuple[int, int],
        timestamp: float,
    ) -> list[HitEvent]:

        width, height = frame_size

        events = []

        for hand_id, hand in enumerate(fingertips):

            for finger_id, landmark in enumerate(hand):

                x = landmark[0] * width
                y = landmark[1] * height

                position = (x, y)

                finger_key = (hand_id, finger_id)

                for pad in self._pads:

                    state_key = (finger_key, pad.pad_id)

                    inside = pad.contains(position)

                    previously_inside = self._inside[state_key]

                    # Finger just entered the pad
                    if (
                        inside
                        and not previously_inside
                        and timestamp - self._last_hit_time[state_key]
                        >= self._config.cooldown_seconds
                    ):

                        events.append(
                            HitEvent(
                                hand_id=hand_id,
                                finger_id=finger_id,
                                pad_id=pad.pad_id,
                                timestamp=timestamp,
                                fingertip_position=position,
                            )
                        )

                        self._last_hit_time[state_key] = timestamp

                    # Update state
                    self._inside[state_key] = inside

        return events

    def reset(self):
        self._inside.clear()
        self._last_hit_time.clear()