"""Velocity-based, sound-free fingertip hit detection for circular drum pads."""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Deque, Iterable, Mapping, Sequence, TypeAlias


PixelPoint: TypeAlias = tuple[float, float]
NormalizedLandmark: TypeAlias = tuple[float, float, float]
FingerKey: TypeAlias = tuple[int, int]


@dataclass(frozen=True)
class CircularPad:
    """A circular pad hit region expressed in display pixel coordinates."""

    pad_id: str
    center: PixelPoint
    radius: float

    def __post_init__(self) -> None:
        if self.radius <= 0:
            raise ValueError("Pad radius must be greater than zero.")

    def contains(self, point: PixelPoint) -> bool:
        """Return whether ``point`` falls inside this pad's circular region."""
        x_offset = point[0] - self.center[0]
        y_offset = point[1] - self.center[1]
        return x_offset * x_offset + y_offset * y_offset <= self.radius * self.radius


@dataclass(frozen=True)
class HitDetectorConfig:
    """Tunable detection parameters, all expressed in seconds or pixels."""

    downward_velocity_threshold = 450.0
    cooldown_seconds: float = 0.08
    history_size: int = 5

    def __post_init__(self) -> None:
        if self.downward_velocity_threshold < 0:
            raise ValueError("The downward velocity threshold cannot be negative.")
        if self.cooldown_seconds < 0:
            raise ValueError("The cooldown cannot be negative.")
        if self.history_size < 2:
            raise ValueError("History size must be at least two samples.")


@dataclass(frozen=True)
class HitEvent:
    """An immutable record of one fingertip entering a pad at hit speed."""

    finger_id: int
    hand_id: int
    pad_id: str
    timestamp: float
    fingertip_position: PixelPoint
    downward_velocity: float


@dataclass(frozen=True)
class _FingertipSample:
    """One timestamped pixel-position sample used to calculate velocity."""

    timestamp: float
    position: PixelPoint


class HitDetector:
    """Detect independent, downward fingertip strikes against circular pads.

    ``update`` consumes the normalized landmark lists returned by
    ``HandTracker.process_frame``. The outer sequence is ordered by hand and
    each inner sequence is ordered by fingertip, so a finger identity is the
    stable ``(hand_id, finger_id)`` pair for that frame.

    No UI, animation, or sound responsibilities are included here.
    """

    def __init__(
        self,
        pads: Iterable[CircularPad],
        config: HitDetectorConfig | None = None,
    ) -> None:
        self._pads = tuple(pads)
        self._validate_pad_ids(self._pads)
        self._config = config or HitDetectorConfig()
        self._histories: dict[FingerKey, Deque[_FingertipSample]] = defaultdict(
            lambda: deque(maxlen=self._config.history_size)
        )
        self._last_hit_times: dict[tuple[FingerKey, str], float] = {}
        self._last_velocities: dict[FingerKey, PixelPoint] = {}

    @property
    def config(self) -> HitDetectorConfig:
        """Return the immutable configuration used by this detector."""
        return self._config

    @property
    def last_velocities(self) -> Mapping[FingerKey, PixelPoint]:
        """Return x/y fingertip velocities in pixels per second for the last frame."""
        return dict(self._last_velocities)

    def update(
        self,
        fingertips: Sequence[Sequence[NormalizedLandmark]],
        frame_size: tuple[int, int],
        timestamp: float,
    ) -> list[HitEvent]:
        """Process one tracker frame and return every hit detected in it.

        ``frame_size`` is ``(width, height)``. ``timestamp`` must be in seconds
        and strictly increase between frames. The first visible frame for a
        fingertip establishes history and therefore cannot produce a hit.
        """
        frame_width, frame_height = frame_size
        if frame_width <= 0 or frame_height <= 0:
            raise ValueError("Frame width and height must be positive.")

        self._last_velocities = {}
        events: list[HitEvent] = []
        for hand_id, hand_fingertips in enumerate(fingertips):
            for finger_id, landmark in enumerate(hand_fingertips):
                position = self._to_pixel_position(landmark, frame_width, frame_height)
                key = (hand_id, finger_id)
                history = self._histories[key]
                previous = history[-1] if history else None
                velocity = self._velocity_between(previous, position, timestamp)
                self._last_velocities[key] = velocity

                if previous is not None:
                    events.extend(
                        self._detect_hits(key, position, previous.position, velocity[1], timestamp)
                    )
                history.append(_FingertipSample(timestamp, position))

        return events

    def reset(self) -> None:
        """Clear motion history, velocities, and cooldown state."""
        self._histories.clear()
        self._last_hit_times.clear()
        self._last_velocities.clear()

    def _detect_hits(
        self,
        finger_key: FingerKey,
        position: PixelPoint,
        previous_position: PixelPoint,
        downward_velocity: float,
        timestamp: float,
    ) -> list[HitEvent]:
        if downward_velocity <= self._config.downward_velocity_threshold:
            return []

        hand_id, finger_id = finger_key
        events: list[HitEvent] = []
        for pad in self._pads:            
            entered_pad = (pad.contains(position)and not pad.contains(previous_position))
            cooldown_key = (finger_key, pad.pad_id)
            cooldown_expired = (timestamp- self._last_hit_times.get(cooldown_key, float("-inf"))>= self._config.cooldown_seconds)
            if entered_pad and cooldown_expired:
                event = HitEvent(
                    finger_id=finger_id,
                    hand_id=hand_id,
                    pad_id=pad.pad_id,
                    timestamp=timestamp,
                    fingertip_position=position,
                    downward_velocity=downward_velocity,
                )
                events.append(event)
                self._last_hit_times[cooldown_key] = timestamp
        return events

    @staticmethod
    def _to_pixel_position(landmark: NormalizedLandmark, width: int, height: int) -> PixelPoint:
        """Convert an ``(x, y, z)`` normalized tracker landmark to pixels."""
        return landmark[0] * width, landmark[1] * height

    @staticmethod
    def _velocity_between(
        previous: _FingertipSample | None,
        position: PixelPoint,
        timestamp: float,
    ) -> PixelPoint:
        """Calculate x/y velocity in pixels per second from the previous sample."""
        if previous is None:
            return 0.0, 0.0
        elapsed = timestamp - previous.timestamp
        if elapsed <= 0:
            raise ValueError("Timestamps must strictly increase for each fingertip.")
        return (
            (position[0] - previous.position[0]) / elapsed,
            (position[1] - previous.position[1]) / elapsed,
        )

    @staticmethod
    def _validate_pad_ids(pads: Sequence[CircularPad]) -> None:
        """Reject ambiguous pad identifiers before detection begins."""
        pad_ids = [pad.pad_id for pad in pads]
        if len(pad_ids) != len(set(pad_ids)):
            raise ValueError("Each drum pad must have a unique pad_id.")
