"""MediaPipe Tasks hand landmark detection and visualization."""

from __future__ import annotations

from pathlib import Path
from typing import Final, TypeAlias

import cv2
import mediapipe as mp
import numpy as np


Landmark: TypeAlias = tuple[float, float, float]
Fingertips: TypeAlias = list[list[Landmark]]


class HandTracker:
    """Detect, visualize, and expose fingertip landmarks for up to two hands."""

    FINGERTIP_IDS: Final[tuple[int, ...]] = (4, 8, 12, 16, 20)
    HAND_CONNECTIONS: Final[tuple[tuple[int, int], ...]] = (
        (0, 1), (1, 2), (2, 3), (3, 4),
        (0, 5), (5, 6), (6, 7), (7, 8),
        (5, 9), (9, 10), (10, 11), (11, 12),
        (9, 13), (13, 14), (14, 15), (15, 16),
        (13, 17), (17, 18), (18, 19), (19, 20), (0, 17),
    )
    _FINGERTIP_COLORS: Final[tuple[tuple[int, int, int], ...]] = (
        (0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 255, 255), (255, 0, 255)
    )
    DEFAULT_MODEL_PATH: Final[Path] = (
        Path(__file__).resolve().parents[2] / "assets" / "hand_landmarker.task"
    )

    def __init__(
        self,
        model_path: str | Path | None = None,
        num_hands: int = 2,
        detection_confidence: float = 0.6,
        presence_confidence: float = 0.6,
        tracking_confidence: float = 0.6,
    ) -> None:
        """Create a synchronous video-mode MediaPipe Tasks hand landmarker."""
        resolved_model_path = Path(model_path) if model_path is not None else self.DEFAULT_MODEL_PATH
        if not resolved_model_path.is_file():
            raise FileNotFoundError(
                f"Hand Landmarker model not found: {resolved_model_path}. "
                "Download hand_landmarker.task into the assets directory."
            )

        options = mp.tasks.vision.HandLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=str(resolved_model_path)),
            running_mode=mp.tasks.vision.RunningMode.VIDEO,
            num_hands=num_hands,
            min_hand_detection_confidence=detection_confidence,
            min_hand_presence_confidence=presence_confidence,
            min_tracking_confidence=tracking_confidence,
        )
        self._landmarker = mp.tasks.vision.HandLandmarker.create_from_options(options)

    def process_frame(self, frame: np.ndarray, timestamp_ms: int) -> tuple[np.ndarray, Fingertips]:
        """Return an annotated BGR frame and fingertip `(x, y, z)` landmarks.

        Coordinates are normalized to the input image dimensions. The outer list
        is one entry per detected hand; fingertip order is thumb through pinky.
        `timestamp_ms` must be strictly increasing between calls.
        """
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb_frame))
        result = self._landmarker.detect_for_video(image, timestamp_ms)

        annotated_frame = frame.copy()
        frame_height, frame_width = frame.shape[:2]
        fingertips: Fingertips = []
        for hand_landmarks in result.hand_landmarks:
            self._draw_hand(annotated_frame, hand_landmarks, frame_width, frame_height)
            fingertips.append(
                [
                    (hand_landmarks[landmark_id].x, hand_landmarks[landmark_id].y, hand_landmarks[landmark_id].z)
                    for landmark_id in self.FINGERTIP_IDS
                ]
            )

        return annotated_frame, fingertips

    def _draw_hand(self, frame: np.ndarray, landmarks: object, width: int, height: int) -> None:
        """Draw all 21 landmarks, their skeletal connections, and colored tips."""
        points = [(int(landmark.x * width), int(landmark.y * height)) for landmark in landmarks]
        for start, end in self.HAND_CONNECTIONS:
            cv2.line(frame, points[start], points[end], (255, 255, 255), 2, cv2.LINE_AA)
        for point in points:
            cv2.circle(frame, point, 4, (0, 180, 255), cv2.FILLED, cv2.LINE_AA)
        for landmark_id, color in zip(self.FINGERTIP_IDS, self._FINGERTIP_COLORS):
            cv2.circle(frame, points[landmark_id], 10, color, cv2.FILLED, cv2.LINE_AA)
            cv2.circle(frame, points[landmark_id], 12, (255, 255, 255), 2, cv2.LINE_AA)

    def close(self) -> None:
        """Release MediaPipe Tasks resources."""
        self._landmarker.close()

    def __enter__(self) -> "HandTracker":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
