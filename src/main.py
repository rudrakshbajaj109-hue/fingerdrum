"""Run the FingerDrum hand-tracking webcam preview."""

from __future__ import annotations

import time

import cv2

from audio.audio_engine import AudioEngine
from drum.hit_detector import CircularPad, HitDetector, HitDetectorConfig
from vision.hand_tracker import HandTracker
from ui.drum_layout import DrumLayout
from ui.drum_renderer import DrumRenderer
from ui.finger_renderer import FingerRenderer
from ui.hit_animation_manager import HitAnimationManager


WINDOW_TITLE = "FingerDrum"


class FingerDrumApp:
    """Coordinate webcam capture, hand visualization, and clean shutdown."""

    def __init__(self, camera_index: int = 0) -> None:
        self._camera_index = camera_index
        self._drum_layout = DrumLayout()
        self._detector_frame_size = (640, 480)
        self._hit_detector_config = HitDetectorConfig()
        self._hit_detector = HitDetector(
            self._build_circular_pads(self._detector_frame_size),
            self._hit_detector_config,
        )
        self._audio_engine = AudioEngine()
        self._hit_animations = HitAnimationManager()
        self._drum_renderer = DrumRenderer(self._drum_layout, self._hit_animations)
        self._finger_renderer = FingerRenderer()

    def run(self) -> None:
        """Start the webcam preview and exit when ESC is pressed."""
        capture = cv2.VideoCapture(self._camera_index)
        if not capture.isOpened():
            capture.release()
            raise RuntimeError(f"Unable to open webcam at index {self._camera_index}.")

        previous_time = time.perf_counter()
        previous_animation_time = previous_time
        try:
            with HandTracker(num_hands=1, draw_landmarks=False) as tracker:
                while True:
                    success, frame = capture.read()
                    if not success:
                        raise RuntimeError("Unable to read a frame from the webcam.")

                    frame = cv2.flip(frame, 1)
                    timestamp_ms = int(time.perf_counter() * 1000)
                    annotated_frame, fingertips = tracker.process_frame(frame,timestamp_ms,)
                    frame_height, frame_width = annotated_frame.shape[:2]
                    self._sync_detector_to_frame((frame_width, frame_height))

                    current_animation_time = time.perf_counter()
                    self._hit_animations.update(
                        current_animation_time - previous_animation_time
                    )
                    previous_animation_time = current_animation_time
                    hits = self._hit_detector.update(fingertips,(frame_width, frame_height),time.perf_counter(),)

                    for hit in hits:
                        print(f"HIT: {hit.pad_id} velocity={hit.downward_velocity:.1f}")
                        self._audio_engine.play_hit(hit)
                        self._hit_animations.register_hit(hit)

                    interface = frame.copy()
                    fingertip_positions = self._fingertip_positions(
                        fingertips,
                        frame_width,
                        frame_height,
                    )

                    self._drum_renderer.draw(interface, fingertip_positions)
                    self._finger_renderer.draw(interface,fingertip_positions,)
                    previous_time = self._draw_fps(interface, previous_time)

                    cv2.imshow(WINDOW_TITLE, interface)
                    if cv2.waitKey(1) & 0xFF == 27:  # ESC
                        break
        finally:
            capture.release()
            self._audio_engine.stop_all()
            cv2.destroyAllWindows()

    def _sync_detector_to_frame(self, frame_size: tuple[int, int]) -> None:
        """Keep detector regions aligned with the active camera resolution."""
        if frame_size == self._detector_frame_size:
            return

        self._detector_frame_size = frame_size
        self._hit_detector = HitDetector(
            self._build_circular_pads(frame_size),
            self._hit_detector_config,
        )

    def _build_circular_pads(
        self,
        frame_size: tuple[int, int],
    ) -> tuple[CircularPad, ...]:
        """Create detector regions from the layout's eight labelled pad models."""
        return tuple(
            CircularPad(pad_id, center, radius)
            for pad_id, (center, radius) in self._drum_layout.pad_regions(frame_size).items()
        )

    @staticmethod
    def _fingertip_positions(
      fingertips,
      frame_width,
      frame_height,):
      positions = []

      for hand in fingertips:
        if len(hand) > 0:
          x, y, _ = hand[0]
          positions.append((x * frame_width, y * frame_height))

      return tuple(positions)
    @staticmethod
    def _draw_fps(frame: object, previous_time: float) -> float:
        """Render the current capture FPS and return the timestamp for the next frame."""
        current_time = time.perf_counter()
        elapsed = current_time - previous_time
        fps = 1.0 / elapsed if elapsed > 0 else 0.0

        cv2.putText(
            frame,
            f"FPS: {fps:.1f}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

        return current_time


if __name__ == "__main__":
    FingerDrumApp().run()