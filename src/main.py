"""Run the FingerDrum hand-tracking webcam preview."""

from __future__ import annotations

import time

import cv2

from vision.hand_tracker import HandTracker
from ui.drum_layout import DrumLayout


WINDOW_TITLE = "FingerDrum"


class FingerDrumApp:
    """Coordinate webcam capture, hand visualization, and clean shutdown."""

    def __init__(self, camera_index: int = 0) -> None:
        self._camera_index = camera_index
        self._drum_layout = DrumLayout()

    def run(self) -> None:
        """Start the webcam preview and exit when ESC is pressed."""
        capture = cv2.VideoCapture(self._camera_index)
        if not capture.isOpened():
            capture.release()
            raise RuntimeError(f"Unable to open webcam at index {self._camera_index}.")

        previous_time = time.perf_counter()
        try:
            with HandTracker(num_hands=2) as tracker:
                while True:
                    success, frame = capture.read()
                    if not success:
                        raise RuntimeError("Unable to read a frame from the webcam.")

                    frame = cv2.flip(frame, 1)
                    timestamp_ms = int(time.perf_counter() * 1000)
                    annotated_frame, _fingertips = tracker.process_frame(frame, timestamp_ms)
                    previous_time = self._draw_fps(annotated_frame, previous_time)
                    interface = self._drum_layout.render(annotated_frame)

                    cv2.imshow(WINDOW_TITLE, interface)
                    if cv2.waitKey(1) & 0xFF == 27:  # ESC
                        break
        finally:
            capture.release()
            cv2.destroyAllWindows()

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
