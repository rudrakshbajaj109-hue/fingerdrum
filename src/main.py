"""Run the FingerDrum hand-tracking webcam preview — Neon Cyberpunk Edition."""

from __future__ import annotations

import time

import cv2

from audio.audio_engine import AudioEngine
from drum.hit_detector import CircularPad, HitDetector, HitDetectorConfig
from vision.hand_tracker import HandTracker
from ui.drum_layout import DrumLayout
from ui.neon_drum_renderer import NeonDrumRenderer, _NEON_COLORS
from ui.neon_finger_renderer import NeonFingerRenderer
from ui.hit_animation_manager import HitAnimationManager
from effects.particle_system import ParticleSystem
from effects.post_processor import PostProcessor


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
        self._drum_renderer = NeonDrumRenderer(self._drum_layout, self._hit_animations)
        self._finger_renderer = NeonFingerRenderer()
        self._particles = ParticleSystem()
        self._post_processor = PostProcessor()

    def run(self) -> None:
      """Start FingerDrum in fullscreen mode."""

      capture = cv2.VideoCapture(self._camera_index)

      if not capture.isOpened():
          capture.release()
          raise RuntimeError(
              f"Unable to open webcam at index {self._camera_index}."
          )

    # Capture in HD, then downscale only for the real-time processing pass.
      capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
      capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
      capture.set(cv2.CAP_PROP_FPS, 60)

      screen_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
      screen_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Fullscreen window
      cv2.namedWindow(WINDOW_TITLE, cv2.WINDOW_NORMAL)

      cv2.setWindowProperty(
          WINDOW_TITLE,
          cv2.WND_PROP_FULLSCREEN,
          cv2.WINDOW_FULLSCREEN,
      )

      previous_time = time.perf_counter()
      previous_animation_time = previous_time

      try:
          with HandTracker(
              num_hands=2,
              draw_landmarks=False,
          ) as tracker:

              while True:

                  success, frame = capture.read()

                  if not success:
                      raise RuntimeError(
                          "Unable to read a frame from the webcam."
                      )

                  # Some camera drivers ignore the requested capture size.
                  # Keep the render and inference workload bounded regardless.
                  if frame.shape[1] > 960 or frame.shape[0] > 540:
                      scale = min(960 / frame.shape[1], 540 / frame.shape[0])
                      processing_size = (
                          max(1, int(frame.shape[1] * scale)),
                          max(1, int(frame.shape[0] * scale)),
                      )
                      frame = cv2.resize(
                          frame,
                          processing_size,
                          interpolation=cv2.INTER_AREA,
                      )
                  frame = cv2.flip(frame, 1)

                  timestamp_ms = int(
                      time.perf_counter() * 1000
                  )

                  annotated_frame, fingertips = tracker.process_frame(
                      frame,
                      timestamp_ms,
                  )

                  frame_height, frame_width = annotated_frame.shape[:2]

                  self._sync_detector_to_frame(
                      (frame_width, frame_height)
                  )

                  current_animation_time = time.perf_counter()
                  dt = current_animation_time - previous_animation_time

                  self._hit_animations.update(dt)
                  self._particles.update(dt)

                  previous_animation_time = (
                      current_animation_time
                  )

                  hits = self._hit_detector.update(
                      fingertips,
                      (frame_width, frame_height),
                      time.perf_counter(),
                  )

                  regions = self._drum_layout.pad_regions(
                      (frame_width, frame_height)
                  )

                  for hit in hits:

                      print(
                          f"HIT -> {hit.pad_id} "
                          f"| Hand {hit.hand_id} "
                          f"| Finger {hit.finger_id}"
                      )

                      self._audio_engine.play_hit(hit)
                      self._hit_animations.register_hit(hit)

                      # Particle burst at hit pad center
                      pad_region = regions.get(hit.pad_id)
                      if pad_region:
                          pad_center, pad_radius = pad_region
                          neon_color = _NEON_COLORS.get(
                              hit.pad_id, (200, 200, 200)
                          )
                          self._particles.emit_burst(
                              pad_center,
                              neon_color,
                              pad_radius,
                              count=18,
                          )

                  # Build the visual interface
                  interface = frame.copy()

                  # Apply post-processing (vignette + color grade)
                  interface = self._post_processor.process(interface)

                  fingertip_positions = (
                      self._fingertip_positions(
                          fingertips,
                          frame_width,
                          frame_height,
                      )
                  )

                  # Draw neon pads
                  self._drum_renderer.draw(
                      interface,
                      fingertip_positions,
                  )

                  # Draw particles and shockwaves
                  self._particles.draw(interface)

                  # Emit sparkle trails at finger positions
                  for pos in fingertip_positions:
                      self._particles.emit_sparkle_trail(
                          pos,
                          (0, 255, 220),
                          count=1,
                      )

                  # Draw neon finger cursors
                  self._finger_renderer.draw(
                      interface,
                      fingertip_positions,
                  )

                  fullscreen_frame = cv2.resize(
                      interface,
                      (screen_width, screen_height),
                      interpolation=cv2.INTER_LINEAR,
                  )

                  # Draw text after scaling so the top bar stays sharp.
                  previous_time = self._draw_hud(
                      fullscreen_frame,
                      previous_time,
                      len(fingertips),
                  )

                  cv2.imshow(
                      WINDOW_TITLE,
                      fullscreen_frame,
                  )

                  key = cv2.waitKey(1) & 0xFF

                  if key == 27:  # ESC
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
    def _draw_hud(frame: object, previous_time: float, hand_count: int) -> float:
        """Draw a sleek cyberpunk-styled live status bar."""
        current_time = time.perf_counter()
        elapsed = current_time - previous_time
        fps = 1.0 / elapsed if elapsed > 0 else 0.0
        height, width = frame.shape[:2]
        margin = max(18, int(width * 0.018))
        panel_height = max(72, int(height * 0.12))

        # Translucent dark header panel
        roi = frame[margin:margin + panel_height, margin:width - margin]
        if roi.size:
            overlay = roi.copy()
            cv2.rectangle(
                overlay,
                (0, 0),
                (roi.shape[1], roi.shape[0]),
                (8, 10, 18),
                cv2.FILLED,
            )
            cv2.addWeighted(overlay, 0.82, roi, 0.18, 0, dst=roi)

            # Neon accent line at bottom of header
            line_y = roi.shape[0] - 1
            cv2.line(
                roi,
                (0, line_y),
                (roi.shape[1], line_y),
                (0, 255, 220),
                1,
                cv2.LINE_AA,
            )

        scale = max(0.48, min(0.8, width / 1600))
        baseline = margin + int(panel_height * 0.48)
        title_x = margin + int(panel_height * 0.38)

        # Neon green dot indicator
        cv2.circle(
            frame,
            (margin + 20, baseline - 7),
            5,
            (0, 255, 180),
            cv2.FILLED,
            cv2.LINE_AA,
        )
        # Dot glow
        cv2.circle(
            frame,
            (margin + 20, baseline - 7),
            9,
            (0, 255, 180),
            1,
            cv2.LINE_AA,
        )

        # Title text with neon tint
        cv2.putText(
            frame,
            "FingerDrum",
            (title_x, baseline),
            cv2.FONT_HERSHEY_DUPLEX,
            scale,
            (0, 255, 220),
            1,
            cv2.LINE_AA,
        )

        right = width - margin - int(panel_height * 0.35)
        creator = "Made by Rudraksh Bajaj"
        creator_scale = scale * 0.62
        creator_size, _ = cv2.getTextSize(
            creator, cv2.FONT_HERSHEY_SIMPLEX, creator_scale, 1
        )
        cv2.putText(
            frame,
            creator,
            (right - creator_size[0], baseline),
            cv2.FONT_HERSHEY_SIMPLEX,
            creator_scale,
            (220, 160, 255),
            1,
            cv2.LINE_AA,
        )

        status = f"{hand_count} HAND{'S' if hand_count != 1 else ''}  |  {fps:04.1f} FPS  |  ESC  EXIT"
        text_size, _ = cv2.getTextSize(status, cv2.FONT_HERSHEY_SIMPLEX, scale * 0.72, 1)
        cv2.putText(
            frame,
            status,
            (right - text_size[0], margin + int(panel_height * 0.84)),
            cv2.FONT_HERSHEY_SIMPLEX,
            scale * 0.72,
            (180, 200, 210),
            1,
            cv2.LINE_AA,
        )

        return current_time


if __name__ == "__main__":
    FingerDrumApp().run()
