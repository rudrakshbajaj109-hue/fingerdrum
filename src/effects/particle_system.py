"""Smooth particle explosion system for drum hit feedback.

All particles are drawn directly onto the frame (no per-particle ROI copy)
for maximum smoothness.  Shockwaves use a single shared overlay per frame.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass

import cv2
import numpy as np


Color = tuple[int, int, int]


@dataclass
class _Particle:
    """A single spark particle with physics."""

    x: float
    y: float
    vx: float
    vy: float
    color: Color
    life: float          # seconds remaining
    max_life: float
    radius: float = 2.5
    gravity: float = 80.0
    drag: float = 0.97


@dataclass
class _Shockwave:
    """An expanding ring with fade-out."""

    cx: int
    cy: int
    color: Color
    age: float = 0.0
    max_age: float = 0.32
    max_radius: float = 100.0


class ParticleSystem:
    """Manage and render particle explosions and shockwaves."""

    _MAX_PARTICLES = 120   # hard cap to prevent frame drops

    def __init__(self) -> None:
        self._particles: list[_Particle] = []
        self._shockwaves: list[_Shockwave] = []

    def emit_burst(
        self,
        center: tuple[int, int],
        color: Color,
        pad_radius: int,
        *,
        count: int = 18,
        speed_range: tuple[float, float] = (140.0, 360.0),
        life_range: tuple[float, float] = (0.20, 0.42),
    ) -> None:
        """Spawn a radial burst of particles from *center*."""
        cx, cy = center

        # Trim if we'd exceed the cap
        budget = max(0, self._MAX_PARTICLES - len(self._particles))
        actual = min(count, budget)

        for _ in range(actual):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(*speed_range)
            life = random.uniform(*life_range)

            # Slight tint variation
            tinted = (
                max(0, min(255, color[0] + random.randint(-20, 20))),
                max(0, min(255, color[1] + random.randint(-20, 20))),
                max(0, min(255, color[2] + random.randint(-20, 20))),
            )

            self._particles.append(
                _Particle(
                    x=float(cx) + random.uniform(-2, 2),
                    y=float(cy) + random.uniform(-2, 2),
                    vx=math.cos(angle) * speed,
                    vy=math.sin(angle) * speed,
                    color=tinted,
                    life=life,
                    max_life=life,
                    radius=random.uniform(1.5, 3.0),
                )
            )

        # Shockwave ring
        self._shockwaves.append(
            _Shockwave(
                cx=cx,
                cy=cy,
                color=color,
                max_radius=float(pad_radius) * 2.2,
            )
        )

    def emit_sparkle_trail(
        self,
        position: tuple[float, float],
        color: Color,
        *,
        count: int = 1,
    ) -> None:
        """Spawn tiny sparkle particles at a fingertip for a trailing effect."""
        if len(self._particles) >= self._MAX_PARTICLES:
            return

        x, y = position
        for _ in range(count):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(10.0, 35.0)
            life = random.uniform(0.06, 0.14)

            self._particles.append(
                _Particle(
                    x=x + random.uniform(-2, 2),
                    y=y + random.uniform(-2, 2),
                    vx=math.cos(angle) * speed,
                    vy=math.sin(angle) * speed,
                    color=color,
                    life=life,
                    max_life=life,
                    radius=random.uniform(1.0, 1.8),
                    gravity=20.0,
                    drag=0.94,
                )
            )

    def update(self, dt: float) -> None:
        """Advance physics for all particles and shockwaves."""
        dt = max(dt, 0.0)

        alive: list[_Particle] = []
        for p in self._particles:
            p.life -= dt
            if p.life <= 0:
                continue
            p.vx *= p.drag
            p.vy *= p.drag
            p.vy += p.gravity * dt
            p.x += p.vx * dt
            p.y += p.vy * dt
            alive.append(p)
        self._particles = alive

        active_waves: list[_Shockwave] = []
        for w in self._shockwaves:
            w.age += dt
            if w.age < w.max_age:
                active_waves.append(w)
        self._shockwaves = active_waves

    def draw(self, frame: np.ndarray) -> None:
        """Render all active particles and shockwaves onto *frame*."""
        height, width = frame.shape[:2]

        # -- Shockwaves: single shared overlay for all active waves --
        if self._shockwaves:
            overlay = frame.copy()
            max_alpha = 0.0
            for w in self._shockwaves:
                progress = w.age / w.max_age
                radius = int(w.max_radius * progress)
                alpha = 0.40 * (1.0 - progress) ** 2

                if radius < 2 or alpha < 0.01:
                    continue
                max_alpha = max(max_alpha, alpha)
                cv2.circle(overlay, (w.cx, w.cy), radius, w.color, 2, cv2.LINE_AA)

            if max_alpha > 0.01:
                cv2.addWeighted(overlay, min(max_alpha, 0.45), frame, 1.0 - min(max_alpha, 0.45), 0, dst=frame)

        # -- Particles: draw directly, no per-particle overlay --
        for p in self._particles:
            ix, iy = int(p.x), int(p.y)
            if ix < 1 or iy < 1 or ix >= width - 1 or iy >= height - 1:
                continue

            fade = max(p.life / p.max_life, 0.0)
            r = max(int(p.radius * fade), 1)

            # Fade colour toward transparency by blending with current pixel
            bg = frame[iy, ix].astype(np.float32)
            fg = np.array(p.color, dtype=np.float32)
            blended = (fg * fade + bg * (1.0 - fade)).astype(np.uint8)

            cv2.circle(frame, (ix, iy), r, blended.tolist(), -1, cv2.LINE_AA)

    @property
    def active_count(self) -> int:
        """Total number of active effects for diagnostics."""
        return len(self._particles) + len(self._shockwaves)
