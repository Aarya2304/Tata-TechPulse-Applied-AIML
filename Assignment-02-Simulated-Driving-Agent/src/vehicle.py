"""Simple 2D vehicle with an arcade-style kinematic model.

State: position (x, y), heading angle, scalar speed plus the latest steering
and throttle commands.  Movement uses a kinematic bicycle approximation with
speed-dependent steering effectiveness.  Not physically accurate -- the goal
is a stable environment where steering decisions genuinely matter.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from src import config


@dataclass
class Vehicle:
    """2D kinematic vehicle on the track plane."""

    x: float
    y: float
    heading: float          # radians, 0 = +x axis
    speed: float = 0.0      # m/s, always >= 0

    steering: float = 0.0   # current steering command (rad)
    throttle: float = 0.0   # current throttle command in [-1, 1]
    on_track: bool = True   # last boundary check result

    def __post_init__(self) -> None:
        self.clamp_state()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def clamp_state(self) -> None:
        self.steering = max(-config.VEHICLE_MAX_STEER,
                            min(config.VEHICLE_MAX_STEER, self.steering))
        self.throttle = max(-1.0, min(1.0, self.throttle))
        self.speed = max(0.0, min(config.VEHICLE_MAX_SPEED, self.speed))

    def reset(self, x: float, y: float, heading: float) -> None:
        self.x = float(x)
        self.y = float(y)
        self.heading = float(heading)
        self.speed = 0.0
        self.steering = 0.0
        self.throttle = 0.0
        self.on_track = True

    # ------------------------------------------------------------------
    # Dynamics
    # ------------------------------------------------------------------
    def step(self, dt: float) -> None:
        """Advance the vehicle physics by ``dt`` seconds using the current
        steering and throttle commands (arcade bicycle model)."""
        self.clamp_state()

        # 1) longitudinal dynamics
        if self.throttle >= 0.0:
            accel = self.throttle * config.VEHICLE_ACCEL
        else:
            accel = self.throttle * config.VEHICLE_BRAKE  # braking / reverse-free
        accel -= config.VEHICLE_DRAG * self.speed
        self.speed += accel * dt
        self.clamp_state()

        # 2) steering -> heading rate (bicycle model, speed-normalised).
        #    At zero speed the vehicle cannot turn.
        effective_wheelbase = config.VEHICLE_WHEELBASE * (
            1.0 + min(self.speed / config.VEHICLE_MAX_SPEED, 1.0)
        )
        heading_rate = (self.speed / effective_wheelbase) * math.tan(self.steering)
        self.heading += heading_rate * dt

        # 3) position update
        self.x += self.speed * math.cos(self.heading) * dt
        self.y += self.speed * math.sin(self.heading) * dt

    def distance_to(self, other_x: float, other_y: float) -> float:
        return math.hypot(self.x - other_x, self.y - other_y)

    def normalized_speed(self) -> float:
        return self.speed / config.VEHICLE_MAX_SPEED
