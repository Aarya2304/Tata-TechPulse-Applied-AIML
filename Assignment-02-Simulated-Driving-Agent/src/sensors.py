"""Ray-based distance sensors for the driving agent.

Seven rays fan out from the vehicle's heading (far-left to far-right) and
measure the distance to the nearest track boundary along each direction.
Distances are normalised to [0, 1] where 1.0 = clear road up to
``SENSOR_MAX_RANGE``.

Two APIs are provided:

* :meth:`SensorSuite.cast_single` / :meth:`SensorSuite.observe` -- simple,
  single-vehicle marching used for clarity and testing.
* :meth:`SensorSuite.cast_batched` / :meth:`SensorSuite.observe_batched` --
  fully vectorised ray casting used by the training loop; it applies the
  same geometry with a march-free slice test, so results agree with the
  single-ray version.

The sensor module is independent of the vehicle/GA and fully unit-testable.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from src import config


@dataclass
class SensorSuite:
    """Configurable ray-caster against a :class:`src.track.Track`."""

    track: object                       # src.track.Track
    ray_offsets: tuple = config.SENSOR_RAY_OFFSETS
    max_range: float = config.SENSOR_MAX_RANGE
    step: float = 1.0                   # marching step in meters (single mode)

    # ------------------------------------------------------------------
    # Single-ray marching (clarity / reference implementation)
    # ------------------------------------------------------------------
    def cast_single(self, x: float, y: float, angle: float) -> float:
        """Distance from (x, y) along ``angle`` to the first off-track point
        (capped at ``max_range``).  Returns meters.

        Uses the same slice sampling and distance-grid geometry as the
        batched path, so single and batched observations agree exactly.
        """
        n_steps = int(np.ceil(self.max_range / self.step))
        d = np.arange(1, n_steps + 1, dtype=float) * self.step
        px = x + math.cos(angle) * d
        py = y + math.sin(angle) * d
        inside = self.track.is_on_track_batched(px, py)
        miss = ~inside
        if miss.any():
            return float(d[int(np.argmax(miss))])
        return float(self.max_range)

    def observe(self, vehicle) -> np.ndarray:
        """Return the 8-dim observation vector for ``vehicle``.

        Layout: 7 normalised ray distances followed by the normalised speed.
        """
        obs = np.empty(len(self.ray_offsets) + 1, dtype=float)
        for i, offset in enumerate(self.ray_offsets):
            angle = vehicle.heading + offset
            distance = self.cast_single(vehicle.x, vehicle.y, angle)
            obs[i] = min(distance / self.max_range, 1.0)
        obs[-1] = vehicle.normalized_speed()
        return obs

    # ------------------------------------------------------------------
    # Batched ray casting (training fast path)
    # ------------------------------------------------------------------
    def cast_batched(self, xs: np.ndarray, ys: np.ndarray,
                     angles: np.ndarray) -> np.ndarray:
        """Vectorised ray distances (meters) for arrays of rays.

        For each ray, the first sample point along the ray that lies outside
        the track is located with a binary search over a slice grid (the road
        is star-shaped with respect to every on-road origin: once a slice
        point leaves the road, the boundary was crossed at or before it).
        """
        xs = np.asarray(xs, dtype=float).ravel()
        ys = np.asarray(ys, dtype=float).ravel()
        angles = np.asarray(angles, dtype=float).ravel()
        n_rays = xs.size
        n_steps = int(np.ceil(self.max_range / self.step))
        step_multipliers = (np.arange(1, n_steps + 1, dtype=float)
                            * self.step)[:, None]     # (n_steps, 1)

        cos_a = np.cos(angles)[None, :]
        sin_a = np.sin(angles)[None, :]
        px = xs[None, :] + step_multipliers * cos_a    # (n_steps, n_rays)
        py = ys[None, :] + step_multipliers * sin_a

        inside = self.track.is_on_track_batched(px, py)
        hit = ~inside                                  # first True per column

        distances = np.full(n_rays, self.max_range, dtype=float)
        any_hit = hit.any(axis=0)
        first_hit = np.argmax(hit, axis=0)             # 0 if no hit
        hit_dist = (first_hit + 1) * self.step
        distances[any_hit] = hit_dist[any_hit]
        return distances

    def observe_batched(self, xs: np.ndarray, ys: np.ndarray,
                        headings: np.ndarray,
                        speeds: np.ndarray) -> np.ndarray:
        """Return an (n_agents, 8) observation matrix.

        Row layout: 7 normalised ray distances + normalised speed.
        """
        xs = np.asarray(xs, dtype=float)[:, None]      # (n, 1) broadcast
        ys = np.asarray(ys, dtype=float)[:, None]
        headings = np.asarray(headings, dtype=float)[:, None]
        offsets = np.asarray(self.ray_offsets, dtype=float)[None, :]
        angles = headings + offsets                    # (n, n_rays)

        distances = self.cast_batched(np.broadcast_to(xs, angles.shape),
                                      np.broadcast_to(ys, angles.shape),
                                      angles).reshape(angles.shape)
        obs = np.empty((distances.shape[0], len(self.ray_offsets) + 1),
                       dtype=float)
        obs[:, :-1] = np.minimum(distances / self.max_range, 1.0)
        obs[:, -1] = np.asarray(speeds, dtype=float) / config.VEHICLE_MAX_SPEED
        return obs
