"""Tests for the ray-based sensor suite."""

from __future__ import annotations

import numpy as np
import pytest

from src import config
from src.sensors import SensorSuite
from src.track import Track
from src.vehicle import Vehicle


@pytest.fixture(scope="module")
def track() -> Track:
    return Track()


class TestSensorConfig:
    def test_seven_rays(self):
        track = Track()
        sensors = SensorSuite(track=track)
        assert len(sensors.ray_offsets) == 7

    def test_observation_vector_layout(self, track):
        sensors = SensorSuite(track=track)
        v = Vehicle(x=track.centerline[track.start_index][0],
                    y=track.centerline[track.start_index][1],
                    heading=track.start_heading)
        obs = sensors.observe(v)
        assert obs.shape == (8,)  # 7 rays + 1 speed

    def test_speed_channel(self, track):
        sensors = SensorSuite(track=track)
        v = Vehicle(x=track.centerline[track.start_index][0],
                    y=track.centerline[track.start_index][1],
                    heading=track.start_heading)
        v.speed = config.VEHICLE_MAX_SPEED / 2
        obs = sensors.observe(v)
        assert obs[-1] == pytest.approx(0.5)


class TestSensorValues:
    def test_values_normalized(self, track):
        sensors = SensorSuite(track=track)
        v = Vehicle(x=track.centerline[track.start_index][0],
                    y=track.centerline[track.start_index][1],
                    heading=track.start_heading)
        for _ in range(15):
            v.step(0.1)
        obs = sensors.observe(v)
        assert np.all(obs >= 0.0) and np.all(obs <= 1.0)

    def test_values_finite(self, track):
        sensors = SensorSuite(track=track)
        v = Vehicle(x=track.centerline[0][0], y=track.centerline[0][1],
                    heading=track.start_heading)
        obs = sensors.observe(v)
        assert np.all(np.isfinite(obs))

    def test_wall_close_ahead_reads_low(self, track):
        """Facing the boundary directly should give a small forward value."""
        sensors = SensorSuite(track=track, step=0.5)
        x, y = track.point_at_index(0)
        # Heading straight at the outer boundary: compute outward direction.
        cx, cy = track.center
        angle = np.arctan2(y - cy, x - cx)
        v = Vehicle(x=x, y=y, heading=float(angle))
        obs = sensors.observe(v)
        assert obs[3] < 0.35   # forward ray hits the wall quickly

    def test_ray_hits_far_when_along_track(self, track):
        """Ray aligned with the centerline tangent should see far."""
        sensors = SensorSuite(track=track)
        x, y = track.point_at_index(0)
        v = Vehicle(x=x, y=y, heading=track.start_heading)
        obs = sensors.observe(v)
        assert obs[3] > 0.5

    def test_distance_capped_at_max_range(self, track):
        """A very short max_range from a point ON the centerline (tangent
        direction) must return exactly max_range (no wall hit within it)."""
        sensors = SensorSuite(track=track, max_range=2.0, step=0.5)
        x, y = track.point_at_index(0)
        d = sensors.cast_single(x, y, track.start_heading)
        assert d == sensors.max_range
