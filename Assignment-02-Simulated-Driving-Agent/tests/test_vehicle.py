"""Tests for the vehicle module."""

from __future__ import annotations

import math

import pytest

from src import config
from src.vehicle import Vehicle


class TestVehicleState:
    def test_initial_state_valid(self):
        v = Vehicle(x=10.0, y=-5.0, heading=0.7)
        assert v.x == 10.0 and v.y == -5.0
        assert v.speed == 0.0
        assert v.on_track is True

    def test_reset_restores_state(self):
        v = Vehicle(x=0, y=0, heading=0)
        v.speed = 30.0
        v.steering = 0.5
        v.reset(x=5, y=6, heading=1.0)
        assert (v.x, v.y, v.heading, v.speed) == (5.0, 6.0, 1.0, 0.0)

    def test_commands_clamped(self):
        v = Vehicle(x=0, y=0, heading=0)
        v.steering = 99.0
        v.throttle = 5.0
        v.clamp_state()
        assert v.steering == config.VEHICLE_MAX_STEER
        assert v.throttle == 1.0


class TestVehicleMovement:
    def _drive(self, v: Vehicle, steps: int, dt: float = 0.1):
        for _ in range(steps):
            v.step(dt)

    def test_forward_movement(self):
        v = Vehicle(x=0, y=0, heading=0)
        v.throttle = 1.0
        self._drive(v, 20)
        assert v.x > 10          # moved clearly along +x
        assert v.speed > 5

    def test_heading_changes_with_steering(self):
        v1 = Vehicle(x=0, y=0, heading=0)
        v1.throttle = 1.0
        v1.steering = 0.4
        self._drive(v1, 20)
        assert v1.heading > 0.05  # turned left

        v2 = Vehicle(x=0, y=0, heading=0)
        v2.throttle = 1.0
        v2.steering = -0.4
        self._drive(v2, 20)
        assert v2.heading < -0.05  # turned right

    def test_no_turn_at_zero_speed(self):
        v = Vehicle(x=0, y=0, heading=0.3)
        v.steering = 0.5
        self._drive(v, 10)
        assert v.heading == pytest.approx(0.3)

    def test_speed_bounded(self):
        v = Vehicle(x=0, y=0, heading=0)
        v.throttle = 1.0
        self._drive(v, 300)
        assert 0.0 <= v.speed <= config.VEHICLE_MAX_SPEED + 1e-9

    def test_braking_reduces_speed(self):
        v = Vehicle(x=0, y=0, heading=0)
        v.throttle = 1.0
        self._drive(v, 30)
        fast = v.speed
        v.throttle = -1.0
        self._drive(v, 10)
        assert v.speed < fast

    def test_movement_deterministic(self):
        a = Vehicle(x=0, y=0, heading=0.1)
        b = Vehicle(x=0, y=0, heading=0.1)
        for va in (a, b):
            va.throttle = 0.8
            va.steering = 0.2
            for _ in range(50):
                va.step(0.1)
        assert (a.x, a.y, a.heading, a.speed) == (b.x, b.y, b.heading, b.speed)
