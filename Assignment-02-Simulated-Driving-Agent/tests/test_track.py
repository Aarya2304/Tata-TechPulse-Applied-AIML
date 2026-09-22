"""Tests for the procedural track module."""

from __future__ import annotations

import math

import numpy as np
import pytest

from src.track import Track


class TestTrackGeneration:
    def test_centerline_exists_and_shape(self):
        track = Track()
        assert track.centerline is not None
        assert track.centerline.shape == (track.num_points, 2)

    def test_track_is_closed_loop(self):
        track = Track()
        pts = track.centerline
        # Consecutive gaps roughly equal, loop closes on itself.
        gaps = np.hypot(*(pts - np.roll(pts, -1, axis=0)).T)
        assert gaps.max() / gaps.min() < 2.0

    def test_boundaries_valid(self):
        track = Track()
        outer, inner = track.outer_boundary, track.inner_boundary
        assert outer.shape == (track.num_points, 2)
        assert inner.shape == (track.num_points, 2)
        # Boundaries offset from centerline by ~ width/2 everywhere.
        d_out = np.hypot(*(outer - track.centerline).T)
        d_in = np.hypot(*(inner - track.centerline).T)
        assert np.allclose(d_out, track.width / 2, rtol=0.05)
        assert np.allclose(d_in, track.width / 2, rtol=0.05)
        # Total length positive and plausible.
        assert track.total_length > 2 * math.pi * (track.radius_base - 10)

    def test_deterministic_seed_reproducible(self):
        a, b = Track(seed=123), Track(seed=123)
        assert np.allclose(a.centerline, b.centerline)
        c = Track(seed=124)
        assert not np.allclose(a.centerline, c.centerline)

    def test_curvature_present(self):
        """Heading must actually change around the loop (steering needed)."""
        track = Track()
        pts = track.centerline
        headings = np.arctan2(*np.diff(np.vstack([pts, pts[:1]]), axis=0).T[::-1])
        heading_change = np.abs(np.diff(np.unwrap(headings))).max()
        assert heading_change > 0.05

    def test_start_on_track(self):
        track = Track()
        sx, sy = track.point_at_index(track.start_index)
        assert track.is_on_track(sx, sy)


class TestTrackQueries:
    def test_nearest_index_on_centerline(self):
        track = Track()
        for i in (0, 50, 199, 357):
            x, y = track.point_at_index(i)
            assert track.nearest_centerline_index(x, y) == i

    def test_distance_to_centerline_zero_on_line(self):
        track = Track()
        x, y = track.point_at_index(100)
        assert track.distance_to_centerline(x, y) < 1e-6

    def test_off_track_detection(self):
        track = Track()
        x, y = track.point_at_index(0)
        # A point far outside the loop must be off track.
        far = (x + (track.width / 2 + 20), y)
        assert not track.is_on_track(*far)

    def test_arc_length_lookup(self):
        track = Track()
        assert track.arc_length_at(0) == 0.0
        assert track.arc_length_at(track.num_points // 2) == \
            pytest.approx(track.total_length / 2, rel=0.15)
