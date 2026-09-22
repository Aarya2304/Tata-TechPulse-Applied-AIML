"""Procedural, deterministic, closed-loop 2D race track.

The centerline is a circle whose radius is perturbed by a fixed number of
smooth sinusoidal "bumps" (deterministic given ``seed``), then smoothed with
a moving average and resampled to ``TRACK_NUM_POINTS`` equally spaced arc
positions.  Inner and outer boundaries are the centerline offset by
``+/- TRACK_WIDTH / 2`` along the local normal.

Everything derives from an explicit seed, so the same seed always yields the
same track (unit-tested).
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

import numpy as np

from src import config


@dataclass
class Track:
    """Closed-loop track sampled at ``num_points`` centerline points."""

    num_points: int = config.TRACK_NUM_POINTS
    width: float = config.TRACK_WIDTH
    seed: int = config.TRACK_SEED
    radius_base: float = config.TRACK_RADIUS_BASE
    radius_variation: float = config.TRACK_RADIUS_VARIATION
    num_bumps: int = config.TRACK_NUM_BUMPS
    smoothing_passes: int = config.TRACK_SMOOTHING_PASSES
    center: tuple = (0.0, 0.0)

    # Filled during __post_init__
    centerline: np.ndarray = field(init=False, repr=False, default=None)
    outer_boundary: np.ndarray = field(init=False, repr=False, default=None)
    inner_boundary: np.ndarray = field(init=False, repr=False, default=None)
    arc_lengths: np.ndarray = field(init=False, repr=False, default=None)
    total_length: float = field(init=False, repr=False, default=0.0)
    start_index: int = field(init=False, repr=False, default=0)
    start_heading: float = field(init=False, repr=False, default=0.0)
    # Cached signed distance grid for fast batched on-track checks
    _df_grid: np.ndarray = field(init=False, repr=False, default=None)
    _df_ox: float = field(init=False, repr=False, default=0.0)
    _df_oy: float = field(init=False, repr=False, default=0.0)
    _df_res: float = field(init=False, repr=False, default=0.0)

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------
    def __post_init__(self) -> None:
        self._build_centerline()
        self._build_boundaries()
        self._compute_arc_lengths()
        self._pick_start()

    def _build_centerline(self) -> None:
        """Deterministic wavy circle, resampled to equally spaced points."""
        rng = random.Random(self.seed)

        # Random-ish (but seeded) phase for each bump frequency component.
        phases = [rng.uniform(0.0, 2.0 * math.pi) for _ in range(self.num_bumps)]

        # Oversample the wavy circle, then resample by arc length so the
        # final points are evenly spaced along the curve.
        oversample = 6
        n_raw = self.num_points * oversample
        angles = [2.0 * math.pi * i / n_raw for i in range(n_raw)]

        radii = []
        for ang in angles:
            r = float(self.radius_base)
            for k, phase in enumerate(phases, start=1):
                r += self.radius_variation / k * math.sin(k * ang + phase)
            radii.append(r)

        # Moving-average smoothing of the radius profile.
        for _ in range(self.smoothing_passes):
            smoothed = []
            n = len(radii)
            for i in range(n):
                smoothed.append(
                    (radii[i - 1] + radii[i] + radii[(i + 1) % n]) / 3.0
                )
            radii = smoothed

        cx, cy = self.center
        raw = np.array(
            [[cx + r * math.cos(a), cy + r * math.sin(a)]
             for r, a in zip(radii, angles)],
            dtype=float,
        )
        self.centerline = self._resample_closed_polyline(raw, self.num_points)

    @staticmethod
    def _resample_closed_polyline(points: np.ndarray, count: int) -> np.ndarray:
        """Resample a closed polyline to ``count`` equally spaced vertices."""
        closed = np.vstack([points, points[:1]])
        seg = np.diff(closed, axis=0)
        seg_len = np.hypot(seg[:, 0], seg[:, 1])
        cumulative = np.concatenate([[0.0], np.cumsum(seg_len)])
        total = cumulative[-1]
        targets = np.linspace(0.0, total, count, endpoint=False)
        idx = np.searchsorted(cumulative, targets, side="right") - 1
        idx = np.clip(idx, 0, len(seg_len) - 1)
        t = (targets - cumulative[idx]) / np.maximum(seg_len[idx], 1e-12)
        resampled = points[idx] + seg[idx] * t[:, None]
        return resampled

    def _build_boundaries(self) -> None:
        """Offset the centerline by +/- width/2 along the local normal."""
        pts = self.centerline
        tangents = np.gradient(pts, axis=0)
        norms = np.hypot(tangents[:, 0], tangents[:, 1])
        norms[norms == 0] = 1.0
        tangents = tangents / norms[:, None]
        normals = np.column_stack([-tangents[:, 1], tangents[:, 0]])
        half = self.width / 2.0
        self.outer_boundary = pts + normals * half
        self.inner_boundary = pts - normals * half

    def _compute_arc_lengths(self) -> None:
        pts = np.vstack([self.centerline, self.centerline[:1]])
        seg = np.diff(pts, axis=0)
        seg_len = np.hypot(seg[:, 0], seg[:, 1])
        self.arc_lengths = np.concatenate([[0.0], np.cumsum(seg_len)])
        self.total_length = float(self.arc_lengths[-1])

    def _pick_start(self) -> None:
        # Start on the flattest section: pick the vertex whose local
        # curvature is smallest, biased toward the rightmost point of the
        # loop for a natural "start line" feel.
        pts = self.centerline
        cx = float(pts[:, 0].mean())
        right_most = int(np.argmax(pts[:, 0]))
        self.start_index = right_most
        prev = pts[(right_most - 1) % self.num_points]
        nxt = pts[(right_most + 1) % self.num_points]
        self.start_heading = math.atan2(nxt[1] - prev[1], nxt[0] - prev[0])
        del cx  # unused; kept for readability

    # ------------------------------------------------------------------
    # Geometry queries
    # ------------------------------------------------------------------
    def _build_distance_grid(self, resolution: float = 1.5) -> None:
        """Cache a signed distance-to-centerline grid for fast queries.

        Values are exact at grid nodes (min over all segments) and bilinearly
        interpolated in between.  Used by ``is_on_track_batched``.
        """
        pts = self.centerline
        pad = self.width / 2.0 + 10.0
        x_min, y_min = pts.min(axis=0) - pad
        x_max, y_max = pts.max(axis=0) + pad
        nx = int((x_max - x_min) / resolution) + 2
        ny = int((y_max - y_min) / resolution) + 2

        gx = x_min + np.arange(nx) * resolution
        gy = y_min + np.arange(ny) * resolution
        X, Y = np.meshgrid(gx, gy, indexing="ij")

        # Min distance from every grid node to every centerline segment.
        a = pts
        b = np.roll(pts, -1, axis=0)
        seg = b - a
        seg_len2 = np.einsum("ij,ij->i", seg, seg)
        seg_len2[seg_len2 == 0] = 1e-12

        # Process grid in flat chunks to bound memory.
        dmin = np.empty(X.size, dtype=float)
        flat_x, flat_y = X.ravel(), Y.ravel()
        chunk = 20000
        for s in range(0, flat_x.size, chunk):
            px = flat_x[s:s + chunk][:, None]
            py = flat_y[s:s + chunk][:, None]
            ax, ay = a[:, 0][None, :], a[:, 1][None, :]
            sx, sy = seg[:, 0][None, :], seg[:, 1][None, :]
            t = np.clip(((px - ax) * sx + (py - ay) * sy) / seg_len2[None, :],
                        0.0, 1.0)
            proj_x = ax + t * sx
            proj_y = ay + t * sy
            dmin[s:s + chunk] = np.min(
                (px - proj_x) ** 2 + (py - proj_y) ** 2, axis=1)
        self._df_grid = np.sqrt(dmin).reshape(nx, ny)
        self._df_ox, self._df_oy = x_min, y_min
        self._df_res = resolution

    def is_on_track_batched(self, xs, ys, margin: float = 0.0) -> np.ndarray:
        """Vectorised version of ``is_on_track`` for arrays of points."""
        if self._df_grid is None:
            self._build_distance_grid()
        xs = np.asarray(xs, dtype=float)
        ys = np.asarray(ys, dtype=float)
        fx = np.clip((xs - self._df_ox) / self._df_res, 0,
                     self._df_grid.shape[0] - 1.001)
        fy = np.clip((ys - self._df_oy) / self._df_res, 0,
                     self._df_grid.shape[1] - 1.001)
        i0 = fx.astype(int)
        j0 = fy.astype(int)
        tx = fx - i0
        ty = fy - j0
        g = self._df_grid
        d = (g[i0, j0] * (1 - tx) * (1 - ty)
             + g[i0 + 1, j0] * tx * (1 - ty)
             + g[i0, j0 + 1] * (1 - tx) * ty
             + g[i0 + 1, j0 + 1] * tx * ty)
        return d <= self.width / 2.0 + margin

    def nearest_centerline_index(self, x: float, y: float) -> int:
        """Index of the closest centerline vertex to (x, y)."""
        d2 = (self.centerline[:, 0] - x) ** 2 + (self.centerline[:, 1] - y) ** 2
        return int(np.argmin(d2))

    def nearest_centerline_index_batched(self, xs, ys) -> np.ndarray:
        """Vectorised nearest centerline vertex for arrays of points."""
        xs = np.asarray(xs, dtype=float)[:, None]
        ys = np.asarray(ys, dtype=float)[:, None]
        d2 = (self.centerline[:, 0][None, :] - xs) ** 2 + \
             (self.centerline[:, 1][None, :] - ys) ** 2
        return np.argmin(d2, axis=1)

    def nearest_centerline_point(self, x: float, y: float) -> tuple:
        """Closest point ON the centerline as (x, y, index)."""
        i = self.nearest_centerline_index(x, y)
        a = self.centerline[i]
        b = self.centerline[(i + 1) % self.num_points]
        p = self.centerline[(i - 1) % self.num_points]
        # Project onto both adjacent segments and keep the better one.
        best = (float(a[0]), float(a[1]), i)
        best_d = (best[0] - x) ** 2 + (best[1] - y) ** 2
        for u, v in ((a, b), (p, a)):
            seg = v - u
            denom = float(seg @ seg)
            if denom < 1e-12:
                continue
            t = float(np.clip(((x - u[0]) * seg[0] + (y - u[1]) * seg[1]) / denom, 0.0, 1.0))
            proj = (u[0] + t * seg[0], u[1] + t * seg[1])
            d = (proj[0] - x) ** 2 + (proj[1] - y) ** 2
            if d < best_d:
                best_d = d
                best = (proj[0], proj[1], i)
        return best

    def distance_to_centerline(self, x: float, y: float) -> float:
        px, py, _ = self.nearest_centerline_point(x, y)
        return math.hypot(px - x, py - y)

    def is_on_track(self, x: float, y: float, margin: float = 0.0) -> bool:
        """True when (x, y) is within width/2 (+ margin) of the centerline."""
        return self.distance_to_centerline(x, y) <= self.width / 2.0 + margin

    def arc_length_at(self, index: int) -> float:
        return float(self.arc_lengths[index])

    def point_at_index(self, index: int) -> tuple:
        i = index % self.num_points
        return float(self.centerline[i][0]), float(self.centerline[i][1])
