"""Episode simulation: rolls out controllers on the track and scores them.

Fitness (deterministic, documented in the README)
-------------------------------------------------
    fitness = 1.0 * progress_meters            # forward centerline progress
            + 0.35 * speed_integral            # sum(speed * dt) while driving
            + 250.0 * laps_completed           # fractional laps counted
            - 40.0  * crashed

Only *forward* progress counts.  Progress is arc length along the centerline
via a nearest-vertex tracker that handles the loop wrap and clamps per-step
deltas, so back-and-forth wiggling cannot fake progress.

Two execution paths:

* :meth:`Simulator.run_episode` -- one vehicle at a time (used by the replay
  and the tests).
* :meth:`Simulator.run_population` -- all genomes evaluated simultaneously
  with fully vectorised physics, sensing and network forward passes.  The
  dynamics and fitness are identical to the single-vehicle path; this is
  purely a performance optimisation for the GA training loop.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from src import config
from src.neural_network import NeuralNetwork
from src.sensors import SensorSuite
from src.track import Track
from src.vehicle import Vehicle


class ProgressTracker:
    """Measures forward progress as exact arc length along the centerline.

    The vehicle position is projected onto the nearest centerline segment;
    the signed arc position (plus completed laps) is tracked across steps.
    Per-step movement is clamped to the physically possible maximum
    (``VEHICLE_MAX_SPEED * dt`` plus one segment length of slack for the
    nearest-segment handover at the seam), so neither vertex-index jitter
    nor cutting across the loop can fake progress.  Backward driving simply
    reduces the accumulated total.
    """

    def __init__(self, track: Track) -> None:
        self.track = track
        self.last_s = track.arc_length_at(track.start_index)
        self.lap_offset = 0.0        # accumulated wrap correction (m)
        self.progress = 0.0
        self.laps = 0.0

    def update(self, x: float, y: float) -> float:
        """Feed the current position; returns the new total progress (m)."""
        n = self.track.num_points
        total = self.track.total_length
        seg_len = total / n

        # Nearest centerline VERTEX, then exact projection onto the best
        # of the two adjacent segments.
        i = self.track.nearest_centerline_index(x, y)
        a = self.track.centerline[i]
        b = self.track.centerline[(i + 1) % n]
        c = self.track.centerline[(i - 1) % n]
        best_s = self.track.arc_length_at(i)
        best_d = (x - a[0]) ** 2 + (y - a[1]) ** 2
        for u, v, s_u in ((a, b, self.track.arc_lengths[i]),
                          (c, a, self.track.arc_lengths[(i - 1) % n])):
            seg = v - u
            denom = float(seg[0] * seg[0] + seg[1] * seg[1])
            if denom < 1e-12:
                continue
            t = float(np.clip(((x - u[0]) * seg[0] + (y - u[1]) * seg[1])
                              / denom, 0.0, 1.0))
            d = (x - (u[0] + t * seg[0])) ** 2 + \
                (y - (u[1] + t * seg[1])) ** 2
            if d < best_d:
                best_d = d
                best_s = s_u + t * seg_len

        s_abs = best_s + self.lap_offset
        ds = s_abs - self.last_s

        # Loop wrap handling: crossing the start/finish line moves the
        # absolute arc coordinate by +/- the full track length.
        if ds > total / 2.0:
            self.lap_offset -= total
            ds -= total
        elif ds < -total / 2.0:
            self.lap_offset += total
            ds += total

        # Physical clamp: max possible movement per step (+ handover slack).
        max_step = seg_len + config.VEHICLE_MAX_SPEED * config.EPISODE_TIME_STEP
        ds = max(-max_step, min(max_step, ds))

        self.progress += ds
        self.last_s = s_abs
        self.laps = self.progress / total
        return self.progress

    def forward_progress(self) -> float:
        return max(self.progress, 0.0)


@dataclass
class EpisodeResult:
    """Outcome of one simulated episode."""

    fitness: float
    progress: float
    laps: float
    steps_survived: int
    crashed: bool
    reason: str
    trajectory: list = field(default_factory=list)
    sensor_history: list = field(default_factory=list)


class Simulator:
    """Runs episodes of vehicles controlled by neural networks."""

    def __init__(self, track: Track) -> None:
        self.track = track
        self.sensors = SensorSuite(track=track)
        # Build the cached distance grid once, up front.
        self.track._build_distance_grid()

    # ------------------------------------------------------------------
    # Fitness
    # ------------------------------------------------------------------
    @staticmethod
    def compute_fitness(progress, speed_integral,
                        laps, crashed):
        """Deterministic fitness from the documented formula.

        Accepts scalars or numpy arrays (for batched evaluation)."""
        progress = np.asarray(progress, dtype=float)
        fitness = (config.FITNESS_PROGRESS_WEIGHT * np.maximum(progress, 0.0)
                   + config.FITNESS_SPEED_BONUS * np.asarray(speed_integral, dtype=float)
                   + config.FITNESS_LAP_BONUS * np.asarray(laps, dtype=float)
                   - (config.FITNESS_OFF_TRACK_PENALTY * np.asarray(crashed, dtype=bool)))
        if fitness.ndim == 0:
            return float(fitness)
        return fitness

    # ------------------------------------------------------------------
    # Single-vehicle episode (clarity / replay / tests)
    # ------------------------------------------------------------------
    def run_episode(self, network: NeuralNetwork,
                    record: bool = False) -> EpisodeResult:
        """Simulate one episode for ``network``; returns fitness + stats.

        Uses the exact same slice-grid geometry as the batched training
        path, so the replayed fitness matches the training fitness of the
        saved agent.
        """
        cfg = config
        vehicle = Vehicle(
            x=self.track.centerline[self.track.start_index][0],
            y=self.track.centerline[self.track.start_index][1],
            heading=self.track.start_heading,
        )
        tracker = ProgressTracker(self.track)

        progress = 0.0
        speed_integral = 0.0
        stuck_steps = 0
        crashed = False
        reason = "max_steps"
        trajectory = []
        sensor_history = []

        for step in range(cfg.EPISODE_MAX_STEPS):
            obs = self.sensors.observe(vehicle)
            steering, throttle = network.forward(obs)

            vehicle.steering = steering * cfg.VEHICLE_MAX_STEER
            vehicle.throttle = throttle
            vehicle.step(cfg.EPISODE_TIME_STEP)

            if not self.track.is_on_track_batched(
                    np.array([vehicle.x]), np.array([vehicle.y]))[0]:
                crashed = True
                reason = "off_track"
                break

            progress = tracker.update(vehicle.x, vehicle.y)
            speed_integral += vehicle.speed * cfg.EPISODE_TIME_STEP

            if progress >= cfg.FITNESS_STUCK_THRESHOLD:
                stuck_steps = 0
            else:
                stuck_steps += 1
                if stuck_steps >= cfg.FITNESS_STUCK_PATIENCE:
                    reason = "stuck"
                    break

            if record:
                trajectory.append((vehicle.x, vehicle.y, vehicle.heading))
                sensor_history.append(obs.copy())

        fitness = self.compute_fitness(
            progress=tracker.forward_progress(),
            speed_integral=speed_integral,
            laps=tracker.laps,
            crashed=crashed,
        )
        return EpisodeResult(
            fitness=fitness,
            progress=tracker.forward_progress(),
            laps=tracker.laps,
            steps_survived=step + 1,
            crashed=crashed,
            reason=reason,
            trajectory=trajectory if record else [],
            sensor_history=sensor_history if record else [],
        )

    # ------------------------------------------------------------------
    # Vectorised population evaluation (training fast path)
    # ------------------------------------------------------------------
    def _arc_positions_batched(self, xs, ys, idx) -> np.ndarray:
        """Exact arc positions along the centerline for points and their
        nearest vertex indices (batched mirror of ProgressTracker.update)."""
        n = self.track.num_points
        pts = self.track.centerline
        arc = self.track.arc_lengths
        seg_len = self.track.total_length / n

        a = pts[idx]
        b = pts[(idx + 1) % n]
        c = pts[(idx - 1) % n]
        s_a = arc[idx]
        s_c = arc[(idx - 1) % n]

        # Projection onto segment a->b.
        seg_ab = b - a
        len2_ab = np.einsum("ij,ij->i", seg_ab, seg_ab)
        len2_ab[len2_ab == 0] = 1e-12
        t_ab = np.clip(((xs - a[:, 0]) * seg_ab[:, 0]
                        + (ys - a[:, 1]) * seg_ab[:, 1]) / len2_ab, 0.0, 1.0)
        d_ab = ((xs - (a[:, 0] + t_ab * seg_ab[:, 0])) ** 2
                + (ys - (a[:, 1] + t_ab * seg_ab[:, 1])) ** 2)

        # Projection onto segment c->a.
        seg_ca = a - c
        len2_ca = np.einsum("ij,ij->i", seg_ca, seg_ca)
        len2_ca[len2_ca == 0] = 1e-12
        t_ca = np.clip(((xs - c[:, 0]) * seg_ca[:, 0]
                        + (ys - c[:, 1]) * seg_ca[:, 1]) / len2_ca, 0.0, 1.0)
        d_ca = ((xs - (c[:, 0] + t_ca * seg_ca[:, 0])) ** 2
                + (ys - (c[:, 1] + t_ca * seg_ca[:, 1])) ** 2)

        return np.where(d_ab <= d_ca,
                        s_a + t_ab * seg_len,
                        s_c + t_ca * seg_len)

    def run_population(self, genomes: np.ndarray) -> tuple:
        """Evaluate a whole population simultaneously.

        Returns ``(fitnesses, progresses, laps, crashed_flags)`` as numpy
        arrays.  Physics, sensing and fitness are identical to
        :meth:`run_episode`; only the batching differs.
        """
        cfg = config
        genomes = np.atleast_2d(np.asarray(genomes, dtype=float))
        n = genomes.shape[0]

        # Reshape genomes into network parameters (matches NeuralNetwork
        # layout: W1, b1, W2, b2).
        in_h = cfg.NN_INPUT_SIZE * cfg.NN_HIDDEN_SIZE
        h_o = cfg.NN_HIDDEN_SIZE * cfg.NN_OUTPUT_SIZE
        W1 = genomes[:, :in_h].reshape(n, cfg.NN_INPUT_SIZE, cfg.NN_HIDDEN_SIZE)
        b1 = genomes[:, in_h:in_h + cfg.NN_HIDDEN_SIZE]
        i2 = in_h + cfg.NN_HIDDEN_SIZE
        W2 = genomes[:, i2:i2 + h_o].reshape(n, cfg.NN_HIDDEN_SIZE,
                                             cfg.NN_OUTPUT_SIZE)
        b2 = genomes[:, i2 + h_o:]

        x = np.full(n, float(self.track.centerline[self.track.start_index][0]))
        y = np.full(n, float(self.track.centerline[self.track.start_index][1]))
        heading = np.full(n, float(self.track.start_heading))
        speed = np.zeros(n)

        last_s = np.full(n, float(self.track.arc_lengths[self.track.start_index]))
        lap_offset = np.zeros(n)
        progress = np.zeros(n)
        laps = np.zeros(n)
        speed_integral = np.zeros(n)
        crashed = np.zeros(n, dtype=bool)
        alive = np.ones(n, dtype=bool)
        stuck = np.zeros(n, dtype=int)
        total_len = self.track.total_length
        seg_len = total_len / self.track.num_points
        max_step = seg_len + (config.VEHICLE_MAX_SPEED
                              * config.EPISODE_TIME_STEP)

        for _ in range(cfg.EPISODE_MAX_STEPS):
            if not alive.any():
                break

            # --- sense (only for living agents) -------------------------
            obs = self.sensors.observe_batched(
                x[alive], y[alive], heading[alive], speed[alive])

            # --- think (vectorised forward pass, per-agent matmul) ------
            W1_a, b1_a = W1[alive], b1[alive]
            W2_a, b2_a = W2[alive], b2[alive]
            hidden = np.tanh(np.einsum("ni,nih->nh", obs, W1_a) + b1_a)
            raw = np.einsum("nh,nho->no", hidden, W2_a) + b2_a
            steer = np.tanh(raw[:, 0])
            thr = 2.0 / (1.0 + np.exp(-raw[:, 1])) - 1.0

            # --- act (arcade bicycle model, batched) --------------------
            steer_angle = steer * cfg.VEHICLE_MAX_STEER
            acc = np.where(thr >= 0.0,
                           thr * cfg.VEHICLE_ACCEL,
                           thr * cfg.VEHICLE_BRAKE)
            speed_a = (speed[alive]
                       - cfg.VEHICLE_DRAG * speed[alive] * cfg.EPISODE_TIME_STEP
                       + acc * cfg.EPISODE_TIME_STEP)
            speed_a = np.clip(speed_a, 0.0, cfg.VEHICLE_MAX_SPEED)
            heading_rate = (speed_a / (cfg.VEHICLE_WHEELBASE *
                                       (1.0 + speed_a / cfg.VEHICLE_MAX_SPEED))) \
                * np.tan(steer_angle)
            heading_a = heading[alive] + heading_rate * cfg.EPISODE_TIME_STEP
            x_a = x[alive] + speed_a * np.cos(heading_a) * cfg.EPISODE_TIME_STEP
            y_a = y[alive] + speed_a * np.sin(heading_a) * cfg.EPISODE_TIME_STEP

            # --- crash detection ----------------------------------------
            idx_alive = alive.nonzero()[0]
            on = self.track.is_on_track_batched(x_a, y_a)
            just_crashed = idx_alive[~on]
            crashed[just_crashed] = True
            survivors = idx_alive[on]

            # --- progress for survivors (exact arc projection) -----------
            if survivors.size:
                idx_new = self.track.nearest_centerline_index_batched(
                    x_a[on], y_a[on])
                s_new = self._arc_positions_batched(x_a[on], y_a[on], idx_new)
                s_abs = s_new + lap_offset[survivors]
                ds = s_abs - last_s[survivors]

                # Loop wrap handling (same rule as ProgressTracker).
                wrap_fwd = ds > total_len / 2.0
                wrap_bwd = ds < -total_len / 2.0
                lap_offset[survivors[wrap_fwd]] -= total_len
                ds[wrap_fwd] -= total_len
                lap_offset[survivors[wrap_bwd]] += total_len
                ds[wrap_bwd] += total_len

                ds = np.clip(ds, -max_step, max_step)
                progress[survivors] += ds
                last_s[survivors] = s_abs
                speed_integral[survivors] += \
                    speed_a[on] * cfg.EPISODE_TIME_STEP
                laps[survivors] = progress[survivors] / total_len

                stuck[survivors] = np.where(
                    progress[survivors] >= cfg.FITNESS_STUCK_THRESHOLD,
                    0, stuck[survivors] + 1)

            # --- update state / deactivate crashed & stuck agents --------
            x[survivors] = x_a[on]
            y[survivors] = y_a[on]
            heading[survivors] = heading_a[on]
            speed[survivors] = speed_a[on]
            alive[just_crashed] = False
            stuck_out = survivors[stuck[survivors]
                                  >= cfg.FITNESS_STUCK_PATIENCE]
            alive[stuck_out] = False

        # Crashed/stuck agents keep the progress they earned.
        progress_fwd = np.maximum(progress, 0.0)
        fitness = np.where(
            crashed,
            self.compute_fitness(progress_fwd, speed_integral, laps, True),
            self.compute_fitness(progress_fwd, speed_integral, laps, False),
        )
        return fitness, progress_fwd, laps, crashed
