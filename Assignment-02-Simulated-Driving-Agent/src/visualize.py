"""Visualisation utilities and best-agent replay (no GUI required).

Figures (saved under ``artifacts/plots/``):

* ``plot_track``            -- centerline + boundaries
* ``plot_best_trajectory``  -- track + best-agent path + start marker
* ``plot_fitness_curve``    -- best/average fitness per generation
* ``plot_sensor_fan``       -- sensor rays for the best agent at a given step
* ``run_replay``            -- CLI entry point (``python -m src.visualize``)

All functions use the Agg backend; nothing opens a window.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from src import config
from src.neural_network import NeuralNetwork
from src.simulation import Simulator
from src.track import Track

TRACK_COLOR = "#444444"
PATH_COLOR = "#1f77b4"


# ---------------------------------------------------------------------------
# Basic plots
# ---------------------------------------------------------------------------
def _draw_track(ax: plt.Axes, track: Track) -> None:
    outer = track.outer_boundary
    inner = track.inner_boundary
    center = track.centerline
    ax.plot(outer[:, 0], outer[:, 1], color=TRACK_COLOR, lw=1.2)
    ax.plot(inner[:, 0], inner[:, 1], color=TRACK_COLOR, lw=1.2)
    ax.plot(center[:, 0], center[:, 1], color="#bbbbbb", lw=0.6,
            ls="--", alpha=0.6)
    ax.set_aspect("equal")


def plot_track(track: Track, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 8))
    _draw_track(ax, track)
    ax.set_title("Generated Track (deterministic seed = "
                 f"{track.seed})")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_best_trajectory(track: Track, trajectory: list, out_path: Path,
                         title: str = "Best agent trajectory") -> None:
    fig, ax = plt.subplots(figsize=(8, 8))
    _draw_track(ax, track)
    if trajectory:
        pts = np.array(trajectory)
        # Colour the path by time to show direction of travel.
        ax.scatter(pts[:, 0], pts[:, 1], c=np.arange(len(pts)),
                   cmap="viridis", s=6, zorder=3)
        ax.plot(pts[:, 0], pts[:, 1], color=PATH_COLOR, lw=0.8, alpha=0.5,
                zorder=2)
        start = pts[0]
        ax.plot(start[0], start[1], marker="*", color="red", ms=16,
                zorder=4, ls="none", label="start")
        # Direction arrow halfway along the path (points are (x, y, heading)).
        mid = len(pts) // 2
        if 1 < mid < len(pts):
            ax.annotate("", xy=pts[mid, :2], xytext=pts[mid - 1, :2],
                        arrowprops=dict(arrowstyle="-|>", color="crimson",
                                        lw=2))
        ax.legend(loc="upper right")
    ax.set_title(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_fitness_curve(history_csv: Path, out_path: Path) -> None:
    """Best/average fitness per generation from training_history.csv."""
    import csv as _csv

    with history_csv.open() as fh:
        rows = list(_csv.DictReader(fh))
    gens = [int(r["generation"]) for r in rows]
    best = [float(r["best_fitness"]) for r in rows]
    avg = [float(r["average_fitness"]) for r in rows]

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.plot(gens, best, color="#d62728", lw=2, label="best fitness")
    ax.plot(gens, avg, color="#1f77b4", lw=2, label="average fitness")
    ax.set_xlabel("Generation")
    ax.set_ylabel("Fitness")
    ax.set_title("Evolution of driving fitness")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_sensor_fan(track: Track, trajectory: list, sensor_history: list,
                    step: int, out_path: Path) -> None:
    """Draw the sensor rays of the recorded best agent at ``step``."""
    fig, ax = plt.subplots(figsize=(8, 8))
    _draw_track(ax, track)
    x, y, heading = trajectory[step]
    obs = sensor_history[step]
    for i, offset in enumerate(config.SENSOR_RAY_OFFSETS):
        angle = heading + offset
        distance = obs[i] * config.SENSOR_MAX_RANGE
        ex = x + math.cos(angle) * distance
        ey = y + math.sin(angle) * distance
        color = "tab:green" if obs[i] > 0.6 else ("tab:orange"
                                                  if obs[i] > 0.3 else "tab:red")
        ax.plot([x, ex], [y, ey], color=color, lw=1.4)
        ax.plot(ex, ey, ".", color=color, ms=6)
    ax.plot(x, y, "o", color="blue", ms=9, zorder=5)
    ax.set_title(f"Sensor rays at recorded step {step}")
    ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Replay of the saved best agent
# ---------------------------------------------------------------------------
def load_best_agent(path: Path = None) -> dict:
    path = path or config.BEST_AGENT_PATH
    return json.loads(Path(path).read_text())


def run_replay(out_dir: Path = None, verbose: bool = True) -> dict:
    """Replay the saved best genome and render replay figures."""
    out_dir = out_dir or config.PLOTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    payload = load_best_agent()
    net = NeuralNetwork(
        input_size=payload["architecture"]["input_size"],
        hidden_size=payload["architecture"]["hidden_size"],
        output_size=payload["architecture"]["output_size"],
    ).set_genome(payload["genome"])

    track = Track(seed=payload["track"]["track_seed"],
                  num_points=payload["track"]["num_points"],
                  width=payload["track"]["width"])
    track._build_distance_grid()
    simulator = Simulator(track)
    result = simulator.run_episode(net, record=True)

    fig_path = config.BEST_AGENT_REPLAY_PATH if hasattr(
        config, "BEST_AGENT_REPLAY_PATH") else out_dir / "best_agent_replay.png"

    plot_best_trajectory(
        track, result.trajectory, out_path=Path(fig_path),
        title=(f"Replay of best agent - fitness {result.fitness:.1f}, "
               f"progress {result.progress:.0f} m, "
               f"{result.laps:.2f} laps ({result.reason})"),
    )
    plot_track(track, out_dir / "track_layout.png")

    # Sensor-fan snapshot mid-episode for illustration.
    if result.trajectory and result.sensor_history:
        mid_step = min(len(result.trajectory) - 1,
                       max(60, len(result.trajectory) // 3))
        plot_sensor_fan(track, result.trajectory, result.sensor_history,
                        step=mid_step, out_path=out_dir / "sensor_fan.png")

    if verbose:
        print("=" * 64)
        print("REPLAY of saved best agent")
        print("=" * 64)
        print(f"  fitness  : {result.fitness:.2f}")
        print(f"  progress : {result.progress:.1f} m "
              f"({result.progress / track.total_length:.1%} of track)")
        print(f"  laps     : {result.laps:.2f}")
        print(f"  steps    : {result.steps_survived} ({result.reason})")
        print(f"  replay   -> {fig_path}")
        print(f"  sensors  -> {out_dir / 'sensor_fan.png'}")

    return {
        "fitness": result.fitness,
        "progress": result.progress,
        "laps": result.laps,
        "steps": result.steps_survived,
        "reason": result.reason,
        "replay_png": str(fig_path),
    }


if __name__ == "__main__":
    run_replay()
