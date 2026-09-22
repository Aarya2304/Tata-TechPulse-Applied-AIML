"""Central configuration for the simulated driving agent.

All important knobs live here: track geometry, vehicle dynamics, sensor
layout, neural-network architecture, GA hyper-parameters and output paths.
Nothing is hardcoded deep inside the modules.
"""

from __future__ import annotations

import math
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
PLOTS_DIR = ARTIFACTS_DIR / "plots"
BEST_AGENT_PATH = ARTIFACTS_DIR / "best_agent.json"
TRAINING_HISTORY_PATH = ARTIFACTS_DIR / "training_history.csv"
BEST_AGENT_REPLAY_PATH = ARTIFACTS_DIR / "best_agent_replay.png"

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
SEED = 42

# ---------------------------------------------------------------------------
# Track
# ---------------------------------------------------------------------------
TRACK_SEED = 42
TRACK_NUM_POINTS = 400          # centerline samples
TRACK_WIDTH = 34.0              # full drivable width (meters)
TRACK_RADIUS_BASE = 95.0        # base radius of the loop (meters)
TRACK_RADIUS_VARIATION = 26.0   # radial variation amplitude (meters)
TRACK_NUM_BUMPS = 7             # number of radial perturbations
TRACK_SMOOTHING_PASSES = 2      # moving-average passes over radii

# ---------------------------------------------------------------------------
# Vehicle (arcade-style kinematic model)
# ---------------------------------------------------------------------------
VEHICLE_MAX_SPEED = 55.0        # m/s hard cap
VEHICLE_MAX_STEER = 0.62        # rad, max steering angle
VEHICLE_ACCEL = 26.0            # m/s^2 at full throttle
VEHICLE_BRAKE = 38.0            # m/s^2 at full brake
VEHICLE_DRAG = 0.35             # passive drag coefficient (1/s)
VEHICLE_WHEELBASE = 2.6         # meters, bicycle-model wheelbase

# ---------------------------------------------------------------------------
# Episode / simulation
# ---------------------------------------------------------------------------
EPISODE_MAX_STEPS = 900         # per-agent simulation step budget
EPISODE_TIME_STEP = 0.1         # seconds per simulation step

# ---------------------------------------------------------------------------
# Sensors
# ---------------------------------------------------------------------------
SENSOR_RAY_OFFSETS = (          # radians relative to heading, 7 rays
    -1.10,   # far-left
    -0.60,   # left
    -0.25,   # slight-left
     0.0,    # forward
     0.25,   # slight-right
     0.60,   # right
     1.10,   # far-right
)
SENSOR_MAX_RANGE = 120.0        # meters; rays beyond this report 1.0 (clear)

# ---------------------------------------------------------------------------
# Neural network
# ---------------------------------------------------------------------------
NN_INPUT_SIZE = 8               # 7 ray distances + normalized speed
NN_HIDDEN_SIZE = 8
NN_OUTPUT_SIZE = 2              # [steering, throttle]

# ---------------------------------------------------------------------------
# Genetic algorithm
# ---------------------------------------------------------------------------
GA_POPULATION_SIZE = 80
GA_GENERATIONS = 50
GA_TOURNAMENT_SIZE = 4
GA_ELITE_COUNT = 4              # copied unchanged into the next generation
GA_CROSSOVER_RATE = 0.9
GA_MUTATION_RATE = 0.12         # per-gene probability
GA_MUTATION_STRENGTH = 0.25     # gaussian std added to a gene
GA_SEED = 42

# ---------------------------------------------------------------------------
# Fitness function
# ---------------------------------------------------------------------------
FITNESS_PROGRESS_WEIGHT = 1.0   # per meter of forward centerline progress
FITNESS_LAP_BONUS = 250.0       # bonus per completed lap
FITNESS_SPEED_BONUS = 0.35      # per (m/s * s) accumulated while on track
FITNESS_OFF_TRACK_PENALTY = 40.0  # one-time penalty when the agent crashes
FITNESS_STUCK_PATIENCE = 90     # steps without progress before early stop
FITNESS_STUCK_THRESHOLD = 0.5   # meters of progress considered "movement"
