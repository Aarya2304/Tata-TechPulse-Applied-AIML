# Assignment 2 - Simulated Driving Agent Behavior

**Tata Technologies TechPulse FY-26 · Applied AI/ML Course**

> Official objective: *"Build a self-driving agent using Python and genetic
> algorithms."*

A population of neural-network driving agents is evolved with a Genetic
Algorithm on a procedurally generated, closed-loop 2D race track. The agents
perceive the track through 7 ray sensors plus their own speed, decide steering
and throttle with a tiny NumPy neural network, and are selected purely on how
far they drive along the track. After 50 generations the best agent completes
**8.04 laps** of the 668 m track, while the initial population manages only
**0.13 lap**.

This is a **simplified simulated environment** for educational purposes — not a
real autonomous vehicle, not photorealistic, and not physically accurate.

---

## Objective

Reference the official assignment objective:

"Build a self-driving agent using Python and genetic algorithms."

## Problem Statement

A simulated car starts on a curved, closed-loop track and must learn to drive
around it for as long and as far as possible. The agent:

* sees the world only through **7 ray sensors** that measure the distance to
  the track boundaries (plus its own normalised speed),
* controls **steering** and **throttle/brake** each simulation step,
* has **no map, no GPS and no hand-coded rules** — its entire behavior comes
  from 90 evolved neural-network weights.

"Learning" means evolution: agents that drive farther reproduce more, so
successive generations should show measurably greater track progress.

## Approach

1. **Track generation** (`src/track.py`) — a deterministic closed loop: a base
   circle of radius 95 m perturbed by 7 seeded sinusoidal "bumps" (amplitude
   26 m), smoothed and resampled to 400 equally spaced centerline points.
   Inner/outer boundaries are offset ±17 m from the centerline. Total length
   **668 m**; everything derives from `TRACK_SEED = 42`.
2. **Vehicle simulation** (`src/vehicle.py`) — an arcade kinematic bicycle
   model: position, heading, scalar speed; acceleration 26 m/s², braking
   38 m/s², drag 0.35/s, hard speed limit 55 m/s, max steering ±0.62 rad.
   Heading rate = `(speed / effective_wheelbase) · tan(steer)`, so the car
   cannot turn while stationary. Any position farther than half the track
   width from the centerline is **off-track (crash)**.
3. **Ray sensors** (`src/sensors.py`) — 7 rays at −1.10, −0.60, −0.25, 0,
   +0.25, +0.60, +1.10 rad relative to the heading (far-left … far-right).
   Each measures the first off-track sample along its direction (marched in
   1 m steps, capped at 120 m) and is normalised to [0, 1] (1 = clear road).
   Together with normalised speed this yields the **8-dim observation**.
4. **Neural network controller** (`src/neural_network.py`) — pure NumPy.
5. **Fitness function** (`src/simulation.py`) — see the exact formula below.
6. **Genetic Algorithm** (`src/genetic_algorithm.py`) — see configuration
   below. Each generation all 80 agents are evaluated, ranked, the best are
   copied unchanged, and the rest of the next generation is bred by
   tournament selection + uniform crossover + Gaussian mutation.
7. **Evolution across generations** (`src/train.py`) — 50 generations; the
   per-generation best/average fitness and best progress/laps are recorded to
   `artifacts/training_history.csv`, and the best-ever genome is saved.

**Performance note:** the training loop evaluates all 80 agents
*simultaneously* with vectorised NumPy physics, sensing and network forward
passes (a cached distance grid replaces per-sample boundary checks). The
dynamics, geometry and fitness are identical to the step-by-step
single-vehicle path, and both paths were verified to agree to
floating-point precision. Training takes ~2 minutes on a normal CPU.

## Neural Network

```
8 inputs  ->  8 hidden neurons (tanh)  ->  2 outputs
```

* **Inputs (8):** 7 normalised ray distances + 1 normalised speed.
* **Hidden layer:** 8 neurons, `tanh` activation.
* **Outputs (2):**
  * `steering` in [-1, 1] (`tanh`), scaled to ±0.62 rad — negative = right,
    positive = left;
  * `throttle` in [-1, 1] (`2·sigmoid − 1`) — positive = accelerate up to
    26 m/s², negative = brake up to 38 m/s².

The network has `8·8 + 8 + 8·2 + 2 = 90` parameters, flattened into the
**genome** that the GA evolves. The class supports forward pass, genome
extraction (`get_genome`) and exact reconstruction (`set_genome`).

## Genetic Algorithm

| Parameter | Value (actual configuration from `src/config.py`) |
|---|---|
| Population size | 80 |
| Generations | 50 |
| Selection | Tournament (size 4) |
| Crossover | Uniform, rate 0.9 |
| Mutation | Gaussian, per-gene rate 0.12, strength σ = 0.25 |
| Elitism | Top 4 genomes copied unchanged |
| Random seeds | GA seed 42, track seed 42, episode deterministic |

Generation cycle: evaluate all agents → rank by fitness → preserve elites →
tournament-select parents → uniform crossover → Gaussian mutation → next
generation. The best genome ever seen is saved to `artifacts/best_agent.json`.

## Fitness Function

The exact formula (deterministic, computed once per episode):

```
fitness = 1.00 · forward_progress_meters
        + 0.35 · speed_integral            (Σ speed·dt while on track)
        + 250  · laps_completed            (fractional laps count)
        − 40   · crashed                   (one-time off-track penalty)
```

**Track progress** is the driver term. It is measured as *exact arc length*
along the centerline: each position is projected onto the nearest centerline
segment and the signed arc position is integrated across steps, with
closed-loop wrap-around handled at the start/finish line. Per-step progress is
clamped to the physically possible maximum (`55 m/s · 0.1 s` plus one segment
of slack), so neither sensor-vertex jitter nor cutting across the loop can
fake progress. Driving backwards simply reduces the total, and oscillating in
place earns nothing — the fitness cannot be exploited by back-and-forth
movement.

**Anti-exploitation properties:** no reward for raw survival time (a stationary
car earns 0.35·Σspeed·dt = 0), negative reward for crashing, and a
stuck-timeout (90 steps without ≥ 0.5 m progress) ends futile episodes early.

## Results

Real values from the final run of `python -m src.train`
(Python 3.11, NumPy 1.26, seed 42, 118 s wall time). Every number below is
copied from `artifacts/training_history.csv` / `training_log.txt` — nothing is
fabricated.

| Metric | Initial (gen 0) | Final (gen 49) |
|---|---:|---:|
| Best Fitness | 108.55 | **9079.55** |
| Average Fitness | 6.69 | **5666.22** |
| Best Progress | 84.66 m | **5369.2 m** |
| Best Laps | 0.127 | **8.038** |

Run configuration: population 80, generations 50, 900 simulation steps per
episode (dt = 0.1 s), track length 668 m.

| Generation | Best Fitness | Average Fitness | Best Progress | Best Laps |
|---:|---:|---:|---:|---:|
| 0 | 108.55 | 6.69 | 84.66 m | 0.127 |
| 10 | 8645.00 | 1948.62 | 5069.08 m | 7.589 |
| 20 | 8994.51 | 4537.65 | 5313.38 m | 7.954 |
| 30 | 9042.25 | 5833.45 | 5345.40 m | 8.002 |
| 40 | 9079.55 | 5155.68 | 5369.20 m | 8.038 |
| 49 | 9079.55 | 5666.22 | 5369.20 m | 8.038 |

Interpretation:

* The very first population already contains one lucky driver (84.7 m), but
  the average agent barely moves (avg fitness 6.7 ≈ almost zero progress).
* By generation 10 the best genome rounds the track 7.6 times; by generation
  30 the *average* agent completes ~8 laps (avg fitness 5833), showing the
  behavior spread through the population rather than one outlier.
* The final best agent drives the full 900-step episode without crashing and
  accumulates 5369 m of centerline progress = 8.04 laps.
* Sanity: the geometric bound for centerline progress is 900 steps × 55 m/s ×
  0.1 s × (outer-lane factor ≈ 1.22) ≈ 6000 m, so 5369 m is physically
  plausible. Independent verification of the replay trajectory: all 900
  recorded positions are on-track (max 15.92 m from the centerline, limit
  17.0 m) and the largest single step is exactly 5.5 m.

## Visualizations

All figures are generated by the run and stored under `artifacts/plots/`
(linked as files; open them from the repository):

| Figure | File |
|---|---|
| Fitness curve (best + average per generation) | `artifacts/plots/fitness_curve.png` |
| Best-agent trajectory from training | `artifacts/plots/best_agent_trajectory.png` |
| Replay of the saved best agent (required artifact) | `artifacts/best_agent_replay.png` |
| Sensor-ray snapshot mid-replay | `artifacts/plots/sensor_fan.png` |
| Track layout (centerline + boundaries) | `artifacts/plots/track_layout.png` |

The trajectory plots show the track (inner/outer boundaries + dashed
centerline), the agent's path coloured by time, the start position (star) and
a direction arrow at mid-path.

## Testing

`python -m pytest tests -v` — **53 tests, all passing** (0.9 s), covering:

* **Track (9):** generation, centerline shape, closed loop, valid boundaries
  (exact ±width/2 offsets), seed determinism, curvature present, start on
  track, nearest-index queries, off-track detection, arc-length lookup.
* **Vehicle (9):** valid initial state, reset, command clamping, forward
  motion, heading changes with steering (both directions), no turning at
  zero speed, speed bounded, braking, determinism.
* **Sensors (8):** exactly 7 rays, 8-dim observation layout, speed channel,
  normalisation to [0, 1], finite values, wall detection ahead, clear-road
  rays, range capping.
* **Neural network (8):** architecture sizes, genome-size formula, output
  shape/range, deterministic forward pass, input validation, genome
  extraction/round-trip, reconstruction reproduces outputs, random genomes
  differ.
* **Genetic algorithm (19):** population size, genome shapes, reproducible
  initial population, tournament selection (deterministic + statistical),
  crossover dimension preservation and mixing, mutation rates, elitism
  (best agents survive unchanged, mismatched genome size rejected),
  population-size stability, stat reporting, and an end-to-end "GA improves
  the best fitness on a toy landscape" test.

No test runs the full 50-generation training; the heaviest test finishes in
milliseconds. (Project-local `pytest.ini` disables the broken third-party
`pytest-flask` plugin installed in this machine's global Python — it is
unrelated to this assignment.)

## How to Run

```bash
# from this directory (Assignment-02-Simulated-Driving-Agent/)
python -m pip install -r requirements.txt

# train the agents (deterministic, ~2 minutes on a normal CPU)
python -m src.train

# replay the saved best agent and regenerate its figures
python -m src.visualize

# run the automated test suite
python -m pytest tests -v
```

Training writes `artifacts/training_history.csv`, `artifacts/best_agent.json`
and all figures; visualization only reads the saved agent and re-renders its
replay.

## Reproducibility

All randomness is seeded: track geometry (`TRACK_SEED = 42`), GA population /
selection / crossover / mutation (`GA_SEED = 42`), and every episode is fully
deterministic (no exploration noise inside episodes). Running
`python -m src.train` twice with the same configuration reproduces the same
history and the same final agent. This was verified: the replay of the saved
best agent reproduces its training fitness exactly (9079.55), because
training and replay use the same deterministic simulator.

## Project Structure

```
Assignment-02-Simulated-Driving-Agent/
├── README.md
├── requirements.txt
├── conftest.py                    # pytest import bootstrap
├── pytest.ini                     # test configuration
├── src/
│   ├── __init__.py
│   ├── config.py                  # all tunable parameters & paths
│   ├── track.py                   # procedural closed-loop track + distance grid
│   ├── vehicle.py                 # arcade kinematic bicycle model
│   ├── sensors.py                 # 7 ray sensors (single + batched)
│   ├── neural_network.py          # NumPy MLP with genome <-> parameters
│   ├── genetic_algorithm.py       # selection / crossover / mutation / elites
│   ├── simulation.py              # episode rollout, progress, fitness
│   ├── train.py                   # GA training entry point (python -m src.train)
│   └── visualize.py               # figures + best-agent replay
├── tests/
│   ├── __init__.py
│   ├── test_track.py
│   ├── test_vehicle.py
│   ├── test_sensors.py
│   ├── test_neural_network.py
│   └── test_genetic_algorithm.py
└── artifacts/
    ├── best_agent.json            # best genome + architecture + config
    ├── training_history.csv       # per-generation metrics (50 rows)
    ├── best_agent_replay.png      # replay trajectory of the saved agent
    ├── training_log.txt           # console log of the final run
    └── plots/
        ├── fitness_curve.png
        ├── best_agent_trajectory.png
        ├── sensor_fan.png
        └── track_layout.png
```

## Conclusion

Across 50 generations the evolved population improved from a best fitness of
108.6 (0.13 lap) to 9079.6 (8.04 laps without crashing), and the *average*
agent went from nearly motionless (fitness 6.7) to completing ~8 laps
(fitness 5666) — clear evidence that the Genetic Algorithm genuinely evolved
driving behavior rather than simulating random motion. The learned controller
uses only 90 floating-point weights and receives no information beyond 7
boundary-distance rays and its own speed, yet it steers through the track's
curves, manages its speed, and keeps all 900 replay positions inside the
track boundaries.

What this demonstrates: neuroevolution can discover competent sensorimotor
policies from a simple, well-shaped fitness signal — forward progress — with
no gradient descent, no labels and no hand-coded driving rules. What it does
*not* claim: this is a toy 2D kinematic simulation on one deterministic track;
the "vehicle" ignores inertia, slip and dynamics beyond an arcade bicycle
model, and the evolved behavior is specific to this track geometry and seed.

## References

* Mitchell, M. (1998). *An Introduction to Genetic Algorithms*. MIT Press.
* Such, F. P. et al. (2017). *Deep Neuroevolution: Genetic Algorithms Are a
  Competitive Alternative for Training Deep Neural Networks*. arXiv:1712.06567.
* sensor/vehicle conventions follow common practice in simple 2D driving
  simulators (ray casting + kinematic bicycle).
