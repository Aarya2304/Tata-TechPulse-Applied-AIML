"""Evolve driving agents with a Genetic Algorithm (entry point).

Usage
-----
    python -m src.train

Pipeline: deterministic track -> GA loop (evaluate all genomes, elitism,
tournament selection, uniform crossover, gaussian mutation) -> save
``training_history.csv`` and ``best_agent.json`` -> render plots.
"""

from __future__ import annotations

import csv
import json
import time

import numpy as np

from src import config
from src.genetic_algorithm import GeneticAlgorithm
from src.neural_network import NeuralNetwork
from src.simulation import Simulator
from src.track import Track
from src.visualize import plot_fitness_curve, plot_best_trajectory


def evaluate_population(ga: GeneticAlgorithm, simulator: Simulator) -> tuple:
    """Evaluate every genome in the GA population (vectorised)."""
    genomes = np.array(ga.get_population(), dtype=float)
    return simulator.run_population(genomes)


def run_training(generations: int = config.GA_GENERATIONS,
                 population_size: int = config.GA_POPULATION_SIZE,
                 seed: int = config.GA_SEED,
                 verbose: bool = True) -> dict:
    """Run the full evolution loop; returns a summary + history rows."""
    start_time = time.time()

    track = Track()
    simulator = Simulator(track)
    genome_size = NeuralNetwork().genome_size
    ga = GeneticAlgorithm(
        genome_size=genome_size,
        population_size=population_size,
        seed=seed,
    )

    history = []
    best_ever = None       # (fitness, genome, progress, laps)
    best_progress = 0.0
    best_laps = 0.0

    for gen in range(generations):
        fitnesses, progresses, laps, crashed = \
            evaluate_population(ga, simulator)

        gen_best_progress = float(progresses.max())
        gen_best_laps = float(laps.max())
        best_progress = max(best_progress, gen_best_progress)
        best_laps = max(best_laps, gen_best_laps)

        # Capture this generation's best genome BEFORE next_generation()
        # rebuilds the population (elitism then carries it forward).
        gen_best_idx = int(np.argmax(fitnesses))
        gen_best_genome = np.array(ga.get_population()[gen_best_idx],
                                   dtype=float, copy=True)

        stats = ga.next_generation(list(fitnesses))

        # Keep the best genome ever seen (deterministic track => stable
        # fitness across generations, so elites re-evaluate identically).
        if best_ever is None or stats["best_fitness"] > best_ever[0]:
            best_ever = (stats["best_fitness"], gen_best_genome,
                         gen_best_progress, gen_best_laps)

        history.append({
            "generation": gen,
            "best_fitness": round(stats["best_fitness"], 4),
            "average_fitness": round(stats["average_fitness"], 4),
            "best_progress": round(gen_best_progress, 2),
            "best_laps": round(gen_best_laps, 3),
        })

        if verbose:
            print(f"gen {gen:3d} | best {stats['best_fitness']:8.2f} | "
                  f"avg {stats['average_fitness']:8.2f} | "
                  f"progress {gen_best_progress:7.1f} m | "
                  f"laps {gen_best_laps:5.2f}")

    elapsed = time.time() - start_time
    summary = {
        "generations": generations,
        "population_size": population_size,
        "seed": seed,
        "elapsed_seconds": round(elapsed, 1),
        "best_fitness": best_ever[0],
        "best_genome": best_ever[1].tolist(),
        "best_progress": best_progress,
        "best_laps": best_laps,
        "history": history,
        "track_length": track.total_length,
    }
    return summary


def save_artifacts(summary: dict) -> None:
    """Write training_history.csv and best_agent.json."""
    config.ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    config.PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    # --- training history -------------------------------------------------
    with config.TRAINING_HISTORY_PATH.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=[
            "generation", "best_fitness", "average_fitness",
            "best_progress", "best_laps"])
        writer.writeheader()
        writer.writerows(summary["history"])

    # --- best agent -------------------------------------------------------
    net = NeuralNetwork()
    payload = {
        "architecture": {
            "input_size": net.input_size,
            "hidden_size": net.hidden_size,
            "output_size": net.output_size,
            "hidden_activation": "tanh",
            "output_activation": ["tanh", "sigmoid_scaled"],
        },
        "genome": summary["best_genome"],
        "genome_size": len(summary["best_genome"]),
        "fitness": summary["best_fitness"],
        "seed": summary["seed"],
        "track": {
            "track_seed": config.TRACK_SEED,
            "num_points": config.TRACK_NUM_POINTS,
            "width": config.TRACK_WIDTH,
        },
        "episode": {
            "max_steps": config.EPISODE_MAX_STEPS,
            "time_step": config.EPISODE_TIME_STEP,
        },
    }
    config.BEST_AGENT_PATH.write_text(json.dumps(payload, indent=2))


def main() -> None:
    print("=" * 64)
    print("Simulated Driving Agent - Genetic Algorithm training")
    print("=" * 64)

    summary = run_training()

    save_artifacts(summary)

    print(f"\nTraining finished in {summary['elapsed_seconds']} s")
    print(f"Best fitness : {summary['best_fitness']:.2f}")
    print(f"Best progress: {summary['best_progress']:.1f} m "
          f"({summary['best_laps']:.2f} laps of "
          f"{summary['track_length']:.0f} m)")

    # ---- plots -----------------------------------------------------------
    config.PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    plot_fitness_curve(config.TRAINING_HISTORY_PATH,
                       config.PLOTS_DIR / "fitness_curve.png")

    # Replay the best genome once with trajectory recording.
    track = Track()
    simulator = Simulator(track)
    best_net = NeuralNetwork().set_genome(summary["best_genome"])
    replay = simulator.run_episode(best_net, record=True)
    plot_best_trajectory(
        track, replay.trajectory,
        out_path=config.PLOTS_DIR / "best_agent_trajectory.png",
        title=(f"Best agent - fitness {replay.fitness:.1f}, "
               f"progress {replay.progress:.0f} m "
               f"({replay.laps:.2f} laps)"),
    )
    print(f"Plots saved to {config.PLOTS_DIR}")


if __name__ == "__main__":
    main()
