"""Genetic Algorithm evolving neural-network driving controllers.

Per generation:
1. evaluate every genome on the deterministic track
2. rank by fitness
3. copy the top ``GA_ELITE_COUNT`` genomes unchanged (elitism)
4. fill the rest by tournament selection -> uniform crossover -> gaussian
   per-gene mutation
"""

from __future__ import annotations

import numpy as np

from src import config
from src.neural_network import NeuralNetwork


class GeneticAlgorithm:
    """Steady-generational GA over fixed-length float genomes.

    ``genome_size`` must match a ``NeuralNetwork`` parameter count for the
    requested architecture (default: the configured 8-8-2 controller).
    """

    def __init__(self,
                 genome_size: int,
                 population_size: int = config.GA_POPULATION_SIZE,
                 elite_count: int = config.GA_ELITE_COUNT,
                 tournament_size: int = config.GA_TOURNAMENT_SIZE,
                 crossover_rate: float = config.GA_CROSSOVER_RATE,
                 mutation_rate: float = config.GA_MUTATION_RATE,
                 mutation_strength: float = config.GA_MUTATION_STRENGTH,
                 seed: int = config.GA_SEED,
                 input_size: int = config.NN_INPUT_SIZE,
                 hidden_size: int = config.NN_HIDDEN_SIZE,
                 output_size: int = config.NN_OUTPUT_SIZE) -> None:
        self.genome_size = genome_size
        self.population_size = population_size
        self.elite_count = min(elite_count, population_size)
        self.tournament_size = max(2, tournament_size)
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.mutation_strength = mutation_strength
        self.rng = np.random.default_rng(seed)

        expected = NeuralNetwork(input_size, hidden_size, output_size).genome_size
        if genome_size != expected:
            raise ValueError(
                f"genome_size {genome_size} does not match network "
                f"architecture ({input_size}-{hidden_size}-{output_size} "
                f"= {expected} parameters)")

        self.population = [
            NeuralNetwork.random_genome(self.rng, input_size, hidden_size,
                                        output_size)
            for _ in range(population_size)
        ]

    # ------------------------------------------------------------------
    # GA operators
    # ------------------------------------------------------------------
    def tournament_select(self, fitnesses: list) -> np.ndarray:
        """Pick the best genome among ``tournament_size`` random contestants."""
        idxs = self.rng.integers(0, self.population_size,
                                 size=self.tournament_size)
        best = max(idxs, key=lambda i: fitnesses[i])
        return self.population[best]

    def crossover(self, parent_a: np.ndarray,
                  parent_b: np.ndarray) -> tuple:
        """Uniform crossover with probability ``crossover_rate``."""
        if self.rng.random() > self.crossover_rate:
            return parent_a.copy(), parent_b.copy()
        mask = self.rng.random(self.genome_size) < 0.5
        child1 = np.where(mask, parent_a, parent_b)
        child2 = np.where(mask, parent_b, parent_a)
        return child1, child2

    def mutate(self, genome: np.ndarray) -> np.ndarray:
        """Gaussian mutation applied per-gene with probability ``rate``."""
        mutated = genome.copy()
        mask = self.rng.random(self.genome_size) < self.mutation_rate
        noise = self.rng.normal(0.0, self.mutation_strength, self.genome_size)
        mutated[mask] += noise[mask]
        return mutated

    # ------------------------------------------------------------------
    # Generation step
    # ------------------------------------------------------------------
    def next_generation(self, fitnesses: list) -> dict:
        """Produce the next population.  Returns summary statistics."""
        fitnesses = np.asarray(fitnesses, dtype=float)
        order = np.argsort(fitnesses)[::-1]
        ranked = [self.population[i] for i in order]

        best_idx = int(order[0])
        stats = {
            "best_fitness": float(fitnesses[order[0]]),
            "best_index": best_idx,
            "average_fitness": float(fitnesses.mean()),
            "worst_fitness": float(fitnesses[order[-1]]),
        }

        next_pop = [ranked[i].copy() for i in range(self.elite_count)]

        while len(next_pop) < self.population_size:
            parent_a = self.tournament_select(fitnesses)
            parent_b = self.tournament_select(fitnesses)
            child_a, child_b = self.crossover(parent_a, parent_b)
            next_pop.append(self.mutate(child_a))
            if len(next_pop) < self.population_size:
                next_pop.append(self.mutate(child_b))

        self.population = next_pop[:self.population_size]
        return stats

    def set_population(self, population: list) -> None:
        self.population = [np.asarray(g, dtype=float) for g in population]

    def get_population(self) -> list:
        return self.population
