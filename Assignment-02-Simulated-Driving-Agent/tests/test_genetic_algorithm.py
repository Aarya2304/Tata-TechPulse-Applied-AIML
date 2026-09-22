"""Tests for the Genetic Algorithm operators."""

from __future__ import annotations

import numpy as np
import pytest

from src import config
from src.genetic_algorithm import GeneticAlgorithm
from src.neural_network import NeuralNetwork


@pytest.fixture()
def ga() -> GeneticAlgorithm:
    return GeneticAlgorithm(genome_size=90, population_size=12, seed=99)


class TestPopulation:
    def test_population_size_correct(self, ga):
        assert len(ga.get_population()) == ga.population_size

    def test_genome_shapes(self, ga):
        for genome in ga.get_population():
            assert genome.shape == (90,)

    def test_reproducible_initial_population(self):
        a = GeneticAlgorithm(genome_size=90, population_size=5, seed=7)
        b = GeneticAlgorithm(genome_size=90, population_size=5, seed=7)
        for ga_, gb in zip(a.get_population(), b.get_population()):
            assert np.allclose(ga_, gb)


class TestSelection:
    def test_tournament_returns_member(self, ga):
        fitnesses = list(range(ga.population_size, 0, -1))
        child = ga.tournament_select(fitnesses)
        assert child.shape == (90,)
        assert any(np.allclose(child, g) for g in ga.get_population())

    def test_tournament_prefers_high_fitness(self, ga):
        """Champion must win when every member is drawn (k = N), and must
        win far above the uniform 1/12 baseline with the default k."""
        fitnesses = [0.0] * ga.population_size
        champion = 3
        fitnesses[champion] = 1000.0
        champion_genome = ga.get_population()[champion]

        # Deterministic case: force a draw that always includes the champion.
        class _FixedDraw:
            def integers(self, *a, **k):
                return np.array([5, 3, 9, 1])

        ga.tournament_size = 4
        ga.rng = _FixedDraw()
        for _ in range(10):
            assert np.allclose(ga.tournament_select(fitnesses),
                               champion_genome)

        # Statistical case: default k=4 with replacement -> P(champion drawn)
        # = 1 - (11/12)^4 = 0.294, versus 1/12 = 0.083 for uniform picking.
        ga.rng = np.random.default_rng(123)
        ga.tournament_size = 4
        wins = sum(
            np.allclose(ga.tournament_select(fitnesses), champion_genome)
            for _ in range(500)
        )
        assert wins / 500 > 0.15


class TestCrossover:
    def test_preserves_genome_dimensions(self, ga):
        pa = ga.get_population()[0]
        pb = ga.get_population()[1]
        c1, c2 = ga.crossover(pa, pb)
        assert c1.shape == (90,) and c2.shape == (90,)

    def test_uniform_mix_of_parents(self, ga):
        pa = np.zeros(90)
        pb = np.ones(90)
        ga.crossover_rate = 1.0
        for _ in range(30):
            c1, _ = ga.crossover(pa, pb)
            assert set(np.unique(c1)).issubset({0.0, 1.0})

    def test_crossover_rate_zero_copies_parents(self, ga):
        pa = np.zeros(90)
        pb = np.ones(90)
        ga.crossover_rate = 0.0
        c1, c2 = ga.crossover(pa, pb)
        assert np.allclose(c1, pa) and np.allclose(c2, pb)


class TestMutation:
    def test_mutation_changes_some_genes(self, ga):
        genome = np.zeros(90)
        ga.mutation_rate = 0.5
        ga.mutation_strength = 0.5
        mutated = ga.mutate(genome)
        changed = np.count_nonzero(mutated)
        assert 0 < changed < 90

    def test_mutation_rate_zero_is_identity(self, ga):
        genome = np.arange(90, dtype=float)
        ga.mutation_rate = 0.0
        assert np.allclose(ga.mutate(genome), genome)

    def test_mutation_preserves_shape(self, ga):
        out = ga.mutate(np.zeros((90,)))
        assert out.shape == (90,)


class TestElitism:
    def test_elites_are_best_agents(self):
        ga = GeneticAlgorithm(genome_size=90, population_size=8,
                              elite_count=2, seed=5)
        old = [g.copy() for g in ga.get_population()]
        fitnesses = [float(i) for i in range(8)]  # index 7 is best
        ga.next_generation(fitnesses)
        new_pop = ga.get_population()
        assert any(np.allclose(old[7], g) for g in new_pop[:2])
        assert any(np.allclose(old[6], g) for g in new_pop[:2])

    def test_mismatched_genome_size_rejected(self):
        with pytest.raises(ValueError):
            GeneticAlgorithm(genome_size=10, population_size=4, seed=0)


class TestNextGeneration:
    def test_population_size_stable(self, ga):
        fitnesses = list(np.random.default_rng(0).random(ga.population_size))
        ga.next_generation(fitnesses)
        assert len(ga.get_population()) == ga.population_size

    def test_returns_stats(self, ga):
        fitnesses = list(np.random.default_rng(1).random(ga.population_size))
        stats = ga.next_generation(fitnesses)
        assert {"best_fitness", "average_fitness", "best_index"} <= set(stats)

    def test_evolution_improves_best_over_generations(self):
        """On a trivial 90-d landscape the GA must approach the optimum."""
        target = np.random.default_rng(42).normal(0, 1, 90)
        ga = GeneticAlgorithm(genome_size=90, population_size=30,
                              elite_count=2, seed=1)
        best_history = []
        for _ in range(15):
            fitnesses = [-float(np.mean((g - target) ** 2))
                         for g in ga.get_population()]
            best_history.append(max(fitnesses))
            ga.next_generation(fitnesses)
        assert best_history[-1] > best_history[0]
