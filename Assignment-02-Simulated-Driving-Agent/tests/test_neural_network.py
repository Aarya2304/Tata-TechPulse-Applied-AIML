"""Tests for the NumPy neural-network controller."""

from __future__ import annotations

import numpy as np
import pytest

from src import config
from src.neural_network import NeuralNetwork


class TestArchitecture:
    def test_default_sizes(self):
        net = NeuralNetwork()
        assert net.input_size == config.NN_INPUT_SIZE == 8
        assert net.hidden_size == config.NN_HIDDEN_SIZE == 8
        assert net.output_size == config.NN_OUTPUT_SIZE == 2

    def test_genome_size_formula(self):
        net = NeuralNetwork(input_size=8, hidden_size=8, output_size=2)
        expected = 8 * 8 + 8 + 8 * 2 + 2
        assert net.genome_size == expected == 90


class TestForwardPass:
    def test_output_shape_and_range(self):
        net = NeuralNetwork()
        rng = np.random.default_rng(0)
        net.set_genome(rng.normal(0, 0.5, net.genome_size))
        out = net.forward(np.zeros(8))
        assert out.shape == (2,)
        assert -1.0 <= out[0] <= 1.0      # steering (tanh)
        assert -1.0 <= out[1] <= 1.0      # throttle (scaled sigmoid)

    def test_deterministic_forward(self):
        net = NeuralNetwork()
        rng = np.random.default_rng(7)
        net.set_genome(rng.normal(0, 0.5, net.genome_size))
        obs = rng.random(8)
        o1 = net.forward(obs)
        o2 = net.forward(obs)
        assert np.array_equal(o1, o2)

    def test_wrong_input_size_raises(self):
        net = NeuralNetwork()
        with pytest.raises(ValueError):
            net.forward(np.zeros(5))


class TestGenome:
    def test_genome_extraction_shape(self):
        net = NeuralNetwork()
        g = net.get_genome()
        assert g.shape == (net.genome_size,)
        assert g.dtype == float

    def test_genome_roundtrip(self):
        net = NeuralNetwork()
        rng = np.random.default_rng(3)
        genome = rng.normal(0, 0.5, net.genome_size)
        net.set_genome(genome)
        assert np.allclose(net.get_genome(), genome)

    def test_reconstruction_reproduces_outputs(self):
        net_a = NeuralNetwork()
        rng = np.random.default_rng(11)
        genome = rng.normal(0, 0.5, net_a.genome_size)
        net_a.set_genome(genome)
        net_b = NeuralNetwork().set_genome(genome)
        obs = rng.random(8)
        assert np.allclose(net_a.forward(obs), net_b.forward(obs))

    def test_set_genome_wrong_size_raises(self):
        net = NeuralNetwork()
        with pytest.raises(ValueError):
            net.set_genome(np.zeros(10))

    def test_random_genome_initialization(self):
        rng = np.random.default_rng(5)
        g1 = NeuralNetwork.random_genome(rng)
        g2 = NeuralNetwork.random_genome(rng)
        assert g1.shape == (90,)
        assert not np.allclose(g1, g2)  # random draws differ
