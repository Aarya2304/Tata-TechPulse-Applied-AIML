"""Tiny feed-forward neural network controller implemented in pure NumPy.

Architecture (configurable in ``config.py``):

    8 inputs -> 8 hidden neurons (tanh) -> 2 outputs

Outputs
-------
* ``steering``  in [-1, 1]  (tanh)   -> scaled to +/- VEHICLE_MAX_STEER
* ``throttle``  in [ 0, 1]  (sigmoid, shifted) -> mapped to [-1, 1] command

The GA evolves the flattened weight vector ("genome").  The network can be
serialised to and reconstructed from a genome, which is all the GA needs.
"""

from __future__ import annotations

import math

import numpy as np

from src import config


class NeuralNetwork:
    """2-layer feed-forward network with genome <-> parameters conversion."""

    def __init__(self, input_size: int = config.NN_INPUT_SIZE,
                 hidden_size: int = config.NN_HIDDEN_SIZE,
                 output_size: int = config.NN_OUTPUT_SIZE) -> None:
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.W1 = np.zeros((input_size, hidden_size))
        self.b1 = np.zeros(hidden_size)
        self.W2 = np.zeros((hidden_size, output_size))
        self.b2 = np.zeros(output_size)

    # ------------------------------------------------------------------
    # Genome handling
    # ------------------------------------------------------------------
    @property
    def genome_size(self) -> int:
        return (self.input_size * self.hidden_size + self.hidden_size
                + self.hidden_size * self.output_size + self.output_size)

    def get_genome(self) -> np.ndarray:
        """Flatten all parameters into a 1-D genome vector."""
        return np.concatenate([self.W1.ravel(), self.b1,
                               self.W2.ravel(), self.b2])

    def set_genome(self, genome: np.ndarray) -> "NeuralNetwork":
        """Load parameters from a flat genome (returns self for chaining)."""
        genome = np.asarray(genome, dtype=float).ravel()
        if genome.size != self.genome_size:
            raise ValueError(
                f"genome size {genome.size} != expected {self.genome_size}")
        i = 0
        n = self.input_size * self.hidden_size
        self.W1 = genome[i:i + n].reshape(self.input_size, self.hidden_size)
        i += n
        self.b1 = genome[i:i + self.hidden_size]
        i += self.hidden_size
        n = self.hidden_size * self.output_size
        self.W2 = genome[i:i + n].reshape(self.hidden_size, self.output_size)
        i += n
        self.b2 = genome[i:i + self.output_size]
        return self

    @staticmethod
    def random_genome(rng: np.random.Generator,
                      input_size: int = config.NN_INPUT_SIZE,
                      hidden_size: int = config.NN_HIDDEN_SIZE,
                      output_size: int = config.NN_OUTPUT_SIZE) -> np.ndarray:
        """Xavier-ish initialisation of a fresh genome."""
        w1_scale = math.sqrt(1.0 / input_size)
        w2_scale = math.sqrt(1.0 / hidden_size)
        n1 = input_size * hidden_size
        n2 = hidden_size * output_size
        genome = np.concatenate([
            rng.normal(0.0, w1_scale, n1),
            np.zeros(hidden_size),
            rng.normal(0.0, w2_scale, n2),
            np.zeros(output_size),
        ])
        return genome

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------
    def forward(self, observation: np.ndarray) -> np.ndarray:
        """Map an observation to [steering, throttle] outputs.

        steering: tanh  -> [-1, 1]
        throttle: 2*sigmoid - 1 -> [-1, 1]  (negative = brake)
        """
        obs = np.asarray(observation, dtype=float).ravel()
        if obs.size != self.input_size:
            raise ValueError(f"expected {self.input_size} inputs, "
                             f"got {obs.size}")
        hidden = np.tanh(obs @ self.W1 + self.b1)
        raw = hidden @ self.W2 + self.b2
        steering = math.tanh(float(raw[0]))
        throttle = 2.0 / (1.0 + math.exp(-float(raw[1]))) - 1.0
        return np.array([steering, throttle])

    def predict(self, observation: np.ndarray) -> np.ndarray:
        """Alias used by the simulation loop."""
        return self.forward(observation)
