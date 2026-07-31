import numpy as np
from dataclasses import dataclass, field
from typing import List


@dataclass
class AuctionConfigM3:
    """Three-group auction configuration aligned with experiment 3."""

    EXPERIMENT_ROUNDS = 500
    SEED: np.uint32 = np.uint32(314159)
    SEED_TRAIN: np.uint32 = np.uint32(12345678)
    SEED_OF_VALUE: np.uint32 = np.uint32(0)
    SEED_OF_MATRIX: np.uint32 = np.uint32(0)
    SEED_OF_GROUPING: np.uint32 = np.uint32(0)

    T: int = 5
    DELTA: float = 0.6
    EPSILON: float = 5

    EPOCHS: int = 100
    NUM_SETTINGS: int = 1152
    TRAIN_SPLIT: int = 1024
    EPSILON_VALUES: List[int] = field(default_factory=lambda: [5])

    CALC_GROUP_SIZES: List[int] = field(default_factory=lambda: [20, 20, 20])
    NUM_AGENTS: int = 60

    VALUATION_RANGES: List[tuple[int, int]] = field(
        default_factory=lambda: [(360, 401), (300, 341), (240, 281)]
    )
    MAX_VALUATION: int = 400

    def reset_data_seed(self, seed_of_value, seed_of_matrix, seed_of_grouping):
        self.SEED_OF_VALUE = seed_of_value
        self.SEED_OF_MATRIX = seed_of_matrix
        self.SEED_OF_GROUPING = seed_of_grouping

    def reset_fairness_epsilon(self, epsilon):
        self.EPSILON = epsilon

    def generate_random_value(self, seed_of_value: np.uint32 = np.uint32(0)):
        rng = np.random.default_rng(seed_of_value)
        v0 = rng.choice(range(*self.VALUATION_RANGES[0]), 2, replace=False)
        v1 = rng.choice(range(*self.VALUATION_RANGES[1]), 2, replace=False)
        v2 = rng.choice(range(*self.VALUATION_RANGES[2]), 2, replace=False)
        return v0, v1, v2

    def generate_random_matrix(
        self, count=1, seed_of_matrix: np.uint32 = np.uint32(0)
    ):
        rng = np.random.default_rng(seed_of_matrix)
        matrices = rng.random((count, 2, 2))
        matrices /= matrices.sum(axis=2, keepdims=True)
        return matrices if count > 1 else matrices[0]
