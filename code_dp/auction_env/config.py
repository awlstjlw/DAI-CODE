import numpy as np
from dataclasses import dataclass, field
from typing import List

@dataclass
class AuctionConfig:
    EXPERIMENT_ROUNDS = 500
    SEED: np.uint32=np.uint32(314159)
    SEED_TRAIN: np.uint32=np.uint32(12345678)
    SEED_OF_VALUE: np.uint32=np.uint32(0)
    SEED_OF_MATRIX: np.uint32=np.uint32(0)
    SEED_OF_GROUPING: np.uint32=np.uint32(0)
    T: int = 5
    DELTA: float = 0.6
    EPSILON: float = 1
    EPOCHS: int = 100
    NUM_SETTINGS: int = 1152
    TRAIN_SPLIT: int = 1024
    EPSILON_VALUES: List[int] = field(default_factory=lambda: [1])
    
    CALC_GROUP_SIZES: List[int] = field(default_factory=list)
    NUM_AGENTS: int = 60
    
    VALUATION_RANGES: List[tuple[int, int]] = field(default_factory=list)
    MAX_VALUATION: int = 400
    
    def __post_init__(self):
        try:
            with open("calc_group_sizes.txt", 'r', encoding='utf-8') as f:
                sizes = f.read().strip().split(',')
                self.CALC_GROUP_SIZES = [int(sizes[0]), int(sizes[1])]
                print(f"Loaded CALC_GROUP_SIZES from file: {self.CALC_GROUP_SIZES}")
        except FileNotFoundError:
            if self.CALC_GROUP_SIZES is None:
                self.CALC_GROUP_SIZES = [20, 20]
            print(f"Using default CALC_GROUP_SIZES: {self.CALC_GROUP_SIZES}")
        
        if not self.VALUATION_RANGES:
            self.VALUATION_RANGES = [(360, 401), (240, 281)]
    
    def reset_data_seed(self, seed_of_value, seed_of_matrix, seed_of_grouping): 
        self.SEED_OF_VALUE = seed_of_value
        self.SEED_OF_MATRIX = seed_of_matrix
        self.SEED_OF_GROUPING = seed_of_grouping
    
    def reset_fairness_epsilon(self, epsilon):
        self.EPSILON = epsilon
    
    def generate_random_value(self, seed_of_value: np.uint32=np.uint32(0)):
        rng = np.random.default_rng(seed_of_value)
        v0 = rng.choice(range(*self.VALUATION_RANGES[0]), 2, replace=False)
        v1 = rng.choice(range(*self.VALUATION_RANGES[1]), 2, replace=False)
        return v0, v1

    def generate_random_matrix(self, count=1, seed_of_matrix: np.uint32=np.uint32(0)):
        rng = np.random.default_rng(seed_of_matrix)
        P_all = rng.random((count, 2, 2))
        P_all /= P_all.sum(axis=2, keepdims=True)
        return P_all if count > 1 else P_all[0]