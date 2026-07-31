from dataclasses import dataclass
import numpy as np

@dataclass
class Group:
    id: int
    matrix0: np.ndarray
    matrix1: np.ndarray
    valuations: np.ndarray