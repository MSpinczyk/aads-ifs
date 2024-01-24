from dataclasses import dataclass
from typing import ClassVar
import numpy as np

@dataclass
class Parameters:
    n_points: int = 5000
    initial_point: ClassVar[np.ndarray] = np.array([0.0, 0.0])
    fitness_radius: float = 0.001
    min_singel_dim: int = 4
    max_singel_dim: int = 4
    population_size: int = 6
    elite_fitness_threshold: float = 0.5
