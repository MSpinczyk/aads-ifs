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
    initial_population_size: int = 6
    recombination_population_size: int = 4
    max_self_creation_population_size: int = 20
    reassortment_population_size: int = 4
    elite_fitness_threshold: float = 0.01
    p_arithmetic_crossover: float = 0.5
    p_vector_crossover: float = 1 - p_arithmetic_crossover
    a: float = 0.5
