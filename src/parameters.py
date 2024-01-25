from dataclasses import dataclass
from typing import ClassVar
import numpy as np

@dataclass
class Parameters:
    n_points: int = 25000
    initial_point: ClassVar[np.ndarray] = np.array([0.0, 0.0])
    fitness_radius: float = 0.005
    min_individual_degree: int = 4
    max_individual_degree: int = 4
    initial_population_size: int = 10
    recombination_population_size: int = 4
    max_self_creation_population_size: int = 10
    reassortment_population_size: int = 20
    elite_fitness_threshold: float = 0.1
    p_arithmetic_crossover: float = 0.5
    p_vector_crossover: float = 1 - p_arithmetic_crossover
    a: float = 0.5
    max_singel_coefficient: float = 1
    min_singel_coefficient: float = -1
    guassian_mutation_radius: float = 0.01
    mutation_probability: float = 0.12
