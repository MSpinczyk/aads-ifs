from dataclasses import dataclass
from typing import ClassVar
import numpy as np

@dataclass
class Parameters:
    n_points: int = 5000
    initial_point: ClassVar[np.ndarray] = np.array([0.0, 0.0])
