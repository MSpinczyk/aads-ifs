from collections import namedtuple
from ifs import IteratedFunctionSystem
from parameters import Parameters
import numpy as np


class Runner:
    def __init__(self, target_ifs: IteratedFunctionSystem):
        self.target_ifs = target_ifs
        self.target_ifs_points = target_ifs.generate_points(
            Parameters.n_points, Parameters.initial_point
        )
        self._cached_ifs_bounds = None

    def calculate_fitness(self, ifs: IteratedFunctionSystem):
        ifs_points = ifs.generate_points(Parameters.n_points, Parameters.initial_point)
        n_a = len(ifs_points)
        n_i = len(self.target_ifs_points)
        n_nd = self.calculate_n_nd(ifs_points)
        n_nn = self.calculate_n_nn(ifs_points)
        # relative coverage of the attractor
        r_c = n_nd/n_i
        # attractor points outside the image
        r_o = n_nn/n_a

        # fitness to maximise
        fitness = (1 - r_c) + (1 - r_o)
        raise fitness

    def calculate_n_nd(self, ifs_points: np.ndarray):
        """not drawn points - present in the image but not in the attractor"""
        raise NotImplementedError

    def calculate_n_nn(self, ifs_points: np.ndarray):
        """points not needed - present in the attractor but not in the image"""
        raise NotImplementedError

    @property
    def target_ifs_bounds(self):
        if self._cached_ifs_bounds is None:
            Bounds = namedtuple('PointBounds', ['lower_x', 'upper_x', 'lower_y', 'upper_y'])

            lower_x = min(self.target_ifs_points[:,0])
            upper_x = max(self.target_ifs_points[:,0])
            lower_y = min(self.target_ifs_points[:,1])
            upper_y = max(self.target_ifs_points[:,1])

            self._cached_ifs_bounds = Bounds(lower_x, upper_x, lower_y, upper_y)

        return self._cached_ifs_bounds
