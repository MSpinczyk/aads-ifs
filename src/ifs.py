import numpy as np
import matplotlib.pyplot as plt

from src.parameters import Parameters
from matplotlib.colors import LogNorm


class IteratedFunctionSystem:
    def __init__(self):
        self.functions: list[np.ndarray] = []
        self.fitness = 0

    def add_function(self, function):
        """
        Add a Barnsley function to the IFS.

        :param function: 6 element array representing the Barnsley function.
        """

        if len(function) != 6:
            raise ValueError('Barnsley function should have 6 elements')

        self.functions.append(function)

    def apply_function(self, point):
        """
        Apply a random function to a point.

        :param point: 1x2 array representing the point (x, y).
        :return: Transformed point.
        """
        function_idx = np.random.choice(len(self.functions))
        function = self.functions[function_idx]

        # aix1 +bix2 +ei, cix1 +dix2 +fi
        return np.array([
            function[0]*point[0] + function[1]*point[1] + function[4],
            function[2]*point[0] + function[3]*point[1] + function[5]
        ])

    def generate_points(self, num_points, initial_point=None):
        """
        Generate a set of points using the Iterated Function System.

        :param num_points: Number of points to generate.
        :param initial_point: Initial point (default is [0, 0]).
        :return: List of generated points.
        """
        if initial_point is None:
            initial_point = np.random.random(size=2)

        points = np.empty((num_points, 2))
        points[0] = initial_point

        for i in range(1, num_points):
            points[i] = self.apply_function(points[i - 1])

        return points

    def plot_fractal(self, num_points=Parameters.n_points, initial_point=Parameters.initial_point):
        """
        Generate and plot a fractal using the Iterated Function System.

        :param num_points: Number of points to generate.
        :param initial_point: Initial point (default is random).
        """
        points = self.generate_points(num_points, initial_point)
        plt.scatter(points[:, 0], points[:, 1], s=1, c='black', alpha=0.5)
        plt.title('Iterated Function System Fractal')
        # plt.gca().set_aspect('equal', adjustable='box')
        plt.show()

    def plot_heatmap(self, num_points=Parameters.n_points, initial_point=Parameters.initial_point, bins=100):
        points = self.generate_points(num_points, initial_point)
        heatmap, xedges, yedges = np.histogram2d(points[:, 0], points[:, 1], bins=bins)
        extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]

        plt.clf()
        plt.imshow(heatmap.T, extent=extent, origin='lower')
        plt.show()

    @classmethod
    def create_fern(cls):
        ifs = cls()
        ifs.add_function(np.array([0, 0, 0, 0.16, 0, 0.0]))
        ifs.add_function(np.array([0.85, 0.04, -0.04, 0.85, 0.0, 1.6]))
        ifs.add_function(np.array([0.2, -0.26, 0.23, 0.22, 0.0, 1.6]))
        ifs.add_function(np.array([-0.15, 0.28, 0.26, 0.24, 0.0, 0.44]))
        return ifs
