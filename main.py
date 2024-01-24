from src.ifs import IteratedFunctionSystem
from src.parameters import Parameters
from src.runner import Runner
import numpy as np


# Example usage:
target = IteratedFunctionSystem.create_fern()
a_ifs = IteratedFunctionSystem()
a_ifs.add_operator(np.array([0, 0, 0, 0.12, 0, 0]))
a_ifs.add_operator(np.array([0.76, 0.045, -0.04, 0.85, 0.0, 1.6]))
a_ifs.add_operator(np.array([0.2, -0.26, 0.21, 0.22, 0.0, 1.6]))
a_ifs.add_operator(np.array([-0.20, 0.24, 0.22, 0.24, 0.0, 0.44]))
a_ifs.plot_fractal(num_points=Parameters.n_points, initial_point=Parameters.initial_point)

runner = Runner(target)
runner.generate_first_population()
# runner.population[0].plot_fractal(num_points=Parameters.n_points, initial_point=Parameters.initial_point)
runner.step()


# Generate and plot the fractal:
# target.plot_fractal(num_points=Parameters.n_points, initial_point=Parameters.initial_point)
# a_ifs.plot_fractal(num_points=Parameters.n_points, initial_point=Parameters.initial_point)
