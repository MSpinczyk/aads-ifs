from src.ifs import IteratedFunctionSystem
from src.parameters import Parameters
from src.runner import Runner
import numpy as np
import scipy


# Example usage:
target = IteratedFunctionSystem.create_fern()
# a_ifs = IteratedFunctionSystem()
# a_ifs.add_function(np.array([0, 0, 0, 0.12, 0, 0]))
# a_ifs.add_function(np.array([0.76, 0.045, -0.04, 0.85, 0.0, 1.6]))
# a_ifs.add_function(np.array([0.2, -0.26, 0.21, 0.22, 0.0, 1.6]))
# a_ifs.add_function(np.array([-0.20, 0.24, 0.22, 0.24, 0.0, 0.44]))
# a_ifs.plot_fractal(num_points=Parameters.n_points, initial_point=Parameters.initial_point)
# data = target.generate_points(1_000_000)
runner = Runner(target)
runner.generate_first_population()

for i in range(4000 + 1):
    runner.step()
    print(f'----------Generation {i}----------')
    print(f'Mean Fitness:{runner.mean_fitness}')
    print(f'Best Fitness:{runner.best.fitness}')
    if i % 100 == 0:
        data = runner.best.generate_points(250_000)
        data.tofile(f'data_{i}.bin')
        runner.plot_fitness_grid(i)
# 
# breakpoint()
# for ifs in runner.population:
#     runner.calculate_fitness(ifs)
# 
# runner.mean_fitness = np.mean([i.fitness for i in runner.population])
# runner.population = sorted(runner.population, key=lambda individual: individual.fitness, reverse=True)
# 
# elite = runner.population[0]
# breakpoint()

# Generate and plot the fractal:
# target.plot_fractal(num_points=Parameters.n_points, initial_point=Parameters.initial_point)
# a_ifs.plot_fractal(num_points=Parameters.n_points, initial_point=Parameters.initial_point)
