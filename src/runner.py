from collections import namedtuple
from src.ifs import IteratedFunctionSystem
from src.parameters import Parameters
import numpy as np
from scipy.spatial import KDTree
import random
from src.operators import arithmetic_crossover, one_point_crossover, reassortment, random_mutation, gaussian_mutation, binary_mutation
import copy
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt


class Runner:
    def __init__(self, target_ifs: IteratedFunctionSystem):
        self.population: list[IteratedFunctionSystem] = []
        self.target_ifs = target_ifs
        self.target_ifs_points = target_ifs.generate_points(
            Parameters.n_points, Parameters.initial_point
        )
        self._cached_ifs_bounds = None
        self.mean_fitness = None
        self.best = IteratedFunctionSystem()

    def step(self):
        new_pop = []

        for ifs in self.population:
            self.calculate_fitness(ifs)

        self.mean_fitness = np.mean([i.fitness for i in self.population])
        self.population = sorted(self.population, key=lambda individual: individual.fitness, reverse=True)

        elite = self.population[0]

        if self.population[0].fitness > self.best.fitness:
            self.best = copy.deepcopy(self.population[0])

        if elite.fitness > Parameters.elite_fitness_threshold:
            new_pop.append(elite) 

        new_pop += self.recombination_offspring()
        new_pop += self.self_creation_offspring()
        new_pop += self.reassortment_offspring()

        for ifs in new_pop:
            if random.random() < Parameters.mutation_probability:
                if random.random() < self.mean_fitness / 2:
                    gaussian_mutation(ifs)
                else:
                    random_mutation(ifs) if random.random() < 0.5 else binary_mutation(ifs)

        self.population = new_pop


    @staticmethod
    def find_closest_point(point, grid_points):
        distances = cdist([point], grid_points, metric='euclidean')
        closest_index = np.argmin(distances)
        return grid_points[closest_index]

    def calculate_fitness(self, attractor_ifs: IteratedFunctionSystem):
        attractor_points = attractor_ifs.generate_points(Parameters.n_points, Parameters.initial_point)
        target_points = self.target_ifs_points

        # Create the grid only once if it remains constant
        if not hasattr(self, 'grid_points'):
            x_grid, y_grid = np.meshgrid(np.linspace(self.target_ifs_bounds.x_min, self.target_ifs_bounds.x_max, Parameters.fitness_grid_resolution),
                                        np.linspace(self.target_ifs_bounds.y_min, self.target_ifs_bounds.y_max, Parameters.fitness_grid_resolution))
            self.grid_points = np.column_stack((x_grid.flatten(), y_grid.flatten()))


        attractor_points_inside_bounds = [
            point for point in attractor_points if self.target_ifs_bounds.x_min <= point[0] <= self.target_ifs_bounds.x_max and self.target_ifs_bounds.y_min <= point[1] <= self.target_ifs_bounds.y_max
        ]

        target_closest_points = [self.find_closest_point(target_point, self.grid_points) for target_point in target_points]
        attractor_closest_points = [self.find_closest_point(attractor_point, self.grid_points) for attractor_point in attractor_points_inside_bounds]

        attractor_set = set(map(tuple, attractor_closest_points))
        target_set = set(map(tuple, target_closest_points))

        n_nn = len(attractor_set - target_set) + len(attractor_points) - len(attractor_points_inside_bounds)
        n_nd = len(target_set - attractor_set)

        n_a = len(attractor_points)
        n_i = len(self.target_ifs_points)

        # Avoid division by zero
        r_c = n_nd / n_i if n_i != 0 else 1
        r_o = n_nn / n_a if n_a != 0 else 1

        # fitness to maximize
        fitness = Parameters.p_rc * (1 - r_c) + Parameters.p_ro * (1 - r_o)

        attractor_ifs.fitness = fitness

        return fitness

    def plot_fitness_grid(self):
        attractor_ifs = self.best

        attractor_points = attractor_ifs.generate_points(Parameters.n_points, Parameters.initial_point)
        target_points = self.target_ifs_points

        if not hasattr(self, 'grid_points'):
            x_grid, y_grid = np.meshgrid(np.linspace(self.target_ifs_bounds.x_min, self.target_ifs_bounds.x_max, Parameters.fitness_grid_resolution),
                                        np.linspace(self.target_ifs_bounds.y_min, self.target_ifs_bounds.y_max, Parameters.fitness_grid_resolution))
            self.grid_points = np.column_stack((x_grid.flatten(), y_grid.flatten()))


        attractor_points_inside_bounds = [
            point for point in attractor_points if self.target_ifs_bounds.x_min <= point[0] <= self.target_ifs_bounds.x_max and self.target_ifs_bounds.y_min <= point[1] <= self.target_ifs_bounds.y_max
        ]

        target_closest_points = [self.find_closest_point(target_point, self.grid_points) for target_point in target_points]
        attractor_closest_points = [self.find_closest_point(attractor_point, self.grid_points) for attractor_point in attractor_points_inside_bounds]

        attractor_set = set(map(tuple, attractor_closest_points))
        target_set = set(map(tuple, target_closest_points))

        plt.scatter(*zip(*attractor_points), color='red', s=1, label='Attractor Points')
        plt.scatter(*zip(*target_points), color='blue', s=1, label='Target Points')
        # plt.scatter(*zip(*target_closest_points), color='green', s=1.5, label='Target Closest Point')
        # plt.scatter(*zip(*attractor_closest_points), color='orange', s=1.5, label='Attractor Closest Point')
        plt.scatter(*zip(*(attractor_set - target_set)), color='violet', s=1.5, label='Points not needed')
        plt.scatter(*zip(*(target_set - attractor_set)), color='olive', s=1.5, label='Points not drawn')
        plt.scatter(*zip(*self.grid_points), color='gray', s=1, alpha=0.2, marker='.')
        plt.legend()
        plt.show()

    def generate_first_population(self):
        for _ in range(Parameters.initial_population_size):
            size = np.random.randint(Parameters.min_individual_degree, Parameters.max_individual_degree + 1)
            ifs = IteratedFunctionSystem()
            for _ in range(size):
                ifs.add_function(np.random.uniform(-1, 1, size=(6)))
            self.population.append(ifs)

    def fitness_proportional_selection(self) -> list[IteratedFunctionSystem]:
        total_fitness = sum(individual.fitness for individual in self.population)
        selection_probabilities = [individual.fitness / total_fitness for individual in self.population]

        selected_individuals = []
        for _ in range(2):
            selected_individual = random.choices(self.population, weights=selection_probabilities)[0]
            selected_individuals.append(selected_individual)

        return selected_individuals

    def recombination_offspring(self) -> list[IteratedFunctionSystem]:
        offspring = []
        for _ in range(Parameters.recombination_population_size // 2):
            parents = self.fitness_proportional_selection()
            if random.random() <= Parameters.p_arithmetic_crossover:
                children = arithmetic_crossover(*parents)
            else:
                children = one_point_crossover(*parents)

            offspring += children
        return offspring

    def self_creation_offspring(self) -> list[IteratedFunctionSystem]:
        offspring = []

        if 1/self.mean_fitness > Parameters.max_self_creation_population_size:
            n_specimen = Parameters.max_self_creation_population_size
        else:
            n_specimen = int(1/self.mean_fitness)

        genetic_universum = [function for ifs in self.population for function in ifs.functions]
        size = np.random.randint(Parameters.min_individual_degree, Parameters.max_individual_degree + 1)

        for _ in range(n_specimen):
            ifs = IteratedFunctionSystem()
            ifs.functions += random.sample(genetic_universum, size)
            offspring.append(ifs)

        return offspring

    def reassortment_offspring(self) -> list[IteratedFunctionSystem]:
        offspring = []
        for _ in range(Parameters.reassortment_population_size // 2):
            parents = self.fitness_proportional_selection()
            children = reassortment(*parents)
            offspring += children

        return offspring

    @property
    def target_ifs_bounds(self):
        if self._cached_ifs_bounds is None:
            Bounds = namedtuple('PointBounds', ['x_min', 'x_max', 'y_min', 'y_max'])

            lower_x = min(self.target_ifs_points[:,0])
            upper_x = max(self.target_ifs_points[:,0])
            lower_y = min(self.target_ifs_points[:,1])
            upper_y = max(self.target_ifs_points[:,1])

            self._cached_ifs_bounds = Bounds(lower_x, upper_x, lower_y, upper_y)

        return self._cached_ifs_bounds
