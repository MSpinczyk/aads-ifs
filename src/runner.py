from collections import namedtuple
from src.ifs import IteratedFunctionSystem
from src.parameters import Parameters
import numpy as np
from scipy.spatial import KDTree
import random
from src.operators import arithmetic_crossover, one_point_crossover, reassortment, random_mutation, gaussian_mutation, binary_mutation
import copy


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


    def calculate_fitness(self, attractor_ifs: IteratedFunctionSystem):
        attractor_points = attractor_ifs.generate_points(Parameters.n_points, Parameters.initial_point)
        attractor_tree = KDTree(attractor_points)
        target_tree = KDTree(self.target_ifs_points)

        attractor_neighbors = attractor_tree.query_ball_point(self.target_ifs_points, Parameters.fitness_radius, p=np.inf)
        target_neighbors = target_tree.query_ball_point(attractor_points, Parameters.fitness_radius, p=np.inf)

        # not drawn points - present in the image but not in the attractor
        n_nd = np.sum([len(neighbors) == 0 for neighbors in target_neighbors])
        # points not needed - present in the attractor but not in the image
        n_nn = np.sum([len(neighbors) == 0 for neighbors in attractor_neighbors])

        n_a = len(attractor_points)
        n_i = len(self.target_ifs_points)

        # relative coverage of the attractor
        r_c = n_nd/n_i
        # attractor points outside the image
        r_o = n_nn/n_a

        # fitness to maximise
        fitness = (1 - r_c) + (1 - r_o)

        attractor_ifs.fitness = fitness

        return fitness

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
            Bounds = namedtuple('PointBounds', ['lower_x', 'upper_x', 'lower_y', 'upper_y'])

            lower_x = min(self.target_ifs_points[:,0])
            upper_x = max(self.target_ifs_points[:,0])
            lower_y = min(self.target_ifs_points[:,1])
            upper_y = max(self.target_ifs_points[:,1])

            self._cached_ifs_bounds = Bounds(lower_x, upper_x, lower_y, upper_y)

        return self._cached_ifs_bounds
