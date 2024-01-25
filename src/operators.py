from src.ifs import IteratedFunctionSystem
from src.parameters import Parameters
import numpy as np
import random

def arithmetic_crossover(p1: IteratedFunctionSystem, p2: IteratedFunctionSystem) -> list[IteratedFunctionSystem]:
    c1 = IteratedFunctionSystem()
    c2 = IteratedFunctionSystem()

    c1.functions = [f1*(1-Parameters.a) + f2*Parameters.a for f1, f2 in zip(p1.functions, p2.functions)]
    c2.functions = [f1*Parameters.a + f2*(1-Parameters.a) for f1, f2 in zip(p1.functions, p2.functions)]

    return [c1, c2]

def one_point_crossover(p1: IteratedFunctionSystem, p2: IteratedFunctionSystem) -> list[IteratedFunctionSystem]:
    # instead of exchanging functions, you should exchange coefficients of two random functions

    crossover_point = random.randint(0, len(p1.functions) - 1)

    c1 = IteratedFunctionSystem()
    c2 = IteratedFunctionSystem()

    c1.functions = p1.functions[:crossover_point] + p2.functions[crossover_point:]
    c2.functions = p2.functions[:crossover_point] + p1.functions[crossover_point:]

    return [c1, c2]

def reassortment(p1: IteratedFunctionSystem, p2: IteratedFunctionSystem) -> list[IteratedFunctionSystem]:
    all_functions = p1.functions + p2.functions
    shuffled_functions = random.sample(all_functions, len(all_functions))

    c1 = IteratedFunctionSystem()
    c2 = IteratedFunctionSystem()

    while shuffled_functions:
        c1.add_function(shuffled_functions.pop())
        c2.add_function(shuffled_functions.pop())

    return [c1, c2]

def random_mutation(ifs: IteratedFunctionSystem):
    a = Parameters.min_singel_coefficient
    b = Parameters.max_singel_coefficient
    mutated_functions = []
    for function in ifs.functions:
        mutated_functions.append(np.array([random.uniform(a, b) for _ in function]))

    ifs.functions = mutated_functions

def gaussian_mutation(ifs: IteratedFunctionSystem):
    a = Parameters.min_singel_coefficient
    b = Parameters.max_singel_coefficient
    r = Parameters.guassian_mutation_radius
    mutated_functions = []
    for function in ifs.functions:
        mutated_coefficients = [np.clip(np.random.normal(loc=coefficient, scale=r * (b - a)), a, b) for coefficient in function]
        mutated_functions.append(np.array(mutated_coefficients))

    ifs.functions = mutated_functions

def binary_mutation(ifs: IteratedFunctionSystem):
    mutated_functions = []
    for function in ifs.functions:
        mutated_coefficients = []
        for coefficient in function:
            binary_representation = bin(int(coefficient * 100))[2:]  # Assuming coefficients are in the range [-1, 1]
            mutated_binary = ''.join([bit if random.random() > 0.5 else ('1' if bit == '0' else '0') for bit in binary_representation])
            mutated_binary = mutated_binary[1:] if mutated_binary.startswith('b') else mutated_binary
            mutated_coefficient = int(mutated_binary, 2) / 100.0
            mutated_coefficients.append(mutated_coefficient)
        mutated_functions.append(np.array(mutated_coefficients))

    ifs.functions = mutated_functions
