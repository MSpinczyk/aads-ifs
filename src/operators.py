from src.ifs import IteratedFunctionSystem
from src.parameters import Parameters
import random

def arithmetic_crossover(p1: IteratedFunctionSystem, p2: IteratedFunctionSystem) -> list[IteratedFunctionSystem]:
    c1 = IteratedFunctionSystem()
    c2 = IteratedFunctionSystem()

    c1.operators = [o1*(1-Parameters.a) + o2*Parameters.a for o1, o2 in zip(p1.operators, p2.operators)]
    c2.operators = [o1*Parameters.a + o2*(1-Parameters.a) for o1, o2 in zip(p1.operators, p2.operators)]

    return [c1, c2]

def one_point_crossover(p1: IteratedFunctionSystem, p2: IteratedFunctionSystem) -> list[IteratedFunctionSystem]:
    # instead of exchanging operators, you should exchange coefficients of two random operators

    crossover_point = random.randint(0, len(p1.operators) - 1)

    c1 = IteratedFunctionSystem()
    c2 = IteratedFunctionSystem()

    c1.operators = p1.operators[:crossover_point] + p2.operators[crossover_point:]
    c2.operators = p2.operators[:crossover_point] + p1.operators[crossover_point:]

    return [c1, c2]

def reassortment(p1: IteratedFunctionSystem, p2: IteratedFunctionSystem) -> list[IteratedFunctionSystem]:
    all_operators = p1.operators + p2.operators
    shuffled_operators = random.sample(all_operators, len(all_operators))

    c1 = IteratedFunctionSystem()
    c2 = IteratedFunctionSystem()

    while shuffled_operators:
        c1.add_operator(shuffled_operators.pop())
        c2.add_operator(shuffled_operators.pop())

    return [c1, c2]
