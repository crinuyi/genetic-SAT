import random

from deap import creator, base
from deap.benchmarks import tools
from numpy.ma import absolute


def check_disjunction(clause, chromosome):
    result = 0
    for element_index in range(len(clause)):
        variable_index = absolute(clause[element_index])
        if clause[element_index] > 0:
            result |= chromosome[variable_index]
        else:
            result |= not chromosome[variable_index]
    return result


class GeneticSAT:
    def __init__(self, formula):
        self.formula = formula
        self.number_of_variables = self.count_number_of_variables()

    def count_number_of_variables(self):
        max_in_formula = 0
        for clause in self.formula:
            max_in_clause = max(absolute(clause))
            if max_in_clause > max_in_formula:
                max_in_formula = max_in_clause
        return max_in_formula

    def fitness(self, chromosome):
        result = 0
        for clause in self.formula:
            result += check_disjunction(clause, chromosome)
        return result

    def run(self):
        creator.create("FitnessSAT", base.Fitness)
        creator.create("Individual", list, fitness=creator.FitnessSAT)

        toolbox = base.Toolbox()
        toolbox.register("attr_bool", random.randint, 0, 1)
        toolbox.register("individual", tools.initRepeat, creator.Individual,
                         toolbox.attr_bool, self.count_number_of_variables())
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)