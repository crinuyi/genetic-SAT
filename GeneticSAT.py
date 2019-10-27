import random

import numpy
from deap import creator, base, tools, algorithms
from numpy.ma import absolute


def check_disjunction(clause, chromosome):
    result = 0
    for element in clause:
        variable_index = absolute(element) - 1
        if element > 0:
            result |= chromosome[variable_index]
        else:
            result |= not chromosome[variable_index]
    return result


class GeneticSAT:
    def __init__(self, formula, population_size, generations):
        self.formula = formula
        self.population_size = population_size
        self.generations = generations
        self.population = None
        self.best = None
        self.log = None
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
        return [result]

    def run(self):
        creator.create("FitnessSAT", base.Fitness, weights=(1.0,))
        creator.create("Individual", list, fitness=creator.FitnessSAT)

        toolbox = base.Toolbox()
        toolbox.register("attr_bool", random.randint, 0, 1)
        toolbox.register("individual", tools.initRepeat, creator.Individual,
                         toolbox.attr_bool, self.count_number_of_variables())
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)

        toolbox.register("evaluate", self.fitness)
        toolbox.register("mate", tools.cxTwoPoint)
        toolbox.register("mutate", tools.mutFlipBit, indpb=0.2)
        toolbox.register("select", tools.selTournament, tournsize=3)

        self.population = toolbox.population(n=self.population_size)

        self.best = tools.HallOfFame(1)
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", numpy.mean)
        stats.register("std", numpy.std)
        stats.register("min", numpy.min)
        stats.register("max", numpy.max)

        self.population, self.log = algorithms.eaSimple(
            self.population,
            toolbox,
            cxpb=0.5,
            mutpb=0.2,
            ngen=self.generations,
            stats=stats,
            halloffame=self.best,
            verbose=False)
