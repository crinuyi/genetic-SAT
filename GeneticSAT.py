def check_disjunction(clause):
    result = 0
    for element_index in range(len(clause)):
        if element_index > 0:
            result |= clause[element_index]
        else:
            result |= not clause[-element_index]
    return result


class GeneticSAT:
    def __init__(self, formula):
        self.formula = formula

    def fitness(self, chromosome):
        result = 0
        for clause in self.formula:
            result += check_disjunction(clause)
        return result
