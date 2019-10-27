from GeneticSAT import GeneticSAT


def main():
    formula = [
        [-1, 2, 4],
        [-2, 3, 4],
        [1, -3, 4],
        [1, -2, -4],
        [2, -3, -4],
        [2, -3, -4],
        [-1, 3, -4],
        [1, 2, 3]
    ]
    genetic_sat = GeneticSAT(formula, 200, 500)
    genetic_sat.run()
    print(genetic_sat.log)


if __name__ == '__main__':
    main()
