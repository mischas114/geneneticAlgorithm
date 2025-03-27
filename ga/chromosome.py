# Defines chromosome, gene range, and fitness evaluation
import numpy as np
from ga.utils import evaluate_function, mean_squared_error

GENE_SIZE = 8  # [a, b, c, d, e, g, h, i]
GENE_MIN = -10
GENE_MAX = 10

class Chromosome:
    def __init__(self, genes=None):
        if genes is None:
            # Random initialization
            self.genes = np.random.uniform(GENE_MIN, GENE_MAX, GENE_SIZE)
        else:
            self.genes = np.array(genes)
        
        self.fitness = None  # Placeholder for fitness value

    # Evaluate fitness of the chromosome
    def evaluate(self, x_vals, y_vals):
        y_pred = evaluate_function(x_vals, self.genes)
        self.fitness = mean_squared_error(y_vals, y_pred)
        return self.fitness

    # Copy the chromosome
    def copy(self):
        # Useful for cloning chromosomes
        return Chromosome(genes=self.genes.copy())