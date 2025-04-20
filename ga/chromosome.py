# ga/chromosome.py

import numpy as np
from ga.utils import evaluate_function, mean_squared_error

GENE_SIZE = 8  # [a, b, c, d, e, g, h, i]
GENE_MIN = -10
GENE_MAX = 10

class Chromosome:
    def __init__(self, genes=None):
        # default for minimization
        self.fitness = float('inf')
        
        if genes is not None:
            # coerce into a numpy array of floats
            self.genes = np.array(genes, dtype=float)
        else:
            # small random real values in [-1, +1]
            self.genes = np.random.uniform(GENE_MIN/10,
                                           GENE_MAX/10,
                                           size=GENE_SIZE)

    def evaluate(self, x_vals, y_vals):
        try:
            y_pred = evaluate_function(self.genes, x_vals)
            self.fitness = mean_squared_error(y_vals, y_pred)
            if self.fitness is None or not np.isfinite(self.fitness):
                self.fitness = 1e10
            return self.fitness
        except Exception as e:
            print(f"Error evaluating chromosome: {e}")
            self.fitness = 1e10
            return self.fitness

    def copy(self):
        # genes.copy() gives a new NumPy array
        return Chromosome(genes=self.genes.copy())
