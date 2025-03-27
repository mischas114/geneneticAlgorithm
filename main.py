import numpy as np
from ga.utils import load_target_values
from ga.chromosome import Chromosome  # Import Chromosome class

# This is so the chromosomes are the same across runs
# because we want to compare the effects of different operators
np.random.seed(42)

def main():
    # Load target values
    x_vals, y_vals = load_target_values("data/target_values.csv")

    # Initialize population (this will now be deterministic)
    population_size = 100
    population = [Chromosome() for _ in range(population_size)]

    # Evaluate fitness
    for chrom in population:
        chrom.evaluate(x_vals, y_vals)

    # Debug print to verify things are consistent
    print("First chromosome genes:", population[0].genes)
    print("First chromosome fitness:", population[0].fitness)

# TODO: Define fitness function
# - Use evaluate_function(coeffs, x_vals) to compute predicted y
# - Compute MSE between predicted y and actual y_vals

# TODO: Run genetic algorithm loop
# - For each generation:
#     - Evaluate fitness for all individuals
#     - Select parents (use Tournament or Roulette)
#     - Apply crossover (one-point, two-point, or uniform)
#     - Apply mutation (swap, scramble, or inversion)
#     - Create new population

# TODO: Track statistics
# - Save best fitness per generation
# - Print/log best individual and its fitness

# TODO: Run experiments with different operator combinations
# - Store results for comparison (e.g., in CSV or in-memory)

# TODO: Plot convergence (fitness over generations)
# - Use matplotlib to show how quickly/slowly the GA converges

# TODO: Save final best solution
# - Print or export best set of coefficients [a–i]

if __name__ == "__main__":
    main()
