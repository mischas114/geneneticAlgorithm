from ga.utils import load_target_values

def main():
    x_vals, y_vals = load_target_values("src/data/target_values.csv")
    print("Loaded data points:", len(x_vals))
    print("Sample:", list(zip(x_vals, y_vals))[:5])

# TODO: Load target data from CSV
# - Use the load_target_values() function
# - Store x and y (input and expected output)

# TODO: Initialize population
# - Each chromosome should be a list of 8 real-valued genes [a, b, c, d, e, g, h, i]
# - Randomly initialize values in range [-10, 10]

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
