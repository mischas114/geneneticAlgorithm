import random
import numpy as np

# Implements 3 crossover strategies: one-point, two-point, uniform
# one-point


def one_point_crossover(parent1, parent2):
    """Perform one-point crossover between two parents."""
    if len(parent1) != len(parent2):
        raise ValueError("Parents must be of the same length")  
    
    # Select a random crossover point
    # np is better than random for reproducibility and performance
    point = np.random.randint(1, len(parent1) - 1)  # Ensure at least one gene is swapped
    child1 = np.concatenate((parent1[:point], parent2[point:]), axis=0)
    child2 = np.concatenate((parent2[:point], parent1[point:]), axis=0)

    return child1, child2
# two-point
def two_point_crossover(parent1, parent2):
    """Perform two-point crossover between two parents."""
    if len(parent1) != len(parent2):
        raise ValueError("Parents must be of the same length")  
    
    # Select two random crossover points
    point1 = np.random.randint(1, len(parent1) - 1)
    point2 = np.random.randint(point1 + 1, len(parent1))

    child1 = np.concatenate((parent1[:point1], parent2[point1:point2], parent1[point2:]), axis=0)
    child2 = np.concatenate((parent2[:point1], parent1[point1:point2], parent2[point2:]), axis=0)

    return child1, child2
# uniform
def uniform_crossover(parent1, parent2):
    """Perform uniform crossover between two parents."""
    if len(parent1) != len(parent2):
        raise ValueError("Parents must be of the same length")  
    
    # Create a mask for crossover
    mask = np.random.randint(0, 2, size=len(parent1), dtype=bool)
    child1 = np.where(mask, parent1, parent2)
    child2 = np.where(mask, parent2, parent1)

    return child1, child2