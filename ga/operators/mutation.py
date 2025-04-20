import numpy as np


# Swap Mutation
def swap_mutation(individual):
    """Swap two random positions in the individual."""
    ind = individual.copy()
    idx1, idx2 = np.random.choice(len(ind), size=2, replace=False)
    ind[idx1], ind[idx2] = ind[idx2], ind[idx1]
    return ind

# Scramble Mutation
def scramble_mutation(individual):
    """Scramble a random subset of the individual."""
    ind = individual.copy()
    start = np.random.randint(0, len(ind) - 2)
    end = np.random.randint(start + 2, len(ind))  # ensure at least 2 elements
    subset = ind[start:end].copy()
    np.random.shuffle(subset)
    ind[start:end] = subset
    return ind

# Inversion Mutation
def inversion_mutation(individual):
    """Reverse a random subset of the individual."""
    ind = individual.copy()
    start = np.random.randint(0, len(ind) - 2)
    end = np.random.randint(start + 2, len(ind))
    ind[start:end] = ind[start:end][::-1]
    return ind
