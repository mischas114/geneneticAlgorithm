import numpy as np

# Tournament Selection
def tournament_selection(population, fitnesses, tournament_size=3):
    """Select one individual using tournament selection."""
    selected_indices = np.random.choice(len(population), size=tournament_size, replace=False)
    best_index = selected_indices[np.argmax([fitnesses[i] for i in selected_indices])]
    return population[best_index]

# Roulette Wheel Selection
def roulette_wheel_selection(population, fitnesses):
    """Select one individual using roulette wheel (fitness proportionate) selection."""
    total_fitness = sum(fitnesses)
    if total_fitness == 0:
        # Avoid division by zero: return random individual
        return population[np.random.randint(0, len(population))]

    probabilities = [f / total_fitness for f in fitnesses]
    selected_index = np.random.choice(len(population), p=probabilities)
    return population[selected_index]
