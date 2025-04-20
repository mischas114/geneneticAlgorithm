import numpy as np
import matplotlib.pyplot as plt
import os
import time
import pandas as pd
import matplotlib.pyplot as plt

from ga.utils import load_target_values, evaluate_function
from ga.chromosome import Chromosome

from ga.operators.selection import tournament_selection, roulette_wheel_selection
from ga.operators.crossover import one_point_crossover, two_point_crossover, uniform_crossover
from ga.operators.mutation import swap_mutation, scramble_mutation, inversion_mutation

from ga.utils import load_target_values, evaluate_function
# (run_genetic_algorithm is already in scope)

# Ensure results directory exists
os.makedirs("results", exist_ok=True)


SELECTIONS = {
    'tournament': tournament_selection,
    'roulette':  roulette_wheel_selection
}
CROSSOVERS = {
    'one_point': one_point_crossover,
    'two_point': two_point_crossover,
    'uniform':   uniform_crossover
}
MUTATIONS = {
    'swap':     swap_mutation,
    'scramble': scramble_mutation,
    'inversion': inversion_mutation
}

def run_genetic_algorithm(
    selection_op: str,
    crossover_op: str,
    mutation_op: str,
    population_size=100,
    generations=200,
    crossover_rate=0.9,
    mutation_rate=0.2,
    tournament_size=3,
    elite_size=4
):
    # 1) load data
    x_vals, y_vals = load_target_values("data/target_values.csv")

    # 2) init population of real‐valued Chromosomes
    if population_size % 2 != 0:
        population_size += 1
    population = [Chromosome() for _ in range(population_size)]

    # 3) pick operator functions
    select_fn = SELECTIONS[selection_op]
    cross_fn  = CROSSOVERS[crossover_op]
    mut_fn    = MUTATIONS[mutation_op]

    best_history = []
    avg_history  = []

    for gen in range(generations):
        # --- evaluate all ---
        for chrom in population:
            chrom.evaluate(x_vals, y_vals)

        # --- sort by fitness (MSE) ascending ---
        population.sort(key=lambda c: c.fitness)

        # record stats
        best_f = population[0].fitness
        avg_f  = np.mean([c.fitness for c in population])
        best_history.append(best_f)
        avg_history.append(avg_f)

        # --- elitism ---
        new_pop = [population[i].copy() for i in range(elite_size)]

        # invert fitness for selection (higher is better)
        inv_fitnesses = [1.0/c.fitness if c.fitness>0 else 1e6 for c in population]

        # --- fill rest of population ---
        while len(new_pop) < population_size:
            # parent selection
            if selection_op == 'tournament':
                p1 = select_fn(population, inv_fitnesses, tournament_size)
                p2 = select_fn(population, inv_fitnesses, tournament_size)
            else:
                p1 = select_fn(population, inv_fitnesses)
                p2 = select_fn(population, inv_fitnesses)

            # crossover
            import secrets
            if secrets.randbelow(100) / 100 < crossover_rate:
                c1_genes, c2_genes = cross_fn(p1.genes, p2.genes)
            else:
                c1_genes, c2_genes = p1.genes.copy(), p2.genes.copy()

            if secrets.randbelow(100) / 100 < mutation_rate:
                c1_genes = mut_fn(c1_genes)
            if secrets.randbelow(100) / 100 < mutation_rate:
                c2_genes = mut_fn(c2_genes)

            # wrap back into Chromosome
            new_pop.append(Chromosome(c1_genes))
            if len(new_pop) < population_size:
                new_pop.append(Chromosome(c2_genes))

        population = new_pop

    # final evaluation to grab best ever
    for chrom in population:
        chrom.evaluate(x_vals, y_vals)
    population.sort(key=lambda c: c.fitness)

    best = population[0]
    return best, best_history, avg_history

if __name__ == "__main__":
    # --- 1) Single‐run demo ---
    print(">>> SINGLE RUN DEMO <<<")
    best, best_hist, avg_hist = run_genetic_algorithm(
        selection_op='tournament',
        crossover_op='two_point',
        mutation_op='scramble',
        population_size=150,
        generations=300,
        crossover_rate=0.9,
        mutation_rate=0.2,
        tournament_size=5,
        elite_size=4
    )
    print(f"Best MSE: {best.fitness:.6f}")
    print(f"Best genes: {best.genes}\n")

    # plot its convergence and save
    plt.figure(figsize=(6,4))
    plt.plot(best_hist, label='Best')
    plt.plot(avg_hist, label='Average', linestyle='--')
    plt.xlabel("Generation")
    plt.ylabel("MSE")
    plt.title("Demo Convergence (tournament/two_point/scramble)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("results/demo_convergence.png")
    plt.close()

    # --- 2) Full operator‐comparison ---
    combos = [
        ('tournament','one_point','swap'),
        ('roulette','two_point','scramble'),
        ('tournament','uniform','inversion'),
        ('roulette','uniform','swap')
    ]

    records = []
    print(">>> COMPARISON RUNS <<<")
    for sel, crs, mut in combos:
        print(f"Running {sel}/{crs}/{mut}...", end=' ')
        t0 = time.perf_counter()
        b, bh, ah = run_genetic_algorithm(
            selection_op=sel,
            crossover_op=crs,
            mutation_op=mut,
            population_size=120,
            generations=150,
            crossover_rate=0.9,
            mutation_rate=0.2,
            tournament_size=5,
            elite_size=4
        )
        dt = time.perf_counter() - t0
        gens = [i for i,m in enumerate(bh) if m < 1.0]
        gen_to1 = gens[0] if gens else None

        records.append({
            'sel': sel,
            'crs': crs,
            'mut': mut,
            'best_mse': b.fitness,
            'gen<MSE1': gen_to1,
            'time_s': dt,
            'best_hist': bh,
            'avg_hist': ah,
            'best_chrom': b
        })
        print(f"done in {dt:.2f}s, final MSE={b.fitness:.4f}")

    # build summary DataFrame
    summary = pd.DataFrame([{
        'Selection': r['sel'],
        'Crossover': r['crs'],
        'Mutation':  r['mut'],
        'Final MSE': r['best_mse'],
        'Gen<MSE1':  r['gen<MSE1'],
        'Time (s)':  r['time_s']
    } for r in records])

    print("\n=== OPERATOR COMPARISON SUMMARY ===")
    print(summary.to_string(index=False))

    # save summary CSV
    summary_path = "results/operator_comparison_summary.csv"
    summary.to_csv(summary_path, index=False)
    print(f"Summary saved to {summary_path}")

    # --- 3) Overlaid convergence (linear) ---
    plt.figure(figsize=(7,4))
    for r in records:
        label = f"{r['sel']}/{r['crs']}/{r['mut']}"
        plt.plot(r['best_hist'], label=label)
    plt.xlabel("Generation")
    plt.ylabel("Best MSE")
    plt.title("Convergence Comparison (linear)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("results/convergence_linear.png")
    plt.close()

    # --- 4) Overlaid convergence (log‑y) ---
    plt.figure(figsize=(7,4))
    for r in records:
        label = f"{r['sel']}/{r['crs']}/{r['mut']}"
        plt.plot(r['best_hist'], label=label)
    plt.yscale('log')
    plt.xlabel("Generation")
    plt.ylabel("Best MSE (log scale)")
    plt.title("Convergence Comparison (log‑y)")
    plt.legend()
    plt.grid(True, which='both')
    plt.tight_layout()
    plt.savefig("results/convergence_log.png")
    plt.close()

    # --- 5) Best fit vs target for top combo ---
    best_rec = min(records, key=lambda x: x['best_mse'])
    bc = best_rec['best_chrom']
    x_vals, y_vals = load_target_values("data/target_values.csv")
    y_pred = evaluate_function(bc.genes, x_vals)

     # **SORT before plotting** to avoid zig-zags:
    idx = np.argsort(x_vals)
    x_sorted      = x_vals[idx]
    y_pred_sorted = y_pred[idx]

    plt.figure(figsize=(6,4))
    plt.scatter(x_vals, y_vals, label='Target', s=30)
    plt.plot(x_sorted, y_pred_sorted,
             label=f"GA fit ({best_rec['sel']}/{best_rec['crs']}/{best_rec['mut']})",
             linewidth=2)
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.title("Best GA Fit vs Target Data")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("results/best_fit_vs_target.png")
    plt.close()

    print("All plots saved in the 'results/' folder.")
    print("Done.")