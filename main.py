"""
Main entry point for 'Swarm Behaviour Evolution Using Genetic Algorithms (GA)'

This script coordinates the overall pipeline:
- Generates map configurations with obstacle layouts and leader paths
- Evolves agent behavior using a genetic algorithm to optimize swarm performance
- Tests the best-evolved genome on unseen environments

All tunable parameters are defined in configs.py.

Authors:
    - @lexph
    - [Add collaborator name here]

Date:
    - July 2025

Description:
    This module orchestrates training and evaluation of swarm agent behavior using
    evolutionary algorithms. It integrates map generation, simulation execution,
    and genome evolution.

Cite:
    Developed in collaboration with Microsoft's Copilot AI (2025),
    with assistance on documentation and modular refactoring.
"""


from map.map_generation import generate_n_map_configs
from simulation.run_sim import run_simulation
from simulation.evolution import run_genetic_algorithm
import time
from configs import (NUM_TRAINING_CONFIGS, NUM_TESTING_CONFIGS, ENABLE_TESTING)
from pathlib import Path
from datetime import datetime
from simulation.plotting import (
    plot_fitness_convergence_band,
    plot_gene_evolution,
    plot_checkpoint_population_gene_boxplots,
    plot_sim_metrics_separate_boxplots,
    plot_metric_vs_gene
)


def print_simulation_summary(summary: dict, map_index: int):
    print(f"\nMap {map_index + 1}: {summary['map_config']}")
    print(f"  → Fitness: {float(summary['fitness']):.4f}")
    print(f"  → Num Followers: {summary['num_followers']}")
    print(f"  → Leader Path: {summary['leader_path']}")
    
    for j, agent in enumerate(summary['agents']):
        print(f"    Agent {j + 1}:")
        print(f"      steps: {agent['step_count']}")
        print(f"      obst. collisions: {agent['obstacle_collision_count']}")
        print(f"      agent collisions: {agent['agent_collision_count']}")
        print(f"      path_dist: {float(agent['path_distance_sum']):.1f}")
        print(f"      leader_dist: {float(agent['leader_distance_sum']):.1f}")
        print(f"      path: {agent['path']}")





def main():
    # Generate training maps
    training_map_configs = generate_n_map_configs(NUM_TRAINING_CONFIGS)

    # Evolve genome
    print("Starting genetic algorithm evolution...\n")
    start_GA = time.time()
    top_genomes, checkpoint_stats, mean_fitnesses, worst_fitnesses, checkpoint_populations = run_genetic_algorithm(training_map_configs)
    end_GA = time.time()

    # Create timestamped log directory
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_dir = Path("logs") / timestamp
    log_dir.mkdir(parents=True, exist_ok=True)
    best_genome = top_genomes[-1]
    print(f"Best genome found: {best_genome}")
    print(f"Best genome fitness: {best_genome.fitness.values[0]}")
    print(f"Time taken for evolution: {end_GA - start_GA:.2f} seconds\n")
    print(f"Done with evolution!")

    ## ----------- Plots ----------------
    # Plot gene evolution
    plot_gene_evolution(top_genomes, save=True, log_dir=log_dir)

    # Plot fitness convergence band -> save in logs/<timestamp>/fitness_convergence_band
    plot_fitness_convergence_band(top_genomes, mean_fitnesses, worst_fitnesses, save=True, log_dir=log_dir)

    # Box plot of gene variance in a population at select generations: start, middle, end
    top_genomes_dict = {
        "start": top_genomes[0],
        "mid": top_genomes[len(top_genomes) // 2],
        "end": top_genomes[-1],
    }
    plot_checkpoint_population_gene_boxplots(
        checkpoint_populations=checkpoint_populations,
        top_genomes=top_genomes_dict,
        save=True,
        log_dir=log_dir
    )

    # Box plot of each sim metric variance in the population at select generations: start, middle, end
    plot_sim_metrics_separate_boxplots(checkpoint_stats, save=True, log_dir=log_dir)

    plot_metric_vs_gene(top_genomes, gene_idx=4, metric_key="avg_leader_distance",
                        title="Avg Leader Distance vs. Leader Distance Weight",
                        x_label="Leader Weight", y_label="Avg Leader Distance",
                        save=True, log_dir=log_dir)

    plot_metric_vs_gene(top_genomes, gene_idx=3, metric_key="avg_path_distance",
                        title="Avg Distance to Path vs. Path Distance Weight",
                        x_label="Path Weight", y_label="Avg Path Distance",
                        save=True, log_dir=log_dir)

    plot_metric_vs_gene(top_genomes, gene_idx=2, metric_key="avg_obstacle_collisions",
                        title="Obstacle Collisions vs. Obstacle Weight",
                        x_label="Obstacle Avoidance Weight", y_label="Avg Obstacle Collisions",
                        save=True, log_dir=log_dir)

    plot_metric_vs_gene(top_genomes, gene_idx=1, metric_key="avg_agent_collisions",
                        title="Avg Agent Collisions vs. Separation Weight",
                        x_label="Separation Weight", y_label="Avg Agent Collisions",
                        save=True, log_dir=log_dir)
    

    # --- Training Summary ---
    print("\n--- Top Genome Training Summary ---")
    best_stats = [run_simulation(best_genome, mc, visualize=True)[1] for mc in training_map_configs]
    for i, stat in enumerate(best_stats):
        print_simulation_summary(stat, i)

    # --- Testing Section (feature flag) ---
    if ENABLE_TESTING:
        print("\nTesting enabled. Running best genome on unseen maps...\n")
        test_map_configs = generate_n_map_configs(NUM_TESTING_CONFIGS)
        for i, test_map_config in enumerate(test_map_configs):
            fitness, summary = run_simulation(best_genome, test_map_config, visualize=True)
            print_simulation_summary(summary, i)
            print(f"   Test fitness: {fitness:.4f}")


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.set_start_method("spawn", force=True)
    main()