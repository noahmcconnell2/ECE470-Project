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
    best_genome = run_genetic_algorithm(training_map_configs)
    end_GA = time.time()

    print(f"Best genome found: {best_genome}")
    print(f"Best genome fitness: {best_genome.fitness.values[0]}")
    print(f"Time taken for evolution: {end_GA - start_GA:.2f} seconds\n")
    print("Done with evolution!")

    # --- Training Summary ---
    print("\n--- Training Summary ---")
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