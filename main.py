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
from configs import (NUM_TRAINING_CONFIGS, NUM_TESTING_CONFIGS,)


def main():

    # Generate 5 map configurations
    training_map_configs = generate_n_map_configs(NUM_TRAINING_CONFIGS)

    # Find best genome using the genetic algorithm
    best_genome = run_genetic_algorithm(training_map_configs)

    test_map_configs = generate_n_map_configs(NUM_TESTING_CONFIGS)

    # Run final simulation with the best genome
    for test_map_config in test_map_configs:
        print(f"Running simulation on test map config with genome: {best_genome}")
        best_fitness = run_simulation(best_genome, test_map_config, Visualize=True)
        print(f"Best fitness for test map config: {best_fitness}")
    

if __name__ == "__main__":
    main()