"""
Main entry point for: Swarm Behaviour Evolution Using Genetic Algorithms (GA)

This module orchestrates the full evolutionary pipeline:
- Generates randomized obstacle-filled map configurations with valid leader paths.
- Evolves decentralized agent behaviors via a genetic algorithm.
- Evaluates swarm performance using metrics like leader proximity, path following, and collision counts.
- Plots evolution progress and saves top-performing simulation videos.
- Optionally tests the final genome on unseen maps for generalization.

All core parameters are defined in configs.py for flexible experimentation.

Authors:
    - @lexph
    - @noahmcconnell2
    - @jacksoneasden
    - @staroak

Date:
    - July 2025

Description:
    This script coordinates the full simulation lifecycle: training (evolution),
    post-training evaluation, metric logging, and result visualization.
    It integrates modules across mapping, simulation, evolution, and plotting.

Cite:
    Developed in collaboration with Microsoft's Copilot AI (2025),
    with assistance from OpenAI's ChatGPT for modular refactoring and documentation.

Dependencies:
    - DEAP for evolutionary strategy
    - Pygame for animation
    - Matplotlib / Seaborn for result visualization
"""


from map.map_generation import generate_n_map_configs
from simulation.run_sim import run_simulation
from simulation.evolution import run_genetic_algorithm
from simulation.movement import GeneIndex
import time
from configs import (NUM_TRAINING_CONFIGS, NUM_TESTING_CONFIGS, ENABLE_TESTING, TOP_GENOME, REPEATED_TESTING, EVOLUTION)
from pathlib import Path
from datetime import datetime
import shutil
import contextlib
import sys
import winsound
from simulation.plotting import (
    plot_fitness_convergence_band,
    plot_gene_evolution,
    plot_checkpoint_population_gene_boxplots,
    plot_sim_metrics_separate_boxplots,
    plot_metric_vs_gene
)

import signal

exit_requested = False

def signal_handler(sig, frame):
    global exit_requested
    exit_requested = True
    print("\n[Signal] Exit requested via keyboard interrupt.", file=sys.stderr)

signal.signal(signal.SIGINT, signal_handler)


def print_simulation_summary(summary: dict, map_index: int):
    print(f"\nMap {map_index + 1}: {summary['map_config']}")
    print(f"  -> Fitness: {float(summary['fitness']):.4f}")
    print(f"  -> Num Followers: {summary['num_followers']}")
    print(f"  -> Leader Path: {summary['leader_path']}")
    
    for j, agent in enumerate(summary['agents']):
        print(f"    Agent {j + 1}:")
        print(f"      steps: {agent['step_count']}")
        print(f"      obst. collisions: {agent['obstacle_collision_count']}")
        print(f"      agent collisions: {agent['agent_collision_count']}")
        print(f"      path_dist: {float(agent['path_distance_sum']):.1f}")
        print(f"      leader_dist: {float(agent['leader_distance_sum']):.1f}")
        print(f"      path: {agent['path']}")


def main():
    # Previous best genome
    best_genome = TOP_GENOME

    # Create timestamped log directory
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_dir = Path("logs") / timestamp
    log_dir.mkdir(parents=True, exist_ok=True)

    # Save a copy of the configs.py file
    shutil.copyfile("configs.py", log_dir / "configs.txt")

    if EVOLUTION:
        # Generate training maps
        training_map_configs = generate_n_map_configs(NUM_TRAINING_CONFIGS)

        # Create output log file
        output_log_path = log_dir / "output.txt"
        with open(output_log_path, "w") as f, contextlib.redirect_stdout(f):
            # Evolve genome
            print("Starting genetic algorithm evolution...\n")
            start_GA = time.time()
            previous_genomes = [TOP_GENOME]
            top_genomes, checkpoint_stats, mean_fitnesses, worst_fitnesses, checkpoint_populations = run_genetic_algorithm(training_map_configs, previous_genomes=previous_genomes)
            end_GA = time.time()

            best_genome = top_genomes[-1]
            print(f"Best genome found: {best_genome}")
            print(f"Best genome fitness: {best_genome.fitness.values[0]}")
            print(f"Worst genome in last generation: {top_genomes[0]}")
            print(f"Worst genome in last generation fitness: {top_genomes[0].fitness.values[0]}")
            print(f"Time taken for evolution: {end_GA - start_GA:.2f} seconds\n")
            print(f"Done with evolution!")
            winsound.PlaySound("assets/714564__lilmati__balcony-view-over-la.wav", winsound.SND_FILENAME | winsound.SND_ASYNC)

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

            plot_metric_vs_gene(top_genomes, gene_idx=GeneIndex.LEADER_DISTANCE, metric_key="avg_leader_distance",
                                title="Avg Leader Distance vs. Leader Distance Weight",
                                x_label="Leader Weight", y_label="Avg Leader Distance",
                                save=True, log_dir=log_dir)

            plot_metric_vs_gene(top_genomes, gene_idx=GeneIndex.PATH_FOLLOWING, metric_key="avg_path_distance",
                                title="Avg Distance to Path vs. Path Distance Weight",
                                x_label="Path Weight", y_label="Avg Path Distance",
                                save=True, log_dir=log_dir)

            plot_metric_vs_gene(top_genomes, gene_idx=GeneIndex.AVOIDANCE, metric_key="avg_obstacle_collisions",
                                title="Obstacle Collisions vs. Obstacle Weight",
                                x_label="Obstacle Avoidance Weight", y_label="Avg Obstacle Collisions",
                                save=True, log_dir=log_dir)

            plot_metric_vs_gene(top_genomes, gene_idx=GeneIndex.SEPARATION, metric_key="avg_agent_collisions",
                                title="Avg Agent Collisions vs. Separation Weight",
                                x_label="Separation Weight", y_label="Avg Agent Collisions",
                                save=True, log_dir=log_dir)
            
            plot_metric_vs_gene(top_genomes, gene_idx=GeneIndex.PATH_FOLLOWING, metric_key="fitness",
                                title="Avg fitness vs. Path Following Weight",
                                x_label="Path Following Weight", y_label="Avg Fitness",
                                save=True, log_dir=log_dir)
            

            # --- Training Summary ---
            print("\n--- Top Genome Training Summary ---")
            for i, mc in enumerate(training_map_configs):
                video_out_path = log_dir / f"training_map_{i+1}_swarm.mp4"
                _, stat = run_simulation(best_genome, mc, visualize=True, video_path=video_out_path)
                print_simulation_summary(stat, i)

    # --- Testing Section (feature flag) ---
    if ENABLE_TESTING:
        print("\nTesting enabled. Running best genome on unseen maps...\n")
        if not REPEATED_TESTING:
            test_map_configs = generate_n_map_configs(NUM_TESTING_CONFIGS)
            for i, test_map_config in enumerate(test_map_configs):
                video_out_path = log_dir / f"test_map_{i+1}_swarm.mp4"
                fitness, summary = run_simulation(best_genome, test_map_config, visualize=True, video_path=video_out_path)
                print_simulation_summary(summary, i)
                print(f"   Test fitness: {fitness:.4f}")
        elif REPEATED_TESTING:
            print("Repeated testing enabled. Press Ctrl + c in the terminal to stop after simulation window close...\n")
            i = 0
            while True:
                if exit_requested:
                    print("Early exit due to user request.")
                    break  # or return as needed
                test_map_config = generate_n_map_configs(1)[0]
                video_out_path = log_dir / f"test_map_{i+1}_swarm.mp4"
                fitness, summary = run_simulation(best_genome, test_map_config, visualize=True, video_path=video_out_path)
                print(f"   Test fitness: {fitness:.4f}")
                i+=1


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.set_start_method("spawn", force=True)
    main()