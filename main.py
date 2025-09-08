"""
Swarm Behaviour Evolution Using Genetic Algorithms (GA)

Refactored entry point:
- Orchestrates training (evolution), plotting, and testing.
- Keeps side effects (files, videos, plots, sounds) behind clear functions.
- Avoids global state; handles Ctrl-C gracefully.

Authors: @lexph, @noahmcconnell2, @jacksoneasden, @staroak
Date: July 2025
"""

from __future__ import annotations

import contextlib
import shutil
import signal
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Tuple

# --- External modules (unchanged) ---
from map.map_generation import generate_n_map_configs
from simulation.run_sim import run_simulation
from simulation.evolution import run_genetic_algorithm
from simulation.movement import GeneIndex
from simulation.plotting import (
    plot_fitness_convergence_band,
    plot_gene_evolution,
    plot_checkpoint_population_gene_boxplots,
    plot_sim_metrics_separate_boxplots,
    plot_metric_vs_gene,
)

# --- Project configs ---
from configs import (
    NUM_TRAINING_CONFIGS,
    NUM_TESTING_CONFIGS,
    ENABLE_TESTING,
    TOP_GENOME,
    REPEATED_TESTING,
    EVOLUTION,
)

# Windows-only; guarded so non-Windows envs won’t explode
try:
    import winsound  # type: ignore
except Exception:  # pragma: no cover
    winsound = None


# ----------------------------
# Data structures & utilities
# ----------------------------

@dataclass(frozen=True)
class RunContext:
    timestamp: str
    log_dir: Path
    output_log_path: Path

def make_run_context(base_dir: Path = Path("logs")) -> RunContext:
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_dir = base_dir / ts
    log_dir.mkdir(parents=True, exist_ok=True)
    # retain a copy of configs used for traceability
    shutil.copyfile("configs.py", log_dir / "configs.txt")
    return RunContext(timestamp=ts, log_dir=log_dir, output_log_path=log_dir / "output.txt")


def bell(path: str = "assets/714564__lilmati__balcony-view-over-la.wav") -> None:
    """Play a sound if available; otherwise no-op."""
    if winsound:
        winsound.PlaySound(path, winsound.SND_FILENAME | winsound.SND_ASYNC)


# ----------------------------
# Pretty printing / summaries
# ----------------------------

def print_simulation_summary(summary: dict, map_index: int) -> None:
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


# ----------------------------
# Plotting bundle
# ----------------------------

def generate_plots(
    *,
    log_dir: Path,
    top_genomes: List,
    checkpoint_stats: List[dict],
    mean_fitnesses: List[float],
    worst_fitnesses: List[float],
    checkpoint_populations: List[Iterable],
) -> None:
    # 1) gene evolution & fitness band
    plot_gene_evolution(top_genomes, save=True, log_dir=log_dir)
    plot_fitness_convergence_band(
        top_genomes, mean_fitnesses, worst_fitnesses, save=True, log_dir=log_dir
    )

    # 2) population spread snapshots
    tg = {
        "start": top_genomes[0],
        "mid": top_genomes[len(top_genomes) // 2],
        "end": top_genomes[-1],
    }
    plot_checkpoint_population_gene_boxplots(
        checkpoint_populations=checkpoint_populations,
        top_genomes=tg,
        save=True,
        log_dir=log_dir,
    )

    # 3) metric spreads
    plot_sim_metrics_separate_boxplots(checkpoint_stats, save=True, log_dir=log_dir)

    # 4) helpful metric-vs-gene slices
    pairs: List[Tuple[GeneIndex, str, str, str]] = [
        (GeneIndex.LEADER_DISTANCE, "avg_leader_distance",
         "Avg Leader Distance vs. Leader Distance Weight",
         "Leader Weight"),
        (GeneIndex.PATH_FOLLOWING, "avg_path_distance",
         "Avg Distance to Path vs. Path Distance Weight",
         "Path Weight"),
        (GeneIndex.AVOIDANCE, "avg_obstacle_collisions",
         "Obstacle Collisions vs. Obstacle Weight",
         "Obstacle Avoidance Weight"),
        (GeneIndex.SEPARATION, "avg_agent_collisions",
         "Avg Agent Collisions vs. Separation Weight",
         "Separation Weight"),
    ]
    for gidx, mkey, title, xlabel in pairs:
        plot_metric_vs_gene(
            top_genomes,
            gene_idx=gidx,
            metric_key=mkey,
            title=title,
            x_label=xlabel,
            y_label=title.split(" vs. ")[0],
            save=True,
            log_dir=log_dir,
        )

    plot_metric_vs_gene(
        top_genomes,
        gene_idx=GeneIndex.PATH_FOLLOWING,
        metric_key="fitness",
        title="Avg Fitness vs. Path Following Weight",
        x_label="Path Following Weight",
        y_label="Avg Fitness",
        save=True,
        log_dir=log_dir,
    )


# ----------------------------
# Runners: evolution & testing
# ----------------------------

def run_evolution(ctx: RunContext, previous_genomes: List) -> Tuple:
    """Run GA on generated training maps and return artifacts."""
    training_map_configs = generate_n_map_configs(NUM_TRAINING_CONFIGS)

    print("Starting genetic algorithm evolution...\n")
    t0 = time.time()
    results = run_genetic_algorithm(
        training_map_configs, previous_genomes=previous_genomes
    )
    top_genomes, checkpoint_stats, mean_fitnesses, worst_fitnesses, checkpoint_populations = results
    t1 = time.time()

    best_genome = top_genomes[-1]
    print(f"Best genome found: {best_genome}")
    print(f"Best genome fitness: {best_genome.fitness.values[0]}")
    print(f"Worst genome in last generation: {top_genomes[0]}")
    print(f"Worst genome in last generation fitness: {top_genomes[0].fitness.values[0]}")
    print(f"Time taken for evolution: {t1 - t0:.2f} seconds\n")
    print("Done with evolution!")

    bell()  # non-blocking

    # Plots
    generate_plots(
        log_dir=ctx.log_dir,
        top_genomes=top_genomes,
        checkpoint_stats=checkpoint_stats,
        mean_fitnesses=mean_fitnesses,
        worst_fitnesses=worst_fitnesses,
        checkpoint_populations=checkpoint_populations,
    )

    # Training summaries + videos
    print("\n--- Top Genome Training Summary ---")
    for i, mc in enumerate(training_map_configs):
        video_out = ctx.log_dir / f"training_map_{i + 1}_swarm.mp4"
        _, stat = run_simulation(best_genome, mc, visualize=True, video_path=video_out)
        print_simulation_summary(stat, i)

    return best_genome, top_genomes


def run_testing(ctx: RunContext, best_genome, *, repeated: bool) -> None:
    print("\nTesting enabled. Running best genome on unseen maps...\n")

    if not repeated:
        test_map_configs = generate_n_map_configs(NUM_TESTING_CONFIGS)
        for i, mc in enumerate(test_map_configs):
            video_out = ctx.log_dir / f"test_map_{i + 1}_swarm.mp4"
            fitness, summary = run_simulation(
                best_genome, mc, visualize=True, video_path=video_out
            )
            print_simulation_summary(summary, i)
            print(f"   Test fitness: {fitness:.4f}")
        return

    # Repeated testing loop; exit with Ctrl-C
    print("Repeated testing enabled. Press Ctrl+C after a sim window closes to stop...\n")
    i = 0
    try:
        while True:
            mc = generate_n_map_configs(1)[0]
            video_out = ctx.log_dir / f"test_map_{i + 1}_swarm.mp4"
            fitness, _ = run_simulation(
                best_genome, mc, visualize=True, video_path=video_out
            )
            print(f"   Test fitness: {fitness:.4f}")
            i += 1
    except KeyboardInterrupt:
        print("\nEarly exit requested. Stopping repeated testing.")


# ----------------------------
# Main
# ----------------------------

def main() -> None:
    # Prepare run context & redirect stdout to a file (keeps console clean)
    ctx = make_run_context()
    previous_best = TOP_GENOME

    with open(ctx.output_log_path, "w") as fout, contextlib.redirect_stdout(fout):
        # graceful Ctrl-C for the whole run
        def _sigint_handler(sig, frame):
            print("\n[Signal] Ctrl-C received. Attempting graceful shutdown...", file=sys.stderr)
            raise KeyboardInterrupt

        old_handler = signal.signal(signal.SIGINT, _sigint_handler)
        try:
            best_genome = previous_best
            if EVOLUTION:
                best_genome, _ = run_evolution(ctx, previous_genomes=[TOP_GENOME])

            if ENABLE_TESTING:
                run_testing(ctx, best_genome, repeated=REPEATED_TESTING)
        except KeyboardInterrupt:
            # Make sure we flush and leave logs in a sane state
            print("\n[Main] Interrupted by user. Exiting cleanly.")
        finally:
            signal.signal(signal.SIGINT, old_handler)  # restore

if __name__ == "__main__":
    import multiprocessing
    multiprocessing.set_start_method("spawn", force=True)
    main()
