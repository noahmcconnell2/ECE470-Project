"""
Plotting utilities for Swarm Behaviour Evolution analysis.

This module contains reusable functions for visualizing:
- Fitness convergence trends
- Gene evolution over generations
- Population variance of genes and simulation metrics
- Correlation between gene weights and performance metrics

All plots optionally save to a timestamped log directory for organized analysis.

Author:
    @lexph (with assistance from OpenAI's ChatGPT, July 2025)
"""

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import re

def _safe_filename(s):
    """Sanitize a string to be safe for filesystem naming."""
    return re.sub(r"[^\w\-_.]", "_", s)

def _save_or_show(fig, plot_name: str, save: bool, log_dir: Path):
    """
    Saves the plot to disk if `save` is True, otherwise shows it interactively.

    Args:
        fig: matplotlib figure
        plot_name: name of the file (without extension)
        save: whether to save or show
        log_dir: path to output directory if saving
    """
    if save:
        filename = _safe_filename(plot_name)
        path = log_dir / f"{filename}.png"
        fig.savefig(path, bbox_inches='tight')
        print(f"Saved plot to {path}")
    else:
        fig.show()
    plt.close(fig)

def plot_fitness_convergence_band(top_genomes, mean_fitnesses, worst_fitnesses, save=False, log_dir=None):
    """
    Plots best, mean, and worst fitness values across generations.

    Args:
        top_genomes: list of top-performing genomes per generation
        mean_fitnesses: mean fitness values per generation
        worst_fitnesses: worst fitness values per generation
        save: if True, save plot to log_dir
        log_dir: directory to save plots
    """
    best_fitnesses = [g.fitness.values[0] for g in top_genomes]
    generations = range(len(best_fitnesses))

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(generations, best_fitnesses, label="Best", color="green")
    ax.plot(generations, mean_fitnesses, label="Mean", color="blue")
    ax.plot(generations, worst_fitnesses, label="Worst", color="red", linestyle="--")
    ax.fill_between(generations, best_fitnesses, worst_fitnesses, alpha=0.1, color="gray")

    ax.set_xlabel("Generation")
    ax.set_ylabel("Fitness")
    ax.set_title("Fitness Convergence Over Generations")
    ax.legend()
    ax.grid(True)

    _save_or_show(fig, "fitness_convergence_band", save, log_dir)

def plot_gene_evolution(top_genomes, save=False, log_dir=None):
    """
    Plots evolution of each gene's value across generations.

    Args:
        top_genomes: list of top-performing genomes per generation
        save: if True, save plot to log_dir
        log_dir: directory to save plots
    """
    generations = range(len(top_genomes))
    fig, ax = plt.subplots(figsize=(10, 6))

    for i in range(len(top_genomes[0])):
        gene_vals = [genome[i] for genome in top_genomes]
        ax.plot(generations, gene_vals, label=f"Gene {i+1}")

    ax.set_xlabel("Generation")
    ax.set_ylabel("Gene Value")
    ax.set_title("Gene Evolution Over Generations")
    ax.legend()
    ax.grid(True)

    _save_or_show(fig, "gene_evolution", save, log_dir)


def plot_sim_metrics_separate_boxplots(checkpoint_stats, save=False, log_dir=None):
    """
    Creates a separate box plot for each simulation metric across start, mid, and end generations.

    Parameters:
        checkpoint_stats (dict): Dictionary of metrics for each generation tag.
        save (bool): If True, saves each plot to `log_dir`.
        log_dir (Path): Directory to save plots if `save` is True.
    """
    import seaborn as sns
    import pandas as pd

    metrics = ["leader_distance", "path_distance", "obstacle_collisions", "agent_collisions", "step_count"]

    for metric in metrics:
        data = []
        for tag in checkpoint_stats:
            values = checkpoint_stats[tag][metric]
            for v in values:
                data.append({"Generation": tag, "Value": v})

        df = pd.DataFrame(data)
        fig, ax = plt.subplots(figsize=(7, 5))
        sns.boxplot(x="Generation", y="Value", data=df, ax=ax)
        ax.set_title(f"{metric.replace('_', ' ').title()} Across Generations")
        ax.set_ylabel(metric.replace('_', ' ').title())
        ax.set_xlabel("Generation")

        _save_or_show(fig, f"boxplot_{metric}", save, log_dir)


def plot_metric_vs_gene(top_genomes, gene_idx, metric_key, title, x_label, y_label, save=False, log_dir=None):
    """
    Scatter plot of a specific simulation metric vs a specific gene weight.

    Args:
        top_genomes: list of top-performing genomes
        gene_idx: index of the gene to use on x-axis
        metric_key: stat key to plot on y-axis (e.g., "avg_leader_distance")
        title: title of the plot
        x_label: label for the x-axis
        y_label: label for the y-axis
        save: if True, save plot to log_dir
        log_dir: directory to save plots
    """
    gene_vals = [g[gene_idx] for g in top_genomes]
    metric_vals = [g.stats[metric_key] for g in top_genomes if g.stats is not None]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(gene_vals, metric_vals, alpha=0.7)

    # Optional trend line
    try:
        from scipy.stats import linregress
        slope, intercept, *_ = linregress(gene_vals, metric_vals)
        ax.plot(gene_vals, [slope * x + intercept for x in gene_vals], color='orange', label='Trend')
        ax.legend()
    except Exception:
        pass

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.grid(True)

    filename = f"{metric_key}_vs_gene{gene_idx+1}"
    _save_or_show(fig, filename, save, log_dir)


def plot_checkpoint_population_gene_boxplots(checkpoint_populations, top_genomes, save=False, log_dir=None):
    """
    Creates a stacked image of gene variance in the population at 3 key generations (start, mid, end),
    using boxplots. Highlights the top genome gene values at each checkpoint for reference.

    Args:
        checkpoint_populations: dict of generation_index -> list of genome lists
        top_genomes: list of top genomes per generation (indexed by generation)
        save: if True, saves the plot to log_dir
        log_dir: directory to save output
    """
    import seaborn as sns
    import pandas as pd
    import matplotlib.pyplot as plt

    generation_tags = {
        "start": "Start Generation",
        "mid": "Mid Generation",
        "end": "Final Generation"
    }

    fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(10, 12), sharex=True)

    for idx, gen in enumerate(["start", "mid", "end"]):
        data = []
        population = checkpoint_populations[gen]

        for genome in population:
            for i, gene_val in enumerate(genome):
                data.append({"Gene": f"Gene {i+1}", "Value": gene_val})

        df = pd.DataFrame(data)
        ax = axes[idx]
        sns.boxplot(x="Gene", y="Value", data=df, ax=ax, color='skyblue')
        ax.set_title(generation_tags.get(gen, f"Generation {gen}"))
        ax.grid(True)

        # Add red scatter points for top genome
        top_genome = top_genomes[gen]  # Uses actual generation number to index directly
        ax.scatter(
            [i for i in range(len(top_genome))],
            [top_genome[i] for i in range(len(top_genome))],
            color='red', label='Top Genome', zorder=5, marker='o'
        )
        ax.legend(loc='upper right')

    fig.suptitle("Population Gene Variance at Key Generations", fontsize=14)
    fig.tight_layout(rect=[0, 0.03, 1, 0.97])

    _save_or_show(fig, "checkpoint_population_gene_boxplots", save, log_dir)
