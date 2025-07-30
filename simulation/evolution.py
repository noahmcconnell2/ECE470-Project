import random
import numpy as np
from simulation.run_sim import run_simulation
from deap import base, creator, tools, algorithms
from functools import partial
from typing import List
from configs import (GA_POPULATION_SIZE, GA_GENERATIONS, GENOME_RANGE, 
                    NUM_ELITES, TOURNAMENT_GROUP_SIZE, RANDOM_SEED, 
                    ETA, MU, SIGMA, INDPB, K_RANDOMS, K_MAX, EPSILON, CXPB, MUTPB, INTRODUCE_RANDOMS,
                    VISUALIZATION_PLAN, ENABLE_VISUALIZATION) 
import multiprocessing

def run_genetic_algorithm(map_configs: list,
                          generations: int = GA_GENERATIONS,
                          population_size: int = GA_POPULATION_SIZE,
                          genome_range: tuple[int, int] = GENOME_RANGE,
                          num_elites: int = NUM_ELITES,
                          tournament_group_size: int = TOURNAMENT_GROUP_SIZE,
                          random_seed: int = RANDOM_SEED,
                          previous_genomes: List[tuple] = None
                         ) -> object:

    random.seed(random_seed)
    np.random.seed(random_seed)

    # Create DEAP classes for fitness and genome representation
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))  # Minimize fitness
    creator.create("Genome", list, fitness=creator.FitnessMin)

    # Register DEAP toolbox functions
    toolbox = base.Toolbox()
    toolbox.register("attr_float", random.uniform, genome_range[0], genome_range[1])        # Random float in our genome range
    toolbox.register("genome", tools.initRepeat, creator.Genome, toolbox.attr_float, n=6)   # Genome contains 6 genes
    toolbox.register("population", tools.initRepeat, list, toolbox.genome)                  # Population is a list of genomes
    toolbox.register("mate", tools.cxSimulatedBinaryBounded, eta=ETA, low=genome_range[0], up=genome_range[1])  # Crossover operator
    toolbox.register(
        "mutate",
        make_clamped_gaussian_mutation(mu=MU, sigma=SIGMA, indpb=INDPB, low=genome_range[0], up=genome_range[1])
    )
    toolbox.register("select", tools.selTournament, tournsize=tournament_group_size)  # Selection operator
    toolbox.register("evaluate", partial(evaluate_genome, map_configs=map_configs, visualize=False))  # Evaluation function

    pool = multiprocessing.Pool()
    toolbox.register("map", pool.map)

    # Initialize population and evaluate fitness - starting genomes
    population = initialize_and_evaluate_population(toolbox, population_size, previous_genomes)

    top_genomes = []
    mean_fitnesses = []
    worst_fitnesses = []
    
    checkpoint_tags = {
        "start": 0,
        "mid": generations // 2,
        "end": generations - 1
    }
    checkpoint_stats = {
        tag: {
            "leader_distance": [],
            "path_distance": [],
            "obstacle_collisions": [],
            "agent_collisions": [],
            "step_count": []
        } for tag in checkpoint_tags
    }
    checkpoint_populations = {tag: [] for tag in checkpoint_tags}
    no_improvement_counter = 0
    best_so_far = float('inf')
  

    for gen in range(generations):
        pre_var_best = tools.selBest(population, k=1)[0]

        # === VARIATION ===
        offspring = algorithms.varAnd(population, toolbox, cxpb=CXPB, mutpb=MUTPB)

        # === EVALUATE new offspring ===
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        results = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, (fit, summary) in zip(invalid_ind, results):
            ind.fitness.values = (fit,)
            ind.stats = summary

        # === ELITISM: keep top N elites from previous generation ===
        elites = tools.selBest(population, k=num_elites)

        
        if INTRODUCE_RANDOMS:
            # === STAGNATION TRACKING ===
            # if mean_fitnesses and mean_fitnesses[-1] >= best_so_far - EPSILON:
            #     no_improvement_counter += 1
            # else:
            #     best_so_far = mean_fitnesses[-1] if mean_fitnesses else float('inf')
            #     no_improvement_counter = 0

            # === HYBRID INJECTION STRATEGY ===
            # 1. Linearly decaying injection amount over generations
            scheduled_k = int(K_MAX * (1 - gen / generations)**1.2)

            # 2. Stagnation-triggered fallback
            stagnation_k = K_RANDOMS + no_improvement_counter

            # 3. Final k: max of scheduled and stagnation amounts (capped)
            k = min(max(scheduled_k, stagnation_k), K_MAX)
            k = max(k, K_RANDOMS)  # Enforce minimum baseline

            # === Create and evaluate k random genomes ===
            random_genomes = [toolbox.genome() for _ in range(k)]
            results = toolbox.map(toolbox.evaluate, random_genomes)
            for ind, (fit, summary) in zip(random_genomes, results):
                ind.fitness.values = (fit,)
                ind.stats = summary

            # === Replace k worst individuals in offspring with random genomes ===
            k = min(k, len(offspring) - 1) if len(offspring) > 1 else 0
            offspring_sorted = sorted(offspring, key=lambda ind: ind.fitness.values[0], reverse=True)
            preserved = offspring_sorted[:-k] if k > 0 else offspring_sorted
            offspring = preserved + random_genomes

        # === TOURNAMENT SELECTION: Fill remainder of population excluding elites ===
        available = len(offspring)
        needed = population_size - num_elites

        if available < needed:
            print(f"[Warning] Only {available} offspring available, needed {needed}.")
            needed = available  # Prevent crash if we’re short

        rest = toolbox.select(offspring, k=needed)

        # === POPULATION UPDATE ===
        population = elites + rest

        # === TRACKING ===
        fitnesses = [ind.fitness.values[0] for ind in population]
        mean_fitnesses.append(np.mean(fitnesses))
        worst_fitnesses.append(max(fitnesses))
        top_genomes.append(tools.selBest(population, k=1)[0])  

        # === VISUALIZATION ===
        visualize_at_generation(gen, population, map_configs)

        # === LOG CHECKPOINT DATA ===
        for tag, gnum in checkpoint_tags.items():
            if gen == gnum:
                log_checkpoint_stats(tag, population, checkpoint_stats, checkpoint_populations)


    pool.close()
    pool.join()

    return top_genomes, checkpoint_stats, mean_fitnesses, worst_fitnesses, checkpoint_populations


def log_checkpoint_stats(tag, population, checkpoint_stats, checkpoint_populations):
    for ind in population:
        # Track sim stats
        if hasattr(ind, "stats") and ind.stats:
            checkpoint_stats[tag]["leader_distance"].append(ind.stats["avg_leader_distance"])
            checkpoint_stats[tag]["path_distance"].append(ind.stats["avg_path_distance"])
            checkpoint_stats[tag]["obstacle_collisions"].append(ind.stats["avg_obstacle_collisions"])
            checkpoint_stats[tag]["agent_collisions"].append(ind.stats["avg_agent_collisions"])
            checkpoint_stats[tag]["step_count"].append(ind.stats["avg_step_count"])

        # Track gene values
        checkpoint_populations[tag].append(list(ind))  # Shallow copy of 6 gene values



def initialize_and_evaluate_population(toolbox, population_size, previous_genomes=None):
    """Initializes a population and evaluates their fitness."""
    population = toolbox.population(n=population_size)

    # Inject previous genomes if provided
    if previous_genomes:
        for i, prev in enumerate(previous_genomes):
            genome = creator.Genome(prev)
            population[i % population_size] = genome

    # Evaluate the population        
    invalid_ind = [ind for ind in population if not ind.fitness.valid]
    results = toolbox.map(toolbox.evaluate, invalid_ind)
    for ind, (fit, summary) in zip(invalid_ind, results):
        ind.fitness.values = (fit,)
        ind.stats = summary
    return population



def make_clamped_gaussian_mutation(mu, sigma, indpb, low, up):
    def mutate(genome):
        tools.mutGaussian(genome, mu=mu, sigma=sigma, indpb=indpb)
        for i in range(len(genome)):
            genome[i] = min(max(genome[i], low), up)
        return (genome,)
    return mutate


def evaluate_genome(genome, map_configs, visualize=False):
    
    results = [
        run_simulation(genome, map_config, visualize=visualize)
        for map_config in map_configs
    ]

    fitness_scores = [fit for fit, _ in results]
    summaries = [summary for _, summary in results]

    genome.stats = summaries

    avg_fitness = np.mean(fitness_scores)
    condensed_summary = create_condensed_summary(avg_fitness, summaries)
    return (avg_fitness, condensed_summary)


def create_condensed_summary(avg_fitness, summaries):
    num_maps = len(summaries)
    num_followers_total = 0

    # Accumulators
    step_count_sum = 0
    leader_dist_sum = 0
    path_dist_sum = 0
    obstacle_collisions = 0
    agent_collisions = 0

    for summary in summaries:
        followers = summary["agents"]
        num_followers = len(followers)
        num_followers_total += num_followers

        for agent in followers:
            step_count_sum += agent["step_count"]
            leader_dist_sum += agent["leader_distance_sum"]
            path_dist_sum += agent["path_distance_sum"]
            obstacle_collisions += agent["obstacle_collision_count"]
            agent_collisions += agent["agent_collision_count"]

    if num_followers_total == 0:
        return None  # Prevent division by zero

    return {
        "fitness": avg_fitness,
        "avg_step_count": step_count_sum / num_followers_total,
        "avg_leader_distance": leader_dist_sum / num_followers_total,
        "avg_path_distance": path_dist_sum / num_followers_total,
        "avg_obstacle_collisions": obstacle_collisions / num_followers_total,
        "avg_agent_collisions": agent_collisions / num_followers_total,
    }


def visualize_at_generation(gen, population, map_configs):
    if not ENABLE_VISUALIZATION:
        return

    plan = VISUALIZATION_PLAN.get(gen, [])
    for ind_rank, map_idx in plan:
        if ind_rank < len(population) and map_idx < len(map_configs):
            genome = tools.selBest(population, k=ind_rank + 1)[ind_rank]
            print(f"Visualizing gen {gen}, rank {ind_rank} on map {map_idx}")
            run_simulation(genome, map_configs[map_idx], visualize=True)


