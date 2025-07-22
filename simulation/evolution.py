import random
import numpy as np
from simulation.run_sim import run_simulation
from deap import base, creator, tools, algorithms
from functools import partial
from configs import (GA_POPULATION_SIZE, GA_GENERATIONS, GENOME_RANGE, 
                    NUM_ELITES, TOURNAMENT_GROUP_SIZE, RANDOM_SEED, 
                    ETA, MU, SIGMA, INDPB, VISUALIZATION_PLAN, ENABLE_VISUALIZATION) 
import multiprocessing

def run_genetic_algorithm(map_configs: list,
                          generations: int = GA_GENERATIONS,
                          population_size: int = GA_POPULATION_SIZE,
                          genome_range: tuple[int, int] = GENOME_RANGE,
                          num_elites: int = NUM_ELITES,
                          tournament_group_size: int = TOURNAMENT_GROUP_SIZE,
                          random_seed: int = RANDOM_SEED,
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
    population = initialize_and_evaluate_population(toolbox, population_size)  
  

    for gen in range(generations):
        pre_var_best = tools.selBest(population, k=1)[0]

        # Variation
        offspring = algorithms.varAnd(population, toolbox, cxpb=0.5, mutpb=0.2)

        # Evaluate new individuals
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # Select elites from current population
        elites = tools.selBest(population, k=num_elites)

        # Select rest from offspring
        rest = toolbox.select(offspring, k=population_size - num_elites)

        # Combine elites and selected offspring
        population = elites + rest
        
        post_var_best = tools.selBest(population, k=1)[0]

        # Optional visualization
        visualize_at_generation(gen, population, map_configs)

    pool.close()
    pool.join()

    return tools.selBest(population, k=1)[0]



def initialize_and_evaluate_population(toolbox, population_size):
    """Initializes a population and evaluates their fitness."""
    population = toolbox.population(n=population_size)
    invalid_ind = [ind for ind in population if not ind.fitness.valid]
    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit
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

    return (avg_fitness,)


def visualize_at_generation(gen, population, map_configs):
    if not ENABLE_VISUALIZATION:
        return

    plan = VISUALIZATION_PLAN.get(gen, [])
    for ind_rank, map_idx in plan:
        if ind_rank < len(population) and map_idx < len(map_configs):
            genome = tools.selBest(population, k=ind_rank + 1)[ind_rank]
            print(f"Visualizing gen {gen}, rank {ind_rank} on map {map_idx}")
            run_simulation(genome, map_configs[map_idx], visualize=True)


