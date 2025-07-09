import random
import numpy as np
from simulation.run_sim import run_simulation
from configs import (GA_POPULATION_SIZE, GA_GENERATIONS, GENOME_RANGE, 
                    NUM_ELITES, TOURNAMENT_GROUP_SIZE, RANDOM_SEED) 

def run_genetic_algorithm(map_configs: list,
                          generations: int = GA_GENERATIONS,
                          population_size: int = GA_POPULATION_SIZE,
                          genome_range: tuple[int, int] = GENOME_RANGE,
                          num_elites: int = NUM_ELITES,
                          tournament_group_size: int = TOURNAMENT_GROUP_SIZE,
                          random_seed: int = RANDOM_SEED,
                         ) -> object:
    
    # Randomly Generate Initial Genome Population
    genome_population = generate_genomes(population_size, genome_range)

    # Run Genetic Algorithm for a fixed number of generations
    for _ in range(generations):

        # Run simulation for each genome across all map configurations, recording fitness scores
        for genome in genome_population:
            fitness_scores = [
                run_simulation(genome, map_config, Visualize=False) for map_config in map_configs
            ]
            genome.fitness = np.mean(fitness_scores)

        # Choose Elites to keep unchanged in the next generation
        elites = [select_k_elites(genome_population, k = NUM_ELITES)]
        
        # Choose parents through tournament selection
        num_tournaments = population_size - num_elites
        parents = [tournament_selection(genome_population, k=tournament_group_size) for _ in range(num_tournaments)] 

        # Generate offspring through crossover and mutation
        random.seed(random_seed)  # For reproducibility
        random.shuffle(parents)
        offspring = []

        for i in range(0, len(parents), 2):
            child1, child2 = crossover(parents[i], parents[i + 1])
            child1 = mutate(child1)
            child2 = mutate(child2)
            offspring.extend([child1, child2])

        # Next generation
        genome_population = elites + offspring

    # Choose best genome from the final population
    return min(genome_population, key=lambda g: g.fitness)


def generate_genomes(population_size: int, genome_range: tuple[int, int]) -> list:
    pass


def select_k_elites(genome_population: list, k: int) -> list:
    pass

def tournament_selection(genome_population: list, k: int) -> object:
    pass

def crossover(parent1, parent2) -> tuple:
    pass

def mutate(genome) -> object:
    pass