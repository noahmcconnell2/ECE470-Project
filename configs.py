"""
Configuration settings for the Swarm Behaviour Evolution Simulation.

This module contains all constants used for:
- Training/test map generation
- Agent/environment parameters
- Genetic algorithm settings
- Fitness evaluation weights
- Visualization control
"""

from typing import NamedTuple
import numpy as np

# ===============================
# Train/Test Configurations
# ===============================
EVOLUTION = False
NUM_TRAINING_CONFIGS = 2    # Number of training maps used for fitness evaluation
NUM_TESTING_CONFIGS = 1     # Number of testing maps used for post-training evaluation

# ===============================
# Testing Settings
# ===============================
ENABLE_TESTING = True
REPEATED_TESTING = True

# ===============================
# Grid and Environment Settings
# ===============================
GRID_DIM = (50, 50)         # Size of the 2D environment grid (rows, columns)
PERCENT_OBSTACLES = 0.2    # Fraction of grid cells to fill with obstacles
ENTRANCE_SIZE = 5           # Width of openings in obstacle funnels
MIN_LEADER_PATH_DISTANCE = int(GRID_DIM[0] * 0.5)   # Minimum path length the leader must travel (used to prevent trivial routes)

# ===============================
# Agent Settings
# ===============================
NUM_AGENTS = 30          # Number of follower agents spawned per simulation
PERCEPTION_RANGE = 5    # Agents can sense other entities within this Chebyshev radius


# ===============================
# Movement Settings
# ===============================
ISOLATION_PENALTY = round((np.sqrt(2 * PERCEPTION_RANGE**2) / 2) + 4, 2)    # Penalty for agents that are isolated from group, or leader — encourages staying close
                                                                            # Formula is derived from half-diagonal of perception square plus offset
MAX_DISTANCE = np.sqrt(GRID_DIM[0]**2 + GRID_DIM[1]**2)     # Used to normalize distances, represents the longest possible distance in the grid
SEDENTARY_PENALTY = 0.4     # Penalty applied if agent does not move during a tick
MAX_OSC_PENALTY = 20        # Maximum penalty for excessive oscillation (frequent back-and-forth moves)
POST_GOAL_BUFFER_STEPS = 20  # Time buffer after leader reaches goal to let followers regroup before simulation ends
FOLLOWER_STAGGER_INTERVAL = 1  # Interval (in ticks) between spawning successive followers


# ===============================
# Genetic Algorithm Settings
# ===============================
GA_POPULATION_SIZE = 80      # Number of genomes in the population per generation
GA_GENERATIONS = 70          # Total number of generations for evolution
GENOME_RANGE = (0.0, 5.0)    # Allowed gene weight values (inclusive)
NUM_ELITES = 1               # Number of top genomes preserved without modification in next generation
TOURNAMENT_GROUP_SIZE = 3    # Size of tournament selection pool
RANDOM_SEED = 42             # Seed for reproducibility

# --- Cross-over and mutation parameters ---
ETA = 9        # SBX crossover parameter; controls spread of offspring (smaller = more exploratory)
MU = 0.0       # Mean for Gaussian mutation; centered around current gene value
SIGMA = 0.3    # Std dev for Gaussian mutation; controls magnitude of mutation
INDPB = 0.3    # Probability of mutating each individual gene within a genome
CXPB = 0.6     # Probability that an offspring is generated via crossover
MUTPB = 0.4    # Probability that an offspring is mutated (may override crossover child)

TOP_GENOME = (0.49352261464562763, 3.7783354429176788, 1.2222925721559357e-06, 2.5095838545005735, 3.8697468401329056, 3.481150923791909)
INTRODUCE_RANDOMS = True                       # Flag for injecting random genomes into populous
K_RANDOMS = int(0.01 * GA_POPULATION_SIZE)      # Baseline number of random genomes injected each generation
K_MAX = int(0.15 * GA_POPULATION_SIZE)           # Max number of random genomes injected due to stagnation
EPSILON = 1e-2                                  # Minimum required improvement to reset stagnation counter

RECORD_VIDEO = True             # Flag to enable pygame animation recordings
ENABLE_VISUALIZATION = False    # Flag to enable Pygame animation of individual evaluations
TILE_SIZE = 16                  # Pixel size of grid cells when visualizing

VISUALIZATION_PLAN = {
    # generation: [(individual_rank, map_index), ...]
    0: [(0, 0), (1, 0), (2, 1)],       # Top 3 individuals, different maps
    10: [(0, 0)],
    25: [(2, 0), (2, 1), (2, 2)],
    49: [(0, 0), (4, 1)]
}


# ===============================
# Fitness Settings
# ===============================
class FitnessWeights(NamedTuple):
    """Fitness evaluation weights:
    - leader_dist: Avg distance to leader
    - path_dist: Avg distance to leader path
    - obstacle_collisions: Obstacle collisions per step
    - agent_collisions: Agent collisions per step
    """
    leader_dist: float
    path_dist: float
    obstacle_collisions: float
    agent_collisions: float

FITNESS_WEIGHTS = FitnessWeights(3.0, 2.8, 1.5, 1.5)
MAX_COLLISIONS = 1.0
MAX_FITNESS = 1e6  # Arbitrary large value for no followers

