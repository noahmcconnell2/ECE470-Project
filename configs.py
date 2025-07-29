# configs.py
from typing import NamedTuple
import numpy as np

# ===============================
# Train/Test Configurations
# ===============================
NUM_TRAINING_CONFIGS = 1    # Number of training maps used for fitness evaluation
NUM_TESTING_CONFIGS = 3     # Number of testing maps used for post-training evaluation

# ===============================
# Grid and Environment Settings
# ===============================
GRID_DIM = (50, 50)         # Size of the 2D environment grid (rows, columns)
PERCENT_OBSTACLES = 0.15    # Fraction of grid cells to fill with obstacles
ENTRANCE_SIZE = 3           # Width of openings in obstacle funnels
MIN_LEADER_PATH_DISTANCE = int(GRID_DIM[0] * 0.5)   # Minimum path length the leader must travel (used to prevent trivial routes)

# ===============================
# Agent Settings
# ===============================
NUM_AGENTS = 5          # Number of follower agents spawned per simulation
PERCEPTION_RANGE = 5    # Agents can sense other entities within this Manhattan radius


# ===============================
# Movement Settings
# ===============================
ISOLATION_PENALTY = round((np.sqrt(2 * PERCEPTION_RANGE**2) / 2) + 4, 2)    # Penalty for agents that are isolated from group — encourages staying close
                                                                            # Formula is derived from half-diagonal of perception square plus offset
MAX_DISTANCE = np.sqrt(GRID_DIM[0]**2 + GRID_DIM[1]**2)     # Used to normalize distances, represents the longest possible distance in the grid
SEDENTARY_PENALTY = 0.4     # Penalty applied if agent does not move during a tick
MAX_OSC_PENALTY = 20        # Maximum penalty for excessive oscillation (frequent back-and-forth moves)
POST_GOAL_BUFFER_STEPS = 10  # Time buffer after leader reaches goal to let followers regroup before simulation ends
FOLLOWER_STAGGER_INTERVAL = 1  # Interval (in ticks) between spawning successive followers


# ===============================
# Genetic Algorithm Settings
# ===============================
GA_POPULATION_SIZE = 100     # Number of genomes in the population per generation
GA_GENERATIONS = 35          # Total number of generations for evolution
GENOME_RANGE = (0.0, 5.0)    # Allowed gene weight values (inclusive)
NUM_ELITES = 2               # Number of top genomes preserved without modification in next generation
TOURNAMENT_GROUP_SIZE = 3    # Size of tournament selection pool
RANDOM_SEED = 42             # Seed for reproducibility

# --- Cross-over and mutation parameters ---
ETA = 8        # SBX crossover parameter; controls spread of offspring (smaller = more exploratory)
MU = 0.0       # Mean for Gaussian mutation; centered around current gene value
SIGMA = 0.25    # Std dev for Gaussian mutation; controls magnitude of mutation
INDPB = 0.3    # Probability of mutating each individual gene within a genome
CXPB = 0.6     # Probability that an offspring is generated via crossover
MUTPB = 0.4    # Probability that an offspring is mutated (may override crossover child)

INTRODUCE_RANDOMS = True                       # Flag for injecting random genomes into populous
K_RANDOMS = int(0.01 * GA_POPULATION_SIZE)      # Baseline number of random genomes injected each generation
K_MAX = int(0.15 * GA_POPULATION_SIZE)           # Max number of random genomes injected due to stagnation
EPSILON = 1e-2                                  # Minimum required improvement to reset stagnation counter

ENABLE_VISUALIZATION = False    # Flag to enable Pygame animation of individual evaluations
TILE_SIZE = 20                  # Pixel size of grid cells when visualizing

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

FITNESS_WEIGHTS = FitnessWeights(1.5, 1.5, 1.0, 1.0)
MAX_COLLISIONS = 1.0
MAX_FITNESS = 1e6  # Arbitrary large value for no followers

# ===============================
# Testing Settings
# ===============================
ENABLE_TESTING = False