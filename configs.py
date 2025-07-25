# configs.py
from typing import NamedTuple
import numpy as np

# ===============================
# Train/Test Configurations
# ===============================
NUM_TRAINING_CONFIGS = 5
NUM_TESTING_CONFIGS = 1

# ===============================
# Grid and Environment Settings
# ===============================
GRID_DIM = (50, 50)
PERCENT_OBSTACLES = 0.20
ENTRANCE_SIZE = 3
MIN_LEADER_PATH_DISTANCE = int(GRID_DIM[0] * 0.2)

# ===============================
# Agent Settings
# ===============================
NUM_AGENTS = 5
PERCEPTION_RANGE = 5


# ===============================
# Movement Settings
# ===============================
ISOLATION_PENALTY = round((np.sqrt(2 * PERCEPTION_RANGE**2) / 2) + 4, 2)
MAX_DISTANCE = np.sqrt(GRID_DIM[0]**2 + GRID_DIM[1]**2)
SEDENTARY_PENALTY = 0.4  # Penalty for not moving
POST_GOAL_BUFFER_STEPS = 8  # Extra steps after leader reaches goal to allow followers to catch up
FOLLOWER_STAGGER_INTERVAL = 2  # Only spawn a new follower every 2 ticks


# ===============================
# Genetic Algorithm Settings
# ===============================
GA_POPULATION_SIZE = 300
GA_GENERATIONS = 60
GENOME_RANGE = (0.0, 5.0)
NUM_ELITES = 1
TOURNAMENT_GROUP_SIZE = 3
RANDOM_SEED = 42
# --- Cross-over and mutation parameters ---
ETA = 7       # Crossover parameter, balanced between exploration and exploitation
MU = 0.0       # Mean for Gaussian mutation
SIGMA = 2.0    # Standard deviation for Gaussian mutation
INDPB = 0.6    # Independent probability for mutation
K_RANDOMS = int(0.01 * GA_POPULATION_SIZE)
K_MAX = int(0.3 * GA_POPULATION_SIZE)
EPSILON = 1e-2
ENABLE_VISUALIZATION = False
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

FITNESS_WEIGHTS = FitnessWeights(1.0, 1.0, 1.0, 1.0)
MAX_COLLISIONS = 1.0
MAX_FITNESS = 1e6  # Arbitrary large value for no followers

# ===============================
# Testing Settings
# ===============================
ENABLE_TESTING = False