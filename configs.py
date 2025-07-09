# configs.py
from typing import NamedTuple

# ===============================
# Train/Test Configurations
# ===============================
NUM_TRAINING_CONFIGS = 5
NUM_TESTING_CONFIGS = 1

# ===============================
# Grid and Environment Settings
# ===============================
GRID_DIM = (50, 50)
PERCENT_OBSTACLES = 0.2
ENTRANCE_SIZE = 5
MIN_LEADER_PATH_DISTANCE = int(GRID_DIM[0] * 0.2)

# ===============================
# Agent Settings
# ===============================
NUM_AGENTS = 10
PERCEPTION_RANGE = 5

# ===============================
# Genetic Algorithm Settings
# ===============================
GA_POPULATION_SIZE = 50
GA_GENERATIONS = 100
GENOME_RANGE = (0, 5)
NUM_ELITES = 2
TOURNAMENT_GROUP_SIZE = 3
RANDOM_SEED = 42

# ===============================
# Fitness Weights (Normalized Importance)
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