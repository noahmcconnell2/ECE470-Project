import numpy as np
from agent.agent import Agent
from typing import List
from configs import FitnessWeights, FITNESS_WEIGHTS, GRID_DIM


def calculate_fitness(followers: list[Agent], weights: FitnessWeights = FITNESS_WEIGHTS) -> float:
    """
    Calculate the fitness scores for a list of agents based on their metrics and given weights.
    
    Args:
        followers: List of Agent objects representing the followers.
        weights: List of weights corresponding to each metric.
        
    Returns:
        float: The average fitness score across all agents.
    """
    total_fitness = 0.0

    # Per step metric maximums
    w, h = GRID_DIM
    max_distance = np.sqrt(w**2 + h**2)  # Maximum distance in the grid
    max_collisions = 1.0

    # Normalize metrics for each agent
    for agent in followers:
        per_step_metrics = [
            (agent.leader_distance_sum / agent.step_count) / max_distance,
            (agent.path_distance_sum / agent.step_count) / max_distance,
            (agent.obstacle_collision_count / agent.step_count) / max_collisions,
            (agent.agent_collision_count / agent.step_count) / max_collisions
        ]
        
        # Calculate weighted sum of metrics
        fitness_score = sum(w * m for w, m in zip(weights, per_step_metrics))
        total_fitness += fitness_score
    
    return total_fitness / len(followers) if followers else 1e6 # Return a high fitness score if no followers are present