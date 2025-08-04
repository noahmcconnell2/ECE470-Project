"""
Fitness evaluation module for swarm agents.

This module defines the function used to compute the fitness of follower agents
based on their stepwise performance metrics during simulation. Metrics are normalized
and combined via a weighted sum to reflect desired swarm behavior traits.
"""


import numpy as np
from agent.agent import Agent
from typing import List
from configs import FitnessWeights, FITNESS_WEIGHTS, MAX_DISTANCE, MAX_COLLISIONS, MAX_FITNESS


def calculate_fitness(followers: list[Agent], weights: FitnessWeights = FITNESS_WEIGHTS) -> float:
    """
    Calculate the average fitness score for a list of follower agents.

    The fitness function is a weighted sum of four normalized performance metrics:
    - Average distance to the leader
    - Average distance to the leader's path
    - Obstacle collision rate
    - Agent collision rate

    All metrics are normalized to a [0, 1] range based on known maximums.
    A lower score indicates better performance.

    Args:
        followers (List[Agent]): List of follower agents with simulation metrics.
        weights (FitnessWeights): Weights assigned to each of the four metrics.

    Returns:
        float: Average fitness score across all agents. If no agents are present,
               returns MAX_FITNESS as a penalty.
    """
    total_fitness = 0.0

    # Normalize metrics for each agent
    for agent in followers:
        per_step_metrics = [
            (agent.leader_distance_sum / agent.step_count) / MAX_DISTANCE,
            (agent.path_distance_sum / agent.step_count) / MAX_DISTANCE,
            (agent.obstacle_collision_count / agent.step_count) / MAX_COLLISIONS,
            (agent.agent_collision_count / agent.step_count) / MAX_COLLISIONS
        ]
        
        # Calculate weighted sum of metrics
        fitness_score = sum(w * m for w, m in zip(weights, per_step_metrics))
        total_fitness += fitness_score
    
    return total_fitness / len(followers) if followers else MAX_FITNESS # Return a high fitness score if no followers are present