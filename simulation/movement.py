"""
This module provides movement scoring utilities for agents in a swarm simulation.

Each agent evaluates 9 possible moves (Moore neighborhood + stay) using a weighted sum 
of six normalized behavioral features:

    - Cohesion: Attraction toward the local center of nearby agents
    - Separation: Repulsion from the nearest neighboring agent
    - Obstacle Avoidance: Tendency to avoid nearby obstacles
    - Path-Following: Tendency to follow the leader's intended path
    - Leader Distance: Tendency to stay close to the leader
    - Alignment: Tendency to match the heading direction of nearby agents

Each feature is normalized to the range [0, 1] based on its theoretical maximum 
(e.g., maximum perception radius or grid distance). This ensures no single feature 
dominates the score due to scale differences.

The genome is a list of six non-negative weights (one per feature), typically within 
a bounded range such as [0, 5]:

    genome = [
        cohesion_weight,
        separation_weight,
        obstacle_avoidance_weight,
        path_following_weight,
        leader_distance_weight,
        alignment_weight
    ]

The total score for each move is calculated as:

    score = (cohesion       * weight_1
           + separation      * weight_2
           + obstacle_avoid  * weight_3
           + path_following  * weight_4
           + leader_distance * weight_5
           + alignment       * weight_6)

Score range: [genome_min * 6, genome_max * 6] (e.g., [0, 30] if weights range from 0 to 5).
Lower scores are preferred, representing more desirable moves.
"""

from agent import Agent
from typing import List, Tuple
from map_config import MapConfig


def calculate_best_move(map_config, agent, leader):
    """
    Calculate the best move for an agent based on the weighted score of behavioral features.

    Args:
        map_config (MapConfig): The current map configuration including grid, agent index, and distance maps.
        agent (Agent): The follower agent for which to compute the next move.
        leader (Agent): The leader agent whose path and position are used for reference.

    Returns:
        tuple[int, int]: The (x, y) coordinates of the best next move.
    """

    # Get all valid candidate moves (Moore neighborhood + stay)
    possible_moves = get_valid_moves(agent.position, map_config.grid)

    move_scores = []

    for move in possible_moves:
        # Calculate the heading vector for the move
        heading = (move[0] - agent.position[0], move[1] - agent.position[1])

        # Get agents in perception range
        nearby_agents = get_agents_in_perception(move, agent.perception_range, map_config)

        # Calculate normalized feature costs
        cohesion   = calculate_cohesion(move, nearby_agents)
        separation = calculate_separation(move, nearby_agents)
        avoidance  = calculate_obstacle_avoidance(move, map_config)
        path_follow = calculate_path_following(move, map_config)
        leader_dist = calculate_leader_distance(move, leader.position)
        alignment   = calculate_alignment(move, heading, nearby_agents)

        # Compute total score using genome weights
        total_score = (
            cohesion     * agent.genome[0] +
            separation   * agent.genome[1] +
            avoidance    * agent.genome[2] +
            path_follow  * agent.genome[3] +
            leader_dist  * agent.genome[4] +
            alignment    * agent.genome[5]
        )

        move_scores.append((total_score, move))

    # Choose the move with the lowest total score
    best_move = min(move_scores, key=lambda x: x[0])[1]
    return best_move


def get_valid_moves(position: Tuple[int, int], grid) -> List[Tuple[int, int]]:
    pass

def get_agents_in_perception(move: Tuple[int, int], percept_range: int, map_config: MapConfig) -> List[Agent]:
    pass

def calculate_cohesion(position: Tuple[int, int], nearby_agents: List[Agent]) -> float:
    pass

def calculate_separation(position: Tuple[int, int], nearby_agents: List[Agent]) -> float:
    pass

def calculate_obstacle_avoidance(position: Tuple[int, int], map_config: MapConfig) -> float:
    pass

def calculate_path_following(position: Tuple[int, int], map_config: MapConfig) -> float:
    pass

def calculate_leader_distance(position: Tuple[int, int], leader_position: Tuple[int, int]) -> float:
    pass

def calculate_alignment(position: Tuple[int, int], heading: Tuple[int, int], nearby_agents: List[Agent]) -> float:
    pass


