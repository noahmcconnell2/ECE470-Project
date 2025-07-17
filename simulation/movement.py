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

import numpy as np
from agent.agent import Agent
from typing import List, Tuple
from map.map_structures import MapConfig
from map.grid_utils import FREE
from configs import ISOLATION_PENALTY, MAX_DISTANCE


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
        # Get agents in perception range as well as the nearest agent
        nearby_agents, nearest_agent = get_agents_in_perception(move, agent.perception_range, map_config)

        # Calculate normalized feature costs
        cohesion   = calculate_cohesion(move, nearby_agents)
        separation = calculate_separation(move, nearest_agent)
        avoidance  = calculate_obstacle_avoidance(move, map_config)
        path_follow = calculate_path_following(move, map_config)
        leader_dist = calculate_leader_distance(move, leader.position)
        alignment   = calculate_alignment(agent.position, move, nearby_agents)

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
    return [
        move for move in grid.get_neighborhood(position)
        if grid.get(move) == FREE
    ]



def get_agents_in_perception(agent: Agent, move: Tuple[int, int], percept_range: int, map_config: MapConfig) -> Tuple[List[Agent], Agent]:
    """
    Get agents within the perception range of the agent at the specified move position.
    Args:
        agent (Agent): The agent for which to find nearby agents.
        move (Tuple[int, int]): The intended move position of the agent.
        percept_range (int): The perception range of the agent.
        map_config (MapConfig): The current map configuration.
    Returns:
        Tuple[List[Agent], Agent]: A list of nearby agents and the nearest agent within perception range.
    """
    nearby_agents = []
    nearest_agent = None
    min_distance = float('inf')
    pw = percept_range // 2
    x_range = (move[0] - pw, move[0] + pw)
    y_range = (move[1] - pw, move[1] + pw)

    for nearby_agent in map_config.agent_index.values():
        if nearby_agent is not None and nearby_agent != agent:
            nax, nay = nearby_agent.position
            if x_range[0] <= nax <= x_range[1] and y_range[0] <= nay <= y_range[1]:
                distance_to_nearby_agent = np.sqrt((nax - agent.position[0]) ** 2 + (nay - agent.position[1]) ** 2)
                if distance_to_nearby_agent < min_distance:
                    min_distance = distance_to_nearby_agent
                    nearest_agent = nearby_agent
                # Add to nearby agents if within perception range
                nearby_agents.append(nearby_agent)
    return nearby_agents, nearest_agent


def normalize_feature(value: float, max_value: float) -> float:
    """
    Normalize a value to [0, 1] given its theoretical maximum.
    Caps output at 1.0 for safety.
    """
    if max_value == 0:
        return 0.0
    return min(value / max_value, 1.0)


def calculate_cohesion(next_move: Tuple[int, int], nearby_agents: List[Agent]) -> float:
    """
    Calculate the cohesion score based on the average position of nearby agents.
    Args:
        next_move (Tuple[int, int]): The intended next move position of the agent.
        nearby_agents (List[Agent]): List of agents within perception range.
    Returns:    
        float: Normalized cohesion score (0 = fully cohesive, 1 = fully isolated).
    """
    if not nearby_agents:
        return 1.0  # Full penalty when isolated

    xs, ys = zip(*(agent.position for agent in nearby_agents))
    mx = np.mean(xs)
    my = np.mean(ys)

    distance = np.sqrt((next_move[0] - mx) ** 2 + (next_move[1] - my) ** 2)
    return normalize_feature(distance, ISOLATION_PENALTY)


def calculate_separation(next_move: Tuple[int, int], nearest_agent: Agent) -> float:
    """
    Calculate the separation score based on the distance to the nearest agent.
    Args:
        next_move (Tuple[int, int]): The intended next move position of the agent.
        nearest_agent (Agent): The nearest agent within perception range.
    Returns:
        float: Normalized separation score (0 = distant, 1 = extremely close).

    1 / normalized_distance
    """
    if not nearest_agent:
        return 1.0  # Full penalty when isolated 
    
    distance = np.sqrt((nearest_agent.position[0] - next_move[0]) ** 2 + (nearest_agent.position[1] - next_move[1]) ** 2)
    normalized_distance = normalize_feature(distance, ISOLATION_PENALTY)

    return 1/ normalized_distance 


def calculate_obstacle_avoidance(next_move: Tuple[int, int], map_config: MapConfig) -> float:
    """
    Calculate the obstacle avoidance score based on the distance to the nearest obstacle.
    Args:
        next_move (Tuple[int, int]): The intended next move position of the agent.
        map_config (MapConfig): The current map configuration containing the obstacle distance map. 
    Returns:    
        float: Normalized obstacle avoidance score (0 = fully avoiding, 1 = about to collide).

    1 / normalized_distance
    """
    normalized_distance = normalize_feature(map_config.obstacle_distance_map.get(next_move), MAX_DISTANCE)

    return 1 / normalized_distance
    

def calculate_path_following(next_move: Tuple[int, int], map_config: MapConfig) -> float:
    """
    Calculate the path-following score based on the distance to the leader's intended path.
    Args:
        next_move (Tuple[int, int]): The intended next move position of the agent.
        map_config (MapConfig): The current map configuration containing the leader path distance map.
    Returns:
        float: Normalized path-following score (0 = fully following, 1 = deviating).
    """
    normalized_distance = normalize_feature(map_config.leader_path_distance_map.get(next_move), MAX_DISTANCE)

    return normalized_distance

def calculate_leader_distance(next_move: Tuple[int, int], leader_position: Tuple[int, int]) -> float:
    """
    Calculate the distance to the leader's position.
    Args:
        next_move (Tuple[int, int]): The intended next move position of the agent.
        leader_position (Tuple[int, int]): The current position of the leader agent.
    Returns:    
        float: Normalized distance to the leader (0 = at leader, 1 = farthest possible).
    """
    distance = np.sqrt((leader_position[0] - next_move[0]) ** 2 + (leader_position[1] - next_move[1]) ** 2)
    normalized_distance = normalize_feature(distance, MAX_DISTANCE)

    return normalized_distance

def calculate_alignment(position: Tuple[int, int], next_move: Tuple[int, int], nearby_agents: List[Agent]) -> float:
    """
    Calculate the alignment score based on the heading direction of nearby agents.
    Args:
        position (Tuple[int, int]): The current position of the agent.
        next_move (Tuple[int, int]): The intended next move position of the agent.
        nearby_agents (List[Agent]): List of agents within perception range.
    Returns:
        float: Normalized alignment score (0 = fully aligned, 1 = fully misaligned).
    """
    if not nearby_agents:
        return 1.0  # Fully misaligned when alone

    # Calculate unit vector of intended movement
    move_vec = np.array([next_move[0] - position[0], next_move[1] - position[1]])
    move_norm = np.linalg.norm(move_vec)
    if move_norm == 0:
        return 1.0  # No movement → undefined alignment
    move_unit = move_vec / move_norm

    # Collect unit heading vectors of nearby agents
    headings = [
        np.array(agent.heading) / np.linalg.norm(agent.heading)
        for agent in nearby_agents if np.linalg.norm(agent.heading) > 0
    ]
    if not headings:
        return 1.0  # No valid heading data

    avg_heading = np.mean(headings, axis=0)
    avg_norm = np.linalg.norm(avg_heading)
    if avg_norm == 0:
        return 1.0  # Net direction cancelled
    avg_unit = avg_heading / avg_norm

    # Compare alignment via cosine similarity
    dot = np.clip(np.dot(move_unit, avg_unit), -1.0, 1.0)

    normalized_alignment = normalize_feature(1.0 - dot, 2.0)  # Normalize to [0, 1]
    
    return normalized_alignment
    


