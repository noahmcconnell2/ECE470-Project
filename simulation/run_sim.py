import numpy as np
from map_config import MapConfig
from agent import Agent, AgentRole
from grid import TileType
from simulation.movement import calculate_best_move
from simulation.fitness import calculate_fitness
from config import NUM_AGENTS, ENTRANCE_SIZE

def run_simulation(genome, map_config: MapConfig) -> float:
    """
    Runs a simulation for a given genome on a specified map configuration.
    Args: 
        genome: The genome to simulate.
        map_config: The map configuration containing the grid, leader path, and obstacle distance map.
    Returns:
        float: The fitness score of the genome after the simulation.
    """

    # List of all agents
    followers = []
    
    # Initialize leader and add to datastructures
    leader = Agent(AgentRole.LEADER, map_config.leader_path[0])
    map_config.update(old_position=None, agent=leader)  # Update agent_index and grid with the leader's initial position

    # Loop until leader reaches goal
    while leader.position != map_config.leader_path[-1]:
        # Move leader along its path
        next_move = map_config.leader_path[leader.step_count + 1]
        old_position = leader.position
        leader.move(next_move)  # Move leader to next position
        map_config.update(old_position, leader) # update agent_index and grid using the next move and leaders current position 
        leader.step_count += 1  # Increment step count for leader

        # Add followers to entrance after first step until NUM_followers are present
        if 0 < len(followers) < NUM_AGENTS - 1:
            map_config.add_followers(ENTRANCE_SIZE, followers, start_position=leader.position, genome=genome)

        # Advance each agent
        for agent in followers:
            if agent.role == AgentRole.FOLLOWER:
                next_move = calculate_best_move(map_config, agent, leader)
                if map_config.grid[next_move] == TileType.EMPTY:
                   old_position = agent.position
                   agent.move(next_move)
                   agent.update_heading(next_move, old_position)  # Update heading based on the new position
                   map_config.update(old_position, agent)  # Update agent_index and grid with the new position

                elif map_config.grid[next_move] == TileType.OBSTACLE:
                    agent.obstacle_collision_count += 1
                
                elif map_config.grid[next_move] == TileType.AGENT:
                    agent.agent_collision_count += 1
                
                # Update agent metrics
                agent.step_count += 1
                agent.leader_distance_sum += np.linalg.norm(np.array(agent.position) - np.array(leader.position))
                agent.path_distance_sum += map_config.leader_path_distance_map[agent.position]

    # Calculate fitness for map config     
    return calculate_fitness(followers)