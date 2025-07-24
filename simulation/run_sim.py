import numpy as np
from map.map_structures import MapConfig
from agent.agent import Agent, AgentRole
from map.grid_utils import TileType
from simulation.movement import rank_moves_by_score
from simulation.fitness import calculate_fitness
from configs import NUM_AGENTS, ENTRANCE_SIZE, POST_GOAL_BUFFER_STEPS, FOLLOWER_STAGGER_INTERVAL

def run_simulation(genome, map_config: MapConfig, visualize: bool= False) -> float:
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
    
    # Initial heading
    initial_heading = tuple(np.subtract(map_config.leader_path[1], map_config.leader_path[0]))

    # Initialize leader and add to datastructures
    leader = Agent(AgentRole.LEADER, map_config.leader_path[0], initial_heading, genome=genome)
    map_config.update(old_position=None, agent=leader)  # Update agent_index and grid with the leader's initial position

    # Loop until leader reaches goal plus a few extra steps
    for count in range(len(map_config.leader_path) + POST_GOAL_BUFFER_STEPS):  # Allow some extra steps to ensure followers can catch up
        # --- Update leader's attributes ---
        if leader.step_count + 1 < len(map_config.leader_path):
            next_move = map_config.leader_path[leader.step_count + 1]
            old_position = leader.position
            leader.move(next_move)  # Move leader to next position
            leader.heading = (next_move[0] - old_position[0], next_move[1] - old_position[1])
            map_config.update(old_position, leader) # update agent_index and grid using the next move and leaders current position

        leader.path.append(leader.position)
        leader.step_count += 1  # Increment step count for leader
        # ---- End of leader update --

        # Add followers to entrance after first step until NUM_followers are present
        if (count + 1) % FOLLOWER_STAGGER_INTERVAL == 0 and len(followers) < NUM_AGENTS - 1:
            map_config.add_followers(ENTRANCE_SIZE, followers, genome=genome, leader=leader)

        # print(f"Step {leader.step_count}: Followers = {len(followers)}")
        


        # Advance each agent
        for agent in followers:
            if agent.role == AgentRole.LEADER:
                # Leader's movement is already handled above
                continue

            ranked_moves = rank_moves_by_score(map_config, agent, leader)
            for score, next_move in ranked_moves:
                if next_move == agent.position or map_config.grid.get(next_move) == TileType.EMPTY:
                    old_position = agent.position
                    agent.move(next_move)
                    agent.path.append(agent.position)
                    agent.heading = (next_move[0] - old_position[0], next_move[1] - old_position[1])
                    map_config.update(old_position, agent)
                    
                    # Update metrics after successful move
                    agent.step_count += 1
                    agent.leader_distance_sum += np.linalg.norm(np.array(agent.position) - np.array(leader.position))
                    agent.path_distance_sum += map_config.leader_path_distance_map.get(agent.position)
                    break

                elif map_config.grid.get(next_move) == TileType.OBSTACLE:
                    agent.obstacle_collision_count += 1
                
                elif map_config.grid.get(next_move) == TileType.AGENT:
                    agent.agent_collision_count += 1
                
                # print(f"Leader Position: {leader.position}, Agent Position: {agent.position}, Next Move: {next_move}, Heading: {agent.heading}")



    fitness = calculate_fitness(followers)

    summary = {
        "fitness": fitness,
        "genome": genome,
        "map_config": map_config.name,
        "leader_path": leader.path,
        "num_followers": len(followers),
        "agents": [{
            "path": agent.path,
            "step_count": agent.step_count,
            "leader_distance_sum": agent.leader_distance_sum,
            "path_distance_sum": agent.path_distance_sum,
            "obstacle_collision_count": agent.obstacle_collision_count,
            "agent_collision_count": agent.agent_collision_count,
            
        } for agent in followers]
    }
    
 
    return fitness, summary