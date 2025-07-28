from agent.agent import Agent, AgentRole
from map.grid_utils import GridWrapper, TileType
from dataclasses import dataclass
from configs import NUM_AGENTS
import numpy as np


@dataclass
class MapConfig:
    """Configuration for a map used in the simulation.
    Contains the grid, leader path, obstacle distance map, and agent index.
    (x, y) coordinates starting from the top-left corner (0, 0); y increasing downwards, x increasing to the right.
    """
    grid: GridWrapper # np.ndarray wrapper for converting (x,y) coordinates to (y,x) for numpy array access
    leader_path: list[tuple[int, int]]
    obstacle_distance_map: np.ndarray
    leader_path_distance_map: np.ndarray
    agent_index: dict[tuple[int, int], Agent] = None
    name: str = "UnnamedMap"

    def update(self, old_position: tuple[int, int], agent: Agent):
        """
        Update the agent's position in the grid and agent index.
        Args:
            position: The new position of the agent.
            agent: The agent to update.
        """
        if self.agent_index is None:
            self.agent_index = {}

        new_position = agent.position # assumes position is valid

        # Remove agent from old position in agent index and grid
        if old_position is not None and old_position in self.agent_index:
            del self.agent_index[old_position]
            self.grid.set(old_position, TileType.EMPTY)

        # Add agent to new position in agent index and grid
        self.agent_index[new_position] = agent
        self.grid.set(new_position, TileType.AGENT)

    def add_followers(self, entrance_size: int, followers: list[Agent], genome: list[float], leader: Agent):
        x, y = self.leader_path[0]  # Start position of the leader
        w, h = self.grid.shape()  # Width and height of the grid
        half = entrance_size // 2
        positions = []

        if y == 0:  # Top edge → followers go ABOVE (y=0), varying in x
            for dx in range(-half, half + 1):
                fx = x + dx
                fy = y  # stay on edge
                if 0 <= fx < w:
                    positions.append((fx, fy))

        elif y == h - 1:  # Bottom edge
            for dx in range(-half, half + 1):
                fx = x + dx
                fy = y
                if 0 <= fx < w:
                    positions.append((fx, fy))

        elif x == 0:  # Left edge → followers on x=0, vary y
            for dy in range(-half, half + 1):
                fx = x
                fy = y + dy
                if 0 <= fy < h:
                    positions.append((fx, fy))

        elif x == w - 1:  # Right edge
            for dy in range(-half, half + 1):
                fx = x
                fy = y + dy
                if 0 <= fy < h:
                    positions.append((fx, fy))

        # Loop through all possible entrance positions and add followers if empty
        for pos in positions:
            if len(followers) >= NUM_AGENTS - 1:  # Stop if we have enough followers
                break
            # Create a new follower agent at the position
            if self.grid.get(pos) == TileType.EMPTY:
                agent = Agent(AgentRole.FOLLOWER, pos, leader.heading, genome)
                followers.append(agent)
                self.update(old_position=None, agent=agent)  # Update agent index and grid with the new follower
                agent.step_count += 1
                
            # else:
            #     print(f"Blocked: Cannot add follower at {pos} – Tile = {self.grid.get(pos)}")


    def get_entrance_positions(self) -> list[tuple[int, int]]:
        """
        Return a static list of entrance coordinates based on the initial leader position and entrance size.
        """
        from configs import ENTRANCE_SIZE  # Local import to avoid circular dependency

        x, y = self.leader_path[0]
        w, h = self.grid.shape()
        half = ENTRANCE_SIZE // 2
        entrance = []

        if y == 0 or y == h - 1:
            for dx in range(-half, half + 1):
                fx = x + dx
                if 0 <= fx < w:
                    entrance.append((fx, y))
        elif x == 0 or x == w - 1:
            for dy in range(-half, half + 1):
                fy = y + dy
                if 0 <= fy < h:
                    entrance.append((x, fy))

        return entrance
    

    def reset_agents(self):
        """Remove all agents from the map (grid and agent index)."""
        positions = list(self.agent_index.keys())
        for pos in positions:
            if self.grid.get(pos) == TileType.AGENT:
                self.grid.set(pos, TileType.EMPTY)
        self.agent_index.clear()
