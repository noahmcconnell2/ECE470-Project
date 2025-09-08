"""
Agent Representation for Swarm Simulation

This module defines the core agent structure used in the swarm behavior simulation.
Agents can be leaders or followers, and they maintain internal metrics such as
movement path, heading direction, and interaction counts for fitness evaluation.

Each agent is initialized with a behavioral genome (list of weights) and tracks
its behavior over time, including steps taken, cumulative distances, and collisions.

"""

import enum
from typing import List
from dataclasses import dataclass, field
from configs import PERCEPTION_RANGE

class AgentRole(enum.IntEnum):
    """Enum to distinguish between leader and follower roles."""
    LEADER = 0
    FOLLOWER = 1

@dataclass
class Agent:
    """
    Represents an agent in the swarm simulation.

    Attributes:
        role (AgentRole): Agent's role (LEADER or FOLLOWER).
        position (Tuple[int, int]): Current grid coordinates (x, y).
        heading (Tuple[int, int]): Current movement direction (dx, dy).
        genome (List[float]): Behavioral gene weights.
        perception_range (int): Chebyshev radius for local perception.
        path (List[Tuple[int, int]]): History of positions visited.
        osc_penalty (float): Penalty counter for oscillatory movement.
        complete (bool): Flag indicating whether the agent completed its goal.

        step_count (int): Total movement steps taken.
        leader_distance_sum (float): Cumulative distance to leader.
        path_distance_sum (float): Cumulative distance to leader's path.
        obstacle_collision_count (int): Number of obstacle collisions.
        agent_collision_count (int): Number of collisions with other agents.
    """
    role: AgentRole  # Type of agent (LEADER or FOLLOWER)
    position: tuple[int, int]  # (x, y) coordinates in the grid
    heading: tuple[int, int]  # (dx, dy) direction vector for movement
    genome: List[float]  # Genetic representation of the agent's behavior or traits
    perception_range: int = PERCEPTION_RANGE
    path: List[tuple[int, int]] = field(default_factory=list)  # Path taken by the agent during the simulation
    osc_penalty: float = 0.0
    complete: bool = False

    # Metrics for fitness evaluation
    step_count: int = 0  # Number of steps taken by the agent
    leader_distance_sum: float = 0.0  # Cumulative distance to the leader
    path_distance_sum: float = 0.0  # Cumulative distance from nearest point along leader path
    obstacle_collision_count: int = 0  # Number of collisions with obstacles
    agent_collision_count: int = 0  # Number of collisions with other agents
    
    def __repr__(self):
        """
        Return a string representation showing role and position.
        """
        return f"Agent(role={self.role.name}, position={self.position})"

    def move(self, new_position: tuple[int, int]):
        """
        Update the agent's position.

        Args:
            new_position (Tuple[int, int]): New (x, y) location.
        """
        self.position = new_position

    def update_heading(self, next_move: tuple[int, int]):
        """
        Recalculate the agent's heading vector based on a target move.

        Args:
            next_move (Tuple[int, int]): Target position to update heading toward.
        """
        dx = next_move[0] - self.position[0]
        dy = next_move[1] - self.position[1]
        self.heading = (dx, dy)
