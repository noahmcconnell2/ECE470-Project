import enum
from typing import List
from dataclasses import dataclass, field
from configs import PERCEPTION_RANGE

class AgentRole(enum.IntEnum):
    LEADER = 0
    FOLLOWER = 1

@dataclass
class Agent:
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
        return f"Agent(role={self.role.name}, position={self.position})"

    def move(self, new_position: tuple[int, int]):
        self.position = new_position

    def update_heading(self, next_move: tuple[int, int]):
        dx = next_move[0] - self.position[0]
        dy = next_move[1] - self.position[1]
        self.heading = (dx, dy)
