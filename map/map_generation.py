import numpy as np
from map.map_structures import MapConfig
from map.grid_utils import GridWrapper
from configs import GRID_DIM, PERCENT_OBSTACLES, MIN_LEADER_PATH_DISTANCE

def generate_n_map_configs(n: int) -> list[MapConfig]:
    """ Generates a list of n map configurations."""
    map_configs = []
    for _ in range(5):
        map_config = generate_map_config()
        map_configs.append(map_config)
    return map_configs


def generate_map_config(grid_dim: tuple[int, int]= GRID_DIM, 
                        percent_obstacles: float = PERCENT_OBSTACLES,
                        min_leader_path_distance: int = MIN_LEADER_PATH_DISTANCE
                        ) -> MapConfig:
    
    grid = generate_populated_map(grid_dim, percent_obstacles)

    # Extract raw grid numpy array for computation
    grid_np = grid.raw()

    leader_path = generate_leader_path(grid_np, min_leader_path_distance) 
    obstacle_distance_map = compute_obstacle_distance_map(grid_np)
    leader_path_distance_map = compute_leader_path_distance_map(leader_path, grid_dim)
    agent_index = {} # Add all grid positions to agent index with value None

    return MapConfig(grid, leader_path, obstacle_distance_map, leader_path_distance_map, agent_index)


def get_valid_leader_starts(grid_shape: tuple[int, int], entrance_width: int) -> list[tuple[int, int]]:
    w, h = grid_shape
    half = entrance_width // 2
    starts = []

    # Top edge (avoid corners)
    for x in range(half, w - half):
        starts.append((x, 0))

    # Bottom edge
    for x in range(half, w - half):
        starts.append((x, h - 1))

    # Left edge
    for y in range(half, h - half):
        starts.append((0, y))

    # Right edge
    for y in range(half, h - half):
        starts.append((w - 1, y))

    return starts


def generate_populated_map(grid_dim: tuple[int, int], percent_obstacles: float) -> GridWrapper:
    pass

def generate_leader_path(grid: np.ndarray, min_distance: int) -> list[tuple[int, int]]:
    # uses get_valid_leader_starts to get valid starting positions for the leader
    pass

def compute_obstacle_distance_map(grid: np.ndarray) -> GridWrapper:
    pass

def compute_leader_path_distance_map(leader_path: list[tuple[int, int]], grid_shape: tuple[int, int]) -> GridWrapper:
    pass