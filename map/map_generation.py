import numpy as np
import heapq
import random
from map.map_structures import MapConfig
from map.grid_utils import GridWrapper
from map.grid_utils import TileType
from configs import GRID_DIM, PERCENT_OBSTACLES, MIN_LEADER_PATH_DISTANCE
from collections import deque

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

def heuristic(a: tuple[int,int], b: tuple[int,int]) ->int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def a_star(grid: np.ndarray, start: tuple[int,int], goal: tuple[int,int]) -> list[tuple[int,int]]:
    w, h = grid.shape
    open_set= []
    heapq.heappush(open_set, (0+heuristic(start, goal),0, start,[start]))
    visited = set()

    while open_set:
        f, g, current, path = heapq.heappop(open_set)
        if current == goal:
            return path
        if current in visited:
            continue
        visited.add(current)

        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx, ny = current[0] + dx, current[1] + dy
            if w > nx >= 0 == grid[nx][ny] and 0 <= ny < h:
                next_node = (nx, ny)
                heapq.heappush(open_set, (g + 1 + heuristic(next_node, goal), g + 1, next_node, path + [next_node]))

    return []

def generate_populated_map(grid_dim: tuple[int, int], percent_obstacles: float) -> GridWrapper:
    width, height = grid_dim
    total_cells = width * height
    num_obstacles = int(total_cells * percent_obstacles)

    # Start with an empty grid
    grid_array = np.full((height, width), TileType.EMPTY, dtype=int)

    # Flatten indices and randomly choose obstacle positions
    all_positions = [(x, y) for x in range(width) for y in range(height)]
    obstacle_positions = random.sample(all_positions, num_obstacles)

    for pos in obstacle_positions:
        x, y = pos
        grid_array[y, x] = TileType.OBSTACLE  # numpy: (row, col) = (y, x)

    return GridWrapper(grid_array)

def generate_leader_path(grid: np.ndarray, min_distance: int) -> list[tuple[int, int]]:
    entrance_width = 3
    possible_endpoints = get_valid_leader_starts(grid.shape[::-1], entrance_width)

    #set a max number of attempts to find a valid start and end
    max_attempts = 100
    attempt = 0

    while attempt < max_attempts:
        attempt +=1

        #choose a random point from possible_endpoints
        if len(possible_endpoints) == 0:
            print ("Not eneough endpoints, for a valid leader path")
            return []
        #pick a random endpoint and start
        start, goal = random.choice(possible_endpoints)

        if grid[start[1]][start[0]] == TileType.OBSTACLE or grid[goal[1]][goal[0]] == TileType.OBSTACLE:
            continue
        #perform a_star from start to goal if its valid
        path = a_star(grid, start, goal)
        #make sure that the path we got is bigger than the min distance that we set
        if path and len(path) >= min_distance:
            return path

    print("Could not find a valid path!")
    return []




def compute_obstacle_distance_map(grid: np.ndarray) -> GridWrapper:
    pass

#needs to return List not Gridwrapper
def compute_leader_path_distance_map(leader_path: list[tuple[int, int]], grid_shape: tuple[int, int]) -> GridWrapper:
    #given a leader path and grid shape, build a 2d array where each cell holds Manhattan distance to the closest point on the leader path

    pass
