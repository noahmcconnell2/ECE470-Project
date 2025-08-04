"""
grid_utils.py

Provides utility structures and enums for grid-based simulations.

Includes:
- `TileType`: Enum for identifying cell types (EMPTY, OBSTACLE, AGENT).
- `GridWrapper`: Wrapper class around a NumPy array that provides intuitive (x, y) access,
  boundary checking, and neighborhood queries for use in swarm and pathfinding simulations.

Designed for use in simulations involving agent movement on a 2D grid.
"""

from dataclasses import dataclass
import numpy as np
from typing import Tuple
import enum

class TileType(enum.IntEnum):
    """
    Enum representing different types of tiles in the grid.

    Attributes:
        EMPTY: A free tile that can be traversed.
        OBSTACLE: A tile occupied by an obstacle.
        AGENT: A tile currently occupied by an agent.
    """
    EMPTY = 0
    OBSTACLE = 1
    AGENT = 2

@dataclass
class GridWrapper:
    """
    A wrapper around a 2D NumPy array representing a grid.

    This class provides (x, y) based access to the underlying (row, col) NumPy structure,
    and includes helper methods for checking bounds and getting Moore neighborhoods.

       x →
    y  +--------------------→
    ↓  | (0,0) (1,0) (2,0) ...
       | (0,1) (1,1) (2,1)
       | ... 
       | (0,n) (1,n) (2,n) ...
    """
    grid: np.ndarray

    def raw(self) -> np.ndarray:
        """Returns the raw NumPy array representing the grid."""
        return self.grid

    def get(self, pos: Tuple[int, int]):
        """
        Returns the value at position (x, y).

        Args:
            pos: A tuple (x, y) representing grid coordinates.

        Raises:
            IndexError: If the position is out of bounds.
        """
        if not self.in_bounds(pos):
            raise IndexError(f"Position {pos} is out of bounds for grid shape {self.grid.shape}.") 
        x, y = pos
        return self.grid[y, x]  # NumPy uses [row, col] = [y, x]

    def set(self, pos: Tuple[int, int], value):
        """
        Sets the value at position (x, y).

        Args:
            pos: A tuple (x, y) position in the grid.
            value: The value to assign to that position.
        """
        x, y = pos
        self.grid[y, x] = value

    def shape(self):
        """
        Returns the shape of the grid as (width, height),
        converting from NumPy's (rows, cols) to (x, y) logic.

        Returns:
            Tuple[int, int]: (width, height)
        """
        h, w = self.grid.shape
        return (w, h)

    def in_bounds(self, pos: Tuple[int, int]) -> bool:
        """
        Checks if the given (x, y) position is within the grid bounds.

        Args:
            pos: A tuple (x, y) position.

        Returns:
            bool: True if in bounds, False otherwise.
        """
        x, y = pos
        return 0 <= y < self.grid.shape[0] and 0 <= x < self.grid.shape[1]
    
    def get_neighborhood(self, pos: Tuple[int, int]) -> list[Tuple[int, int]]:
        """
        Returns the Moore neighborhood (8 surrounding + center) of a given (x, y) position.

        Args:
            pos: A tuple (x, y) position.

        Returns:
            list[Tuple[int, int]]: List of valid neighbor coordinates within bounds.
        """
        x, y = pos
        neighborhood = []
        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                tx, ty = x + dx, y + dy
                if self.in_bounds((tx, ty)):
                    neighborhood.append((tx, ty))

        return neighborhood
