from dataclasses import dataclass
import numpy as np
from typing import Tuple, List
import enum

class TileType(enum.IntEnum):
    EMPTY = 0
    OBSTACLE = 1
    AGENT = 2

@dataclass
class GridWrapper:
    """
    A wrapper for a numpy array to represent a grid.
    Allows for (x, y) access to numpy array (y, x) coordinates.

       x →
    y  +--------------------→
    ↓  | (0,0) (1,0) (2,0) ...
       | (0,1) (1,1) (2,1)
       | ... 
       | (0,n) (1,n) (2,n) ...
    """
    grid: np.ndarray

    def get(self, pos: Tuple[int, int]):
        """Get the value at (x, y) position."""
        x, y = pos
        return self.grid[y, x]  # NumPy uses [row, col] = [y, x]

    def set(self, pos: Tuple[int, int], value):
        """Set the value at (x, y) position."""
        x, y = pos
        self.grid[y, x] = value

    def shape(self):
        """Return the shape of the grid as (width, height)"""
        h, w = self.grid.shape
        return (w, h)

    def in_bounds(self, pos: Tuple[int, int]) -> bool:
        """Check if (x, y) position is inside the grid bounds."""
        x, y = pos
        return 0 <= y < self.grid.shape[0] and 0 <= x < self.grid.shape[1]
