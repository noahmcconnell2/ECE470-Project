# Simulation Visualization Module
import pygame
import numpy as np
from map.map_structures import MapConfig
from agent.agent import Agent, AgentRole
from map.grid_utils import TileType

TILE_SIZE = 20 # edge length in pixels of each tile


# Colour definitions (R, G, B)
COLOURS = {
    'empty': (255, 255, 255), # white (background/screen colour)
    'obstacle': (0, 0, 0), # black
    'leader': (255, 0, 0), # red
    'follower': (0, 100, 255), # blue
    'path': (255, 255, 0), # yellow
}


class SwarmVisualizer:
    def __init__(self, map_config: MapConfig):
        self.map_config = map_config
        self.grid_width, self.grid_height = map_config.grid.shape()

        # Calculate window dimensions
        self.window_width = self.grid_width * TILE_SIZE
        self.window_height = self.grid_height * TILE_SIZE
        
        # Initialize pygame
        pygame.init()
        self.screen = pygame.display.set_mode((self.window_width, self.window_height))
        pygame.display.set_caption("Swarm Simulation")

        self.clock = pygame.time.Clock() 


    def draw_tile(self, x: int, y: int, colour: tuple):
        """
        Draw a single tile at grid position (x, y) with the given colour.
        Converts grid coordinates to pixel coordinates and draws a rectangle.
        """

        pixel_x = x * TILE_SIZE
        pixel_y = y * TILE_SIZE
        pygame.draw.rect(self.screen, colour, (pixel_x, pixel_y, TILE_SIZE, TILE_SIZE))


    def draw_grid(self):
        """
        Draw the complete grid (including obstacles, path and agents).
        """

        self.screen.fill(COLOURS['empty']) # fill background

        for x in range(self.grid_width):
            for y in range(self.grid_height):
                tile_type = self.map_config.grid.get((x, y))
                if tile_type == TileType.OBSTACLE:
                    self.draw_tile(x, y, COLOURS['obstacle'])
                elif tile_type == TileType.EMPTY:
                    self.draw_tile(x, y, COLOURS['empty'])
                elif tile_type == TileType.AGENT:
                    self.draw_tile(x, y, COLOURS['empty'])

        # Draw leader path
        for x, y in self.map_config.leader_path:
            # Only draw path if no agent is currently on this tile
            if self.map_config.grid.get((x, y)) != TileType.AGENT:
                self.draw_tile(x, y, COLOURS['path'])
            
        # Draw agents on top of everything else
        if self.map_config.agent_index:
            for position, agent in self.map_config.agent_index.items():
                if agent.role == AgentRole.LEADER:
                    self.draw_tile(position[0], position[1], COLOURS['leader'])
                elif agent.role == AgentRole.FOLLOWER:
                    self.draw_tile(position[0], position[1], COLOURS['follower'])

        # Update the display
        pygame.display.flip()


    def run_frame(self, fps=7) -> bool:
        """
        Draw one frame and handle events.
        - fps: Frames per second target
        - wait_for_keypress: If True, pauses simulation waiting for R or Enter key
        Returns:
            False if window closed, True otherwise.
        """
        # Handle pygame events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False 
        
        # Draw the current state
        self.draw_grid()
        self.clock.tick(fps)  # control FPS 
        return True


    def close(self):
        """
        Clean up pygame resources.
        """
        pygame.quit()


def visualize_simulation(map_config: MapConfig, delay_ms: int = 100):
    """
    Create and run a simple visualization of the current map state.
    
    Args:
        map_config: The current map configuration to visualize
        delay_ms: Milliseconds to wait between frames (controls speed)
    """
    visualizer = SwarmVisualizer(map_config)
    
    # Keep window open until user closes it
    running = True
    clock = pygame.time.Clock() # helps control framerate
    
    while running:
        running = visualizer.run_frame()
        clock.tick(1000 // delay_ms)  # convert delay to FPS
    
    visualizer.close()