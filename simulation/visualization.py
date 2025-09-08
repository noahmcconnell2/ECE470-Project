"""
Swarm Simulation Visualization Module

Provides a real-time Pygame-based UI for visualizing swarm behavior evolution. 
Supports rendering agents, obstacles, leader paths, entrance zones, and agent movement.

Key Features:
- Adjustable tile size with zoom support (+ / - keys)
- Agent color-coding by role (leader, follower)
- Optional video recording using `imageio`
- Interactive controls for quitting or restarting the simulation
"""

import pygame
import numpy as np
from map.map_structures import MapConfig
from agent.agent import Agent, AgentRole
from map.grid_utils import TileType
from utils.imageio_recorder import ImageioVideoWriter
from configs import RECORD_VIDEO
from pathlib import Path


# Colour definitions (R, G, B)
COLOURS = {
    'empty': (255, 255, 255), # white (background/screen colour)
    'obstacle': (0, 0, 0), # black
    'leader': (255, 0, 0), # red
    'follower': (0, 100, 255), # blue
    'path': (255, 255, 0), # yellow
    'entrance_border': (0, 255, 0), # green
}


class SwarmVisualizer:
    """
    Pygame-based visualizer for swarm agent simulation.

    Attributes:
        map_config (MapConfig): The simulation map and agent configuration.
        tile_size (int): Pixel size of each grid tile.
        record (bool): Whether to record frames into a video.
        recorder (ImageioVideoWriter): Recorder instance if video saving is enabled.
    """
    def __init__(self, map_config: MapConfig, tile_size, record: bool = False, save_path: Path = None):
        """
        Initializes the Pygame visualizer for the simulation.

        Args:
            map_config (MapConfig): Map data with grid and agents.
            tile_size (int): Size of each tile in pixels.
            record (bool): Whether to enable video recording.
            save_path (Path): Where to save the video file if recording.
        """
        pygame.init()

        self.map_config = map_config
        self.grid_width, self.grid_height = map_config.grid.shape()
        self.tile_size = tile_size  # mutable but doesn't affect window size

        self.window_width = self.grid_width * tile_size
        self.window_height = self.grid_height * tile_size

        self.font = pygame.font.SysFont('Arial', 16)
        self.screen = pygame.display.set_mode((self.window_width, self.window_height), pygame.RESIZABLE)
        pygame.display.set_caption("Swarm Simulation")
        self.clock = pygame.time.Clock()

        self.record = record and RECORD_VIDEO and save_path is not None
        self.recorder = ImageioVideoWriter(str(save_path), fps=7) if self.record else None

    def draw_overlay(self):
        """Draws control instructions overlayed at the bottom of the screen."""
        instructions = "R = Restart   |   Q = Quit   |   + / - = Zoom"
        text_surface = self.font.render(instructions, True, (50, 50, 50))
        self.screen.blit(text_surface, (10, self.window_height - 25))

    def draw_tile(self, x: int, y: int, colour: tuple):
        """Draws a filled tile at the given grid position with the given color."""
        pixel_x = x * self.tile_size
        pixel_y = y * self.tile_size
        pygame.draw.rect(self.screen, colour, (pixel_x, pixel_y, self.tile_size, self.tile_size))

    def draw_border(self, x: int, y: int, colour: tuple, border_thickness=2):
        """Draws a border around a tile at (x, y) to highlight entrances."""
        pixel_x = x * self.tile_size
        pixel_y = y * self.tile_size
        pygame.draw.rect(
            self.screen, colour,
            (pixel_x, pixel_y, self.tile_size, self.tile_size),
            width=border_thickness
        )

    def draw_grid(self):
        """Renders the entire simulation grid including tiles, agents, and overlays."""
        self.screen.fill(COLOURS['empty'])

        for x in range(self.grid_width):
            for y in range(self.grid_height):
                tile_type = self.map_config.grid.get((x, y))
                if tile_type == TileType.OBSTACLE:
                    self.draw_tile(x, y, COLOURS['obstacle'])

        for x, y in self.map_config.leader_path:
            if self.map_config.grid.get((x, y)) != TileType.AGENT:
                self.draw_tile(x, y, COLOURS['path'])

        if self.map_config.agent_index:
            for position, agent in self.map_config.agent_index.items():
                if agent.role == AgentRole.LEADER:
                    self.draw_tile(*position, COLOURS['leader'])
                elif agent.role == AgentRole.FOLLOWER:
                    self.draw_tile(*position, COLOURS['follower'])

        # Draw entrance tile borders (if available)
        if hasattr(self.map_config, "get_entrance_positions"):
            for x, y in self.map_config.get_entrance_positions():
                self.draw_border(x, y, COLOURS['entrance_border'])

        self.draw_overlay()
        pygame.display.flip()

        if self.recorder:
            self.recorder.capture(self.screen)

    def run_frame(self, fps=7) -> bool:
        """
        Processes a single frame of the simulation and handles events.

        Args:
            fps (int): Frames per second cap for rendering.

        Returns:
            str: "quit", "restart", or "continue" depending on user input.
        """
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return "quit"
            elif event.type == pygame.KEYDOWN:
                key = event.key

                if key == pygame.K_q or key == pygame.K_c:
                    return "quit"
                elif key == pygame.K_r:
                    return "restart"
                elif key in (pygame.K_PLUS, pygame.K_EQUALS, pygame.K_KP_PLUS):
                    self.tile_size = min(self.tile_size + 2, 100)
                elif key in (pygame.K_MINUS, pygame.K_KP_MINUS):
                    self.tile_size = max(self.tile_size - 2, 5)

        self.draw_grid()
        self.clock.tick(fps)
        return True

    def close(self):
        """Closes the simulation window and stops any video recording."""
        if self.recorder:
            self.recorder.close()
        pygame.display.quit()
        pygame.quit()


def visualize_simulation(map_config: MapConfig, delay_ms: int = 100, tile_size=20):
    """
    Launches an interactive visualization window for the provided map config.

    Args:
        map_config (MapConfig): Grid and agent state to visualize.
        delay_ms (int): Delay between frames in milliseconds (inverse of FPS).
        tile_size (int): Size of each grid tile in pixels.
    """
    visualizer = SwarmVisualizer(map_config, tile_size=tile_size)
    
    # Keep window open until user closes it
    running = True
    clock = pygame.time.Clock() # helps control framerate
    
    while running:
        running = visualizer.run_frame()
        clock.tick(1000 // delay_ms)  # convert delay to FPS
    
    visualizer.close()