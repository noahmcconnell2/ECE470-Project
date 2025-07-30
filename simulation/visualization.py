# Simulation Visualization Module
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
    def __init__(self, map_config: MapConfig, tile_size, record: bool = False, save_path: Path = None):
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
        instructions = "R = Restart   |   Q = Quit   |   + / - = Zoom"
        text_surface = self.font.render(instructions, True, (50, 50, 50))
        self.screen.blit(text_surface, (10, self.window_height - 25))

    def draw_tile(self, x: int, y: int, colour: tuple):
        pixel_x = x * self.tile_size
        pixel_y = y * self.tile_size
        pygame.draw.rect(self.screen, colour, (pixel_x, pixel_y, self.tile_size, self.tile_size))

    def draw_border(self, x: int, y: int, colour: tuple, border_thickness=2):
        pixel_x = x * self.tile_size
        pixel_y = y * self.tile_size
        pygame.draw.rect(
            self.screen, colour,
            (pixel_x, pixel_y, self.tile_size, self.tile_size),
            width=border_thickness
        )

    def draw_grid(self):
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
        if self.recorder:
            self.recorder.close()
        pygame.display.quit()
        pygame.quit()


def visualize_simulation(map_config: MapConfig, delay_ms: int = 100, tile_size=20):
    """
    Create and run a simple visualization of the current map state.
    
    Args:
        map_config: The current map configuration to visualize
        delay_ms: Milliseconds to wait between frames (controls speed)
    """
    visualizer = SwarmVisualizer(map_config, tile_size=tile_size)
    
    # Keep window open until user closes it
    running = True
    clock = pygame.time.Clock() # helps control framerate
    
    while running:
        running = visualizer.run_frame()
        clock.tick(1000 // delay_ms)  # convert delay to FPS
    
    visualizer.close()