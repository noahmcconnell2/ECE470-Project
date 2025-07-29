import imageio
import numpy as np
import pygame

class ImageioVideoWriter:
    def __init__(self, filename: str, fps: int = 7):
        self.filename = filename
        self.fps = fps
        self.writer = imageio.get_writer(self.filename, fps=self.fps, codec='libx264')

    def capture(self, surface):
        """Capture the current Pygame surface and write it directly to video."""
        rgb_array = pygame.surfarray.array3d(surface)
        rgb_array = np.transpose(rgb_array, (1, 0, 2))  # Convert to (height, width, channels)
        self.writer.append_data(rgb_array)

    def close(self):
        """Finalize and close the video file."""
        self.writer.close()
