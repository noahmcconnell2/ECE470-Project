"""
Imageio-based Video Recording Utility for Pygame Frames

This class provides a lightweight wrapper around `imageio` to capture
and record Pygame surface frames into an MP4 video using the libx264 codec.

Designed for recording frames during swarm simulation visualizations.

Dependencies:
    - imageio
    - numpy
    - pygame

Author:
    @lexph (structure and base implementation guided by OpenAI's ChatGPT, July 2025)

Notes:
    Structure and functionality (e.g., using `pygame.surfarray`, transpose to (H, W, C),
    and appending frames via `imageio`) were generated with assistance from OpenAI's ChatGPT.
    Code was customized and validated for use in swarm visualization recordings.
"""

import imageio
import numpy as np
import pygame

class ImageioVideoWriter:
    """
    A utility class for recording Pygame surface frames to a video file.

    Attributes:
        filename (str): Output video file path.
        fps (int): Frames per second for the output video.
        writer (imageio.Writer): Internal video writer object.
    """

    def __init__(self, filename: str, fps: int = 7):
        """
        Initialize the video writer with a target filename and frame rate.

        Args:
            filename (str): Path where the video will be saved.
            fps (int): Frames per second for the video. Default is 7.
        """
        self.filename = filename
        self.fps = fps
        self.writer = imageio.get_writer(self.filename, fps=self.fps, codec='libx264')

    def capture(self, surface):
        """
        Capture the current Pygame surface and write it to the video file.

        Args:
            surface (pygame.Surface): The surface to capture from the screen.
        """
        rgb_array = pygame.surfarray.array3d(surface)
        rgb_array = np.transpose(rgb_array, (1, 0, 2))  # Convert to (height, width, channels)
        self.writer.append_data(rgb_array)

    def close(self):
        """
        Finalize and close the video file. Should be called after recording is complete.
        """
        self.writer.close()
