"""Observation history buffer."""

from collections import deque

import numpy as np
import torch

__all__ = ["HistoryBuffer"]


class HistoryBuffer:
    def __init__(self, length):
        self.length = length
        self.image_buffer = deque(maxlen=length)
        self.scalar_buffer = deque(maxlen=length)

    def reset(self, observation, scalar_input):
        self.image_buffer.clear()
        self.scalar_buffer.clear()

        for _ in range(self.length):
            self.image_buffer.append(observation.copy())

            self.scalar_buffer.append(scalar_input.copy())

    def append(self, observation, scalar_input):
        self.image_buffer.append(observation.copy())

        self.scalar_buffer.append(scalar_input.copy())

    def tensors(self):
        images = torch.tensor(np.stack(self.image_buffer), dtype=torch.float32)

        scalars = torch.tensor(np.stack(self.scalar_buffer), dtype=torch.float32)

        return images, scalars

    def terminal_tensors(self, terminal_observation, terminal_scalars):
        """
        Builds the history tensors as they would look one step past the
        current buffer, ending in the true terminal observation/scalars
        (i.e. before the episode-end auto-reset). Used only to bootstrap
        the value function on time-limit truncations. Does not mutate
        this buffer.
        """

        images = list(self.image_buffer)[1:] + [terminal_observation]
        scalars = list(self.scalar_buffer)[1:] + [terminal_scalars]

        images = torch.tensor(np.stack(images), dtype=torch.float32)
        scalars = torch.tensor(np.stack(scalars), dtype=torch.float32)

        return images, scalars