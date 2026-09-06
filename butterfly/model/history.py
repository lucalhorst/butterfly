"""Observation history buffer."""

from collections import deque

import numpy as np
import torch

__all__ = ["HistoryBuffer", "stack_scalar_dicts"]


def stack_scalar_dicts(list_of_dicts):
    """Stack a list of per-timestep scalar dicts into a dict of tensors.

    ``list_of_dicts`` is a list of dicts (e.g. the outputs of
    ``ButterflyEnv.get_scalar_inputs``) each mapping a field name to a
    numpy array. Returns a dict mapping the same field names to stacked
    ``torch.Tensor``s, one dimension deeper along the stack axis.

    This is the single source of truth for stacking the structured scalar
    observations; both ``HistoryBuffer`` and the training rollout code use
    it so the logic lives in exactly one place.
    """
    return {
        key: torch.tensor(np.stack([d[key] for d in list_of_dicts]))
        for key in list_of_dicts[0]
    }


class HistoryBuffer:
    def __init__(self, length):
        self.length = length
        self.image_buffer = deque(maxlen=length)
        # One deque per scalar field (see ButterflyEnv.get_scalar_inputs
        # for the set of keys), kept in lockstep with image_buffer.
        self.scalar_buffers = {}

    def reset(self, observation, scalar_input):
        self.image_buffer.clear()

        for _ in range(self.length):
            self.image_buffer.append(observation.copy())

        self.scalar_buffers = {
            key: deque(maxlen=self.length) for key in scalar_input.keys()
        }
        for key, value in scalar_input.items():
            for _ in range(self.length):
                self.scalar_buffers[key].append(value.copy())

    def append(self, observation, scalar_input):
        self.image_buffer.append(observation.copy())

        for key, value in scalar_input.items():
            self.scalar_buffers[key].append(value.copy())

    def tensors(self):
        images = torch.tensor(np.stack(self.image_buffer), dtype=torch.float32)

        scalars = {
            key: torch.tensor(np.stack(deques)) for key, deques in self.scalar_buffers.items()
        }

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
        images = torch.tensor(np.stack(images), dtype=torch.float32)

        scalars = {}
        for key, deques in self.scalar_buffers.items():
            field = list(deques)[1:] + [terminal_scalars[key]]
            scalars[key] = torch.tensor(np.stack(field))

        return images, scalars
