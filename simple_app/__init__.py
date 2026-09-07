"""Standalone training app for the butterfly environment.

Imports only the butterfly environment (plus its config, which is needed
to construct it) from the main package -- no butterfly.model / butterfly.algo
code is reused. Ships its own simple MLP + ResNet-18 policy and PPO loop.
"""

from simple_app.policy import SimplePolicy
from simple_app.ppo import compute_gae, run_episodes, train

__all__ = ["SimplePolicy", "compute_gae", "run_episodes", "train"]