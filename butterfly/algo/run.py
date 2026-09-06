"""Shared logic for running a trained policy in the environment.

Used by both the rendered playback (human_player / ai_player --render) and
headless evaluation (ai_player). Handles checkpoint loading and stepping an
autonomous policy through episodes.
"""

from pathlib import Path

import torch

from butterfly.config import Config
from butterfly.env.environment import ButterflyEnv
from butterfly.model.history import HistoryBuffer
from butterfly.model.policy import ButterflyPolicy
from butterfly.utils import DEVICE

__all__ = ["load_policy", "run_ai_episodes"]


def load_policy(config: Config, weights, device=DEVICE) -> ButterflyPolicy:
    """Load a trained ButterflyPolicy from a checkpoint (plain state_dict
    or a ``{"policy": ...}`` dict) onto ``device``."""
    weights = Path(weights)

    if not weights.exists():
        raise FileNotFoundError(f"Weights file does not exist: {weights}")

    policy = ButterflyPolicy(config).to(device)

    checkpoint = torch.load(weights, map_location=device)

    if isinstance(checkpoint, dict) and "policy" in checkpoint:
        policy.load_state_dict(checkpoint["policy"])
    else:
        # Backward compatibility: older checkpoints were a raw policy
        # state_dict rather than a {"policy": ..., "optimizer": ...} dict.
        policy.load_state_dict(checkpoint)

    policy.eval()
    return policy


def run_ai_episodes(
    config: Config,
    policy,
    *,
    episodes,
    seed,
    device=DEVICE,
    render=False,
    render_speed=None,
):
    """Run ``episodes`` autonomous episodes of ``policy`` in the env.

    If ``render`` is True, opens a pygame window and displays the AI playing.
    Otherwise runs headless. Yields a per-episode stats dict.
    """
    if render:
        import time

        clock = _make_clock()
        speed = (
            render_speed if render_speed is not None else config.rendering.play_speed
        )

    env = ButterflyEnv(config=config, seed=seed, render=render)

    for episode in range(episodes):
        observation, scalar_input = env.reset()

        history = HistoryBuffer(config.environment.history_length)
        history.reset(observation, scalar_input)

        done = False
        cumulative_reward = 0.0

        while not done:
            with torch.no_grad():
                image_sequence, scalar_dict = history.tensors()

                image_sequence = image_sequence.unsqueeze(0).to(device)
                scalar_dict = {
                    key: tensor.unsqueeze(0).to(device)
                    for key, tensor in scalar_dict.items()
                }

                action, _, value = policy.sample_action(image_sequence, scalar_dict)

            action = action[0].cpu().numpy()

            env.display_stats = {
                "action": action.tolist(),
                "value": value[0].item(),
                "cumulative_reward": cumulative_reward,
            }

            next_observation, next_scalars, reward, terminated, truncated, info = (
                env.step(action)
            )

            cumulative_reward += reward

            history.append(next_observation, next_scalars)
            done = terminated or truncated

            if render:
                clock.tick(speed)

            if render and (not env.render_enabled):
                # User closed the window (or rendering was disabled mid-run);
                # stop early.
                done = True

        yield {
            "episode": episode,
            "food_collected": info["food_collected"],
            "hunger": info["hunger"],
            "time_step": info["time_step"],
            "bird_state": info["bird_state"],
            "total_reward": cumulative_reward,
        }


def _make_clock():
    import pygame

    pygame.init()
    return pygame.time.Clock()

