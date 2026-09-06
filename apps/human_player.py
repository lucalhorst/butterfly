"""Human player app — keyboard-controlled butterfly in a rendered environment.

Controls:
    Arrow keys / WASD   move the butterfly
    TAB                 toggle full stats
    F1                  toggle the debug stats panel
    ESC / window close  quit

Usage:
    butterfly-play --config butterfly.toml
"""

import argparse
import os

import numpy as np
import torch

from butterfly.config import apply_cli_overrides, build_config_arg_group, load_config
from butterfly.env.environment import ButterflyEnv
from butterfly.utils import DEVICE, seed_everything


def read_keyboard_action(speed, max_speed):
    """Return a 2D normalized action from the arrow keys / WASD held state.

    Returns None if there is no movement input.
    """
    import pygame

    keys = pygame.key.get_pressed()

    dx = 0.0
    dy = 0.0

    if keys[pygame.K_LEFT] or keys[pygame.K_a]:
        dx -= 1.0
    if keys[pygame.K_RIGHT] or keys[pygame.K_d]:
        dx += 1.0
    if keys[pygame.K_UP] or keys[pygame.K_w]:
        dy -= 1.0
    if keys[pygame.K_DOWN] or keys[pygame.K_s]:
        dy += 1.0

    if dx == 0.0 and dy == 0.0:
        return None

    # Normalize so diagonal movement isn't faster than cardinal.
    mag = float(np.hypot(dx, dy))
    return np.array([dx / mag, dy / mag], dtype=np.float32)


def build_parser():
    parser = argparse.ArgumentParser(
        description="Virtual Butterfly RL agent - human play",
    )

    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to TOML config file (default: butterfly.toml)",
    )

    parser.add_argument("--seed", type=int, default=123)

    parser.add_argument(
        "--threads",
        type=int,
        default=None,
        help=(
            "Number of CPU threads for torch to use (only relevant when "
            "running without CUDA). Defaults to os.cpu_count()."
        ),
    )

    build_config_arg_group(parser)

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    cfg = load_config(args.config)
    cfg = apply_cli_overrides(cfg, args)

    seed_everything(args.seed)

    if DEVICE == "cpu":
        thread_count = args.threads if args.threads is not None else os.cpu_count() or 1
        torch.set_num_threads(thread_count)

    import pygame

    pygame.init()

    env = ButterflyEnv(config=cfg, seed=args.seed, render=True)
    env.human_control = True

    clock = pygame.time.Clock()

    try:
        while env.render_enabled:
            observation, scalar_input = env.reset()
            done = False
            cumulative_reward = 0.0

            while not done and env.render_enabled:
                env.display_stats = {
                    "action": None,
                    "value": 0.0,
                    "cumulative_reward": cumulative_reward,
                }

                action = read_keyboard_action(
                    cfg.environment.butterfly_speed, cfg.environment.butterfly_speed
                )

                if action is None:
                    action = np.zeros(2, dtype=np.float32)

                next_obs, next_scalars, reward, terminated, truncated, info = env.step(
                    action
                )
                cumulative_reward += reward
                done = terminated or truncated

                clock.tick(60)

            if env.render_enabled:
                print(
                    "Episode finished | "
                    f"food collected={info['food_collected']} | "
                    f"hunger={info['hunger']:.2f} | "
                    f"time_step={info['time_step']} | "
                    f"bird_state={info['bird_state']} | "
                    f"total reward={cumulative_reward:.2f}"
                )
    finally:
        pygame.quit()


if __name__ == "__main__":
    main()
