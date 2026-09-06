"""AI player app — run a trained checkpoint autonomously.

Headless by default (prints per-episode stats); pass --render to open a
pygame window showing the agent play.

Usage:
    butterfly-ai --weights models/checkpoint.pt --episodes 5
    butterfly-ai --weights models/checkpoint.pt --render
"""

import argparse
import os

import torch

from butterfly.algo.run import load_policy, run_ai_episodes
from butterfly.config import load_config, build_config_arg_group, apply_cli_overrides
from butterfly.utils import DEVICE, newest_model, seed_everything


def build_parser():
    parser = argparse.ArgumentParser(
        description="Virtual Butterfly RL agent - run checkpoint",
    )

    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to TOML config file (default: butterfly.toml)",
    )

    parser.add_argument(
        "--weights",
        type=str,
        default=None,
        help=(
            "Weights file to load. If omitted, the newest model in "
            "models/ is used."
        ),
    )

    parser.add_argument("--episodes", type=int, default=1)

    parser.add_argument("--seed", type=int, default=None)

    parser.add_argument(
        "--render",
        action="store_true",
        help="Open a pygame window showing the agent playing.",
    )

    parser.add_argument(
        "--speed",
        type=float,
        default=None,
        help="Rendering speed (steps/second) when --render is set.",
    )

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

    if args.seed is None:
        seed = cfg.rendering.ai_player_seed if args.render else 123
    else:
        seed = args.seed

    seed_everything(seed)

    if DEVICE == "cpu":
        thread_count = args.threads if args.threads is not None else os.cpu_count() or 1
        torch.set_num_threads(thread_count)

    weights = args.weights if args.weights is not None else newest_model()
    print(f"Loading model: {weights}")
    policy = load_policy(cfg, weights)

    for stats in run_ai_episodes(
        cfg,
        policy,
        episodes=args.episodes,
        seed=seed,
        render=args.render,
        render_speed=args.speed,
    ):
        print(
            f"Episode {stats['episode'] + 1} | "
            f"food collected={stats['food_collected']} | "
            f"hunger={stats['hunger']:.2f} | "
            f"time_step={stats['time_step']} | "
            f"bird_state={stats['bird_state']} | "
            f"total reward={stats['total_reward']:.2f}"
        )


if __name__ == "__main__":
    main()
