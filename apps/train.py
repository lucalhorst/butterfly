"""Training app — PPO training of the butterfly agent.

Usage:
    butterfly-train --config butterfly.toml --updates 1000
"""

import argparse
import os

import torch

from butterfly.algo.ppo import train
from butterfly.config import build_config_arg_group, load_config, apply_cli_overrides
from butterfly.utils import DEVICE, seed_everything


def build_parser():
    parser = argparse.ArgumentParser(
        description="Virtual Butterfly RL agent - training",
    )

    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to TOML config file (default: butterfly.toml)",
    )

    parser.add_argument("--updates", type=int, default=1000)

    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help=(
            "Path to a checkpoint to resume training from (restores "
            "policy weights and optimizer state). The update counter "
            "always restarts at 0."
        ),
    )

    parser.add_argument("--seed", type=int, default=42)

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
        print(f"CPU device: using {thread_count} torch threads.")

    try:
        train(cfg, total_updates=args.updates, resume_from=args.resume)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
