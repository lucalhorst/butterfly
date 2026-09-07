"""Command-line interface for the standalone simple_app trainer.

Usage:
    python -m simple_app.train --updates 200
    python -m simple_app.train --eval --checkpoint simple_app/checkpoints/simple_...pt --episodes 10
"""

import argparse

from butterfly.config import Config, load_config

__all__ = ["main"]


def _modify_training(config: Config, rollout_length=None, num_envs=None) -> Config:
    changes = {}
    if rollout_length is not None:
        changes["rollout_length"] = rollout_length
    if num_envs is not None:
        changes["num_envs"] = num_envs
    if not changes:
        return config
    return config.model_copy(
        update={"training": config.training.model_copy(update=changes)}
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description="Simple MLP+ResNet18 butterfly trainer")
    parser.add_argument("--config", default=None, help="Path to a butterfly TOML config")
    parser.add_argument("--seed", type=int, default=0, help="RNG seed")
    parser.add_argument("--device", default=None, help="torch device (default: auto)")

    parser.add_argument("--eval", action="store_true", help="Evaluate a checkpoint instead of training")
    parser.add_argument("--updates", type=int, default=100, help="Number of PPO updates")
    parser.add_argument("--rollout-length", type=int, default=None, help="Override rollout length")
    parser.add_argument("--num-envs", type=int, default=None, help="Override number of envs")
    parser.add_argument("--checkpoint", default=None, help="Train output path / eval input path")
    parser.add_argument("--resume", default=None, help="Resume training from an existing checkpoint")
    parser.add_argument("--episodes", type=int, default=5, help="Eval-only: episodes to run")
    parser.add_argument("--render", action="store_true", help="Eval-only: open a pygame window")
    parser.add_argument("--render-speed", type=float, default=20.0)

    args = parser.parse_args(argv)

    from simple_app.ppo import run_episodes, train

    config = load_config(args.config)

    if args.eval:
        if args.checkpoint is None:
            parser.error("--eval requires --checkpoint")
        run_episodes(
            config,
            args.checkpoint,
            episodes=args.episodes,
            seed=args.seed,
            render=args.render,
            render_speed=args.render_speed,
            device=args.device,
        )
    else:
        import torch
        import numpy as np

        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

        config = _modify_training(config, args.rollout_length, args.num_envs)

        train(
            config,
            total_updates=args.updates,
            output_model=args.checkpoint,
            resume_from=args.resume,
            device=args.device,
        )


if __name__ == "__main__":
    main()