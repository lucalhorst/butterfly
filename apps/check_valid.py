import argparse
import os
from pathlib import Path

from butterfly.algo.run import load_policy
from butterfly.config import apply_cli_overrides, build_config_arg_group, load_config


def build_parser():
    parser = argparse.ArgumentParser(
        description="Virtual Butterfly RL agent - test checkpoint if they match the model",
    )

    parser.add_argument(
        "--weight-folder",
        type=str,
        default="./configs/",
        help=("folder of the model checkpoints. If omitted, the folder is models/."),
    )
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    cfg = load_config(args.config)
    cfg = apply_cli_overrides(cfg, args)

    for file in os.walk(Path(args.weight_folder)):
        policy = load_policy(cfg, file)
        del policy


if __name__ == "__main__":
    main()
