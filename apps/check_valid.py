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
        "--config",
        type=str,
        default=None,
        help="Path to TOML config file (default: butterfly.toml)",
    )
    parser.add_argument(
        "--weight-folder",
        type=str,
        default="./models/",
        help="Folder of the model checkpoints (default: ./models/)",
    )
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    cfg = load_config(args.config)
    cfg = apply_cli_overrides(cfg, args)

    weight_folder = Path(args.weight_folder)
    for root, _, files in os.walk(weight_folder):
        for file in files:
            if file.endswith((".pt", ".pth", ".ckpt")):
                checkpoint_path = os.path.join(root, file)
                print(f"Testing model: {checkpoint_path}.".ljust(100), end=" ")
                try:
                    policy = load_policy(cfg, checkpoint_path)
                    print(f"model can be loaded")
                    # Optionally: Add validation logic here
                except Exception as e:
                    print(f"loading failed")


if __name__ == "__main__":
    main()
