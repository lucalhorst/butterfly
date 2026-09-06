"""Shared utility helpers."""

import math
import os
import random
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

__all__ = [
    "DEVICE",
    "seed_everything",
    "clamp",
    "print_progress_bar",
    "get_git_model_label",
    "create_model_filename",
    "newest_model",
]


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def clamp(value, low, high):
    return max(low, min(high, value))


def print_progress_bar(current, total, prefix="", bar_length=30, start_time=None):
    """
    Prints a single-line, in-place progress bar using carriage returns.
    Call once per step; call print() (or otherwise emit a newline)
    after the loop finishes so subsequent output starts on a fresh line.

    If start_time (a time.time() captured before the loop began) is
    given, also shows elapsed time and an ETA for the remaining steps,
    based on the current average rate.
    """

    fraction = current / total if total else 1.0
    fraction = clamp(fraction, 0.0, 1.0)

    filled = int(bar_length * fraction)
    bar = "#" * filled + "-" * (bar_length - filled)

    timing = ""

    if start_time is not None:
        elapsed = time.time() - start_time

        if current > 0 and elapsed > 0:
            rate = current / elapsed
            remaining = (total - current) / rate if rate > 0 else 0.0

            timing = f" | {elapsed:5.1f}s elapsed, ETA {remaining:5.1f}s"

    sys.stdout.write(f"\r{prefix}[{bar}] {current}/{total}{timing}")
    sys.stdout.flush()


def run_git_command(*args):
    try:
        result = subprocess.run(
            ["git", *args],
            capture_output=True,
            text=True,
            check=True,
        )

        return result.stdout.strip()

    except (
        subprocess.CalledProcessError,
        FileNotFoundError,
    ):
        return None


def get_git_model_label():
    """
    Examples:

        v1.4.0
        v1.4.0_dirty
        a1b2c3d
        a1b2c3d_dirty
        nogit
    """

    # Prefer an exact tag on the current commit.
    git_tag = run_git_command(
        "describe",
        "--tags",
        "--exact-match",
        "--abbrev=0",
    )

    if git_tag:
        label = git_tag
    else:
        # Fall back to the short commit hash.
        git_hash = run_git_command(
            "rev-parse",
            "--short",
            "HEAD",
        )

        label = git_hash if git_hash else "nogit"

    # Include staged, unstaged, and untracked-file changes.
    status = run_git_command(
        "status",
        "--porcelain",
    )

    if status:
        label += "_dirty"

    # Avoid characters that are awkward in filenames.
    label = label.replace("/", "-")
    label = label.replace(" ", "-")

    return label


def create_model_filename():
    model_dir = Path("models")
    model_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    git_label = get_git_model_label()

    return model_dir / (f"{timestamp}_{git_label}_butterfly.pt")


def newest_model():
    """
    Returns the most recently modified *_butterfly.pt file in MODEL_DIR.

    Raises FileNotFoundError if no model files exist.
    """

    model_dir = Path("models")

    if not model_dir.exists():
        raise FileNotFoundError(f"Model directory does not exist: {model_dir}")

    candidates = sorted(
        model_dir.glob("*_butterfly.pt"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )

    if not candidates:
        raise FileNotFoundError(f"No model files found in: {model_dir}")

    return candidates[0]
