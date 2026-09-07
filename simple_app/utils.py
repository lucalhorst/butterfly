"""Local console helpers mirroring the main trainer's stdout logging.

Kept local so simple_app keeps its "only butterfly.env / config imports"
contract instead of depending on butterfly.utils.
"""

import sys
import time

__all__ = ["clamp", "print_progress_bar"]


def clamp(value, low, high):
    return max(low, min(high, value))


def print_progress_bar(current, total, prefix="", bar_length=30, start_time=None):
    """Single-line in-place progress bar (same format as the main trainer).

    Call once per step; emit a newline (print()) after the loop so
    subsequent output starts on a fresh line. With ``start_time``
    (time.time() captured before the loop) also shows elapsed + ETA.
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