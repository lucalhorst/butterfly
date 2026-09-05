"""Training metric logging: CSV persistence + TensorBoard scalar mirror.

``METRIC_TAGS`` is the single source of truth for which metrics are
collected each PPO update. It drives both the CSV column order and the
TensorBoard scalar tags so the two channels never drift apart.
"""

import csv
from datetime import datetime
from pathlib import Path

from butterfly.utils import get_git_model_label

__all__ = [
    "METRIC_TAGS",
    "CsvMetricLogger",
    "create_metrics_csv_path",
    "create_tensorboard_logdir",
]

METRIC_TAGS = [
    "update",
    "timestamp",
    "rollout_time_s",
    "backprop_time_s",
    "update_time_s",
    "rollout_steps_s",
    "reward_mean",
    "reward_std",
    "reward_min",
    "reward_max",
    "return_mean",
    "return_std",
    "return_min",
    "return_max",
    "episode_count",
    "episode_return_mean",
    "value_mean",
    "value_std",
    "value_min",
    "value_max",
    "advantage_mean",
    "advantage_std",
    "advantage_min",
    "advantage_max",
    "loss_total",
    "loss_policy",
    "loss_value",
    "loss_entropy",
    "entropy",
    "log_prob_mean",
    "kl_approx",
    "clip_fraction",
    "ratio_mean",
    "grad_norm",
    "action_sat",
    "action_mag_mean",
    "action_mag_max",
    "action_mean_x",
    "action_mean_y",
]


def create_metrics_csv_path(output_model: Path) -> Path:
    """Default CSV path sits next to the checkpoint, same timestamp+git stem."""
    return output_model.with_suffix(".csv")


def create_tensorboard_logdir() -> Path:
    """Default TensorBoard run dir: ``runs/<timestamp>_<git>``."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    git_label = get_git_model_label().replace("/", "-").replace(" ", "-")
    return Path("runs") / f"{timestamp}_{git_label}"


class CsvMetricLogger:
    """Writes one row per PPO update to a CSV file.

    Opening is append-aware: if the target file already exists (e.g. a
    resumed run against an explicit checkpoint), rows are appended and the
    header is not rewritten. Each row is flushed immediately so an
    in-progress run can be ``tail -f``-ed.
    """

    def __init__(self, path):
        self.path = Path(path)
        self._file = None
        self._writer = None

    def open(self):
        if self._file is not None:
            return self

        self.path.parent.mkdir(parents=True, exist_ok=True)

        write_header = not self.path.exists() or self.path.stat().st_size == 0

        self._file = open(self.path, "a", newline="", encoding="utf-8")

        if write_header:
            self._writer = csv.DictWriter(self._file, fieldnames=METRIC_TAGS)
            self._writer.writeheader()
            self._file.flush()
        else:
            self._writer = csv.DictWriter(self._file, fieldnames=METRIC_TAGS)

        return self

    def log(self, row: dict[str, Any]):
        if self._writer is None:
            self.open()

        # Only keep known columns; tolerate missing/extra keys.
        csv_row = {tag: row.get(tag, "") for tag in METRIC_TAGS}
        self._writer.writerow(csv_row)
        self._file.flush()

    def close(self):
        if self._file is not None:
            self._file.close()
            self._file = None
            self._writer = None

    def __enter__(self):
        return self.open()

    def __exit__(self, *exc):
        self.close()