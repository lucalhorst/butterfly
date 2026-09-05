"""Pydantic configuration models for the butterfly project.

Configuration is validated/coerced by pydantic and can be loaded from a
TOML file (default ``butterfly.toml``) merged over the schema defaults.
"""

from pathlib import Path
from typing import Optional

import tomllib
from pydantic import BaseModel, Field

__all__ = [
    "TrainingConfig",
    "EnvironmentConfig",
    "RewardsConfig",
    "WorldConfig",
    "PredatorConfig",
    "NetworkConfig",
    "RenderingConfig",
    "Config",
    "load_config",
]


class TrainingConfig(BaseModel):
    rollout_length: int = 1000
    ppo_epochs: int = 4
    minibatch_size: int = 256
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.2
    learning_rate: float = 3e-4
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    num_envs: int = 4
    gradient_max_norm: float = 0.5
    checkpoint_interval: int = 5
    log_interval: int = 5


class RewardsConfig(BaseModel):
    base_step: float = -0.002
    movement_scale: float = 0.02
    eating_distance: float = 0.055
    eating_reward: float = 2.0
    hunger_death_penalty: float = -5.0
    bird_kill_penalty: float = -10.0


class EnvironmentConfig(BaseModel):
    image_size: int = 64
    max_steps: int = 10000
    history_length: int = 8
    food_track_limit: int = 15
    food_detection_radius: float = 1.0
    initial_hunger: float = 1.0
    hunger_depletion_per_step: float = 0.002
    food_hunger_restore: float = 0.35
    butterfly_speed: float = 0.035
    rewards: RewardsConfig = Field(default_factory=RewardsConfig)

    def scalar_input_size(self) -> int:
        """Length of the scalar feature vector produced by
        ``ButterflyEnv.get_scalar_inputs`` (hunger + per-food triples +
        time + is_daytime + 5 bird features + bird_detected)."""
        return 1 + self.food_track_limit * 3 + 1 + 1 + 5 + 1


class WorldConfig(BaseModel):
    chunk_size: int = 16
    chunks_loaded: int = 4
    plants_per_chunk: int = 20
    day_cycle_length: int = 100
    day_duration: int = 50
    night_duration: int = 50
    plant_cooldown: int = 25
    interval_phases_max: int = 3
    day_night_cycle_enabled: bool = True
    plant_types: list[str] = ["day", "night", "interval", "random"]
    plant_type_probs: list[float] = [0.35, 0.35, 0.2, 0.1]
    plant_types_enabled: dict[str, bool] = Field(
        default_factory=lambda: {
            "day": True,
            "night": True,
            "interval": True,
            "random": True,
        }
    )


class PredatorConfig(BaseModel):
    enabled: bool = True
    speed: float = 0.025
    detection_range: float = 0.3
    chase_speed: float = 0.035
    patrol_range: float = 2.0


class NetworkConfig(BaseModel):
    feature_size: int = 256
    transformer_heads: int = 8
    transformer_feedforward: int = 512
    transformer_dropout: float = 0.1
    transformer_layers: int = 3


class RenderingConfig(BaseModel):
    window_size: int = 600
    play_seed: int = 123
    play_speed: float = 20.0
    ai_player_seed: int = 123


class Config(BaseModel):
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    environment: EnvironmentConfig = Field(default_factory=EnvironmentConfig)
    world: WorldConfig = Field(default_factory=WorldConfig)
    predator: PredatorConfig = Field(default_factory=PredatorConfig)
    network: NetworkConfig = Field(default_factory=NetworkConfig)
    rendering: RenderingConfig = Field(default_factory=RenderingConfig)


def load_config(config_path: Optional[str] = None) -> Config:
    """Load configuration from a TOML file, falling back to defaults.

    If ``config_path`` is given and exists, its contents are parsed and
    merged over the schema defaults; otherwise the default ``butterfly.toml``
    in the current directory is used if present. Fields not present in the
    file keep their pydantic defaults.
    """
    toml_data = {}

    path = None
    if config_path is not None:
        path = Path(config_path)
    else:
        default_path = Path("butterfly.toml")
        if default_path.exists():
            path = default_path

    if path is not None and path.exists():
        with open(path, "rb") as f:
            toml_data = tomllib.load(f)

    return Config.model_validate(toml_data)
