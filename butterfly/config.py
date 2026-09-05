"""Pydantic configuration models for the butterfly project.

Configuration is validated/coerced by pydantic and can be loaded from a
TOML file (default ``butterfly.toml``) merged over the schema defaults.
"""

import argparse
from pathlib import Path
from typing import Any, Optional

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
    "build_config_arg_group",
    "apply_cli_overrides",
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


# ============================================================
# CLI config overrides
# ============================================================


def _str_to_bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("true", "1", "yes"):
        return True
    if v.lower() in ("false", "0", "no"):
        return False
    raise argparse.ArgumentTypeError(f"Boolean value expected, got {v!r}")


# Group name -> {argparse dest: dotted config path}.
CONFIG_ARG_DESTS = {
    "training": {
        "training_rollout_length": "training.rollout_length",
        "training_ppo_epochs": "training.ppo_epochs",
        "training_minibatch_size": "training.minibatch_size",
        "training_gamma": "training.gamma",
        "training_gae_lambda": "training.gae_lambda",
        "training_clip_epsilon": "training.clip_epsilon",
        "training_learning_rate": "training.learning_rate",
        "training_entropy_coef": "training.entropy_coef",
        "training_value_coef": "training.value_coef",
        "training_num_envs": "training.num_envs",
        "training_gradient_max_norm": "training.gradient_max_norm",
        "training_checkpoint_interval": "training.checkpoint_interval",
        "training_log_interval": "training.log_interval",
    },
    "environment": {
        "environment_image_size": "environment.image_size",
        "environment_max_steps": "environment.max_steps",
        "environment_history_length": "environment.history_length",
        "environment_food_track_limit": "environment.food_track_limit",
        "environment_food_detection_radius": "environment.food_detection_radius",
        "environment_initial_hunger": "environment.initial_hunger",
        "environment_hunger_depletion_per_step": "environment.hunger_depletion_per_step",
        "environment_food_hunger_restore": "environment.food_hunger_restore",
        "environment_butterfly_speed": "environment.butterfly_speed",
    },
    "environment.rewards": {
        "environment_rewards_base_step": "environment.rewards.base_step",
        "environment_rewards_movement_scale": "environment.rewards.movement_scale",
        "environment_rewards_eating_distance": "environment.rewards.eating_distance",
        "environment_rewards_eating_reward": "environment.rewards.eating_reward",
        "environment_rewards_hunger_death_penalty": "environment.rewards.hunger_death_penalty",
        "environment_rewards_bird_kill_penalty": "environment.rewards.bird_kill_penalty",
    },
    "world": {
        "world_chunk_size": "world.chunk_size",
        "world_chunks_loaded": "world.chunks_loaded",
        "world_plants_per_chunk": "world.plants_per_chunk",
        "world_day_cycle_length": "world.day_cycle_length",
        "world_day_duration": "world.day_duration",
        "world_night_duration": "world.night_duration",
        "world_plant_cooldown": "world.plant_cooldown",
        "world_interval_phases_max": "world.interval_phases_max",
        "world_day_night_cycle_enabled": "world.day_night_cycle_enabled",
    },
    "predator": {
        "predator_enabled": "predator.enabled",
        "predator_speed": "predator.speed",
        "predator_detection_range": "predator.detection_range",
        "predator_chase_speed": "predator.chase_speed",
        "predator_patrol_range": "predator.patrol_range",
    },
    "network": {
        "network_feature_size": "network.feature_size",
        "network_transformer_heads": "network.transformer_heads",
        "network_transformer_feedforward": "network.transformer_feedforward",
        "network_transformer_dropout": "network.transformer_dropout",
        "network_transformer_layers": "network.transformer_layers",
    },
    "rendering": {
        "rendering_window_size": "rendering.window_size",
        "rendering_play_seed": "rendering.play_seed",
    },
}

# Dest -> type for the CLI args, so argparse can coerce values.
CONFIG_ARG_TYPES = {
    "training_rollout_length": int,
    "training_ppo_epochs": int,
    "training_minibatch_size": int,
    "training_gamma": float,
    "training_gae_lambda": float,
    "training_clip_epsilon": float,
    "training_learning_rate": float,
    "training_entropy_coef": float,
    "training_value_coef": float,
    "training_num_envs": int,
    "training_gradient_max_norm": float,
    "training_checkpoint_interval": int,
    "training_log_interval": int,
    "environment_image_size": int,
    "environment_max_steps": int,
    "environment_history_length": int,
    "environment_food_track_limit": int,
    "environment_food_detection_radius": float,
    "environment_initial_hunger": float,
    "environment_hunger_depletion_per_step": float,
    "environment_food_hunger_restore": float,
    "environment_butterfly_speed": float,
    "environment_rewards_base_step": float,
    "environment_rewards_movement_scale": float,
    "environment_rewards_eating_distance": float,
    "environment_rewards_eating_reward": float,
    "environment_rewards_hunger_death_penalty": float,
    "environment_rewards_bird_kill_penalty": float,
    "world_chunk_size": int,
    "world_chunks_loaded": int,
    "world_plants_per_chunk": int,
    "world_day_cycle_length": int,
    "world_day_duration": int,
    "world_night_duration": int,
    "world_plant_cooldown": int,
    "world_interval_phases_max": int,
    "world_day_night_cycle_enabled": _str_to_bool,
    "predator_enabled": _str_to_bool,
    "predator_speed": float,
    "predator_detection_range": float,
    "predator_chase_speed": float,
    "predator_patrol_range": float,
    "network_feature_size": int,
    "network_transformer_heads": int,
    "network_transformer_feedforward": int,
    "network_transformer_dropout": float,
    "network_transformer_layers": int,
    "rendering_window_size": int,
    "rendering_play_seed": int,
}


def build_config_arg_group(parser):
    """Register the per-section config override flags on ``parser``.

    Returns an argparse argument group containing the overrides.
    """
    for group_name in CONFIG_ARG_DESTS:
        group = parser.add_argument_group(group_name)
        for dest in CONFIG_ARG_DESTS[group_name]:
            arg_type = CONFIG_ARG_TYPES[dest]
            group.add_argument(
                f"--{dest.replace('_', '-')}", type=arg_type, default=None
            )

    world = parser.add_argument_group("world")
    world.add_argument(
        "--world.enabled-plant-types",
        dest="world_enabled_plant_types",
        type=str,
        default=None,
        help="Comma-separated list of enabled plant types (e.g. 'day,night,interval'). Overrides plant_types_enabled.",
    )


def apply_cli_overrides(config: Config, args) -> Config:
    """Apply CLI config overrides (from argparse ``args``) onto ``config``.

    If ``args.world_enabled_plant_types`` is set, it overrides
    ``world.plant_types_enabled``. Returns a new Config with the merged
    values applied.
    """
    dest_to_config = {}
    for mapping in CONFIG_ARG_DESTS.values():
        dest_to_config.update(mapping)

    overrides = {}
    for dest, value in vars(args).items():
        if value is None or dest not in dest_to_config:
            continue
        overrides[dest_to_config[dest]] = value

    enabled_plant_types_str = getattr(args, "world_enabled_plant_types", None)
    if enabled_plant_types_str is not None:
        enabled = [t.strip() for t in enabled_plant_types_str.split(",")]
        all_types = config.world.plant_types
        overrides["world.plant_types_enabled"] = {t: (t in enabled) for t in all_types}

    if not overrides:
        return config

    data = config.model_dump()
    for dotted_key, value in overrides.items():
        parts = dotted_key.split(".")
        obj = data
        for part in parts[:-1]:
            obj = obj[part]
        obj[parts[-1]] = value

    return Config.model_validate(data)
