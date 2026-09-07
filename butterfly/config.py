"""Pydantic configuration models for the butterfly project.

Configuration is validated/coerced by pydantic and can be loaded from a
TOML file (default ``butterfly.toml``) merged over the schema defaults.

Groups are organized by entity (butterfly / bird / plant) plus cross-cutting
concerns (training, environment, world, network, rendering). Reward scalars
are grouped by the entity that receives them (always the butterfly), and the
butterfly's perception/tracking limits live under :class:`PerceptionConfig`.

CLI config overrides are derived automatically from the model schema
(:func:`_collect_fields`), so every leaf field gets a matching ``--group-field``
flag and grouping changes propagate to the CLI without hand-maintained maps.
"""

import argparse
from pathlib import Path
from typing import Any, Optional, Union

import tomllib
from pydantic import BaseModel, Field

__all__ = [
    "TrainingConfig",
    "EnvironmentConfig",
    "PerceptionConfig",
    "ButterflyRewardsConfig",
    "ButterflyConfig",
    "BirdConfig",
    "PlantConfig",
    "WorldConfig",
    "NetworkConfig",
    "TemporalTransformerConfig",
    "EntityTransformerConfig",
    "RenderingConfig",
    "Config",
    "FOOD_TYPE_TO_ID",
    "FOOD_PAD_ID",
    "FOOD_TYPE_EMBEDDING_COUNT",
    "BIRD_STATE_TO_ID",
    "BIRD_PAD_ID",
    "BIRD_STATE_EMBEDDING_COUNT",
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
    log_csv: bool = True
    log_tensorboard: bool = True
    tensorboard_logdir: Optional[str] = None


class EnvironmentConfig(BaseModel):
    image_size: int = 64
    pixels_per_unit: float = 1.0
    max_steps: int = 10000
    history_length: int = 8


class PerceptionConfig(BaseModel):
    detection_radius: float = 1.0
    track_limit: int = 15
    max_tracked_birds: int = 1


class ButterflyRewardsConfig(BaseModel):
    base_step: float = -0.002
    movement_scale: float = 0.02
    eating_distance: float = 0.055
    eating_reward: float = 2.0
    hunger_death_penalty: float = -5.0
    kill_penalty: float = -10.0


class ButterflyConfig(BaseModel):
    speed: float = 0.035
    initial_hunger: float = 1.0
    hunger_depletion_per_step: float = 0.002
    perception: PerceptionConfig = Field(default_factory=PerceptionConfig)
    rewards: ButterflyRewardsConfig = Field(default_factory=ButterflyRewardsConfig)


class BirdConfig(BaseModel):
    enabled: bool = True
    population: int = 1
    speed: float = 0.025
    detection_radius: float = 0.3
    chase_speed: float = 0.035
    patrol_range: float = 2.0
    spawn_radius: float = 50.0


class PlantConfig(BaseModel):
    plants_per_chunk: int = 20
    cooldown: int = 25
    types: list[str] = ["day", "night", "interval", "random"]
    type_probs: list[float] = [0.35, 0.35, 0.2, 0.1]
    types_enabled: dict[str, bool] = Field(
        default_factory=lambda: {
            "day": True,
            "night": True,
            "interval": True,
            "random": True,
        }
    )
    hunger_restore: float = 0.35
    interval_phases_max: int = 3


class WorldConfig(BaseModel):
    chunk_size: int = 16
    chunks_loaded: int = 4
    day_cycle_length: int = 100
    day_duration: int = 50
    night_duration: int = 50
    day_night_cycle_enabled: bool = True


# Food type <-> id mapping used both by the environment (to build the
# structured food_token observations) and by the model (to size/lookup the
# food type embedding). Hardcoded per design decision; PAD_ID sits after the
# real types and is used for the padding slots in a food token list that has
# fewer real tokens than food_track_limit.
FOOD_TYPE_TO_ID = {"day": 0, "night": 1, "interval": 2, "random": 3}
FOOD_PAD_ID = len(FOOD_TYPE_TO_ID)
FOOD_TYPE_EMBEDDING_COUNT = len(FOOD_TYPE_TO_ID) + 1

# Bird state <-> id mapping (see BirdState in env.entities). PAD_ID used for
# padding bird slots when fewer real birds exist than max_tracked_birds.
BIRD_STATE_TO_ID = {"roam": 0, "chase": 1, "return": 2}
BIRD_PAD_ID = len(BIRD_STATE_TO_ID)
BIRD_STATE_EMBEDDING_COUNT = len(BIRD_STATE_TO_ID) + 1


class TemporalTransformerConfig(BaseModel):
    heads: int = 8
    feedforward: int = 512
    dropout: float = 0.1
    layers: int = 3


class EntityTransformerConfig(BaseModel):
    heads: int = 2
    feedforward: int = 192
    dropout: float = 0.1
    layers: int = 1


class NetworkConfig(BaseModel):
    feature_size: int = 256

    # Temporal transformer: runs over the per-frame summary vectors across
    # the history_length window. (Formerly named transformer_*; renamed for
    # clarity once a per-frame entity transformer was added.)
    temporal_transformer: TemporalTransformerConfig = Field(
        default_factory=TemporalTransformerConfig
    )

    # Entity transformer: a separate, smaller transformer that runs once per
    # timestep over the image + context + food + bird entity tokens.
    entity_transformer: EntityTransformerConfig = Field(
        default_factory=EntityTransformerConfig
    )

    # Per-token embedding dims for the food type and bird state categorical
    # inputs (each +1 slot for the PAD id).
    food_type_embedding_dim: int = 8
    bird_state_embedding_dim: int = 8


class RenderingConfig(BaseModel):
    window_size: int = 600
    stats_panel_width: int = 350
    obs_panel_width: int = 140
    play_seed: int = 123
    play_speed: float = 20.0
    ai_player_seed: int = 123


class Config(BaseModel):
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    environment: EnvironmentConfig = Field(default_factory=EnvironmentConfig)
    butterfly: ButterflyConfig = Field(default_factory=ButterflyConfig)
    bird: BirdConfig = Field(default_factory=BirdConfig)
    plant: PlantConfig = Field(default_factory=PlantConfig)
    world: WorldConfig = Field(default_factory=WorldConfig)
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
# CLI config overrides (schema-driven)
# ============================================================


def _str_to_bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("true", "1", "yes"):
        return True
    if v.lower() in ("false", "0", "no"):
        return False
    raise argparse.ArgumentTypeError(f"Boolean value expected, got {v!r}")


def _collect_fields(model: type[BaseModel], prefix: str = "") -> list[tuple[str, str, Any]]:
    """Recursively collect ``(dotted_path, field_name, annotation)`` tuples.

    Nested BaseModels are walked depth-first, so a field ``heads`` inside
    ``network.temporal_transformer`` yields ``("network.temporal_transformer.heads",
    ...)``. ``Optional[T]`` annotations are unwrapped to ``T`` so the argtype
    resolver sees the concrete scalar type.
    """
    fields = []
    for name, field_info in model.model_fields.items():
        dotted = f"{prefix}.{name}" if prefix else name
        annotation = field_info.annotation
        origin = getattr(annotation, "__origin__", None)
        if origin is Union:
            args = [a for a in getattr(annotation, "__args__", ()) if a is not type(None)]
            if args:
                annotation = args[0]
        if isinstance(annotation, type) and issubclass(annotation, BaseModel):
            fields.extend(_collect_fields(annotation, dotted))
        else:
            fields.append((dotted, name, annotation))
    return fields


def _type_to_argtype(annotation):
    """Map a config field's runtime annotation to an argparse type converter.

    Returns None for complex types (list/dict) that don't map to a scalar
    CLI flag.
    """
    if annotation is bool:
        return _str_to_bool
    if annotation is int:
        return int
    if annotation is float:
        return float
    if annotation is str:
        return str
    return None


def build_config_arg_group(parser):
    """Register the per-section config override flags on ``parser``.

    One argparse argument group per top-level config section is created. The
    flags are derived from the model schema via :func:`_collect_fields`, so
    adding/renaming/regrouping a field updates the CLI automatically.
    """
    sections = {}
    for dotted_key, _name, annotation in _collect_fields(Config):
        arg_type = _type_to_argtype(annotation)
        if arg_type is None:
            continue
        section = dotted_key.split(".")[0]
        sections.setdefault(section, []).append((dotted_key, arg_type))

    for section, fields in sections.items():
        group = parser.add_argument_group(section)
        for dotted_key, arg_type in fields:
            group.add_argument(
                f"--{dotted_key.replace('.', '-').replace('_', '-')}",
                dest=dotted_key.replace(".", "_"),
                type=arg_type,
                default=None,
            )

    plant = parser.add_argument_group("plant")
    plant.add_argument(
        "--plant.enabled-types",
        dest="plant_enabled_types",
        type=str,
        default=None,
        help="Comma-separated list of enabled plant types (e.g. 'day,night,interval'). Overrides plant.types_enabled.",
    )


def apply_cli_overrides(config: Config, args) -> Config:
    """Apply CLI config overrides (from argparse ``args``) onto ``config``.

    If ``args.plant_enabled_types`` is set, it overrides
    ``plant.types_enabled``. Returns a new Config with the merged
    values applied.
    """
    dest_to_config = {
        dotted_key.replace(".", "_"): dotted_key
        for dotted_key, _name, _annotation in _collect_fields(Config)
    }

    overrides = {}
    for dest, value in vars(args).items():
        if value is None or dest not in dest_to_config:
            continue
        overrides[dest_to_config[dest]] = value

    enabled_types_str = getattr(args, "plant_enabled_types", None)
    if enabled_types_str is not None:
        enabled = [t.strip() for t in enabled_types_str.split(",")]
        all_types = config.plant.types
        overrides["plant.types_enabled"] = {t: (t in enabled) for t in all_types}

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