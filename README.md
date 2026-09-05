# Butterfly AI

Reinforcement learning (PPO) project training a virtual butterfly agent to forage for flowers in a procedurally-generated, open-world 2D environment.

The agent learns from raw pixels (64x64 RGB) plus scalar inputs (hunger, nearby food, time of day, bird predator state) using a ResNet-18 visual encoder fused with a learned scalar encoder, processed through a Transformer over an 8-frame temporal history, outputting continuous 2D movement actions and a value estimate.

## Features

- **Procedural world** with chunk-based generation, day/night cycles, and 4 plant types (day, night, interval, random)
- **Bird predator** with roam/chase/return state machine
- **PPO training** with GAE, gradient clipping, and multi-process vectorized environments
- **ResNet-18 + Transformer** policy architecture with visual-scalar fusion
- **Attribution viewer** for analyzing what the agent attends to (saliency, scalar importance, attention weights)

## Installation

```bash
pip install -e .
```

Requires Python >= 3.10. Dependencies: PyTorch, torchvision, numpy, pygame-ce, pydantic, pillow.

## Usage

### Train

```bash
butterfly-train --config butterfly.toml --updates 1000
```

### Play (keyboard control)

```bash
butterfly-play --config butterfly.toml
```

### Run AI agent

```bash
butterfly-ai --weights models/checkpoint.pt --episodes 5
butterfly-ai --weights models/checkpoint.pt --render  # with pygame window
```

### Export attribution data

```bash
butterfly-attrib --checkpoint models/checkpoint.pt --output attribution.json
```

Then open `viewer/attribution_viewer.html` in a browser to explore the results.

## Configuration

All settings live in `butterfly.toml` with pydantic-validated defaults. Sections:

| Section | Description |
|---|---|
| `[training]` | PPO hyperparameters (rollout length, epochs, learning rate, etc.) |
| `[environment]` | Agent and env settings (hunger, speed, detection radius) |
| `[environment.rewards]` | Reward shaping (eating, movement, death penalties) |
| `[world]` | Chunk generation, day/night cycle, plant types |
| `[predator]` | Bird predator behavior (speed, detection, chase) |
| `[network]` | Model architecture (feature size, transformer config) |
| `[rendering]` | Pygame window settings |

GPU tier presets are provided in `gpu_04gb.toml` through `gpu_32gb.toml` — these override `num_envs`, `rollout_length`, `minibatch_size`, and checkpoint/log intervals for different VRAM budgets.

## Project Structure

```
butterfly/           Core library
  config.py          Pydantic config models + TOML loader
  utils.py           Shared helpers (seeding, progress bar, model filenames)
  env/               Environment module
    bird.py          Bird predator with state machine
    chunk.py         Procedural chunk generation
    world.py         World manager (chunk loading, plant logic)
    butterfly_env.py Main ButterflyEnv (reset/step/render)
    vec_env.py       Multi-process vectorized environments
  model/             Model module
    encoder.py       ResNet-18 visual encoder
    policy.py        ButterflyPolicy (actor-critic with Transformer)
    history.py       Observation history buffer
  algo/              Algorithm module
    ppo.py           PPO training loop + GAE
    run.py           Shared checkpoint loading + episode runner
apps/                Entry points
  train.py           Training app
  human_player.py    Keyboard-controlled play
  ai_player.py       Autonomous AI agent
  attribution_viewer.py  Attribution data export
viewer/
  attribution_viewer.html  Interactive HTML viewer for attribution data
```

## Checkpoints

Checkpoints are saved to `models/` with timestamped filenames: `YYYYMMDD_HHMMSS_<git-hash>_butterfly.pt`. Each checkpoint contains `{"policy": state_dict, "optimizer": state_dict}`.

## License

Not specified.
