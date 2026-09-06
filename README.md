# Butterfly AI

Reinforcement learning (PPO) project training a virtual butterfly agent to forage for flowers in a procedurally-generated, open-world 2D environment.

The agent learns from raw pixels (64x64 RGB) plus scalar inputs (hunger, nearby food, time of day, bird predator state) using a ResNet-18 visual encoder fused with an entity encoder, processed through a Transformer over an 8-frame temporal history, outputting continuous 2D movement actions and a value estimate.

## Features

- **Procedural world** with chunk-based generation, day/night cycles, and 4 plant types (day, night, interval, random)
- **Bird predator** with roam/chase/return state machine
- **PPO training** with GAE, gradient clipping, and multi-process vectorized environments
- **ResNet-18 + Transformer** policy architecture with visual-scalar fusion
- **Entity transformer** encoding food and predator features with learned type embeddings
- **Config override system** — every config field overridable via CLI flags (`--section-key value`)
- **TensorBoard + CSV logging** with per-episode metric summaries
- **Attribution viewer** for analyzing what the agent attends to (saliency, scalar importance, attention weights)
- **GPU tier presets** tuned for 4 GB to 32 GB VRAM

## Installation

```bash
pip install -e .
```

Requires Python >= 3.10. Dependencies: PyTorch, torchvision, numpy, pygame-ce, pydantic, pillow, tensorboard.

## Usage

### Train

```bash
butterfly-train --config butterfly.toml --updates 1000
```

| Flag | Description |
|---|---|
| `--config` | Path to TOML config (default: `butterfly.toml`) |
| `--updates` | Number of PPO updates (default: 1000) |
| `--resume` | Path to checkpoint to resume from |
| `--seed` | Random seed (default: 42) |
| `--threads` | Number of worker threads |

### Play (keyboard control)

```bash
butterfly-play --config butterfly.toml
```

Controls: Arrow keys / WASD to move, TAB toggle stats, F1 toggle debug panel, ESC quit.

### Run AI agent

```bash
butterfly-ai --weights models/checkpoint.pt --episodes 5
butterfly-ai --weights models/checkpoint.pt --render  # with pygame window
```

If `--weights` is omitted, the newest checkpoint in `models/` is used automatically.

### Export attribution data

```bash
butterfly-attrib --checkpoint models/checkpoint.pt --output attribution.json
```

| Flag | Description |
|---|---|
| `--checkpoint` | Path to model checkpoint (required) |
| `--output` | Output JSON path (default: `attribution.json`) |
| `--steps` | Number of steps to record (default: 30) |
| `--seed` | Random seed (default: 123) |
| `--saliency-res` | Saliency heatmap resolution (default: 16) |
| `--stochastic` | Use stochastic policy instead of deterministic |

Then open `viewer/attribution_viewer.html` in a browser to explore the results.

### CLI config overrides

Every config field can be overridden from the command line using `--section-key value`:

```bash
butterfly-train --training-learning-rate 1e-4 --environment-max-birds 3 --world-day-cycle-length 200
```

## Configuration

All settings live in `butterfly.toml` with pydantic-validated defaults. Sections:

| Section | Description |
|---|---|
| `[training]` | PPO hyperparameters — rollout length, epochs, learning rate, num_envs, GAE, gradient clipping, checkpoint/log intervals |
| `[environment]` | Agent and env settings — image size, max steps, history length, hunger, speed, detection radius, max birds |
| `[environment.rewards]` | Reward shaping — eating, movement, hunger death, bird kill penalties |
| `[world]` | Chunk generation, day/night cycle, plant types and probabilities |
| `[world.plant_types_enabled]` | Toggle individual plant types on/off |
| `[predator]` | Bird predator behavior — speed, detection range, chase speed, patrol range |
| `[network]` | Model architecture — feature size, temporal transformer config, entity transformer config, embedding dimensions |
| `[rendering]` | Pygame window — window size, stats panel width, play speed |

## GPU Presets

Preset files override `[training]` and `[network]` for different VRAM budgets:

| Preset | VRAM | Example Hardware | num_envs | rollout_length | minibatch_size |
|---|---|---|---|---|---|
| `gpu_04gb.toml` | ~4 GB | GTX 1650, RTX 3050 laptop | 8 | 128 | 128 |
| `gpu_08gb.toml` | ~8 GB | RTX 3060, RTX 4060, RTX 2070 | 16 | 256 | 256 |
| `gpu_16gb.toml` | ~16 GB | RTX 4080, RTX 4070 Ti Super | 32 | 512 | 512 |
| `gpu_24gb.toml` | ~24 GB | RTX 3090, RTX 4090, A5000 | 48 | 512 | 1024 |
| `gpu_32gb.toml` | ~32 GB | V100 32 GB, A100, RTX 5090 | 64 | 1024 | 2048 |

```bash
butterfly-train --config gpu_08gb.toml --updates 2000
```

## Project Structure

```
butterfly/                Core library
  config.py              Pydantic config models + TOML loader
  utils.py               Shared helpers (seeding, progress bar, model filenames)
  metrics.py             CSV and TensorBoard metric logging
  env/                   Environment module
    entities.py          Bird, BirdState, Chunk, WorldManager
    environment.py       ButterflyEnv (reset / step / render)
    vec_env.py           Multi-process vectorized environments
  model/                 Model module
    encoder.py           ResNet-18 visual encoder
    policy.py            ButterflyPolicy (actor-critic) + EntityEncoder
    history.py           Observation history buffer
  algo/                  Algorithm module
    ppo.py               PPO training loop + GAE
    run.py               Shared checkpoint loading + episode runner
apps/                    Entry points
  train.py               Training app
  human_player.py        Keyboard-controlled play
  ai_player.py           Autonomous AI agent
  attribution_viewer.py  Attribution data export
viewer/
  attribution_viewer.html  Interactive HTML viewer for attribution data
```

## Checkpoints

Checkpoints are saved to `models/` with timestamped filenames: `YYYYMMDD_HHMMSS_<git-hash>_butterfly.pt`. Each checkpoint contains `{"policy": state_dict, "optimizer": state_dict}`. A companion `.csv` file is saved alongside each checkpoint with training metrics.

## License

Not specified.
