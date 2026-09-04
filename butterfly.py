import argparse
import math
import multiprocessing as mp
import random
import subprocess
import sys
from collections import deque
from datetime import datetime
from pathlib import Path

import numpy as np
import pygame
import torch
import torch.nn as nn
import torch.optim as optim

# ============================================================
# Configuration
# ============================================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

IMAGE_SIZE = 64
WORLD_SIZE = 1.0

MAX_STEPS = 500
FOOD_COUNT = 12
HISTORY_LENGTH = 8

NUM_ENVS = 4
ROLLOUT_LENGTH = 128
PPO_EPOCHS = 4
MINIBATCH_SIZE = 256

GAMMA = 0.99
GAE_LAMBDA = 0.95
CLIP_EPSILON = 0.2
LEARNING_RATE = 3e-4
ENTROPY_COEF = 0.01
VALUE_COEF = 0.5

MODEL_DIR = Path("models")

FOOD_TRACK_LIMIT = 10
FOOD_DETECTION_RADIUS = 0.45

INITIAL_HUNGER = 1.0
HUNGER_DEPLETION_PER_STEP = 0.002
FOOD_HUNGER_RESTORE = 0.35

SCALAR_INPUT_SIZE = 1 + FOOD_TRACK_LIMIT * 2

# ============================================================
# Utility functions
# ============================================================


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def clamp(value, low, high):
    return max(low, min(high, value))


def print_progress_bar(current, total, prefix="", bar_length=30):
    """
    Prints a single-line, in-place progress bar using carriage returns.
    Call once per step; call print() (or otherwise emit a newline)
    after the loop finishes so subsequent output starts on a fresh line.
    """

    fraction = current / total if total else 1.0
    fraction = clamp(fraction, 0.0, 1.0)

    filled = int(bar_length * fraction)
    bar = "#" * filled + "-" * (bar_length - filled)

    sys.stdout.write(f"\r{prefix}[{bar}] {current}/{total}")
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
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    git_label = get_git_model_label()

    return MODEL_DIR / (f"{timestamp}_{git_label}_butterfly.pt")


def newest_model():
    """
    Returns the most recently modified *_butterfly.pt file in MODEL_DIR.

    Raises FileNotFoundError if no model files exist.
    """

    if not MODEL_DIR.exists():
        raise FileNotFoundError(f"Model directory does not exist: {MODEL_DIR}")

    candidates = sorted(
        MODEL_DIR.glob("*_butterfly.pt"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )

    if not candidates:
        raise FileNotFoundError(f"No model files found in: {MODEL_DIR}")

    return candidates[0]


# ============================================================
# Virtual butterfly environment
# ============================================================


class ButterflyEnv:
    """
    A simple 2D world.

    The butterfly must search for flowers and collect nectar.
    Observation: RGB image from the butterfly's point of view, PLUS a
    scalar vector of hunger + angle/distance to nearby food.

    NOTE (documented tradeoff, not a bug): the image and scalar
    observations are largely redundant -- both encode the relative
    position of nearby food, just in different formats. This is kept
    intentionally (it can help the visual encoder learn useful
    features, and mirrors how partial/full-precision sensors might
    coexist in a real system) rather than "fixed", since collapsing
    them into one modality would change the task itself. Worth
    knowing if you're debugging why the two encoders learn similar
    things.
    """

    def __init__(self, seed=None, render=False):
        self.rng = np.random.default_rng(seed)
        self.render_enabled = render

        self.width = 1.0
        self.height = 1.0

        self.hunger = INITIAL_HUNGER

        self.butterfly = np.zeros(2, dtype=np.float32)
        self.food = []
        self.collected = 0
        self.steps = 0

        self.window = None
        self.clock = None

    def reset(self):
        self.butterfly = self.rng.uniform(low=0.15, high=0.85, size=2).astype(
            np.float32
        )

        self.food = [
            self.rng.uniform(0.05, 0.95, size=2).astype(np.float32)
            for _ in range(FOOD_COUNT)
        ]

        self.collected = 0
        self.steps = 0
        self.hunger = INITIAL_HUNGER

        observation = self.render_observation()
        scalar_inputs = self.get_scalar_inputs()

        return observation, scalar_inputs

    def step(self, action):
        self.steps += 1

        action = np.asarray(action, dtype=np.float32)
        action = np.clip(action, -1.0, 1.0)

        old_position = self.butterfly.copy()

        speed = 0.035
        self.butterfly += action * speed
        self.butterfly = np.clip(self.butterfly, 0.02, 0.98)

        # Hunger continuously depletes.
        self.hunger -= HUNGER_DEPLETION_PER_STEP
        self.hunger = max(0.0, self.hunger)

        reward = -0.002

        movement = np.linalg.norm(self.butterfly - old_position)

        reward += float(movement) * 0.02

        remaining_food = []

        for item in self.food:
            distance = np.linalg.norm(self.butterfly - item)

            if distance < 0.055:
                reward += 2.0

                # Eating restores hunger.
                self.hunger += FOOD_HUNGER_RESTORE
                self.hunger = min(1.0, self.hunger)

                self.collected += 1
            else:
                remaining_food.append(item)

        self.food = remaining_food

        # End the episode when hunger reaches zero.
        hunger_dead = self.hunger <= 0.0

        # Successfully collecting all food also ends the episode.
        all_food_collected = len(self.food) == 0

        terminated = hunger_dead or all_food_collected
        truncated = self.steps >= MAX_STEPS

        if hunger_dead:
            reward -= 5.0

        if all_food_collected:
            reward += 5.0

        observation = self.render_observation()
        scalar_inputs = self.get_scalar_inputs()

        if self.render_enabled:
            self.render()

        return (
            observation,
            scalar_inputs,
            reward,
            terminated,
            truncated,
            {
                "food_collected": self.collected,
                "food_remaining": len(self.food),
                "hunger": self.hunger,
                "hunger_dead": hunger_dead,
            },
        )

    def get_scalar_inputs(self):
        """
        Returns:

            hunger:
                Normalized hunger value in [0, 1].

            food_inputs:
                Ten pairs of [angle, radius].

                angle is normalized to [-1, 1].
                radius is normalized to [0, 1].
        """

        hunger = np.array([self.hunger], dtype=np.float32)

        detectable_food = []

        for food_position in self.food:
            offset = food_position - self.butterfly
            distance = float(np.linalg.norm(offset))

            if distance <= FOOD_DETECTION_RADIUS:
                angle = math.atan2(float(offset[1]), float(offset[0]))

                # Normalize angle from [-pi, pi] to [-1, 1].
                normalized_angle = angle / math.pi

                # Normalize radius from [0, detection radius] to [0, 1].
                normalized_radius = distance / FOOD_DETECTION_RADIUS

                detectable_food.append((distance, normalized_angle, normalized_radius))

        # Closest food first.
        detectable_food.sort(key=lambda item: item[0])

        food_inputs = []

        for i in range(FOOD_TRACK_LIMIT):
            if i < len(detectable_food):
                _, angle, radius = detectable_food[i]

                food_inputs.extend([angle, radius])
            else:
                # No food in this slot.
                food_inputs.extend([0.0, 1.0])

        food_inputs = np.asarray(food_inputs, dtype=np.float32)

        return np.concatenate([hunger, food_inputs])

    def render_observation(self):
        """
        Creates a top-down RGB image.

        The butterfly is in the center of the image.
        Food is represented as colored points relative to it.
        """
        image = np.zeros((3, IMAGE_SIZE, IMAGE_SIZE), dtype=np.float32)

        # Dark green background.
        image[0, :, :] = 0.04
        image[1, :, :] = 0.12
        image[2, :, :] = 0.05

        # Draw food.
        for item in self.food:
            relative = item - self.butterfly

            # Local observation radius.
            pixel_x = int(IMAGE_SIZE / 2 + relative[0] * IMAGE_SIZE)
            pixel_y = int(IMAGE_SIZE / 2 + relative[1] * IMAGE_SIZE)

            if 2 <= pixel_x < IMAGE_SIZE - 2 and 2 <= pixel_y < IMAGE_SIZE - 2:
                image[0, pixel_y - 2 : pixel_y + 3, pixel_x - 2 : pixel_x + 3] = 1.0
                image[1, pixel_y - 2 : pixel_y + 3, pixel_x - 2 : pixel_x + 3] = 0.55
                image[2, pixel_y - 2 : pixel_y + 3, pixel_x - 2 : pixel_x + 3] = 0.05

        # Draw the butterfly at the center.
        center = IMAGE_SIZE // 2
        image[0, center - 2 : center + 3, center - 2 : center + 3] = 0.9
        image[1, center - 2 : center + 3, center - 2 : center + 3] = 0.2
        image[2, center - 2 : center + 3, center - 2 : center + 3] = 0.9

        # Convert to HWC for pygame if necessary, but keep CHW for PyTorch.
        return image

    def render(self):
        if self.window is None:
            pygame.init()
            self.window = pygame.display.set_mode((600, 600))
            pygame.display.set_caption("Virtual Butterfly")

        if self.clock is None:
            self.clock = pygame.time.Clock()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.render_enabled = False

        self.window.fill((20, 60, 25))

        # Food.
        for item in self.food:
            x = int(item[0] * 600)
            y = int(item[1] * 600)
            pygame.draw.circle(self.window, (255, 190, 30), (x, y), 7)

        # Butterfly.
        bx = int(self.butterfly[0] * 600)
        by = int(self.butterfly[1] * 600)

        pygame.draw.circle(self.window, (240, 70, 220), (bx - 10, by), 10)
        pygame.draw.circle(self.window, (240, 70, 220), (bx + 10, by), 10)
        pygame.draw.circle(self.window, (30, 20, 30), (bx, by), 5)

        pygame.display.flip()
        self.clock.tick(60)


# ============================================================
# Multiprocessing environment workers
#
# Each worker owns one ButterflyEnv in its own process, so env
# stepping (numpy work, image rendering) actually happens in
# parallel instead of a sequential Python loop across 16 envs.
# ============================================================


def _env_worker(remote, seed):
    env = ButterflyEnv(seed=seed, render=False)

    try:
        while True:
            command, payload = remote.recv()

            if command == "reset":
                observation, scalars = env.reset()

                remote.send((observation, scalars))

            elif command == "step":
                observation, scalars, reward, terminated, truncated, info = env.step(
                    payload
                )

                if terminated or truncated:
                    # Stash the true terminal observation/scalars so the
                    # main process can bootstrap the value function for
                    # time-limit truncations (see compute_gae / train()).
                    info = dict(info)
                    info["terminal_observation"] = observation
                    info["terminal_scalars"] = scalars

                    observation, scalars = env.reset()

                remote.send((observation, scalars, reward, terminated, truncated, info))

            elif command == "close":
                remote.close()
                break

            else:
                raise ValueError(f"Unknown worker command: {command}")

    except (EOFError, KeyboardInterrupt):
        pass


class SubprocVecEnv:
    """
    Runs one ButterflyEnv per worker process for true parallelism.

    Uses the default multiprocessing start method (fork on Linux),
    guarded by the __main__ check at the bottom of this file.
    """

    def __init__(self, seeds):
        self.num_envs = len(seeds)

        self.remotes, worker_remotes = zip(*[mp.Pipe() for _ in seeds])

        self.processes = []

        for worker_remote, seed in zip(worker_remotes, seeds):
            process = mp.Process(
                target=_env_worker,
                args=(worker_remote, seed),
                daemon=True,
            )
            process.start()
            self.processes.append(process)

            # The main process doesn't need its own handle to the
            # worker's end of the pipe.
            worker_remote.close()

    def reset(self):
        for remote in self.remotes:
            remote.send(("reset", None))

        results = [remote.recv() for remote in self.remotes]

        observations, scalars = zip(*results)

        return list(observations), list(scalars)

    def step(self, actions):
        for remote, action in zip(self.remotes, actions):
            remote.send(("step", action))

        return [remote.recv() for remote in self.remotes]

    def close(self):
        for remote in self.remotes:
            remote.send(("close", None))

        for process in self.processes:
            process.join()


# ============================================================
# ResNet visual encoder
# ============================================================


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
        )

        self.activation = nn.ReLU()

    def forward(self, x):
        return self.activation(x + self.block(x))


class SmallResNet(nn.Module):
    def __init__(self, feature_size=256):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 5, stride=2, padding=2),
            nn.ReLU(),
            ResidualBlock(32),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(),
            ResidualBlock(64),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.ReLU(),
            ResidualBlock(128),
            nn.AdaptiveAvgPool2d((1, 1)),
        )

        self.projection = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, feature_size),
            nn.ReLU(),
        )

    def forward(self, x):
        x = self.encoder(x)
        return self.projection(x)


# ============================================================
# ResNet + Transformer policy
# ============================================================


class ButterflyPolicy(nn.Module):
    def __init__(self, feature_size=256, action_size=2, history_length=HISTORY_LENGTH):
        super().__init__()

        self.history_length = history_length

        self.visual_encoder = SmallResNet(feature_size)

        self.scalar_encoder = nn.Sequential(
            nn.Linear(SCALAR_INPUT_SIZE, 128),
            nn.ReLU(),
            nn.Linear(128, feature_size),
            nn.ReLU(),
        )

        # Fuse visual and scalar features via concatenation + a learned
        # projection, instead of adding them. Addition forces both
        # modalities into the same 256-dim subspace with no way to
        # learn how to weight/mix them; concatenation lets the
        # network learn that mixing.
        self.fusion = nn.Sequential(
            nn.Linear(feature_size * 2, feature_size),
            nn.ReLU(),
        )

        self.position_embedding = nn.Parameter(
            torch.zeros(1, history_length, feature_size)
        )

        transformer_layer = nn.TransformerEncoderLayer(
            d_model=feature_size,
            nhead=8,
            dim_feedforward=512,
            dropout=0.1,
            batch_first=True,
            activation="gelu",
        )

        self.transformer = nn.TransformerEncoder(transformer_layer, num_layers=3)

        self.policy_head = nn.Sequential(
            nn.Linear(feature_size, 128),
            nn.Tanh(),
            nn.Linear(128, action_size),
        )

        self.value_head = nn.Sequential(
            nn.Linear(feature_size, 128),
            nn.Tanh(),
            nn.Linear(128, 1),
        )

        self.log_std = nn.Parameter(torch.zeros(action_size))

    def encode_inputs(self, image_sequence, scalar_sequence):
        """
        image_sequence:
            [batch, history, 3, 64, 64]

        scalar_sequence:
            [batch, history, 21]
        """

        batch_size, history, channels, height, width = image_sequence.shape

        images = image_sequence.reshape(batch_size * history, channels, height, width)

        image_features = self.visual_encoder(images)

        image_features = image_features.reshape(batch_size, history, -1)

        scalar_features = self.scalar_encoder(scalar_sequence)

        # Concatenate along the feature dim, then project back down to
        # feature_size with a learned layer (see self.fusion above).
        combined_features = torch.cat([image_features, scalar_features], dim=-1)
        combined_features = self.fusion(combined_features)

        return combined_features

    def forward(self, image_sequence, scalar_sequence):
        features = self.encode_inputs(image_sequence, scalar_sequence)

        sequence_length = features.shape[1]

        features = features + self.position_embedding[:, :sequence_length]

        # NOTE: this causal mask has no effect on correctness as written,
        # since only the *last* timestep's transformer output is ever
        # consumed below (current_state = transformed[:, -1]) -- a token
        # can attend to itself and everything before it either way. It's
        # kept here because it's harmless and would matter if this policy
        # is ever extended to consume outputs from multiple timesteps.
        causal_mask = torch.triu(
            torch.ones(sequence_length, sequence_length, device=features.device),
            diagonal=1,
        ).bool()

        transformed = self.transformer(features, mask=causal_mask)

        current_state = transformed[:, -1]

        action_mean = self.policy_head(current_state)

        value = self.value_head(current_state).squeeze(-1)

        std = self.log_std.exp().expand_as(action_mean)

        return action_mean, std, value

    def sample_action(self, image_sequence, scalar_sequence):
        mean, std, value = self.forward(image_sequence, scalar_sequence)

        distribution = torch.distributions.Normal(mean, std)

        raw_action = distribution.rsample()
        action = torch.tanh(raw_action)

        log_probability = distribution.log_prob(raw_action)

        log_probability -= torch.log(1 - action.pow(2) + 1e-6)

        log_probability = log_probability.sum(dim=-1)

        return action, log_probability, value

    def evaluate_actions(self, image_sequence, scalar_sequence, actions):
        mean, std, value = self.forward(image_sequence, scalar_sequence)

        distribution = torch.distributions.Normal(mean, std)

        clipped_actions = actions.clamp(-0.999, 0.999)

        raw_action = torch.atanh(clipped_actions)

        log_probability = distribution.log_prob(raw_action)

        log_probability -= torch.log(1 - clipped_actions.pow(2) + 1e-6)

        log_probability = log_probability.sum(dim=-1)
        entropy = distribution.entropy().sum(dim=-1)

        return log_probability, entropy, value


# ============================================================
# Observation history
# ============================================================


class HistoryBuffer:
    def __init__(self, length=HISTORY_LENGTH):
        self.length = length
        self.image_buffer = deque(maxlen=length)
        self.scalar_buffer = deque(maxlen=length)

    def reset(self, observation, scalar_input):
        self.image_buffer.clear()
        self.scalar_buffer.clear()

        for _ in range(self.length):
            self.image_buffer.append(observation.copy())

            self.scalar_buffer.append(scalar_input.copy())

    def append(self, observation, scalar_input):
        self.image_buffer.append(observation.copy())

        self.scalar_buffer.append(scalar_input.copy())

    def tensors(self):
        images = torch.tensor(np.stack(self.image_buffer), dtype=torch.float32)

        scalars = torch.tensor(np.stack(self.scalar_buffer), dtype=torch.float32)

        return images, scalars

    def terminal_tensors(self, terminal_observation, terminal_scalars):
        """
        Builds the history tensors as they would look one step past the
        current buffer, ending in the true terminal observation/scalars
        (i.e. before the episode-end auto-reset). Used only to bootstrap
        the value function on time-limit truncations. Does not mutate
        this buffer.
        """

        images = list(self.image_buffer)[1:] + [terminal_observation]
        scalars = list(self.scalar_buffer)[1:] + [terminal_scalars]

        images = torch.tensor(np.stack(images), dtype=torch.float32)
        scalars = torch.tensor(np.stack(scalars), dtype=torch.float32)

        return images, scalars


# ============================================================
# PPO training
# ============================================================


def compute_gae(rewards, values, dones):
    """
    dones marks any episode boundary (terminated OR truncated). For
    truncated episodes, train() has already added a bootstrapped
    gamma * V(s_terminal) term onto the reward at that step (see
    train()), so treating truncation and termination identically here
    is correct: the bootstrap information is already folded into
    `rewards` rather than needing a second "next value" lookup.
    """

    advantages = np.zeros_like(rewards, dtype=np.float32)
    last_advantage = 0.0

    for t in reversed(range(len(rewards))):
        if t == len(rewards) - 1:
            next_value = 0.0
            next_nonterminal = 1.0 - dones[t]
        else:
            next_value = values[t + 1]
            next_nonterminal = 1.0 - dones[t]

        delta = rewards[t] + GAMMA * next_value * next_nonterminal - values[t]

        last_advantage = delta + GAMMA * GAE_LAMBDA * next_nonterminal * last_advantage

        advantages[t] = last_advantage

    returns = advantages + values
    return advantages, returns


def train(total_updates=1000, output_model=None):
    if output_model is None:
        output_model = create_model_filename()
    else:
        output_model = Path(output_model)
        output_model.parent.mkdir(parents=True, exist_ok=True)

    print(f"Training on device: {DEVICE}")
    print(f"Model will be saved to: {output_model}")

    vec_env = SubprocVecEnv(seeds=list(range(NUM_ENVS)))

    observations, scalars = vec_env.reset()

    histories = []

    for observation, scalar_input in zip(observations, scalars):
        history = HistoryBuffer()
        history.reset(observation, scalar_input)

        histories.append(history)

    policy = ButterflyPolicy().to(DEVICE)
    optimizer = optim.Adam(policy.parameters(), lr=LEARNING_RATE)

    try:
        for update in range(total_updates):
            rollout_images = []
            rollout_scalars = []
            rollout_actions = []
            rollout_log_probs = []
            rollout_rewards = []
            rollout_values = []
            rollout_dones = []

            for step in range(ROLLOUT_LENGTH):
                print_progress_bar(
                    step + 1,
                    ROLLOUT_LENGTH,
                    prefix=f"Update {update:05d}/{total_updates} rollout ",
                )

                batch_images = []
                batch_scalars = []

                for history in histories:
                    images, scalars_tensor = history.tensors()

                    batch_images.append(images)
                    batch_scalars.append(scalars_tensor)

                batch_images = torch.stack(batch_images).to(DEVICE)

                batch_scalars = torch.stack(batch_scalars).to(DEVICE)

                with torch.no_grad():
                    actions, log_probs, values = policy.sample_action(
                        batch_images, batch_scalars
                    )

                actions_np = actions.cpu().numpy()

                step_results = vec_env.step(list(actions_np))

                step_rewards = []
                step_dones = []

                for i, (
                    next_observation,
                    next_scalars,
                    reward,
                    terminated,
                    truncated,
                    info,
                ) in enumerate(step_results):
                    done = terminated or truncated

                    if done and truncated and not terminated:
                        # Time-limit cutoff, not a "real" ending: bootstrap
                        # using the value network's estimate of the true
                        # terminal state, instead of letting GAE treat this
                        # like the episode's return is exactly 0 afterward.
                        terminal_images, terminal_scalars = histories[
                            i
                        ].terminal_tensors(
                            info["terminal_observation"], info["terminal_scalars"]
                        )

                        with torch.no_grad():
                            _, _, bootstrap_value = policy.forward(
                                terminal_images.unsqueeze(0).to(DEVICE),
                                terminal_scalars.unsqueeze(0).to(DEVICE),
                            )

                        reward = reward + GAMMA * bootstrap_value.item()

                    step_rewards.append(reward)
                    step_dones.append(float(done))

                    if done:
                        histories[i].reset(next_observation, next_scalars)
                    else:
                        histories[i].append(next_observation, next_scalars)

                rollout_images.append(batch_images.cpu())

                rollout_scalars.append(batch_scalars.cpu())
                rollout_actions.append(actions.cpu())
                rollout_log_probs.append(log_probs.cpu())
                rollout_rewards.append(torch.tensor(step_rewards, dtype=torch.float32))
                rollout_values.append(values.cpu())
                rollout_dones.append(torch.tensor(step_dones, dtype=torch.float32))

            print()  # move past the in-place rollout progress bar

            images = torch.stack(rollout_images)
            scalars_tensor = torch.stack(rollout_scalars)
            actions = torch.stack(rollout_actions)
            old_log_probs = torch.stack(rollout_log_probs)
            rewards = torch.stack(rollout_rewards)
            values = torch.stack(rollout_values)
            dones = torch.stack(rollout_dones)

            images = images.reshape(
                ROLLOUT_LENGTH * NUM_ENVS, HISTORY_LENGTH, 3, IMAGE_SIZE, IMAGE_SIZE
            )

            scalars_tensor = scalars_tensor.reshape(
                ROLLOUT_LENGTH * NUM_ENVS, HISTORY_LENGTH, SCALAR_INPUT_SIZE
            )

            actions = actions.reshape(ROLLOUT_LENGTH * NUM_ENVS, 2)

            old_log_probs = old_log_probs.reshape(-1)
            rewards_np = rewards.numpy()
            values_np = values.numpy()
            dones_np = dones.numpy()

            advantages = []
            returns = []

            for env_index in range(NUM_ENVS):
                env_rewards = rewards_np[:, env_index]
                env_values = values_np[:, env_index]
                env_dones = dones_np[:, env_index]

                env_advantages, env_returns = compute_gae(
                    env_rewards, env_values, env_dones
                )

                advantages.append(env_advantages)
                returns.append(env_returns)

            advantages = np.stack(advantages, axis=1)
            returns = np.stack(returns, axis=1)

            advantages = torch.tensor(advantages.reshape(-1), dtype=torch.float32)

            returns = torch.tensor(returns.reshape(-1), dtype=torch.float32)

            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            dataset_size = ROLLOUT_LENGTH * NUM_ENVS
            indices = np.arange(dataset_size)

            for _ in range(PPO_EPOCHS):
                np.random.shuffle(indices)

                for start in range(0, dataset_size, MINIBATCH_SIZE):
                    batch_indices = indices[start : start + MINIBATCH_SIZE]

                    batch_images = images[batch_indices].to(DEVICE)
                    batch_scalars = scalars_tensor[batch_indices].to(DEVICE)
                    batch_actions = actions[batch_indices].to(DEVICE)
                    batch_old_log_probs = old_log_probs[batch_indices].to(DEVICE)
                    batch_advantages = advantages[batch_indices].to(DEVICE)
                    batch_returns = returns[batch_indices].to(DEVICE)

                    new_log_probs, entropy, new_values = policy.evaluate_actions(
                        batch_images, batch_scalars, batch_actions
                    )

                    ratio = (new_log_probs - batch_old_log_probs).exp()

                    unclipped = ratio * batch_advantages
                    clipped = (
                        ratio.clamp(1.0 - CLIP_EPSILON, 1.0 + CLIP_EPSILON)
                        * batch_advantages
                    )

                    policy_loss = -torch.min(unclipped, clipped).mean()

                    value_loss = (batch_returns - new_values).pow(2).mean()

                    entropy_loss = -entropy.mean()

                    loss = (
                        policy_loss
                        + VALUE_COEF * value_loss
                        + ENTROPY_COEF * entropy_loss
                    )

                    optimizer.zero_grad()
                    loss.backward()

                    torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=0.5)

                    optimizer.step()

            if update % 10 == 0:
                average_reward = rewards.mean().item()
                average_value = values.mean().item()
                average_advantage = advantages.mean().item()

                print(
                    f"Update {update:05d}/{total_updates} | "
                    f"reward={average_reward: .4f} | "
                    f"value={average_value: .4f} | "
                    f"advantage={average_advantage: .4f} | "
                    f"loss={loss.item(): .4f}"
                )

            if update % 100 == 0:
                torch.save(policy.state_dict(), output_model)
                print(f"Checkpoint saved: {output_model}")

        torch.save(policy.state_dict(), output_model)

        print()
        print("Training complete.")
        print(f"Saved model: {output_model}")

    finally:
        vec_env.close()


# ============================================================
# Play the trained butterfly
# ============================================================


def play(weights=None):
    if weights is None:
        weights = newest_model()
    else:
        weights = Path(weights)

        if not weights.exists():
            raise FileNotFoundError(f"Weights file does not exist: {weights}")

    print(f"Loading model: {weights}")

    env = ButterflyEnv(seed=123, render=True)
    policy = ButterflyPolicy().to(DEVICE)

    policy.load_state_dict(torch.load(weights, map_location=DEVICE))

    policy.eval()

    while env.render_enabled:
        observation, scalar_input = env.reset()

        history = HistoryBuffer()
        history.reset(observation, scalar_input)

        done = False

        while not done and env.render_enabled:
            with torch.no_grad():
                image_sequence, scalar_sequence = history.tensors()

                image_sequence = image_sequence.unsqueeze(0).to(DEVICE)
                scalar_sequence = scalar_sequence.unsqueeze(0).to(DEVICE)

                action, _, _ = policy.sample_action(image_sequence, scalar_sequence)

            action = action[0].cpu().numpy()

            next_observation, next_scalars, reward, terminated, truncated, info = (
                env.step(action)
            )

            history.append(next_observation, next_scalars)
            done = terminated or truncated

            if done:
                print(
                    "Episode finished | "
                    f"food collected={info['food_collected']} | "
                    f"food remaining={info['food_remaining']}"
                )


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--mode", choices=["train", "run", "play"], default="train")

    parser.add_argument("--updates", type=int, default=1000)

    parser.add_argument(
        "--weights",
        type=str,
        default=None,
        help=(
            "Weights file to load when using --mode run. "
            "If omitted, the newest model is used."
        ),
    )

    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    seed_everything(args.seed)

    try:
        if args.mode == "train":
            train(total_updates=args.updates)
        elif args.mode in ["run", "play"]:
            play(weights=args.weights)
    except KeyboardInterrupt:
        pass
