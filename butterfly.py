import argparse
import math
import multiprocessing as mp
import os
import random
import subprocess
import sys
import time
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

MAX_STEPS = 10000
HISTORY_LENGTH = 8

NUM_ENVS = 16
ROLLOUT_LENGTH = 256
PPO_EPOCHS = 4
MINIBATCH_SIZE = 256

GAMMA = 0.99
GAE_LAMBDA = 0.95
CLIP_EPSILON = 0.2
LEARNING_RATE = 3e-4
ENTROPY_COEF = 0.01
VALUE_COEF = 0.5

MODEL_DIR = Path("models")

FOOD_TRACK_LIMIT = 15
FOOD_DETECTION_RADIUS = 0.45

INITIAL_HUNGER = 1.0
HUNGER_DEPLETION_PER_STEP = 0.0002
FOOD_HUNGER_RESTORE = 0.35

CHUNK_SIZE = 16
CHUNKS_LOADED = 2
PLANTS_PER_CHUNK = 20
DAY_CYCLE_LENGTH = 100
DAY_DURATION = 50
NIGHT_DURATION = 50
PLANT_COOLDOWN = 25
INTERVAL_PHASES_MAX = 3
BIRD_SPEED = 0.025
BIRD_DETECTION_RANGE = 0.3
BIRD_CHASE_SPEED = 0.035
BIRD_PATROL_RANGE = 2.0

PLANT_TYPES = ["day", "night", "interval", "random"]
PLANT_TYPE_PROBS = [0.35, 0.35, 0.2, 0.1]

SCALAR_INPUT_SIZE = 1 + FOOD_TRACK_LIMIT * 3 + 1 + 1 + 5 + 1

# ============================================================
# Utility functions
# ============================================================


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
# Bird entity
# ============================================================


class BirdState:
    ROAM = "roam"
    CHASE = "chase"
    RETURN = "return"


class Bird:
    def __init__(self, spawn_pos, rng):
        self.pos = spawn_pos.copy().astype(np.float32)
        self.spawn_pos = spawn_pos.copy().astype(np.float32)
        self.state = BirdState.ROAM
        self.rng = rng
        self.target_pos = None

    def update(self, butterfly_pos):
        if self.state == BirdState.ROAM:
            return self._do_roam(butterfly_pos)
        elif self.state == BirdState.CHASE:
            return self._do_chase(butterfly_pos)
        elif self.state == BirdState.RETURN:
            return self._do_return()
        return None

    def _do_roam(self, butterfly_pos):
        if self.target_pos is None or self._reached_target():
            angle = self.rng.uniform(0, 2 * math.pi)
            dist = self.rng.uniform(0, BIRD_PATROL_RANGE)
            self.target_pos = self.spawn_pos + np.array(
                [dist * math.cos(angle), dist * math.sin(angle)], dtype=np.float32
            )

        direction = self.target_pos - self.pos
        dist = float(np.linalg.norm(direction))

        if dist > 0.1:
            self.pos += (direction / dist) * BIRD_SPEED

        dist_to_butterfly = float(np.linalg.norm(butterfly_pos - self.pos))
        if dist_to_butterfly < BIRD_DETECTION_RANGE:
            self.state = BirdState.CHASE

        return None

    def _do_chase(self, butterfly_pos):
        direction = butterfly_pos - self.pos
        dist = float(np.linalg.norm(direction))

        if dist > 0.05:
            self.pos += (direction / dist) * BIRD_CHASE_SPEED

        if dist < 0.05:
            self.state = BirdState.RETURN
            return "kill"

        if dist > BIRD_DETECTION_RANGE * 2:
            self.state = BirdState.RETURN

        return None

    def _do_return(self):
        direction = self.spawn_pos - self.pos
        dist = float(np.linalg.norm(direction))

        if dist > 0.5:
            self.pos += (direction / dist) * BIRD_SPEED
        else:
            self.state = BirdState.ROAM
            self.target_pos = None

        return None

    def _reached_target(self):
        if self.target_pos is None:
            return True
        return float(np.linalg.norm(self.target_pos - self.pos)) < 0.5

    def get_relative_info(self, butterfly_pos):
        offset = self.pos - butterfly_pos
        distance = float(np.linalg.norm(offset))
        angle = math.atan2(float(offset[1]), float(offset[0])) / math.pi
        normalized_dist = min(distance / (BIRD_DETECTION_RANGE * 2), 1.0)
        return angle, normalized_dist, self.state


# ============================================================
# Chunk and world system
# ============================================================


class Chunk:
    def __init__(self, chunk_x, chunk_y):
        self.chunk_x = chunk_x
        self.chunk_y = chunk_y
        self.plants = self._generate_plants()

    def _generate_plants(self):
        plants = []
        chunk_seed = hash((self.chunk_x, self.chunk_y)) % (2**31)
        chunk_rng = np.random.default_rng(chunk_seed)

        for _ in range(PLANTS_PER_CHUNK):
            local_pos = chunk_rng.uniform(0, CHUNK_SIZE, size=2).astype(np.float32)

            plant_type = chunk_rng.choice(PLANT_TYPES, p=PLANT_TYPE_PROBS)

            plant = {
                "type": plant_type,
                "local_pos": local_pos,
                "active": True,
                "cooldown_until": 0,
            }

            if plant_type == "interval":
                num_phases = int(chunk_rng.integers(1, INTERVAL_PHASES_MAX + 1))
                plant["phases"] = []
                for _ in range(num_phases):
                    start = int(chunk_rng.integers(0, DAY_CYCLE_LENGTH))
                    duration = int(chunk_rng.integers(5, 20))
                    plant["phases"].append((start, duration))
            elif plant_type == "random":
                plant["appear_prob"] = float(chunk_rng.uniform(0.01, 0.05))
                plant["disappear_prob"] = float(chunk_rng.uniform(0.01, 0.05))
                plant["active"] = bool(chunk_rng.random() < 0.5)

            plants.append(plant)

        return plants


class WorldManager:
    def __init__(self, rng):
        self.rng = rng
        self.loaded_chunks = {}
        self.current_chunk = (0, 0)

    def update(self, butterfly_world_pos):
        chunk_x = int(math.floor(butterfly_world_pos[0] / CHUNK_SIZE))
        chunk_y = int(math.floor(butterfly_world_pos[1] / CHUNK_SIZE))

        self.current_chunk = (chunk_x, chunk_y)

        needed_chunks = set()
        half = CHUNKS_LOADED // 2
        for dx in range(-half, half + 1):
            for dy in range(-half, half + 1):
                needed_chunks.add((chunk_x + dx, chunk_y + dy))

        to_remove = [k for k in self.loaded_chunks if k not in needed_chunks]
        for k in to_remove:
            del self.loaded_chunks[k]

        for chunk_pos in needed_chunks:
            if chunk_pos not in self.loaded_chunks:
                self.loaded_chunks[chunk_pos] = Chunk(chunk_pos[0], chunk_pos[1])

    def get_all_plants(self, butterfly_world_pos):
        all_plants = []

        for chunk_pos, chunk in self.loaded_chunks.items():
            chunk_origin_x = chunk_pos[0] * CHUNK_SIZE
            chunk_origin_y = chunk_pos[1] * CHUNK_SIZE

            for plant in chunk.plants:
                world_pos = np.array(
                    [
                        chunk_origin_x + plant["local_pos"][0],
                        chunk_origin_y + plant["local_pos"][1],
                    ],
                    dtype=np.float32,
                )

                all_plants.append(
                    {
                        "world_pos": world_pos,
                        "type": plant["type"],
                        "active": plant["active"],
                        "plant_ref": plant,
                    }
                )

        return all_plants

    def get_visible_plants(self, butterfly_world_pos, current_step):
        visible = []

        for chunk_pos, chunk in self.loaded_chunks.items():
            chunk_origin_x = chunk_pos[0] * CHUNK_SIZE
            chunk_origin_y = chunk_pos[1] * CHUNK_SIZE

            for plant in chunk.plants:
                is_active = self._check_plant_active(plant, current_step)

                if is_active:
                    world_pos = np.array(
                        [
                            chunk_origin_x + plant["local_pos"][0],
                            chunk_origin_y + plant["local_pos"][1],
                        ],
                        dtype=np.float32,
                    )

                    visible.append(
                        {
                            "world_pos": world_pos,
                            "type": plant["type"],
                            "plant_ref": plant,
                        }
                    )

        return visible

    def _check_plant_active(self, plant, current_step):
        if current_step < plant["cooldown_until"]:
            return False

        step_in_cycle = current_step % DAY_CYCLE_LENGTH
        is_daytime = step_in_cycle < DAY_DURATION

        if plant["type"] == "day":
            return is_daytime
        elif plant["type"] == "night":
            return not is_daytime
        elif plant["type"] == "interval":
            return self._check_interval_active(plant, step_in_cycle)
        elif plant["type"] == "random":
            return self._check_random_active(plant, current_step)

        return False

    def _check_interval_active(self, plant, step_in_cycle):
        for start, duration in plant.get("phases", []):
            end = (start + duration) % DAY_CYCLE_LENGTH
            if start < end:
                if start <= step_in_cycle < end:
                    return True
            else:
                if step_in_cycle >= start or step_in_cycle < end:
                    return True
        return False

    def _check_random_active(self, plant, current_step):
        if plant["active"]:
            if self.rng.random() < plant.get("disappear_prob", 0.02):
                plant["active"] = False
        else:
            if self.rng.random() < plant.get("appear_prob", 0.02):
                plant["active"] = True
        return plant["active"]


# ============================================================
# Virtual butterfly environment
# ============================================================


class ButterflyEnv:
    """
    An open-world 2D environment with procedural chunk generation.

    The butterfly searches for flowers in a world with day-night cycles,
    multiple plant types, and a bird predator. Observation: RGB image
    showing only eatable plants, plus scalar vector with all nearby
    plants, time info, and bird proximity.
    """

    def __init__(self, seed=None, render=False):
        self.rng = np.random.default_rng(seed)
        self.render_enabled = render

        self.hunger = INITIAL_HUNGER
        self.butterfly_world_pos = np.zeros(2, dtype=np.float32)
        self.collected = 0
        self.steps = 0
        self.time_step = 0
        self.is_daytime = True

        self.world = None
        self.bird = None

        self.window = None
        self.clock = None
        self.font = None
        self.show_full_stats = False
        self.display_stats = {}

    def reset(self):
        self.time_step = 0
        self.is_daytime = True
        self.steps = 0
        self.hunger = INITIAL_HUNGER
        self.collected = 0

        self.world = WorldManager(self.rng)

        self.butterfly_world_pos = self.rng.uniform(
            -CHUNK_SIZE / 2, CHUNK_SIZE / 2, size=2
        ).astype(np.float32)

        self.world.update(self.butterfly_world_pos)

        bird_spawn = self.rng.uniform(
            -CHUNK_SIZE, CHUNK_SIZE, size=2
        ).astype(np.float32)
        self.bird = Bird(bird_spawn, self.rng)

        observation = self.render_observation()
        scalar_inputs = self.get_scalar_inputs()

        return observation, scalar_inputs

    def step(self, action):
        self.steps += 1
        self.time_step += 1
        self.is_daytime = (self.time_step % DAY_CYCLE_LENGTH) < DAY_DURATION

        action = np.asarray(action, dtype=np.float32)
        action = np.clip(action, -1.0, 1.0)

        old_position = self.butterfly_world_pos.copy()

        speed = 0.035
        self.butterfly_world_pos += action * speed

        self.world.update(self.butterfly_world_pos)

        visible_plants = self.world.get_visible_plants(
            self.butterfly_world_pos, self.time_step
        )

        self.hunger -= HUNGER_DEPLETION_PER_STEP
        self.hunger = max(0.0, self.hunger)

        reward = -0.002

        movement = np.linalg.norm(self.butterfly_world_pos - old_position)
        reward += float(movement) * 0.02

        for plant_info in visible_plants:
            distance = np.linalg.norm(
                self.butterfly_world_pos - plant_info["world_pos"]
            )
            if distance < 0.055:
                reward += 2.0
                self.hunger += FOOD_HUNGER_RESTORE
                self.hunger = min(1.0, self.hunger)
                self.collected += 1

                plant_ref = plant_info["plant_ref"]
                plant_ref["active"] = False
                plant_ref["cooldown_until"] = self.time_step + PLANT_COOLDOWN

        bird_result = self.bird.update(self.butterfly_world_pos)
        bird_killed = bird_result == "kill"

        hunger_dead = self.hunger <= 0.0
        terminated = hunger_dead or bird_killed
        truncated = self.steps >= MAX_STEPS

        if hunger_dead:
            reward -= 5.0

        if bird_killed:
            reward -= 10.0

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
                "hunger": self.hunger,
                "time_step": self.time_step,
                "is_daytime": self.is_daytime,
                "bird_state": self.bird.state,
                "bird_killed": bird_killed,
            },
        )

    def get_scalar_inputs(self):
        hunger = np.array([self.hunger], dtype=np.float32)

        time_normalized = (self.time_step % DAY_CYCLE_LENGTH) / DAY_CYCLE_LENGTH
        time_input = np.array([time_normalized], dtype=np.float32)
        is_day_input = np.array([1.0 if self.is_daytime else 0.0], dtype=np.float32)

        bird_angle = 0.0
        bird_dist = 1.0
        bird_state_roam = 1.0
        bird_state_chase = 0.0
        bird_state_return = 0.0
        bird_detected = 0.0

        if self.bird is not None:
            b_angle, b_dist, b_state = self.bird.get_relative_info(
                self.butterfly_world_pos
            )
            bird_angle = b_angle
            bird_dist = b_dist

            if b_state == BirdState.ROAM:
                bird_state_roam = 1.0
            elif b_state == BirdState.CHASE:
                bird_state_chase = 1.0
            elif b_state == BirdState.RETURN:
                bird_state_return = 1.0

            bird_detected = 1.0 if b_dist < 1.0 else 0.0

        bird_input = np.array(
            [bird_angle, bird_dist, bird_state_roam, bird_state_chase, bird_state_return],
            dtype=np.float32,
        )
        bird_detected_input = np.array([bird_detected], dtype=np.float32)

        all_plants = self.world.get_all_plants(self.butterfly_world_pos)

        detectable_food = []
        for plant_info in all_plants:
            offset = plant_info["world_pos"] - self.butterfly_world_pos
            distance = float(np.linalg.norm(offset))

            if distance <= FOOD_DETECTION_RADIUS:
                angle = math.atan2(float(offset[1]), float(offset[0]))
                normalized_angle = angle / math.pi
                normalized_radius = distance / FOOD_DETECTION_RADIUS
                is_eatable = 1.0 if plant_info["active"] else 0.0
                detectable_food.append(
                    (distance, normalized_angle, normalized_radius, is_eatable)
                )

        detectable_food.sort(key=lambda item: item[0])

        food_inputs = []
        for i in range(FOOD_TRACK_LIMIT):
            if i < len(detectable_food):
                _, angle, radius, is_eatable = detectable_food[i]
                food_inputs.extend([angle, radius, is_eatable])
            else:
                food_inputs.extend([0.0, 1.0, 0.0])

        food_inputs = np.asarray(food_inputs, dtype=np.float32)

        return np.concatenate(
            [hunger, food_inputs, time_input, is_day_input, bird_input, bird_detected_input]
        )

    def render_observation(self):
        image = np.zeros((3, IMAGE_SIZE, IMAGE_SIZE), dtype=np.float32)

        if self.is_daytime:
            image[0, :, :] = 0.1
            image[1, :, :] = 0.25
            image[2, :, :] = 0.08
        else:
            image[0, :, :] = 0.02
            image[1, :, :] = 0.05
            image[2, :, :] = 0.12

        visible_plants = self.world.get_visible_plants(
            self.butterfly_world_pos, self.time_step
        )

        for plant_info in visible_plants:
            relative = plant_info["world_pos"] - self.butterfly_world_pos
            pixel_x = int(IMAGE_SIZE / 2 + relative[0] * IMAGE_SIZE)
            pixel_y = int(IMAGE_SIZE / 2 + relative[1] * IMAGE_SIZE)

            if 2 <= pixel_x < IMAGE_SIZE - 2 and 2 <= pixel_y < IMAGE_SIZE - 2:
                color = self._get_plant_color(plant_info["type"])
                image[
                    0, pixel_y - 2 : pixel_y + 3, pixel_x - 2 : pixel_x + 3
                ] = color[0]
                image[
                    1, pixel_y - 2 : pixel_y + 3, pixel_x - 2 : pixel_x + 3
                ] = color[1]
                image[
                    2, pixel_y - 2 : pixel_y + 3, pixel_x - 2 : pixel_x + 3
                ] = color[2]

        if self.bird is not None:
            bird_relative = self.bird.pos - self.butterfly_world_pos
            bird_px = int(IMAGE_SIZE / 2 + bird_relative[0] * IMAGE_SIZE)
            bird_py = int(IMAGE_SIZE / 2 + bird_relative[1] * IMAGE_SIZE)

            if 0 <= bird_px < IMAGE_SIZE and 0 <= bird_py < IMAGE_SIZE:
                image[
                    0, bird_py - 1 : bird_py + 2, bird_px - 1 : bird_px + 2
                ] = 0.8
                image[
                    1, bird_py - 1 : bird_py + 2, bird_px - 1 : bird_px + 2
                ] = 0.1
                image[
                    2, bird_py - 1 : bird_py + 2, bird_px - 1 : bird_px + 2
                ] = 0.1

        center = IMAGE_SIZE // 2
        image[0, center - 2 : center + 3, center - 2 : center + 3] = 0.9
        image[1, center - 2 : center + 3, center - 2 : center + 3] = 0.2
        image[2, center - 2 : center + 3, center - 2 : center + 3] = 0.9

        return image

    def _get_plant_color(self, plant_type):
        colors = {
            "day": (1.0, 0.8, 0.1),
            "night": (0.3, 0.1, 0.8),
            "interval": (0.1, 0.8, 0.8),
            "random": (0.8, 0.3, 0.8),
        }
        return colors.get(plant_type, (1.0, 1.0, 1.0))

    def render(self):
        if self.window is None:
            pygame.init()
            self.window = pygame.display.set_mode((600, 600))
            pygame.display.set_caption("Virtual Butterfly")

        if self.clock is None:
            self.clock = pygame.time.Clock()

        if self.font is None:
            self.font = pygame.font.SysFont("consolas", 16)
            self.font_small = pygame.font.SysFont("consolas", 13)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.render_enabled = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_TAB:
                    self.show_full_stats = not self.show_full_stats

        if self.is_daytime:
            bg_color = (20, 80, 30)
        else:
            bg_color = (10, 20, 40)

        self.window.fill(bg_color)

        visible_plants = self.world.get_visible_plants(
            self.butterfly_world_pos, self.time_step
        )

        for plant_info in visible_plants:
            relative = plant_info["world_pos"] - self.butterfly_world_pos
            screen_x = 300 + int(relative[0] * 300)
            screen_y = 300 + int(relative[1] * 300)

            if 0 <= screen_x < 600 and 0 <= screen_y < 600:
                color = self._get_plant_color_pygame(plant_info["type"])
                pygame.draw.circle(self.window, color, (screen_x, screen_y), 7)

        if self.bird is not None:
            bird_relative = self.bird.pos - self.butterfly_world_pos
            bird_x = 300 + int(bird_relative[0] * 300)
            bird_y = 300 + int(bird_relative[1] * 300)

            if 0 <= bird_x < 600 and 0 <= bird_y < 600:
                pygame.draw.circle(self.window, (200, 30, 30), (bird_x, bird_y), 10)

        bx = 300
        by = 300
        pygame.draw.circle(self.window, (240, 70, 220), (bx - 10, by), 10)
        pygame.draw.circle(self.window, (240, 70, 220), (bx + 10, by), 10)
        pygame.draw.circle(self.window, (30, 20, 30), (bx, by), 5)

        y_off = 10

        hunger_color = (
            int(50 + 205 * self.hunger),
            int(180 * self.hunger),
            50,
        )

        time_str = "Day" if self.is_daytime else "Night"
        cycle_pos = self.time_step % DAY_CYCLE_LENGTH
        lines = [
            (f"Hunger: {self.hunger:.2f}", hunger_color),
            (f"Food: {self.collected} collected", (220, 220, 220)),
            (f"Time: {time_str} ({cycle_pos}/{DAY_CYCLE_LENGTH})", (200, 200, 150)),
        ]

        if self.show_full_stats:
            lines.append((f"Step: {self.steps}/{MAX_STEPS}", (180, 180, 180)))
            lines.append(
                (
                    f"Reward: {self.display_stats.get('cumulative_reward', 0.0):.2f}",
                    (180, 180, 180),
                )
            )
            lines.append(
                (f"Value: {self.display_stats.get('value', 0.0):.3f}", (180, 180, 180))
            )
            action = self.display_stats.get("action", None)
            if action is not None:
                lines.append(
                    (f"Action: [{action[0]:+.3f}, {action[1]:+.3f}]", (180, 180, 180))
                )
            bird_state_str = self.bird.state if self.bird else "N/A"
            lines.append((f"Bird: {bird_state_str}", (200, 100, 100)))
            lines.append(("TAB: hide full stats", (100, 100, 100)))
        else:
            lines.append(("TAB: full stats", (100, 100, 100)))

        bar_x, bar_y, bar_w, bar_h = 10, y_off + len(lines) * 20 + 4, 120, 8
        pygame.draw.rect(self.window, (40, 40, 40), (bar_x, bar_y, bar_w, bar_h))
        fill_w = int(bar_w * clamp(self.hunger, 0.0, 1.0))
        pygame.draw.rect(self.window, hunger_color, (bar_x, bar_y, fill_w, bar_h))

        for text, color in lines:
            surf = self.font.render(text, True, color)
            self.window.blit(surf, (10, y_off))
            y_off += 20

        pygame.display.flip()
        self.clock.tick(60)

    def _get_plant_color_pygame(self, plant_type):
        colors = {
            "day": (255, 200, 30),
            "night": (80, 30, 200),
            "interval": (30, 200, 200),
            "random": (200, 80, 200),
        }
        return colors.get(plant_type, (255, 255, 255))


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
    """
    Channel widths halved from the original (32/64/128 -> 16/32/64).
    Conv2d compute scales with in_channels * out_channels, so this cuts
    convolution FLOPs roughly 4x -- the dominant cost in this whole
    model on CPU, independent of env count or rollout/minibatch size.
    """

    def __init__(self, feature_size=256):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, 5, stride=2, padding=2),
            nn.ReLU(),
            ResidualBlock(16),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            ResidualBlock(32),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(),
            ResidualBlock(64),
            nn.AdaptiveAvgPool2d((1, 1)),
        )

        self.projection = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64, feature_size),
            nn.ReLU(),
        )

    def forward(self, x):
        x = self.encoder(x)
        return self.projection(x)


def encode_images_deduped(visual_encoder, images):
    """
    images: [N, C, H, W] -- a flattened batch*history stack of frames
    from a single minibatch forward pass.

    Overlapping history windows from nearby samples in the same
    minibatch can share identical raw frames (e.g. sample t's window
    and sample t+1's window overlap in 7 of 8 frames). This finds
    exact-duplicate frames within the minibatch, runs the (expensive)
    visual_encoder on each unique frame only once, then scatters the
    results back out to every original position.

    This is "minibatch-local" dedup: it stays fully correct under PPO
    (every minibatch still uses the current, un-cached weights -- no
    gradient staleness), unlike caching features across minibatches or
    epochs would. Because minibatches are randomly shuffled from the
    whole rollout, the number of duplicate frames found here varies
    run to run -- savings are real but modest, not a guaranteed 8x.
    """

    flat = images.reshape(images.shape[0], -1)

    unique_flat, inverse_indices = torch.unique(flat, dim=0, return_inverse=True)

    unique_images = unique_flat.reshape(-1, *images.shape[1:])

    unique_features = visual_encoder(unique_images)

    return unique_features[inverse_indices]


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

        image_features = encode_images_deduped(self.visual_encoder, images)

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

            rollout_start_time = time.time()

            for step in range(ROLLOUT_LENGTH):
                print_progress_bar(
                    step + 1,
                    ROLLOUT_LENGTH,
                    prefix=f"Update {update:05d}/{total_updates} rollout ",
                    start_time=rollout_start_time,
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

            minibatches_per_epoch = math.ceil(dataset_size / MINIBATCH_SIZE)
            total_minibatches = PPO_EPOCHS * minibatches_per_epoch
            minibatch_counter = 0
            backprop_start_time = time.time()

            for _ in range(PPO_EPOCHS):
                np.random.shuffle(indices)

                for start in range(0, dataset_size, MINIBATCH_SIZE):
                    minibatch_counter += 1

                    print_progress_bar(
                        minibatch_counter,
                        total_minibatches,
                        prefix=f"Update {update:05d}/{total_updates} backprop ",
                        start_time=backprop_start_time,
                    )

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

            print()  # move past the in-place backprop progress bar

            if update % 5 == 0:
                average_reward = rewards.mean().item()
                average_value = values.mean().item()
                average_advantage = advantages.mean().item()

                print(
                    f"Update {update:05d}/{total_updates} | "
                    f"reward={average_reward: .4f} | "
                    f"value={average_value: .4f} | "
                    f"advantage={average_advantage: .4f} | "
                    f"loss={loss.item(): .4f}",
                )

            if update % 10 == 0:
                torch.save(policy.state_dict(), output_model)
                print(f"Checkpoint saved: {output_model}")

        torch.save(policy.state_dict(), output_model)

        print()
        print("Training complete.")
        print(f"Saved model: {output_model}")
    except KeyboardInterrupt:
        torch.save(policy.state_dict(), output_model)

        print()
        print("Training incomplete, but exiting on request")
        print(f"Saved model: {output_model}")
        raise
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
        cumulative_reward = 0.0

        while not done and env.render_enabled:
            with torch.no_grad():
                image_sequence, scalar_sequence = history.tensors()

                image_sequence = image_sequence.unsqueeze(0).to(DEVICE)
                scalar_sequence = scalar_sequence.unsqueeze(0).to(DEVICE)

                action, _, value = policy.sample_action(image_sequence, scalar_sequence)

            action = action[0].cpu().numpy()

            env.display_stats = {
                "action": action.tolist(),
                "value": value[0].item(),
                "cumulative_reward": cumulative_reward,
            }

            next_observation, next_scalars, reward, terminated, truncated, info = (
                env.step(action)
            )

            cumulative_reward += reward

            history.append(next_observation, next_scalars)
            done = terminated or truncated

            if done:
                print(
                    "Episode finished | "
                    f"food collected={info['food_collected']} | "
                    f"hunger={info['hunger']:.2f} | "
                    f"time_step={info['time_step']} | "
                    f"bird_state={info['bird_state']} | "
                    f"total reward={cumulative_reward:.2f}"
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

    parser.add_argument(
        "--threads",
        type=int,
        default=None,
        help=(
            "Number of CPU threads for torch to use (only relevant when "
            "running without CUDA). Defaults to os.cpu_count(). Try your "
            "physical core count (not hyperthreads) if unsure -- more "
            "threads isn't always faster for this workload."
        ),
    )

    args = parser.parse_args()

    seed_everything(args.seed)

    if DEVICE == "cpu":
        thread_count = args.threads if args.threads is not None else os.cpu_count()
        torch.set_num_threads(thread_count)
        print(f"CPU device: using {thread_count} torch threads.")

    try:
        if args.mode == "train":
            train(total_updates=args.updates)
        elif args.mode in ["run", "play"]:
            play(weights=args.weights)
    except KeyboardInterrupt:
        pass
