"""The virtual butterfly environment."""

import math

import numpy as np

from butterfly.config import (
    BIRD_PAD_ID,
    BIRD_STATE_TO_ID,
    FOOD_PAD_ID,
    FOOD_TYPE_TO_ID,
    Config,
)
from butterfly.env.entities import Bird, WorldManager
from butterfly.utils import clamp

__all__ = ["ButterflyEnv"]

# Multiplicative dim factor applied to a plant's color in the image
# observation when it is currently inactive (not eatable). Purely visual.
DIM_INACTIVE_PLANT = 0.4


class ButterflyEnv:
    """
    An open-world 2D environment with procedural chunk generation.

    The butterfly searches for flowers in a world with day-night cycles,
    multiple plant types, and a bird predator. Observation: RGB image
    showing all plants in view (inactive ones dimmed), plus a structured
    dict of scalar arrays with nearby eatable plants, time info, and bird
    proximity.
    """

    def __init__(self, config: Config, seed=None, render=False):
        self.config = config
        self.rng = np.random.default_rng(seed)
        self.render_enabled = render

        self.hunger = config.environment.initial_hunger
        self.butterfly_world_pos = np.zeros(2, dtype=np.float32)
        self.collected = 0
        self.steps = 0
        self.time_step = 0
        self.is_daytime = True

        self.world: WorldManager | None = None
        self.bird = None

        self.window = None
        self.clock = None
        self.font = None
        self.show_full_stats = False
        self.display_stats = {}

        # Human keyboard control (used by the human_player app).
        self.human_control = False

    def reset(self):
        cfg = self.config
        self.time_step = 0
        self.is_daytime = True
        self.steps = 0
        self.hunger = cfg.environment.initial_hunger
        self.collected = 0

        self.world = WorldManager(cfg.world, self.rng)

        self.butterfly_world_pos = self.rng.uniform(
            -cfg.world.chunk_size / 2, cfg.world.chunk_size / 2, size=2
        ).astype(np.float32)

        self.world.update(self.butterfly_world_pos)

        if cfg.predator.enabled:
            bird_spawn = self.rng.uniform(
                -cfg.world.chunk_size, cfg.world.chunk_size, size=2
            ).astype(np.float32)
            self.bird = Bird(cfg.predator, bird_spawn, self.rng)
        else:
            self.bird = None

        observation = self.render_observation()
        scalar_inputs = self.get_scalar_inputs()

        return observation, scalar_inputs

    def step(self, action):
        cfg = self.config
        self.steps += 1
        self.time_step += 1

        if cfg.world.day_night_cycle_enabled:
            self.is_daytime = (
                self.time_step % cfg.world.day_cycle_length
            ) < cfg.world.day_duration
        else:
            self.is_daytime = True

        action = np.asarray(action, dtype=np.float32)
        action = np.clip(action, -1.0, 1.0)

        old_position = self.butterfly_world_pos.copy()

        self.butterfly_world_pos += action * cfg.environment.butterfly_speed

        self.world.update(self.butterfly_world_pos)

        visible_plants = self.world.get_visible_plants(
            self.butterfly_world_pos, self.time_step
        )

        self.hunger -= cfg.environment.hunger_depletion_per_step
        self.hunger = max(0.0, self.hunger)

        reward = cfg.environment.rewards.base_step

        movement = np.linalg.norm(self.butterfly_world_pos - old_position)
        reward += float(movement) * cfg.environment.rewards.movement_scale

        for plant_info in visible_plants:
            distance = np.linalg.norm(
                self.butterfly_world_pos - plant_info["world_pos"]
            )
            if distance < cfg.environment.rewards.eating_distance:
                reward += cfg.environment.rewards.eating_reward
                self.hunger += cfg.environment.food_hunger_restore
                self.hunger = min(1.0, self.hunger)
                self.collected += 1

                plant_ref = plant_info["plant_ref"]
                plant_ref["active"] = False
                plant_ref["cooldown_until"] = self.time_step + cfg.world.plant_cooldown

        bird_killed = False
        if cfg.predator.enabled and self.bird is not None:
            bird_result = self.bird.update(self.butterfly_world_pos)
            bird_killed = bird_result == "kill"

        hunger_dead = self.hunger <= 0.0
        terminated = hunger_dead or bird_killed
        truncated = self.steps >= cfg.environment.max_steps

        if hunger_dead:
            reward += cfg.environment.rewards.hunger_death_penalty

        if bird_killed:
            reward += cfg.environment.rewards.bird_kill_penalty

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
                "bird_state": self.bird.state if self.bird is not None else None,
                "bird_killed": bird_killed,
            },
        )

    def get_scalar_inputs(self):
        cfg = self.config
        env_cfg = cfg.environment
        food_track_limit = env_cfg.food_track_limit
        max_birds = env_cfg.max_birds

        hunger = np.array([self.hunger], dtype=np.float32)

        time_normalized = (
            self.time_step % cfg.world.day_cycle_length
        ) / cfg.world.day_cycle_length
        time_input = np.array([time_normalized], dtype=np.float32)
        is_day_input = np.array([1.0 if self.is_daytime else 0.0], dtype=np.float32)

        context = np.concatenate([hunger, time_input, is_day_input]).astype(np.float32)

        bird_angle = np.zeros(max_birds, dtype=np.float32)
        bird_dist = np.ones(max_birds, dtype=np.float32)
        bird_state_id = np.full(max_birds, BIRD_PAD_ID, dtype=np.int64)
        bird_detected = np.zeros(max_birds, dtype=np.float32)
        bird_mask = np.zeros(max_birds, dtype=bool)

        if self.bird is not None:
            b_angle, b_dist, b_state = self.bird.get_relative_info(
                self.butterfly_world_pos
            )
            idx = 0
            bird_angle[idx] = b_angle
            bird_dist[idx] = b_dist
            bird_state_id[idx] = BIRD_STATE_TO_ID[b_state]
            bird_detected[idx] = 1.0 if b_dist < 1.0 else 0.0
            bird_mask[idx] = True

        # Food candidate pool: only *currently eatable* (active) plants.
        # This eliminates the old truncation ambiguity -- every listed slot
        # is actionable, since inactive plants are never candidates.
        all_plants = self.world.get_all_plants(self.butterfly_world_pos)

        detectable_food = []
        for plant_info in all_plants:
            if not plant_info["active"]:
                continue

            offset = plant_info["world_pos"] - self.butterfly_world_pos
            distance = float(np.linalg.norm(offset))

            if distance <= env_cfg.food_detection_radius:
                angle = math.atan2(float(offset[1]), float(offset[0]))
                normalized_angle = angle / math.pi
                normalized_radius = distance / env_cfg.food_detection_radius
                detectable_food.append(
                    (
                        distance,
                        normalized_angle,
                        normalized_radius,
                        FOOD_TYPE_TO_ID[plant_info["type"]],
                    )
                )

        detectable_food.sort(key=lambda item: item[0])

        food_angle = np.zeros(food_track_limit, dtype=np.float32)
        food_radius = np.ones(food_track_limit, dtype=np.float32)
        food_type_id = np.full(food_track_limit, FOOD_PAD_ID, dtype=np.int64)
        food_mask = np.zeros(food_track_limit, dtype=bool)

        num_tracked = min(len(detectable_food), food_track_limit)
        for i in range(num_tracked):
            _, angle, radius, type_id = detectable_food[i]
            food_angle[i] = angle
            food_radius[i] = radius
            food_type_id[i] = type_id
            food_mask[i] = True

        return {
            "context": context,
            "food_angle": food_angle,
            "food_radius": food_radius,
            "food_type_id": food_type_id,
            "food_mask": food_mask,
            "bird_angle": bird_angle,
            "bird_dist": bird_dist,
            "bird_state_id": bird_state_id,
            "bird_detected": bird_detected,
            "bird_mask": bird_mask,
        }

    def render_observation(self):
        cfg = self.config
        image_size = cfg.environment.image_size
        image = np.zeros((3, image_size, image_size), dtype=np.float32)

        if self.is_daytime:
            image[0, :, :] = 0.1
            image[1, :, :] = 0.25
            image[2, :, :] = 0.08
        else:
            image[0, :, :] = 0.02
            image[1, :, :] = 0.05
            image[2, :, :] = 0.12

        all_plants = self.world.get_all_plants(self.butterfly_world_pos)

        for plant_info in all_plants:
            relative = plant_info["world_pos"] - self.butterfly_world_pos
            pixel_x = int(image_size / 2 + relative[0] * image_size)
            pixel_y = int(image_size / 2 + relative[1] * image_size)

            if 2 <= pixel_x < image_size - 2 and 2 <= pixel_y < image_size - 2:
                color = self._get_plant_color(plant_info["type"])

                if not plant_info["active"]:
                    color = tuple(c * DIM_INACTIVE_PLANT for c in color)

                image[0, pixel_y - 2 : pixel_y + 3, pixel_x - 2 : pixel_x + 3] = color[
                    0
                ]
                image[1, pixel_y - 2 : pixel_y + 3, pixel_x - 2 : pixel_x + 3] = color[
                    1
                ]
                image[2, pixel_y - 2 : pixel_y + 3, pixel_x - 2 : pixel_x + 3] = color[
                    2
                ]

        if self.bird is not None:
            bird_relative = self.bird.pos - self.butterfly_world_pos
            bird_px = int(image_size / 2 + bird_relative[0] * image_size)
            bird_py = int(image_size / 2 + bird_relative[1] * image_size)

            if 0 <= bird_px < image_size and 0 <= bird_py < image_size:
                image[0, bird_py - 1 : bird_py + 2, bird_px - 1 : bird_px + 2] = 0.8
                image[1, bird_py - 1 : bird_py + 2, bird_px - 1 : bird_px + 2] = 0.1
                image[2, bird_py - 1 : bird_py + 2, bird_px - 1 : bird_px + 2] = 0.1

        center = image_size // 2
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
        cfg = self.config
        window_size = cfg.rendering.window_size
        half = window_size // 2

        if self.window is None:
            import pygame

            pygame.init()
            self.window = pygame.display.set_mode((window_size, window_size))
            pygame.display.set_caption("Virtual Butterfly")

        if self.clock is None:
            import pygame

            self.clock = pygame.time.Clock()

        if self.font is None:
            import pygame

            self.font = pygame.font.SysFont("consolas", 16)
            self.font_small = pygame.font.SysFont("consolas", 13)

        import pygame

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
            screen_x = half + int(relative[0] * half)
            screen_y = half + int(relative[1] * half)

            if 0 <= screen_x < window_size and 0 <= screen_y < window_size:
                color = self._get_plant_color_pygame(plant_info["type"])
                pygame.draw.circle(self.window, color, (screen_x, screen_y), 7)

        if self.bird is not None:
            bird_relative = self.bird.pos - self.butterfly_world_pos
            bird_x = half + int(bird_relative[0] * half)
            bird_y = half + int(bird_relative[1] * half)

            if 0 <= bird_x < window_size and 0 <= bird_y < window_size:
                radius = 10
                heading = self.bird.heading  # radians, world space

                # Nose point: straight ahead along heading.
                nose = (
                    bird_x + radius * math.cos(heading),
                    bird_y + radius * math.sin(heading),
                )
                # Back two corners: offset +/-140 degrees from heading, pulled
                # in to about 70% of radius so the triangle doesn't look too
                # needle-thin.
                back_spread = math.radians(140)
                back_left = (
                    bird_x + 0.7 * radius * math.cos(heading + back_spread),
                    bird_y + 0.7 * radius * math.sin(heading + back_spread),
                )
                back_right = (
                    bird_x + 0.7 * radius * math.cos(heading - back_spread),
                    bird_y + 0.7 * radius * math.sin(heading - back_spread),
                )

                pygame.draw.polygon(
                    self.window, (200, 30, 30), [nose, back_left, back_right]
                )

        bx = half
        by = half
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
        cycle_pos = self.time_step % cfg.world.day_cycle_length
        lines = [
            (f"Hunger: {self.hunger:.2f}", hunger_color),
            (f"Food: {self.collected} collected", (220, 220, 220)),
            (
                f"Time: {time_str} ({cycle_pos}/{cfg.world.day_cycle_length})",
                (200, 200, 150),
            ),
        ]

        if self.show_full_stats:
            lines.append(
                (f"Step: {self.steps}/{cfg.environment.max_steps}", (180, 180, 180))
            )
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
