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

_TYPE_ID_TO_NAME = {v: k for k, v in FOOD_TYPE_TO_ID.items()}
_BIRD_ID_TO_NAME = {v: k for k, v in BIRD_STATE_TO_ID.items()}

_FOOD_SHORT = {"day": "day", "night": "night", "interval": "int", "random": "rand"}
_BIRD_STATE_STYLE = {
    "roam": ((120, 90, 30), (255, 230, 170)),
    "chase": ((130, 40, 40), (255, 190, 190)),
    "return": ((40, 70, 130), (190, 215, 255)),
}
_FOOD_TYPE_STYLE = {
    0: ((60, 110, 50), (210, 255, 210)),
    1: ((70, 60, 140), (215, 200, 255)),
    2: ((40, 110, 110), (190, 255, 255)),
    3: ((120, 60, 120), (255, 200, 255)),
    4: ((55, 55, 66), (140, 140, 150)),
}
_PANEL_BG = (24, 24, 42)
_PANEL_LABEL = (132, 132, 164)
_PANEL_VALUE = (230, 230, 244)
_PANEL_MUTED = (95, 95, 125)
_PANEL_GOOD = (120, 230, 140)
_PANEL_BAD = (244, 120, 120)
_PANEL_WARN = (240, 195, 90)
_SECTION_COLORS = {
    "env": (90, 200, 255),
    "scalars": (200, 150, 255),
    "bird": (255, 110, 110),
    "world": (130, 220, 140),
    "ai": (255, 200, 120),
}
_STATS_HEADER_H = 26
_STATS_ROW_H = 14
_FOOD_ROWS_SHOWN = 9


class ButterflyEnv:
    """
    An open-world 2D environment with procedural chunk generation.

    The butterfly searches for flowers in a world with day-night cycles,
    multiple plant types, and a bird predator. Observation: RGB image
    showing all plants in view (inactive ones dimmed), plus a structured
    dict of scalar arrays with nearby eatable plants, time info, and bird
    proximity.
    """

    __slots__ = (
        "config",
        "collected",
        "steps",
        "rng",
        "render_enabled",
        "hunger",
        "butterfly_world_pos",
        "time_step",
        "world",
        "birds",
        "window",
        "clock",
        "font",
        "font_small",
        "show_full_stats",
        "show_stats_panel",
        "is_daytime",
        "display_stats",
        "_last_scalar_inputs",
        "human_control",
    )

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

        # self.world: WorldManager | None = None
        self.birds: list[Bird] = []

        self.window = None
        self.clock = None
        self.font = None
        self.show_full_stats = False
        self.show_stats_panel = False
        self.display_stats = {}
        self._last_scalar_inputs = {}

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

        self.birds = []
        if cfg.predator.enabled:
            for _ in range(cfg.environment.max_birds):
                bird_spawn = self.rng.uniform(
                    -cfg.world.chunk_size, cfg.world.chunk_size, size=2
                ).astype(np.float32)
                self.birds.append(Bird(cfg.predator, bird_spawn, self.rng))

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
        if cfg.predator.enabled:
            for bird in self.birds:
                bird_result = bird.update(self.butterfly_world_pos)
                if bird_result == "kill":
                    bird_killed = True

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
                "bird_states": [b.state for b in self.birds],
                "bird_state": self.birds[0].state if self.birds else None,
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

        bird_info = []
        for bird in self.birds:
            b_angle, b_dist, b_state = bird.get_relative_info(self.butterfly_world_pos)
            bird_info.append((b_dist, b_angle, b_dist, b_state))

        bird_info.sort(key=lambda x: x[0])

        num_birds = min(len(bird_info), max_birds)
        for i in range(num_birds):
            _, b_angle, b_dist, b_state = bird_info[i]
            bird_angle[i] = b_angle
            bird_dist[i] = b_dist
            bird_state_id[i] = BIRD_STATE_TO_ID[b_state]
            bird_detected[i] = 1.0 if b_dist < 1.0 else 0.0
            bird_mask[i] = True

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

        result = {
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
        self._last_scalar_inputs = result
        return result

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

        for bird in self.birds:
            bird_relative = bird.pos - self.butterfly_world_pos
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
        import pygame

        cfg = self.config
        window_size = cfg.rendering.window_size
        stats_width = cfg.rendering.stats_panel_width
        full_width = window_size + stats_width if self.show_stats_panel else window_size
        half = window_size // 2

        if self.window is None or self.window.get_width() != full_width:
            pygame.init()
            self.window = pygame.display.set_mode((full_width, window_size))
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
                elif event.key == pygame.K_F1:
                    self.show_stats_panel = not self.show_stats_panel

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

        for bird in self.birds:
            bird_relative = bird.pos - self.butterfly_world_pos
            bird_x = half + int(bird_relative[0] * half)
            bird_y = half + int(bird_relative[1] * half)

            if 0 <= bird_x < window_size and 0 <= bird_y < window_size:
                radius = 10
                heading = bird.heading

                nose = (
                    bird_x + radius * math.cos(heading),
                    bird_y + radius * math.sin(heading),
                )
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
            if self.birds:
                state_counts = {}
                for b in self.birds:
                    state_counts[b.state] = state_counts.get(b.state, 0) + 1
                bird_summary = ", ".join(f"{c} {s}" for s, c in state_counts.items())
                lines.append(
                    (f"Birds: {len(self.birds)} ({bird_summary})", (200, 100, 100))
                )
            else:
                lines.append(("Birds: none", (200, 100, 100)))
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

        if self.show_stats_panel:
            self._render_stats_panel()

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

    def _render_stats_panel(self):
        import pygame

        cfg = self.config
        panel_w = cfg.rendering.stats_panel_width
        panel_h = cfg.rendering.window_size
        boot_x = cfg.rendering.window_size

        surface = pygame.Surface((panel_w, panel_h))
        surface.fill(_PANEL_BG)

        title = self.font.render("DEBUG STATS", True, _PANEL_VALUE)
        surface.blit(title, (14, 5))
        hint = self.font_small.render("F1: close", True, _PANEL_MUTED)
        surface.blit(hint, (panel_w - hint.get_width() - 12, 7))
        pygame.draw.line(surface, (48, 48, 74), (8, 25), (panel_w - 8, 25))
        y = 30

        y = self._render_env_section(surface, y)
        y = self._render_scalar_section(surface, y)
        y = self._render_bird_section(surface, y)
        y = self._render_world_section(surface, y)
        y = self._render_ai_section(surface, y)

        tab_hint = self.font_small.render("TAB: inline stats", True, _PANEL_MUTED)
        surface.blit(tab_hint, (16, panel_h - 18))

        pygame.draw.line(surface, (70, 70, 100), (0, 0), (0, panel_h), 2)
        self.window.blit(surface, (boot_x, 0))

    def _render_env_section(self, surface, y):
        import pygame

        cfg = self.config
        y = self._draw_section_header(surface, "ENVIRONMENT", _SECTION_COLORS["env"], y)

        pos = self.butterfly_world_pos
        y = self._draw_kv(surface, "pos", f"({pos[0]:.3f}, {pos[1]:.3f})", y)
        y = self._draw_kv(
            surface,
            "step",
            f"{self.steps} / {cfg.environment.max_steps}",
            y,
        )
        hunger_color = (
            int(50 + 205 * self.hunger),
            int(180 * self.hunger),
            50,
        )
        y = self._draw_kv(
            surface,
            "food",
            f"{self.collected} eaten",
            y,
            color=_PANEL_GOOD,
        )
        y = self._draw_progress_row(surface, "hunger", self.hunger, hunger_color, y)

        cycle_pos = self.time_step % cfg.world.day_cycle_length
        cycle_frac = cycle_pos / cfg.world.day_cycle_length
        y = self._draw_progress_row(
            surface, "cycle", cycle_frac, _SECTION_COLORS["env"], y
        )

        style = (
            ((45, 110, 65), (200, 255, 210))
            if self.is_daytime
            else ((70, 60, 150), (210, 200, 255))
        )
        surface.blit(self.font_small.render("phase", True, _PANEL_LABEL), (16, y))
        self._draw_badge(
            surface,
            "DAY" if self.is_daytime else "NIGHT",
            style,
            78,
            y,
        )
        return y + _STATS_ROW_H

    def _render_scalar_section(self, surface, y):
        import pygame

        scalars = self._last_scalar_inputs
        y = self._draw_section_header(
            surface, "SCALAR INPUTS", _SECTION_COLORS["scalars"], y
        )

        if not scalars:
            surface.blit(
                self.font_small.render("unavailable", True, _PANEL_MUTED),
                (16, y),
            )
            return y + _STATS_ROW_H

        context = scalars["context"]
        ctx_text = f"[{context[0]:.3f}, {context[1]:.3f}, {context[2]:.2f}]"
        y = self._draw_kv(surface, "context", ctx_text, y)

        food_angle = scalars["food_angle"]
        food_radius = scalars["food_radius"]
        food_type = scalars["food_type_id"]
        food_mask = scalars["food_mask"]
        active_food = int(np.count_nonzero(food_mask))
        shown = 0

        y = self._draw_table_header(
            surface,
            f"FOOD TRACKS ({active_food} in view)",
            y,
        )
        for i in range(len(food_angle)):
            if not food_mask[i]:
                continue
            if shown >= _FOOD_ROWS_SHOWN:
                break
            type_id = int(food_type[i])
            name = _TYPE_ID_TO_NAME.get(type_id, "pad")
            frac = 1.0 - float(food_radius[i])
            y = self._draw_track_row(
                surface,
                i,
                float(food_angle[i]),
                frac,
                _FOOD_SHORT.get(name, name),
                _FOOD_TYPE_STYLE.get(type_id, _FOOD_TYPE_STYLE[4]),
                y,
            )
            shown += 1

        if active_food > shown:
            more = active_food - shown
            more_surf = self.font_small.render(f"+{more} more", True, _PANEL_MUTED)
            surface.blit(more_surf, (16, y))
            y += _STATS_ROW_H

        bird_angle = scalars["bird_angle"]
        bird_dist = scalars["bird_dist"]
        bird_state = scalars["bird_state_id"]
        bird_detected = scalars["bird_detected"]
        bird_mask = scalars["bird_mask"]

        y = self._draw_table_header(surface, "BIRD TRACKS", y)
        for i in range(len(bird_angle)):
            if not bird_mask[i]:
                continue
            state_id = int(bird_state[i])
            name = _BIRD_ID_TO_NAME.get(state_id, "roam")
            style = _BIRD_STATE_STYLE.get(name, _FOOD_TYPE_STYLE[4])
            frac = 1.0 - float(bird_dist[i])
            row_y = y
            self._draw_track_row(
                surface,
                i,
                float(bird_angle[i]),
                frac,
                name,
                style,
                row_y,
            )
            if bird_detected[i]:
                det = self.font_small.render("DET", True, _PANEL_BAD)
                surface.blit(det, (234, row_y))
            y = row_y + _STATS_ROW_H

        return y

    def _render_bird_section(self, surface, y):
        import pygame

        y = self._draw_section_header(surface, "BIRDS", _SECTION_COLORS["bird"], y)

        if not self.birds:
            surface.blit(
                self.font_small.render("predator disabled", True, _PANEL_MUTED),
                (16, y),
            )
            return y + _STATS_ROW_H

        for idx, bird in enumerate(self.birds):
            if idx > 0:
                pygame.draw.line(
                    surface, (48, 48, 74), (16, y), (surface.get_width() - 16, y)
                )
                y += 2

            surface.blit(
                self.font_small.render(f"[{idx}]", True, _PANEL_MUTED), (16, y)
            )

            pos = bird.pos
            surface.blit(
                self.font_small.render(
                    f"pos ({pos[0]:.2f}, {pos[1]:.2f})", True, _PANEL_VALUE
                ),
                (40, y),
            )
            y += _STATS_ROW_H

            h = float(bird.heading)
            state_name = bird.state if bird.state else "roam"
            dist = float(np.linalg.norm(bird.pos - self.butterfly_world_pos))
            dist_col = _PANEL_GOOD if dist < 0.1 else _PANEL_VALUE

            surface.blit(self.font_small.render("state", True, _PANEL_LABEL), (16, y))
            self._draw_badge(
                surface,
                state_name.upper(),
                _BIRD_STATE_STYLE.get(state_name, _FOOD_TYPE_STYLE[4]),
                78,
                y,
            )
            dist_surf = self.font_small.render(f"dist {dist:.3f}", True, dist_col)
            surface.blit(
                dist_surf, (surface.get_width() - dist_surf.get_width() - 12, y)
            )
            y += _STATS_ROW_H

        return y

    def _render_world_section(self, surface, y):
        import pygame

        y = self._draw_section_header(surface, "WORLD", _SECTION_COLORS["world"], y)

        chunk = self.world.current_chunk
        y = self._draw_kv(surface, "chunk", f"({chunk[0]}, {chunk[1]})", y)
        y = self._draw_kv(
            surface,
            "chunks loaded",
            str(len(self.world.loaded_chunks)),
            y,
        )

        all_plants = self.world.get_all_plants(self.butterfly_world_pos)
        total = len(all_plants)
        active = sum(1 for p in all_plants if p["active"])
        y = self._draw_kv(
            surface,
            "plants",
            f"{total} ({active} active / {total - active} inert)",
            y,
        )

        counts = {}
        for p in all_plants:
            counts[p["type"]] = counts.get(p["type"], 0) + 1
        type_text = " ".join(f"{k}:{v}" for k, v in counts.items())
        if not type_text:
            type_text = "none"
        surface.blit(self.font_small.render("types", True, _PANEL_LABEL), (16, y))
        surface.blit(self.font_small.render(type_text, True, _PANEL_VALUE), (78, y))
        return y + _STATS_ROW_H

    def _render_ai_section(self, surface, y):
        import pygame

        y = self._draw_section_header(surface, "AI OUTPUT", _SECTION_COLORS["ai"], y)

        action = self.display_stats.get("action")
        if action is None:
            action_text = "—"
            action_col = _PANEL_MUTED
        else:
            arr = np.asarray(action, dtype=np.float32)
            action_text = "[" + ", ".join(f"{a:+.3f}" for a in arr) + "]"
            motion = float(np.linalg.norm(arr))
            action_col = _PANEL_GOOD if motion > 0.01 else _PANEL_MUTED
        y = self._draw_kv(surface, "action", action_text, y, color=action_col)

        value = self.display_stats.get("value", 0.0)
        y = self._draw_kv(surface, "value", f"{value:.4f}", y)

        reward = self.display_stats.get("cumulative_reward", 0.0)
        reward_col = _PANEL_GOOD if reward > 0.0 else _PANEL_BAD
        y = self._draw_kv(surface, "reward", f"{reward:.3f}", y, color=reward_col)
        return y

    def _draw_section_header(self, surface, title, color, y):
        import pygame

        surf = self.font.render(title, True, _PANEL_VALUE)
        surface.blit(surf, (14, y + 4))
        pygame.draw.rect(surface, color, (8, y + 3, 4, _STATS_HEADER_H - 9))
        pygame.draw.line(
            surface,
            (48, 48, 74),
            (8, y + _STATS_HEADER_H - 1),
            (surface.get_width() - 8, y + _STATS_HEADER_H - 1),
        )
        return y + _STATS_HEADER_H

    def _draw_kv(
        self, surface, label, value, y, color=_PANEL_VALUE, row_h=_STATS_ROW_H
    ):
        surface.blit(self.font_small.render(label, True, _PANEL_LABEL), (16, y))
        val_surf = self.font_small.render(value, True, color)
        surface.blit(
            val_surf,
            (surface.get_width() - val_surf.get_width() - 12, y),
        )
        return y + row_h

    def _draw_progress_row(self, surface, label, frac, color, y):
        import pygame

        surface.blit(self.font_small.render(label, True, _PANEL_LABEL), (16, y))
        bar_x, bar_w, bar_h = 82, 150, 8
        bar_y = y + 3
        pygame.draw.rect(
            surface,
            (48, 48, 74),
            (bar_x, bar_y, bar_w, bar_h),
            border_radius=4,
        )
        fill = int(bar_w * clamp(frac, 0.0, 1.0))
        if fill > 0:
            pygame.draw.rect(
                surface, color, (bar_x, bar_y, fill, bar_h), border_radius=4
            )
        pct = self.font_small.render(f"{frac:.0%}", True, _PANEL_VALUE)
        surface.blit(pct, (bar_x + bar_w + 8, y))
        return y + _STATS_ROW_H

    def _draw_table_header(self, surface, text, y):
        surface.blit(self.font_small.render(text, True, _PANEL_MUTED), (16, y))
        return y + _STATS_ROW_H

    def _draw_track_row(
        self,
        surface,
        i,
        angle,
        frac,
        badge_name,
        badge_style,
        y,
        row_h=_STATS_ROW_H,
    ):
        import pygame

        surface.blit(self.font_small.render(f"{i:02d}", True, _PANEL_MUTED), (16, y))
        surface.blit(
            self.font_small.render(f"ang {angle:+.2f}", True, _PANEL_VALUE),
            (32, y),
        )
        bar_x, bar_w, bar_h = 100, 60, 7
        pygame.draw.rect(
            surface,
            (48, 48, 74),
            (bar_x, y + 3, bar_w, bar_h),
            border_radius=3,
        )
        fill = int(bar_w * clamp(frac, 0.0, 1.0))
        col = (
            _PANEL_GOOD if frac > 0.7 else (_PANEL_WARN if frac > 0.4 else _PANEL_VALUE)
        )
        if fill > 0:
            pygame.draw.rect(surface, col, (bar_x, y + 3, fill, bar_h), border_radius=3)
        self._draw_badge(surface, badge_name, badge_style, 166, y)
        return y + row_h

    def _draw_badge(self, surface, text, style, x, y):
        import pygame

        bg, fg = style
        text_surf = self.font_small.render(text, True, fg)
        w = text_surf.get_width() + 14
        pygame.draw.rect(surface, bg, (x, y, w, 14), border_radius=7)
        surface.blit(text_surf, (x + 7, y + 1))
        return w
