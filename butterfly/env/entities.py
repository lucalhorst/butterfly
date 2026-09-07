"""Bird predator and procedural world entities."""

import math

import numpy as np

from butterfly.config import BirdConfig, Config, PlantConfig, WorldConfig

__all__ = ["BirdState", "Bird", "Chunk", "WorldManager"]


# "Up" heading constant: atan2 convention used throughout this file is
# (dy, dx), and "up" on screen/world is -y, so this is -pi/2.
_HEADING_UP = -math.pi / 2


class BirdState:
    ROAM = "roam"
    CHASE = "chase"
    RETURN = "return"


class Bird:
    def __init__(self, bird: BirdConfig, spawn_pos, rng):
        self.bird = bird
        self.pos = spawn_pos.copy().astype(np.float32)
        self.spawn_pos = spawn_pos.copy().astype(np.float32)
        self.state = BirdState.ROAM
        self.rng = rng
        self.target_pos = None
        # NEW: current facing direction in radians (world space, atan2
        # convention matching the rest of the file). Defaults to "up"
        # until the bird has a real target to face.
        self.heading = _HEADING_UP

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
            dist = self.rng.uniform(0, self.bird.patrol_range)
            self.target_pos = self.spawn_pos + np.array(
                [dist * math.cos(angle), dist * math.sin(angle)], dtype=np.float32
            )

        direction = self.target_pos - self.pos
        dist = float(np.linalg.norm(direction))

        # NEW: always face the roam target, whether or not we're close
        # enough this frame to actually move. Falls back to "up" only
        # in the degenerate case of dist == 0 (already at target).
        self.heading = (
            math.atan2(float(direction[1]), float(direction[0]))
            if dist > 1e-6
            else _HEADING_UP
        )

        if dist > 0.1:
            self.pos += (direction / dist) * self.bird.speed

        dist_to_butterfly = float(np.linalg.norm(butterfly_pos - self.pos))
        if dist_to_butterfly < self.bird.detection_radius:
            self.state = BirdState.CHASE

        return None

    def _do_chase(self, butterfly_pos):
        direction = butterfly_pos - self.pos
        dist = float(np.linalg.norm(direction))

        # NEW: always face the butterfly while chasing.
        self.heading = (
            math.atan2(float(direction[1]), float(direction[0]))
            if dist > 1e-6
            else _HEADING_UP
        )

        if dist > 0.05:
            self.pos += (direction / dist) * self.bird.chase_speed

        if dist < 0.05:
            self.state = BirdState.RETURN
            return "kill"

        if dist > self.bird.detection_radius * 2:
            self.state = BirdState.RETURN

        return None

    def _do_return(self):
        direction = self.spawn_pos - self.pos
        dist = float(np.linalg.norm(direction))

        # NEW: always face the spawn point while returning.
        self.heading = (
            math.atan2(float(direction[1]), float(direction[0]))
            if dist > 1e-6
            else _HEADING_UP
        )

        if dist > 0.5:
            self.pos += (direction / dist) * self.bird.speed
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
        normalized_dist = min(distance / (self.bird.detection_radius * 2), 1.0)
        return angle, normalized_dist, self.state


# Chunk and WorldManager are UNCHANGED from the original entities.py.
# Omitted here for brevity -- copy them as-is from the original file.


class Chunk:
    def __init__(self, world: WorldConfig, plant: PlantConfig, chunk_x, chunk_y):
        self.world = world
        self.plant = plant
        self.chunk_x = chunk_x
        self.chunk_y = chunk_y
        self.plants = self._generate_plants()

    def _generate_plants(self):
        plants = []
        chunk_seed = hash((self.chunk_x, self.chunk_y)) % (2**31)
        chunk_rng = np.random.default_rng(chunk_seed)

        plant = self.plant
        world = self.world
        enabled_dict = plant.types_enabled
        enabled_types = [t for t in plant.types if enabled_dict.get(t, True)]
        enabled_probs = [
            p
            for t, p in zip(plant.types, plant.type_probs)
            if enabled_dict.get(t, True)
        ]

        if not enabled_types:
            enabled_types = ["day"]
            enabled_probs = [1.0]

        total = sum(enabled_probs)
        enabled_probs = [p / total for p in enabled_probs]

        for _ in range(plant.plants_per_chunk):
            local_pos = chunk_rng.uniform(0, world.chunk_size, size=2).astype(
                np.float32
            )

            plant_type = chunk_rng.choice(enabled_types, p=enabled_probs)

            plant_info = {
                "type": plant_type,
                "local_pos": local_pos,
                "active": True,
                "cooldown_until": 0,
            }

            if plant_type == "interval":
                num_phases = int(
                    chunk_rng.integers(1, plant.interval_phases_max + 1)
                )
                plant_info["phases"] = []
                for _ in range(num_phases):
                    start = int(chunk_rng.integers(0, world.day_cycle_length))
                    duration = int(chunk_rng.integers(5, 20))
                    plant_info["phases"].append((start, duration))
            elif plant_type == "random":
                plant_info["appear_prob"] = float(chunk_rng.uniform(0.01, 0.05))
                plant_info["disappear_prob"] = float(chunk_rng.uniform(0.01, 0.05))
                plant_info["active"] = bool(chunk_rng.random() < 0.5)

            plants.append(plant_info)

        return plants


class WorldManager:
    def __init__(self, world: WorldConfig, plant: PlantConfig, rng):
        self.world = world
        self.plant = plant
        self.rng = rng
        self.loaded_chunks = {}
        self.current_chunk = (0, 0)

    def update(self, butterfly_world_pos):
        chunk_x = int(math.floor(butterfly_world_pos[0] / self.world.chunk_size))
        chunk_y = int(math.floor(butterfly_world_pos[1] / self.world.chunk_size))

        self.current_chunk = (chunk_x, chunk_y)

        needed_chunks = set()
        half = self.world.chunks_loaded // 2
        for dx in range(-half, half + 1):
            for dy in range(-half, half + 1):
                needed_chunks.add((chunk_x + dx, chunk_y + dy))

        to_remove = [k for k in self.loaded_chunks if k not in needed_chunks]
        for k in to_remove:
            del self.loaded_chunks[k]

        for chunk_pos in needed_chunks:
            if chunk_pos not in self.loaded_chunks:
                self.loaded_chunks[chunk_pos] = Chunk(
                    self.world, self.plant, chunk_pos[0], chunk_pos[1]
                )

    def get_all_plants(self, butterfly_world_pos):
        all_plants = []

        for chunk_pos, chunk in self.loaded_chunks.items():
            chunk_origin_x = chunk_pos[0] * self.world.chunk_size
            chunk_origin_y = chunk_pos[1] * self.world.chunk_size

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
            chunk_origin_x = chunk_pos[0] * self.world.chunk_size
            chunk_origin_y = chunk_pos[1] * self.world.chunk_size

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

        if self.world.day_night_cycle_enabled:
            step_in_cycle = current_step % self.world.day_cycle_length
            is_daytime = step_in_cycle < self.world.day_duration
        else:
            is_daytime = True

        if plant["type"] == "day":
            return is_daytime
        elif plant["type"] == "night":
            return not is_daytime
        elif plant["type"] == "interval":
            if not self.world.day_night_cycle_enabled:
                return True
            step_in_cycle = current_step % self.world.day_cycle_length
            return self._check_interval_active(plant, step_in_cycle)
        elif plant["type"] == "random":
            return self._check_random_active(plant, current_step)

        return False

    def _check_interval_active(self, plant, step_in_cycle):
        for start, duration in plant.get("phases", []):
            end = (start + duration) % self.world.day_cycle_length
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
