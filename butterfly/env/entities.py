"""Bird predator and procedural world entities."""

import math

import numpy as np

from butterfly.config import Config, WorldConfig, PredatorConfig

__all__ = ["BirdState", "Bird", "Chunk", "WorldManager"]


class BirdState:
    ROAM = "roam"
    CHASE = "chase"
    RETURN = "return"


class Bird:
    def __init__(self, predator: PredatorConfig, spawn_pos, rng):
        self.predator = predator
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
            dist = self.rng.uniform(0, self.predator.patrol_range)
            self.target_pos = self.spawn_pos + np.array(
                [dist * math.cos(angle), dist * math.sin(angle)], dtype=np.float32
            )

        direction = self.target_pos - self.pos
        dist = float(np.linalg.norm(direction))

        if dist > 0.1:
            self.pos += (direction / dist) * self.predator.speed

        dist_to_butterfly = float(np.linalg.norm(butterfly_pos - self.pos))
        if dist_to_butterfly < self.predator.detection_range:
            self.state = BirdState.CHASE

        return None

    def _do_chase(self, butterfly_pos):
        direction = butterfly_pos - self.pos
        dist = float(np.linalg.norm(direction))

        if dist > 0.05:
            self.pos += (direction / dist) * self.predator.chase_speed

        if dist < 0.05:
            self.state = BirdState.RETURN
            return "kill"

        if dist > self.predator.detection_range * 2:
            self.state = BirdState.RETURN

        return None

    def _do_return(self):
        direction = self.spawn_pos - self.pos
        dist = float(np.linalg.norm(direction))

        if dist > 0.5:
            self.pos += (direction / dist) * self.predator.speed
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
        normalized_dist = min(distance / (self.predator.detection_range * 2), 1.0)
        return angle, normalized_dist, self.state


class Chunk:
    def __init__(self, world: WorldConfig, chunk_x, chunk_y):
        self.world = world
        self.chunk_x = chunk_x
        self.chunk_y = chunk_y
        self.plants = self._generate_plants()

    def _generate_plants(self):
        plants = []
        chunk_seed = hash((self.chunk_x, self.chunk_y)) % (2**31)
        chunk_rng = np.random.default_rng(chunk_seed)

        world = self.world
        enabled_dict = world.plant_types_enabled
        enabled_types = [t for t in world.plant_types if enabled_dict.get(t, True)]
        enabled_probs = [
            p
            for t, p in zip(world.plant_types, world.plant_type_probs)
            if enabled_dict.get(t, True)
        ]

        if not enabled_types:
            enabled_types = ["day"]
            enabled_probs = [1.0]

        total = sum(enabled_probs)
        enabled_probs = [p / total for p in enabled_probs]

        for _ in range(world.plants_per_chunk):
            local_pos = chunk_rng.uniform(0, world.chunk_size, size=2).astype(
                np.float32
            )

            plant_type = chunk_rng.choice(enabled_types, p=enabled_probs)

            plant = {
                "type": plant_type,
                "local_pos": local_pos,
                "active": True,
                "cooldown_until": 0,
            }

            if plant_type == "interval":
                num_phases = int(
                    chunk_rng.integers(1, world.interval_phases_max + 1)
                )
                plant["phases"] = []
                for _ in range(num_phases):
                    start = int(chunk_rng.integers(0, world.day_cycle_length))
                    duration = int(chunk_rng.integers(5, 20))
                    plant["phases"].append((start, duration))
            elif plant_type == "random":
                plant["appear_prob"] = float(chunk_rng.uniform(0.01, 0.05))
                plant["disappear_prob"] = float(chunk_rng.uniform(0.01, 0.05))
                plant["active"] = bool(chunk_rng.random() < 0.5)

            plants.append(plant)

        return plants


class WorldManager:
    def __init__(self, world: WorldConfig, rng):
        self.world = world
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
                    self.world, chunk_pos[0], chunk_pos[1]
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
