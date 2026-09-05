"""Multi-process vectorized environment."""

import multiprocessing as mp

from butterfly.config import Config
from butterfly.env.environment import ButterflyEnv

__all__ = ["SubprocVecEnv"]


def _env_worker(remote, seed, config_dict):
    config = Config.model_validate(config_dict)

    env = ButterflyEnv(config=config, seed=seed, render=False)

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

    def __init__(self, seeds, config: Config):
        self.num_envs = len(seeds)

        self.remotes, worker_remotes = zip(*[mp.Pipe() for _ in seeds])

        self.processes = []

        config_dict = config.model_dump()

        for worker_remote, seed in zip(worker_remotes, seeds):
            process = mp.Process(
                target=_env_worker,
                args=(worker_remote, seed, config_dict),
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