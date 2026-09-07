"""PPO training / evaluation for the SimplePolicy.

Self-contained (no ``butterfly.algo`` reuse); imports only the butterfly
environment and config.
"""

import math
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim

from butterfly.config import Config
from butterfly.env import ButterflyEnv

from simple_app.history import HistoryBuffer, stack_scalar_dicts
from simple_app.policy import SimplePolicy
from simple_app.utils import print_progress_bar

__all__ = ["compute_gae", "train", "run_episodes"]


def compute_gae(rewards, values, dones, gamma, gae_lambda):
    """dones marks any episode boundary (terminated OR truncated).

    Truncation bootstrap (gamma * V(terminal)) is folded into `rewards`
    before this is called, so the two cases can be treated identically here.
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

        delta = rewards[t] + gamma * next_value * next_nonterminal - values[t]

        last_advantage = (
            delta + gamma * gae_lambda * next_nonterminal * last_advantage
        )

        advantages[t] = last_advantage

    returns = advantages + values
    return advantages, returns


def _default_checkpoint_path():
    directory = Path(__file__).parent / "checkpoints"
    directory.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return directory / f"simple_{stamp}.pt"


def train(
    config: Config,
    total_updates=1000,
    output_model=None,
    resume_from=None,
    device=None,
):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if output_model is None:
        output_model = _default_checkpoint_path()
    else:
        output_model = Path(output_model)
        output_model.parent.mkdir(parents=True, exist_ok=True)

    training = config.training
    env_cfg = config.environment

    print(f"Training on device: {device}")
    print(f"Checkpoints will be saved to: {output_model}")
    print(
        f"num_envs={training.num_envs} rollout_length={training.rollout_length} "
        f"history_length={env_cfg.history_length} ppo_epochs={training.ppo_epochs} "
        f"minibatch_size={training.minibatch_size}"
    )

    envs = [
        ButterflyEnv(config=config, seed=seed)
        for seed in range(training.num_envs)
    ]

    histories = []
    for env in envs:
        observation, scalars = env.reset()
        history = HistoryBuffer(env_cfg.history_length)
        history.reset(observation, scalars)
        histories.append(history)

    policy = SimplePolicy(config).to(device)
    optimizer = optim.Adam(policy.parameters(), lr=training.learning_rate)

    if resume_from is not None:
        resume_path = Path(resume_from)
        if not resume_path.exists():
            raise FileNotFoundError(f"Resume checkpoint does not exist: {resume_path}")

        checkpoint = torch.load(resume_path, map_location=device)
        if isinstance(checkpoint, dict) and "policy" in checkpoint:
            policy.load_state_dict(checkpoint["policy"])
            if "optimizer" in checkpoint:
                optimizer.load_state_dict(checkpoint["optimizer"])
            print(f"Resumed policy + optimizer state from: {resume_path}")
        else:
            policy.load_state_dict(checkpoint)
            print(
                f"Resumed policy weights only from: {resume_path} "
                "(no optimizer state; optimizer starts fresh)"
            )

    try:
        for update in range(total_updates):
            rollout_images = []
            rollout_scalars = []
            rollout_actions = []
            rollout_log_probs = []
            rollout_rewards = []
            rollout_values = []
            rollout_dones = []

            episode_return_accum = [0.0] * training.num_envs
            episode_returns = []

            rollout_start_time = time.time()

            for step in range(training.rollout_length):
                print_progress_bar(
                    step + 1,
                    training.rollout_length,
                    prefix=f"Update {update:05d}/{total_updates} rollout ",
                    start_time=rollout_start_time,
                )

                batch_images = []
                batch_scalar_dicts = []

                for history in histories:
                    images, scalars_dict = history.tensors()
                    batch_images.append(images)
                    batch_scalar_dicts.append(scalars_dict)

                batch_images = torch.stack(batch_images).to(device)
                batch_scalars = {
                    key: tensor.to(device)
                    for key, tensor in stack_scalar_dicts(batch_scalar_dicts).items()
                }

                with torch.no_grad():
                    actions, log_probs, values = policy.sample_action(
                        batch_images, batch_scalars
                    )

                actions_np = actions.cpu().numpy()

                step_rewards = []
                step_dones = []

                for i, env in enumerate(envs):
                    next_observation, next_scalars, reward, terminated, truncated, _info = env.step(
                        actions_np[i]
                    )
                    done = terminated or truncated

                    if done and truncated and not terminated:
                        terminal_images, terminal_scalars = histories[
                            i
                        ].terminal_tensors(next_observation, next_scalars)

                        with torch.no_grad():
                            _, _, bootstrap_value = policy.forward(
                                terminal_images.unsqueeze(0).to(device),
                                {
                                    key: tensor.unsqueeze(0).to(device)
                                    for key, tensor in terminal_scalars.items()
                                },
                            )

                        reward = reward + training.gamma * bootstrap_value.item()

                    step_rewards.append(reward)
                    step_dones.append(float(done))

                    episode_return_accum[i] += reward

                    if done:
                        episode_returns.append(episode_return_accum[i])
                        episode_return_accum[i] = 0.0
                        observation, scalars = env.reset()
                        histories[i].reset(observation, scalars)
                    else:
                        histories[i].append(next_observation, next_scalars)

                rollout_images.append(batch_images.cpu())
                rollout_scalars.append(
                    {key: tensor.cpu() for key, tensor in batch_scalars.items()}
                )
                rollout_actions.append(actions.cpu())
                rollout_log_probs.append(log_probs.cpu())
                rollout_rewards.append(torch.tensor(step_rewards, dtype=torch.float32))
                rollout_values.append(values.cpu())
                rollout_dones.append(torch.tensor(step_dones, dtype=torch.float32))

            print()

            images = torch.stack(rollout_images)
            scalars_tensor = {}
            for key in rollout_scalars[0]:
                scalars_tensor[key] = torch.stack([d[key] for d in rollout_scalars])
            actions = torch.stack(rollout_actions)
            old_log_probs = torch.stack(rollout_log_probs)
            rewards = torch.stack(rollout_rewards)
            values = torch.stack(rollout_values)
            dones = torch.stack(rollout_dones)

            rollout_length = training.rollout_length
            num_envs = training.num_envs

            images = images.reshape(rollout_length * num_envs, env_cfg.history_length, 3, env_cfg.image_size, env_cfg.image_size)
            for key, tensor in scalars_tensor.items():
                scalars_tensor[key] = tensor.reshape(rollout_length * num_envs, env_cfg.history_length, -1)
            actions = actions.reshape(rollout_length * num_envs, 2)
            old_log_probs = old_log_probs.reshape(-1)
            rewards_np = rewards.numpy()
            values_np = values.numpy()
            dones_np = dones.numpy()

            advantages = []
            returns = []
            for env_index in range(num_envs):
                env_advantages, env_returns = compute_gae(
                    rewards_np[:, env_index],
                    values_np[:, env_index],
                    dones_np[:, env_index],
                    training.gamma,
                    training.gae_lambda,
                )
                advantages.append(env_advantages)
                returns.append(env_returns)

            advantages = np.stack(advantages, axis=1)
            returns = np.stack(returns, axis=1)

            advantages = torch.tensor(advantages.reshape(-1), dtype=torch.float32)
            returns = torch.tensor(returns.reshape(-1), dtype=torch.float32)
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            dataset_size = rollout_length * num_envs
            indices = np.arange(dataset_size)
            minibatches_per_epoch = math.ceil(dataset_size / training.minibatch_size)
            total_minibatches = training.ppo_epochs * minibatches_per_epoch

            policy_losses = []
            value_losses = []
            entropy_losses = []
            total_losses = []

            backprop_start_time = time.time()
            minibatch_counter = 0

            for _ in range(training.ppo_epochs):
                np.random.shuffle(indices)

                for start in range(0, dataset_size, training.minibatch_size):
                    minibatch_counter += 1
                    print_progress_bar(
                        minibatch_counter,
                        total_minibatches,
                        prefix=f"Update {update:05d}/{total_updates} backprop ",
                        start_time=backprop_start_time,
                    )

                    batch_indices = indices[start : start + training.minibatch_size]

                    batch_images = images[batch_indices].to(device)
                    batch_scalars = {
                        key: tensor[batch_indices].to(device)
                        for key, tensor in scalars_tensor.items()
                    }
                    batch_actions = actions[batch_indices].to(device)
                    batch_old_log_probs = old_log_probs[batch_indices].to(device)
                    batch_advantages = advantages[batch_indices].to(device)
                    batch_returns = returns[batch_indices].to(device)

                    new_log_probs, entropy, new_values = policy.evaluate_actions(
                        batch_images, batch_scalars, batch_actions
                    )

                    ratio = (new_log_probs - batch_old_log_probs).exp()

                    unclipped = ratio * batch_advantages
                    clipped = (
                        ratio.clamp(
                            1.0 - training.clip_epsilon,
                            1.0 + training.clip_epsilon,
                        )
                        * batch_advantages
                    )
                    policy_loss = -torch.min(unclipped, clipped).mean()
                    value_loss = (batch_returns - new_values).pow(2).mean()
                    entropy_loss = -entropy.mean()

                    loss = (
                        policy_loss
                        + training.value_coef * value_loss
                        + training.entropy_coef * entropy_loss
                    )

                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        policy.parameters(), max_norm=training.gradient_max_norm
                    )
                    optimizer.step()

                    policy_losses.append(policy_loss.item())
                    value_losses.append(value_loss.item())
                    entropy_losses.append(entropy_loss.item())
                    total_losses.append(loss.item())

            print()

            rollout_time = time.time() - rollout_start_time
            backprop_time = time.time() - backprop_start_time

            reward_values = rewards.flatten()
            value_values = values.flatten()
            advantage_values = advantages.cpu().flatten()

            if update % training.log_interval == 0:
                episode_count = int(dones.sum().item())
                episode_return_mean = (
                    float(np.mean(episode_returns)) if episode_returns else float("nan")
                )
                avg_total_loss = float(np.mean(total_losses))

                print(
                    f"Update {update:05d}/{total_updates} | "
                    f"reward={reward_values.mean().item(): .4f} | "
                    f"value={value_values.mean().item(): .4f} | "
                    f"advantage={advantage_values.mean().item(): .4f} | "
                    f"loss={avg_total_loss: .4f} "
                    f"(policy={np.mean(policy_losses): .4f}, "
                    f"value={np.mean(value_losses): .4f}, "
                    f"entropy={np.mean(entropy_losses): .4f}) | "
                    f"action_sat={(actions.abs() > 0.99).float().mean().item():.1%} | "
                    f"episodes={episode_count} | "
                    f"episode_return={episode_return_mean: .3f} | "
                    f"rollout={rollout_time:5.1f}s backprop={backprop_time:5.1f}s"
                )

            if update % training.checkpoint_interval == 0:
                torch.save(
                    {
                        "config": config.model_dump(),
                        "policy": policy.state_dict(),
                        "optimizer": optimizer.state_dict(),
                    },
                    output_model,
                )
                print(f"Checkpoint saved: {output_model}")

        torch.save(
            {
                "config": config.model_dump(),
                "policy": policy.state_dict(),
                "optimizer": optimizer.state_dict(),
            },
            output_model,
        )

        print()
        print("Training complete.")
        print(f"Saved model: {output_model}")
    except KeyboardInterrupt:
        torch.save(
            {
                "config": config.model_dump(),
                "policy": policy.state_dict(),
                "optimizer": optimizer.state_dict(),
            },
            output_model,
        )
        print()
        print("Training interrupted; checkpoint saved.")
        raise


def run_episodes(
    config: Config,
    checkpoint,
    episodes=10,
    seed=0,
    render=False,
    render_speed=20.0,
    device=None,
):
    """Run the trained SimplePolicy for ``episodes`` and report stats."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = Path(checkpoint)
    if not checkpoint.exists():
        raise FileNotFoundError(f"Weights file does not exist: {checkpoint}")

    policy = SimplePolicy(config).to(device)
    loaded = torch.load(checkpoint, map_location=device)
    if isinstance(loaded, dict) and "policy" in loaded:
        policy.load_state_dict(loaded["policy"])
    else:
        policy.load_state_dict(loaded)
    policy.eval()

    env = ButterflyEnv(config=config, seed=seed, render=render)
    history = HistoryBuffer(config.environment.history_length)

    if render:
        import pygame

        pygame.init()
        clock = pygame.time.Clock()

    for episode in range(episodes):
        observation, scalar_input = env.reset()
        history.reset(observation, scalar_input)

        done = False
        cumulative_reward = 0.0
        steps = 0

        while not done:
            with torch.no_grad():
                image_sequence, scalar_dict = history.tensors()
                image_sequence = image_sequence.unsqueeze(0).to(device)
                scalar_dict = {
                    key: tensor.unsqueeze(0).to(device)
                    for key, tensor in scalar_dict.items()
                }
                action, _, _ = policy.sample_action(image_sequence, scalar_dict)

            action = action[0].cpu().numpy()

            next_observation, next_scalars, reward, terminated, truncated, _ = (
                env.step(action)
            )

            cumulative_reward += reward
            steps += 1
            done = terminated or truncated

            history.append(next_observation, next_scalars)

            if render:
                clock.tick(render_speed)

        print(
            f"Episode {episode:03d} | steps={steps:5d} | "
            f"reward={cumulative_reward:7.3f} | hunger_dead={terminated and not truncated}"
        )