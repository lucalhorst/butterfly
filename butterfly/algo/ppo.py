"""PPO training."""

import math
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from butterfly.config import Config
from butterfly.env.vec_env import SubprocVecEnv
from butterfly.metrics import (
    METRIC_TAGS,
    CsvMetricLogger,
    create_metrics_csv_path,
    create_tensorboard_logdir,
)
from butterfly.model.history import HistoryBuffer
from butterfly.model.policy import ButterflyPolicy
from butterfly.utils import DEVICE, create_model_filename, print_progress_bar

__all__ = ["compute_gae", "train"]


def compute_gae(rewards, values, dones, gamma, gae_lambda):
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

        delta = rewards[t] + gamma * next_value * next_nonterminal - values[t]

        last_advantage = (
            delta + gamma * gae_lambda * next_nonterminal * last_advantage
        )

        advantages[t] = last_advantage

    returns = advantages + values
    return advantages, returns


def train(
    config: Config,
    total_updates=1000,
    output_model=None,
    resume_from=None,
    device=DEVICE,
    csv_output=None,
    tb_logdir=None,
):
    if output_model is None:
        output_model = create_model_filename()
    else:
        output_model = Path(output_model)
        output_model.parent.mkdir(parents=True, exist_ok=True)

    training = config.training
    env_cfg = config.environment

    print(f"Training on device: {device}")
    print(f"Model will be saved to: {output_model}")

    config_summary = {
        "device": device,
        "learning_rate": training.learning_rate,
        "gamma": training.gamma,
        "gae_lambda": training.gae_lambda,
        "clip_epsilon": training.clip_epsilon,
        "entropy_coef": training.entropy_coef,
        "value_coef": training.value_coef,
        "rollout_length": training.rollout_length,
        "num_envs": training.num_envs,
        "ppo_epochs": training.ppo_epochs,
        "minibatch_size": training.minibatch_size,
        "resume_from": resume_from,
    }

    csv_logger = None
    tb_writer = None

    if training.log_csv:
        csv_path = (
            Path(csv_output) if csv_output is not None else create_metrics_csv_path(output_model)
        )
        csv_logger = CsvMetricLogger(csv_path).open()
        print(f"Metrics CSV: {csv_path}")

    if training.log_tensorboard:
        tb_dir = Path(tb_logdir) if tb_logdir is not None else create_tensorboard_logdir()
        tb_writer = SummaryWriter(log_dir=str(tb_dir))
        tb_writer.add_text(
            "config",
            "\n".join(f"{key}={value}" for key, value in config_summary.items()),
        )
        print(f"TensorBoard run: {tb_dir}")

    vec_env = SubprocVecEnv(seeds=list(range(training.num_envs)), config=config)

    observations, scalars = vec_env.reset()

    histories = []

    for observation, scalar_input in zip(observations, scalars):
        history = HistoryBuffer(env_cfg.history_length)
        history.reset(observation, scalar_input)

        histories.append(history)

    policy = ButterflyPolicy(config).to(device)
    optimizer = optim.Adam(policy.parameters(), lr=training.learning_rate)

    if resume_from is not None:
        resume_path = Path(resume_from)

        if not resume_path.exists():
            raise FileNotFoundError(f"Resume checkpoint does not exist: {resume_path}")

        checkpoint = torch.load(resume_path, map_location=device)

        if isinstance(checkpoint, dict) and "policy" in checkpoint:
            policy.load_state_dict(checkpoint["policy"])
            optimizer.load_state_dict(checkpoint["optimizer"])
            print(f"Resumed policy + optimizer state from: {resume_path}")
        else:
            # Backward compatibility: older checkpoints were a raw
            # policy state_dict with no optimizer state saved, so the
            # optimizer just starts fresh.
            policy.load_state_dict(checkpoint)
            print(
                f"Resumed policy weights only from: {resume_path} "
                "(older checkpoint format has no optimizer state; "
                "optimizer starts fresh)"
            )

        # Note: the update counter always restarts at 0 on resume (this
        # only affects logging/checkpoint-interval cadence), and rollout
        # history buffers, env state, and RNG streams are NOT restored --
        # a resumed run starts collecting fresh trajectories against the
        # loaded weights.

    scalar_input_size = env_cfg.scalar_input_size()

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

            episode_return_accum = [0.0] * training.num_envs
            episode_returns = []

            for step in range(training.rollout_length):
                print_progress_bar(
                    step + 1,
                    training.rollout_length,
                    prefix=f"Update {update:05d}/{total_updates} rollout ",
                    start_time=rollout_start_time,
                )

                batch_images = []
                batch_scalars = []

                for history in histories:
                    images, scalars_tensor = history.tensors()

                    batch_images.append(images)
                    batch_scalars.append(scalars_tensor)

                batch_images = torch.stack(batch_images).to(device)

                batch_scalars = torch.stack(batch_scalars).to(device)

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
                                terminal_images.unsqueeze(0).to(device),
                                terminal_scalars.unsqueeze(0).to(device),
                            )

                        reward = reward + training.gamma * bootstrap_value.item()

                    step_rewards.append(reward)
                    step_dones.append(float(done))

                    episode_return_accum[i] += reward

                    if done:
                        episode_returns.append(episode_return_accum[i])
                        episode_return_accum[i] = 0.0
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
                training.rollout_length * training.num_envs,
                env_cfg.history_length,
                3,
                env_cfg.image_size,
                env_cfg.image_size,
            )

            scalars_tensor = scalars_tensor.reshape(
                training.rollout_length * training.num_envs,
                env_cfg.history_length,
                scalar_input_size,
            )

            actions = actions.reshape(
                training.rollout_length * training.num_envs, 2
            )

            old_log_probs = old_log_probs.reshape(-1)
            rewards_np = rewards.numpy()
            values_np = values.numpy()
            dones_np = dones.numpy()

            advantages = []
            returns = []

            for env_index in range(training.num_envs):
                env_rewards = rewards_np[:, env_index]
                env_values = values_np[:, env_index]
                env_dones = dones_np[:, env_index]

                env_advantages, env_returns = compute_gae(
                    env_rewards,
                    env_values,
                    env_dones,
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

            dataset_size = training.rollout_length * training.num_envs
            indices = np.arange(dataset_size)

            minibatches_per_epoch = math.ceil(dataset_size / training.minibatch_size)
            total_minibatches = training.ppo_epochs * minibatches_per_epoch
            minibatch_counter = 0
            backprop_start_time = time.time()

            policy_losses = []
            value_losses = []
            entropy_losses = []
            total_losses = []
            kl_divs = []
            clip_fractions = []
            ratio_means = []
            grad_norms = []

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
                    batch_scalars = scalars_tensor[batch_indices].to(device)
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

                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        policy.parameters(), max_norm=training.gradient_max_norm
                    )

                    optimizer.step()

                    policy_losses.append(policy_loss.item())
                    value_losses.append(value_loss.item())
                    entropy_losses.append(entropy_loss.item())
                    total_losses.append(loss.item())
                    kl_divs.append(
                        (new_log_probs - batch_old_log_probs).mean().item()
                    )
                    clip_fractions.append(
                        ((ratio - 1.0).abs() > training.clip_epsilon)
                        .float()
                        .mean()
                        .item()
                    )
                    ratio_means.append(ratio.mean().item())
                    grad_norms.append(grad_norm.item())

            print()  # move past the in-place backprop progress bar

            rollout_time = time.time() - rollout_start_time
            backprop_time = time.time() - backprop_start_time

            reward_values = rewards.flatten()
            return_values = returns.flatten()
            value_values = values.flatten()
            advantage_values = advantages.flatten()
            action_magnitudes = actions.norm(dim=-1)

            avg_total_loss = float(np.mean(total_losses))
            avg_policy_loss = float(np.mean(policy_losses))
            avg_value_loss = float(np.mean(value_losses))
            avg_entropy_loss = float(np.mean(entropy_losses))

            metrics = {
                "update": update,
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "rollout_time_s": rollout_time,
                "backprop_time_s": backprop_time,
                "update_time_s": rollout_time + backprop_time,
                "rollout_steps_s": (
                    training.rollout_length * training.num_envs / rollout_time
                ),
                "reward_mean": reward_values.mean().item(),
                "reward_std": reward_values.std().item(),
                "reward_min": reward_values.min().item(),
                "reward_max": reward_values.max().item(),
                "return_mean": return_values.mean().item(),
                "return_std": return_values.std().item(),
                "return_min": return_values.min().item(),
                "return_max": return_values.max().item(),
                "episode_count": int(dones.sum().item()),
                "episode_return_mean": (
                    float(np.mean(episode_returns)) if episode_returns else float("nan")
                ),
                "value_mean": value_values.mean().item(),
                "value_std": value_values.std().item(),
                "value_min": value_values.min().item(),
                "value_max": value_values.max().item(),
                "advantage_mean": advantage_values.mean().item(),
                "advantage_std": advantage_values.std().item(),
                "advantage_min": advantage_values.min().item(),
                "advantage_max": advantage_values.max().item(),
                "loss_total": avg_total_loss,
                "loss_policy": avg_policy_loss,
                "loss_value": avg_value_loss,
                "loss_entropy": avg_entropy_loss,
                "entropy": -avg_entropy_loss,
                "log_prob_mean": old_log_probs.mean().item(),
                "kl_approx": float(np.mean(kl_divs)),
                "clip_fraction": float(np.mean(clip_fractions)),
                "ratio_mean": float(np.mean(ratio_means)),
                "grad_norm": float(np.mean(grad_norms)),
                "action_sat": (actions.abs() > 0.99).float().mean().item(),
                "action_mag_mean": action_magnitudes.mean().item(),
                "action_mag_max": action_magnitudes.max().item(),
                "action_mean_x": actions[:, 0].mean().item(),
                "action_mean_y": actions[:, 1].mean().item(),
            }

            if csv_logger is not None:
                csv_logger.log(metrics)

            if tb_writer is not None:
                for tag in METRIC_TAGS:
                    value = metrics[tag]
                    if isinstance(value, (int, float)):
                        tb_writer.add_scalar(tag, value, global_step=update)

                tb_writer.add_histogram(
                    "rollout/reward", reward_values.numpy(), global_step=update
                )
                tb_writer.add_histogram(
                    "rollout/advantage", advantage_values.numpy(), global_step=update
                )
                tb_writer.add_histogram(
                    "rollout/value", value_values.numpy(), global_step=update
                )
                tb_writer.add_histogram(
                    "rollout/action_magnitude",
                    action_magnitudes.numpy(),
                    global_step=update,
                )

            if update % training.log_interval == 0:
                print(
                    f"Update {update:05d}/{total_updates} | "
                    f"reward={metrics['reward_mean']: .4f} | "
                    f"value={metrics['value_mean']: .4f} | "
                    f"advantage={metrics['advantage_mean']: .4f} | "
                    f"loss={metrics['loss_total']: .4f} "
                    f"(policy={metrics['loss_policy']: .4f}, "
                    f"value={metrics['loss_value']: .4f}, "
                    f"entropy={metrics['loss_entropy']: .4f}) | "
                    f"action_sat={metrics['action_sat']:.1%}",
                )

            if update % training.checkpoint_interval == 0:
                torch.save(
                    {
                        "policy": policy.state_dict(),
                        "optimizer": optimizer.state_dict(),
                    },
                    output_model,
                )
                print(f"Checkpoint saved: {output_model}")

        torch.save(
            {"policy": policy.state_dict(), "optimizer": optimizer.state_dict()},
            output_model,
        )

        print()
        print("Training complete.")
        print(f"Saved model: {output_model}")
    except KeyboardInterrupt:
        torch.save(
            {"policy": policy.state_dict(), "optimizer": optimizer.state_dict()},
            output_model,
        )

        print()
        print("Training incomplete, but exiting on request")
        print(f"Saved model: {output_model}")
        raise
    finally:
        if csv_logger is not None:
            csv_logger.close()
        if tb_writer is not None:
            tb_writer.close()
        vec_env.close()