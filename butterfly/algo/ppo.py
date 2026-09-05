"""PPO training."""

import math
import time

import numpy as np
import torch
import torch.optim as optim

from butterfly.config import Config
from butterfly.env.vec_env import SubprocVecEnv
from butterfly.model.history import HistoryBuffer
from butterfly.model.policy import ButterflyPolicy
from butterfly.utils import create_model_filename, print_progress_bar
from pathlib import Path

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

                    torch.nn.utils.clip_grad_norm_(
                        policy.parameters(), max_norm=training.gradient_max_norm
                    )

                    optimizer.step()

                    policy_losses.append(policy_loss.item())
                    value_losses.append(value_loss.item())
                    entropy_losses.append(entropy_loss.item())
                    total_losses.append(loss.item())

            print()  # move past the in-place backprop progress bar

            if update % training.log_interval == 0:
                average_reward = rewards.mean().item()
                average_value = values.mean().item()
                average_advantage = advantages.mean().item()

                avg_total_loss = float(np.mean(total_losses))
                avg_policy_loss = float(np.mean(policy_losses))
                avg_value_loss = float(np.mean(value_losses))
                avg_entropy_loss = float(np.mean(entropy_losses))

                action_saturation = (actions.abs() > 0.99).float().mean().item()

                print(
                    f"Update {update:05d}/{total_updates} | "
                    f"reward={average_reward: .4f} | "
                    f"value={average_value: .4f} | "
                    f"advantage={average_advantage: .4f} | "
                    f"loss={avg_total_loss: .4f} "
                    f"(policy={avg_policy_loss: .4f}, value={avg_value_loss: .4f}, "
                    f"entropy={avg_entropy_loss: .4f}) | "
                    f"action_sat={action_saturation:.1%}",
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
        vec_env.close()