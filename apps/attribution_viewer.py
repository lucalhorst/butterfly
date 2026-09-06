"""Attribution viewer export app.

Runs a trained ButterflyPolicy checkpoint through a short episode and
exports per-step attribution data for the interactive viewer:

  - image saliency:      gradient of the objective w.r.t. each of the
                          8 history frames (per-pixel, downsampled)
  - scalar attribution:  gradient magnitude for every float scalar input
                          (hunger, time, food angle/radius per slot, bird
                          angle/dist/detected) at the focused timestep
  - transformer attention: per-layer, per-head attention weights of the
                          temporal transformer over the 8-timestep history
                          (normally discarded by PyTorch's fast attention
                          path -- this monkey-patches them back on)
  - branch balance:      relative gradient norm flowing through the image
                          token vs. the entity (context/food/bird) tokens
                          as they enter the per-frame entity transformer

Everything is computed for two objectives -- the action output and the
value estimate -- so the viewer can toggle between "what drives the
action" and "what drives the value estimate".

Usage:
    butterfly-attrib \\
        --checkpoint models/20260101_120000_abc1234_butterfly.pt \\
        --steps 30 \\
        --output attribution.json

Requires: torch, numpy, pillow (for encoding preview frames as PNG).
"""

import argparse
import base64
import io
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from butterfly.config import load_config
from butterfly.env.environment import ButterflyEnv
from butterfly.model.history import HistoryBuffer
from butterfly.model.policy import ButterflyPolicy

try:
    from PIL import Image
except ImportError:
    Image = None


def patch_attention_capture(policy):
    """
    nn.TransformerEncoderLayer calls self_attn with need_weights=False
    by default, to use the fused fast-path kernel, so attention weights
    are normally thrown away. This wraps each layer's self_attn.forward
    to force need_weights=True and average_attn_weights=False (keep
    per-head weights), and stashes the result on the layer object so we
    can read it back after each forward pass.
    """
    for layer in policy.transformer.layers:
        original_forward = layer.self_attn.forward

        def make_wrapper(orig, layer_ref):
            def wrapper(*args, **kwargs):
                kwargs["need_weights"] = True
                kwargs["average_attn_weights"] = False
                output, weights = orig(*args, **kwargs)
                layer_ref._captured_attention = weights.detach()
                return output, weights

            return wrapper

        layer.self_attn.forward = make_wrapper(original_forward, layer)


def get_captured_attention(policy):
    """Returns [num_layers][num_heads][seq][seq] as nested python lists."""
    all_layers = []
    for layer in policy.transformer.layers:
        weights = layer._captured_attention[0]  # drop batch dim (batch size 1)
        all_layers.append(weights.cpu().numpy().round(4).tolist())
    return all_layers


def downsample_saliency(grad_chw, out_size):
    """
    grad_chw: torch tensor [3, H, W], the gradient of some scalar
    objective w.r.t. one history frame.

    Returns an out_size x out_size grid: per-pixel gradient magnitude
    (L2 across channels), average-pooled down to out_size x out_size,
    then min-max normalized to [0, 1] so it can be drawn as a heatmap
    regardless of the objective's raw gradient scale.
    """
    magnitude = grad_chw.detach().pow(2).sum(dim=0).sqrt()
    magnitude = magnitude.unsqueeze(0).unsqueeze(0)
    pooled = F.adaptive_avg_pool2d(magnitude, (out_size, out_size))[0, 0]
    pooled = pooled - pooled.min()
    max_val = pooled.max()
    if max_val > 1e-8:
        pooled = pooled / max_val
    return pooled.round(decimals=4).cpu().numpy().tolist()


def encode_frame_png(image_chw):
    """Encodes a [3, H, W] float tensor (roughly in [0, 1]) as a base64 PNG data URI."""
    if Image is None:
        return None
    arr = image_chw.detach().cpu().clamp(0, 1).mul(255).byte().permute(1, 2, 0).numpy()
    img = Image.fromarray(arr, mode="RGB")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def scalar_feature_labels(food_track_limit, max_birds):
    """Mirrors the float feature order attributed in run_episode.

    Only the float (gradient-bearing) fields are attributed: the int
    type/state ids and bool masks carry no autograd signal. Kept in the
    same order as ``get_scalar_inputs`` returns the dict fields.
    """
    labels = ["hunger", "time_normalized", "is_daytime"]
    for i in range(food_track_limit):
        labels += [f"food{i}_angle", f"food{i}_radius"]
    for i in range(max_birds):
        labels += [f"bird{i}_angle", f"bird{i}_dist", f"bird{i}_detected"]
    return labels


def build_scalar_grads(grad_scalars, food_track_limit, max_birds):
    """Assemble the per-timestep scalar gradient vector from the grad-bearing
    float fields of the scalar dict (each [1, history, dim]).

    Returns a [history, num_float_features] numpy array in the same order
    as ``scalar_feature_labels``.
    """
    context_grad = grad_scalars["context"].grad[0]  # [history, 3]
    food_angle_grad = grad_scalars["food_angle"].grad[0]  # [history, F]
    food_radius_grad = grad_scalars["food_radius"].grad[0]
    bird_angle_grad = grad_scalars["bird_angle"].grad[0]  # [history, M]
    bird_dist_grad = grad_scalars["bird_dist"].grad[0]
    bird_detected_grad = grad_scalars["bird_detected"].grad[0]

    values = torch.cat(
        [
            context_grad,
            food_angle_grad,
            food_radius_grad,
            bird_angle_grad,
            bird_dist_grad,
            bird_detected_grad,
        ],
        dim=-1,
    )
    return values.detach().cpu().numpy().round(4).tolist()


def run_episode(config, policy, device, steps, seed, deterministic, saliency_res):
    env = ButterflyEnv(config=config, seed=seed, render=False)
    history = HistoryBuffer(config.environment.history_length)

    observation, scalar_input = env.reset()
    history.reset(observation, scalar_input)

    labels = scalar_feature_labels(
        config.environment.food_track_limit, config.environment.max_birds
    )
    frames = []

    for step in range(steps):
        image_seq, scalar_dict = history.tensors()
        image_seq = image_seq.unsqueeze(0).to(device).clone().requires_grad_(True)

        # Only float fields can carry autograd gradients (int type ids and
        # bool masks are used as-is during the forward pass).
        grad_scalars = {}
        for key, tensor in scalar_dict.items():
            tensor = tensor.unsqueeze(0).to(device).clone()
            if tensor.dtype.is_floating_point:
                tensor.requires_grad_(True)
            grad_scalars[key] = tensor

        raw_frames_png = [
            encode_frame_png(image_seq[0, t]) for t in range(image_seq.shape[1])
        ]

        # --- Manual forward pass, mirroring ButterflyPolicy.forward through
        # the hierarchical EntityEncoder, keeping the intermediate per-branch
        # tensors around so we can attribute the objective to the image token
        # vs. the entity tokens (context + food + bird) separately. ---
        batch, hist, ch, h, w = image_seq.shape
        flat_n = batch * hist

        flat_images = image_seq.reshape(flat_n, ch, h, w)
        # NOTE: intentionally bypasses encode_images_deduped (torch.unique
        # has no backward pass). Only 8 frames per export step, so there's
        # no dedup benefit to lose.
        ent = policy.entity_encoder
        image_features = ent.visual_encoder(flat_images)
        image_features = image_features.reshape(batch, hist, -1)
        image_features.retain_grad()

        food_track_limit = config.environment.food_track_limit
        max_birds = config.environment.max_birds

        context = grad_scalars["context"].reshape(flat_n, -1)
        context_features = ent.context_encoder(context)
        context_features.retain_grad()

        food_angle = grad_scalars["food_angle"].reshape(flat_n, food_track_limit, 1)
        food_radius = grad_scalars["food_radius"].reshape(flat_n, food_track_limit, 1)
        food_type_ids = grad_scalars["food_type_id"].reshape(flat_n, food_track_limit)
        food_type_emb = ent.food_type_embedding(food_type_ids)
        food_features = ent.food_token_encoder(
            torch.cat([food_angle, food_radius, food_type_emb], dim=-1)
        )

        bird_angle = grad_scalars["bird_angle"].reshape(flat_n, max_birds, 1)
        bird_dist = grad_scalars["bird_dist"].reshape(flat_n, max_birds, 1)
        bird_state_ids = grad_scalars["bird_state_id"].reshape(flat_n, max_birds)
        bird_state_emb = ent.bird_state_embedding(bird_state_ids)
        bird_detected = grad_scalars["bird_detected"].reshape(flat_n, max_birds, 1)
        bird_features = ent.bird_token_encoder(
            torch.cat([bird_angle, bird_dist, bird_state_emb, bird_detected], dim=-1)
        )

        # Entity branch = all non-image tokens (context + food + bird).
        entity_features = torch.cat(
            [
                context_features.reshape(flat_n, 1, -1),
                food_features,
                bird_features,
            ],
            dim=1,
        )
        entity_features.retain_grad()

        entity_tokens = torch.cat(
            [image_features.reshape(flat_n, 1, -1), entity_features], dim=1
        )

        food_mask = grad_scalars["food_mask"].reshape(flat_n, food_track_limit)
        bird_mask = grad_scalars["bird_mask"].reshape(flat_n, max_birds)
        padding_mask = torch.zeros(
            entity_tokens.shape[:2], dtype=torch.bool, device=device
        )
        # Token order is [image, context, food..., bird...]; image and
        # context are always real (False), food/bird are NOT of their
        # real-token masks.
        food_start = 2
        padding_mask[:, food_start : food_start + food_track_limit] = ~food_mask
        padding_mask[:, food_start + food_track_limit :] = ~bird_mask

        pooled = ent.entity_transformer(
            entity_tokens, src_key_padding_mask=padding_mask
        )
        valid_mask = ~padding_mask
        pooled = (pooled * valid_mask.unsqueeze(-1).float()).sum(dim=1)
        pooled = pooled / valid_mask.sum(dim=1, keepdim=True).float().clamp(min=1.0)
        fused = pooled.reshape(batch, hist, -1)

        seq_len = fused.shape[1]
        fused = fused + policy.position_embedding[:, :seq_len]

        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=device), diagonal=1
        ).bool()

        transformed = policy.transformer(fused, mask=causal_mask)
        current_state = transformed[:, -1]

        action_mean = policy.policy_head(current_state)
        value = policy.value_head(current_state).squeeze(-1)

        # Attention weights come from this single shared forward pass,
        # so they're identical regardless of which objective we backprop.
        attention = get_captured_attention(policy)

        def collect_grads():
            image_grad = [
                downsample_saliency(image_seq.grad[0, t], saliency_res)
                for t in range(hist)
            ]
            scalar_grad = build_scalar_grads(
                grad_scalars, food_track_limit, max_birds
            )
            image_branch_norm = float(image_features.grad.norm().item())
            scalar_branch_norm = float(entity_features.grad.norm().item())
            return {
                "image": image_grad,
                "scalar": scalar_grad,
                "image_branch_norm": image_branch_norm,
                "scalar_branch_norm": scalar_branch_norm,
            }

        # Objective 1: overall action magnitude
        policy.zero_grad(set_to_none=True)
        action_mean.pow(2).sum().backward(retain_graph=True)
        action_attribution = collect_grads()

        # Objective 2: value estimate
        image_seq.grad = None
        for tensor in grad_scalars.values():
            tensor.grad = None
        image_features.grad = None
        entity_features.grad = None
        policy.zero_grad(set_to_none=True)
        value.backward()
        value_attribution = collect_grads()

        with torch.no_grad():
            if deterministic:
                action = torch.tanh(action_mean)
            else:
                std = policy.log_std.exp().expand_as(action_mean)
                sampled = torch.distributions.Normal(action_mean, std).sample()
                action = torch.tanh(sampled)

        frames.append(
            {
                "step": step,
                "action_mean": action_mean.detach().cpu().numpy().round(4).tolist()[0],
                "action_taken": action.cpu().numpy().round(4).tolist()[0],
                "value": round(float(value.item()), 4),
                "raw_frames": raw_frames_png,
                "attribution": {
                    "action": action_attribution,
                    "value": value_attribution,
                },
                "attention": attention,
            }
        )

        action_np = action.detach().cpu().numpy()[0]
        next_obs, next_scalars, reward, terminated, truncated, info = env.step(
            action_np
        )
        if terminated or truncated:
            history.reset(next_obs, next_scalars)
        else:
            history.append(next_obs, next_scalars)

    return {
        "scalar_labels": labels,
        "num_transformer_layers": config.network.temporal_transformer_layers,
        "num_transformer_heads": config.network.temporal_transformer_heads,
        "history_length": config.environment.history_length,
        "saliency_resolution": saliency_res,
        "frames": frames,
    }


def build_parser():
    parser = argparse.ArgumentParser(
        description="Export attribution data from a ButterflyPolicy checkpoint for the interactive viewer"
    )
    parser.add_argument(
        "--checkpoint", required=True, help="Path to a .pt checkpoint"
    )
    parser.add_argument(
        "--config", default=None, help="Path to a TOML config, if you used a non-default one"
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=30,
        help="Number of episode steps to record",
    )
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument(
        "--saliency-res",
        type=int,
        default=16,
        help="Downsampled saliency grid resolution (NxN)",
    )
    parser.add_argument(
        "--stochastic",
        action="store_true",
        help="Sample actions from the policy distribution instead of using the deterministic mean",
    )
    parser.add_argument("--output", default="attribution.json")
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    if Image is None:
        print(
            "Warning: Pillow not installed -- frame previews will be omitted. "
            "`pip install pillow` to include them."
        )

    config = load_config(args.config)

    device = "cpu"  # attribution is a single small forward/backward pass; CPU is fine
    policy = ButterflyPolicy(config).to(device)

    checkpoint = torch.load(args.checkpoint, map_location=device)
    if isinstance(checkpoint, dict) and "policy" in checkpoint:
        policy.load_state_dict(checkpoint["policy"])
    else:
        policy.load_state_dict(checkpoint)
    policy.eval()

    patch_attention_capture(policy)

    print(
        f"Running {args.steps} steps and computing attribution (seed={args.seed})..."
    )
    data = run_episode(
        config,
        policy,
        device,
        steps=args.steps,
        seed=args.seed,
        deterministic=not args.stochastic,
        saliency_res=args.saliency_res,
    )

    output_path = Path(args.output)
    with open(output_path, "w") as f:
        json.dump(data, f)

    size_kb = output_path.stat().st_size / 1024
    print(f"Wrote {output_path} ({size_kb:.1f} KB)")
    print("Load this file in the attribution_viewer.html tool to explore it.")


if __name__ == "__main__":
    main()
