"""
export_attribution.py

Runs your trained ButterflyPolicy checkpoint through a short episode and
exports per-step attribution data for the interactive viewer:

  - image saliency:      gradient of the objective w.r.t. each of the
                          8 history frames (per-pixel, downsampled)
  - scalar attribution:  gradient magnitude for every scalar input
                          (hunger, food entries, time, bird info, ...)
  - transformer attention: per-layer, per-head attention weights over
                          the 8-timestep history (normally discarded
                          by PyTorch's fast attention path -- this
                          monkey-patches them back on)
  - fusion balance:      relative gradient norm flowing through the
                          vision branch vs. the scalar branch at the
                          point they're fused

Everything is computed for two objectives -- the action output and the
value estimate -- so the viewer can toggle between "what drives the
action" and "what drives the value estimate".

This imports your training script directly (rather than redefining the
model classes here), so it always matches your actual architecture and
config -- if you change butterfly.py, this script picks it up
automatically. It does NOT execute your script's __main__ block.

Usage:
    python export_attribution.py \\
        --script butterfly.py \\
        --checkpoint models/20260101_120000_abc1234_butterfly.pt \\
        --steps 30 \\
        --output attribution.json

Requires: torch, numpy, pillow (for encoding preview frames as PNG).
    pip install pillow   # if not already installed
"""

import argparse
import base64
import importlib.util
import io
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

try:
    from PIL import Image
except ImportError:
    Image = None


def load_user_module(script_path):
    """
    Loads the user's training script as an importable module without
    running its `if __name__ == "__main__":` block, so we get access to
    ButterflyPolicy, ButterflyEnv, HistoryBuffer, Config, load_config,
    and encode_images_deduped exactly as defined there.
    """
    script_path = Path(script_path)
    if not script_path.exists():
        raise FileNotFoundError(f"Script not found: {script_path}")

    spec = importlib.util.spec_from_file_location("user_butterfly_module", script_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


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


def scalar_feature_labels(food_track_limit):
    """Mirrors the exact concatenation order in ButterflyEnv.get_scalar_inputs."""
    labels = ["hunger"]
    for i in range(food_track_limit):
        labels += [f"food{i}_angle", f"food{i}_radius", f"food{i}_eatable"]
    labels += ["time_normalized", "is_daytime"]
    labels += [
        "bird_angle",
        "bird_dist",
        "bird_state_roam",
        "bird_state_chase",
        "bird_state_return",
    ]
    labels += ["bird_detected"]
    return labels


def run_episode(module, policy, device, steps, seed, deterministic, saliency_res):
    env = module.ButterflyEnv(seed=seed, render=False)
    history = module.HistoryBuffer()

    observation, scalar_input = env.reset()
    history.reset(observation, scalar_input)

    labels = scalar_feature_labels(module.cfg.environment.food_track_limit)
    frames = []

    for step in range(steps):
        image_seq, scalar_seq = history.tensors()
        image_seq = image_seq.unsqueeze(0).to(device).clone().requires_grad_(True)
        scalar_seq = scalar_seq.unsqueeze(0).to(device).clone().requires_grad_(True)

        raw_frames_png = [encode_frame_png(image_seq[0, t]) for t in range(image_seq.shape[1])]

        # --- Manual forward pass, mirroring ButterflyPolicy.forward, but
        # keeping the intermediate per-branch tensors around so we can
        # attribute the objective to the vision branch vs. the scalar
        # branch separately. ---
        batch, hist, ch, h, w = image_seq.shape
        flat_images = image_seq.reshape(batch * hist, ch, h, w)
        # NOTE: intentionally bypasses module.encode_images_deduped here.
        # That helper uses torch.unique() to skip re-encoding duplicate
        # frames during training minibatches, but torch.unique has no
        # backward pass -- it breaks as soon as the input requires grad,
        # which it does here (we need gradients w.r.t. the pixels for
        # saliency). With only `hist` (8) frames per export step there's
        # no meaningful dedup benefit to lose by calling the encoder
        # directly; the output values are identical either way.
        image_features = policy.visual_encoder(flat_images)
        image_features = image_features.reshape(batch, hist, -1)
        image_features.retain_grad()

        scalar_features = policy.scalar_encoder(scalar_seq)
        scalar_features.retain_grad()

        combined = torch.cat([image_features, scalar_features], dim=-1)
        fused = policy.fusion(combined)

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
            scalar_grad = scalar_seq.grad[0].detach().cpu().numpy().round(4).tolist()
            image_branch_norm = float(image_features.grad.norm().item())
            scalar_branch_norm = float(scalar_features.grad.norm().item())
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
        scalar_seq.grad = None
        image_features.grad = None
        scalar_features.grad = None
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
        next_obs, next_scalars, reward, terminated, truncated, info = env.step(action_np)
        if terminated or truncated:
            history.reset(next_obs, next_scalars)
        else:
            history.append(next_obs, next_scalars)

    return {
        "scalar_labels": labels,
        "num_transformer_layers": module.cfg.network.transformer_layers,
        "num_transformer_heads": module.cfg.network.transformer_heads,
        "history_length": module.cfg.environment.history_length,
        "saliency_resolution": saliency_res,
        "frames": frames,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Export attribution data from a ButterflyPolicy checkpoint for the interactive viewer"
    )
    parser.add_argument("--script", required=True, help="Path to your training script, e.g. butterfly.py")
    parser.add_argument("--checkpoint", required=True, help="Path to a .pt checkpoint")
    parser.add_argument("--config", default=None, help="Path to a TOML config, if you used a non-default one")
    parser.add_argument("--steps", type=int, default=30, help="Number of episode steps to record")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--saliency-res", type=int, default=16, help="Downsampled saliency grid resolution (NxN)")
    parser.add_argument(
        "--stochastic",
        action="store_true",
        help="Sample actions from the policy distribution instead of using the deterministic mean",
    )
    parser.add_argument("--output", default="attribution.json")
    args = parser.parse_args()

    if Image is None:
        print("Warning: Pillow not installed -- frame previews will be omitted. `pip install pillow` to include them.")

    module = load_user_module(args.script)
    module.cfg = module.load_config(args.config)

    device = "cpu"  # attribution is a single small forward/backward pass; CPU is fine and avoids CUDA setup here
    policy = module.ButterflyPolicy().to(device)

    checkpoint = torch.load(args.checkpoint, map_location=device)
    if isinstance(checkpoint, dict) and "policy" in checkpoint:
        policy.load_state_dict(checkpoint["policy"])
    else:
        policy.load_state_dict(checkpoint)
    policy.eval()

    patch_attention_capture(policy)

    print(f"Running {args.steps} steps and computing attribution (seed={args.seed})...")
    data = run_episode(
        module,
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
