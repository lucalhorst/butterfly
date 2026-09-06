"""ResNet visual encoder."""

import torch
import torch.nn as nn

__all__ = ["ResNetEncoder", "encode_images_deduped"]


class ResNetEncoder(nn.Module):
    """
    Standard torchvision ResNet-18 backbone, trained from scratch (no
    ImageNet-pretrained weights -- the environment's frames are flat
    synthetic color blocks, not natural photos, so pretrained features
    are unlikely to transfer, and pretrained weights assume 224x224
    input/ImageNet normalization that this 64x64 pipeline doesn't use).

    The final fully-connected classification layer is replaced with a
    plain linear projection to `feature_size` so the output matches
    what ButterflyPolicy and encode_images_deduped expect: [N, 3, H, W]
    in, [N, feature_size] out.

    Heads-up: resnet18 is far heavier (~11M params, full 4-stage
    BasicBlock design) than the previous hand-rolled SmallResNet, which
    was deliberately shrunk for CPU speed since this network runs once
    per env per rollout step (num_envs of them) and again every PPO
    minibatch. Expect noticeably slower training on CPU.
    """

    def __init__(self, feature_size=256):
        super().__init__()

        from torchvision.models import resnet18

        self.backbone = resnet18(weights=None)

        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_features, feature_size)

    def forward(self, x):
        return self.backbone(x)


def encode_images_deduped(visual_encoder, images):
    """
    images: [N, C, H, W] -- a flattened batch*history stack of frames
    from a single minibatch forward pass.

    Overlapping history windows from nearby samples in the same
    minibatch can share identical raw frames (e.g. sample t's window
    and sample t+1's window overlap in 7 of 8 frames). This finds
    exact-duplicate frames within the minibatch, runs the (expensive)
    visual_encoder on each unique frame only once, then scatters the
    results back out to every original position.

    This is "minibatch-local" dedup: it stays fully correct under PPO
    (every minibatch still uses the current, un-cached weights -- no
    gradient staleness), unlike caching features across minibatches or
    epochs would. Because minibatches are randomly shuffled from the
    whole rollout, the number of duplicate frames found here varies
    run to run -- savings are real but modest, not a guaranteed 8x.
    """

    flat = images.reshape(images.shape[0], -1)

    unique_flat, inverse_indices = torch.unique(flat, dim=0, return_inverse=True)

    unique_images = unique_flat.reshape(-1, *images.shape[1:])

    unique_features = visual_encoder(unique_images)

    return unique_features[inverse_indices]