"""Simple MLP + ResNet-18 actor-critic policy for the butterfly env.

Architecture (single frame -> per-frame latent -> history MLP):

    image [3,64,64]           --torchvision resnet18, fc->proj_dim--> image_feat
    scalar fields             --per-group MLPs + masked pool--> scalar_feat
    concat(image_feat, scalar_feat) --fusion MLP--> latent          (per frame)

    last ``history_length`` per-frame latents are stacked [L, latent] and
    flattened into a --history MLP--> shared features,
    split into actor (tanh-squashed diagonal Gaussian) and critic heads.

The fusion MLP therefore runs once per frame of the history window, and its
output for each of the L frames feeds the history MLP.
"""

import torch
import torch.nn as nn
from torchvision.models import resnet18

from butterfly.config import (
    BIRD_STATE_EMBEDDING_COUNT,
    FOOD_TYPE_EMBEDDING_COUNT,
    Config,
)

__all__ = ["SimplePolicy"]


class _Block(nn.Module):
    """Two-layer MLP with ReLU between layers."""

    def __init__(self, in_dim, hidden, out_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x):
        return self.net(x)


class SimplePolicy(nn.Module):
    def __init__(
        self,
        config: Config,
        action_size=2,
        history_length=None,
        proj_dim=None,
        hidden_dim=128,
        latent_dim=128,
    ):
        super().__init__()

        if history_length is None:
            history_length = config.environment.history_length
        if proj_dim is None:
            proj_dim = config.network.feature_size

        self.history_length = history_length
        self.action_size = action_size

        self._track_limit = config.butterfly.perception.track_limit
        self._max_birds = config.butterfly.perception.max_tracked_birds

        # Image branch: ResNet-18 trained from scratch, fc swapped for a
        # projection to proj_dim.
        backbone = resnet18(weights=None)
        backbone.fc = nn.Linear(backbone.fc.in_features, proj_dim)
        self.image_encoder = backbone

        # Scalar branch. Categorical fields go through embeddings; real
        # fields are fed through per-group MLPs and masked-mean-pooled over
        # the padded slots.
        self.food_type_embedding = nn.Embedding(
            FOOD_TYPE_EMBEDDING_COUNT, config.network.food_type_embedding_dim
        )
        self.bird_state_embedding = nn.Embedding(
            BIRD_STATE_EMBEDDING_COUNT, config.network.bird_state_embedding_dim
        )

        self.context_mlp = _Block(3, hidden_dim, proj_dim)
        self.food_mlp = _Block(
            3 + config.network.food_type_embedding_dim, hidden_dim, proj_dim
        )
        self.bird_mlp = _Block(
            4 + config.network.bird_state_embedding_dim, hidden_dim, proj_dim
        )
        self.scalar_mlp = _Block(proj_dim * 3, hidden_dim, proj_dim)

        # Per-frame fusion of image + scalar features into a latent vector.
        self.fusion_mlp = _Block(proj_dim * 2, hidden_dim, latent_dim)

        # Temporal aggregation over the stacked per-frame latents.
        self.history_mlp = _Block(
            latent_dim * history_length, hidden_dim * 2, hidden_dim
        )
        self.policy_head = nn.Linear(hidden_dim, action_size)
        self.value_head = nn.Linear(hidden_dim, 1)
        self.log_std = nn.Parameter(torch.zeros(action_size))

    def _scalar_features(self, scalar_dict, flat_n):
        context = scalar_dict["context"].reshape(flat_n, 3).float()
        context_features = self.context_mlp(context)

        food_angle = scalar_dict["food_angle"].reshape(
            flat_n, self._track_limit, 1
        ).float()
        food_radius = scalar_dict["food_radius"].reshape(
            flat_n, self._track_limit, 1
        ).float()
        food_mask = scalar_dict["food_mask"].reshape(
            flat_n, self._track_limit, 1
        ).float()
        food_ids = scalar_dict["food_type_id"].reshape(
            flat_n, self._track_limit
        ).long()
        food_emb = self.food_type_embedding(food_ids)
        food_input = torch.cat(
            [food_angle, food_radius, food_mask, food_emb], dim=-1
        )
        food_features = self.food_mlp(food_input)
        food_pool = self._masked_mean(food_features, food_mask)

        bird_angle = scalar_dict["bird_angle"].reshape(
            flat_n, self._max_birds, 1
        ).float()
        bird_dist = scalar_dict["bird_dist"].reshape(
            flat_n, self._max_birds, 1
        ).float()
        bird_detected = scalar_dict["bird_detected"].reshape(
            flat_n, self._max_birds, 1
        ).float()
        bird_mask = scalar_dict["bird_mask"].reshape(
            flat_n, self._max_birds, 1
        ).float()
        bird_ids = scalar_dict["bird_state_id"].reshape(
            flat_n, self._max_birds
        ).long()
        bird_emb = self.bird_state_embedding(bird_ids)
        bird_input = torch.cat(
            [bird_angle, bird_dist, bird_detected, bird_mask, bird_emb], dim=-1
        )
        bird_features = self.bird_mlp(bird_input)
        bird_pool = self._masked_mean(bird_features, bird_mask)

        scalar_input = torch.cat(
            [context_features, food_pool, bird_pool], dim=-1
        )
        return self.scalar_mlp(scalar_input)

    @staticmethod
    def _masked_mean(features, mask):
        count = mask.sum(dim=1).clamp(min=1.0)
        return (features * mask).sum(dim=1) / count

    def encode_inputs(self, image_sequence, scalar_dict):
        """Per-frame fusion latents, [batch, history, latent_dim]."""
        batch_size, history, channels, height, width = image_sequence.shape
        flat_n = batch_size * history

        images = image_sequence.reshape(flat_n, channels, height, width)
        image_features = self.image_encoder(images)

        scalar_features = self._scalar_features(scalar_dict, flat_n)

        fused = self.fusion_mlp(
            torch.cat([image_features, scalar_features], dim=-1)
        )
        return fused.reshape(batch_size, history, -1)

    def forward(self, image_sequence, scalar_dict):
        """image_sequence [B, L, 3, H, W]; scalar dict of [B, L, ...].

        Returns (action_mean, std, value).
        """
        features = self.encode_inputs(image_sequence, scalar_dict)
        shared = self.history_mlp(features.reshape(
            features.shape[0], -1
        ))
        action_mean = self.policy_head(shared)
        value = self.value_head(shared).squeeze(-1)
        std = self.log_std.exp().expand_as(action_mean)
        return action_mean, std, value

    def sample_action(self, image_sequence, scalar_dict):
        mean, std, value = self.forward(image_sequence, scalar_dict)
        distribution = torch.distributions.Normal(mean, std)
        raw_action = distribution.rsample()
        action = torch.tanh(raw_action)
        log_probability = distribution.log_prob(raw_action)
        log_probability -= torch.log(1 - action.pow(2) + 1e-6)
        log_probability = log_probability.sum(dim=-1)
        return action, log_probability, value

    def evaluate_actions(self, image_sequence, scalar_dict, actions):
        mean, std, value = self.forward(image_sequence, scalar_dict)
        distribution = torch.distributions.Normal(mean, std)
        clipped_actions = actions.clamp(-0.999, 0.999)
        raw_action = torch.atanh(clipped_actions)
        log_probability = distribution.log_prob(raw_action)
        log_probability -= torch.log(1 - clipped_actions.pow(2) + 1e-6)
        log_probability = log_probability.sum(dim=-1)
        entropy = distribution.entropy().sum(dim=-1)
        return log_probability, entropy, value