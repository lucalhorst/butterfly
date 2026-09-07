"""ResNet + hierarchical entity/temporal Transformer policy network."""

import torch
import torch.nn as nn

from butterfly.config import (
    BIRD_STATE_EMBEDDING_COUNT,
    FOOD_TYPE_EMBEDDING_COUNT,
    Config,
)
from butterfly.model.encoder import ResNetEncoder, encode_images_deduped

__all__ = ["EntityEncoder", "ButterflyPolicy"]


class EntityEncoder(nn.Module):
    """
    Per-frame entity encoder: builds a set of tokens (image + context +
    food + bird) for each timestep, runs a small entity transformer over
    them, and masked-mean-pools to a per-frame summary vector.

    The same weights are applied across all ``history_length`` frames --
    the caller batches the time dimension into the batch dimension
    (``[batch*history, ...]``) for this stage, like the image encoder.
    """

    def __init__(self, config: Config, action_size=2):
        super().__init__()

        net = config.network
        perception = config.butterfly.perception
        feature_size = net.feature_size
        self.food_track_limit = perception.track_limit
        self.max_birds = perception.max_tracked_birds

        self.visual_encoder = ResNetEncoder(feature_size)

        self.context_encoder = nn.Sequential(
            nn.Linear(3, 128),
            nn.ReLU(),
            nn.Linear(128, feature_size),
        )

        # Food token: cat(angle, radius, type_embedding) per slot.
        self.food_type_embedding = nn.Embedding(
            FOOD_TYPE_EMBEDDING_COUNT, net.food_type_embedding_dim
        )
        self.food_token_encoder = nn.Sequential(
            nn.Linear(2 + net.food_type_embedding_dim, 128),
            nn.ReLU(),
            nn.Linear(128, feature_size),
        )

        # Bird token: cat(angle, dist, state_embedding, detected) per bird.
        self.bird_state_embedding = nn.Embedding(
            BIRD_STATE_EMBEDDING_COUNT, net.bird_state_embedding_dim
        )
        self.bird_token_encoder = nn.Sequential(
            nn.Linear(3 + net.bird_state_embedding_dim, 128),
            nn.ReLU(),
            nn.Linear(128, feature_size),
        )

        entity_layer = nn.TransformerEncoderLayer(
            d_model=feature_size,
            nhead=net.entity_transformer.heads,
            dim_feedforward=net.entity_transformer.feedforward,
            dropout=net.entity_transformer.dropout,
            batch_first=True,
            activation="gelu",
        )
        self.entity_transformer = nn.TransformerEncoder(
            entity_layer, num_layers=net.entity_transformer.layers
        )

    def forward(self, image_sequence, scalar_dict):
        """
        image_sequence:
            [batch, history, 3, H, W]
        scalar_dict:
            dict of tensors, each [batch, history, field_dim...]

        Returns the per-frame summary vector [batch, history, feature_size].
        """
        batch_size, history, channels, height, width = image_sequence.shape
        flat_n = batch_size * history

        images = image_sequence.reshape(flat_n, channels, height, width)
        image_features = encode_images_deduped(self.visual_encoder, images)

        context = scalar_dict["context"].reshape(flat_n, -1)
        context_features = self.context_encoder(context)

        food_angle = scalar_dict["food_angle"].reshape(flat_n, self.food_track_limit, 1)
        food_radius = scalar_dict["food_radius"].reshape(
            flat_n, self.food_track_limit, 1
        )
        food_type_ids = scalar_dict["food_type_id"].reshape(
            flat_n, self.food_track_limit
        )
        food_type_emb = self.food_type_embedding(food_type_ids)
        food_input = torch.cat([food_angle, food_radius, food_type_emb], dim=-1)
        food_features = self.food_token_encoder(food_input)

        bird_angle = scalar_dict["bird_angle"].reshape(flat_n, self.max_birds, 1)
        bird_dist = scalar_dict["bird_dist"].reshape(flat_n, self.max_birds, 1)
        bird_state_ids = scalar_dict["bird_state_id"].reshape(flat_n, self.max_birds)
        bird_state_emb = self.bird_state_embedding(bird_state_ids)
        bird_detected = scalar_dict["bird_detected"].reshape(
            flat_n, self.max_birds, 1
        )
        bird_input = torch.cat(
            [bird_angle, bird_dist, bird_state_emb, bird_detected], dim=-1
        )
        bird_features = self.bird_token_encoder(bird_input)

        # Assemble the entity token sequence: image, context, then all food
        # tokens, then all bird tokens.
        entity_tokens = torch.cat(
            [
                image_features.unsqueeze(1),
                context_features.unsqueeze(1),
                food_features,
                bird_features,
            ],
            dim=1,
        )

        # Build the transformer key_padding_mask (True = ignore). Image and
        # context are always real (False to keep them). Food/bird positions
        # are NOT of their real-token masks.
        food_mask = scalar_dict["food_mask"].reshape(
            flat_n, self.food_track_limit
        )
        bird_mask = scalar_dict["bird_mask"].reshape(flat_n, self.max_birds)

        padding_mask = torch.zeros(
            entity_tokens.shape[:2],
            dtype=torch.bool,
            device=entity_tokens.device,
        )
        padding_mask[:, 2 : 2 + self.food_track_limit] = ~food_mask
        padding_mask[:, 2 + self.food_track_limit :] = ~bird_mask

        transformed = self.entity_transformer(entity_tokens, src_key_padding_mask=padding_mask)

        # Masked mean-pool: average over real positions only. Image and
        # context are included unconditionally, so the pool is never empty
        # (safe for the all-padding edge case of no food and no bird).
        valid_mask = ~padding_mask
        entity_sum = (transformed * valid_mask.unsqueeze(-1).float()).sum(dim=1)
        entity_count = valid_mask.sum(dim=1, keepdim=True).float().clamp(min=1.0)
        pooled = entity_sum / entity_count

        return pooled.reshape(batch_size, history, -1)


class ButterflyPolicy(nn.Module):
    def __init__(self, config: Config, action_size=2, history_length=None):
        super().__init__()

        if history_length is None:
            history_length = config.environment.history_length

        self.history_length = history_length
        feature_size = config.network.feature_size

        # One module handle for the entity stage (shares the ResNet, builds
        # the per-frame tokens, transforms + pools them to a summary vector).
        self.entity_encoder = EntityEncoder(config, action_size)

        self.position_embedding = nn.Parameter(
            torch.zeros(1, history_length, feature_size)
        )

        transformer_layer = nn.TransformerEncoderLayer(
            d_model=feature_size,
            nhead=config.network.temporal_transformer.heads,
            dim_feedforward=config.network.temporal_transformer.feedforward,
            dropout=config.network.temporal_transformer.dropout,
            batch_first=True,
            activation="gelu",
        )

        self.transformer = nn.TransformerEncoder(
            transformer_layer,
            num_layers=config.network.temporal_transformer.layers,
        )

        self.policy_head = nn.Sequential(
            nn.Linear(feature_size, 128),
            nn.Tanh(),
            nn.Linear(128, action_size),
        )

        self.value_head = nn.Sequential(
            nn.Linear(feature_size, 128),
            nn.Tanh(),
            nn.Linear(128, 1),
        )

        self.log_std = nn.Parameter(torch.zeros(action_size))

    def encode_inputs(self, image_sequence, scalar_dict):
        """
        image_sequence:
            [batch, history, 3, 64, 64]

        scalar_dict:
            dict of tensors, each [batch, history, ...]

        Returns per-frame summary vectors [batch, history, feature_size].
        """
        return self.entity_encoder(image_sequence, scalar_dict)

    def forward(self, image_sequence, scalar_dict):
        features = self.encode_inputs(image_sequence, scalar_dict)

        sequence_length = features.shape[1]

        features = features + self.position_embedding[:, :sequence_length]

        # NOTE: this causal mask has no effect on correctness as written,
        # since only the *last* timestep's transformer output is ever
        # consumed below (current_state = transformed[:, -1]) -- a token
        # can attend to itself and everything before it either way. It's
        # kept here because it's harmless and would matter if this policy
        # is ever extended to consume outputs from multiple timesteps.
        causal_mask = torch.triu(
            torch.ones(sequence_length, sequence_length, device=features.device),
            diagonal=1,
        ).bool()

        transformed = self.transformer(features, mask=causal_mask)

        current_state = transformed[:, -1]

        action_mean = self.policy_head(current_state)

        value = self.value_head(current_state).squeeze(-1)

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
