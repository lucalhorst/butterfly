"""ResNet + Transformer policy network."""

import torch
import torch.nn as nn

from butterfly.config import Config
from butterfly.model.encoder import ResNetEncoder, encode_images_deduped

__all__ = ["ButterflyPolicy"]


class ButterflyPolicy(nn.Module):
    def __init__(self, config: Config, action_size=2, history_length=None):
        super().__init__()

        if history_length is None:
            history_length = config.environment.history_length

        self.history_length = history_length

        self.visual_encoder = ResNetEncoder(config.network.feature_size)

        scalar_input_size = config.environment.scalar_input_size()

        self.scalar_encoder = nn.Sequential(
            nn.Linear(scalar_input_size, 128),
            nn.ReLU(),
            nn.Linear(128, config.network.feature_size),
            nn.ReLU(),
        )

        # Fuse visual and scalar features via concatenation + a learned
        # projection, instead of adding them. Addition forces both
        # modalities into the same 256-dim subspace with no way to
        # learn how to weight/mix them; concatenation lets the
        # network learn that mixing.
        self.fusion = nn.Sequential(
            nn.Linear(config.network.feature_size * 2, config.network.feature_size),
            nn.ReLU(),
        )

        self.position_embedding = nn.Parameter(
            torch.zeros(1, history_length, config.network.feature_size)
        )

        transformer_layer = nn.TransformerEncoderLayer(
            d_model=config.network.feature_size,
            nhead=config.network.transformer_heads,
            dim_feedforward=config.network.transformer_feedforward,
            dropout=config.network.transformer_dropout,
            batch_first=True,
            activation="gelu",
        )

        self.transformer = nn.TransformerEncoder(
            transformer_layer, num_layers=config.network.transformer_layers
        )

        self.policy_head = nn.Sequential(
            nn.Linear(config.network.feature_size, 128),
            nn.Tanh(),
            nn.Linear(128, action_size),
        )

        self.value_head = nn.Sequential(
            nn.Linear(config.network.feature_size, 128),
            nn.Tanh(),
            nn.Linear(128, 1),
        )

        self.log_std = nn.Parameter(torch.zeros(action_size))

    def encode_inputs(self, image_sequence, scalar_sequence):
        """
        image_sequence:
            [batch, history, 3, 64, 64]

        scalar_sequence:
            [batch, history, 21]
        """

        batch_size, history, channels, height, width = image_sequence.shape

        images = image_sequence.reshape(batch_size * history, channels, height, width)

        image_features = encode_images_deduped(self.visual_encoder, images)

        image_features = image_features.reshape(batch_size, history, -1)

        scalar_features = self.scalar_encoder(scalar_sequence)

        # Concatenate along the feature dim, then project back down to
        # feature_size with a learned layer (see self.fusion above).
        combined_features = torch.cat([image_features, scalar_features], dim=-1)
        combined_features = self.fusion(combined_features)

        return combined_features

    def forward(self, image_sequence, scalar_sequence):
        features = self.encode_inputs(image_sequence, scalar_sequence)

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

    def sample_action(self, image_sequence, scalar_sequence):
        mean, std, value = self.forward(image_sequence, scalar_sequence)

        distribution = torch.distributions.Normal(mean, std)

        raw_action = distribution.rsample()
        action = torch.tanh(raw_action)

        log_probability = distribution.log_prob(raw_action)

        log_probability -= torch.log(1 - action.pow(2) + 1e-6)

        log_probability = log_probability.sum(dim=-1)

        return action, log_probability, value

    def evaluate_actions(self, image_sequence, scalar_sequence, actions):
        mean, std, value = self.forward(image_sequence, scalar_sequence)

        distribution = torch.distributions.Normal(mean, std)

        clipped_actions = actions.clamp(-0.999, 0.999)

        raw_action = torch.atanh(clipped_actions)

        log_probability = distribution.log_prob(raw_action)

        log_probability -= torch.log(1 - clipped_actions.pow(2) + 1e-6)

        log_probability = log_probability.sum(dim=-1)
        entropy = distribution.entropy().sum(dim=-1)

        return log_probability, entropy, value
