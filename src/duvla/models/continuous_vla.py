"""A small continuous-action VLA policy skeleton.

This is an original implementation of the SmolVLA-style decomposition:
multimodal VLM context + robot state -> compact action expert -> action chunk.
The Qwen processor/backbone is intentionally injected later through
``vlm_features`` so this module can be tested without downloading code or
constructing a second copy of the Qwen weights.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class ContinuousVLAConfig:
    """Shape and capacity choices for the first RTX 5060-compatible policy."""

    state_dim: int = 8
    action_dim: int = 7
    gripper_loss_weight: float = 1.0
    vlm_feature_dim: int = 2048
    hidden_dim: int = 256
    action_horizon: int = 8
    max_context_tokens: int = 8
    context_tokens_per_observation: int = 8
    expert_layers: int = 2
    expert_heads: int = 4
    dropout: float = 0.0

    def __post_init__(self) -> None:
        if self.state_dim <= 0 or self.action_dim <= 0 or self.vlm_feature_dim <= 0:
            raise ValueError("feature dimensions must be positive")
        if self.gripper_loss_weight <= 0:
            raise ValueError("gripper_loss_weight must be positive")
        if (
            self.hidden_dim <= 0
            or self.action_horizon <= 0
            or self.max_context_tokens <= 0
            or self.context_tokens_per_observation <= 0
        ):
            raise ValueError(
                "hidden_dim, action_horizon, max_context_tokens and "
                "context_tokens_per_observation must be positive"
            )
        if self.hidden_dim % self.expert_heads != 0:
            raise ValueError("hidden_dim must be divisible by expert_heads")


class StateEncoder(nn.Module):
    """Encode the opaque dataset state vector without assigning semantics."""

    def __init__(self, state_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.projection = nn.Sequential(
            nn.LayerNorm(state_dim),
            nn.Linear(state_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, state: Tensor) -> Tensor:
        if state.ndim != 2:
            raise ValueError(f"state must have shape [batch, state_dim], got {tuple(state.shape)}")
        return self.projection(state)


class ActionExpert(nn.Module):
    """Compact Transformer action expert conditioned on context tokens."""

    def __init__(self, config: ContinuousVLAConfig) -> None:
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_dim,
            nhead=config.expert_heads,
            dim_feedforward=4 * config.hidden_dim,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=config.expert_layers)
        self.context_position = nn.Parameter(
            torch.randn(config.max_context_tokens, config.hidden_dim) * 0.02
        )
        self.action_queries = nn.Parameter(
            torch.randn(config.action_horizon, config.hidden_dim) * 0.02
        )
        self.action_projection = nn.Linear(config.hidden_dim, config.action_dim)

    def forward(self, context: Tensor, state_token: Tensor) -> Tensor:
        if context.ndim != 3:
            raise ValueError(f"context must have shape [batch, tokens, hidden], got {tuple(context.shape)}")
        if context.shape[1] > self.context_position.shape[0]:
            raise ValueError(
                f"context has {context.shape[1]} tokens, maximum is {self.context_position.shape[0]}"
            )
        if state_token.ndim != 3 or state_token.shape[1] != 1:
            raise ValueError("state_token must have shape [batch, 1, hidden]")
        batch_size = context.shape[0]
        queries = self.action_queries.unsqueeze(0).expand(batch_size, -1, -1)
        context = context + self.context_position[: context.shape[1]].unsqueeze(0)
        tokens = torch.cat((context, state_token, queries), dim=1)
        encoded = self.transformer(tokens)
        return self.action_projection(encoded[:, -queries.shape[1] :])


class ContinuousVLAPolicy(nn.Module):
    """Qwen-feature-conditioned continuous action policy.

    ``vlm_features`` may be a pooled Qwen feature ``[B, D]`` or a sequence of
    multimodal hidden states ``[B, T, D]``.  Keeping this boundary explicit
    lets the Qwen wrapper evolve without changing the action expert contract.
    """

    def __init__(self, config: ContinuousVLAConfig | None = None) -> None:
        super().__init__()
        self.config = config or ContinuousVLAConfig()
        self.vlm_projection = nn.Linear(self.config.vlm_feature_dim, self.config.hidden_dim)
        self.state_encoder = StateEncoder(self.config.state_dim, self.config.hidden_dim)
        self.action_expert = ActionExpert(self.config)

    def forward(self, vlm_features: Tensor, state: Tensor) -> Tensor:
        if vlm_features.ndim == 2:
            vlm_features = vlm_features.unsqueeze(1).unsqueeze(1)
        elif vlm_features.ndim == 3:
            vlm_features = vlm_features.unsqueeze(1)
        if vlm_features.ndim != 4:
            raise ValueError(
                "vlm_features must have shape [batch, feature_dim], [batch, tokens, feature_dim], "
                "or [batch, history, tokens, feature_dim]"
            )
        if vlm_features.shape[-1] != self.config.vlm_feature_dim:
            raise ValueError(
                f"expected VLM feature dim {self.config.vlm_feature_dim}, "
                f"got {vlm_features.shape[-1]}"
            )
        if state.ndim == 3:
            state = state[:, -1]
        if state.ndim != 2 or state.shape[-1] != self.config.state_dim:
            raise ValueError(
                f"expected state shape [batch, {self.config.state_dim}] or [batch, history, state_dim], "
                f"got {tuple(state.shape)}"
            )
        if vlm_features.shape[0] != state.shape[0]:
            raise ValueError("VLM features and state must have the same batch size")
        context_tokens = vlm_features.shape[1] * vlm_features.shape[2]
        if context_tokens > self.config.max_context_tokens:
            raise ValueError(
                f"VLM context has {context_tokens} tokens, maximum is {self.config.max_context_tokens}"
            )
        vlm_features = vlm_features.reshape(vlm_features.shape[0], context_tokens, -1)
        context = self.vlm_projection(vlm_features.to(self.vlm_projection.weight.dtype))
        state_projection = self.state_encoder.projection[1]
        state_token = self.state_encoder(state.to(state_projection.weight.dtype)).unsqueeze(1)
        return self.action_expert(context, state_token)

    def loss(self, prediction: Tensor, target: Tensor, valid_mask: Tensor) -> Tensor:
        """Compute masked Huber loss for normalized or raw action chunks."""

        expected = (
            prediction.ndim == 3
            and prediction.shape[1] == self.config.action_horizon
            and prediction.shape[2] == self.config.action_dim
        )
        if not expected or target.shape != prediction.shape:
            raise ValueError("prediction and target must both have shape [batch, horizon, action_dim]")
        if valid_mask.shape != prediction.shape[:2]:
            raise ValueError("valid_mask must have shape [batch, horizon]")
        element_loss = nn.functional.smooth_l1_loss(prediction, target, reduction="none")
        action_weights = prediction.new_ones(self.config.action_dim)
        action_weights[-1] = self.config.gripper_loss_weight
        element_loss = element_loss * action_weights.view(1, 1, -1)
        mask = valid_mask.to(dtype=element_loss.dtype).unsqueeze(-1)
        denominator = mask.sum() * action_weights.sum()
        if denominator.item() == 0:
            raise ValueError("valid_mask must contain at least one valid action")
        return (element_loss * mask).sum() / denominator
