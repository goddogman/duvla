"""Low-rank frozen-Qwen layer fusion for representation-ceiling experiments."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class QwenLayerFusionConfig:
    """Configuration for a parameter-matched layer-fusion sidecar.

    The parent layer-14 representation is always preserved as the residual
    anchor.  Every auxiliary layer is provided as a compact, train-split-only
    projection with shape ``[..., compressed_dim]``.  Only the scalar gates
    and the up projections are trainable.
    """

    layer_ids: tuple[int, ...] = (12, 14, 18, -1)
    feature_dim: int = 2048
    compressed_dim: int = 12
    base_layer: int = 14
    gate_init: float = 0.01

    def __post_init__(self) -> None:
        if not self.layer_ids:
            raise ValueError("layer_ids cannot be empty")
        if len(set(self.layer_ids)) != len(self.layer_ids):
            raise ValueError("layer_ids must be unique")
        if self.base_layer not in self.layer_ids:
            raise ValueError("base_layer must be present in layer_ids")
        if self.feature_dim <= 0 or self.compressed_dim <= 0:
            raise ValueError("feature_dim and compressed_dim must be positive")
        if not 0.0 < self.gate_init < 1.0:
            raise ValueError("gate_init must be between zero and one")


class FrozenQwenLayerFusion(nn.Module):
    """Fuse compact Qwen layers into an exactly preserved layer-14 anchor.

    ``base_features`` has shape ``[B, C, T, D]`` and ``compressed_layers``
    has shape ``[B, L, C, T, R]``.  Each layer receives a trainable ``R→D``
    projection and a bounded scalar gate.  All up projections are initialized
    to zero while gates start small but non-zero, so an untrained module is
    bitwise-equivalent to the parent and projection gradients are live on the
    first optimizer step.
    """

    def __init__(self, config: QwenLayerFusionConfig) -> None:
        super().__init__()
        self.config = config
        self.projections = nn.ModuleList(
            nn.Linear(config.compressed_dim, config.feature_dim, bias=False)
            for _ in config.layer_ids
        )
        gate_logit = torch.atanh(torch.tensor(config.gate_init)).item()
        self.gate_logits = nn.Parameter(
            torch.full((len(config.layer_ids),), gate_logit)
        )
        for projection in self.projections:
            nn.init.zeros_(projection.weight)

    @property
    def layer_weights(self) -> Tensor:
        """Return bounded signed residual weights used by every layer."""

        return torch.tanh(self.gate_logits)

    def forward(self, base_features: Tensor, compressed_layers: Tensor) -> Tensor:
        if base_features.ndim != 4:
            raise ValueError("base_features must have shape [B, C, T, D]")
        if compressed_layers.ndim != 5:
            raise ValueError("compressed_layers must have shape [B, L, C, T, R]")
        expected_base = (*compressed_layers.shape[:1], *compressed_layers.shape[2:4])
        if tuple(base_features.shape[:3]) != expected_base:
            raise ValueError("base and compressed layer batch/camera/token axes must match")
        if base_features.shape[-1] != self.config.feature_dim:
            raise ValueError("base_features do not match feature_dim")
        if compressed_layers.shape[1] != len(self.config.layer_ids):
            raise ValueError("compressed layer axis does not match layer_ids")
        if compressed_layers.shape[-1] != self.config.compressed_dim:
            raise ValueError("compressed_layers do not match compressed_dim")

        dtype = self.projections[0].weight.dtype
        residual = torch.zeros_like(base_features, dtype=dtype)
        weights = self.layer_weights.to(dtype=dtype)
        compact = compressed_layers.to(dtype=dtype)
        for index, projection in enumerate(self.projections):
            residual = residual + weights[index] * projection(compact[:, index])
        return base_features.to(dtype=dtype) + residual


def trainable_parameter_count(module: nn.Module) -> int:
    """Return the exact number of parameters that receive gradients."""

    return sum(parameter.numel() for parameter in module.parameters() if parameter.requires_grad)
