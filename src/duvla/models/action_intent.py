"""Structured coarse action intents for Duvla V3.13.

The codebook is fitted only to training-split residuals relative to the frozen
V2.6 median policy.  The online reasoner receives natural-language policy
context, robot state, and the median action chunk; it never receives a LIBERO
task index, reward, success predicate, or simulator state.

``P4`` is retained only as the internal experiment-stage identifier for
backward-compatible artifact paths.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class ActionIntentConfig:
    context_dim: int = 768
    state_dim: int = 8
    action_dim: int = 7
    action_horizon: int = 8
    executed_steps: int = 2
    code_count: int = 64
    hidden_dim: int = 256
    transformer_layers: int = 2
    attention_heads: int = 8
    dropout: float = 0.0

    def __post_init__(self) -> None:
        positive = (
            self.context_dim,
            self.state_dim,
            self.action_dim,
            self.action_horizon,
            self.executed_steps,
            self.code_count,
            self.hidden_dim,
            self.transformer_layers,
            self.attention_heads,
        )
        if any(value <= 0 for value in positive):
            raise ValueError("all action-intent dimensions must be positive")
        if self.executed_steps > self.action_horizon:
            raise ValueError("executed_steps cannot exceed action_horizon")
        if self.hidden_dim % self.attention_heads:
            raise ValueError("hidden_dim must be divisible by attention_heads")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")


def nearest_action_intent(residual_prefix: Tensor, codebook: Tensor) -> Tensor:
    """Return the nearest train-derived residual code under mean squared error."""

    if residual_prefix.ndim != 3 or codebook.ndim != 3:
        raise ValueError("residual_prefix and codebook must be [rows/codes, steps, action]")
    if residual_prefix.shape[1:] != codebook.shape[1:]:
        raise ValueError("residual prefix and codebook shapes differ")
    flat = residual_prefix.float().flatten(1)
    codes = codebook.float().flatten(1)
    distance = (
        flat.square().sum(dim=1, keepdim=True)
        + codes.square().sum(dim=1)[None]
        - 2.0 * flat @ codes.transpose(0, 1)
    ).clamp_min_(0.0)
    return distance.argmin(dim=1)


def fit_action_intent_codebook(
    residual_prefix: Tensor,
    *,
    code_count: int,
    iterations: int,
    seed: int,
    assignment_batch_size: int = 4096,
) -> tuple[Tensor, Tensor]:
    """Fit a deterministic k-means++ codebook and return codes and counts."""

    if residual_prefix.ndim != 3 or not residual_prefix.is_floating_point():
        raise ValueError("residual_prefix must be a floating [rows, steps, action] tensor")
    if not bool(torch.isfinite(residual_prefix).all()):
        raise ValueError("residual_prefix contains non-finite values")
    if not 1 <= code_count <= residual_prefix.shape[0]:
        raise ValueError("code_count must be in [1, rows]")
    if iterations <= 0 or assignment_batch_size <= 0:
        raise ValueError("iterations and assignment_batch_size must be positive")
    samples = residual_prefix.float().flatten(1).cpu()
    generator = torch.Generator().manual_seed(seed)
    first = int(torch.randint(samples.shape[0], (1,), generator=generator))
    selected = [first]
    closest = (samples - samples[first]).square().sum(dim=1)
    for _ in range(1, code_count):
        total = closest.sum()
        if float(total) <= 0.0:
            remaining = torch.ones(samples.shape[0], dtype=torch.bool)
            remaining[selected] = False
            candidate = remaining.nonzero(as_tuple=False)[0, 0]
        else:
            candidate = torch.multinomial(
                closest / total, 1, generator=generator
            )[0]
        index = int(candidate)
        selected.append(index)
        closest = torch.minimum(
            closest, (samples - samples[index]).square().sum(dim=1)
        )
    centers = samples[selected].clone()
    counts = torch.zeros(code_count, dtype=torch.long)
    for _ in range(iterations):
        sums = torch.zeros_like(centers)
        counts.zero_()
        for chunk in samples.split(assignment_batch_size):
            distance = torch.cdist(chunk, centers)
            assignment = distance.argmin(dim=1)
            sums.index_add_(0, assignment, chunk)
            counts.index_add_(
                0, assignment, torch.ones(assignment.shape[0], dtype=torch.long)
            )
        active = counts > 0
        centers[active] = sums[active] / counts[active, None]
    return centers.reshape(code_count, *residual_prefix.shape[1:]), counts.clone()


class StructuredActionIntentReasoner(nn.Module):
    """Predict coarse residual action codes from grounded policy context."""

    def __init__(self, config: ActionIntentConfig, codebook: Tensor) -> None:
        super().__init__()
        if codebook.shape != (
            config.code_count,
            config.executed_steps,
            config.action_dim,
        ):
            raise ValueError("action-intent codebook has an incompatible shape")
        if not bool(torch.isfinite(codebook.float()).all()):
            raise ValueError("action-intent codebook contains non-finite values")
        self.config = config
        self.register_buffer("codebook", codebook.float().clone())
        self.context_projection = nn.Sequential(
            nn.LayerNorm(config.context_dim),
            nn.Linear(config.context_dim, config.hidden_dim),
        )
        self.state_projection = nn.Sequential(
            nn.LayerNorm(config.state_dim),
            nn.Linear(config.state_dim, config.hidden_dim),
        )
        self.action_projection = nn.Linear(config.action_dim, config.hidden_dim)
        self.action_positions = nn.Parameter(
            torch.randn(config.executed_steps, config.hidden_dim) * 0.02
        )
        self.intent_token = nn.Parameter(torch.randn(config.hidden_dim) * 0.02)
        self.context_attention = nn.MultiheadAttention(
            config.hidden_dim,
            config.attention_heads,
            dropout=config.dropout,
            batch_first=True,
        )
        layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_dim,
            nhead=config.attention_heads,
            dim_feedforward=config.hidden_dim * 4,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.temporal_encoder = nn.TransformerEncoder(
            layer, num_layers=config.transformer_layers
        )
        self.classifier = nn.Linear(config.hidden_dim, config.code_count)

    def forward(self, context: Tensor, state: Tensor, base_actions: Tensor) -> Tensor:
        config = self.config
        if context.ndim != 3 or context.shape[-1] != config.context_dim:
            raise ValueError("context must be [batch, tokens, context_dim]")
        if state.shape != (context.shape[0], config.state_dim):
            raise ValueError("state must be [batch, state_dim]")
        if base_actions.shape != (
            context.shape[0],
            config.action_horizon,
            config.action_dim,
        ):
            raise ValueError("base_actions do not match the action-intent contract")
        dtype = self.context_projection[0].weight.dtype
        memory = self.context_projection(context.to(dtype=dtype))
        state_token = self.state_projection(state.to(dtype=dtype))[:, None]
        action = self.action_projection(
            base_actions[:, : config.executed_steps].to(dtype=dtype)
        )
        action = action + self.action_positions[None] + state_token
        summary = self.intent_token[None, None].expand(context.shape[0], 1, -1)
        queries = torch.cat((summary + state_token, action), dim=1)
        grounded, _ = self.context_attention(
            queries, memory, memory, need_weights=False
        )
        encoded = self.temporal_encoder(queries + grounded)
        return self.classifier(encoded[:, 0])

    def candidates(
        self,
        context: Tensor,
        state: Tensor,
        base_actions: Tensor,
        *,
        top_k: int = 5,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Return top-k structured alternatives followed by the unchanged base."""

        if not 1 <= top_k <= self.config.code_count:
            raise ValueError("top_k must be in [1, code_count]")
        logits = self(context, state, base_actions)
        scores, indices = logits.topk(top_k, dim=1)
        residual = self.codebook[indices].to(dtype=base_actions.dtype)
        alternatives = base_actions[:, None].expand(-1, top_k, -1, -1).clone()
        alternatives[:, :, : self.config.executed_steps] += residual
        candidates = torch.cat((alternatives, base_actions[:, None]), dim=1)
        return candidates, indices, scores
