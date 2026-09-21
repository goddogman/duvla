"""Small action-head training utilities for the frozen-Qwen milestone."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from duvla.models.continuous_vla import ContinuousVLAPolicy


@dataclass(frozen=True)
class ActionHeadTrainResult:
    initial_loss: float
    final_loss: float
    steps: int


def fit_action_head(
    policy: ContinuousVLAPolicy,
    vlm_features: Tensor,
    states: Tensor,
    targets: Tensor,
    valid_mask: Tensor,
    *,
    steps: int,
    learning_rate: float = 1e-4,
    batch_size: int | None = None,
    seed: int = 17,
) -> ActionHeadTrainResult:
    """Fit the compact action head on cached, frozen-Qwen features.

    The function intentionally receives features rather than a Qwen module;
    this makes the frozen-backbone boundary and memory behavior explicit.
    """

    if steps <= 0 or learning_rate <= 0:
        raise ValueError("steps and learning_rate must be positive")
    if vlm_features.shape[0] != states.shape[0] or states.shape[0] != targets.shape[0]:
        raise ValueError("features, states, and targets must have the same batch size")
    if batch_size is not None and batch_size <= 0:
        raise ValueError("batch_size must be positive when supplied")
    if seed < 0:
        raise ValueError("seed must be non-negative")
    sample_count = vlm_features.shape[0]
    effective_batch_size = sample_count if batch_size is None else min(batch_size, sample_count)
    generator = torch.Generator(device=vlm_features.device)
    generator.manual_seed(seed)
    optimizer = torch.optim.AdamW(policy.parameters(), lr=learning_rate)
    policy.train()
    losses: list[float] = []
    for _ in range(steps):
        if effective_batch_size == sample_count:
            indices = torch.arange(sample_count, device=vlm_features.device)
        else:
            indices = torch.randint(
                sample_count,
                (effective_batch_size,),
                generator=generator,
                device=vlm_features.device,
            )
        optimizer.zero_grad(set_to_none=True)
        prediction = policy(
            vlm_features.index_select(0, indices), states.index_select(0, indices)
        )
        loss = policy.loss(
            prediction,
            targets.index_select(0, indices),
            valid_mask.index_select(0, indices),
        )
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach()))
    return ActionHeadTrainResult(initial_loss=losses[0], final_loss=losses[-1], steps=steps)
