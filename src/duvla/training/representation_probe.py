"""Lightweight probes for auditing frozen-Qwen token representations.

These helpers are diagnostic only.  Task labels may be used as a control
variable in a probe, but are never an input to the deployed policy.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class RidgeProbe:
    """Standardized multi-output ridge regressor."""

    feature_mean: Tensor
    feature_std: Tensor
    target_mean: Tensor
    weights: Tensor

    def predict(self, features: Tensor) -> Tensor:
        normalized = (features.float() - self.feature_mean) / self.feature_std
        design = torch.cat(
            (normalized, torch.ones(normalized.shape[0], 1, dtype=normalized.dtype)),
            dim=1,
        )
        return design @ self.weights + self.target_mean


def select_task_blocks(
    task_indices: Tensor,
    *,
    samples_per_task: int,
    blocks_per_task: int,
    task_count: int = 40,
) -> Tensor:
    """Select compact, phase-spanning blocks for every task.

    Contiguous blocks keep the diagnostic inexpensive because feature-cache
    shards are ordered, while block centres distributed through each task
    avoid measuring only the beginning of demonstrations.
    """

    if task_indices.ndim != 1:
        raise ValueError("task_indices must have shape [samples]")
    if samples_per_task <= 0 or blocks_per_task <= 0:
        raise ValueError("samples_per_task and blocks_per_task must be positive")
    if blocks_per_task > samples_per_task:
        raise ValueError("blocks_per_task cannot exceed samples_per_task")
    selected: list[Tensor] = []
    base = samples_per_task // blocks_per_task
    remainder = samples_per_task % blocks_per_task
    for task in range(task_count):
        rows = torch.nonzero(task_indices == task, as_tuple=False).flatten()
        if rows.numel() < samples_per_task:
            raise ValueError(
                f"task {task} has {rows.numel()} rows, fewer than {samples_per_task}"
            )
        task_selection: list[Tensor] = []
        for block in range(blocks_per_task):
            length = base + int(block < remainder)
            centre = round((block + 0.5) * rows.numel() / blocks_per_task)
            start = max(0, min(centre - length // 2, rows.numel() - length))
            task_selection.append(rows[start : start + length])
        task_rows = torch.cat(task_selection)
        if torch.unique(task_rows).numel() != samples_per_task:
            raise RuntimeError(f"task {task} block selection contains duplicates")
        selected.append(task_rows)
    return torch.sort(torch.cat(selected)).values


def aggregate_visual_tokens(
    features: Tensor,
    *,
    token_bins: int,
) -> tuple[Tensor, Tensor]:
    """Return pooled and order-preserving summaries of selected channels.

    ``features`` is ``[B, cameras, tokens, channels]``.  The ordered output
    preserves coarse sequence bins only; it must not be described as a 2-D
    image grid because the V40 cache contains ordered Qwen prompt-token bins.
    """

    if features.ndim != 4:
        raise ValueError("features must have shape [batch, cameras, tokens, channels]")
    if token_bins <= 0 or features.shape[2] % token_bins != 0:
        raise ValueError("token count must be divisible by token_bins")
    values = features.float()
    pooled = values.mean(dim=2).flatten(1)
    ordered = values.reshape(
        values.shape[0],
        values.shape[1],
        token_bins,
        values.shape[2] // token_bins,
        values.shape[3],
    ).mean(dim=3)
    return pooled, ordered.flatten(1)


def aggregate_spatial_grid_tokens(
    features: Tensor,
    *,
    input_grid: tuple[int, int],
    output_grid: tuple[int, int],
) -> tuple[Tensor, Tensor]:
    """Return pooled and true 2-D block summaries of raster image tokens."""

    if features.ndim != 4:
        raise ValueError("features must have shape [batch, cameras, tokens, channels]")
    input_height, input_width = input_grid
    output_height, output_width = output_grid
    if input_height * input_width != features.shape[2]:
        raise ValueError("input grid does not match the token count")
    if (
        min(input_height, input_width, output_height, output_width) <= 0
        or input_height % output_height
        or input_width % output_width
    ):
        raise ValueError("output grid must divide the positive input grid")
    values = features.float().reshape(
        features.shape[0],
        features.shape[1],
        input_height,
        input_width,
        features.shape[3],
    )
    pooled = values.mean(dim=(2, 3)).flatten(1)
    spatial = values.reshape(
        values.shape[0],
        values.shape[1],
        output_height,
        input_height // output_height,
        output_width,
        input_width // output_width,
        values.shape[-1],
    ).mean(dim=(3, 5))
    return pooled, spatial.flatten(1)


def make_probe_design(
    visual: Tensor | None,
    states: Tensor,
    task_indices: Tensor,
    *,
    task_count: int = 40,
) -> Tensor:
    """Build a diagnostic design matrix with task/state controls."""

    if states.ndim != 2:
        raise ValueError("states must have shape [batch, state_dim]")
    if task_indices.shape != (states.shape[0],):
        raise ValueError("task_indices must match the batch dimension")
    if bool(((task_indices < 0) | (task_indices >= task_count)).any()):
        raise ValueError("task index is outside the diagnostic task range")
    task_control = torch.nn.functional.one_hot(
        task_indices.long(), num_classes=task_count
    ).float()
    parts = [states.float(), task_control]
    if visual is not None:
        if visual.ndim != 2 or visual.shape[0] != states.shape[0]:
            raise ValueError("visual features must have shape [batch, feature_dim]")
        parts.append(visual.float())
    return torch.cat(parts, dim=1)


def fit_ridge_probe(features: Tensor, targets: Tensor, *, alpha: float) -> RidgeProbe:
    """Fit a small multi-output ridge probe without extra dependencies."""

    if features.ndim != 2 or targets.ndim != 2:
        raise ValueError("features and targets must both be rank two")
    if features.shape[0] != targets.shape[0] or features.shape[0] < 2:
        raise ValueError("features and targets need matching non-trivial batches")
    if alpha <= 0:
        raise ValueError("alpha must be positive")
    x = features.float()
    y = targets.float()
    feature_mean = x.mean(dim=0)
    feature_std = x.std(dim=0, unbiased=False).clamp_min(1e-6)
    target_mean = y.mean(dim=0)
    x = (x - feature_mean) / feature_std
    y = y - target_mean
    design = torch.cat((x, torch.ones(x.shape[0], 1, dtype=x.dtype)), dim=1)
    gram = design.transpose(0, 1) @ design
    penalty = torch.eye(gram.shape[0], dtype=gram.dtype) * alpha
    penalty[-1, -1] = 0.0
    rhs = design.transpose(0, 1) @ y
    try:
        weights = torch.linalg.solve(gram + penalty, rhs)
    except torch.linalg.LinAlgError:
        weights = torch.linalg.lstsq(gram + penalty, rhs).solution
    return RidgeProbe(feature_mean, feature_std, target_mean, weights)


def probe_metrics(predictions: Tensor, targets: Tensor, *, horizon: int) -> dict[str, float]:
    """Compute normalized arm-action errors for a flattened action chunk."""

    expected = horizon * 6
    if predictions.shape != targets.shape or predictions.shape[1] != expected:
        raise ValueError(f"probe tensors must have shape [batch, {expected}]")
    errors = (predictions.float() - targets.float()).reshape(-1, horizon, 6).square()
    total = float(errors.mean())
    position = float(errors[..., :3].mean())
    rotation = float(errors[..., 3:].mean())
    target_variance = float(
        (targets.float() - targets.float().mean(dim=0, keepdim=True)).square().mean()
    )
    return {
        "arm_mse": total,
        "position_mse": position,
        "rotation_mse": rotation,
        "r2": 1.0 - total / max(target_variance, 1e-12),
    }
