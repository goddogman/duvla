"""Contracts for V101 training-only branch-outcome caches."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
from typing import Mapping, Sequence

import torch
from torch import Tensor

from .contracts import ACTION_DIM, STATE_DIM, ContractError


REQUIRED_OUTCOME_FIELDS = frozenset(
    {
        "source_dataset_indices",
        "candidate_actions",
        "candidate_mask",
        "future_features",
        "progress",
        "failure",
        "interaction_event",
        "preferred_index",
        "candidate_error",
        "baseline_error",
        "episode_index",
        "frame_index",
    }
)

FORBIDDEN_TRAINING_OUTCOME_FIELDS = frozenset(
    {
        "benchmark_reward",
        "benchmark_success",
        "reward",
        "success",
        "suite_task_id",
        "task_index",
        "evaluation_initial_state",
    }
)


@dataclass(frozen=True)
class OutcomeCacheManifest:
    schema_version: int
    source_role: str
    candidate_count: int
    action_horizon: int
    executed_action_steps: int
    context_dim: int
    state_dim: int
    action_dim: int
    camera_keys: tuple[str, ...]
    candidate_generator_checkpoint: str
    source_manifest_sha256: str
    recorded_fields: frozenset[str]
    uses_benchmark_reward: bool
    uses_benchmark_task_index: bool
    uses_evaluation_initial_states: bool


@dataclass(frozen=True)
class OutcomeShard:
    """Compact V101 targets aligned row-for-row with one source feature shard."""

    candidate_actions: Tensor
    candidate_mask: Tensor
    future_features: Tensor
    progress: Tensor
    failure: Tensor
    interaction_event: Tensor
    preferred_index: Tensor
    candidate_error: Tensor
    baseline_error: Tensor
    episode_indices: Tensor
    frame_indices: Tensor
    dataset_indices: tuple[int, ...]


def save_outcome_shard(path: str | Path, shard: OutcomeShard) -> None:
    """Validate and atomically publish one compact candidate-target shard."""

    actions = shard.candidate_actions
    if actions.ndim != 4:
        raise ValueError("candidate_actions must have shape [batch, candidates, horizon, action_dim]")
    batch, candidates, _, _ = actions.shape
    matrix_shape = (batch, candidates)
    if tuple(shard.candidate_mask.shape) != matrix_shape or shard.candidate_mask.dtype != torch.bool:
        raise ValueError("candidate_mask must be boolean with shape [batch, candidates]")
    for name, value in (
        ("progress", shard.progress),
        ("failure", shard.failure),
        ("interaction_event", shard.interaction_event),
        ("candidate_error", shard.candidate_error),
    ):
        if tuple(value.shape) != matrix_shape:
            raise ValueError(f"{name} must have shape [batch, candidates]")
    if shard.future_features.ndim != 2 or shard.future_features.shape[0] != batch:
        raise ValueError("future_features must have shape [batch, future_dim]")
    for name, value in (
        ("preferred_index", shard.preferred_index),
        ("baseline_error", shard.baseline_error),
        ("episode_indices", shard.episode_indices),
        ("frame_indices", shard.frame_indices),
    ):
        if tuple(value.shape) != (batch,):
            raise ValueError(f"{name} must have shape [batch]")
    if len(shard.dataset_indices) != batch:
        raise ValueError("dataset_indices must match shard batch size")
    if not bool(shard.candidate_mask.any(dim=1).all()):
        raise ValueError("every row must contain at least one valid candidate")
    if not bool(((shard.preferred_index >= 0) & (shard.preferred_index < candidates)).all()):
        raise ValueError("preferred_index is outside candidate range")
    floating = (
        actions,
        shard.future_features,
        shard.progress,
        shard.candidate_error,
        shard.baseline_error,
    )
    if any(not bool(torch.isfinite(value.float()).all()) for value in floating):
        raise ValueError("outcome shard contains non-finite values")
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    torch.save(
        {
            "candidate_actions": actions.cpu(),
            "candidate_mask": shard.candidate_mask.cpu(),
            "future_features": shard.future_features.cpu(),
            "progress": shard.progress.cpu(),
            "failure": shard.failure.cpu(),
            "interaction_event": shard.interaction_event.cpu(),
            "preferred_index": shard.preferred_index.cpu(),
            "candidate_error": shard.candidate_error.cpu(),
            "baseline_error": shard.baseline_error.cpu(),
            "episode_indices": shard.episode_indices.cpu(),
            "frame_indices": shard.frame_indices.cpu(),
            "dataset_indices": shard.dataset_indices,
        },
        temporary,
    )
    temporary.replace(destination)


def load_outcome_shard(path: str | Path) -> OutcomeShard:
    payload = torch.load(Path(path), map_location="cpu", weights_only=True)
    return OutcomeShard(
        candidate_actions=payload["candidate_actions"],
        candidate_mask=payload["candidate_mask"],
        future_features=payload["future_features"],
        progress=payload["progress"],
        failure=payload["failure"],
        interaction_event=payload["interaction_event"],
        preferred_index=payload["preferred_index"],
        candidate_error=payload["candidate_error"],
        baseline_error=payload["baseline_error"],
        episode_indices=payload["episode_indices"],
        frame_indices=payload["frame_indices"],
        dataset_indices=tuple(int(index) for index in payload["dataset_indices"]),
    )


def list_outcome_shards(directory: str | Path) -> tuple[Path, ...]:
    return tuple(sorted(Path(directory).glob("shard-*.pt")))


def proxy_candidate_targets(
    candidate_actions: Tensor,
    expert_actions: Tensor,
    valid_mask: Tensor,
    baseline_actions: Tensor,
    *,
    gripper_weight: float = 0.25,
) -> dict[str, Tensor]:
    """Build reward-free imitation warm-up targets for candidate ranking.

    This is intentionally labelled a proxy: it ranks candidates by closeness
    to the train demonstration and never claims to represent closed-loop task
    success.  True outcome ranking remains a later train-side branch stage.
    """

    if candidate_actions.ndim != 4:
        raise ValueError("candidate_actions must have rank 4")
    batch, candidates, horizon, action_dim = candidate_actions.shape
    expected_actions = (batch, horizon, action_dim)
    if tuple(expert_actions.shape) != expected_actions or tuple(baseline_actions.shape) != expected_actions:
        raise ValueError(f"expert_actions and baseline_actions must have shape {expected_actions}")
    if tuple(valid_mask.shape) != (batch, horizon) or valid_mask.dtype != torch.bool:
        raise ValueError("valid_mask must be boolean with shape [batch, horizon]")
    if gripper_weight < 0.0:
        raise ValueError("gripper_weight must be non-negative")
    weights = valid_mask.to(candidate_actions.dtype)
    denominator = weights.sum(dim=1).clamp_min(1.0)
    arm_error = (candidate_actions[..., :-1] - expert_actions[:, None, :, :-1]).abs().mean(dim=-1)
    gripper_error = (
        (candidate_actions[..., -1] >= 0)
        != (expert_actions[:, None, :, -1] >= 0)
    ).to(candidate_actions.dtype)
    candidate_error = (
        ((arm_error + gripper_weight * gripper_error) * weights[:, None, :]).sum(dim=-1)
        / denominator[:, None]
    )
    baseline_arm = (baseline_actions[..., :-1] - expert_actions[..., :-1]).abs().mean(dim=-1)
    baseline_gripper = (
        (baseline_actions[..., -1] >= 0) != (expert_actions[..., -1] >= 0)
    ).to(candidate_actions.dtype)
    baseline_error = (
        ((baseline_arm + gripper_weight * baseline_gripper) * weights).sum(dim=-1)
        / denominator
    )
    preferred_index = candidate_error.argmin(dim=1)
    minimum = candidate_error.gather(1, preferred_index[:, None])
    scale = candidate_error.std(dim=1, unbiased=False).clamp_min(0.05)[:, None]
    progress = torch.exp(-(candidate_error - minimum) / scale).clamp(0.0, 1.0)
    failure = candidate_error > baseline_error[:, None]
    expert_sign = expert_actions[..., -1] >= 0
    first_sign = expert_sign[:, :1]
    changed = (expert_sign != first_sign) & valid_mask
    candidate_event_match = (
        ((candidate_actions[..., -1] >= 0) == expert_sign[:, None, :]) | ~valid_mask[:, None, :]
    ).all(dim=-1)
    interaction_event = changed.any(dim=1)[:, None].expand(-1, candidates) & candidate_event_match
    return {
        "progress": progress,
        "failure": failure,
        "interaction_event": interaction_event,
        "preferred_index": preferred_index,
        "candidate_error": candidate_error,
        "baseline_error": baseline_error,
    }


def manifest_sha256(payload: Mapping[str, object]) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return sha256(canonical.encode("utf-8")).hexdigest()


def file_sha256(path: str | Path) -> str:
    digest = sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _positive_int(payload: Mapping[str, object], key: str) -> int:
    value = payload.get(key)
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ContractError(f"{key} must be a positive integer")
    return value


def _string(payload: Mapping[str, object], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value:
        raise ContractError(f"{key} must be a non-empty string")
    return value


def _boolean(payload: Mapping[str, object], key: str) -> bool:
    value = payload.get(key)
    if not isinstance(value, bool):
        raise ContractError(f"{key} must be boolean")
    return value


def _strings(payload: Mapping[str, object], key: str) -> tuple[str, ...]:
    value = payload.get(key)
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ContractError(f"{key} must be a string sequence")
    result = tuple(value)
    if not result or any(not isinstance(item, str) or not item for item in result):
        raise ContractError(f"{key} must contain non-empty strings")
    return result


def validate_outcome_cache_manifest(
    payload: Mapping[str, object],
    *,
    source_manifest: Mapping[str, object],
    expected_source_role: str | None = None,
) -> OutcomeCacheManifest:
    """Validate an outcome cache without allowing evaluation data leakage."""

    schema_version = _positive_int(payload, "schema_version")
    if schema_version != 1:
        raise ContractError("unsupported outcome cache schema_version")
    source_role = _string(payload, "source_role")
    if source_role not in {"train", "validation"}:
        raise ContractError("source_role must be train or validation")
    if source_manifest.get("split") != source_role:
        raise ContractError("source_role does not match the source manifest split")
    if expected_source_role is not None and source_role != expected_source_role:
        raise ContractError(f"outcome cache must have source_role={expected_source_role}")
    candidate_count = _positive_int(payload, "candidate_count")
    if candidate_count < 2:
        raise ContractError("candidate_count must be at least two for ranking")
    action_horizon = _positive_int(payload, "action_horizon")
    executed_action_steps = _positive_int(payload, "executed_action_steps")
    if executed_action_steps > action_horizon:
        raise ContractError("executed_action_steps cannot exceed action_horizon")
    context_dim = _positive_int(payload, "context_dim")
    state_dim = _positive_int(payload, "state_dim")
    action_dim = _positive_int(payload, "action_dim")
    if state_dim != STATE_DIM or action_dim != ACTION_DIM:
        raise ContractError("state/action dimensions do not match the Duvla contract")
    if context_dim != source_manifest.get("feature_dim"):
        raise ContractError("context_dim does not match the source Qwen cache")
    if action_horizon != source_manifest.get("horizon"):
        raise ContractError("action_horizon does not match the source train cache")

    camera_keys = _strings(payload, "camera_keys")
    if list(camera_keys) != source_manifest.get("camera_keys"):
        raise ContractError("camera_keys/order does not match the source train cache")
    recorded_fields = frozenset(_strings(payload, "recorded_fields"))
    missing = REQUIRED_OUTCOME_FIELDS.difference(recorded_fields)
    if missing:
        raise ContractError(f"outcome cache is missing fields: {', '.join(sorted(missing))}")
    forbidden = FORBIDDEN_TRAINING_OUTCOME_FIELDS.intersection(recorded_fields)
    if forbidden:
        raise ContractError(
            "outcome training cache contains forbidden supervision fields: "
            + ", ".join(sorted(forbidden))
        )

    source_hash = _string(payload, "source_manifest_sha256")
    if source_hash != manifest_sha256(source_manifest):
        raise ContractError("source_manifest_sha256 does not match the supplied manifest")
    for statistic in ("state_mean", "state_std", "action_mean", "action_std"):
        if payload.get(statistic) != source_manifest.get(statistic):
            raise ContractError(f"{statistic} must be copied unchanged from train manifest")

    uses_benchmark_reward = _boolean(payload, "uses_benchmark_reward")
    uses_benchmark_task_index = _boolean(payload, "uses_benchmark_task_index")
    uses_evaluation_initial_states = _boolean(payload, "uses_evaluation_initial_states")
    if uses_benchmark_reward or uses_benchmark_task_index or uses_evaluation_initial_states:
        raise ContractError("outcome cache contains forbidden benchmark/evaluation supervision")

    return OutcomeCacheManifest(
        schema_version=schema_version,
        source_role=source_role,
        candidate_count=candidate_count,
        action_horizon=action_horizon,
        executed_action_steps=executed_action_steps,
        context_dim=context_dim,
        state_dim=state_dim,
        action_dim=action_dim,
        camera_keys=camera_keys,
        candidate_generator_checkpoint=_string(payload, "candidate_generator_checkpoint"),
        source_manifest_sha256=source_hash,
        recorded_fields=recorded_fields,
        uses_benchmark_reward=uses_benchmark_reward,
        uses_benchmark_task_index=uses_benchmark_task_index,
        uses_evaluation_initial_states=uses_evaluation_initial_states,
    )
