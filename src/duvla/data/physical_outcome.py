"""Training-only physical branch outcomes for the V1.101 verifier prototype.

The cache contract deliberately excludes LIBERO evaluation initial states,
benchmark rewards, success labels, and benchmark task indices.  Supervision is
obtained by restoring a state from an expert *training demonstration*, applying
each candidate action prefix, and measuring the resulting simulator state
against the expert future state from the same demonstration.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Mapping, Sequence

import torch
from torch import Tensor

from .contracts import ACTION_DIM, STATE_DIM, ContractError


LEGACY_PHYSICAL_OUTCOME_FIELDS = frozenset(
    {
        "source_keys",
        "context",
        "state",
        "candidate_actions",
        "candidate_mask",
        "future_states",
        "progress",
        "failure",
        "preferred_index",
        "candidate_error",
        "current_error",
        "baseline_error",
    }
)

PHYSICAL_OUTCOME_FIELDS = LEGACY_PHYSICAL_OUTCOME_FIELDS | frozenset(
    {
        "simulator_states",
        "current_simulator_state",
        "expert_future_simulator_state",
        "simulator_scale",
        "simulator_weights",
    }
)

FORBIDDEN_PHYSICAL_OUTCOME_FIELDS = frozenset(
    {
        "benchmark_reward",
        "benchmark_success",
        "evaluation_initial_state",
        "reward",
        "success",
        "suite_task_id",
        "task_index",
    }
)


@dataclass(frozen=True)
class PhysicalOutcomeManifest:
    schema_version: int
    source_role: str
    candidate_count: int
    flow_sample_count: int
    baseline_candidate_index: int
    action_horizon: int
    executed_action_steps: int
    context_dim: int
    context_tokens: int
    context_tokens_per_camera: int
    state_dim: int
    action_dim: int
    candidate_generator_checkpoint: str
    candidate_generator_sha256: str
    source_revision: str
    recorded_fields: frozenset[str]
    uses_training_demonstration_states: bool
    uses_benchmark_reward: bool
    uses_benchmark_task_index: bool
    uses_evaluation_initial_states: bool


@dataclass(frozen=True)
class PhysicalOutcomeShard:
    """One compact V1.101 physical-outcome cache shard.

    ``context`` contains deterministically reduced frozen-Qwen tokens.  The
    full 128-token context is used to generate candidates but is not copied to
    disk, keeping the branch cache small.
    """

    context: Tensor
    state: Tensor
    candidate_actions: Tensor
    candidate_mask: Tensor
    future_states: Tensor
    progress: Tensor
    failure: Tensor
    preferred_index: Tensor
    candidate_error: Tensor
    current_error: Tensor
    baseline_error: Tensor
    source_keys: tuple[str, ...]
    simulator_states: Tensor | None = None
    current_simulator_state: Tensor | None = None
    expert_future_simulator_state: Tensor | None = None
    simulator_scale: Tensor | None = None
    simulator_weights: Tensor | None = None


def reduce_ordered_camera_context(context: Tensor, *, tokens_per_camera: int = 8) -> Tensor:
    """Reduce ordered per-camera Qwen tokens with deterministic contiguous bins.

    The operation preserves camera order and token order.  It accepts either
    ``[batch, cameras, tokens, width]`` or ``[cameras, tokens, width]`` and
    returns the same rank with ``tokens_per_camera`` tokens.
    """

    if tokens_per_camera <= 0:
        raise ValueError("tokens_per_camera must be positive")
    squeeze = context.ndim == 3
    if squeeze:
        context = context.unsqueeze(0)
    if context.ndim != 4:
        raise ValueError("context must have shape [batch, cameras, tokens, width]")
    batch, cameras, tokens, width = context.shape
    if cameras != 2:
        raise ValueError("Duvla requires exactly two ordered camera streams")
    if tokens % tokens_per_camera:
        raise ValueError("source token count must be divisible by tokens_per_camera")
    reduced = context.reshape(
        batch, cameras, tokens_per_camera, tokens // tokens_per_camera, width
    ).mean(dim=3)
    return reduced.squeeze(0) if squeeze else reduced


def physical_candidate_targets(
    candidate_states: Tensor,
    current_state: Tensor,
    expert_future_state: Tensor,
    scale: Tensor,
    weights: Tensor,
    *,
    baseline_index: int,
    failure_margin: float = 0.10,
) -> dict[str, Tensor]:
    """Convert true branch endpoints into ranking/calibration targets.

    Inputs are one restored demonstration state with ``K`` candidate branch
    endpoints.  ``scale`` and ``weights`` are task-local physical-state
    normalizers computed only from training demonstrations.  No reward or
    success predicate participates in this target.
    """

    if candidate_states.ndim != 2:
        raise ValueError("candidate_states must have shape [candidates, simulator_state_dim]")
    candidates, state_width = candidate_states.shape
    if current_state.shape != (state_width,):
        raise ValueError("current_state width does not match candidate states")
    if expert_future_state.shape != (state_width,):
        raise ValueError("expert_future_state width does not match candidate states")
    if scale.shape != (state_width,) or weights.shape != (state_width,):
        raise ValueError("scale and weights must match simulator state width")
    if baseline_index < 0 or baseline_index >= candidates:
        raise ValueError("baseline_index is outside candidate range")
    if failure_margin < 0.0:
        raise ValueError("failure_margin must be non-negative")
    floating = (candidate_states, current_state, expert_future_state, scale, weights)
    if any(not bool(torch.isfinite(value.float()).all()) for value in floating):
        raise ValueError("physical target input contains non-finite values")
    if bool((scale <= 0).any()) or bool((weights < 0).any()) or not bool((weights > 0).any()):
        raise ValueError("scale must be positive and weights must be non-negative/non-empty")

    normalized = (candidate_states - expert_future_state[None, :]) / scale[None, :]
    squared = normalized.square() * weights[None, :]
    candidate_error = (squared.sum(dim=-1) / weights.sum()).sqrt()
    current_normalized = (current_state - expert_future_state) / scale
    current_error = (
        (current_normalized.square() * weights).sum() / weights.sum()
    ).sqrt()
    preferred_index = candidate_error.argmin()
    # Express progress relative to the remaining expert-state distance.  This
    # makes calibration comparable across tasks without forcing the best item
    # in a bad candidate pool to look successful.  The clamp only controls the
    # near-zero numerical regime; it does not change candidate ordering.
    relative_gain = (current_error - candidate_error) / current_error.clamp_min(0.1)
    progress = torch.sigmoid(relative_gain)
    baseline_error = candidate_error[baseline_index]
    failure = candidate_error > current_error * (1.0 + failure_margin)
    return {
        "progress": progress,
        "failure": failure,
        "preferred_index": preferred_index,
        "candidate_error": candidate_error,
        "current_error": current_error,
        "baseline_error": baseline_error,
    }


def _validate_shard(shard: PhysicalOutcomeShard) -> tuple[int, int, int, int]:
    if shard.context.ndim != 4:
        raise ValueError("context must have shape [batch, cameras, tokens, context_dim]")
    batch, cameras, _tokens, _width = shard.context.shape
    if cameras != 2:
        raise ValueError("context must preserve two ordered cameras")
    if shard.state.shape != (batch, STATE_DIM):
        raise ValueError(f"state must have shape [batch, {STATE_DIM}]")
    if shard.candidate_actions.ndim != 4:
        raise ValueError("candidate_actions must have rank four")
    action_batch, candidates, horizon, action_dim = shard.candidate_actions.shape
    if action_batch != batch or action_dim != ACTION_DIM:
        raise ValueError("candidate action shape violates the Duvla contract")
    matrix = (batch, candidates)
    if shard.candidate_mask.shape != matrix or shard.candidate_mask.dtype != torch.bool:
        raise ValueError("candidate_mask must be boolean [batch, candidates]")
    if shard.future_states.shape != (batch, candidates, STATE_DIM):
        raise ValueError("future_states must be [batch, candidates, state_dim]")
    for name, value in (
        ("progress", shard.progress),
        ("failure", shard.failure),
        ("candidate_error", shard.candidate_error),
    ):
        if value.shape != matrix:
            raise ValueError(f"{name} must have shape [batch, candidates]")
    if shard.failure.dtype != torch.bool:
        raise ValueError("failure must be boolean")
    if shard.preferred_index.shape != (batch,) or shard.preferred_index.dtype != torch.long:
        raise ValueError("preferred_index must be int64 [batch]")
    if shard.baseline_error.shape != (batch,):
        raise ValueError("baseline_error must have shape [batch]")
    if shard.current_error.shape != (batch,):
        raise ValueError("current_error must have shape [batch]")
    raw = (
        shard.simulator_states,
        shard.current_simulator_state,
        shard.expert_future_simulator_state,
        shard.simulator_scale,
        shard.simulator_weights,
    )
    if any(value is not None for value in raw):
        if any(value is None for value in raw):
            raise ValueError("raw simulator outcome fields must be all present or all absent")
        assert shard.simulator_states is not None
        assert shard.current_simulator_state is not None
        assert shard.expert_future_simulator_state is not None
        assert shard.simulator_scale is not None
        assert shard.simulator_weights is not None
        if shard.simulator_states.ndim != 3:
            raise ValueError("simulator_states must have rank three")
        simulator_width = shard.simulator_states.shape[-1]
        if shard.simulator_states.shape != (batch, candidates, simulator_width):
            raise ValueError("simulator_states must be [batch, candidates, simulator_width]")
        if shard.current_simulator_state.shape != (batch, simulator_width):
            raise ValueError("current_simulator_state width mismatch")
        if shard.expert_future_simulator_state.shape != (batch, simulator_width):
            raise ValueError("expert_future_simulator_state width mismatch")
        if shard.simulator_scale.shape != (simulator_width,):
            raise ValueError("simulator_scale must be [simulator_width]")
        if shard.simulator_weights.shape != (simulator_width,):
            raise ValueError("simulator_weights must be [simulator_width]")
        if bool((shard.simulator_scale <= 0).any()):
            raise ValueError("simulator_scale must be positive")
        if bool((shard.simulator_weights < 0).any()) or not bool(
            (shard.simulator_weights > 0).any()
        ):
            raise ValueError("simulator_weights must be non-negative/non-empty")
    if len(shard.source_keys) != batch or len(set(shard.source_keys)) != batch:
        raise ValueError("source_keys must be unique and match batch size")
    if not bool(shard.candidate_mask.any(dim=1).all()):
        raise ValueError("every row must contain a valid candidate")
    if not bool(((shard.preferred_index >= 0) & (shard.preferred_index < candidates)).all()):
        raise ValueError("preferred_index is outside candidate range")
    floating = (
        shard.context,
        shard.state,
        shard.candidate_actions,
        shard.future_states,
        shard.progress,
        shard.candidate_error,
        shard.current_error,
        shard.baseline_error,
        *(value for value in raw if value is not None),
    )
    if any(not bool(torch.isfinite(value.float()).all()) for value in floating):
        raise ValueError("physical outcome shard contains non-finite values")
    return batch, candidates, horizon, action_dim


def save_physical_outcome_shard(path: str | Path, shard: PhysicalOutcomeShard) -> None:
    _validate_shard(shard)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    torch.save(
        {
            "context": shard.context.cpu(),
            "state": shard.state.cpu(),
            "candidate_actions": shard.candidate_actions.cpu(),
            "candidate_mask": shard.candidate_mask.cpu(),
            "future_states": shard.future_states.cpu(),
            "progress": shard.progress.cpu(),
            "failure": shard.failure.cpu(),
            "preferred_index": shard.preferred_index.cpu(),
            "candidate_error": shard.candidate_error.cpu(),
            "current_error": shard.current_error.cpu(),
            "baseline_error": shard.baseline_error.cpu(),
            "source_keys": shard.source_keys,
            "simulator_states": shard.simulator_states.cpu()
            if shard.simulator_states is not None
            else None,
            "current_simulator_state": shard.current_simulator_state.cpu()
            if shard.current_simulator_state is not None
            else None,
            "expert_future_simulator_state": shard.expert_future_simulator_state.cpu()
            if shard.expert_future_simulator_state is not None
            else None,
            "simulator_scale": shard.simulator_scale.cpu()
            if shard.simulator_scale is not None
            else None,
            "simulator_weights": shard.simulator_weights.cpu()
            if shard.simulator_weights is not None
            else None,
        },
        temporary,
    )
    temporary.replace(destination)


def load_physical_outcome_shard(path: str | Path) -> PhysicalOutcomeShard:
    payload = torch.load(Path(path), map_location="cpu", weights_only=True)
    shard = PhysicalOutcomeShard(
        context=payload["context"],
        state=payload["state"],
        candidate_actions=payload["candidate_actions"],
        candidate_mask=payload["candidate_mask"],
        future_states=payload["future_states"],
        progress=payload["progress"],
        failure=payload["failure"],
        preferred_index=payload["preferred_index"],
        candidate_error=payload["candidate_error"],
        current_error=payload["current_error"],
        baseline_error=payload["baseline_error"],
        source_keys=tuple(str(value) for value in payload["source_keys"]),
        simulator_states=payload.get("simulator_states"),
        current_simulator_state=payload.get("current_simulator_state"),
        expert_future_simulator_state=payload.get("expert_future_simulator_state"),
        simulator_scale=payload.get("simulator_scale"),
        simulator_weights=payload.get("simulator_weights"),
    )
    _validate_shard(shard)
    return shard


def list_physical_outcome_shards(directory: str | Path) -> tuple[Path, ...]:
    return tuple(sorted(Path(directory).glob("shard-*.pt")))


def physical_outcome_episode_key(source_key: str) -> str:
    """Return the demonstration identity encoded by a branch-cache source key."""

    parts = source_key.split("/")
    if len(parts) != 4 or not all(parts[:3]):
        raise ValueError(
            "physical outcome source key must be suite/file/demo/frame"
        )
    return "/".join(parts[:3])


def physical_outcome_episode_fold(source_key: str, *, folds: int) -> int:
    """Assign a whole demonstration to a stable, process-independent fold."""

    if folds < 2:
        raise ValueError("folds must be at least two")
    episode = physical_outcome_episode_key(source_key)
    digest = hashlib.sha256(episode.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False) % folds


def physical_outcome_source_inventory(
    paths: Sequence[str | Path],
) -> tuple[frozenset[str], frozenset[str], int]:
    """Audit globally unique rows and collect their isolated demo identities."""

    source_keys: set[str] = set()
    episode_keys: set[str] = set()
    rows = 0
    for path in paths:
        shard = load_physical_outcome_shard(path)
        overlap = source_keys.intersection(shard.source_keys)
        if overlap:
            example = min(overlap)
            raise ValueError(f"duplicate physical outcome source key: {example}")
        source_keys.update(shard.source_keys)
        episode_keys.update(physical_outcome_episode_key(key) for key in shard.source_keys)
        rows += len(shard.source_keys)
    return frozenset(source_keys), frozenset(episode_keys), rows


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


def validate_physical_outcome_manifest(payload: Mapping[str, object]) -> PhysicalOutcomeManifest:
    """Reject benchmark leakage before a physical-outcome cache enters training."""

    schema_version = _positive_int(payload, "schema_version")
    if schema_version not in {2, 3}:
        raise ContractError("physical outcome cache requires schema_version=2 or 3")
    source_role = _string(payload, "source_role")
    if source_role not in {"train", "validation"}:
        raise ContractError("source_role must be train or validation")
    candidate_count = _positive_int(payload, "candidate_count")
    flow_sample_count = _positive_int(payload, "flow_sample_count")
    if candidate_count != flow_sample_count + 1:
        raise ContractError("candidate_count must equal flow_sample_count plus median baseline")
    baseline_candidate_index = payload.get("baseline_candidate_index")
    if not isinstance(baseline_candidate_index, int) or isinstance(baseline_candidate_index, bool):
        raise ContractError("baseline_candidate_index must be an integer")
    if baseline_candidate_index != candidate_count - 1:
        raise ContractError("median baseline must be the final candidate")
    action_horizon = _positive_int(payload, "action_horizon")
    executed_action_steps = _positive_int(payload, "executed_action_steps")
    if executed_action_steps > action_horizon:
        raise ContractError("executed_action_steps cannot exceed action_horizon")
    context_dim = _positive_int(payload, "context_dim")
    context_tokens = _positive_int(payload, "context_tokens")
    context_tokens_per_camera = _positive_int(payload, "context_tokens_per_camera")
    if context_tokens != 2 * context_tokens_per_camera:
        raise ContractError("context_tokens must equal two ordered camera streams")
    state_dim = _positive_int(payload, "state_dim")
    action_dim = _positive_int(payload, "action_dim")
    if state_dim != STATE_DIM or action_dim != ACTION_DIM:
        raise ContractError("state/action dimensions violate the Duvla contract")
    recorded_fields = frozenset(_strings(payload, "recorded_fields"))
    required_fields = (
        PHYSICAL_OUTCOME_FIELDS if schema_version == 3 else LEGACY_PHYSICAL_OUTCOME_FIELDS
    )
    missing = required_fields.difference(recorded_fields)
    if missing:
        raise ContractError("physical outcome cache is missing fields: " + ", ".join(sorted(missing)))
    forbidden = FORBIDDEN_PHYSICAL_OUTCOME_FIELDS.intersection(recorded_fields)
    if forbidden:
        raise ContractError(
            "physical outcome cache contains forbidden supervision fields: "
            + ", ".join(sorted(forbidden))
        )
    uses_training_demonstration_states = _boolean(
        payload, "uses_training_demonstration_states"
    )
    uses_benchmark_reward = _boolean(payload, "uses_benchmark_reward")
    uses_benchmark_task_index = _boolean(payload, "uses_benchmark_task_index")
    uses_evaluation_initial_states = _boolean(payload, "uses_evaluation_initial_states")
    if not uses_training_demonstration_states:
        raise ContractError("physical outcome cache must be rooted in training demonstrations")
    if uses_benchmark_reward or uses_benchmark_task_index or uses_evaluation_initial_states:
        raise ContractError("physical outcome cache contains forbidden benchmark/evaluation supervision")
    target_definition = _string(payload, "target_definition")
    if "reward" in target_definition.casefold() or "success" in target_definition.casefold():
        raise ContractError("target_definition must not use reward or success labels")
    candidate_generator_sha256 = _string(payload, "candidate_generator_sha256")
    if len(candidate_generator_sha256) != 64 or any(
        character not in "0123456789abcdef" for character in candidate_generator_sha256.casefold()
    ):
        raise ContractError("candidate_generator_sha256 must be a SHA-256 hex digest")
    return PhysicalOutcomeManifest(
        schema_version=schema_version,
        source_role=source_role,
        candidate_count=candidate_count,
        flow_sample_count=flow_sample_count,
        baseline_candidate_index=baseline_candidate_index,
        action_horizon=action_horizon,
        executed_action_steps=executed_action_steps,
        context_dim=context_dim,
        context_tokens=context_tokens,
        context_tokens_per_camera=context_tokens_per_camera,
        state_dim=state_dim,
        action_dim=action_dim,
        candidate_generator_checkpoint=_string(payload, "candidate_generator_checkpoint"),
        candidate_generator_sha256=candidate_generator_sha256.casefold(),
        source_revision=_string(payload, "source_revision"),
        recorded_fields=recorded_fields,
        uses_training_demonstration_states=uses_training_demonstration_states,
        uses_benchmark_reward=uses_benchmark_reward,
        uses_benchmark_task_index=uses_benchmark_task_index,
        uses_evaluation_initial_states=uses_evaluation_initial_states,
    )


def write_json_atomic(path: str | Path, payload: Mapping[str, object]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(destination)
