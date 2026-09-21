"""Leakage-resistant cache contract for V3.25 recovery supervision."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import torch
from torch import Tensor

from .contracts import ACTION_DIM, STATE_DIM, ContractError


RECOVERY_CACHE_FIELDS = frozenset(
    {
        "source_keys", "label_sources", "context", "state", "base_actions",
        "target_actions", "action_mask", "gate_target", "positive",
        "position_error_before", "position_error_after", "physical_error_before",
        "physical_error_after", "object_error_before", "object_error_after",
    }
)

FORBIDDEN_RECOVERY_FIELDS = frozenset(
    {
        "benchmark_reward", "benchmark_success", "evaluation_initial_state",
        "reward", "success", "suite_task_id", "task_index",
    }
)


@dataclass(frozen=True)
class RecoveryCacheManifest:
    schema_version: int
    source_role: str
    rows: int
    positive_rows: int
    negative_rows: int
    context_tokens: int
    context_dim: int
    state_dim: int
    action_horizon: int
    action_dim: int
    parent_checkpoint: str
    parent_checkpoint_sha256: str
    qwen_hidden_layers: tuple[int, ...]
    qwen_context_mode: str
    recorded_fields: frozenset[str]


@dataclass(frozen=True)
class RecoveryCacheShard:
    context: Tensor
    state: Tensor
    base_actions: Tensor
    target_actions: Tensor
    action_mask: Tensor
    gate_target: Tensor
    positive: Tensor
    source_keys: tuple[str, ...]
    label_sources: tuple[str, ...]
    position_error_before: Tensor
    position_error_after: Tensor
    physical_error_before: Tensor
    physical_error_after: Tensor
    object_error_before: Tensor
    object_error_after: Tensor


def _validate_shard(shard: RecoveryCacheShard) -> tuple[int, int, int, int]:
    if shard.context.ndim != 3:
        raise ValueError("context must have shape [batch,tokens,width]")
    batch, tokens, width = shard.context.shape
    if batch <= 0 or min(tokens, width) <= 0:
        raise ValueError("recovery shard dimensions must be positive")
    if shard.state.shape != (batch, STATE_DIM):
        raise ValueError(f"state must have shape [batch,{STATE_DIM}]")
    if shard.base_actions.ndim != 3:
        raise ValueError("base_actions must have shape [batch,horizon,action_dim]")
    if shard.target_actions.shape != shard.base_actions.shape:
        raise ValueError("target_actions must match base_actions")
    action_batch, horizon, action_dim = shard.base_actions.shape
    if action_batch != batch or action_dim != ACTION_DIM:
        raise ValueError("action tensors violate the Duvla contract")
    matrix = (batch, horizon)
    if shard.action_mask.shape != matrix or shard.action_mask.dtype != torch.bool:
        raise ValueError("action_mask must be boolean [batch,horizon]")
    if shard.gate_target.shape != matrix or shard.gate_target.dtype != torch.bool:
        raise ValueError("gate_target must be boolean [batch,horizon]")
    if shard.positive.shape != (batch,) or shard.positive.dtype != torch.bool:
        raise ValueError("positive must be boolean [batch]")
    if not bool(shard.action_mask.any(dim=1).all()):
        raise ValueError("every recovery row must supervise at least one action")
    if bool((shard.gate_target & ~shard.action_mask).any()):
        raise ValueError("gate targets cannot be positive outside the action mask")
    if not torch.equal(shard.gate_target.any(dim=1), shard.positive):
        raise ValueError("positive rows must agree with gate_target")
    if len(shard.source_keys) != batch or len(set(shard.source_keys)) != batch:
        raise ValueError("source_keys must be unique within a shard")
    if len(shard.label_sources) != batch:
        raise ValueError("label_sources must match the batch")
    allowed_sources = {"cartesian_servo", "demo_replay", "identity"}
    if any(value not in allowed_sources for value in shard.label_sources):
        raise ValueError("label_sources contain an unsupported recovery target")
    for index, label in enumerate(shard.label_sources):
        if (label != "identity") != bool(shard.positive[index]):
            raise ValueError("identity labels must be negative and corrections positive")
    metrics = (
        shard.position_error_before, shard.position_error_after,
        shard.physical_error_before, shard.physical_error_after,
        shard.object_error_before, shard.object_error_after,
    )
    if any(value.shape != (batch,) for value in metrics):
        raise ValueError("recovery metrics must have shape [batch]")
    floating = (
        shard.context, shard.state, shard.base_actions, shard.target_actions, *metrics,
    )
    if any(not bool(torch.isfinite(value.float()).all()) for value in floating):
        raise ValueError("recovery shard contains non-finite values")
    return batch, tokens, width, horizon


def save_recovery_cache_shard(path: str | Path, shard: RecoveryCacheShard) -> None:
    _validate_shard(shard)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    torch.save(
        {
            "context": shard.context.cpu(),
            "state": shard.state.cpu(),
            "base_actions": shard.base_actions.cpu(),
            "target_actions": shard.target_actions.cpu(),
            "action_mask": shard.action_mask.cpu(),
            "gate_target": shard.gate_target.cpu(),
            "positive": shard.positive.cpu(),
            "source_keys": shard.source_keys,
            "label_sources": shard.label_sources,
            "position_error_before": shard.position_error_before.cpu(),
            "position_error_after": shard.position_error_after.cpu(),
            "physical_error_before": shard.physical_error_before.cpu(),
            "physical_error_after": shard.physical_error_after.cpu(),
            "object_error_before": shard.object_error_before.cpu(),
            "object_error_after": shard.object_error_after.cpu(),
        },
        temporary,
    )
    temporary.replace(destination)


def load_recovery_cache_shard(
    path: str | Path, *, mmap: bool = False
) -> RecoveryCacheShard:
    payload = torch.load(
        Path(path), map_location="cpu", weights_only=True, mmap=mmap
    )
    shard = RecoveryCacheShard(
        context=payload["context"], state=payload["state"],
        base_actions=payload["base_actions"], target_actions=payload["target_actions"],
        action_mask=payload["action_mask"], gate_target=payload["gate_target"],
        positive=payload["positive"],
        source_keys=tuple(str(value) for value in payload["source_keys"]),
        label_sources=tuple(str(value) for value in payload["label_sources"]),
        position_error_before=payload["position_error_before"],
        position_error_after=payload["position_error_after"],
        physical_error_before=payload["physical_error_before"],
        physical_error_after=payload["physical_error_after"],
        object_error_before=payload["object_error_before"],
        object_error_after=payload["object_error_after"],
    )
    _validate_shard(shard)
    return shard


def list_recovery_cache_shards(directory: str | Path) -> tuple[Path, ...]:
    return tuple(sorted(Path(directory).glob("shard-*.pt")))


def _integer(payload: Mapping[str, object], name: str, *, minimum: int = 1) -> int:
    value = payload.get(name)
    if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
        raise ContractError(f"{name} must be an integer >= {minimum}")
    return value


def _string(payload: Mapping[str, object], name: str) -> str:
    value = payload.get(name)
    if not isinstance(value, str) or not value:
        raise ContractError(f"{name} must be a non-empty string")
    return value


def _boolean(payload: Mapping[str, object], name: str) -> bool:
    value = payload.get(name)
    if not isinstance(value, bool):
        raise ContractError(f"{name} must be boolean")
    return value


def _sequence(payload: Mapping[str, object], name: str) -> Sequence[object]:
    value = payload.get(name)
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ContractError(f"{name} must be a sequence")
    return value


def validate_recovery_cache_manifest(payload: Mapping[str, object]) -> RecoveryCacheManifest:
    if _string(payload, "format") != "duvla_v3_25_recovery_cache":
        raise ContractError("unexpected recovery cache format")
    schema_version = _integer(payload, "schema_version")
    if schema_version != 1:
        raise ContractError("recovery cache requires schema_version=1")
    source_role = _string(payload, "source_role")
    if source_role not in {"train", "validation"}:
        raise ContractError("source_role must be train or validation")
    rows = _integer(payload, "rows")
    positive_rows = _integer(payload, "positive_rows", minimum=0)
    negative_rows = _integer(payload, "negative_rows", minimum=0)
    if positive_rows <= 0 or negative_rows <= 0 or positive_rows + negative_rows != rows:
        raise ContractError("recovery cache must contain paired positive and negative rows")
    state_dim = _integer(payload, "state_dim")
    action_dim = _integer(payload, "action_dim")
    if state_dim != STATE_DIM or action_dim != ACTION_DIM:
        raise ContractError("state/action dimensions violate the Duvla contract")
    sha = _string(payload, "parent_checkpoint_sha256").casefold()
    if len(sha) != 64 or any(character not in "0123456789abcdef" for character in sha):
        raise ContractError("parent_checkpoint_sha256 must be a SHA-256 digest")
    layers = tuple(int(value) for value in _sequence(payload, "qwen_hidden_layers"))
    if layers != (12, 14, 18, -1):
        raise ContractError("V3.25 requires Qwen layers 12/14/18/final")
    recorded = frozenset(str(value) for value in _sequence(payload, "recorded_fields"))
    missing = RECOVERY_CACHE_FIELDS.difference(recorded)
    if missing:
        raise ContractError("recovery cache is missing fields: " + ", ".join(sorted(missing)))
    forbidden = FORBIDDEN_RECOVERY_FIELDS.intersection(recorded)
    if forbidden:
        raise ContractError(
            "recovery cache contains forbidden fields: " + ", ".join(sorted(forbidden))
        )
    if not _boolean(payload, "uses_training_demonstration_states"):
        raise ContractError("recovery cache must use training demonstration states")
    for flag in (
        "uses_benchmark_reward", "uses_success",
        "uses_benchmark_task_index_as_model_input", "uses_evaluation_initial_states",
    ):
        if _boolean(payload, flag):
            raise ContractError(f"{flag} must be false")
    return RecoveryCacheManifest(
        schema_version=schema_version, source_role=source_role, rows=rows,
        positive_rows=positive_rows, negative_rows=negative_rows,
        context_tokens=_integer(payload, "context_tokens"),
        context_dim=_integer(payload, "context_dim"), state_dim=state_dim,
        action_horizon=_integer(payload, "action_horizon"), action_dim=action_dim,
        parent_checkpoint=_string(payload, "parent_checkpoint"),
        parent_checkpoint_sha256=sha,
        qwen_hidden_layers=layers,
        qwen_context_mode=_string(payload, "qwen_context_mode"),
        recorded_fields=recorded,
    )
