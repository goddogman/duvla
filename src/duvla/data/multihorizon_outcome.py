"""Compact multi-horizon physical outcomes joined to an existing branch cache."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor

from .contracts import STATE_DIM

PHYSICAL_OBJECT_SLOTS = 2
PHYSICAL_ARTICULATION_SLOTS = 2
PHYSICAL_FEATURE_DIM = (
    STATE_DIM + PHYSICAL_OBJECT_SLOTS * 7 + PHYSICAL_ARTICULATION_SLOTS
)


@dataclass(frozen=True)
class PhysicalDofSlots:
    """Task-local MuJoCo DOFs selected from training demonstrations only."""

    object_qpos_addresses: tuple[int, ...]
    articulation_qpos_addresses: tuple[int, ...]


def select_salient_physical_dofs(
    state_sequences: tuple[Tensor, ...],
    *,
    joint_names: tuple[str, ...],
    joint_types: tuple[int, ...],
    joint_qpos_addresses: tuple[int, ...],
    nq: int,
    object_motion_threshold: float = 0.02,
    articulation_motion_threshold: float = 0.02,
) -> PhysicalDofSlots:
    """Select moving non-robot objects/joints without using a task id or reward.

    MuJoCo flattened states start with simulation time followed by qpos.  Free
    joints use seven qpos values (xyz + quaternion); slide/hinge joints use one.
    Selection is based only on motion observed in official training demos.
    """

    if not state_sequences or nq <= 0:
        raise ValueError("state sequences and nq must be non-empty/positive")
    if object_motion_threshold < 0.0 or articulation_motion_threshold < 0.0:
        raise ValueError("physical motion thresholds must be non-negative")
    count = len(joint_names)
    if not (
        len(joint_types) == count == len(joint_qpos_addresses)
    ):
        raise ValueError("joint metadata lengths differ")
    sequences = tuple(value.float() for value in state_sequences)
    if any(value.ndim != 2 or value.shape[0] < 2 for value in sequences):
        raise ValueError("each state sequence must contain at least two rows")
    if any(value.shape[1] < 1 + nq for value in sequences):
        raise ValueError("flattened state is shorter than time plus qpos")

    object_scores: list[tuple[float, str, int]] = []
    articulation_scores: list[tuple[float, str, int]] = []
    for name, joint_type, address in zip(
        joint_names, joint_types, joint_qpos_addresses, strict=True
    ):
        if name.startswith(("robot0_", "gripper0_")):
            continue
        if not 0 <= address < nq:
            raise ValueError("joint qpos address is outside nq")
        flat_address = 1 + address
        if joint_type == 0:
            if address + 7 > nq:
                raise ValueError("free joint exceeds qpos width")
            score = max(
                float(
                    (sequence[:, flat_address : flat_address + 3] - sequence[0, flat_address : flat_address + 3])
                    .norm(dim=-1)
                    .max()
                )
                for sequence in sequences
            )
            if score >= object_motion_threshold:
                object_scores.append((score, name, address))
        else:
            score = max(
                float(
                    (sequence[:, flat_address] - sequence[0, flat_address])
                    .abs()
                    .max()
                )
                for sequence in sequences
            )
            if score >= articulation_motion_threshold:
                articulation_scores.append((score, name, address))
    object_scores.sort(key=lambda item: (-item[0], item[1]))
    articulation_scores.sort(key=lambda item: (-item[0], item[1]))
    return PhysicalDofSlots(
        object_qpos_addresses=tuple(
            item[2] for item in object_scores[:PHYSICAL_OBJECT_SLOTS]
        ),
        articulation_qpos_addresses=tuple(
            item[2] for item in articulation_scores[:PHYSICAL_ARTICULATION_SLOTS]
        ),
    )


def physical_outcome_features(
    simulator_states: Tensor,
    robot_states: Tensor,
    *,
    current_simulator_state: Tensor,
    task_scale: Tensor,
    slots: PhysicalDofSlots,
) -> tuple[Tensor, Tensor]:
    """Build a fixed 24D robot/object/articulation outcome descriptor.

    Object and articulation values are changes relative to the current state;
    robot state remains the normalized absolute 8D deployment observation.
    Missing slots are zero padded and excluded by the returned mask.
    """

    if simulator_states.shape[:-1] != robot_states.shape[:-1]:
        raise ValueError("simulator and robot state prefix shapes differ")
    if robot_states.shape[-1] != STATE_DIM:
        raise ValueError(f"robot state must end in {STATE_DIM} values")
    width = simulator_states.shape[-1]
    if current_simulator_state.shape != (width,) or task_scale.shape != (width,):
        raise ValueError("current simulator state and scale must match state width")
    if not bool(torch.isfinite(simulator_states.float()).all()) or not bool(
        torch.isfinite(robot_states.float()).all()
    ):
        raise ValueError("physical outcome inputs must be finite")
    values = [robot_states.float()]
    mask_values = [torch.ones(STATE_DIM, dtype=torch.bool, device=robot_states.device)]
    current = current_simulator_state.to(simulator_states)
    scale = task_scale.to(simulator_states).clamp_min(1e-2)
    prefix = simulator_states.shape[:-1]
    for slot_index in range(PHYSICAL_OBJECT_SLOTS):
        if slot_index >= len(slots.object_qpos_addresses):
            values.append(torch.zeros(*prefix, 7, dtype=torch.float32, device=robot_states.device))
            mask_values.append(torch.zeros(7, dtype=torch.bool, device=robot_states.device))
            continue
        qpos = 1 + slots.object_qpos_addresses[slot_index]
        translation_scale = scale[qpos : qpos + 3].clamp_min(5e-2)
        translation = (
            simulator_states[..., qpos : qpos + 3] - current[qpos : qpos + 3]
        ) / translation_scale
        current_quaternion = current[qpos + 3 : qpos + 7]
        quaternion = simulator_states[..., qpos + 3 : qpos + 7]
        sign = torch.where(
            (quaternion * current_quaternion).sum(dim=-1, keepdim=True) < 0,
            -torch.ones((), dtype=quaternion.dtype, device=quaternion.device),
            torch.ones((), dtype=quaternion.dtype, device=quaternion.device),
        )
        quaternion_delta = quaternion * sign - current_quaternion
        values.append(torch.cat((translation.float(), quaternion_delta.float()), dim=-1))
        mask_values.append(torch.ones(7, dtype=torch.bool, device=robot_states.device))
    for slot_index in range(PHYSICAL_ARTICULATION_SLOTS):
        if slot_index >= len(slots.articulation_qpos_addresses):
            values.append(torch.zeros(*prefix, 1, dtype=torch.float32, device=robot_states.device))
            mask_values.append(torch.zeros(1, dtype=torch.bool, device=robot_states.device))
            continue
        qpos = 1 + slots.articulation_qpos_addresses[slot_index]
        delta = (simulator_states[..., qpos] - current[qpos]) / scale[qpos]
        values.append(delta.float().unsqueeze(-1))
        mask_values.append(torch.ones(1, dtype=torch.bool, device=robot_states.device))
    features = torch.cat(values, dim=-1)
    mask = torch.cat(mask_values)
    if features.shape[-1] != PHYSICAL_FEATURE_DIM:
        raise RuntimeError("physical feature width invariant failed")
    return features, mask


@dataclass(frozen=True)
class MultiHorizonOutcomeShard:
    source_keys: tuple[str, ...]
    horizons: tuple[int, ...]
    future_states: Tensor
    horizon_mask: Tensor
    progress: Tensor
    failure: Tensor
    preferred_index: Tensor
    candidate_error: Tensor
    current_error: Tensor
    baseline_error: Tensor
    simulator_states: Tensor
    expert_future_states: Tensor
    expert_future_simulator_states: Tensor
    physical_features: Tensor
    expert_physical_features: Tensor
    physical_feature_mask: Tensor


def validate_multihorizon_outcome_shard(
    shard: MultiHorizonOutcomeShard,
) -> tuple[int, int, int, int]:
    if shard.future_states.ndim != 4:
        raise ValueError("future_states must be [batch, candidates, horizons, state_dim]")
    batch, candidates, horizon_count, state_dim = shard.future_states.shape
    if state_dim != STATE_DIM:
        raise ValueError(f"future_states must use the {STATE_DIM}D robot state")
    if len(shard.source_keys) != batch or len(set(shard.source_keys)) != batch:
        raise ValueError("source_keys must be unique and match batch size")
    if len(shard.horizons) != horizon_count:
        raise ValueError("horizons do not match the tensor horizon axis")
    if tuple(sorted(shard.horizons)) != shard.horizons or any(
        value <= 0 for value in shard.horizons
    ):
        raise ValueError("horizons must be positive and strictly ordered")
    if shard.horizon_mask.shape != (batch, horizon_count):
        raise ValueError("horizon_mask must be [batch, horizons]")
    if shard.horizon_mask.dtype != torch.bool:
        raise ValueError("horizon_mask must be boolean")
    matrix = (batch, candidates, horizon_count)
    for name, value in (
        ("progress", shard.progress),
        ("candidate_error", shard.candidate_error),
    ):
        if value.shape != matrix:
            raise ValueError(f"{name} must be [batch, candidates, horizons]")
    if shard.failure.shape != matrix or shard.failure.dtype != torch.bool:
        raise ValueError("failure must be boolean [batch, candidates, horizons]")
    if shard.preferred_index.shape != (batch, horizon_count):
        raise ValueError("preferred_index must be [batch, horizons]")
    if shard.preferred_index.dtype != torch.long:
        raise ValueError("preferred_index must be int64")
    if shard.current_error.shape != (batch, horizon_count):
        raise ValueError("current_error must be [batch, horizons]")
    if shard.baseline_error.shape != (batch, horizon_count):
        raise ValueError("baseline_error must be [batch, horizons]")
    if shard.simulator_states.ndim != 4:
        raise ValueError("simulator_states must be [batch, candidates, horizons, width]")
    width = shard.simulator_states.shape[-1]
    if shard.simulator_states.shape != (batch, candidates, horizon_count, width):
        raise ValueError("simulator_states axes differ from future_states")
    if shard.expert_future_simulator_states.shape != (batch, horizon_count, width):
        raise ValueError("expert future simulator states have an incompatible shape")
    if shard.expert_future_states.shape != (batch, horizon_count, STATE_DIM):
        raise ValueError("expert future robot states have an incompatible shape")
    if shard.physical_features.shape != (
        batch,
        candidates,
        horizon_count,
        PHYSICAL_FEATURE_DIM,
    ):
        raise ValueError("physical_features have an incompatible shape")
    if shard.expert_physical_features.shape != (
        batch,
        horizon_count,
        PHYSICAL_FEATURE_DIM,
    ):
        raise ValueError("expert physical features have an incompatible shape")
    if shard.physical_feature_mask.shape != (batch, PHYSICAL_FEATURE_DIM):
        raise ValueError("physical feature mask has an incompatible shape")
    if shard.physical_feature_mask.dtype != torch.bool:
        raise ValueError("physical feature mask must be boolean")
    valid = shard.horizon_mask[:, None].expand(-1, candidates, -1)
    if bool(valid.any()):
        floating = (
            shard.future_states,
            shard.progress,
            shard.candidate_error,
            shard.current_error,
            shard.baseline_error,
            shard.simulator_states,
            shard.expert_future_states,
            shard.expert_future_simulator_states,
            shard.physical_features,
            shard.expert_physical_features,
        )
        if any(not bool(torch.isfinite(value.float()).all()) for value in floating):
            raise ValueError("multi-horizon outcome contains non-finite values")
    if not bool(((shard.preferred_index >= 0) & (shard.preferred_index < candidates)).all()):
        raise ValueError("preferred_index is outside candidate range")
    return batch, candidates, horizon_count, width


def save_multihorizon_outcome_shard(
    path: str | Path, shard: MultiHorizonOutcomeShard
) -> None:
    validate_multihorizon_outcome_shard(shard)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    torch.save(
        {
            "format": "duvla_v3_1_multihorizon_outcome_shard_v3",
            "source_keys": shard.source_keys,
            "horizons": shard.horizons,
            "future_states": shard.future_states.cpu(),
            "horizon_mask": shard.horizon_mask.cpu(),
            "progress": shard.progress.cpu(),
            "failure": shard.failure.cpu(),
            "preferred_index": shard.preferred_index.cpu(),
            "candidate_error": shard.candidate_error.cpu(),
            "current_error": shard.current_error.cpu(),
            "baseline_error": shard.baseline_error.cpu(),
            "simulator_states": shard.simulator_states.cpu(),
            "expert_future_states": shard.expert_future_states.cpu(),
            "expert_future_simulator_states": shard.expert_future_simulator_states.cpu(),
            "physical_features": shard.physical_features.cpu(),
            "expert_physical_features": shard.expert_physical_features.cpu(),
            "physical_feature_mask": shard.physical_feature_mask.cpu(),
        },
        temporary,
    )
    temporary.replace(destination)


def load_multihorizon_outcome_shard(
    path: str | Path, *, mmap: bool = False
) -> MultiHorizonOutcomeShard:
    payload = torch.load(
        Path(path), map_location="cpu", weights_only=True, mmap=mmap
    )
    if payload.get("format") != "duvla_v3_1_multihorizon_outcome_shard_v3":
        raise ValueError(f"unsupported multi-horizon outcome shard: {path}")
    shard = MultiHorizonOutcomeShard(
        source_keys=tuple(str(value) for value in payload["source_keys"]),
        horizons=tuple(int(value) for value in payload["horizons"]),
        future_states=payload["future_states"],
        horizon_mask=payload["horizon_mask"],
        progress=payload["progress"],
        failure=payload["failure"],
        preferred_index=payload["preferred_index"],
        candidate_error=payload["candidate_error"],
        current_error=payload["current_error"],
        baseline_error=payload["baseline_error"],
        simulator_states=payload["simulator_states"],
        expert_future_states=payload["expert_future_states"],
        expert_future_simulator_states=payload["expert_future_simulator_states"],
        physical_features=payload["physical_features"],
        expert_physical_features=payload["expert_physical_features"],
        physical_feature_mask=payload["physical_feature_mask"],
    )
    validate_multihorizon_outcome_shard(shard)
    return shard


def list_multihorizon_outcome_shards(directory: str | Path) -> tuple[Path, ...]:
    return tuple(sorted(Path(directory).glob("shard-*.pt")))


def aggregate_multihorizon_targets(
    shard: MultiHorizonOutcomeShard,
    *,
    weights: tuple[float, ...] = (0.2, 0.3, 0.5),
) -> dict[str, Tensor]:
    """Flatten future states and aggregate physical labels over valid horizons."""

    batch, candidates, horizon_count, _width = validate_multihorizon_outcome_shard(
        shard
    )
    if len(weights) != horizon_count or any(value < 0.0 for value in weights):
        raise ValueError("weights must be non-negative and match horizons")
    raw_weights = shard.future_states.new_tensor(weights)[None]
    valid_weights = raw_weights * shard.horizon_mask.to(raw_weights.dtype)
    denominator = valid_weights.sum(dim=1, keepdim=True)
    if bool((denominator <= 0.0).any()):
        raise ValueError("every row must contain at least one valid horizon")
    normalized = valid_weights / denominator
    candidate_weights = normalized[:, None]
    aggregate_error = (shard.candidate_error * candidate_weights).sum(dim=-1)
    feature_mask = shard.physical_feature_mask[:, None, None, :]
    future_mask = shard.horizon_mask[:, None, :, None].expand(
        batch, candidates, horizon_count, PHYSICAL_FEATURE_DIM
    ) & feature_mask.expand(batch, candidates, horizon_count, -1)
    plan_mask = shard.horizon_mask[:, :, None].expand(
        batch, horizon_count, PHYSICAL_FEATURE_DIM
    ) & shard.physical_feature_mask[:, None, :]
    return {
        "future_target": shard.physical_features.flatten(2, 3),
        "future_mask": future_mask.flatten(2, 3),
        "planned_future_target": shard.expert_physical_features.flatten(1, 2),
        "planned_future_mask": plan_mask.flatten(1, 2),
        "progress_target": (shard.progress * candidate_weights).sum(dim=-1),
        "failure_target": (
            shard.failure.to(dtype=shard.progress.dtype) * candidate_weights
        ).sum(dim=-1),
        "candidate_error": aggregate_error,
        "current_error": (shard.current_error * normalized).sum(dim=-1),
        "baseline_error": (shard.baseline_error * normalized).sum(dim=-1),
        "preferred_index": aggregate_error.argmin(dim=1),
    }
