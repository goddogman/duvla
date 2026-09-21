"""Task-relational physical outcomes for Duvla V3.1-P3c.

The representation is built only from training-demonstration simulator states.
It describes robot/object/goal relations without exposing a benchmark task id,
reward, success predicate, or evaluation initial state to the policy.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor

from .contracts import STATE_DIM
from .multihorizon_outcome import (
    PHYSICAL_ARTICULATION_SLOTS,
    PHYSICAL_OBJECT_SLOTS,
    PhysicalDofSlots,
)


RELATIONAL_OBJECT_FEATURE_DIM = 16
RELATIONAL_ARTICULATION_FEATURE_DIM = 2
RELATIONAL_FEATURE_DIM = (
    STATE_DIM
    + PHYSICAL_OBJECT_SLOTS * RELATIONAL_OBJECT_FEATURE_DIM
    + PHYSICAL_ARTICULATION_SLOTS * RELATIONAL_ARTICULATION_FEATURE_DIM
)


def relational_source_demo(source_key: str) -> tuple[str, str, str]:
    """Return the suite/file/demo identity encoded in one audited source key."""

    raw = source_key.split("#", 1)[0]
    values = raw.split("/")
    if len(values) != 4:
        raise ValueError(f"invalid relational source key: {source_key}")
    int(values[3])
    return values[0], values[1], values[2]


def validate_relational_source_alignment(
    *,
    base_source_keys: tuple[str, ...],
    outcome_source_keys: tuple[str, ...],
    source_role: str,
    train_demo_inventory: dict[tuple[str, str], set[str]],
) -> None:
    """Reject row drift and detectable train/validation episode crossover."""

    if source_role not in {"train", "validation"}:
        raise ValueError(f"unsupported source role: {source_role}")
    if base_source_keys != outcome_source_keys:
        raise ValueError("base and multi-horizon source keys are not row-aligned")
    for source_key in outcome_source_keys:
        suite, filename, demo = relational_source_demo(source_key)
        in_train_inventory = demo in train_demo_inventory.get((suite, filename), set())
        if source_role == "train" and not in_train_inventory:
            raise ValueError(f"train row is absent from train inventory: {source_key}")
        if source_role == "validation" and in_train_inventory:
            raise ValueError(f"validation row overlaps train inventory: {source_key}")


def _signed_quaternion_delta(value: Tensor, reference: Tensor) -> Tensor:
    sign = torch.where(
        (value * reference).sum(dim=-1, keepdim=True) < 0,
        value.new_tensor(-1.0),
        value.new_tensor(1.0),
    )
    return value * sign - reference


def relational_feature_weights() -> Tensor:
    """Return fixed component weights for relation-space endpoint errors."""

    values = [
        # Absolute robot state is retained for dynamics prediction but is not
        # allowed to dominate object/goal relations.
        torch.tensor([0.25] * 3 + [0.10] * 3 + [0.50] * 2),
    ]
    object_weights = torch.tensor(
        [0.50] * 3  # end effector -> object vector
        + [1.00]  # end effector -> object distance
        + [0.50] * 3  # object -> final demonstrated goal vector
        + [1.00]  # object -> goal distance
        + [0.50] * 3  # object/EEF co-motion residual
        + [1.00]  # co-motion residual norm
        + [0.25] * 4  # sign-canonicalized goal quaternion delta
    )
    values.extend(object_weights.clone() for _ in range(PHYSICAL_OBJECT_SLOTS))
    values.extend(torch.ones(RELATIONAL_ARTICULATION_FEATURE_DIM) for _ in range(PHYSICAL_ARTICULATION_SLOTS))
    result = torch.cat(values)
    if result.shape != (RELATIONAL_FEATURE_DIM,):
        raise RuntimeError("relational feature weights have the wrong width")
    return result


def relational_outcome_features(
    simulator_states: Tensor,
    normalized_robot_states: Tensor,
    *,
    current_simulator_state: Tensor,
    current_normalized_robot_state: Tensor,
    goal_simulator_state: Tensor,
    task_scale: Tensor,
    state_mean: Tensor,
    state_std: Tensor,
    slots: PhysicalDofSlots,
) -> tuple[Tensor, Tensor]:
    """Encode EEF–object–goal and articulation relations at any horizon."""

    if simulator_states.shape[:-1] != normalized_robot_states.shape[:-1]:
        raise ValueError("simulator and robot state prefix shapes differ")
    if normalized_robot_states.shape[-1] != STATE_DIM:
        raise ValueError(f"robot state must end in {STATE_DIM} values")
    width = simulator_states.shape[-1]
    if any(value.shape != (width,) for value in (current_simulator_state, goal_simulator_state, task_scale)):
        raise ValueError("current/goal simulator states and scale must match state width")
    if any(value.shape != (STATE_DIM,) for value in (current_normalized_robot_state, state_mean, state_std)):
        raise ValueError("current robot state and normalization must be 8D")
    floating = (
        simulator_states,
        normalized_robot_states,
        current_simulator_state,
        current_normalized_robot_state,
        goal_simulator_state,
        task_scale,
        state_mean,
        state_std,
    )
    if any(not bool(torch.isfinite(value.float()).all()) for value in floating):
        raise ValueError("relational outcome inputs must be finite")
    if bool((state_std <= 0).any()):
        raise ValueError("state_std must be positive")

    device = normalized_robot_states.device
    dtype = normalized_robot_states.dtype
    current = current_simulator_state.to(device=device, dtype=dtype)
    goal = goal_simulator_state.to(device=device, dtype=dtype)
    scale = task_scale.to(device=device, dtype=dtype).clamp_min(1e-2)
    mean = state_mean.to(device=device, dtype=dtype)
    std = state_std.to(device=device, dtype=dtype)
    raw_robot = normalized_robot_states * std + mean
    current_raw_robot = current_normalized_robot_state.to(device=device, dtype=dtype) * std + mean
    current_eef = current_raw_robot[:3]
    eef = raw_robot[..., :3]
    prefix = simulator_states.shape[:-1]

    values: list[Tensor] = [normalized_robot_states.float()]
    masks: list[Tensor] = [torch.ones(STATE_DIM, dtype=torch.bool, device=device)]
    for slot_index in range(PHYSICAL_OBJECT_SLOTS):
        if slot_index >= len(slots.object_qpos_addresses):
            values.append(
                torch.zeros(
                    *prefix,
                    RELATIONAL_OBJECT_FEATURE_DIM,
                    dtype=torch.float32,
                    device=device,
                )
            )
            masks.append(
                torch.zeros(
                    RELATIONAL_OBJECT_FEATURE_DIM, dtype=torch.bool, device=device
                )
            )
            continue
        address = 1 + slots.object_qpos_addresses[slot_index]
        if address + 7 > width:
            raise ValueError("object qpos slot exceeds simulator state width")
        translation_scale = scale[address : address + 3].clamp_min(5e-2)
        object_position = simulator_states[..., address : address + 3].to(dtype=dtype)
        current_object = current[address : address + 3]
        goal_object = goal[address : address + 3]
        eef_to_object = (object_position - eef) / translation_scale
        object_to_goal = (goal_object - object_position) / translation_scale
        co_motion = (
            (object_position - current_object) - (eef - current_eef)
        ) / translation_scale
        quaternion = simulator_states[..., address + 3 : address + 7].to(dtype=dtype)
        goal_quaternion = goal[address + 3 : address + 7]
        quaternion_to_goal = _signed_quaternion_delta(quaternion, goal_quaternion)
        values.append(
            torch.cat(
                (
                    eef_to_object,
                    eef_to_object.norm(dim=-1, keepdim=True),
                    object_to_goal,
                    object_to_goal.norm(dim=-1, keepdim=True),
                    co_motion,
                    co_motion.norm(dim=-1, keepdim=True),
                    quaternion_to_goal,
                ),
                dim=-1,
            ).float()
        )
        masks.append(
            torch.ones(
                RELATIONAL_OBJECT_FEATURE_DIM, dtype=torch.bool, device=device
            )
        )

    for slot_index in range(PHYSICAL_ARTICULATION_SLOTS):
        if slot_index >= len(slots.articulation_qpos_addresses):
            values.append(torch.zeros(*prefix, 2, dtype=torch.float32, device=device))
            masks.append(torch.zeros(2, dtype=torch.bool, device=device))
            continue
        address = 1 + slots.articulation_qpos_addresses[slot_index]
        if address >= width:
            raise ValueError("articulation qpos slot exceeds simulator state width")
        denominator = scale[address]
        position = simulator_states[..., address].to(dtype=dtype)
        values.append(
            torch.stack(
                (
                    (position - current[address]) / denominator,
                    (goal[address] - position) / denominator,
                ),
                dim=-1,
            ).float()
        )
        masks.append(torch.ones(2, dtype=torch.bool, device=device))

    features = torch.cat(values, dim=-1)
    mask = torch.cat(masks)
    if features.shape[-1] != RELATIONAL_FEATURE_DIM:
        raise RuntimeError("relational feature width invariant failed")
    return features, mask


@dataclass(frozen=True)
class RelationalOutcomeShard:
    source_keys: tuple[str, ...]
    horizons: tuple[int, ...]
    horizon_mask: Tensor
    future_features: Tensor
    expert_future_features: Tensor
    feature_mask: Tensor
    progress: Tensor
    failure: Tensor
    preferred_index: Tensor
    candidate_error: Tensor
    current_error: Tensor
    baseline_error: Tensor


def validate_relational_outcome_shard(
    shard: RelationalOutcomeShard,
) -> tuple[int, int, int]:
    if shard.future_features.ndim != 4:
        raise ValueError("future_features must be [batch, candidates, horizons, width]")
    batch, candidates, horizon_count, width = shard.future_features.shape
    if width != RELATIONAL_FEATURE_DIM:
        raise ValueError("relational outcome has an incompatible feature width")
    if len(shard.source_keys) != batch or len(set(shard.source_keys)) != batch:
        raise ValueError("source_keys must be unique and match batch size")
    if len(shard.horizons) != horizon_count:
        raise ValueError("horizons do not match the feature tensor")
    if shard.horizon_mask.shape != (batch, horizon_count) or shard.horizon_mask.dtype != torch.bool:
        raise ValueError("horizon_mask must be boolean [batch, horizons]")
    if shard.expert_future_features.shape != (batch, horizon_count, width):
        raise ValueError("expert relational features have an incompatible shape")
    if shard.feature_mask.shape != (batch, width) or shard.feature_mask.dtype != torch.bool:
        raise ValueError("feature_mask must be boolean [batch, width]")
    matrix = (batch, candidates, horizon_count)
    for name, value in (
        ("progress", shard.progress),
        ("candidate_error", shard.candidate_error),
    ):
        if value.shape != matrix:
            raise ValueError(f"{name} must be [batch, candidates, horizons]")
    if shard.failure.shape != matrix or shard.failure.dtype != torch.bool:
        raise ValueError("failure must be boolean [batch, candidates, horizons]")
    if shard.preferred_index.shape != (batch, horizon_count) or shard.preferred_index.dtype != torch.long:
        raise ValueError("preferred_index must be int64 [batch, horizons]")
    if shard.current_error.shape != (batch, horizon_count):
        raise ValueError("current_error must be [batch, horizons]")
    if shard.baseline_error.shape != (batch, horizon_count):
        raise ValueError("baseline_error must be [batch, horizons]")
    floating = (
        shard.future_features,
        shard.expert_future_features,
        shard.progress,
        shard.candidate_error,
        shard.current_error,
        shard.baseline_error,
    )
    if any(not bool(torch.isfinite(value.float()).all()) for value in floating):
        raise ValueError("relational outcome contains non-finite values")
    if not bool(((shard.preferred_index >= 0) & (shard.preferred_index < candidates)).all()):
        raise ValueError("preferred_index is outside candidate range")
    return batch, candidates, horizon_count


def save_relational_outcome_shard(
    path: str | Path, shard: RelationalOutcomeShard
) -> None:
    validate_relational_outcome_shard(shard)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    torch.save(
        {
            "format": "duvla_v3_1_relational_outcome_shard_v1",
            **{name: value.cpu() if isinstance(value, Tensor) else value for name, value in vars(shard).items()},
        },
        temporary,
    )
    temporary.replace(destination)


def load_relational_outcome_shard(
    path: str | Path, *, mmap: bool = False
) -> RelationalOutcomeShard:
    payload = torch.load(Path(path), map_location="cpu", weights_only=True, mmap=mmap)
    if payload.get("format") != "duvla_v3_1_relational_outcome_shard_v1":
        raise ValueError(f"unsupported relational outcome shard: {path}")
    shard = RelationalOutcomeShard(
        **{
            name: tuple(payload[name]) if name in {"source_keys", "horizons"} else payload[name]
            for name in RelationalOutcomeShard.__dataclass_fields__
        }
    )
    validate_relational_outcome_shard(shard)
    return shard


def aggregate_relational_targets(
    shard: RelationalOutcomeShard,
    *,
    weights: tuple[float, ...] = (0.2, 0.3, 0.5),
) -> dict[str, Tensor]:
    batch, candidates, horizon_count = validate_relational_outcome_shard(shard)
    if len(weights) != horizon_count or any(value < 0.0 for value in weights):
        raise ValueError("weights must be non-negative and match horizons")
    raw_weights = shard.future_features.new_tensor(weights)[None]
    valid_weights = raw_weights * shard.horizon_mask.to(raw_weights.dtype)
    denominator = valid_weights.sum(dim=1, keepdim=True)
    if bool((denominator <= 0).any()):
        raise ValueError("every row must contain at least one valid horizon")
    normalized = valid_weights / denominator
    candidate_weights = normalized[:, None]
    aggregate_error = (shard.candidate_error * candidate_weights).sum(dim=-1)
    future_mask = shard.feature_mask[:, None, None, :].expand(
        batch, candidates, horizon_count, RELATIONAL_FEATURE_DIM
    ) & shard.horizon_mask[:, None, :, None]
    plan_mask = shard.feature_mask[:, None, :].expand(
        batch, horizon_count, RELATIONAL_FEATURE_DIM
    ) & shard.horizon_mask[:, :, None]
    return {
        "future_target": shard.future_features.flatten(2, 3),
        "future_mask": future_mask.flatten(2, 3),
        "planned_future_target": shard.expert_future_features.flatten(1, 2),
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


def list_relational_outcome_shards(directory: str | Path) -> tuple[Path, ...]:
    return tuple(sorted(Path(directory).glob("shard-*.pt")))
