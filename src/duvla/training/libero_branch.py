"""Pure helpers for train-demonstration physical branch collection."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
import random
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class DemonstrationFrame:
    file_path: Path
    demo_key: str
    frame_index: int

    @property
    def source_key(self) -> str:
        return f"{self.file_path.parent.name}/{self.file_path.name}/{self.demo_key}/{self.frame_index}"


def stable_demo_split(
    demo_keys: Sequence[str],
    *,
    task_key: str,
    seed: int,
    validation_fraction: float = 0.2,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Split demos deterministically without depending on Python hash randomization."""

    if len(demo_keys) < 2:
        raise ValueError("a task needs at least two demonstrations")
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be in (0, 1)")
    if len(set(demo_keys)) != len(demo_keys):
        raise ValueError("demo_keys must be unique")
    digest = sha256(f"{seed}:{task_key}".encode()).digest()
    task_seed = int.from_bytes(digest[:8], "big")
    shuffled = sorted(str(key) for key in demo_keys)
    random.Random(task_seed).shuffle(shuffled)
    count = max(1, round(len(shuffled) * validation_fraction))
    count = min(count, len(shuffled) - 1)
    validation = tuple(sorted(shuffled[:count]))
    train = tuple(sorted(shuffled[count:]))
    return train, validation


def sample_demo_frames(
    file_path: Path,
    demo_lengths: Mapping[str, int],
    demo_keys: Sequence[str],
    *,
    count: int,
    history_length: int,
    executed_action_steps: int,
    frame_stride: int,
) -> tuple[DemonstrationFrame, ...]:
    """Choose a phase-covering, demo-balanced fixed number of source frames."""

    if min(count, executed_action_steps, frame_stride) <= 0 or history_length < 0:
        raise ValueError("count/executed steps/stride must be positive and history non-negative")
    if not demo_keys:
        raise ValueError("demo_keys cannot be empty")
    eligible: dict[str, np.ndarray] = {}
    first = history_length * frame_stride
    # Official HDF5 demonstrations are commonly recorded at 20 Hz while the
    # policy runs at 10 Hz.  ``frame_stride`` is the validated source/policy
    # frequency ratio, so it applies to both history spacing and the expert
    # future endpoint.
    future = executed_action_steps * frame_stride
    for key in demo_keys:
        length = int(demo_lengths[key])
        positions = np.arange(first, length - future, frame_stride, dtype=np.int64)
        if positions.size:
            eligible[str(key)] = positions
    if not eligible:
        raise ValueError("no demonstration contains an eligible branch frame")

    selected: list[DemonstrationFrame] = []
    keys = sorted(eligible)
    base, remainder = divmod(count, len(keys))
    for number, key in enumerate(keys):
        allocation = base + (1 if number < remainder else 0)
        positions = eligible[key]
        if allocation <= 0:
            continue
        indices = np.linspace(0, positions.size - 1, num=allocation, dtype=np.int64)
        chosen = positions[indices]
        selected.extend(
            DemonstrationFrame(file_path=file_path, demo_key=key, frame_index=int(frame))
            for frame in chosen
        )
    if len(selected) != count:
        raise RuntimeError(f"frame sampler produced {len(selected)} rows, expected {count}")
    return tuple(selected)


def hdf5_control_frequency(env_args: object) -> int:
    """Read the source control frequency from LIBERO's HDF5 ``env_args``."""

    value = env_args
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError as exc:
            raise ValueError("HDF5 env_args is not valid JSON") from exc
    if not isinstance(value, Mapping):
        raise ValueError("HDF5 env_args must be a mapping or JSON object")
    nested = value.get("env_kwargs")
    frequency = nested.get("control_freq") if isinstance(nested, Mapping) else None
    if frequency is None:
        frequency = value.get("control_freq")
    if not isinstance(frequency, int) or isinstance(frequency, bool) or frequency <= 0:
        raise ValueError("HDF5 env_args has no positive integer control_freq")
    return frequency


def simulator_state_weights(*, width: int, nq: int, nv: int) -> np.ndarray:
    """Weight qpos more than qvel while explicitly ignoring simulator time."""

    if min(width, nq, nv) <= 0:
        raise ValueError("state width, nq and nv must be positive")
    if 1 + nq + nv > width:
        raise ValueError("flattened simulator state is shorter than time+qpos+qvel")
    result = np.full(width, 0.1, dtype=np.float32)
    result[0] = 0.0
    result[1 : 1 + nq] = 1.0
    result[1 + nq : 1 + nq + nv] = 0.1
    return result


def robust_state_range_scale(
    state_sequences: Sequence[np.ndarray],
    *,
    nq: int,
    nv: int,
    lower_quantile: float = 0.05,
    upper_quantile: float = 0.95,
    qpos_floor: float = 0.01,
    qvel_floor: float = 0.05,
    auxiliary_floor: float = 0.01,
) -> np.ndarray:
    """Fit a task-local state-range scale from training demonstrations only.

    A short-horizon delta is often exactly zero for object coordinates that do
    not move in most frames.  Dividing by that delta amplifies simulator noise
    and makes outcome labels saturate.  A robust trajectory range plus
    component-aware physical floors remains stable for both moving and static
    coordinates.
    """

    if not state_sequences:
        raise ValueError("state_sequences must be non-empty")
    if min(nq, nv) <= 0:
        raise ValueError("nq and nv must be positive")
    if not 0.0 <= lower_quantile < upper_quantile <= 1.0:
        raise ValueError("quantiles must satisfy 0 <= lower < upper <= 1")
    if min(qpos_floor, qvel_floor, auxiliary_floor) <= 0.0:
        raise ValueError("physical scale floors must be positive")
    rows: list[np.ndarray] = []
    width: int | None = None
    for states in state_sequences:
        values = np.asarray(states, dtype=np.float64)
        if values.ndim != 2 or values.shape[0] == 0:
            raise ValueError("each simulator state sequence must be non-empty rank two")
        width = values.shape[1] if width is None else width
        if values.shape[1] != width:
            raise ValueError("simulator state width changed within one task")
        if not np.isfinite(values).all():
            raise ValueError("simulator state sequence contains non-finite values")
        rows.append(values)
    if width is None or 1 + nq + nv > width:
        raise ValueError("flattened simulator state is shorter than time+qpos+qvel")
    joined = np.concatenate(rows, axis=0)
    scale = np.quantile(joined, upper_quantile, axis=0) - np.quantile(
        joined, lower_quantile, axis=0
    )
    floors = np.full(width, auxiliary_floor, dtype=np.float64)
    floors[0] = 1.0  # simulator time is ignored by simulator_state_weights
    floors[1 : 1 + nq] = qpos_floor
    floors[1 + nq : 1 + nq + nv] = qvel_floor
    return np.maximum(scale, floors).astype(np.float32)


def robust_state_delta_scale(
    state_sequences: Sequence[np.ndarray], *, future_offset: int
) -> np.ndarray:
    """Legacy schema-2 scale retained only for loading historical evidence."""

    if future_offset <= 0 or not state_sequences:
        raise ValueError("future_offset and state_sequences must be non-empty")
    deltas: list[np.ndarray] = []
    width: int | None = None
    for states in state_sequences:
        values = np.asarray(states, dtype=np.float64)
        if values.ndim != 2 or values.shape[0] <= future_offset:
            continue
        width = values.shape[1] if width is None else width
        if values.shape[1] != width:
            raise ValueError("simulator state width changed within one task")
        deltas.append(np.abs(values[future_offset:] - values[:-future_offset]))
    if not deltas or width is None:
        raise ValueError("train demonstrations contain no state deltas")
    joined = np.concatenate(deltas, axis=0)
    scale = np.quantile(joined, 0.75, axis=0)
    return np.maximum(scale, 1e-4).astype(np.float32)


def hdf5_robot_state(observation_group: object, frame_index: int) -> np.ndarray:
    """Read the exact 8-D LIBERO state contract from a demonstration obs group."""

    group = observation_group
    if "ee_states" in group and "gripper_states" in group:  # type: ignore[operator]
        ee = np.asarray(group["ee_states"][frame_index], dtype=np.float32)  # type: ignore[index]
        gripper = np.asarray(group["gripper_states"][frame_index], dtype=np.float32)  # type: ignore[index]
    elif all(key in group for key in ("ee_pos", "ee_ori", "gripper_states")):  # type: ignore[operator]
        ee = np.concatenate(
            (
                np.asarray(group["ee_pos"][frame_index], dtype=np.float32),  # type: ignore[index]
                np.asarray(group["ee_ori"][frame_index], dtype=np.float32),  # type: ignore[index]
            )
        )
        gripper = np.asarray(group["gripper_states"][frame_index], dtype=np.float32)  # type: ignore[index]
    else:
        raise ValueError("demonstration obs lacks ee/gripper state fields")
    result = np.concatenate((ee.reshape(-1), gripper.reshape(-1)))
    if result.shape != (8,) or not np.isfinite(result).all():
        raise ValueError(f"demonstration robot state must be finite shape (8,), got {result.shape}")
    return result


def robot_state_alignment_errors(
    cached_state: np.ndarray, source_state: np.ndarray
) -> tuple[float, float]:
    """Return position and gripper max errors for two 8-D LIBERO states.

    Axis-angle coordinates are deliberately excluded: equivalent rotations can
    have discontinuous axis-angle representations.  Position and gripper state
    are the stable, directly actionable checks used before pairing cached Qwen
    context with an official-HDF5 simulator state.
    """

    cached = np.asarray(cached_state, dtype=np.float32)
    source = np.asarray(source_state, dtype=np.float32)
    if cached.shape != (8,) or source.shape != (8,):
        raise ValueError("LIBERO robot states must both have shape (8,)")
    if not np.isfinite(cached).all() or not np.isfinite(source).all():
        raise ValueError("LIBERO robot states must contain finite values")
    return (
        float(np.max(np.abs(cached[:3] - source[:3]))),
        float(np.max(np.abs(cached[6:] - source[6:]))),
    )


def nonzero_arm_action_indices(actions: np.ndarray) -> np.ndarray:
    """Return the source rows retained by the public LeRobot LIBERO conversion.

    The embedded-image ``HuggingFaceVLA/libero`` dataset removes demonstration
    frames whose six arm-control coordinates are all exactly zero.  The
    gripper coordinate is deliberately excluded from this decision.  Keeping
    this small rule explicit lets cached Qwen rows be aligned back to their
    original HDF5 MuJoCo states without relying on task ids at policy runtime.
    """

    values = np.asarray(actions, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] != 7:
        raise ValueError("LIBERO actions must have shape [frames, 7]")
    if not np.isfinite(values).all():
        raise ValueError("LIBERO actions contain non-finite values")
    return np.flatnonzero(np.any(values[:, :6] != 0.0, axis=1)).astype(np.int64)


def align_filtered_action_sequence(
    source_actions: np.ndarray,
    converted_actions: np.ndarray,
) -> np.ndarray:
    """Map one converted episode frame-for-frame to its source HDF5 rows.

    An exact float32 comparison is intentional: it prevents a merely similar
    trajectory from being paired with the wrong simulator state.  Callers may
    skip episodes that fail this audit and sample a different training row.
    """

    source = np.asarray(source_actions, dtype=np.float32)
    converted = np.asarray(converted_actions, dtype=np.float32)
    indices = nonzero_arm_action_indices(source)
    filtered = source[indices]
    if converted.ndim != 2 or converted.shape[1:] != (7,):
        raise ValueError("converted LIBERO actions must have shape [frames, 7]")
    if not np.isfinite(converted).all():
        raise ValueError("converted LIBERO actions contain non-finite values")
    if filtered.shape != converted.shape or not np.array_equal(filtered, converted):
        raise ValueError("converted episode does not exactly match filtered HDF5 actions")
    return indices
