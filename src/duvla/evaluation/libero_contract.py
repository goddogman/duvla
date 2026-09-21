"""Explicit state and action conversions for the official LIBERO adapter.

The conversion is based on the downloaded dataset's ``meta/info.json`` and
the local LeRobot LIBERO processor.  Keeping it in a small pure module makes
the simulator boundary testable without importing MuJoCo.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

from duvla.data.contracts import ACTION_DIM, STATE_DIM, MinMaxStats

STATE_NAMES = ("x", "y", "z", "axis_angle1", "axis_angle2", "axis_angle3", "gripper", "gripper")
ACTION_NAMES = ("x", "y", "z", "axis_angle1", "axis_angle2", "axis_angle3", "gripper")

# The official LIBERO suite orders are different from the 40-task LeRobot
# training order used by this project.  Keeping this mapping in the simulator
# boundary makes task-conditioned inference deterministic without selecting a
# different checkpoint per task.
LIBERO_SUITE_TO_GLOBAL_TASK = {
    "libero_10": (5, 7, 3, 8, 0, 9, 1, 4, 6, 2),
    "libero_goal": (19, 17, 14, 12, 18, 15, 13, 16, 10, 11),
    "libero_object": (24, 22, 26, 23, 21, 28, 27, 25, 29, 20),
    "libero_spatial": (34, 37, 38, 35, 31, 32, 30, 33, 36, 39),
}

# LeRobot's LIBERO environment defaults, derived from the longest training
# demonstration in each suite.  A single 280-step cap truncates Goal and,
# especially, the long-horizon LIBERO-10 suite before the standard deadline.
LIBERO_STANDARD_MAX_STEPS = {
    "libero_spatial": 280,
    "libero_object": 280,
    "libero_goal": 300,
    "libero_10": 520,
}


def libero_standard_max_steps(suite: str) -> int:
    """Return the standard per-suite LIBERO episode budget."""

    try:
        return LIBERO_STANDARD_MAX_STEPS[suite]
    except KeyError as exc:
        raise ValueError(f"unsupported LIBERO suite: {suite!r}") from exc


def libero_global_task_index(suite: str, task_index: int) -> int:
    """Return the global LeRobot task id for an official LIBERO task."""

    try:
        order = LIBERO_SUITE_TO_GLOBAL_TASK[suite]
    except KeyError as exc:
        raise ValueError(f"unsupported LIBERO suite: {suite!r}") from exc
    if not isinstance(task_index, int) or not 0 <= task_index < len(order):
        raise ValueError(f"task_index {task_index} is outside {suite} (0..{len(order) - 1})")
    return order[task_index]


def libero_dummy_action() -> np.ndarray:
    """Return LIBERO's reset-settling no-op with the Panda gripper held open."""

    action = np.zeros(ACTION_DIM, dtype=np.float32)
    action[-1] = -1.0
    return action


def resolve_benchmark_task_index(
    instruction: str,
    benchmark_instructions: Sequence[str],
) -> int:
    """Resolve a dataset instruction to the matching LIBERO task position.

    The LeRobot dataset's ``task_index`` order is not guaranteed to match the
    order returned by a LIBERO benchmark suite.  Exact normalized text
    matching makes that boundary explicit instead of silently pairing two
    different tasks.
    """

    if not isinstance(instruction, str) or not instruction.strip():
        raise ValueError("instruction must be a non-empty string")

    def normalize(value: str) -> str:
        return " ".join(value.split()).casefold()

    target = normalize(instruction)
    matches = [index for index, value in enumerate(benchmark_instructions) if normalize(value) == target]
    if not matches:
        raise ValueError(f"no LIBERO benchmark task matches instruction: {instruction!r}")
    if len(matches) > 1:
        raise ValueError(f"multiple LIBERO benchmark tasks match instruction: {instruction!r}")
    return matches[0]


def orient_libero_view(image: object, *, flip_180: bool = True) -> np.ndarray:
    """Apply the measured LIBERO↔LeRobot camera orientation transform."""

    value = np.asarray(image)
    if value.ndim != 3 or value.shape[2] != 3 or value.dtype != np.uint8:
        raise ValueError("LIBERO view must be a uint8 RGB HWC array")
    return np.flip(value, axis=(0, 1)).copy() if flip_180 else value.copy()


def quaternion_xyzw_to_axis_angle(quaternion: object) -> np.ndarray:
    """Convert one LIBERO quaternion in ``(x, y, z, w)`` order."""

    value = np.asarray(quaternion, dtype=np.float64)
    if value.shape != (4,):
        raise ValueError(f"quaternion must have shape (4,), got {value.shape}")
    if not np.isfinite(value).all():
        raise ValueError("quaternion must contain finite values")
    norm = np.linalg.norm(value)
    if norm <= 1e-12:
        raise ValueError("quaternion norm must be positive")
    value = value / norm
    w = float(np.clip(value[3], -1.0, 1.0))
    denominator = float(np.sqrt(max(1.0 - w * w, 0.0)))
    if denominator <= 1e-10:
        return np.zeros(3, dtype=np.float32)
    angle = 2.0 * np.arccos(w)
    return (value[:3] / denominator * angle).astype(np.float32)


def observation_to_state(observation: Mapping[str, object]) -> np.ndarray:
    """Build the dataset's 8D state from a raw LIBERO observation."""

    required = ("robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos")
    missing = [key for key in required if key not in observation]
    if missing:
        raise ValueError(f"LIBERO observation is missing: {', '.join(missing)}")
    position = np.asarray(observation[required[0]], dtype=np.float32)
    gripper = np.asarray(observation[required[2]], dtype=np.float32)
    if position.shape != (3,):
        raise ValueError(f"eef position must have shape (3,), got {position.shape}")
    if gripper.shape != (2,):
        raise ValueError(f"gripper qpos must have shape (2,), got {gripper.shape}")
    state = np.concatenate(
        [position, quaternion_xyzw_to_axis_angle(observation[required[1]]), gripper]
    ).astype(np.float32)
    if state.shape != (STATE_DIM,):  # defensive guard against future contract drift
        raise ValueError(f"LIBERO state must have shape ({STATE_DIM},), got {state.shape}")
    return state


def normalized_action_to_env(
    normalized_action: object,
    train_action_stats: MinMaxStats,
) -> np.ndarray:
    """Convert a min-max-normalized dataset action to native LIBERO action.

    The dataset stores the gripper as ``[0, 1]``.  LIBERO/robosuite's Panda
    gripper control uses ``-1=open, +1=closed``; the explicit conversion is
    ``-(2 * gripper - 1)``.  The six Cartesian values remain in their native
    dataset order and are not renamed or rotated.
    """

    normalized = np.asarray(normalized_action, dtype=np.float32)
    if normalized.shape != (ACTION_DIM,):
        raise ValueError(
            f"normalized action must have shape ({ACTION_DIM},), got {normalized.shape}"
        )
    if not np.isfinite(normalized).all():
        raise ValueError("normalized action must contain finite values")
    clipped = np.clip(normalized, -1.0, 1.0)
    data_action = np.asarray(train_action_stats.denormalize(clipped.tolist()), dtype=np.float32)
    env_action = data_action.copy()
    env_action[-1] = -(2.0 * data_action[-1] - 1.0)
    return np.clip(env_action, -1.0, 1.0).astype(np.float32)
