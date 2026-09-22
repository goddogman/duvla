"""Small, explicit LIBERO rollout utilities.

The LIBERO dependency is intentionally imported lazily because the training
environment does not contain the separate MuJoCo evaluation stack.  This
module currently exposes only a zero-action baseline.  It is an environment
and video-pipeline smoke test, not a policy evaluation.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np

from duvla.data.contracts import ACTION_DIM
from duvla.evaluation.initial_states import load_libero_initial_states


ActionMode = Literal["zeros"]


@dataclass(frozen=True)
class LiberoRolloutConfig:
    """Configuration for one deterministic, single-task smoke rollout."""

    suite: str = "libero_spatial"
    task_index: int = 0
    init_state_index: int = 0
    steps: int = 100
    camera_height: int = 256
    camera_width: int = 256
    fps: int = 20
    action_mode: ActionMode = "zeros"

    def __post_init__(self) -> None:
        if self.task_index < 0 or self.init_state_index < 0:
            raise ValueError("task and init-state indices must be non-negative")
        if self.steps <= 0 or self.camera_height <= 0 or self.camera_width <= 0:
            raise ValueError("steps and camera dimensions must be positive")
        if self.fps <= 0:
            raise ValueError("fps must be positive")
        if self.action_mode != "zeros":
            raise ValueError("only the explicit zero-action smoke mode is available")


@dataclass(frozen=True)
class LiberoRolloutResult:
    """Summary returned after a video has been written."""

    output_path: Path
    frames: int
    rewards: tuple[float, ...]
    success: bool
    task_name: str


def validate_env_action(action: object) -> np.ndarray:
    """Validate the native LIBERO/robosuite action contract.

    The order and semantics are deliberately not renamed here.  This check
    only enforces the simulator's observed seven-dimensional ``[-1, 1]``
    interface and therefore cannot silently alter action meanings.
    """

    vector = np.asarray(action, dtype=np.float32)
    if vector.shape != (ACTION_DIM,):
        raise ValueError(f"LIBERO action must have shape ({ACTION_DIM},), got {vector.shape}")
    if not np.isfinite(vector).all():
        raise ValueError("LIBERO action must contain finite values")
    if np.any(vector < -1.0) or np.any(vector > 1.0):
        raise ValueError("LIBERO action values must be within [-1, 1]")
    return vector


def compose_views(agentview: object, wristview: object) -> np.ndarray:
    """Place the two explicit LIBERO RGB views side by side for a video."""

    views = [np.asarray(image) for image in (agentview, wristview)]
    if any(image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8 for image in views):
        raise ValueError("LIBERO video views must be uint8 RGB arrays")
    if views[0].shape[0] != views[1].shape[0]:
        raise ValueError("LIBERO video views must have equal heights")
    return np.concatenate(views, axis=1)


def run_zero_action_rollout(
    output_path: str | Path, config: LiberoRolloutConfig = LiberoRolloutConfig()
) -> LiberoRolloutResult:
    """Run one LIBERO task with zero actions and write a side-by-side MP4.

    This function must be called from the dedicated LIBERO environment (for
    example ``/path/to/venv/bin/python``).  It never downloads assets.
    """

    try:
        import imageio.v2 as imageio
        from libero.libero import benchmark
        from libero.libero.envs import OffScreenRenderEnv
    except ImportError as exc:  # pragma: no cover - depends on eval env
        raise RuntimeError(
            "LIBERO rollout requires the separate MuJoCo evaluation environment"
        ) from exc

    suite_factory = benchmark.get_benchmark(config.suite)
    task_suite = suite_factory(task_order_index=0)
    if config.task_index >= task_suite.get_num_tasks():
        raise ValueError(f"task_index {config.task_index} is outside {task_suite.get_num_tasks()} tasks")
    task = task_suite.get_task(config.task_index)
    init_states = load_libero_initial_states(task_suite, config.task_index)
    if config.init_state_index >= len(init_states):
        raise ValueError(f"init_state_index {config.init_state_index} is outside {len(init_states)} states")

    env = OffScreenRenderEnv(
        bddl_file_name=task_suite.get_task_bddl_file_path(config.task_index),
        camera_names=["agentview", "robot0_eye_in_hand"],
        camera_heights=config.camera_height,
        camera_widths=config.camera_width,
        control_freq=config.fps,
        horizon=config.steps,
    )
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    rewards: list[float] = []
    frames = 0
    try:
        observation = env.reset()
        observation = env.set_init_state(init_states[config.init_state_index])
        writer = imageio.get_writer(destination, fps=config.fps, codec="libx264", macro_block_size=1)
        try:
            writer.append_data(
                compose_views(observation["agentview_image"], observation["robot0_eye_in_hand_image"])
            )
            frames += 1
            for _ in range(config.steps):
                observation, reward, done, _info = env.step(validate_env_action(np.zeros(ACTION_DIM)))
                writer.append_data(
                    compose_views(observation["agentview_image"], observation["robot0_eye_in_hand_image"])
                )
                frames += 1
                rewards.append(float(reward))
                if done:
                    break
        finally:
            writer.close()
        success = bool(env.check_success())
    finally:
        env.close()
    return LiberoRolloutResult(
        output_path=destination,
        frames=frames,
        rewards=tuple(rewards),
        success=success,
        task_name=task.name,
    )
