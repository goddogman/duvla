from __future__ import annotations

import numpy as np

from duvla.data.contracts import ACTION_DIM, MinMaxStats
from duvla.evaluation.libero_contract import (
    libero_dummy_action,
    libero_global_task_index,
    libero_standard_max_steps,
    observation_to_state,
    normalized_action_to_env,
    orient_libero_view,
    quaternion_xyzw_to_axis_angle,
    resolve_benchmark_task_index,
)


def test_libero_dummy_action_holds_gripper_open_during_settle() -> None:
    action = libero_dummy_action()

    assert action.dtype == np.float32
    assert action.tolist() == [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0]


def test_quaternion_xyzw_to_axis_angle_identity() -> None:
    np.testing.assert_allclose(quaternion_xyzw_to_axis_angle([0.0, 0.0, 0.0, 1.0]), 0.0)


def test_observation_to_state_matches_eef_axis_angle_and_two_gripper_values() -> None:
    state = observation_to_state(
        {
            "robot0_eef_pos": [1.0, 2.0, 3.0],
            "robot0_eef_quat": [0.0, 0.0, np.sin(np.pi / 4), np.cos(np.pi / 4)],
            "robot0_gripper_qpos": [0.1, 0.2],
        }
    )

    np.testing.assert_allclose(state, [1.0, 2.0, 3.0, 0.0, 0.0, np.pi / 2, 0.1, 0.2])


def test_normalized_action_maps_dataset_gripper_to_libero_convention() -> None:
    stats = MinMaxStats(
        minimum=(-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 0.0),
        maximum=(1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0),
    )
    normalized = np.zeros(ACTION_DIM, dtype=np.float32)
    normalized[-1] = -1.0  # dataset gripper=0 -> simulator +1 (closed)

    action = normalized_action_to_env(normalized, stats)

    np.testing.assert_allclose(action[:6], 0.0)
    assert action[-1] == 1.0


def test_orient_libero_view_flips_both_image_axes_explicitly() -> None:
    image = np.arange(2 * 3 * 3, dtype=np.uint8).reshape(2, 3, 3)

    oriented = orient_libero_view(image)

    np.testing.assert_array_equal(oriented, image[::-1, ::-1])
    np.testing.assert_array_equal(orient_libero_view(image, flip_180=False), image)


def test_resolve_benchmark_task_index_uses_instruction_not_numeric_order() -> None:
    instructions = ("task a", "Pick  up   the bowl", "task c")

    assert resolve_benchmark_task_index("pick up the bowl", instructions) == 1


def test_libero_global_task_index_uses_official_suite_order() -> None:
    assert libero_global_task_index("libero_10", 9) == 2
    assert libero_global_task_index("libero_object", 7) == 25


def test_standard_episode_budget_is_suite_specific() -> None:
    assert libero_standard_max_steps("libero_spatial") == 280
    assert libero_standard_max_steps("libero_object") == 280
    assert libero_standard_max_steps("libero_goal") == 300
    assert libero_standard_max_steps("libero_10") == 520
