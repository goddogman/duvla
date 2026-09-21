from __future__ import annotations

import numpy as np

from duvla.evaluation.flow_contract import (
    StandardizationStats,
    flow_normalized_action_to_env,
)


def test_standardization_round_trip() -> None:
    stats = StandardizationStats((1.0, -2.0), (2.0, 4.0))
    values = np.asarray([[3.0, 6.0]], dtype=np.float32)
    np.testing.assert_allclose(stats.denormalize(stats.normalize(values)), values)


def test_native_gripper_sign_is_preserved_and_bounded() -> None:
    stats = StandardizationStats((0.0,) * 7, (1.0,) * 7)
    normalized = np.asarray([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]])
    result = flow_normalized_action_to_env(normalized, stats)
    assert result[:, 6].tolist() == [-1.0, 1.0]


def test_action_commands_are_clipped_only_at_environment_boundary() -> None:
    stats = StandardizationStats((0.0,) * 7, (1.0,) * 7)
    normalized = np.asarray([[2.0, -2.0, 0.0, 0.0, 0.0, 0.0, 4.0]])
    result = flow_normalized_action_to_env(normalized, stats)
    np.testing.assert_allclose(result[0, [0, 1, 6]], [1.0, -1.0, 1.0])
