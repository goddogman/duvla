"""Pure contracts for train-side Cartesian recovery supervision.

These helpers deliberately operate only on robot poses and never consume a
LIBERO reward, success flag, task id, or evaluation initial state.  They are
used to validate whether perturbed *training demonstration* states can be
rejoined before expensive visual-feature caches are built.
"""

from __future__ import annotations

import numpy as np


OSC_POSITION_SCALE = np.asarray((0.05, 0.05, 0.05), dtype=np.float64)
OSC_ROTATION_SCALE = np.asarray((0.5, 0.5, 0.5), dtype=np.float64)


def _finite_vector(value: object, width: int, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.shape != (width,):
        raise ValueError(f"{name} must have shape ({width},)")
    if not np.isfinite(result).all():
        raise ValueError(f"{name} must contain finite values")
    return result


def quaternion_xyzw_to_matrix(quaternion: object) -> np.ndarray:
    """Convert a normalized-or-unnormalized ``xyzw`` quaternion to a matrix."""

    value = _finite_vector(quaternion, 4, "quaternion")
    norm = float(np.linalg.norm(value))
    if norm <= 1e-12:
        raise ValueError("quaternion norm must be positive")
    x, y, z, w = value / norm
    return np.asarray(
        (
            (1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)),
            (2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)),
            (2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)),
        ),
        dtype=np.float64,
    )


def rotation_matrix_to_axis_angle(matrix: object) -> np.ndarray:
    """Return the shortest rotation vector represented by a 3x3 matrix."""

    value = np.asarray(matrix, dtype=np.float64)
    if value.shape != (3, 3) or not np.isfinite(value).all():
        raise ValueError("rotation matrix must be finite shape (3, 3)")
    cosine = float(np.clip((np.trace(value) - 1.0) / 2.0, -1.0, 1.0))
    angle = float(np.arccos(cosine))
    if angle <= 1e-10:
        return np.zeros(3, dtype=np.float64)
    if np.pi - angle <= 1e-5:
        # Near pi the antisymmetric part is numerically unstable.  Recover an
        # axis from the diagonal and choose signs from the off-diagonal terms.
        diagonal = np.maximum((np.diag(value) + 1.0) / 2.0, 0.0)
        axis = np.sqrt(diagonal)
        largest = int(np.argmax(axis))
        if axis[largest] <= 1e-8:
            raise ValueError("rotation matrix has an indeterminate pi-axis")
        if largest == 0:
            axis[1] = np.copysign(axis[1], value[0, 1] + value[1, 0])
            axis[2] = np.copysign(axis[2], value[0, 2] + value[2, 0])
        elif largest == 1:
            axis[0] = np.copysign(axis[0], value[0, 1] + value[1, 0])
            axis[2] = np.copysign(axis[2], value[1, 2] + value[2, 1])
        else:
            axis[0] = np.copysign(axis[0], value[0, 2] + value[2, 0])
            axis[1] = np.copysign(axis[1], value[1, 2] + value[2, 1])
        axis /= np.linalg.norm(axis)
        return axis * angle
    axis = np.asarray(
        (
            value[2, 1] - value[1, 2],
            value[0, 2] - value[2, 0],
            value[1, 0] - value[0, 1],
        ),
        dtype=np.float64,
    ) / (2.0 * np.sin(angle))
    return axis * angle


def pose_errors(
    current_position: object,
    current_quaternion_xyzw: object,
    target_position: object,
    target_quaternion_xyzw: object,
) -> tuple[float, float]:
    """Return translation norm in metres and geodesic rotation in radians."""

    current_pos = _finite_vector(current_position, 3, "current_position")
    target_pos = _finite_vector(target_position, 3, "target_position")
    current_rotation = quaternion_xyzw_to_matrix(current_quaternion_xyzw)
    target_rotation = quaternion_xyzw_to_matrix(target_quaternion_xyzw)
    rotation_error = rotation_matrix_to_axis_angle(target_rotation @ current_rotation.T)
    return float(np.linalg.norm(target_pos - current_pos)), float(np.linalg.norm(rotation_error))


def cartesian_rejoin_action(
    current_position: object,
    current_quaternion_xyzw: object,
    target_position: object,
    target_quaternion_xyzw: object,
    *,
    gripper_command: float,
    position_gain: float = 1.0,
    rotation_gain: float = 1.0,
) -> np.ndarray:
    """Construct one bounded LIBERO OSC command toward a target end-effector pose.

    LIBERO's audited OSC_POSE controller maps input ``[-1,1]`` to translation
    ``+/-0.05m`` and axis-angle rotation ``+/-0.5rad`` per control step.
    """

    if position_gain <= 0.0 or rotation_gain <= 0.0:
        raise ValueError("rejoin gains must be positive")
    if not np.isfinite(gripper_command):
        raise ValueError("gripper command must be finite")
    current_pos = _finite_vector(current_position, 3, "current_position")
    target_pos = _finite_vector(target_position, 3, "target_position")
    current_rotation = quaternion_xyzw_to_matrix(current_quaternion_xyzw)
    target_rotation = quaternion_xyzw_to_matrix(target_quaternion_xyzw)
    rotation_error = rotation_matrix_to_axis_angle(target_rotation @ current_rotation.T)
    action = np.concatenate(
        (
            position_gain * (target_pos - current_pos) / OSC_POSITION_SCALE,
            rotation_gain * rotation_error / OSC_ROTATION_SCALE,
            np.asarray((gripper_command,), dtype=np.float64),
        )
    )
    return np.clip(action, -1.0, 1.0).astype(np.float32)
