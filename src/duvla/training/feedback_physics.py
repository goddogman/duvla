"""训练侧标签度量；不得作为正式模型输入。"""

from __future__ import annotations

import numpy as np

from .recovery import pose_errors


def quaternion_distance_wxyz(left: np.ndarray, right: np.ndarray) -> float:
    left, right = np.asarray(left, dtype=np.float64), np.asarray(right, dtype=np.float64)
    if left.shape != (4,) or right.shape != (4,) or not np.isfinite([left, right]).all():
        raise ValueError("四元数必须是有限4维数组")
    norms = np.linalg.norm(left) * np.linalg.norm(right)
    if norms < 1e-12:
        raise ValueError("四元数范数为零")
    return float(2 * np.arccos(np.clip(abs(np.dot(left, right)) / norms, 0, 1)))


def physical_snapshot(env: object, observation: dict[str, object]) -> dict[str, object]:
    model = env.sim.model
    objects = {}
    for index in range(model.njnt):
        name = model.joint_id2name(index) or ""
        if name.startswith(("robot", "gripper")):
            continue
        kind = int(model.jnt_type[index])
        start = int(model.jnt_qposadr[index])
        width = {0: 7, 1: 4, 2: 1, 3: 1}[kind]
        objects[name] = (kind, np.asarray(env.sim.data.qpos[start:start + width]).copy())
    if not objects:
        raise ValueError("训练环境没有可审计物体关节")
    fingers = np.asarray(observation["robot0_gripper_qpos"], dtype=np.float64)
    return {"position": np.asarray(observation["robot0_eef_pos"]).copy(),
            "rotation": np.asarray(observation["robot0_eef_quat"]).copy(),
            "aperture": float(abs(fingers[0] - fingers[1])), "objects": objects}


def physical_errors(actual: dict[str, object], target: dict[str, object]) -> np.ndarray:
    pos, rot = pose_errors(actual["position"], actual["rotation"], target["position"], target["rotation"])
    if actual["objects"].keys() != target["objects"].keys():
        raise ValueError("物体关节布局不一致")
    positions, rotations = [], []
    for name, (kind, values) in actual["objects"].items():
        target_kind, expected = target["objects"][name]
        if kind != target_kind:
            raise ValueError("物体关节类型不一致")
        if kind == 0:
            positions.append(float(np.linalg.norm(values[:3] - expected[:3])))
            rotations.append(quaternion_distance_wxyz(values[3:], expected[3:]))
        elif kind == 1:
            rotations.append(quaternion_distance_wxyz(values, expected))
        elif kind == 2:
            positions.append(float(abs(values[0] - expected[0])))
        else:
            # 有限位铰链使用实际坐标差，不能把相隔2pi误认为相同关节状态。
            rotations.append(float(abs(values[0] - expected[0])))
    return np.asarray([pos, rot, max(positions, default=0.), max(rotations, default=0.),
                       abs(actual["aperture"] - target["aperture"])], dtype=np.float32)
