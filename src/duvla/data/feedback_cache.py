"""V3.26 真实执行反馈缓存；物理证据只留在标签侧。"""

from __future__ import annotations

from pathlib import Path

import torch
from torch import Tensor

from .feedback_recovery import FeedbackProvenance


FLOAT_SHAPES = {
    "context": (135, 768), "state_token": (1, 768), "state": (8,),
    "previous_context": (135, 768), "previous_state": (8,),
    "previous_actions": (2, 7), "base_actions": (8, 7), "target_actions": (8, 7),
    "future_actions": (2, 7), "future_state_delta": (8,),
    "future_visual_delta": (2, 768), "parent_errors": (3, 5),
    "correction_errors": (3, 5),
}
BOOL_SHAPES = {"action_mask": (8,), "gate_target": (8,), "gate_mask": (8,)}
META_FIELDS = {"provenance", "source_keys", "decisions", "correction_sources"}


def validate_feedback_shard(payload: dict[str, object]) -> int:
    if set(payload) != set(FLOAT_SHAPES) | set(BOOL_SHAPES) | META_FIELDS:
        raise ValueError("反馈缓存字段不符合白名单")
    count = len(payload["source_keys"])
    if count == 0 or len(set(payload["source_keys"])) != count:
        raise ValueError("缓存起点必须非空且不重复")
    for name, shape in FLOAT_SHAPES.items():
        value = payload[name]
        if not isinstance(value, Tensor) or not value.is_floating_point() or tuple(value.shape) != (count, *shape):
            raise ValueError(f"{name} 形状/类型不正确")
        if not bool(torch.isfinite(value).all()):
            raise ValueError(f"{name} 存在非有限值")
    for name, shape in BOOL_SHAPES.items():
        value = payload[name]
        if not isinstance(value, Tensor) or value.dtype != torch.bool or tuple(value.shape) != (count, *shape):
            raise ValueError(f"{name} 形状/类型不正确")
    if any(len(payload[name]) != count for name in META_FIELDS):
        raise ValueError("元数据数量不符")
    for row, raw in enumerate(payload["provenance"]):
        FeedbackProvenance.from_mapping(raw)
        decision = payload["decisions"][row]
        if decision not in {"beneficial", "keep_parent", "unknown"}:
            raise ValueError("未知标签决策")
        mask = payload["action_mask"][row]
        gate = payload["gate_target"][row]
        if bool(mask[2:].any()) or not torch.equal(mask, payload["gate_mask"][row]):
            raise ValueError("仅监督经过分支检验的两步前缀")
        if decision == "unknown":
            if bool(mask.any() or gate.any()):
                raise ValueError("未知样本不得产生动作/gate监督")
        elif not bool(mask[:2].all()) or not torch.equal(gate[:2], torch.full((2,), decision == "beneficial")):
            raise ValueError("收益标签与mask不一致")
        if decision != "beneficial" and not torch.equal(payload["target_actions"][row], payload["base_actions"][row]):
            raise ValueError("无可靠收益必须保留父动作")
        if not torch.equal(payload["future_actions"][row], payload["target_actions"][row, :2]):
            raise ValueError("未来监督必须对应真实执行的目标前缀")
    return count


def save_feedback_shard(path: Path, payload: dict[str, object]) -> None:
    validate_feedback_shard(payload)
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".pt.tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def load_feedback_shard(path: Path) -> dict[str, object]:
    payload = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    validate_feedback_shard(payload)
    return payload
