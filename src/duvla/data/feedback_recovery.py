"""V3.26 配对纠错监督契约；不包含缓存生成、模拟器或正式训练入口。

物理误差仅为训练侧标签，不是任务成功概率。两个分支必须从同一个训练
演示起点和同源观测出发：执行各自的两步前缀，再由同一个冻结父策略继续。
只有各检测时域都不退化、且至少一项改善时才生成正监督。
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Literal, Mapping, Sequence

import torch
from torch import Tensor

from .contracts import ACTION_DIM, STATE_DIM, ContractError


PHYSICAL_ERROR_NAMES = (
    "eef_position_error_m",
    "eef_rotation_error_rad",
    "object_position_error_m",
    "object_rotation_error_rad",
    "gripper_aperture_error_m",
)
OBSERVATION_DOMAIN = "official_demo_xml_render"
MODEL_INPUT_FIELDS = frozenset({"context", "state", "base_actions", "feedback"})
EXECUTION_FEEDBACK_FIELDS = frozenset({
    "previous_context", "previous_state", "executed_actions", "executed_mask",
    "predicted_state_delta", "predicted_visual_delta", "valid", "prediction_valid",
})


def _sha256(value: str, name: str) -> None:
    if not isinstance(value, str) or len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise ContractError(f"{name} must be a non-empty lowercase SHA256")


def _floating(value: Tensor, name: str) -> None:
    if not isinstance(value, Tensor) or not value.is_floating_point():
        raise ContractError(f"{name} must be a floating tensor")
    if not bool(torch.isfinite(value).all()):
        raise ContractError(f"{name} contains non-finite values")


@dataclass(frozen=True)
class FeedbackProvenance:
    """只用于审计，不传入网络；同一对分支的此对象必须完全相同。"""

    dataset_id: str
    episode_key: str
    role: Literal["train", "validation"]
    source_manifest_sha256: str
    demo_xml_sha256: str
    initial_state_sha256: str
    current_context_sha256: str
    history_context_sha256: str
    normalization_sha256: str
    source_kind: str = "training_demonstration"
    observation_domain: str = OBSERVATION_DOMAIN
    history_domain: str = OBSERVATION_DOMAIN
    origin_kind: str = "demonstration"

    def __post_init__(self) -> None:
        if not self.dataset_id.strip() or not self.episode_key.strip():
            raise ContractError("dataset_id and episode_key must be non-empty")
        if self.role not in {"train", "validation"}:
            raise ContractError(
                "feedback source role must be train or validation, never evaluation"
            )
        if self.source_kind != "training_demonstration":
            raise ContractError("feedback source must originate from training demonstrations")
        if (
            self.observation_domain != OBSERVATION_DOMAIN
            or self.history_domain != OBSERVATION_DOMAIN
        ):
            raise ContractError(
                "current context and history must use the same official XML rendering domain"
            )
        if self.origin_kind not in {"demonstration", "synthetic_perturbation", "parent_rollout"}:
            raise ContractError("unsupported training-side origin_kind")
        for name in (
            "source_manifest_sha256", "demo_xml_sha256", "initial_state_sha256",
            "current_context_sha256", "history_context_sha256", "normalization_sha256",
        ):
            _sha256(getattr(self, name), name)

    @classmethod
    def from_mapping(cls, payload: Mapping[str, object]) -> FeedbackProvenance:
        allowed = {field.name for field in fields(cls)}
        if set(payload) - allowed:
            raise ContractError(f"forbidden provenance fields: {sorted(set(payload) - allowed)}")
        return cls(**dict(payload))  # type: ignore[arg-type]


def validate_feedback_model_inputs(payload: Mapping[str, object]) -> None:
    """严格白名单；未来标签、reward、success、task id 均不能混入输入。"""

    if set(payload) != MODEL_INPUT_FIELDS:
        raise ContractError(
            "model inputs must contain only context, state, base_actions and feedback"
        )
    for name in ("context", "state", "base_actions"):
        _floating(payload[name], name)
    context, state, actions = (payload[name] for name in (
        "context", "state", "base_actions",
    ))
    if context.ndim != 2 or min(context.shape) <= 0:
        raise ContractError("context must have shape [tokens,width]")
    if state.shape != (STATE_DIM,):
        raise ContractError(f"state must have shape [{STATE_DIM}]")
    if actions.ndim != 2 or actions.shape[1] != ACTION_DIM or actions.shape[0] < 2:
        raise ContractError(f"base_actions must have shape [horizon>=2,{ACTION_DIM}]")
    feedback = payload["feedback"]
    if not isinstance(feedback, Mapping) or set(feedback) != EXECUTION_FEEDBACK_FIELDS:
        raise ContractError("feedback must contain only the eight causal ExecutionFeedback fields")
    floating_shapes = {
        "previous_context": context.shape,
        "previous_state": (STATE_DIM,),
        "executed_actions": (2, ACTION_DIM),
        "predicted_state_delta": (STATE_DIM,),
        "predicted_visual_delta": (2, context.shape[-1]),
    }
    for name, shape in floating_shapes.items():
        _floating(feedback[name], name)
        if feedback[name].shape != shape:
            raise ContractError(f"feedback {name} has an invalid shape")
    for name, shape in {"executed_mask": (2,), "valid": (), "prediction_valid": ()}.items():
        value = feedback[name]
        if not isinstance(value, Tensor) or value.dtype != torch.bool or value.shape != shape:
            raise ContractError(f"feedback {name} must be boolean with shape {shape}")
    if not bool(feedback["valid"]) and bool(
        feedback["executed_mask"].any() or feedback["prediction_valid"]
    ):
        raise ContractError("invalid feedback cannot claim executed actions or a valid prediction")
    if bool(feedback["executed_mask"][1] and not feedback["executed_mask"][0]):
        raise ContractError("executed_mask must describe a contiguous executed prefix")
    if bool(feedback["valid"]) and not bool(feedback["executed_mask"].any()):
        raise ContractError("valid feedback must contain at least one executed action")
    if bool(feedback["prediction_valid"]) and not bool(feedback["executed_mask"].all()):
        raise ContractError("valid two-step prediction requires the complete executed prefix")
    if len({value.device for value in (context, state, actions, *feedback.values())}) != 1:
        raise ContractError("model input tensors must be on the same device")


@dataclass(frozen=True)
class FeedbackObservation:
    """单个训练起点的合法输入；不携带任何未来物理标签。

    base_actions 表示父策略动作经过环境单位转换、物理边界裁剪后，再按
    同一训练统计归一化的完整 chunk；不是可能超出执行范围的原始网络预测。
    其前两步必须与 parent 分支实际送入环境的动作对应。feedback 中的
    executed_actions 同样先保留实际裁剪后的环境命令，再归一化。
    """

    provenance: FeedbackProvenance
    context: Tensor
    state: Tensor
    base_actions: Tensor
    feedback: Mapping[str, Tensor]

    def model_inputs(self) -> dict[str, object]:
        result = {name: getattr(self, name) for name in MODEL_INPUT_FIELDS}
        validate_feedback_model_inputs(result)
        return result


@dataclass(frozen=True)
class BranchProtocol:
    """continuation_steps 不包含前缀；默认检测总步数为 2、10、18。"""

    parent_checkpoint_sha256: str
    continuation_policy_sha256: str
    continuation_config_sha256: str
    target_reference_sha256: str
    executed_prefix_steps: int = 2
    continuation_steps: tuple[int, ...] = (0, 8, 16)
    control_frequency_hz: int = 20
    continuation_seed: int = 23

    def __post_init__(self) -> None:
        for name in (
            "parent_checkpoint_sha256", "continuation_policy_sha256",
            "continuation_config_sha256", "target_reference_sha256",
        ):
            _sha256(getattr(self, name), name)
        if self.continuation_policy_sha256 != self.parent_checkpoint_sha256:
            raise ContractError("both branches must continue using the same frozen parent policy")
        if self.executed_prefix_steps != 2 or self.continuation_steps != (0, 8, 16):
            raise ContractError(
                "V3.26 requires two executed steps followed by matched 0/8/16 continuation steps"
            )
        if self.control_frequency_hz != 20:
            raise ContractError("V3.26 branch control frequency must be 20Hz")
        if not isinstance(self.continuation_seed, int) or isinstance(self.continuation_seed, bool):
            raise ContractError("continuation_seed must be an integer")

    @property
    def measurement_steps(self) -> tuple[int, ...]:
        return tuple(self.executed_prefix_steps + steps for steps in self.continuation_steps)


@dataclass(frozen=True)
class PhysicalBranchEvidence:
    """特权物理信息仅存在此标签侧；aperture 误差不等于抓取成功。

    normalized_executed_prefix 来自该分支实际执行的、裁剪后的环境命令，
    使用 provenance.normalization_sha256 对应的统计再次归一化。
    父分支前缀相等检查比较的是对应物理动作，不能用未裁剪预测冒充。
    """

    branch: Literal["parent", "correction"]
    provenance: FeedbackProvenance
    protocol: BranchProtocol
    normalized_executed_prefix: Tensor
    physical_errors: Tensor
    metric_valid: Tensor
    reliable: bool = True

    def validate(self) -> None:
        if self.branch not in {"parent", "correction"}:
            raise ContractError("branch must identify parent or correction")
        _floating(self.normalized_executed_prefix, "normalized_executed_prefix")
        if self.normalized_executed_prefix.shape != (
            self.protocol.executed_prefix_steps, ACTION_DIM,
        ):
            raise ContractError("branch executed action prefix does not match the protocol")
        _floating(self.physical_errors, "physical_errors")
        shape = (len(self.protocol.measurement_steps), len(PHYSICAL_ERROR_NAMES))
        if self.physical_errors.shape != shape or bool((self.physical_errors < 0).any()):
            raise ContractError("physical_errors must be non-negative [matched horizons,5] values")
        if self.metric_valid.shape != shape or self.metric_valid.dtype != torch.bool:
            raise ContractError("metric_valid must be boolean and match physical_errors")
        if not isinstance(self.reliable, bool):
            raise ContractError("reliable must explicitly be boolean")


@dataclass(frozen=True)
class ImprovementRule:
    """各列单位见 PHYSICAL_ERROR_NAMES；margin 是预注册误差容限，不是成功率。"""

    minimum_improvement: tuple[float, ...]
    non_regression_tolerance: tuple[float, ...]

    def tensors(self, reference: Tensor) -> tuple[Tensor, Tensor]:
        improvement = reference.new_tensor(self.minimum_improvement)
        tolerance = reference.new_tensor(self.non_regression_tolerance)
        for name, value in (
            ("minimum_improvement", improvement), ("non_regression_tolerance", tolerance),
        ):
            _floating(value, name)
            if value.shape != (len(PHYSICAL_ERROR_NAMES),) or bool((value < 0).any()):
                raise ContractError(f"{name} must supply five non-negative per-metric margins")
        if bool((improvement <= tolerance).any()):
            raise ContractError(
                "minimum_improvement must exceed non_regression_tolerance for every metric"
            )
        return improvement, tolerance


@dataclass(frozen=True)
class FeedbackRecoveryTarget:
    target_actions: Tensor
    action_mask: Tensor
    gate_target: Tensor
    gate_mask: Tensor
    decision: Literal["beneficial", "keep_parent", "unknown"]
    reason: str


def build_feedback_recovery_target(
    observation: FeedbackObservation,
    parent: PhysicalBranchEvidence,
    correction: PhysicalBranchEvidence,
    rule: ImprovementRule,
) -> FeedbackRecoveryTarget:
    """只构造单样本标签；缺少可靠配对证据时所有损失掩码为 False。"""

    observation.model_inputs()
    parent.validate()
    correction.validate()
    if parent.branch != "parent" or correction.branch != "correction":
        raise ContractError("paired evidence must identify the parent and correction branches")
    if (
        parent.provenance != observation.provenance
        or correction.provenance != observation.provenance
    ):
        raise ContractError(
            "paired branches must have identical source, XML, start and context/history hashes"
        )
    if parent.protocol != correction.protocol:
        raise ContractError(
            "paired branches must have identical horizons, reference and continuation protocol"
        )
    steps = parent.protocol.executed_prefix_steps
    base = observation.base_actions
    if not torch.equal(parent.normalized_executed_prefix.to(base), base[:steps]):
        raise ContractError(
            "parent executed prefix must exactly match the observed base action prefix"
        )
    improvement, tolerance = rule.tensors(parent.physical_errors)
    target = base.detach().clone()
    action_mask = torch.zeros(base.shape[0], dtype=torch.bool, device=base.device)
    gate_target = torch.zeros_like(action_mask)
    gate_mask = torch.zeros_like(action_mask)
    reliable = parent.reliable and correction.reliable and bool(
        parent.metric_valid.all() and correction.metric_valid.all()
    )
    if not reliable:
        return FeedbackRecoveryTarget(
            target, action_mask, gate_target, gate_mask, "unknown",
            "incomplete_or_unreliable_physical_evidence",
        )
    difference = parent.physical_errors - correction.physical_errors.to(parent.physical_errors)
    no_regression = bool((difference >= -tolerance).all())
    measurable_improvement = bool((difference >= improvement).any())
    beneficial = no_regression and measurable_improvement
    action_mask[:steps] = True
    gate_mask[:steps] = True
    if beneficial:
        target[:steps] = correction.normalized_executed_prefix.detach().to(base)
        gate_target[:steps] = True
    return FeedbackRecoveryTarget(
        target, action_mask, gate_target, gate_mask,
        "beneficial" if beneficial else "keep_parent",
        "paired_physical_improvement" if beneficial else "no_advantage_or_physical_regression",
    )


def assert_feedback_episode_isolation(
    train: Sequence[FeedbackProvenance], validation: Sequence[FeedbackProvenance]
) -> None:
    """起点、扰动或分支不同也不能让同一训练 episode 跨 split。"""

    if any(item.role != "train" for item in train) or any(
        item.role != "validation" for item in validation
    ):
        raise ContractError("feedback split roles do not match train/validation")
    train_episodes = {(item.dataset_id, item.episode_key) for item in train}
    validation_episodes = {(item.dataset_id, item.episode_key) for item in validation}
    if train_episodes & validation_episodes:
        raise ContractError("feedback train/validation overlap by demonstration episode")
