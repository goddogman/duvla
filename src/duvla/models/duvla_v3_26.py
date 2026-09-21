"""V3.26 的执行反馈 Flow 原型；此模块不启动训练或模拟器任务。

历史仅包含允许的观测、真实执行动作和先前预测。物体真值、任务编号、
reward/success 均不属于模型输入。预测输出目前只是待监督的辅助能力。
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .duvla_v2_1 import DuvlaV21Config, DuvlaV21Policy, _RecoveryCrossBlock


@dataclass(frozen=True)
class DuvlaV326Config(DuvlaV21Config):
    spatial_tokens: int = 64
    unified_flow_gripper: bool = True
    highres_layer14_visual: bool = True
    dense_interleaved_flow: bool = True
    feedback_hidden_dim: int = 384
    feedback_layers: int = 2
    feedback_heads: int = 6
    feedback_execution_steps: int = 2
    feedback_velocity_limit: float = 1.0
    feedback_gripper_velocity_limit: float = 1.0
    feedback_gate_init: float = 0.01

    def __post_init__(self) -> None:
        super().__post_init__()
        if min(self.feedback_hidden_dim, self.feedback_layers, self.feedback_heads) <= 0:
            raise ValueError("feedback dimensions must be positive")
        if self.feedback_hidden_dim % self.feedback_heads:
            raise ValueError("feedback_hidden_dim must be divisible by feedback_heads")
        if self.feedback_execution_steps != 2 or self.replan_action_steps != 2:
            raise ValueError("V3.26 requires the audited two-step execution contract")
        if self.camera_count != 2 or self.action_dim != 7 or self.state_dim != 8:
            raise ValueError("V3.26 requires two cameras, eight state and seven action dimensions")
        if not self.unified_flow_gripper or self.observation_feedback_recovery:
            raise ValueError("V3.26 requires joint Flow gripper without post-hoc V3.25 recovery")
        if not (self.highres_layer14_visual and self.dense_interleaved_flow):
            raise ValueError("V3.26 retains the dense high-resolution parent context")
        if self.multilayer_spatial_residual:
            raise ValueError("V3.26 does not combine the discarded spatial residual")
        if min(self.feedback_velocity_limit, self.feedback_gripper_velocity_limit) <= 0:
            raise ValueError("feedback velocity limits must be positive")
        if not all(math.isfinite(value) for value in (
            self.feedback_velocity_limit, self.feedback_gripper_velocity_limit,
            self.feedback_gate_init,
        )) or not 0 < self.feedback_gate_init < 1:
            raise ValueError("feedback limits and gate initial probability must be finite")


@dataclass(frozen=True)
class ExecutionFeedback:
    """上一轮实际执行包；所有 state/action 都使用父策略的归一化空间。

    previous_context [B,T,D]，previous_state [B,8]；executed_actions [B,2,7]
    是真正送入环境的动作（裁剪/转换后再归一化），executed_mask [B,2]。
    predicted_state_delta [B,8]、predicted_visual_delta [B,2,D] 必须对应
    这段实际动作；不可用预测由 prediction_valid [B] 明确标记。
    valid [B] 表示有上一轮执行；episode 开始必须为 False。
    """

    previous_context: Tensor
    previous_state: Tensor
    executed_actions: Tensor
    executed_mask: Tensor
    predicted_state_delta: Tensor
    predicted_visual_delta: Tensor
    valid: Tensor
    prediction_valid: Tensor


def _finite(name: str, values: Tensor) -> None:
    if not bool(torch.isfinite(values).all()):
        raise ValueError(f"{name} must contain only finite values")


def _bool_mask(name: str, values: Tensor, shape: tuple[int, ...]) -> None:
    if tuple(values.shape) != shape or values.dtype != torch.bool:
        raise ValueError(f"{name} must be a boolean tensor with shape {shape}")


def _masked(values: Tensor, valid: Tensor) -> Tensor:
    # where 而非乘零，避免无效历史的 NaN 进入运算。
    return torch.where(valid.reshape(valid.shape + (1,) * (values.ndim - valid.ndim)),
                       values, torch.zeros_like(values))


def spatial_summary(context: Tensor, camera_count: int = 2, spatial_tokens: int = 64) -> Tensor:
    """从父 context 前部按相机分别取空间均值，不混入语言/历史 token。"""
    if min(camera_count, spatial_tokens) <= 0 or context.ndim != 3:
        raise ValueError("spatial summary requires [B,T,D] and positive layout dimensions")
    if context.shape[1] < camera_count * spatial_tokens:
        raise ValueError("context is too short for the requested camera spatial layout")
    return context[:, :camera_count * spatial_tokens].reshape(
        context.shape[0], camera_count, spatial_tokens, context.shape[-1]
    ).mean(dim=2)


class ExecutionFeedbackModule(nn.Module):
    """真实状态变化和预测误差条件化的有界 Flow velocity residual。"""

    def __init__(self, config: DuvlaV326Config) -> None:
        super().__init__()
        self.config = config
        width = config.feedback_hidden_dim
        summary_dim = config.camera_count * config.hidden_dim
        action_dim = config.feedback_execution_steps * config.action_dim
        feedback_dim = 3 * config.state_dim + 2 * summary_dim + action_dim + 3
        self.context_projection = nn.Linear(config.hidden_dim, width)
        self.spatial_feedback_projection = nn.Linear(config.hidden_dim, width)
        self.spatial_feedback_position = nn.Parameter(
            torch.randn(config.camera_count * config.spatial_tokens, width) * 0.02
        )
        self.feedback_projection = nn.Sequential(
            nn.Linear(feedback_dim, width), nn.SiLU(), nn.Linear(width, width),
        )
        self.action_projection = nn.Linear(config.action_dim, width)
        self.state_projection = nn.Linear(config.state_dim, width)
        self.time_projection = nn.Sequential(nn.Linear(3, width), nn.SiLU(), nn.Linear(width, width))
        self.position = nn.Parameter(torch.randn(config.action_horizon, width) * 0.02)
        self.blocks = nn.ModuleList(
            _RecoveryCrossBlock(width, config.feedback_heads, config.dropout)
            for _ in range(config.feedback_layers)
        )
        self.velocity_output = nn.Linear(width, config.action_dim)
        self.gate = nn.Linear(width, 1)
        nn.init.zeros_(self.velocity_output.weight)
        nn.init.zeros_(self.velocity_output.bias)
        nn.init.zeros_(self.gate.weight)
        nn.init.constant_(self.gate.bias, math.log(config.feedback_gate_init / (1 - config.feedback_gate_init)))
        limits = torch.full((config.action_dim,), config.feedback_velocity_limit)
        limits[-1] = config.feedback_gripper_velocity_limit
        self.register_buffer("velocity_limits", limits)
        forecast_dim = summary_dim + config.hidden_dim + config.state_dim + action_dim + 2
        self.forecast_encoder = nn.Sequential(
            nn.Linear(forecast_dim, width), nn.SiLU(), nn.Linear(width, width), nn.SiLU(),
        )
        self.future_state = nn.Linear(width, config.state_dim)
        self.future_visual = nn.Linear(width, summary_dim)
        nn.init.zeros_(self.future_state.weight)
        nn.init.zeros_(self.future_state.bias)
        nn.init.zeros_(self.future_visual.weight)
        nn.init.zeros_(self.future_visual.bias)

    def _validate_current(self, context: Tensor, state: Tensor) -> None:
        cfg = self.config
        if context.ndim != 3 or tuple(context.shape[1:]) != (cfg.context_tokens, cfg.hidden_dim):
            raise ValueError("context must preserve the exact parent [B,T,D] layout")
        if tuple(state.shape) != (context.shape[0], cfg.state_dim):
            raise ValueError("state must have shape [B,8]")
        _finite("context", context)
        _finite("state", state)

    def visual_summary(self, context: Tensor) -> Tensor:
        """固定的父 context 双相机均值；无新增可学习目标编码器。"""
        cfg = self.config
        return spatial_summary(context, cfg.camera_count, cfg.spatial_tokens)

    def _execution_actions(self, actions: Tensor, mask: Tensor, batch: int) -> Tensor:
        if tuple(actions.shape) != (batch, 2, self.config.action_dim):
            raise ValueError("executed_actions must have shape [B,2,7]")
        _bool_mask("executed_mask", mask, (batch, 2))
        if bool((mask[:, 1] & ~mask[:, 0]).any()):
            raise ValueError("executed_mask must describe a contiguous executed prefix")
        actions = _masked(actions, mask)
        _finite("valid executed_actions", actions)
        return actions

    def encode_feedback(
        self, context: Tensor, state: Tensor, feedback: ExecutionFeedback | None
    ) -> tuple[Tensor, Tensor]:
        self._validate_current(context, state)
        batch = context.shape[0]
        if feedback is None:
            return context.new_zeros(batch, self.config.feedback_hidden_dim), torch.zeros(
                batch, device=context.device, dtype=torch.bool
            )
        _bool_mask("valid", feedback.valid, (batch,))
        _bool_mask("prediction_valid", feedback.prediction_valid, (batch,))
        expected = {
            "previous_context": context.shape,
            "previous_state": state.shape,
            "predicted_state_delta": state.shape,
            "predicted_visual_delta": (batch, 2, self.config.hidden_dim),
        }
        for name, shape in expected.items():
            if tuple(getattr(feedback, name).shape) != tuple(shape):
                raise ValueError(f"{name} must have shape {tuple(shape)}")
        _bool_mask("executed_mask", feedback.executed_mask, (batch, 2))
        if bool((feedback.valid & ~feedback.executed_mask.any(dim=1)).any()):
            raise ValueError("valid execution feedback must contain an executed action")
        mask = feedback.executed_mask & feedback.valid[:, None]
        actions = self._execution_actions(feedback.executed_actions, mask, batch)
        valid = feedback.valid
        prediction_valid = feedback.prediction_valid & valid
        previous_context = _masked(feedback.previous_context.detach(), valid)
        previous_state = _masked(feedback.previous_state.detach(), valid)
        predicted_state = _masked(feedback.predicted_state_delta.detach(), prediction_valid)
        predicted_visual = _masked(feedback.predicted_visual_delta.detach(), prediction_valid)
        for name, value in (
            ("previous_context", previous_context), ("previous_state", previous_state),
            ("predicted_state_delta", predicted_state), ("predicted_visual_delta", predicted_visual),
        ):
            _finite(name, value)
        state_delta = _masked(state - previous_state, valid)
        visual_delta = _masked(self.visual_summary(context - previous_context), valid)
        state_error = _masked(state_delta - predicted_state, prediction_valid)
        visual_error = _masked(visual_delta - predicted_visual, prediction_valid)
        values = torch.cat((
            state, state_delta, state_error, visual_delta.flatten(1), visual_error.flatten(1),
            actions.detach().flatten(1), mask.to(state.dtype), prediction_valid[:, None].to(state.dtype),
        ), dim=-1)
        embedding = self.feedback_projection(values)
        return _masked(embedding, valid), valid

    def forward(
        self, context: Tensor, state: Tensor, noisy_actions: Tensor, time: Tensor,
        feedback: ExecutionFeedback | None,
    ) -> dict[str, Tensor]:
        embedding, valid = self.encode_feedback(context, state, feedback)
        batch = context.shape[0]
        if tuple(noisy_actions.shape) != (batch, self.config.action_horizon, 7):
            raise ValueError("noisy_actions must have shape [B,H,7]")
        if tuple(time.shape) != (batch,):
            raise ValueError("time must have shape [B]")
        _finite("noisy_actions", noisy_actions)
        _finite("time", time)
        if bool(((time < 0) | (time > 1)).any()):
            raise ValueError("Flow time must lie in [0,1]")
        time_features = torch.stack((time, torch.sin(math.pi * time), torch.cos(math.pi * time)), dim=-1)
        tokens = self.action_projection(noisy_actions) + self.position[None]
        tokens = tokens + (embedding + self.state_projection(state) + self.time_projection(time_features))[:, None]
        memory = self.context_projection(context)
        spatial_count = self.config.camera_count * self.config.spatial_tokens
        # 保留每个相机 raster 位置的真实变化；双相机均值仅用于辅助预测，
        # 不能成为执行反馈中唯一的历史视觉信息。
        spatial_delta = torch.zeros_like(context[:, :spatial_count])
        if feedback is not None:
            previous_spatial = _masked(feedback.previous_context[:, :spatial_count].detach(), valid)
            spatial_delta = _masked(context[:, :spatial_count] - previous_spatial, valid)
        spatial_memory = self.spatial_feedback_projection(spatial_delta) + self.spatial_feedback_position[None]
        memory = torch.cat((memory, _masked(spatial_memory, valid)), dim=1)
        for block in self.blocks:
            tokens = block(tokens, memory)
        gate_logits = self.gate(tokens).squeeze(-1)
        gate = gate_logits.sigmoid() * valid[:, None].to(gate_logits.dtype)
        residual = torch.tanh(self.velocity_output(tokens))
        residual = residual * self.velocity_limits.to(residual)
        return {"velocity_residual": residual * gate[:, :, None], "gate_logits": gate_logits,
                "gate": gate, "feedback_valid": valid}

    def forecast_execution(
        self, context: Tensor, state: Tensor, executed_actions: Tensor, executed_mask: Tensor
    ) -> dict[str, Tensor]:
        self._validate_current(context, state)
        actions = self._execution_actions(executed_actions, executed_mask, context.shape[0])
        latent = self.forecast_encoder(torch.cat((
            self.visual_summary(context).flatten(1), context.mean(dim=1), state,
            actions.flatten(1), executed_mask.to(state.dtype),
        ), dim=-1))
        # 预测时域固定为两步。只执行一步的中断段可用于历史反馈，但不能
        # 被当作完整两步未来标签；tracker 同样使该段 prediction_valid=False。
        valid = executed_mask.all(dim=1)
        return {
            "state_delta": _masked(self.future_state(latent), valid),
            "visual_delta": _masked(self.future_visual(latent).reshape(
                context.shape[0], 2, self.config.hidden_dim), valid),
        }

    @staticmethod
    def relative_improvement_gate_loss(logits: Tensor, target: Tensor, mask: Tensor) -> Tensor:
        """target 必须来自 parent/correction 同状态对照，不能是数据来源标签。"""
        if target.shape != logits.shape:
            raise ValueError("gate improvement target must match gate logits [B,H]")
        _bool_mask("gate_label_mask", mask, tuple(logits.shape))
        target = _masked(target.to(logits.dtype), mask)
        _finite("gate target", target)
        if bool(((target < 0) | (target > 1)).any()):
            raise ValueError("gate target must be a probability in [0,1]")
        terms = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
        return _masked(terms, mask).sum() / mask.sum().clamp_min(1)


class DuvlaV326Policy(DuvlaV21Policy):
    """保留父模型键/采样协议，只训练 execution_feedback。"""

    def __init__(self, config: DuvlaV326Config | None = None) -> None:
        config = DuvlaV326Config() if config is None else config
        super().__init__(config)
        for parameter in self.parameters():
            parameter.requires_grad_(False)
        self.execution_feedback = ExecutionFeedbackModule(config)
        self.train(self.training)

    def train(self, mode: bool = True) -> DuvlaV326Policy:
        super().train(mode)
        for name, module in self.named_children():
            module.train(mode if name == "execution_feedback" else False)
        return self

    def load_parent_state_dict(self, state: Mapping[str, Tensor]) -> None:
        expected = {name for name in self.state_dict() if not name.startswith("execution_feedback.")}
        if set(state) != expected:
            raise ValueError("parent state keys must match exactly; no partial parent loading")
        self.load_state_dict(state, strict=False)

    def forecast_execution(
        self, context: Tensor, state: Tensor, executed_actions: Tensor, executed_mask: Tensor
    ) -> dict[str, Tensor]:
        return self.execution_feedback.forecast_execution(context, state, executed_actions, executed_mask)

    def feedback_velocity_from_context(
        self, context: Tensor, state_token: Tensor, state: Tensor, noisy_actions: Tensor,
        time: Tensor, feedback: ExecutionFeedback | None = None,
    ) -> dict[str, Tensor]:
        if tuple(state_token.shape) != (context.shape[0], 1, self.config.hidden_dim):
            raise ValueError("state_token must have shape [B,1,D]")
        _finite("state_token", state_token)
        context, state, state_token = context.detach(), state.detach(), state_token.detach()
        result = self.execution_feedback(context, state, noisy_actions, time, feedback)
        with torch.no_grad():
            base = self.flow_expert(context, state_token, noisy_actions, time)
            if not isinstance(base, Tensor):
                base = base[0]
        # FP32主参数+BF16 autocast训练时，常量buffer不能把父BF16积分器
        # 悄悄提升为FP32；否则即使残差严格为零也会改变每步舍入后的父动作。
        velocity = base + result["velocity_residual"].to(base.dtype)
        return {**result, "parent_velocity": base, "velocity": velocity}

    def flow_loss_components(self, *args: object, **kwargs: object) -> dict[str, Tensor]:
        raise RuntimeError("V3.26 training must use flow_loss_components_from_context with explicit execution feedback")

    def flow_loss_components_from_context(
        self, context: Tensor, state_token: Tensor, state: Tensor, actions: Tensor,
        action_mask: Tensor, feedback: ExecutionFeedback | None = None, *,
        noise: Tensor | None = None, time: Tensor | None = None,
        previous_actions: Tensor | None = None,
        executed_actions: Tensor | None = None, executed_mask: Tensor | None = None,
        future_state_delta: Tensor | None = None, future_visual_delta: Tensor | None = None,
        gate_improvement_target: Tensor | None = None, gate_label_mask: Tensor | None = None,
        future_state_weight: float = 1.0, future_visual_weight: float = 1.0,
        gate_weight: float = 0.1,
    ) -> dict[str, Tensor]:
        batch = context.shape[0]
        if tuple(actions.shape) != (batch, self.config.action_horizon, 7):
            raise ValueError("actions must have shape [B,H,7]")
        _bool_mask("action_mask", action_mask, tuple(actions.shape[:2]))
        if not bool(action_mask.any()) and future_state_delta is None and future_visual_delta is None:
            raise ValueError("at least one supervised action or future outcome is required")
        # 完整 chunk 进入 Flow 的 self-attention，包括没有 loss 的尾部。
        # 不知道的尾部必须由调用方提供有限父 chunk；不能把 noise 置零。
        actions = actions.detach()
        _finite("complete target action chunk", actions)
        noise = torch.randn_like(actions) if noise is None else noise
        if noise.shape != actions.shape:
            raise ValueError("noise must match actions")
        _finite("complete Flow noise", noise)
        time = torch.distributions.Beta(1.5, 1.0).sample((batch,)).to(actions).clamp_(0.001, 0.999) if time is None else time
        if tuple(time.shape) != (batch,):
            raise ValueError("time must have shape [B]")
        blend = time[:, None, None]
        noisy = (1 - blend) * noise + blend * actions
        result = self.feedback_velocity_from_context(context, state_token, state, noisy, time, feedback)
        error = (result["velocity"] - (actions - noise)).square()
        weights = action_mask[:, :, None].expand_as(actions).to(actions.dtype).clone()
        weights[:, : self.config.replan_action_steps] *= self.config.flow_prefix_weight
        if self.config.flow_gripper_transition_weight > 1:
            if previous_actions is None or tuple(previous_actions.shape) != (batch, 7):
                raise ValueError("transition-weighted Flow requires previous_actions [B,7]")
            previous = torch.cat((previous_actions[:, None, -1], actions[:, :-1, -1]), dim=1)
            transition = (previous > 0) != (actions[:, :, -1] > 0)
            weights[:, :, -1] *= 1 + transition * (self.config.flow_gripper_transition_weight - 1)
        def mean(values: Tensor, mask: Tensor) -> Tensor:
            return (values * mask).sum() / mask.sum().clamp_min(1)
        flow = mean(error, weights)
        endpoint = noisy + (1 - blend) * result["velocity"]
        prefix = action_mask.clone()
        prefix[:, self.config.replan_action_steps:] = False
        arm_mask = prefix[:, :, None].expand(-1, -1, 6).to(actions.dtype)
        losses = {
            "flow": flow,
            "flow_arm": mean(error[:, :, :6], weights[:, :, :6]),
            "flow_gripper": mean(error[:, :, 6], weights[:, :, 6]),
            "endpoint_arm_l1": mean((endpoint[:, :, :6] - actions[:, :, :6]).abs(), arm_mask),
            "endpoint_gripper_l1": mean((endpoint[:, :, 6] - actions[:, :, 6]).abs(), prefix),
        }
        losses["total"] = flow + self.config.flow_endpoint_loss_weight * losses["endpoint_arm_l1"]
        if min(future_state_weight, future_visual_weight, gate_weight) < 0 or not all(
            math.isfinite(value) for value in (future_state_weight, future_visual_weight, gate_weight)
        ):
            raise ValueError("auxiliary loss weights must be finite and non-negative")
        future_values = (executed_actions, executed_mask, future_state_delta, future_visual_delta)
        if any(value is not None for value in future_values):
            if not all(value is not None for value in future_values):
                raise ValueError("future supervision requires execution actions/mask and both delta targets")
            prediction = self.forecast_execution(context.detach(), state.detach(), executed_actions, executed_mask)
            future_valid = executed_mask.all(dim=1)
            for name, target, coefficient in (
                ("state_delta", future_state_delta, future_state_weight),
                ("visual_delta", future_visual_delta, future_visual_weight),
            ):
                if target.shape != prediction[name].shape:
                    raise ValueError(f"future {name} target has an invalid shape")
                target = _masked(target.detach(), future_valid)
                _finite(name, target)
                terms = F.smooth_l1_loss(prediction[name], target, reduction="none").flatten(1).mean(dim=1)
                value = _masked(terms, future_valid).sum() / future_valid.sum().clamp_min(1)
                losses[f"future_{name}"] = value
                losses["total"] = losses["total"] + coefficient * value
        if gate_improvement_target is not None or gate_label_mask is not None:
            if gate_improvement_target is None or gate_label_mask is None:
                raise ValueError("gate supervision requires both relative-improvement targets and mask")
            _bool_mask("gate_label_mask", gate_label_mask, tuple(action_mask.shape))
            mask = gate_label_mask & action_mask & result["feedback_valid"][:, None]
            gate_loss = self.execution_feedback.relative_improvement_gate_loss(
                result["gate_logits"], gate_improvement_target, mask)
            losses["gate"] = gate_loss
            losses["total"] = losses["total"] + gate_weight * gate_loss
        return losses

    @torch.no_grad()
    def sample_actions(
        self, visual: Tensor, semantic: Tensor, state: Tensor, *,
        execution_feedback: ExecutionFeedback | None = None, **options: object,
    ) -> Tensor:
        if execution_feedback is None:
            return super().sample_actions(visual, semantic, state, **options)
        encode_names = {
            "auxiliary_visual", "history_visual", "history_auxiliary_visual", "history_semantic",
            "history_states", "previous_action", "history_previous_actions", "layer14_only",
        }
        allowed = encode_names | {
            "flow_samples", "flow_steps", "noise", "apply_direct", "apply_instruction",
            "apply_gripper_event", "previous_gripper_closed", "apply_recovery",
        }
        if unknown := set(options) - allowed:
            raise TypeError(f"unknown policy input: {sorted(unknown)}")
        context, state_token, language, _weights = self._encode_context(
            visual, semantic, state, **{key: value for key, value in options.items() if key in encode_names})
        samples = self.config.flow_samples if options.get("flow_samples") is None else options["flow_samples"]
        steps = self.config.flow_steps if options.get("flow_steps") is None else options["flow_steps"]
        base = self.sample_feedback_from_context(
            context, state_token, state, execution_feedback, samples=samples,
            steps=steps, noise=options.get("noise"))
        summary = context.mean(dim=1)
        if options.get("apply_direct", True):
            base = base + self.direct_residual(base, summary, state_token[:, 0], language)
        if options.get("apply_instruction", True):
            base = base + self.instruction_residual(base, summary, state_token[:, 0], language)
        return base

    @torch.no_grad()
    def sample_feedback_from_context(
        self, context: Tensor, state_token: Tensor, state: Tensor,
        execution_feedback: ExecutionFeedback, *, samples: int = 5,
        steps: int = 10, noise: Tensor | None = None,
    ) -> Tensor:
        """缓存验证与正式推理共享相同Flow积分，不应用非Flow旧头。"""
        if not isinstance(samples, int) or not isinstance(steps, int) or min(samples, steps) <= 0:
            raise ValueError("flow samples and integration steps must be positive integers")
        batch = context.shape[0]
        shape = (batch, samples, self.config.action_horizon, 7)
        if noise is None:
            actions = torch.randn(shape, device=context.device, dtype=next(self.parameters()).dtype)
        else:
            if not isinstance(noise, Tensor) or tuple(noise.shape) != shape:
                raise ValueError(f"noise must have shape {shape}")
            actions = noise.to(device=context.device, dtype=next(self.parameters()).dtype)
        def repeat(values: Tensor) -> Tensor:
            return values[:, None].expand(-1, samples, *values.shape[1:]).flatten(0, 1)
        repeated_feedback = ExecutionFeedback(**{
            name: repeat(getattr(execution_feedback, name))
            for name in execution_feedback.__dataclass_fields__
        })
        flat = actions.flatten(0, 1)
        repeated_context, repeated_state_token, repeated_state = repeat(context), repeat(state_token), repeat(state)
        for index in range(steps):
            time = torch.full((batch * samples,), index / steps, device=flat.device, dtype=flat.dtype)
            result = self.feedback_velocity_from_context(
                repeated_context, repeated_state_token, repeated_state, flat, time, repeated_feedback)
            flat = flat + result["velocity"] / steps
        return flat.reshape(shape).median(dim=1).values
