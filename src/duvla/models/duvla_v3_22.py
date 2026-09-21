"""Duvla V3.22 Flow residual around a frozen parallel-action prior."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from duvla.models.duvla_v3_21 import DuvlaV321Config, DuvlaV321Policy
from duvla.models.flow_matching_vla import (
    FlowMatchingActionExpert,
    FlowMatchingVLAConfig,
)


@dataclass(frozen=True)
class DuvlaV322Config:
    prior: DuvlaV321Config
    residual_hidden_dim: int = 768
    residual_layers: int = 12
    residual_heads: int = 12
    flow_steps: int = 10
    flow_samples: int = 5
    endpoint_loss_weight: float = 0.02
    dropout: float = 0.0

    def __post_init__(self) -> None:
        if min(
            self.residual_hidden_dim,
            self.residual_layers,
            self.residual_heads,
            self.flow_steps,
            self.flow_samples,
        ) <= 0:
            raise ValueError("V3.22 dimensions and Flow counts must be positive")
        if self.residual_hidden_dim != self.prior.hidden_dim:
            raise ValueError("V3.22 prior and residual hidden dimensions must match")
        if self.residual_hidden_dim % self.residual_heads:
            raise ValueError("residual_hidden_dim must divide residual_heads")
        if self.endpoint_loss_weight < 0.0:
            raise ValueError("endpoint_loss_weight must be non-negative")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")

    @property
    def feature_dim(self) -> int:
        return self.prior.feature_dim

    @property
    def camera_count(self) -> int:
        return self.prior.camera_count

    @property
    def spatial_tokens(self) -> int:
        return self.prior.spatial_tokens

    @property
    def state_dim(self) -> int:
        return self.prior.state_dim

    @property
    def action_dim(self) -> int:
        return self.prior.action_dim

    @property
    def action_horizon(self) -> int:
        return self.prior.action_horizon

    @property
    def hidden_dim(self) -> int:
        return self.prior.hidden_dim

    @property
    def history_length(self) -> int:
        return self.prior.history_length

    @property
    def history_stride(self) -> int:
        return self.prior.history_stride

    @property
    def replan_action_steps(self) -> int:
        return self.prior.replan_action_steps

    @property
    def highres_layer14_visual(self) -> bool:
        return True

    @property
    def visual_feature_layers(self) -> int:
        return 1

    @property
    def context_tokens(self) -> int:
        return self.prior.context_tokens

    @property
    def action_history_conditioning(self) -> bool:
        return False

    @property
    def gripper_action_conditioning(self) -> bool:
        return False

    @property
    def unified_flow_gripper(self) -> bool:
        return True

    @property
    def gripper_control_mode(self) -> str:
        return "absolute"


class DuvlaV322Policy(nn.Module):
    """Frozen deterministic prior plus a multimodal six-dimensional Flow residual."""

    def __init__(self, config: DuvlaV322Config) -> None:
        super().__init__()
        self.config = config
        self.prior = DuvlaV321Policy(config.prior)
        self.prior.requires_grad_(False)
        flow_config = FlowMatchingVLAConfig(
            state_dim=config.state_dim,
            action_dim=config.action_dim - 1,
            vlm_feature_dim=config.feature_dim,
            hidden_dim=config.residual_hidden_dim,
            action_horizon=config.action_horizon,
            max_context_tokens=config.context_tokens,
            expert_layers=config.residual_layers,
            expert_heads=config.residual_heads,
            dropout=config.dropout,
            num_flow_steps=config.flow_steps,
            self_attn_every_n_layers=2,
            time_sampling="beta_1.5_1.0",
            architecture_variant="smol_aligned",
            attention_pattern="alternating",
            ffn_type="swiglu",
            norm_type="rms_norm",
            attention_bias=False,
            state_projection_layers=1,
            context_layout="merged",
            qwen_context_mode="ordered",
        )
        self.residual_flow = FlowMatchingActionExpert(flow_config)

    def _prior_and_context(
        self,
        visual: Tensor,
        semantic: Tensor,
        state: Tensor,
        *,
        history_visual: Tensor | None,
        history_semantic: Tensor | None,
        history_states: Tensor | None,
        layer14_only: bool,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        with torch.no_grad():
            memory = self.prior._encode_memory(
                visual,
                semantic,
                state,
                history_visual=history_visual,
                history_semantic=history_semantic,
                history_states=history_states,
                layer14_only=layer14_only,
            )
            arm, gripper_logits, _ = self.prior._decode_memory(memory)
        state_index = self.config.camera_count * self.config.spatial_tokens + self.config.prior.feature_layers
        state_token = memory[:, state_index : state_index + 1]
        return arm, gripper_logits, memory, state_token

    def flow_loss_components(
        self,
        visual: Tensor,
        semantic: Tensor,
        state: Tensor,
        target_actions: Tensor,
        valid_mask: Tensor,
        *,
        history_visual: Tensor | None = None,
        history_semantic: Tensor | None = None,
        history_states: Tensor | None = None,
        previous_actions: Tensor | None = None,
        history_previous_actions: Tensor | None = None,
        layer14_only: bool = False,
        parent_actions: Tensor | None = None,
        stage: str = "flow",
        noise: Tensor | None = None,
        time: Tensor | None = None,
        sample_weights: Tensor | None = None,
    ) -> dict[str, Tensor]:
        del previous_actions, history_previous_actions, parent_actions
        if stage != "flow":
            raise ValueError("V3.22 has one residual_flow stage")
        if tuple(target_actions.shape[1:]) != (
            self.config.action_horizon,
            self.config.action_dim,
        ) or tuple(valid_mask.shape) != tuple(target_actions.shape[:2]):
            raise ValueError("target_actions or valid_mask violates the V3.22 contract")
        prior_arm, _gripper, context, state_token = self._prior_and_context(
            visual,
            semantic,
            state,
            history_visual=history_visual,
            history_semantic=history_semantic,
            history_states=history_states,
            layer14_only=layer14_only,
        )
        target_residual = target_actions[:, :, :-1] - prior_arm
        if noise is None:
            noise = torch.randn_like(target_residual)
        elif noise.shape[-1] == self.config.action_dim:
            noise = noise[:, :, :-1]
        if tuple(noise.shape) != tuple(target_residual.shape):
            raise ValueError("V3.22 loss noise must match the six-dimensional arm residual")
        if time is None:
            time = torch.distributions.Beta(1.5, 1.0).sample(
                (target_actions.shape[0],)
            ).to(target_actions.device).clamp_(0.001, 0.999)
        time_view = time[:, None, None].to(target_actions.dtype)
        noisy = (1.0 - time_view) * noise + time_view * target_residual
        predicted_velocity = self.residual_flow(context, state_token, noisy, time)
        if not isinstance(predicted_velocity, Tensor):
            predicted_velocity = predicted_velocity[0]
        target_velocity = target_residual - noise
        weights = valid_mask.to(target_actions.dtype)[:, :, None].expand_as(target_residual).clone()
        weights[:, : self.config.replan_action_steps] *= self.config.prior.executed_prefix_weight
        denominator = weights.sum(dim=(1, 2)).clamp_min(1.0)
        flow_per_sample = (
            (predicted_velocity - target_velocity).square() * weights
        ).sum(dim=(1, 2)) / denominator
        endpoint = noisy + (1.0 - time_view) * predicted_velocity
        prefix_weights = weights[:, : self.config.replan_action_steps]
        endpoint_per_sample = (
            (endpoint[:, : self.config.replan_action_steps] - target_residual[:, : self.config.replan_action_steps]).abs()
            * prefix_weights
        ).sum(dim=(1, 2)) / prefix_weights.sum(dim=(1, 2)).clamp_min(1.0)
        if sample_weights is None:
            sample_weights = torch.ones_like(flow_per_sample)
        elif tuple(sample_weights.shape) != tuple(flow_per_sample.shape):
            raise ValueError("sample_weights must have shape [batch]")
        sample_weights = sample_weights.to(flow_per_sample)
        return {
            "residual_flow": (flow_per_sample * sample_weights).mean(),
            "residual_endpoint": self.config.endpoint_loss_weight
            * (endpoint_per_sample * sample_weights).mean(),
        }

    def _sample_residuals(
        self,
        context: Tensor,
        state_token: Tensor,
        *,
        samples: int,
        steps: int,
        noise: Tensor | None,
    ) -> Tensor:
        batch = context.shape[0]
        expected = (
            batch,
            samples,
            self.config.action_horizon,
            self.config.action_dim - 1,
        )
        if noise is None:
            values = torch.randn(expected, device=context.device, dtype=context.dtype)
        else:
            if noise.shape[-1] == self.config.action_dim:
                noise = noise[:, :, :, :-1]
            if tuple(noise.shape) != expected:
                raise ValueError(f"V3.22 inference noise must have shape {expected}")
            values = noise.to(device=context.device, dtype=context.dtype)
        flat = values.flatten(0, 1)
        repeated_context = context[:, None].expand(-1, samples, -1, -1).flatten(0, 1)
        repeated_state = state_token[:, None].expand(-1, samples, -1, -1).flatten(0, 1)
        for index in range(steps):
            time = torch.full(
                (batch * samples,),
                index / steps,
                device=context.device,
                dtype=context.dtype,
            )
            velocity = self.residual_flow(repeated_context, repeated_state, flat, time)
            if not isinstance(velocity, Tensor):
                velocity = velocity[0]
            flat = flat + velocity / steps
        return flat.reshape(expected)

    @torch.no_grad()
    def sample_actions(
        self,
        visual: Tensor,
        semantic: Tensor,
        state: Tensor,
        *,
        history_visual: Tensor | None = None,
        history_semantic: Tensor | None = None,
        history_states: Tensor | None = None,
        previous_action: Tensor | None = None,
        history_previous_actions: Tensor | None = None,
        layer14_only: bool = False,
        flow_samples: int | None = None,
        flow_steps: int | None = None,
        apply_direct: bool = False,
        apply_instruction: bool = False,
        apply_gripper_event: bool = False,
        previous_gripper_closed: Tensor | None = None,
        noise: Tensor | None = None,
    ) -> Tensor:
        del previous_action, history_previous_actions, apply_direct, apply_instruction
        del apply_gripper_event, previous_gripper_closed
        prior_arm, gripper_logits, context, state_token = self._prior_and_context(
            visual,
            semantic,
            state,
            history_visual=history_visual,
            history_semantic=history_semantic,
            history_states=history_states,
            layer14_only=layer14_only,
        )
        samples = self.config.flow_samples if flow_samples is None else flow_samples
        steps = self.config.flow_steps if flow_steps is None else flow_steps
        residual = self._sample_residuals(
            context, state_token, samples=samples, steps=steps, noise=noise
        ).median(dim=1).values
        arm = prior_arm + residual
        closed = gripper_logits.sigmoid().ge(
            self.config.prior.gripper_probability_threshold
        )
        open_value = torch.as_tensor(
            self.config.prior.gripper_open_value, device=arm.device, dtype=arm.dtype
        )
        close_value = torch.as_tensor(
            self.config.prior.gripper_close_value, device=arm.device, dtype=arm.dtype
        )
        gripper = torch.where(closed, close_value, open_value)
        return torch.cat((arm, gripper[:, :, None]), dim=-1)

    def parameter_counts(self) -> dict[str, int]:
        total = sum(parameter.numel() for parameter in self.parameters())
        trainable = sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)
        return {"robot_policy": total, "trainable": trainable}
