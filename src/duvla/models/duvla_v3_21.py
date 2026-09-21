"""Duvla V3.21 parallel continuous action-query policy.

V3.21 is a fresh action generator, not a residual on a V1/V2 checkpoint.  It
keeps the frozen, cached multi-layer Qwen representation used by V2.6, but
replaces stochastic Flow integration with eight learned action queries that
predict an entire continuous chunk in parallel.  The binary gripper command is
modelled explicitly instead of being diffused as a continuous coordinate.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import Tensor, nn

from duvla.models.duvla_v2_1 import DuvlaV21Config, MultiLayerQwenFusion


@dataclass(frozen=True)
class DuvlaV321Config:
    feature_dim: int = 2048
    feature_layers: int = 4
    camera_count: int = 2
    spatial_tokens: int = 64
    semantic_tokens: int = 1
    state_dim: int = 8
    action_dim: int = 7
    action_horizon: int = 8
    hidden_dim: int = 768
    decoder_layers: int = 12
    decoder_heads: int = 12
    history_length: int = 4
    history_stride: int = 2
    replan_action_steps: int = 2
    dropout: float = 0.0
    arm_huber_beta: float = 0.1
    executed_prefix_weight: float = 4.0
    gripper_loss_weight: float = 0.2
    gripper_transition_weight: float = 8.0
    transition_aux_loss_weight: float = 0.05
    gripper_open_value: float = -1.0
    gripper_close_value: float = 1.0
    gripper_probability_threshold: float = 0.5

    def __post_init__(self) -> None:
        dimensions = (
            self.feature_dim,
            self.feature_layers,
            self.camera_count,
            self.spatial_tokens,
            self.semantic_tokens,
            self.state_dim,
            self.action_dim,
            self.action_horizon,
            self.hidden_dim,
            self.decoder_layers,
            self.decoder_heads,
            self.history_length,
            self.history_stride,
            self.replan_action_steps,
        )
        if any(value <= 0 for value in dimensions):
            raise ValueError("all V3.21 dimensions and counts must be positive")
        if self.semantic_tokens != 1:
            raise ValueError("V3.21 requires one cached semantic token per Qwen layer")
        if self.action_dim != 7:
            raise ValueError("V3.21 requires the LIBERO 7-D action contract")
        if self.hidden_dim % self.decoder_heads:
            raise ValueError("hidden_dim must be divisible by decoder_heads")
        if self.replan_action_steps > self.action_horizon:
            raise ValueError("replan_action_steps cannot exceed action_horizon")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if self.arm_huber_beta <= 0.0:
            raise ValueError("arm_huber_beta must be positive")
        if self.executed_prefix_weight < 1.0 or self.gripper_transition_weight < 1.0:
            raise ValueError("prefix and transition weights must be at least one")
        if self.gripper_loss_weight < 0.0 or self.transition_aux_loss_weight < 0.0:
            raise ValueError("loss weights must be non-negative")
        if not 0.0 < self.gripper_probability_threshold < 1.0:
            raise ValueError("gripper_probability_threshold must be in (0, 1)")

    @property
    def highres_layer14_visual(self) -> bool:
        return True

    @property
    def visual_feature_layers(self) -> int:
        return 1

    @property
    def context_tokens(self) -> int:
        return (
            self.camera_count * self.spatial_tokens
            + self.feature_layers
            + 1
            + self.history_length
            - 1
        )

    @property
    def action_history_conditioning(self) -> bool:
        return False

    @property
    def gripper_action_conditioning(self) -> bool:
        return False

    @property
    def gripper_control_mode(self) -> str:
        return "absolute"

    @property
    def unified_flow_gripper(self) -> bool:
        # This compatibility property tells the evaluator not to apply a
        # legacy post-hoc gripper FSM. V3.21 emits its own deterministic state.
        return True

    @property
    def flow_samples(self) -> int:
        return 1

    @property
    def flow_steps(self) -> int:
        return 1


class _ActionQueryBlock(nn.Module):
    def __init__(self, hidden_dim: int, heads: int, dropout: float) -> None:
        super().__init__()
        self.self_norm = nn.RMSNorm(hidden_dim, eps=1e-6)
        self.self_attention = nn.MultiheadAttention(
            hidden_dim, heads, dropout=dropout, bias=False, batch_first=True
        )
        self.cross_norm = nn.RMSNorm(hidden_dim, eps=1e-6)
        self.cross_attention = nn.MultiheadAttention(
            hidden_dim, heads, dropout=dropout, bias=False, batch_first=True
        )
        self.ffn_norm = nn.RMSNorm(hidden_dim, eps=1e-6)
        intermediate = 256 * math.ceil((8 * hidden_dim / 3) / 256)
        self.gate = nn.Linear(hidden_dim, intermediate, bias=False)
        self.up = nn.Linear(hidden_dim, intermediate, bias=False)
        self.down = nn.Linear(intermediate, hidden_dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, queries: Tensor, memory: Tensor) -> Tensor:
        normalized = self.self_norm(queries)
        attended, _ = self.self_attention(
            normalized, normalized, normalized, need_weights=False
        )
        queries = queries + self.dropout(attended)
        normalized = self.cross_norm(queries)
        attended, _ = self.cross_attention(
            normalized, memory, memory, need_weights=False
        )
        queries = queries + self.dropout(attended)
        normalized = self.ffn_norm(queries)
        hidden = torch.nn.functional.silu(self.gate(normalized)) * self.up(normalized)
        return queries + self.down(self.dropout(hidden))


class DuvlaV321Policy(nn.Module):
    """Unified natural-language policy with parallel continuous action queries."""

    def __init__(self, config: DuvlaV321Config | None = None) -> None:
        super().__init__()
        self.config = config or DuvlaV321Config()
        cfg = self.config
        fusion_config = DuvlaV21Config(
            feature_dim=cfg.feature_dim,
            feature_layers=cfg.feature_layers,
            camera_count=cfg.camera_count,
            spatial_tokens=cfg.spatial_tokens,
            semantic_tokens=cfg.semantic_tokens,
            state_dim=cfg.state_dim,
            action_dim=cfg.action_dim,
            action_horizon=cfg.action_horizon,
            hidden_dim=cfg.hidden_dim,
            expert_heads=cfg.decoder_heads,
            history_length=cfg.history_length,
            history_stride=cfg.history_stride,
            replan_action_steps=cfg.replan_action_steps,
            highres_layer14_visual=True,
            dense_interleaved_flow=True,
            context_layers=0,
            unified_flow_gripper=True,
        )
        self.fusion = MultiLayerQwenFusion(fusion_config)
        self.history_position = nn.Parameter(
            torch.randn(cfg.history_length - 1, cfg.hidden_dim) * 0.02
        )
        self.action_queries = nn.Parameter(
            torch.randn(cfg.action_horizon, cfg.hidden_dim) * 0.02
        )
        self.decoder = nn.ModuleList(
            _ActionQueryBlock(cfg.hidden_dim, cfg.decoder_heads, cfg.dropout)
            for _ in range(cfg.decoder_layers)
        )
        self.output_norm = nn.RMSNorm(cfg.hidden_dim, eps=1e-6)
        self.arm_head = nn.Linear(cfg.hidden_dim, cfg.action_dim - 1)
        self.gripper_head = nn.Linear(cfg.hidden_dim, 1)
        self.transition_head = nn.Linear(cfg.hidden_dim, 1)

    def _encode_memory(
        self,
        visual: Tensor,
        semantic: Tensor,
        state: Tensor,
        *,
        history_visual: Tensor | None,
        history_semantic: Tensor | None,
        history_states: Tensor | None,
        layer14_only: bool,
    ) -> Tensor:
        fused_visual, semantic_tokens, state_token, _ = self.fusion(
            visual, semantic, state, layer14_only=layer14_only
        )
        current_visual = fused_visual.flatten(1, 2)
        current_semantic = semantic_tokens
        history_count = self.config.history_length - 1
        if history_visual is None or history_semantic is None or history_states is None:
            summary = (
                current_visual.mean(dim=1)
                + current_semantic.mean(dim=1)
                + state_token
            )
            history_tokens = summary[:, None].expand(-1, history_count, -1)
        else:
            batch = visual.shape[0]
            expected = (
                history_visual.ndim == 6
                and history_visual.shape[1] == history_count
                and history_semantic.ndim == 5
                and history_semantic.shape[1] == history_count
                and tuple(history_states.shape[:2]) == (batch, history_count)
            )
            if not expected:
                raise ValueError("history tensors do not match the V3.21 contract")
            hist_visual, hist_semantic, hist_state, _ = self.fusion(
                history_visual.flatten(0, 1),
                history_semantic.flatten(0, 1),
                history_states.flatten(0, 1),
                layer14_only=layer14_only,
            )
            history_tokens = (
                hist_visual.mean(dim=(1, 2))
                + hist_semantic.mean(dim=1)
                + hist_state
            ).reshape(batch, history_count, self.config.hidden_dim)
        history_tokens = history_tokens + self.history_position[None]
        return torch.cat(
            (current_visual, current_semantic, state_token[:, None], history_tokens),
            dim=1,
        )

    def forward(
        self,
        visual: Tensor,
        semantic: Tensor,
        state: Tensor,
        *,
        history_visual: Tensor | None = None,
        history_semantic: Tensor | None = None,
        history_states: Tensor | None = None,
        layer14_only: bool = False,
    ) -> tuple[Tensor, Tensor, Tensor]:
        memory = self._encode_memory(
            visual,
            semantic,
            state,
            history_visual=history_visual,
            history_semantic=history_semantic,
            history_states=history_states,
            layer14_only=layer14_only,
        )
        return self._decode_memory(memory)

    def _decode_memory(self, memory: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        queries = self.action_queries[None].expand(memory.shape[0], -1, -1)
        for block in self.decoder:
            queries = block(queries, memory)
        queries = self.output_norm(queries)
        return (
            self.arm_head(queries),
            self.gripper_head(queries).squeeze(-1),
            self.transition_head(queries).squeeze(-1),
        )

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
        del history_previous_actions, parent_actions, noise, time
        if stage != "flow":
            raise ValueError("V3.21 has one end-to-end parallel_action_query stage")
        expected = (self.config.action_horizon, self.config.action_dim)
        if tuple(target_actions.shape[1:]) != expected:
            raise ValueError("target_actions do not match the V3.21 action contract")
        if tuple(valid_mask.shape) != tuple(target_actions.shape[:2]):
            raise ValueError("valid_mask must have shape [batch, horizon]")
        arm, gripper_logits, transition_logits = self(
            visual,
            semantic,
            state,
            history_visual=history_visual,
            history_semantic=history_semantic,
            history_states=history_states,
            layer14_only=layer14_only,
        )
        batch = target_actions.shape[0]
        if sample_weights is None:
            sample_weights = torch.ones(batch, device=state.device, dtype=state.dtype)
        elif tuple(sample_weights.shape) != (batch,):
            raise ValueError("sample_weights must have shape [batch]")
        sample_weights = sample_weights.to(device=state.device, dtype=state.dtype)
        if bool((sample_weights < 0).any()) or float(sample_weights.sum()) <= 0.0:
            raise ValueError("sample_weights must be non-negative with a positive sum")

        valid = valid_mask.to(dtype=state.dtype)
        step_weight = valid.clone()
        step_weight[:, : self.config.replan_action_steps] *= self.config.executed_prefix_weight
        arm_error = torch.nn.functional.smooth_l1_loss(
            arm,
            target_actions[:, :, :-1],
            beta=self.config.arm_huber_beta,
            reduction="none",
        ).mean(dim=-1)
        arm_per_sample = (arm_error * step_weight).sum(dim=1) / step_weight.sum(
            dim=1
        ).clamp_min(1.0)

        closed = target_actions[:, :, -1].gt(0.0)
        if previous_actions is None or tuple(previous_actions.shape) != (
            batch,
            self.config.action_dim,
        ):
            raise ValueError("V3.21 requires previous_actions [batch, action_dim]")
        previous_closed = previous_actions[:, -1].gt(0.0)
        transitions = torch.cat(
            (closed[:, :1].ne(previous_closed[:, None]), closed[:, 1:].ne(closed[:, :-1])),
            dim=1,
        )
        event_weight = valid * (
            1.0
            + transitions.to(state.dtype) * (self.config.gripper_transition_weight - 1.0)
        )
        gripper_error = torch.nn.functional.binary_cross_entropy_with_logits(
            gripper_logits, closed.to(gripper_logits.dtype), reduction="none"
        )
        gripper_per_sample = (gripper_error * event_weight).sum(dim=1) / event_weight.sum(
            dim=1
        ).clamp_min(1.0)
        transition_error = torch.nn.functional.binary_cross_entropy_with_logits(
            transition_logits, transitions.to(transition_logits.dtype), reduction="none"
        )
        transition_per_sample = (transition_error * valid).sum(dim=1) / valid.sum(
            dim=1
        ).clamp_min(1.0)

        def weighted_mean(values: Tensor) -> Tensor:
            return (values * sample_weights).mean()

        return {
            "arm_huber": weighted_mean(arm_per_sample),
            "gripper_bce": self.config.gripper_loss_weight
            * weighted_mean(gripper_per_sample),
            "transition_bce": self.config.transition_aux_loss_weight
            * weighted_mean(transition_per_sample),
        }

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
        del previous_action, history_previous_actions, flow_samples, flow_steps
        del apply_direct, apply_instruction, apply_gripper_event, previous_gripper_closed, noise
        arm, gripper_logits, _ = self(
            visual,
            semantic,
            state,
            history_visual=history_visual,
            history_semantic=history_semantic,
            history_states=history_states,
            layer14_only=layer14_only,
        )
        closed = gripper_logits.sigmoid().ge(self.config.gripper_probability_threshold)
        open_value = torch.as_tensor(
            self.config.gripper_open_value, device=arm.device, dtype=arm.dtype
        )
        close_value = torch.as_tensor(
            self.config.gripper_close_value, device=arm.device, dtype=arm.dtype
        )
        gripper = torch.where(closed, close_value, open_value)
        return torch.cat((arm, gripper[:, :, None]), dim=-1)

    def parameter_counts(self) -> dict[str, int]:
        total = sum(parameter.numel() for parameter in self.parameters())
        trainable = sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)
        return {"robot_policy": total, "trainable": trainable}
