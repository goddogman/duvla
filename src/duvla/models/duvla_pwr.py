"""Physical latent planning modules for Duvla V3.1 (DuVLA-PWR)."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from duvla.models.flow_matching_vla import (
    FlowMatchingActionExpert,
    FlowMatchingVLAConfig,
)
from duvla.models.duvla_v2_1 import DuvlaV21Config, DuvlaV21Policy


@dataclass(frozen=True)
class PWRPlannerConfig:
    feature_dim: int = 2048
    semantic_layers: int = 4
    camera_count: int = 2
    spatial_tokens: int = 64
    state_dim: int = 8
    action_dim: int = 7
    action_horizon: int = 8
    future_offsets: tuple[int, ...] = (2, 4, 8)
    hidden_dim: int = 384
    transformer_layers: int = 2
    attention_heads: int = 6
    future_spatial_tokens: int = 16
    future_dim: int = 128
    history_length: int = 4
    history_stride: int = 2
    dropout: float = 0.0
    future_residual_prediction: bool = False

    def __post_init__(self) -> None:
        positive = (
            self.feature_dim,
            self.semantic_layers,
            self.camera_count,
            self.spatial_tokens,
            self.state_dim,
            self.action_dim,
            self.action_horizon,
            self.hidden_dim,
            self.transformer_layers,
            self.attention_heads,
            self.future_spatial_tokens,
            self.future_dim,
            self.history_length,
            self.history_stride,
        )
        if any(value <= 0 for value in positive):
            raise ValueError("all PWR planner dimensions must be positive")
        if not self.future_offsets or any(value <= 0 for value in self.future_offsets):
            raise ValueError("future_offsets must be non-empty and positive")
        if tuple(sorted(self.future_offsets)) != self.future_offsets:
            raise ValueError("future_offsets must be strictly ordered")
        if max(self.future_offsets) > self.action_horizon:
            raise ValueError("future_offsets cannot exceed action_horizon")
        if self.hidden_dim % self.attention_heads:
            raise ValueError("hidden_dim must be divisible by attention_heads")
        side = round(self.spatial_tokens**0.5)
        future_side = round(self.future_spatial_tokens**0.5)
        if side * side != self.spatial_tokens or future_side * future_side != self.future_spatial_tokens:
            raise ValueError("spatial token counts must be square grids")
        if self.spatial_tokens % self.future_spatial_tokens:
            raise ValueError("future spatial grid must evenly pool the input grid")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")


@dataclass(frozen=True)
class PWRPlannerOutput:
    physical_latent: Tensor
    context_memory: Tensor
    future_visual: Tensor
    future_state_delta: Tensor
    action_summary: Tensor
    interaction_logits: Tensor


def compressed_future_visual(values: Tensor, config: PWRPlannerConfig) -> Tensor:
    """Deterministically compress frozen Qwen spatial targets without a teacher."""

    if values.ndim == 5:
        values = values[:, None]
    expected = (
        config.camera_count,
        config.spatial_tokens,
        config.feature_dim,
    )
    if values.ndim != 6 or tuple(values.shape[-3:]) != expected:
        raise ValueError(
            "future visual must have shape [batch, horizons, layers, cameras, tokens, width] "
            "with a singleton visual layer"
        )
    if values.shape[2] != 1:
        raise ValueError("PWR v3.1 initial cache requires one layer-14 spatial stream")
    batch, horizons = values.shape[:2]
    side = round(config.spatial_tokens**0.5)
    future_side = round(config.future_spatial_tokens**0.5)
    spatial = values[:, :, 0].float().reshape(
        batch * horizons * config.camera_count,
        side,
        side,
        config.feature_dim,
    ).permute(0, 3, 1, 2)
    pooled = F.adaptive_avg_pool2d(spatial, (future_side, future_side))
    pooled = pooled.permute(0, 2, 3, 1).reshape(-1, config.feature_dim)
    compressed = F.adaptive_avg_pool1d(
        pooled[:, None], config.future_dim
    ).squeeze(1)
    compressed = F.layer_norm(compressed, (config.future_dim,))
    return compressed.reshape(
        batch,
        horizons,
        config.camera_count,
        config.future_spatial_tokens,
        config.future_dim,
    )


def interaction_event_targets(
    actions: Tensor,
    previous_actions: Tensor,
    offsets: tuple[int, ...],
) -> Tensor:
    """Return HOLD/CLOSE/OPEN events at each supervised physical horizon."""

    if actions.ndim != 3 or previous_actions.ndim != 2:
        raise ValueError("actions and previous_actions must be rank 3 and 2")
    if actions.shape[0] != previous_actions.shape[0] or actions.shape[2] != previous_actions.shape[1]:
        raise ValueError("action batch/dimension contracts differ")
    targets: list[Tensor] = []
    for offset in offsets:
        if not 0 < offset <= actions.shape[1]:
            raise ValueError("interaction offset is outside action horizon")
        current_closed = actions[:, offset - 1, -1] > 0.0
        prior_closed = (
            previous_actions[:, -1] > 0.0
            if offset == 1
            else actions[:, offset - 2, -1] > 0.0
        )
        event = torch.zeros_like(current_closed, dtype=torch.long)
        event = torch.where(current_closed & ~prior_closed, torch.ones_like(event), event)
        event = torch.where(~current_closed & prior_closed, torch.full_like(event, 2), event)
        targets.append(event)
    return torch.stack(targets, dim=1)


class PhysicalLatentPlanner(nn.Module):
    """Predict supervised multi-horizon physical plans from frozen Qwen features."""

    def __init__(self, config: PWRPlannerConfig) -> None:
        super().__init__()
        self.config = config
        self.visual_projection = nn.Sequential(
            nn.RMSNorm(config.feature_dim, eps=1e-6),
            nn.Linear(config.feature_dim, config.hidden_dim, bias=False),
        )
        self.semantic_projection = nn.Sequential(
            nn.RMSNorm(config.feature_dim, eps=1e-6),
            nn.Linear(config.feature_dim, config.hidden_dim, bias=False),
        )
        self.state_projection = nn.Sequential(
            nn.RMSNorm(config.state_dim, eps=1e-6),
            nn.Linear(config.state_dim, config.hidden_dim),
        )
        self.history_projection = nn.Sequential(
            nn.RMSNorm(config.hidden_dim * 3, eps=1e-6),
            nn.Linear(config.hidden_dim * 3, config.hidden_dim),
        )
        self.camera_embedding = nn.Parameter(
            torch.randn(config.camera_count, config.hidden_dim) * 0.02
        )
        self.spatial_embedding = nn.Parameter(
            torch.randn(config.spatial_tokens, config.hidden_dim) * 0.02
        )
        self.semantic_embedding = nn.Parameter(
            torch.randn(config.semantic_layers, config.hidden_dim) * 0.02
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_dim,
            nhead=config.attention_heads,
            dim_feedforward=config.hidden_dim * 4,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.context_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=config.transformer_layers
        )
        self.plan_queries = nn.Parameter(
            torch.randn(len(config.future_offsets), config.hidden_dim) * 0.02
        )
        self.plan_attention = nn.MultiheadAttention(
            config.hidden_dim,
            config.attention_heads,
            dropout=config.dropout,
            batch_first=True,
        )
        future_width = (
            config.camera_count
            * config.future_spatial_tokens
            * config.future_dim
        )
        self.future_head = nn.Linear(config.hidden_dim, future_width)
        if config.future_residual_prediction:
            # The zero-initialized planner is exactly the persistence baseline;
            # training only has to explain the physical change from now to t+Δ.
            nn.init.zeros_(self.future_head.weight)
            nn.init.zeros_(self.future_head.bias)
        self.state_head = nn.Linear(config.hidden_dim, config.state_dim)
        self.action_head = nn.Linear(config.hidden_dim, config.action_dim)
        self.interaction_head = nn.Linear(config.hidden_dim, 3)

    def _context(
        self,
        visual: Tensor,
        semantic: Tensor,
        state: Tensor,
        *,
        history_visual: Tensor | None,
        history_semantic: Tensor | None,
        history_states: Tensor | None,
    ) -> Tensor:
        config = self.config
        expected_visual = (
            1,
            config.camera_count,
            config.spatial_tokens,
            config.feature_dim,
        )
        if visual.ndim != 5 or tuple(visual.shape[1:]) != expected_visual:
            raise ValueError("visual cache contract differs from PWR planner")
        if semantic.shape[1:] != (config.semantic_layers, 1, config.feature_dim):
            raise ValueError("semantic cache contract differs from PWR planner")
        if state.shape[1:] != (config.state_dim,):
            raise ValueError("state cache contract differs from PWR planner")
        dtype = self.visual_projection[0].weight.dtype
        visual = visual[:, 0].to(dtype=dtype)
        semantic = semantic[:, :, 0].to(dtype=dtype)
        state = state.to(dtype=dtype)
        visual_tokens = self.visual_projection(visual)
        visual_tokens = visual_tokens + self.camera_embedding[None, :, None]
        visual_tokens = visual_tokens + self.spatial_embedding[None, None]
        visual_tokens = visual_tokens.flatten(1, 2)
        semantic_tokens = self.semantic_projection(semantic)
        semantic_tokens = semantic_tokens + self.semantic_embedding[None]
        state_token = self.state_projection(state)[:, None]
        tokens = [visual_tokens, semantic_tokens, state_token]
        history_count = config.history_length - 1
        if history_count:
            expected_history_visual = (
                visual.shape[0],
                history_count,
                1,
                config.camera_count,
                config.spatial_tokens,
                config.feature_dim,
            )
            if history_visual is None or tuple(history_visual.shape) != expected_history_visual:
                raise ValueError("history_visual cache contract differs from PWR planner")
            if history_semantic is None or history_states is None:
                raise ValueError("PWR planner history fields are incomplete")
            history_visual_summary = history_visual[:, :, 0].to(dtype=dtype).mean(dim=(2, 3))
            history_semantic_summary = history_semantic.to(dtype=dtype).mean(dim=(2, 3))
            history_state = history_states.to(dtype=dtype)
            history_tokens = self.history_projection(
                torch.cat(
                    (
                        self.visual_projection(history_visual_summary),
                        self.semantic_projection(history_semantic_summary),
                        self.state_projection(history_state),
                    ),
                    dim=-1,
                )
            )
            tokens.append(history_tokens)
        return self.context_encoder(torch.cat(tokens, dim=1))

    def forward(
        self,
        visual: Tensor,
        semantic: Tensor,
        state: Tensor,
        *,
        history_visual: Tensor | None = None,
        history_semantic: Tensor | None = None,
        history_states: Tensor | None = None,
    ) -> PWRPlannerOutput:
        memory = self._context(
            visual,
            semantic,
            state,
            history_visual=history_visual,
            history_semantic=history_semantic,
            history_states=history_states,
        )
        queries = self.plan_queries[None].expand(memory.shape[0], -1, -1)
        latent, _ = self.plan_attention(queries, memory, memory, need_weights=False)
        latent = latent + queries
        config = self.config
        future_delta = self.future_head(latent).reshape(
            memory.shape[0],
            len(config.future_offsets),
            config.camera_count,
            config.future_spatial_tokens,
            config.future_dim,
        )
        if config.future_residual_prediction:
            current = compressed_future_visual(visual, config)[:, 0].to(
                dtype=future_delta.dtype
            )
            future = current[:, None] + future_delta
        else:
            future = future_delta
        return PWRPlannerOutput(
            physical_latent=latent,
            context_memory=memory,
            future_visual=future,
            future_state_delta=self.state_head(latent),
            action_summary=self.action_head(latent),
            interaction_logits=self.interaction_head(latent),
        )

    def losses(
        self,
        output: PWRPlannerOutput,
        *,
        current_visual: Tensor,
        current_state: Tensor,
        future_visual: Tensor,
        future_states: Tensor,
        future_mask: Tensor,
        actions: Tensor,
        valid_mask: Tensor,
        previous_actions: Tensor,
        future_weight: float = 1.0,
        state_weight: float = 1.0,
        action_weight: float = 0.5,
        interaction_weight: float = 0.25,
        interaction_class_weights: Tensor | None = None,
    ) -> dict[str, Tensor]:
        config = self.config
        target_visual = compressed_future_visual(future_visual, config).to(
            dtype=output.future_visual.dtype
        )
        current_target = compressed_future_visual(current_visual, config)[:, 0]
        current_target = current_target[:, None].expand_as(target_visual)
        mask = future_mask.to(dtype=output.future_visual.dtype)
        visual_mask = mask[:, :, None, None, None]
        visual_denominator = visual_mask.expand_as(target_visual).sum().clamp_min(1.0)
        future = (
            F.smooth_l1_loss(output.future_visual, target_visual, reduction="none")
            * visual_mask
        ).sum() / visual_denominator
        persistence = (
            F.smooth_l1_loss(current_target, target_visual, reduction="none")
            * visual_mask
        ).sum() / visual_denominator

        state_target = future_states.to(output.future_state_delta.dtype) - current_state[:, None].to(
            output.future_state_delta.dtype
        )
        state_mask = mask[:, :, None]
        state_denominator = state_mask.expand_as(state_target).sum().clamp_min(1.0)
        state_loss = (
            F.smooth_l1_loss(output.future_state_delta, state_target, reduction="none")
            * state_mask
        ).sum() / state_denominator
        state_zero = (
            F.smooth_l1_loss(torch.zeros_like(state_target), state_target, reduction="none")
            * state_mask
        ).sum() / state_denominator

        action_targets: list[Tensor] = []
        action_valid: list[Tensor] = []
        for offset in config.future_offsets:
            prefix_mask = valid_mask[:, :offset].to(actions.dtype)
            denominator = prefix_mask.sum(dim=1, keepdim=True).clamp_min(1.0)
            action_targets.append(
                (actions[:, :offset] * prefix_mask[:, :, None]).sum(dim=1) / denominator
            )
            action_valid.append(valid_mask[:, offset - 1])
        action_target = torch.stack(action_targets, dim=1).to(output.action_summary.dtype)
        action_mask = torch.stack(action_valid, dim=1) & future_mask
        action_weights = action_mask.to(output.action_summary.dtype)[:, :, None]
        action_denominator = action_weights.expand_as(action_target).sum().clamp_min(1.0)
        action_loss = (
            F.smooth_l1_loss(output.action_summary, action_target, reduction="none")
            * action_weights
        ).sum() / action_denominator

        event_target = interaction_event_targets(
            actions, previous_actions, config.future_offsets
        ).to(output.interaction_logits.device)
        event_loss = F.cross_entropy(
            output.interaction_logits.flatten(0, 1),
            event_target.flatten(),
            weight=(
                interaction_class_weights.to(
                    device=output.interaction_logits.device,
                    dtype=output.interaction_logits.dtype,
                )
                if interaction_class_weights is not None
                else None
            ),
            reduction="none",
        ).reshape_as(event_target)
        event_weights = action_mask.to(event_loss.dtype)
        interaction = (event_loss * event_weights).sum() / event_weights.sum().clamp_min(1.0)
        total = (
            future_weight * future
            + state_weight * state_loss
            + action_weight * action_loss
            + interaction_weight * interaction
        )
        return {
            "total": total,
            "future": future,
            "future_persistence": persistence,
            "state": state_loss,
            "state_zero": state_zero,
            "action": action_loss,
            "interaction": interaction,
        }


@dataclass(frozen=True)
class PWRActionExpertConfig:
    """Hybrid continuous/discrete action expert for Duvla V3.1 P1."""

    hidden_dim: int = 384
    arm_dim: int = 6
    action_horizon: int = 8
    attention_heads: int = 6
    planning_layers: int = 2
    flow_layers: int = 4
    feedforward_multiplier: int = 4
    dropout: float = 0.0
    architecture: str = "compact"
    expert_hidden_dim: int | None = None
    flow_prefix_weight: float = 1.0
    gripper_state_loss_weight: float = 0.0
    feature_dim: int = 2048
    semantic_layers: int = 4
    camera_count: int = 2
    spatial_tokens: int = 64
    state_dim: int = 8
    history_length: int = 4

    def __post_init__(self) -> None:
        positive = (
            self.hidden_dim,
            self.arm_dim,
            self.action_horizon,
            self.attention_heads,
            self.planning_layers,
            self.flow_layers,
            self.feedforward_multiplier,
            self.feature_dim,
            self.semantic_layers,
            self.camera_count,
            self.spatial_tokens,
            self.state_dim,
            self.history_length,
        )
        if any(value <= 0 for value in positive):
            raise ValueError("all PWR action expert dimensions must be positive")
        if self.hidden_dim % self.attention_heads:
            raise ValueError("hidden_dim must be divisible by attention_heads")
        if self.architecture not in {"compact", "dense_interleaved"}:
            raise ValueError("PWR action architecture must be compact or dense_interleaved")
        expert_hidden = self.expert_hidden_dim or self.hidden_dim
        if expert_hidden <= 0 or expert_hidden % self.attention_heads:
            raise ValueError("expert hidden size must be positive and divisible by attention heads")
        if self.architecture == "compact" and expert_hidden != self.hidden_dim:
            raise ValueError("compact PWR expert must use the planner hidden size")
        if self.flow_prefix_weight < 1.0:
            raise ValueError("flow_prefix_weight must be at least one")
        if self.gripper_state_loss_weight < 0.0:
            raise ValueError("gripper_state_loss_weight must be non-negative")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")


@dataclass(frozen=True)
class PWRActionExpertOutput:
    arm_velocity: Tensor
    gripper_event_logits: Tensor
    gripper_state_logits: Tensor | None = None


def gripper_event_targets(
    actions: Tensor,
    previous_actions: Tensor,
    *,
    normalized_closed_threshold: float,
) -> Tensor:
    """Return per-step HOLD/CLOSE/OPEN labels from normalized LIBERO actions."""

    if actions.ndim != 3 or previous_actions.ndim != 2:
        raise ValueError("actions and previous_actions must be rank 3 and 2")
    if actions.shape[0] != previous_actions.shape[0]:
        raise ValueError("action batches differ")
    if actions.shape[-1] != previous_actions.shape[-1]:
        raise ValueError("action dimensions differ")
    current_closed = actions[..., -1] > normalized_closed_threshold
    prior_closed = torch.cat(
        (
            (previous_actions[:, -1] > normalized_closed_threshold)[:, None],
            current_closed[:, :-1],
        ),
        dim=1,
    )
    events = torch.zeros_like(current_closed, dtype=torch.long)
    events = torch.where(current_closed & ~prior_closed, torch.ones_like(events), events)
    events = torch.where(
        ~current_closed & prior_closed,
        torch.full_like(events, 2),
        events,
    )
    return events


def integrate_gripper_events(
    event_logits: Tensor,
    previous_actions: Tensor,
    *,
    normalized_closed_threshold: float,
    normalized_open_value: float,
    normalized_closed_value: float,
    event_class_weights: Tensor | None = None,
) -> Tensor:
    """Convert HOLD/CLOSE/OPEN logits into a normalized gripper action chunk."""

    if event_logits.ndim != 3 or event_logits.shape[-1] != 3:
        raise ValueError("event_logits must have shape [batch, horizon, 3]")
    if previous_actions.ndim != 2 or previous_actions.shape[0] != event_logits.shape[0]:
        raise ValueError("previous_actions must match the event batch")
    if event_class_weights is not None:
        event_logits = calibrated_gripper_event_logits(
            event_logits,
            event_class_weights,
        )
    closed = previous_actions[:, -1] > normalized_closed_threshold
    values: list[Tensor] = []
    for event in event_logits.argmax(dim=-1).unbind(dim=1):
        closed = torch.where(event == 1, torch.ones_like(closed), closed)
        closed = torch.where(event == 2, torch.zeros_like(closed), closed)
        values.append(
            torch.where(
                closed,
                torch.full_like(closed, normalized_closed_value, dtype=event_logits.dtype),
                torch.full_like(closed, normalized_open_value, dtype=event_logits.dtype),
            )
        )
    return torch.stack(values, dim=1)


def calibrated_gripper_event_logits(
    event_logits: Tensor,
    event_class_weights: Tensor,
    *,
    correction_strength: float = 1.0,
) -> Tensor:
    """Undo weighted-cross-entropy prior shift before event argmax.

    Weighted CE learns logits proportional to ``log p(class|x) + log weight``.
    Subtracting the fixed training log-weight recovers an unweighted posterior
    without fitting any validation-dependent threshold.
    """

    if event_logits.ndim != 3 or event_logits.shape[-1] != 3:
        raise ValueError("event_logits must have shape [batch, horizon, 3]")
    if tuple(event_class_weights.shape) != (3,) or bool((event_class_weights <= 0).any()):
        raise ValueError("event_class_weights must be three positive values")
    if not 0.0 <= correction_strength <= 1.0:
        raise ValueError("correction_strength must be in [0, 1]")
    return event_logits - correction_strength * event_class_weights.to(
        device=event_logits.device,
        dtype=event_logits.dtype,
    ).log()[None, None]


def fuse_gripper_event_state_logits(
    event_logits: Tensor,
    state_logits: Tensor,
    previous_actions: Tensor,
    *,
    normalized_closed_threshold: float,
) -> Tensor:
    """Fuse shared-token event and absolute-state evidence without task rules."""

    if event_logits.ndim != 3 or event_logits.shape[-1] != 3:
        raise ValueError("event logits must have shape [batch, horizon, 3]")
    if tuple(state_logits.shape) != tuple(event_logits.shape[:2]):
        raise ValueError("state logits must match the event batch and horizon")
    if previous_actions.ndim != 2 or previous_actions.shape[0] != event_logits.shape[0]:
        raise ValueError("previous actions must match the event batch")
    closed_log = F.logsigmoid(state_logits)
    open_log = F.logsigmoid(-state_logits)
    previous_closed = (
        previous_actions[:, -1] > normalized_closed_threshold
    )[:, None]
    hold_log = torch.where(previous_closed, closed_log, open_log)
    return event_logits + torch.stack((hold_log, closed_log, open_log), dim=-1)


class _FlowTimeEmbedding(nn.Module):
    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, time: Tensor) -> Tensor:
        if time.ndim != 1:
            raise ValueError("flow time must have shape [batch]")
        half = self.hidden_dim // 2
        frequencies = torch.exp(
            -torch.log(torch.tensor(10_000.0, device=time.device))
            * torch.arange(half, device=time.device, dtype=torch.float32)
            / max(half - 1, 1)
        )
        angles = time.float()[:, None] * frequencies[None] * 1000.0
        embedding = torch.cat((angles.sin(), angles.cos()), dim=-1)
        if embedding.shape[-1] < self.hidden_dim:
            embedding = F.pad(embedding, (0, self.hidden_dim - embedding.shape[-1]))
        return self.projection(embedding.to(dtype=self.projection[0].weight.dtype))


class PWRActionExpert(nn.Module):
    """Plan-conditioned 6-D Flow expert with coordinated gripper events.

    A shared set of temporal action queries first cross-attends to the frozen
    physical planner memory.  The stable query state predicts gripper events;
    noisy arm actions and Flow time are then injected into those same tokens to
    predict continuous arm velocity.
    """

    def __init__(self, config: PWRActionExpertConfig) -> None:
        super().__init__()
        self.config = config
        expert_hidden = config.expert_hidden_dim or config.hidden_dim
        self.dense_interleaved = config.architecture == "dense_interleaved"
        if self.dense_interleaved:
            self.context_projection = None
            self.raw_visual_projection = nn.Sequential(
                nn.RMSNorm(config.feature_dim, eps=1e-6),
                nn.Linear(config.feature_dim, expert_hidden, bias=False),
            )
            self.raw_semantic_projection = nn.Sequential(
                nn.RMSNorm(config.feature_dim, eps=1e-6),
                nn.Linear(config.feature_dim, expert_hidden, bias=False),
            )
            self.raw_state_projection = nn.Sequential(
                nn.RMSNorm(config.state_dim, eps=1e-6),
                nn.Linear(config.state_dim, expert_hidden),
            )
            self.raw_action_projection = nn.Sequential(
                nn.RMSNorm(config.arm_dim + 1, eps=1e-6),
                nn.Linear(config.arm_dim + 1, expert_hidden),
            )
            self.physical_projection = nn.Sequential(
                nn.RMSNorm(config.hidden_dim, eps=1e-6),
                nn.Linear(config.hidden_dim, expert_hidden, bias=False),
            )
            self.dense_camera_embedding = nn.Parameter(
                torch.randn(config.camera_count, expert_hidden) * 0.02
            )
            self.dense_spatial_embedding = nn.Parameter(
                torch.randn(config.spatial_tokens, expert_hidden) * 0.02
            )
            self.dense_semantic_embedding = nn.Parameter(
                torch.randn(config.semantic_layers, expert_hidden) * 0.02
            )
            self.dense_history_embedding = nn.Parameter(
                torch.randn(max(config.history_length - 1, 1), expert_hidden) * 0.02
            )
            dense_config = FlowMatchingVLAConfig(
                state_dim=config.hidden_dim,
                action_dim=config.arm_dim,
                vlm_feature_dim=expert_hidden,
                hidden_dim=expert_hidden,
                action_horizon=config.action_horizon,
                max_context_tokens=160,
                expert_layers=config.flow_layers,
                expert_heads=config.attention_heads,
                dropout=config.dropout,
                num_flow_steps=10,
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
            self.dense_flow = FlowMatchingActionExpert(dense_config)
            self.dense_state_token = None
            self.gripper_event_head = nn.Linear(expert_hidden, 3)
            self.gripper_state_head = nn.Linear(expert_hidden, 1)
            self.action_queries = None
            self.plan_type_embedding = None
            self.planning_decoder = None
            self.arm_input = None
            self.time_embedding = None
            self.flow_fusion = None
            self.flow_encoder = None
            self.arm_velocity_head = None
            return

        self.context_projection = None
        self.raw_visual_projection = None
        self.raw_semantic_projection = None
        self.raw_state_projection = None
        self.raw_action_projection = None
        self.physical_projection = None
        self.dense_camera_embedding = None
        self.dense_spatial_embedding = None
        self.dense_semantic_embedding = None
        self.dense_history_embedding = None
        self.dense_flow = None
        self.dense_state_token = None
        self.gripper_state_head = None
        self.action_queries = nn.Parameter(
            torch.randn(config.action_horizon, config.hidden_dim) * 0.02
        )
        self.plan_type_embedding = nn.Parameter(torch.randn(config.hidden_dim) * 0.02)
        planning_layer = nn.TransformerDecoderLayer(
            d_model=config.hidden_dim,
            nhead=config.attention_heads,
            dim_feedforward=config.hidden_dim * config.feedforward_multiplier,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.planning_decoder = nn.TransformerDecoder(
            planning_layer,
            num_layers=config.planning_layers,
            norm=nn.RMSNorm(config.hidden_dim, eps=1e-6),
        )
        self.arm_input = nn.Linear(config.arm_dim, config.hidden_dim)
        self.time_embedding = _FlowTimeEmbedding(config.hidden_dim)
        self.flow_fusion = nn.Sequential(
            nn.RMSNorm(config.hidden_dim * 3, eps=1e-6),
            nn.Linear(config.hidden_dim * 3, config.hidden_dim),
            nn.SiLU(),
        )
        flow_layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_dim,
            nhead=config.attention_heads,
            dim_feedforward=config.hidden_dim * config.feedforward_multiplier,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.flow_encoder = nn.TransformerEncoder(
            flow_layer,
            num_layers=config.flow_layers,
            norm=nn.RMSNorm(config.hidden_dim, eps=1e-6),
        )
        self.arm_velocity_head = nn.Linear(config.hidden_dim, config.arm_dim)
        self.gripper_event_head = nn.Linear(config.hidden_dim, 3)

    def plan_tokens(self, context_memory: Tensor, physical_latent: Tensor) -> Tensor:
        if self.dense_interleaved:
            raise RuntimeError("dense interleaved expert does not collapse context into plan tokens")
        config = self.config
        if context_memory.ndim != 3 or context_memory.shape[-1] != config.hidden_dim:
            raise ValueError("context_memory must have shape [batch, tokens, hidden]")
        if (
            physical_latent.ndim != 3
            or physical_latent.shape[0] != context_memory.shape[0]
            or physical_latent.shape[-1] != config.hidden_dim
        ):
            raise ValueError("physical_latent must match context memory batch/hidden")
        memory = torch.cat(
            (
                context_memory,
                physical_latent + self.plan_type_embedding[None, None],
            ),
            dim=1,
        )
        queries = self.action_queries[None].expand(context_memory.shape[0], -1, -1)
        return self.planning_decoder(queries, memory)

    def forward(
        self,
        context_memory: Tensor,
        physical_latent: Tensor,
        noisy_arm: Tensor,
        time: Tensor,
        *,
        visual: Tensor | None = None,
        semantic: Tensor | None = None,
        state: Tensor | None = None,
        history_visual: Tensor | None = None,
        history_semantic: Tensor | None = None,
        history_states: Tensor | None = None,
        previous_actions: Tensor | None = None,
        history_previous_actions: Tensor | None = None,
    ) -> PWRActionExpertOutput:
        config = self.config
        expected = (context_memory.shape[0], config.action_horizon, config.arm_dim)
        if tuple(noisy_arm.shape) != expected:
            raise ValueError(f"noisy_arm must have shape {expected}")
        if tuple(time.shape) != (context_memory.shape[0],):
            raise ValueError("time must have shape [batch]")
        if self.dense_interleaved:
            return self._dense_forward(
                context_memory,
                physical_latent,
                noisy_arm,
                time,
                visual=visual,
                semantic=semantic,
                state=state,
                history_visual=history_visual,
                history_semantic=history_semantic,
                history_states=history_states,
                previous_actions=previous_actions,
                history_previous_actions=history_previous_actions,
            )
        plan_tokens = self.plan_tokens(context_memory, physical_latent)
        return self.predict_from_plan_tokens(plan_tokens, noisy_arm, time)

    def _dense_condition(
        self,
        context_memory: Tensor,
        physical_latent: Tensor,
        *,
        visual: Tensor | None,
        semantic: Tensor | None,
        state: Tensor | None,
        history_visual: Tensor | None,
        history_semantic: Tensor | None,
        history_states: Tensor | None,
        previous_actions: Tensor | None,
        history_previous_actions: Tensor | None,
    ) -> tuple[Tensor, Tensor]:
        if (
            self.raw_visual_projection is None
            or self.raw_semantic_projection is None
            or self.raw_state_projection is None
            or self.raw_action_projection is None
            or self.physical_projection is None
            or self.dense_camera_embedding is None
            or self.dense_spatial_embedding is None
            or self.dense_semantic_embedding is None
            or self.dense_history_embedding is None
        ):
            raise RuntimeError("dense PWR projections are unavailable")
        config = self.config
        batch = context_memory.shape[0]
        if visual is None or tuple(visual.shape) != (
            batch,
            1,
            config.camera_count,
            config.spatial_tokens,
            config.feature_dim,
        ):
            raise ValueError("dense PWR visual input has an invalid shape")
        if semantic is None or tuple(semantic.shape) != (
            batch,
            config.semantic_layers,
            1,
            config.feature_dim,
        ):
            raise ValueError("dense PWR semantic input has an invalid shape")
        if state is None or tuple(state.shape) != (batch, config.state_dim):
            raise ValueError("dense PWR state input has an invalid shape")
        if previous_actions is None or tuple(previous_actions.shape) != (
            batch,
            config.arm_dim + 1,
        ):
            raise ValueError("dense PWR previous action has an invalid shape")
        visual_tokens = self.raw_visual_projection(visual[:, 0])
        visual_tokens = (
            visual_tokens
            + self.dense_camera_embedding[None, :, None]
            + self.dense_spatial_embedding[None, None]
        ).flatten(1, 2)
        semantic_tokens = self.raw_semantic_projection(semantic[:, :, 0])
        semantic_tokens = semantic_tokens + self.dense_semantic_embedding[None]
        state_token = (
            self.raw_state_projection(state)
            + self.raw_action_projection(previous_actions)
        )[:, None]
        context_parts = [visual_tokens, semantic_tokens]
        history_count = config.history_length - 1
        if history_count:
            if (
                history_visual is None
                or tuple(history_visual.shape)
                != (
                    batch,
                    history_count,
                    1,
                    config.camera_count,
                    config.spatial_tokens,
                    config.feature_dim,
                )
                or history_semantic is None
                or tuple(history_semantic.shape)
                != (
                    batch,
                    history_count,
                    config.semantic_layers,
                    1,
                    config.feature_dim,
                )
                or history_states is None
                or tuple(history_states.shape)
                != (batch, history_count, config.state_dim)
                or history_previous_actions is None
                or tuple(history_previous_actions.shape)
                != (batch, history_count, config.arm_dim + 1)
            ):
                raise ValueError("dense PWR history inputs have invalid shapes")
            history = (
                self.raw_visual_projection(history_visual[:, :, 0]).mean(dim=(2, 3))
                + self.raw_semantic_projection(history_semantic[:, :, :, 0]).mean(dim=2)
                + self.raw_state_projection(history_states)
                + self.raw_action_projection(history_previous_actions)
                + self.dense_history_embedding[None, :history_count]
            )
            context_parts.append(history)
        physical = self.physical_projection(physical_latent)
        context_parts.append(physical)
        return torch.cat(context_parts, dim=1), state_token

    def _dense_forward(
        self,
        context_memory: Tensor,
        physical_latent: Tensor,
        noisy_arm: Tensor,
        time: Tensor,
        **raw_inputs: Tensor | None,
    ) -> PWRActionExpertOutput:
        if self.dense_flow is None or self.gripper_state_head is None:
            raise RuntimeError("dense PWR action expert is unavailable")
        context, state_token = self._dense_condition(
            context_memory, physical_latent, **raw_inputs
        )
        velocity, hidden = self.dense_flow(
            context,
            state_token,
            noisy_arm,
            time,
            return_hidden=True,
        )
        return PWRActionExpertOutput(
            arm_velocity=velocity,
            gripper_event_logits=self.gripper_event_head(hidden),
            gripper_state_logits=self.gripper_state_head(hidden).squeeze(-1),
        )

    def predict_from_plan_tokens(
        self,
        plan_tokens: Tensor,
        noisy_arm: Tensor,
        time: Tensor,
    ) -> PWRActionExpertOutput:
        """Predict from cached plan tokens, avoiding repeated cross-attention."""

        if self.dense_interleaved:
            raise RuntimeError("dense expert prediction requires the full planner context")
        config = self.config
        expected = (plan_tokens.shape[0], config.action_horizon, config.hidden_dim)
        if tuple(plan_tokens.shape) != expected:
            raise ValueError(f"plan_tokens must have shape {expected}")
        if tuple(noisy_arm.shape) != (
            plan_tokens.shape[0],
            config.action_horizon,
            config.arm_dim,
        ):
            raise ValueError("noisy_arm differs from cached plan tokens")
        if tuple(time.shape) != (plan_tokens.shape[0],):
            raise ValueError("time must have shape [batch]")
        time_tokens = self.time_embedding(time)[:, None].expand_as(plan_tokens)
        noisy_tokens = self.arm_input(noisy_arm.to(dtype=plan_tokens.dtype))
        flow_tokens = self.flow_fusion(
            torch.cat((plan_tokens, noisy_tokens, time_tokens), dim=-1)
        )
        hidden = self.flow_encoder(flow_tokens)
        return PWRActionExpertOutput(
            arm_velocity=self.arm_velocity_head(hidden),
            gripper_event_logits=self.gripper_event_head(plan_tokens),
        )

    def arm_flow_loss(
        self,
        output: PWRActionExpertOutput,
        *,
        target_velocity: Tensor,
        valid_mask: Tensor,
    ) -> Tensor:
        if tuple(target_velocity.shape) != tuple(output.arm_velocity.shape):
            raise ValueError("target velocity differs from predicted arm velocity")
        if tuple(valid_mask.shape) != tuple(target_velocity.shape[:2]):
            raise ValueError("valid mask differs from the arm Flow contract")
        weights = valid_mask.to(dtype=output.arm_velocity.dtype)
        if self.config.flow_prefix_weight > 1.0:
            weights = weights.clone()
            weights[:, : min(2, weights.shape[1])] *= self.config.flow_prefix_weight
        expanded = weights[:, :, None].expand_as(output.arm_velocity)
        return (
            F.mse_loss(output.arm_velocity, target_velocity, reduction="none")
            * expanded
        ).sum() / expanded.sum().clamp_min(1.0)

    def losses(
        self,
        output: PWRActionExpertOutput,
        *,
        target_velocity: Tensor,
        actions: Tensor,
        valid_mask: Tensor,
        previous_actions: Tensor,
        normalized_closed_threshold: float,
        event_class_weights: Tensor | None = None,
        gripper_weight: float = 0.1,
    ) -> dict[str, Tensor]:
        config = self.config
        if tuple(target_velocity.shape) != (
            actions.shape[0],
            config.action_horizon,
            config.arm_dim,
        ):
            raise ValueError("target_velocity shape differs from the arm Flow contract")
        if actions.shape[1:] != (config.action_horizon, config.arm_dim + 1):
            raise ValueError("actions differ from the hybrid action contract")
        if tuple(valid_mask.shape) != tuple(actions.shape[:2]):
            raise ValueError("valid_mask differs from the action chunk")
        weights = valid_mask.to(dtype=output.arm_velocity.dtype)
        flow = self.arm_flow_loss(
            output,
            target_velocity=target_velocity,
            valid_mask=valid_mask,
        )
        targets = gripper_event_targets(
            actions,
            previous_actions,
            normalized_closed_threshold=normalized_closed_threshold,
        ).to(device=output.gripper_event_logits.device)
        events = F.cross_entropy(
            output.gripper_event_logits.flatten(0, 1),
            targets.flatten(),
            weight=(
                event_class_weights.to(
                    device=output.gripper_event_logits.device,
                    dtype=output.gripper_event_logits.dtype,
                )
                if event_class_weights is not None
                else None
            ),
            reduction="none",
        ).reshape_as(targets)
        gripper = (events * weights).sum() / weights.sum().clamp_min(1.0)
        losses = {"flow": flow, "gripper": gripper}
        total = flow + gripper_weight * gripper
        if output.gripper_state_logits is not None:
            state_targets = (
                actions[..., -1] > normalized_closed_threshold
            ).to(dtype=output.gripper_state_logits.dtype)
            state_values = F.binary_cross_entropy_with_logits(
                output.gripper_state_logits,
                state_targets,
                reduction="none",
            )
            state = (state_values * weights).sum() / weights.sum().clamp_min(1.0)
            losses["gripper_state"] = state
            total = total + self.config.gripper_state_loss_weight * state
        return {"total": total, **losses}

    @torch.no_grad()
    def sample(
        self,
        context_memory: Tensor,
        physical_latent: Tensor,
        previous_actions: Tensor,
        *,
        noise: Tensor,
        normalized_closed_threshold: float,
        normalized_open_value: float,
        normalized_closed_value: float,
        event_class_weights: Tensor | None = None,
        integration_steps: int = 10,
        visual: Tensor | None = None,
        semantic: Tensor | None = None,
        state: Tensor | None = None,
        history_visual: Tensor | None = None,
        history_semantic: Tensor | None = None,
        history_states: Tensor | None = None,
        history_previous_actions: Tensor | None = None,
    ) -> Tensor:
        if integration_steps <= 0:
            raise ValueError("integration_steps must be positive")
        arm = noise.clone()
        if self.dense_interleaved:
            step_size = 1.0 / integration_steps
            for step in range(integration_steps):
                time = torch.full(
                    (arm.shape[0],),
                    step / integration_steps,
                    device=arm.device,
                    dtype=arm.dtype,
                )
                output = self._dense_forward(
                    context_memory,
                    physical_latent,
                    arm,
                    time,
                    visual=visual,
                    semantic=semantic,
                    state=state,
                    history_visual=history_visual,
                    history_semantic=history_semantic,
                    history_states=history_states,
                    previous_actions=previous_actions,
                    history_previous_actions=history_previous_actions,
                )
                arm = arm + step_size * output.arm_velocity
            final = self._dense_forward(
                context_memory,
                physical_latent,
                arm,
                torch.ones(arm.shape[0], device=arm.device, dtype=arm.dtype),
                visual=visual,
                semantic=semantic,
                state=state,
                history_visual=history_visual,
                history_semantic=history_semantic,
                history_states=history_states,
                previous_actions=previous_actions,
                history_previous_actions=history_previous_actions,
            )
            event_logits = final.gripper_event_logits
            if event_class_weights is not None:
                event_logits = calibrated_gripper_event_logits(
                    event_logits, event_class_weights
                )
            if final.gripper_state_logits is not None:
                event_logits = fuse_gripper_event_state_logits(
                    event_logits,
                    final.gripper_state_logits,
                    previous_actions,
                    normalized_closed_threshold=normalized_closed_threshold,
                )
            gripper = integrate_gripper_events(
                event_logits,
                previous_actions,
                normalized_closed_threshold=normalized_closed_threshold,
                normalized_open_value=normalized_open_value,
                normalized_closed_value=normalized_closed_value,
            )
            return torch.cat((arm, gripper[:, :, None]), dim=-1)
        plan_tokens = self.plan_tokens(context_memory, physical_latent)
        event_logits = self.gripper_event_head(plan_tokens)
        step_size = 1.0 / integration_steps
        for step in range(integration_steps):
            time = torch.full(
                (arm.shape[0],),
                step / integration_steps,
                device=arm.device,
                dtype=arm.dtype,
            )
            output = self.predict_from_plan_tokens(plan_tokens, arm, time)
            arm = arm + step_size * output.arm_velocity
        gripper = integrate_gripper_events(
            event_logits,
            previous_actions,
            normalized_closed_threshold=normalized_closed_threshold,
            normalized_open_value=normalized_open_value,
            normalized_closed_value=normalized_closed_value,
            event_class_weights=event_class_weights,
        )
        return torch.cat((arm, gripper[:, :, None]), dim=-1)


@dataclass(frozen=True)
class DuvlaV31PolicyConfig:
    planner: PWRPlannerConfig
    expert: PWRActionExpertConfig
    flow_steps: int = 10
    flow_samples: int = 5
    replan_action_steps: int = 2
    gripper_event_class_weights: tuple[float, float, float] = (1.0, 16.0, 16.0)
    gripper_correction_strength: float = 0.3
    normalized_closed_threshold: float = 0.0
    normalized_open_value: float = -1.0
    normalized_closed_value: float = 1.0
    highres_layer14_visual: bool = True
    unified_flow_gripper: bool = False
    gripper_control_mode: str = "event3"
    action_history_conditioning: bool = False
    gripper_action_conditioning: bool = True

    @property
    def action_horizon(self) -> int:
        return self.expert.action_horizon

    @property
    def action_dim(self) -> int:
        return self.expert.arm_dim + 1

    @property
    def state_dim(self) -> int:
        return self.planner.state_dim

    @property
    def history_length(self) -> int:
        return self.planner.history_length

    @property
    def history_stride(self) -> int:
        return self.planner.history_stride


class DuvlaV31Policy(nn.Module):
    """Deployable P0b planner + P1 hybrid expert without task-index routing."""

    def __init__(self, config: DuvlaV31PolicyConfig) -> None:
        super().__init__()
        if config.planner.hidden_dim != config.expert.hidden_dim:
            raise ValueError("planner and expert hidden dimensions differ")
        if config.planner.action_horizon != config.expert.action_horizon:
            raise ValueError("planner and expert horizons differ")
        if config.flow_steps <= 0 or config.flow_samples <= 0:
            raise ValueError("Flow inference counts must be positive")
        if not 0.0 <= config.gripper_correction_strength <= 1.0:
            raise ValueError("gripper correction strength must be in [0, 1]")
        self.config = config
        self.planner = PhysicalLatentPlanner(config.planner)
        self.action_expert = PWRActionExpert(config.expert)

    @torch.no_grad()
    def sample_actions(
        self,
        visual: Tensor,
        semantic: Tensor,
        state: Tensor,
        *,
        history_visual: Tensor,
        history_semantic: Tensor,
        history_states: Tensor,
        previous_action: Tensor,
        history_previous_actions: Tensor | None = None,
        flow_samples: int | None = None,
        noise: Tensor | None = None,
        **_unused: object,
    ) -> Tensor:
        samples = flow_samples or self.config.flow_samples
        plan = self.planner(
            visual,
            semantic,
            state,
            history_visual=history_visual,
            history_semantic=history_semantic,
            history_states=history_states,
        )
        batch = visual.shape[0]
        if noise is None:
            noise_arm = torch.randn(
                batch,
                samples,
                self.config.action_horizon,
                self.config.expert.arm_dim,
                device=visual.device,
                dtype=visual.dtype,
            )
        else:
            expected = (
                batch,
                samples,
                self.config.action_horizon,
                self.config.action_dim,
            )
            if tuple(noise.shape) != expected:
                raise ValueError(f"noise must have shape {expected}")
            noise_arm = noise[..., : self.config.expert.arm_dim]
        context = plan.context_memory[:, None].expand(
            -1, samples, -1, -1
        ).flatten(0, 1)
        latent = plan.physical_latent[:, None].expand(
            -1, samples, -1, -1
        ).flatten(0, 1)
        previous = previous_action[:, None].expand(-1, samples, -1).flatten(0, 1)
        raw_visual = visual[:, None].expand(-1, samples, *visual.shape[1:]).flatten(0, 1)
        raw_semantic = semantic[:, None].expand(-1, samples, *semantic.shape[1:]).flatten(0, 1)
        raw_state = state[:, None].expand(-1, samples, *state.shape[1:]).flatten(0, 1)
        raw_history_visual = history_visual[:, None].expand(
            -1, samples, *history_visual.shape[1:]
        ).flatten(0, 1)
        raw_history_semantic = history_semantic[:, None].expand(
            -1, samples, *history_semantic.shape[1:]
        ).flatten(0, 1)
        raw_history_states = history_states[:, None].expand(
            -1, samples, *history_states.shape[1:]
        ).flatten(0, 1)
        raw_history_actions = None
        if history_previous_actions is not None:
            raw_history_actions = history_previous_actions[:, None].expand(
                -1, samples, *history_previous_actions.shape[1:]
            ).flatten(0, 1)
        generated = self.action_expert.sample(
            context,
            latent,
            previous,
            noise=noise_arm.flatten(0, 1),
            normalized_closed_threshold=self.config.normalized_closed_threshold,
            normalized_open_value=self.config.normalized_open_value,
            normalized_closed_value=self.config.normalized_closed_value,
            event_class_weights=torch.tensor(
                self.config.gripper_event_class_weights,
                device=visual.device,
            ).pow(self.config.gripper_correction_strength),
            integration_steps=self.config.flow_steps,
            visual=raw_visual,
            semantic=raw_semantic,
            state=raw_state,
            history_visual=raw_history_visual,
            history_semantic=raw_history_semantic,
            history_states=raw_history_states,
            history_previous_actions=raw_history_actions,
        ).reshape(batch, samples, self.config.action_horizon, self.config.action_dim)
        return generated.median(dim=1).values


@dataclass(frozen=True)
class PWRProgressiveResidualConfig:
    """Parent-preserving residual controller used after the credible P1c base."""

    base_hidden_dim: int = 768
    planner_hidden_dim: int = 384
    action_dim: int = 7
    action_horizon: int = 8
    hidden_dim: int = 384
    layers: int = 4
    attention_heads: int = 6
    feedforward_multiplier: int = 4
    dropout: float = 0.0
    direct_arm_trust_region: float = 0.20
    direct_gripper_trust_region: float = 0.0
    progress_arm_trust_region: float = 0.15
    progress_gripper_trust_region: float = 0.10
    # The output projection is exactly zero-initialized, so a near-closed gate
    # is unnecessary for parent preservation and severely attenuates early
    # residual gradients.  Start neutral and let the bounded trust region carry
    # the safety constraint.
    gate_init: float = 0.50

    def __post_init__(self) -> None:
        positive = (
            self.base_hidden_dim,
            self.planner_hidden_dim,
            self.action_dim,
            self.action_horizon,
            self.hidden_dim,
            self.layers,
            self.attention_heads,
            self.feedforward_multiplier,
        )
        if any(value <= 0 for value in positive):
            raise ValueError("all progressive residual dimensions must be positive")
        if self.hidden_dim % self.attention_heads:
            raise ValueError("residual hidden size must be divisible by attention heads")
        regions = (
            self.direct_arm_trust_region,
            self.direct_gripper_trust_region,
            self.progress_arm_trust_region,
            self.progress_gripper_trust_region,
        )
        if any(value < 0.0 for value in regions):
            raise ValueError("residual trust regions must be non-negative")
        if not 0.0 < self.gate_init < 1.0:
            raise ValueError("residual gate_init must lie strictly between zero and one")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("residual dropout must be in [0, 1)")


class _PWRResidualBlock(nn.Module):
    def __init__(self, config: PWRProgressiveResidualConfig) -> None:
        super().__init__()
        hidden = config.hidden_dim
        self.attention_norm = nn.RMSNorm(hidden, eps=1e-6)
        self.attention = nn.MultiheadAttention(
            hidden,
            config.attention_heads,
            dropout=config.dropout,
            bias=False,
            batch_first=True,
        )
        self.feedforward_norm = nn.RMSNorm(hidden, eps=1e-6)
        self.feedforward_in = nn.Linear(
            hidden,
            2 * config.feedforward_multiplier * hidden,
            bias=False,
        )
        self.feedforward_out = nn.Linear(
            config.feedforward_multiplier * hidden,
            hidden,
            bias=False,
        )

    def forward(self, tokens: Tensor) -> Tensor:
        normalized = self.attention_norm(tokens)
        attended, _ = self.attention(
            normalized,
            normalized,
            normalized,
            need_weights=False,
        )
        tokens = tokens + attended
        value, gate = self.feedforward_in(
            self.feedforward_norm(tokens)
        ).chunk(2, dim=-1)
        return tokens + self.feedforward_out(F.silu(gate) * value)


class _PWRResidualStage(nn.Module):
    def __init__(
        self,
        config: PWRProgressiveResidualConfig,
        *,
        physical_conditioned: bool,
        arm_trust_region: float,
        gripper_trust_region: float,
    ) -> None:
        super().__init__()
        self.physical_conditioned = physical_conditioned
        hidden = config.hidden_dim
        self.action_projection = nn.Linear(config.action_dim, hidden)
        self.context_projection = nn.Sequential(
            nn.RMSNorm(config.base_hidden_dim, eps=1e-6),
            nn.Linear(config.base_hidden_dim, hidden, bias=False),
        )
        self.state_projection = nn.Sequential(
            nn.RMSNorm(config.base_hidden_dim, eps=1e-6),
            nn.Linear(config.base_hidden_dim, hidden, bias=False),
        )
        self.semantic_projection = (
            nn.Sequential(
                nn.RMSNorm(config.base_hidden_dim, eps=1e-6),
                nn.Linear(config.base_hidden_dim, hidden, bias=False),
            )
            if physical_conditioned
            else None
        )
        self.physical_projection = (
            nn.Sequential(
                nn.RMSNorm(config.planner_hidden_dim, eps=1e-6),
                nn.Linear(config.planner_hidden_dim, hidden, bias=False),
            )
            if physical_conditioned
            else None
        )
        self.action_position = nn.Parameter(
            torch.randn(config.action_horizon, hidden) * 0.02
        )
        self.blocks = nn.ModuleList(
            _PWRResidualBlock(config) for _ in range(config.layers)
        )
        self.output = nn.Linear(hidden, config.action_dim)
        self.gate = nn.Parameter(
            torch.full(
                (config.action_dim,),
                torch.logit(torch.tensor(config.gate_init)).item(),
            )
        )
        limits = torch.full((config.action_dim,), arm_trust_region)
        limits[-1] = gripper_trust_region
        self.register_buffer("trust_region", limits, persistent=True)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(
        self,
        actions: Tensor,
        context_summary: Tensor,
        state_token: Tensor,
        semantic_summary: Tensor,
        physical_latent: Tensor,
    ) -> Tensor:
        condition = self.context_projection(context_summary) + self.state_projection(
            state_token
        )
        if self.physical_conditioned:
            if self.semantic_projection is None or self.physical_projection is None:
                raise RuntimeError("physical residual projections are unavailable")
            condition = condition + self.semantic_projection(semantic_summary)
            condition = condition + self.physical_projection(physical_latent).mean(dim=1)
        tokens = (
            self.action_projection(actions)
            + condition[:, None]
            + self.action_position[None]
        )
        for block in self.blocks:
            tokens = block(tokens)
        delta = torch.tanh(self.output(tokens)) * self.trust_region
        return delta * torch.sigmoid(self.gate)[None, None]


class PWRProgressiveResidualController(nn.Module):
    """Direct residual followed by a physical-progress residual."""

    def __init__(self, config: PWRProgressiveResidualConfig) -> None:
        super().__init__()
        self.config = config
        self.direct = _PWRResidualStage(
            config,
            physical_conditioned=False,
            arm_trust_region=config.direct_arm_trust_region,
            gripper_trust_region=config.direct_gripper_trust_region,
        )
        self.progress = _PWRResidualStage(
            config,
            physical_conditioned=True,
            arm_trust_region=config.progress_arm_trust_region,
            gripper_trust_region=config.progress_gripper_trust_region,
        )

    def forward(
        self,
        base_actions: Tensor,
        context_summary: Tensor,
        state_token: Tensor,
        semantic_summary: Tensor,
        physical_latent: Tensor,
        *,
        apply_direct: bool,
        apply_progress: bool,
    ) -> tuple[Tensor, Tensor, Tensor]:
        direct = (
            self.direct(
                base_actions,
                context_summary,
                state_token,
                semantic_summary,
                physical_latent,
            )
            if apply_direct
            else torch.zeros_like(base_actions)
        )
        corrected = base_actions + direct
        progress = (
            self.progress(
                corrected,
                context_summary,
                state_token,
                semantic_summary,
                physical_latent,
            )
            if apply_progress
            else torch.zeros_like(base_actions)
        )
        return corrected + progress, direct, progress


@dataclass(frozen=True)
class DuvlaV31AnchoredPolicyConfig:
    """Exact V2.6 base + P0b + zero-initialized PWR residual stages."""

    base: DuvlaV21Config
    planner: PWRPlannerConfig
    residual: PWRProgressiveResidualConfig
    flow_steps: int = 10
    flow_samples: int = 5
    replan_action_steps: int = 2
    highres_layer14_visual: bool = True

    def __post_init__(self) -> None:
        if self.base.action_dim != self.residual.action_dim:
            raise ValueError("base and residual action dimensions differ")
        if self.base.action_horizon != self.residual.action_horizon:
            raise ValueError("base and residual horizons differ")
        if self.base.hidden_dim != self.residual.base_hidden_dim:
            raise ValueError("base and residual hidden dimensions differ")
        if self.planner.hidden_dim != self.residual.planner_hidden_dim:
            raise ValueError("planner and residual hidden dimensions differ")
        if self.base.history_length != self.planner.history_length:
            raise ValueError("base and planner history lengths differ")
        if self.base.history_stride != self.planner.history_stride:
            raise ValueError("base and planner history strides differ")
        if min(self.flow_steps, self.flow_samples, self.replan_action_steps) <= 0:
            raise ValueError("anchored policy inference counts must be positive")

    @property
    def action_horizon(self) -> int:
        return self.base.action_horizon

    @property
    def action_dim(self) -> int:
        return self.base.action_dim

    @property
    def state_dim(self) -> int:
        return self.base.state_dim

    @property
    def history_length(self) -> int:
        return self.base.history_length

    @property
    def history_stride(self) -> int:
        return self.base.history_stride

    @property
    def unified_flow_gripper(self) -> bool:
        return self.base.unified_flow_gripper

    @property
    def gripper_control_mode(self) -> str:
        return self.base.gripper_control_mode

    @property
    def action_history_conditioning(self) -> bool:
        return self.base.action_history_conditioning

    @property
    def gripper_action_conditioning(self) -> bool:
        return self.base.gripper_action_conditioning


class DuvlaV31AnchoredPolicy(nn.Module):
    """P1c/P2 policy that preserves the exact natural-language V2.6 generator."""

    def __init__(self, config: DuvlaV31AnchoredPolicyConfig) -> None:
        super().__init__()
        self.config = config
        self.base_policy = DuvlaV21Policy(config.base)
        self.planner = PhysicalLatentPlanner(config.planner)
        self.residual = PWRProgressiveResidualController(config.residual)

    def freeze_backbones(self) -> None:
        self.base_policy.eval().requires_grad_(False)
        self.planner.eval().requires_grad_(False)

    @torch.no_grad()
    def sample_actions(
        self,
        visual: Tensor,
        semantic: Tensor,
        state: Tensor,
        *,
        history_visual: Tensor,
        history_semantic: Tensor,
        history_states: Tensor,
        previous_action: Tensor,
        history_previous_actions: Tensor | None = None,
        flow_samples: int | None = None,
        noise: Tensor | None = None,
        apply_direct: bool = True,
        apply_instruction: bool = True,
        **_unused: object,
    ) -> Tensor:
        context, state_token, semantic_summary, _ = self.base_policy._encode_context(
            visual,
            semantic,
            state,
            history_visual=history_visual,
            history_semantic=history_semantic,
            history_states=history_states,
            previous_action=previous_action,
            history_previous_actions=history_previous_actions,
        )
        samples = flow_samples or self.config.flow_samples
        candidates = self.base_policy._integrate_flow_candidates(
            context,
            state_token,
            samples=samples,
            steps=self.config.flow_steps,
            noise=noise,
        )
        base_actions = candidates.median(dim=1).values
        plan = self.planner(
            visual,
            semantic,
            state,
            history_visual=history_visual,
            history_semantic=history_semantic,
            history_states=history_states,
        )
        corrected, _, _ = self.residual(
            base_actions,
            context.mean(dim=1),
            state_token[:, 0],
            semantic_summary,
            plan.physical_latent,
            apply_direct=apply_direct,
            apply_progress=apply_instruction,
        )
        return corrected
