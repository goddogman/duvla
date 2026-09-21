"""Duvla V2.1 multi-layer progressive-residual Flow policy.

This module is deliberately independent from V1.x checkpoint-compatible
policy branches.  It consumes the schema-7 frozen-Qwen cache and never
accepts a benchmark task index.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import Tensor, nn

from duvla.models.flow_matching_vla import (
    FlowMatchingActionExpert,
    FlowMatchingVLAConfig,
)


def aggregate_action_candidates(
    candidates: Tensor,
    *,
    mode: str,
    executed_prefix: int,
    prefix_weight: float,
) -> Tensor:
    """Aggregate normalized ``[B,K,H,D]`` candidates without privileged labels."""
    if candidates.ndim != 4 or not torch.isfinite(candidates).all():
        raise ValueError("finite action candidates [B,K,H,D] required")
    if not 0 < executed_prefix <= candidates.shape[2] or prefix_weight < 1.0:
        raise ValueError("invalid candidate aggregation weights")
    if mode == "coordinate_median":
        return candidates.median(dim=1).values
    if mode != "trajectory_medoid":
        raise ValueError(f"unsupported candidate aggregation: {mode}")
    temporal_weights = candidates.new_ones(candidates.shape[2])
    temporal_weights[:executed_prefix] = prefix_weight
    pairwise = (candidates[:, :, None] - candidates[:, None, :]).abs().mean(dim=-1)
    pairwise = (pairwise * temporal_weights[None, None, None, :]).sum(dim=-1)
    selected = pairwise.sum(dim=-1).argmin(dim=-1)
    batch_indices = torch.arange(candidates.shape[0], device=candidates.device)
    return candidates[batch_indices, selected]


def pool_multilayer_spatial_grid(values: Tensor, *, output_side: int = 4) -> Tensor:
    """Average-pool ``[B,L,C,T,D]`` square rasters without mixing axes."""

    if values.ndim != 5 or output_side <= 0:
        raise ValueError("spatial values must have shape [B,L,C,T,D]")
    input_side = math.isqrt(values.shape[3])
    if input_side * input_side != values.shape[3] or input_side % output_side:
        raise ValueError("input spatial grid must be square and divisible by output_side")
    scale = input_side // output_side
    batch, layers, cameras, _tokens, width = values.shape
    return values.reshape(
        batch,
        layers,
        cameras,
        output_side,
        scale,
        output_side,
        scale,
        width,
    ).mean(dim=(4, 6)).reshape(
        batch, layers, cameras, output_side * output_side, width
    )


@dataclass(frozen=True)
class DuvlaV21Config:
    ordered_language_bridge: bool = False
    ordered_language_bridge_mode: str = "scalar_gate"
    language_residual_scale: float = 0.10
    language_max_tokens: int = 32
    cross_camera_fusion: bool = False
    cross_camera_residual_scale: float = 0.10
    causal_action_attention: bool = True
    candidate_aggregation: str = "coordinate_median"
    feature_dim: int = 2048
    feature_layers: int = 4
    camera_count: int = 2
    spatial_tokens: int = 16
    semantic_tokens: int = 1
    state_dim: int = 8
    action_dim: int = 7
    action_horizon: int = 8
    adapter_rank: int = 128
    hidden_dim: int = 768
    expert_layers: int = 12
    expert_heads: int = 12
    context_layers: int = 4
    residual_layers: int = 2
    history_length: int = 4
    history_stride: int = 1
    flow_steps: int = 10
    flow_samples: int = 5
    replan_action_steps: int = 2
    dropout: float = 0.0
    action_history_conditioning: bool = False
    gripper_action_conditioning: bool = False
    gripper_previous_action_mode: str = "full"
    gripper_control_mode: str = "absolute"
    gripper_hidden_dim: int = 256
    gripper_layers: int = 2
    gripper_heads: int = 4
    gripper_event_change_weight: float = 1.0
    flow_endpoint_loss_weight: float = 0.0
    flow_prefix_weight: float = 1.0
    flow_gripper_transition_weight: float = 1.0
    # Keep the Flow residual and reduction in FP32 when the surrounding
    # forward pass runs under BF16 autocast.
    flow_loss_fp32: bool = False
    gripper_loss_weight: float = 0.05
    gripper_transition_loss_weight: float = 0.02
    # Explicitly overridden by the V2.1 training recipe from train-only
    # transition counts.  A neutral default preserves old checkpoint metadata.
    gripper_transition_positive_weight: float = 1.0
    gripper_event_class_weights: tuple[float, float, float] = (1.0, 1.0, 1.0)
    gripper_event_focal_gamma: float = 0.0
    gripper_event_loss_weight: float = 0.05
    gripper_state_aux_loss_weight: float = 0.01
    gripper_event_probability_threshold: float = 0.5
    gripper_open_threshold: float = 0.4
    gripper_close_threshold: float = 0.6
    gripper_transition_threshold: float = 0.5
    direct_loss_weight: float = 1.0
    instruction_loss_weight: float = 1.0
    monotonic_loss_weight: float = 0.1
    residual_gate_init: float = 0.05
    arm_trust_region: float = 0.35
    gripper_trust_region: float = 0.0
    gripper_open_value: float = -1.0
    gripper_close_value: float = 1.0
    # V2.2 G0 keeps gripper inside the same continuous 7-D Flow output and
    # removes the separate event/FSM override used by the failed V2.1 R1 line.
    unified_flow_gripper: bool = False
    # V2.5 stores one exact layer-14 8x8 raster while retaining four
    # instruction summaries.  The two axes intentionally have different
    # layer counts; semantic adapters remain shared with the corresponding
    # Qwen layers and the visual branch reuses the layer-14 adapter.
    highres_layer14_visual: bool = False
    # V2.6 removes the rank-128/context-preencoder bottleneck.  It projects
    # frozen Qwen tokens at full rank, keeps four semantic tokens explicit,
    # and restores the proven alternating cross/self action-expert pattern.
    dense_interleaved_flow: bool = False
    # V3.23 keeps the dense interleaved action expert but replaces the single
    # layer-14 8x8 raster with four Qwen layers at 4x4 per camera.  The total
    # number of visual tokens stays 128, so this isolates representation depth
    # without increasing the action expert's cross-attention length.
    dense_multilayer_spatial_flow: bool = False
    # V3.24 preserves the exact V2.6 layer-14 8x8 raster and adds a learned
    # residual made from layers 12/14/18/final at 4x4.  This tests multi-layer
    # Qwen spatial information without sacrificing the high-resolution anchor.
    multilayer_spatial_residual: bool = False
    multilayer_spatial_rank: int = 64
    multilayer_spatial_residual_limit: float = 0.10
    # V3.25 keeps the exact V2.6 generator frozen and learns a bounded
    # observation-feedback correction from filtered train-side deviations.
    observation_feedback_recovery: bool = False
    recovery_hidden_dim: int = 384
    recovery_layers: int = 4
    recovery_heads: int = 6
    recovery_arm_trust_region: float = 1.0
    recovery_gripper_trust_region: float = 2.2
    recovery_gate_init: float = 0.01

    def __post_init__(self) -> None:
        positive = (
            self.feature_dim,
            self.feature_layers,
            self.camera_count,
            self.spatial_tokens,
            self.semantic_tokens,
            self.state_dim,
            self.action_dim,
            self.action_horizon,
            self.adapter_rank,
            self.hidden_dim,
            self.expert_layers,
            self.expert_heads,
            self.residual_layers,
            self.history_length,
            self.history_stride,
            self.flow_steps,
            self.flow_samples,
            self.replan_action_steps,
            self.multilayer_spatial_rank,
            self.recovery_hidden_dim,
            self.recovery_layers,
            self.recovery_heads,
            self.gripper_hidden_dim,
            self.gripper_layers,
            self.gripper_heads,
            self.language_max_tokens,
        )
        if any(value <= 0 for value in positive):
            raise ValueError("all Duvla V2.1 dimensions and counts must be positive")
        if self.context_layers < 0:
            raise ValueError("context_layers must be non-negative")
        if self.ordered_language_bridge_mode not in {"scalar_gate", "zero_output"}:
            raise ValueError("unsupported ordered language bridge mode")
        if not 0.0 < self.language_residual_scale <= 1.0:
            raise ValueError("language residual scale must be in (0, 1]")
        if self.language_max_tokens < 32:
            raise ValueError("language_max_tokens must preserve the trained 32-token contract")
        if not 0.0 < self.cross_camera_residual_scale <= 1.0:
            raise ValueError("cross-camera residual scale must be in (0, 1]")
        if self.candidate_aggregation not in {"coordinate_median", "trajectory_medoid"}:
            raise ValueError("unsupported candidate aggregation")
        if self.highres_layer14_visual and self.dense_multilayer_spatial_flow:
            raise ValueError("high-resolution and dense multi-layer spatial modes are exclusive")
        if self.dense_multilayer_spatial_flow and not self.dense_interleaved_flow:
            raise ValueError("dense multi-layer spatial mode requires dense interleaved Flow")
        if self.multilayer_spatial_residual and not (
            self.highres_layer14_visual and self.dense_interleaved_flow
        ):
            raise ValueError(
                "multi-layer spatial residual requires the high-resolution dense path"
            )
        if self.dense_interleaved_flow and not (
            self.highres_layer14_visual or self.dense_multilayer_spatial_flow
        ):
            raise ValueError(
                "dense interleaved Flow requires high-resolution layer-14 or multi-layer spatial tokens"
            )
        if self.semantic_tokens != 1:
            raise ValueError("V2.1 requires exactly one semantic token per layer")
        if self.hidden_dim % self.expert_heads:
            raise ValueError("hidden_dim must be divisible by expert_heads")
        if self.gripper_hidden_dim % self.gripper_heads:
            raise ValueError("gripper_hidden_dim must be divisible by gripper_heads")
        if self.recovery_hidden_dim % self.recovery_heads:
            raise ValueError("recovery_hidden_dim must be divisible by recovery_heads")
        if self.replan_action_steps > self.action_horizon:
            raise ValueError("replan_action_steps cannot exceed action_horizon")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if not 0.0 < self.residual_gate_init < 1.0:
            raise ValueError("residual_gate_init must be in (0, 1)")
        if not 0.0 < self.multilayer_spatial_residual_limit <= 1.0:
            raise ValueError("multi-layer spatial residual limit must be in (0, 1]")
        if min(self.recovery_arm_trust_region, self.recovery_gripper_trust_region) <= 0.0:
            raise ValueError("recovery trust regions must be positive")
        if not 0.0 < self.recovery_gate_init < 1.0:
            raise ValueError("recovery_gate_init must be in (0, 1)")
        if not (
            0.0 < self.gripper_open_threshold < 0.5
            < self.gripper_close_threshold < 1.0
            and 0.0 < self.gripper_transition_threshold < 1.0
        ):
            raise ValueError("gripper hysteresis thresholds are invalid")
        if self.gripper_loss_weight < 0.0 or self.gripper_transition_loss_weight < 0.0:
            raise ValueError("gripper loss weights must be non-negative")
        if self.gripper_transition_positive_weight <= 0.0:
            raise ValueError("gripper transition positive weight must be positive")
        if self.gripper_event_change_weight < 1.0:
            raise ValueError("gripper_event_change_weight must be at least one")
        if self.flow_endpoint_loss_weight < 0.0:
            raise ValueError("flow_endpoint_loss_weight must be non-negative")
        if self.flow_prefix_weight < 1.0:
            raise ValueError("flow_prefix_weight must be at least one")
        if self.flow_gripper_transition_weight < 1.0:
            raise ValueError("flow_gripper_transition_weight must be at least one")
        if self.gripper_previous_action_mode not in {"full", "gripper_only"}:
            raise ValueError(
                "gripper_previous_action_mode must be full or gripper_only"
            )
        if self.gripper_control_mode not in {"absolute", "event3"}:
            raise ValueError("gripper_control_mode must be absolute or event3")
        if (
            len(self.gripper_event_class_weights) != 3
            or any(value <= 0.0 for value in self.gripper_event_class_weights)
        ):
            raise ValueError("gripper_event_class_weights must contain three positive values")
        if self.gripper_event_focal_gamma < 0.0:
            raise ValueError("gripper_event_focal_gamma must be non-negative")
        if self.gripper_event_loss_weight < 0.0 or self.gripper_state_aux_loss_weight < 0.0:
            raise ValueError("gripper event/state loss weights must be non-negative")
        if not 0.0 < self.gripper_event_probability_threshold < 1.0:
            raise ValueError("gripper_event_probability_threshold must be in (0, 1)")

    @property
    def context_tokens(self) -> int:
        visual_tokens = self.camera_count * self.spatial_tokens
        if self.dense_multilayer_spatial_flow:
            visual_tokens *= self.feature_layers
        return (
            visual_tokens
            + (self.feature_layers if self.dense_interleaved_flow else self.semantic_tokens)
            + self.history_length
            - 1
        )

    @property
    def visual_feature_layers(self) -> int:
        return 1 if self.highres_layer14_visual else self.feature_layers


class _LowRankLayerAdapter(nn.Module):
    def __init__(self, feature_dim: int, rank: int, hidden_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.RMSNorm(feature_dim, eps=1e-6),
            nn.Linear(feature_dim, rank, bias=False),
            nn.SiLU(),
            nn.Linear(rank, hidden_dim, bias=False),
        )

    def forward(self, values: Tensor) -> Tensor:
        return self.net(values)


class _DenseLayerAdapter(nn.Module):
    def __init__(self, feature_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.RMSNorm(feature_dim, eps=1e-6),
            nn.Linear(feature_dim, hidden_dim, bias=False),
        )

    def forward(self, values: Tensor) -> Tensor:
        return self.net(values)


class _MultiLayerSpatialResidual(nn.Module):
    """Fuse coarse Qwen layers into a bounded residual on an 8x8 anchor."""

    def __init__(self, config: DuvlaV21Config) -> None:
        super().__init__()
        self.config = config
        self.adapters = nn.ModuleList(
            _LowRankLayerAdapter(
                config.feature_dim,
                config.multilayer_spatial_rank,
                config.hidden_dim,
            )
            for _ in range(config.feature_layers)
        )
        gate_hidden = max(config.hidden_dim // 4, 64)
        self.gates = nn.Sequential(
            nn.RMSNorm(2 * config.hidden_dim, eps=1e-6),
            nn.Linear(2 * config.hidden_dim, gate_hidden),
            nn.SiLU(),
            nn.Linear(gate_hidden, 1),
        )
        self.output = nn.Sequential(
            nn.RMSNorm(config.hidden_dim, eps=1e-6),
            nn.Linear(config.hidden_dim, config.hidden_dim, bias=False),
        )
        output = self.output[-1]
        if not isinstance(output, nn.Linear):  # pragma: no cover - defensive
            raise RuntimeError("multi-layer residual output must be linear")
        nn.init.zeros_(output.weight)

    def forward(
        self,
        auxiliary_visual: Tensor,
        semantic_by_layer: Tensor,
        state_token: Tensor,
    ) -> tuple[Tensor, Tensor]:
        cfg = self.config
        expected = (
            auxiliary_visual.ndim == 5
            and tuple(auxiliary_visual.shape[1:4])
            == (cfg.feature_layers, cfg.camera_count, 16)
            and auxiliary_visual.shape[-1] == cfg.feature_dim
        )
        if not expected:
            raise ValueError(
                "auxiliary multi-layer visual must have shape [B,L,C,16,D]"
            )
        adapted = torch.stack(
            [adapter(auxiliary_visual[:, index]) for index, adapter in enumerate(self.adapters)],
            dim=1,
        )
        pooled = adapted.mean(dim=3)
        gate_input = torch.cat(
            (
                pooled,
                (semantic_by_layer + state_token[:, None, :])[:, :, None, :]
                .expand_as(pooled),
            ),
            dim=-1,
        )
        weights = self.gates(gate_input).squeeze(-1).softmax(dim=1)
        coarse = (adapted * weights[:, :, :, None, None]).sum(dim=1)
        side = math.isqrt(coarse.shape[2])
        target_side = math.isqrt(cfg.spatial_tokens)
        if side * side != coarse.shape[2] or target_side * target_side != cfg.spatial_tokens:
            raise ValueError("spatial token counts must form square grids")
        if target_side % side:
            raise ValueError("auxiliary grid must evenly divide the high-resolution grid")
        scale = target_side // side
        upsampled = coarse.reshape(
            coarse.shape[0], cfg.camera_count, side, side, cfg.hidden_dim
        ).repeat_interleave(scale, dim=2).repeat_interleave(scale, dim=3)
        upsampled = upsampled.reshape(
            coarse.shape[0], cfg.camera_count, cfg.spatial_tokens, cfg.hidden_dim
        )
        residual = cfg.multilayer_spatial_residual_limit * torch.tanh(
            self.output(upsampled)
        )
        return residual, weights


class MultiLayerQwenFusion(nn.Module):
    """Fuse layers dynamically while preserving camera and raster positions."""

    def __init__(self, config: DuvlaV21Config) -> None:
        super().__init__()
        self.config = config
        if config.dense_interleaved_flow:
            self.adapters = nn.ModuleList(
                _DenseLayerAdapter(config.feature_dim, config.hidden_dim)
                for _ in range(config.feature_layers)
            )
        else:
            self.adapters = nn.ModuleList(
                _LowRankLayerAdapter(config.feature_dim, config.adapter_rank, config.hidden_dim)
                for _ in range(config.feature_layers)
            )
        gate_hidden = max(config.hidden_dim // 2, 32)
        self.visual_gate = (
            None
            if config.dense_interleaved_flow
            else nn.Sequential(
                nn.RMSNorm(3 * config.hidden_dim, eps=1e-6),
                nn.Linear(3 * config.hidden_dim, gate_hidden),
                nn.SiLU(),
                nn.Linear(gate_hidden, 1),
            )
        )
        self.semantic_gate = (
            None
            if config.dense_interleaved_flow
            else nn.Sequential(
                nn.RMSNorm(2 * config.hidden_dim, eps=1e-6),
                nn.Linear(2 * config.hidden_dim, gate_hidden),
                nn.SiLU(),
                nn.Linear(gate_hidden, 1),
            )
        )
        self.state_projection = (
            nn.Sequential(
                nn.RMSNorm(config.state_dim, eps=1e-6),
                nn.Linear(config.state_dim, config.hidden_dim),
            )
            if config.dense_interleaved_flow
            else nn.Sequential(
                nn.RMSNorm(config.state_dim, eps=1e-6),
                nn.Linear(config.state_dim, config.hidden_dim),
                nn.SiLU(),
                nn.Linear(config.hidden_dim, config.hidden_dim),
            )
        )
        self.layer_embedding = nn.Parameter(
            torch.randn(config.feature_layers, config.hidden_dim) * 0.02
        )
        self.camera_embedding = nn.Parameter(
            torch.randn(config.camera_count, config.hidden_dim) * 0.02
        )
        self.spatial_embedding = nn.Parameter(
            torch.randn(config.spatial_tokens, config.hidden_dim) * 0.02
        )
        self.semantic_embedding = nn.Parameter(torch.randn(1, config.hidden_dim) * 0.02)
        self.multilayer_spatial_residual = (
            _MultiLayerSpatialResidual(config)
            if config.multilayer_spatial_residual
            else None
        )

    def forward(
        self,
        visual: Tensor,
        semantic: Tensor,
        state: Tensor,
        *,
        auxiliary_visual: Tensor | None = None,
        layer14_only: bool = False,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        cfg = self.config
        expected_visual = (
            visual.ndim == 5
            and tuple(visual.shape[1:4])
            == (cfg.visual_feature_layers, cfg.camera_count, cfg.spatial_tokens)
            and visual.shape[-1] == cfg.feature_dim
        )
        expected_semantic = (
            semantic.ndim == 4
            and tuple(semantic.shape[1:3]) == (cfg.feature_layers, cfg.semantic_tokens)
            and semantic.shape[-1] == cfg.feature_dim
        )
        if not expected_visual:
            raise ValueError("visual must have shape [B, layers, cameras, spatial_tokens, D]")
        if not expected_semantic:
            raise ValueError("semantic must have shape [B, layers, 1, D]")
        if tuple(state.shape) != (visual.shape[0], cfg.state_dim):
            raise ValueError("state must have shape [B, state_dim]")

        if cfg.highres_layer14_visual:
            if cfg.feature_layers < 2:
                raise ValueError("high-resolution visual mode requires cached layer 14")
            adapted_visual = self.adapters[1](visual[:, 0])[:, None]
            visual_layer_embedding = self.layer_embedding[1:2]
        else:
            adapted_visual = torch.stack(
                [adapter(visual[:, index]) for index, adapter in enumerate(self.adapters)],
                dim=1,
            )
            visual_layer_embedding = self.layer_embedding
        adapted_semantic = torch.stack(
            [adapter(semantic[:, index]) for index, adapter in enumerate(self.adapters)],
            dim=1,
        )
        layer_bias = visual_layer_embedding[None, :, None, None, :]
        adapted_visual = adapted_visual + layer_bias
        adapted_semantic = adapted_semantic + self.layer_embedding[None, :, None, :]
        state_token = self.state_projection(state)

        if cfg.dense_interleaved_flow:
            if cfg.dense_multilayer_spatial_flow:
                fused_visual = (
                    adapted_visual
                    + self.camera_embedding[None, None, :, None, :]
                    + self.spatial_embedding[None, None, None, :, :]
                ).flatten(1, 2)
            else:
                fused_visual = adapted_visual[:, 0]
                fused_visual = (
                    fused_visual
                    + self.camera_embedding[None, :, None, :]
                    + self.spatial_embedding[None, None, :, :]
                )
            semantic_tokens = adapted_semantic[:, :, 0] + self.semantic_embedding
            if self.multilayer_spatial_residual is not None:
                if auxiliary_visual is None:
                    raise ValueError("multi-layer spatial residual requires auxiliary_visual")
                residual, residual_weights = self.multilayer_spatial_residual(
                    auxiliary_visual,
                    semantic_tokens,
                    state_token,
                )
                fused_visual = fused_visual + residual
                visual_weights = residual_weights
            elif auxiliary_visual is not None:
                raise ValueError("auxiliary_visual requires multi-layer spatial residual mode")
            visual_weights = (
                torch.ones(
                    visual.shape[0],
                    cfg.feature_layers if cfg.dense_multilayer_spatial_flow else 1,
                    cfg.camera_count,
                    device=visual.device,
                    dtype=adapted_visual.dtype,
                )
                if self.multilayer_spatial_residual is None
                else visual_weights
            )
            return fused_visual, semantic_tokens, state_token, visual_weights

        camera_summary = adapted_visual.mean(dim=3)
        semantic_by_layer = adapted_semantic[:, :, 0]
        visual_semantic = (
            semantic_by_layer[:, 1:2]
            if cfg.highres_layer14_visual
            else semantic_by_layer
        )
        visual_gate_input = torch.cat(
            (
                camera_summary,
                visual_semantic[:, :, None, :].expand_as(camera_summary),
                state_token[:, None, None, :].expand_as(camera_summary),
            ),
            dim=-1,
        )
        if self.visual_gate is None or self.semantic_gate is None:  # pragma: no cover
            raise RuntimeError("dynamic fusion gates are unavailable")
        visual_scores = self.visual_gate(visual_gate_input).squeeze(-1)
        semantic_scores = self.semantic_gate(
            torch.cat(
                (
                    semantic_by_layer,
                    state_token[:, None, :].expand_as(semantic_by_layer),
                ),
                dim=-1,
            )
        ).squeeze(-1)
        if layer14_only and not cfg.highres_layer14_visual:
            if cfg.feature_layers < 2:
                raise ValueError("layer14 control requires the second cached layer")
            visual_scores = torch.full_like(visual_scores, -torch.inf)
            visual_scores[:, 1] = 0.0
            semantic_scores = torch.full_like(semantic_scores, -torch.inf)
            semantic_scores[:, 1] = 0.0
        visual_weights = visual_scores.softmax(dim=1)
        semantic_weights = semantic_scores.softmax(dim=1)
        fused_visual = (
            adapted_visual * visual_weights[:, :, :, None, None]
        ).sum(dim=1)
        fused_visual = (
            fused_visual
            + self.camera_embedding[None, :, None, :]
            + self.spatial_embedding[None, None, :, :]
        )
        fused_semantic = (
            semantic_by_layer * semantic_weights[:, :, None]
        ).sum(dim=1)
        fused_semantic = fused_semantic + self.semantic_embedding
        return fused_visual, fused_semantic, state_token, visual_weights


class _SwiGLUContextBlock(nn.Module):
    def __init__(self, hidden_dim: int, heads: int, dropout: float) -> None:
        super().__init__()
        self.attention_norm = nn.RMSNorm(hidden_dim, eps=1e-6)
        self.attention = nn.MultiheadAttention(
            hidden_dim, heads, dropout=dropout, bias=False, batch_first=True
        )
        self.ffn_norm = nn.RMSNorm(hidden_dim, eps=1e-6)
        intermediate = 256 * math.ceil((8 * hidden_dim / 3) / 256)
        self.gate = nn.Linear(hidden_dim, intermediate, bias=False)
        self.up = nn.Linear(hidden_dim, intermediate, bias=False)
        self.down = nn.Linear(intermediate, hidden_dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, values: Tensor) -> Tensor:
        normalized = self.attention_norm(values)
        attended, _ = self.attention(normalized, normalized, normalized, need_weights=False)
        values = values + self.dropout(attended)
        normalized = self.ffn_norm(values)
        feed_forward = torch.nn.functional.silu(self.gate(normalized)) * self.up(normalized)
        return values + self.down(self.dropout(feed_forward))


class _RecoveryCrossBlock(nn.Module):
    """Cross-attend action queries to the frozen policy's observation context."""

    def __init__(self, hidden_dim: int, heads: int, dropout: float) -> None:
        super().__init__()
        self.query_norm = nn.RMSNorm(hidden_dim, eps=1e-6)
        self.context_norm = nn.RMSNorm(hidden_dim, eps=1e-6)
        self.cross_attention = nn.MultiheadAttention(
            hidden_dim, heads, dropout=dropout, bias=False, batch_first=True
        )
        self.self_block = _SwiGLUContextBlock(hidden_dim, heads, dropout)

    def forward(self, actions: Tensor, context: Tensor) -> Tensor:
        attended, _ = self.cross_attention(
            self.query_norm(actions),
            self.context_norm(context),
            self.context_norm(context),
            need_weights=False,
        )
        return self.self_block(actions + attended)


class ObservationFeedbackRecoveryHead(nn.Module):
    """Predict a gated bounded correction without changing the frozen parent."""

    def __init__(self, config: DuvlaV21Config) -> None:
        super().__init__()
        self.config = config
        hidden = config.recovery_hidden_dim
        self.context_projection = nn.Sequential(
            nn.RMSNorm(config.hidden_dim, eps=1e-6),
            nn.Linear(config.hidden_dim, hidden, bias=False),
        )
        self.state_projection = nn.Sequential(
            nn.RMSNorm(config.state_dim, eps=1e-6),
            nn.Linear(config.state_dim, hidden),
        )
        self.action_projection = nn.Linear(config.action_dim, hidden)
        self.action_position = nn.Parameter(
            torch.randn(config.action_horizon, hidden) * 0.02
        )
        self.blocks = nn.ModuleList(
            _RecoveryCrossBlock(hidden, config.recovery_heads, config.dropout)
            for _ in range(config.recovery_layers)
        )
        self.output = nn.Linear(hidden, config.action_dim)
        self.gate = nn.Linear(hidden, 1)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)
        nn.init.zeros_(self.gate.weight)
        self.gate.bias.data.fill_(
            math.log(config.recovery_gate_init)
            - math.log1p(-config.recovery_gate_init)
        )
        limits = torch.full(
            (config.action_dim,), config.recovery_arm_trust_region
        )
        limits[-1] = config.recovery_gripper_trust_region
        self.register_buffer("trust_region", limits, persistent=True)

    def forward(
        self,
        context: Tensor,
        state: Tensor,
        base_actions: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        cfg = self.config
        if (
            context.ndim != 3
            or context.shape[0] != base_actions.shape[0]
            or context.shape[-1] != cfg.hidden_dim
        ):
            raise ValueError("recovery context must have shape [B,T,hidden_dim]")
        if tuple(state.shape) != (base_actions.shape[0], cfg.state_dim):
            raise ValueError("recovery state must have shape [B,state_dim]")
        if tuple(base_actions.shape[1:]) != (cfg.action_horizon, cfg.action_dim):
            raise ValueError("recovery base actions have an invalid shape")
        projected_context = self.context_projection(context)
        tokens = (
            self.action_projection(base_actions)
            + self.state_projection(state)[:, None, :]
            + self.action_position[None, :, :]
        )
        for block in self.blocks:
            tokens = block(tokens, projected_context)
        residual = torch.tanh(self.output(tokens)) * self.trust_region
        gate = torch.sigmoid(self.gate(tokens))
        corrected = base_actions + gate * residual
        return corrected, residual, gate


class _ProgressiveResidualHead(nn.Module):
    def __init__(self, config: DuvlaV21Config, *, instruction_conditioned: bool) -> None:
        super().__init__()
        self.config = config
        self.instruction_conditioned = instruction_conditioned
        self.action_projection = nn.Linear(config.action_dim, config.hidden_dim)
        condition_inputs = 3 if instruction_conditioned else 2
        self.condition_projection = nn.Sequential(
            nn.RMSNorm(condition_inputs * config.hidden_dim, eps=1e-6),
            nn.Linear(condition_inputs * config.hidden_dim, config.hidden_dim),
            nn.SiLU(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
        )
        self.action_position = nn.Parameter(
            torch.randn(config.action_horizon, config.hidden_dim) * 0.02
        )
        self.blocks = nn.ModuleList(
            _SwiGLUContextBlock(config.hidden_dim, config.expert_heads, config.dropout)
            for _ in range(config.residual_layers)
        )
        self.output = nn.Linear(config.hidden_dim, config.action_dim)
        self.gate = nn.Linear(config.hidden_dim, config.action_dim)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)
        nn.init.zeros_(self.gate.weight)
        self.gate.bias.data.fill_(
            math.log(config.residual_gate_init) - math.log1p(-config.residual_gate_init)
        )
        limits = torch.full((config.action_dim,), config.arm_trust_region)
        limits[-1] = config.gripper_trust_region
        self.register_buffer("trust_region", limits, persistent=True)

    def forward(
        self,
        parent_actions: Tensor,
        context_summary: Tensor,
        state_token: Tensor,
        semantic_token: Tensor,
    ) -> Tensor:
        condition_parts = [context_summary, state_token]
        if self.instruction_conditioned:
            condition_parts.append(semantic_token)
        condition = self.condition_projection(torch.cat(condition_parts, dim=-1))
        tokens = (
            self.action_projection(parent_actions)
            + condition[:, None, :]
            + self.action_position[None, :, :]
        )
        for block in self.blocks:
            tokens = block(tokens)
        delta = torch.tanh(self.output(tokens)) * self.trust_region
        return delta * torch.sigmoid(self.gate(tokens))


class _ActionConditionedGripperHead(nn.Module):
    """Predict gripper control and auxiliary state from the generated arm plan."""

    def __init__(self, config: DuvlaV21Config) -> None:
        super().__init__()
        self.config = config
        hidden = config.gripper_hidden_dim
        self.condition_projection = nn.Sequential(
            nn.RMSNorm(3 * config.hidden_dim, eps=1e-6),
            nn.Linear(3 * config.hidden_dim, hidden),
            nn.SiLU(),
        )
        self.arm_projection = nn.Linear(config.action_dim - 1, hidden)
        previous_action_dim = (
            config.action_dim
            if config.gripper_previous_action_mode == "full"
            else 1
        )
        self.previous_action_projection = nn.Linear(previous_action_dim, hidden)
        self.action_position = nn.Parameter(
            torch.randn(config.action_horizon, hidden) * 0.02
        )
        self.blocks = nn.ModuleList(
            _SwiGLUContextBlock(hidden, config.gripper_heads, config.dropout)
            for _ in range(config.gripper_layers)
        )
        event_classes = 3 if config.gripper_control_mode == "event3" else 1
        self.event_output = nn.Linear(hidden, event_classes)
        self.transition_output = nn.Linear(hidden, 1)

    def forward(
        self,
        actions: Tensor,
        context_summary: Tensor,
        state_token: Tensor,
        semantic_token: Tensor,
        previous_action: Tensor,
    ) -> tuple[Tensor, Tensor]:
        condition = self.condition_projection(
            torch.cat((context_summary, state_token, semantic_token), dim=-1)
        )
        gripper_history = (
            previous_action
            if self.config.gripper_previous_action_mode == "full"
            else previous_action[:, -1:]
        )
        tokens = (
            self.arm_projection(actions[:, :, :-1])
            + self.previous_action_projection(gripper_history)[:, None, :]
            + condition[:, None, :]
            + self.action_position[None, :, :]
        )
        for block in self.blocks:
            tokens = block(tokens)
        event_logits = self.event_output(tokens)
        if self.config.gripper_control_mode == "absolute":
            event_logits = event_logits.squeeze(-1)
        return event_logits, self.transition_output(tokens).squeeze(-1)


class DuvlaV21Policy(nn.Module):
    """Fresh V2.1 policy with no benchmark-index input surface."""

    def __init__(self, config: DuvlaV21Config | None = None) -> None:
        super().__init__()
        self.config = config or DuvlaV21Config()
        cfg = self.config
        self.fusion = MultiLayerQwenFusion(cfg)
        self.history_position = nn.Parameter(
            torch.randn(cfg.history_length - 1, cfg.hidden_dim) * 0.02
        )
        self.previous_action_projection = (
            nn.Sequential(
                nn.RMSNorm(cfg.action_dim, eps=1e-6),
                nn.Linear(cfg.action_dim, cfg.hidden_dim),
                nn.SiLU(),
                nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
            )
            if cfg.action_history_conditioning
            else None
        )
        self.context_blocks = nn.ModuleList(
            _SwiGLUContextBlock(cfg.hidden_dim, cfg.expert_heads, cfg.dropout)
            for _ in range(cfg.context_layers)
        )
        flow_config = FlowMatchingVLAConfig(
            state_dim=cfg.state_dim,
            action_dim=cfg.action_dim,
            vlm_feature_dim=cfg.feature_dim,
            hidden_dim=cfg.hidden_dim,
            action_horizon=cfg.action_horizon,
            max_context_tokens=cfg.context_tokens,
            expert_layers=cfg.expert_layers,
            expert_heads=cfg.expert_heads,
            dropout=cfg.dropout,
            num_flow_steps=cfg.flow_steps,
            self_attn_every_n_layers=2,
            time_sampling="beta_1.5_1.0",
            architecture_variant="smol_aligned",
            attention_pattern=(
                "alternating" if cfg.dense_interleaved_flow else "periodic_cross_self"
            ),
            ffn_type="swiglu",
            norm_type="rms_norm",
            attention_bias=False,
            causal_action_attention=cfg.causal_action_attention,
            state_projection_layers=1 if cfg.dense_interleaved_flow else 2,
            context_layout="merged",
            qwen_context_mode="ordered",
        )
        self.flow_expert = FlowMatchingActionExpert(flow_config)
        self.direct_residual = _ProgressiveResidualHead(
            cfg, instruction_conditioned=False
        )
        self.instruction_residual = _ProgressiveResidualHead(
            cfg, instruction_conditioned=True
        )
        if cfg.unified_flow_gripper:
            if cfg.gripper_action_conditioning:
                raise ValueError(
                    "unified_flow_gripper cannot use a separate action-conditioned head"
                )
            self.action_conditioned_gripper = None
            self.gripper_event_head = None
            self.gripper_transition_head = None
        elif cfg.gripper_action_conditioning:
            self.action_conditioned_gripper = _ActionConditionedGripperHead(cfg)
            self.gripper_event_head = None
            self.gripper_transition_head = None
        else:
            self.action_conditioned_gripper = None
            self.gripper_event_head = nn.Sequential(
                nn.RMSNorm(2 * cfg.hidden_dim, eps=1e-6),
                nn.Linear(2 * cfg.hidden_dim, cfg.hidden_dim),
                nn.SiLU(),
                nn.Linear(cfg.hidden_dim, cfg.action_horizon),
            )
            self.gripper_transition_head = nn.Sequential(
                nn.RMSNorm(2 * cfg.hidden_dim, eps=1e-6),
                nn.Linear(2 * cfg.hidden_dim, cfg.hidden_dim),
                nn.SiLU(),
                nn.Linear(cfg.hidden_dim, cfg.action_horizon - 1),
            )
        self.observation_feedback_recovery = (
            ObservationFeedbackRecoveryHead(cfg)
            if cfg.observation_feedback_recovery
            else None
        )
        # Construct last: shared base initialization remains unchanged by adding this module.
        if cfg.ordered_language_bridge:
            from duvla.models.ordered_language_bridge import OrderedLanguageBridge
            self.language_bridge = OrderedLanguageBridge(
                cfg.feature_dim,
                cfg.hidden_dim,
                cfg.expert_heads,
                cfg.camera_count,
                max_tokens=cfg.language_max_tokens,
                mode=cfg.ordered_language_bridge_mode,
                residual_scale=cfg.language_residual_scale,
            )
        else:
            self.language_bridge = None
        if cfg.cross_camera_fusion:
            from duvla.models.robust_camera_fusion import RobustCrossCameraFusion
            self.cross_camera_bridge = RobustCrossCameraFusion(
                cfg.hidden_dim,
                cfg.expert_heads,
                cfg.camera_count,
                residual_scale=cfg.cross_camera_residual_scale,
            )
        else:
            self.cross_camera_bridge = None


    def _encode_context(
        self,
        visual: Tensor,
        semantic: Tensor,
        state: Tensor,
        *,
        auxiliary_visual: Tensor | None = None,
        language_tokens: Tensor | None = None,
        language_mask: Tensor | None = None,
        history_visual: Tensor | None = None,
        history_auxiliary_visual: Tensor | None = None,
        history_semantic: Tensor | None = None,
        history_states: Tensor | None = None,
        previous_action: Tensor | None = None,
        history_previous_actions: Tensor | None = None,
        layer14_only: bool = False,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        fused_visual, semantic_token, state_token, layer_weights = self.fusion(
            visual,
            semantic,
            state,
            auxiliary_visual=auxiliary_visual,
            layer14_only=layer14_only,
        )
        if self.config.action_history_conditioning:
            if self.previous_action_projection is None:  # pragma: no cover
                raise RuntimeError("action history projection is unavailable")
            if previous_action is None or tuple(previous_action.shape) != (
                visual.shape[0],
                self.config.action_dim,
            ):
                raise ValueError("previous_action must have shape [batch, action_dim]")
            state_token = state_token + self.previous_action_projection(previous_action)
        current_visual = fused_visual.flatten(1, 2)
        if self.language_bridge is not None:
            if language_tokens is None or language_mask is None:
                raise ValueError('V3.29 requires ordered language tokens and mask')
            current_visual = self.language_bridge(current_visual, language_tokens, language_mask)
        semantic_summary = (
            semantic_token.mean(dim=1) if semantic_token.ndim == 3 else semantic_token
        )
        current_semantic = (
            semantic_token if semantic_token.ndim == 3 else semantic_token[:, None, :]
        )
        if self.cross_camera_bridge is not None:
            current_visual = self.cross_camera_bridge(
                current_visual,
                semantic_summary,
                state_token,
            )
        history_count = self.config.history_length - 1
        if history_visual is None or history_semantic is None or history_states is None:
            summary = current_visual.mean(dim=1) + semantic_summary + state_token
            history_tokens = summary[:, None, :].expand(-1, history_count, -1)
        else:
            expected = (
                history_visual.ndim == 6
                and history_visual.shape[1] == history_count
                and history_semantic.ndim == 5
                and history_semantic.shape[1] == history_count
                and tuple(history_states.shape[:2]) == (visual.shape[0], history_count)
            )
            if self.config.multilayer_spatial_residual:
                expected = (
                    expected
                    and history_auxiliary_visual is not None
                    and history_auxiliary_visual.ndim == 6
                    and tuple(history_auxiliary_visual.shape[:2])
                    == (visual.shape[0], history_count)
                )
            if self.config.action_history_conditioning:
                expected = expected and history_previous_actions is not None and tuple(
                    history_previous_actions.shape
                ) == (visual.shape[0], history_count, self.config.action_dim)
            if not expected:
                raise ValueError("history tensors do not match V2.1 history contract")
            batch = visual.shape[0]
            hist_visual = history_visual.flatten(0, 1)
            hist_semantic = history_semantic.flatten(0, 1)
            hist_state = history_states.flatten(0, 1)
            hist_auxiliary = (
                None
                if history_auxiliary_visual is None
                else history_auxiliary_visual.flatten(0, 1)
            )
            hist_fused, hist_language, hist_state_token, _ = self.fusion(
                hist_visual,
                hist_semantic,
                hist_state,
                auxiliary_visual=hist_auxiliary,
                layer14_only=layer14_only,
            )
            if self.config.action_history_conditioning:
                if self.previous_action_projection is None or history_previous_actions is None:
                    raise RuntimeError("history action projection is unavailable")
                hist_state_token = hist_state_token + self.previous_action_projection(
                    history_previous_actions.flatten(0, 1)
                )
            hist_language_summary = (
                hist_language.mean(dim=1) if hist_language.ndim == 3 else hist_language
            )
            history_tokens = (
                hist_fused.mean(dim=(1, 2)) + hist_language_summary + hist_state_token
            ).reshape(batch, history_count, self.config.hidden_dim)
        history_tokens = history_tokens + self.history_position[None, :, :]
        context = torch.cat((current_visual, current_semantic, history_tokens), dim=1)
        for block in self.context_blocks:
            context = block(context)
        return context, state_token[:, None, :], semantic_summary, layer_weights

    def flow_loss_components(
        self,
        visual: Tensor,
        semantic: Tensor,
        state: Tensor,
        target_actions: Tensor,
        valid_mask: Tensor,
        *,
        auxiliary_visual: Tensor | None = None,
        language_tokens: Tensor | None = None,
        language_mask: Tensor | None = None,
        history_visual: Tensor | None = None,
        history_auxiliary_visual: Tensor | None = None,
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
        if stage not in {"flow", "gripper", "direct", "instruction", "joint"}:
            raise ValueError("stage must be flow, gripper, direct, instruction, or joint")
        if tuple(target_actions.shape[1:]) != (
            self.config.action_horizon,
            self.config.action_dim,
        ) or tuple(valid_mask.shape) != tuple(target_actions.shape[:2]):
            raise ValueError("target_actions or valid_mask has an invalid shape")
        context, state_token, language, _weights = self._encode_context(
            visual,
            semantic,
            state,
            auxiliary_visual=auxiliary_visual,
            language_tokens=language_tokens,
            language_mask=language_mask,
            history_visual=history_visual,
            history_auxiliary_visual=history_auxiliary_visual,
            history_semantic=history_semantic,
            history_states=history_states,
            previous_action=previous_actions,
            history_previous_actions=history_previous_actions,
            layer14_only=layer14_only,
        )
        if sample_weights is None:
            sample_weights = torch.ones(
                target_actions.shape[0], device=target_actions.device, dtype=target_actions.dtype
            )
        elif tuple(sample_weights.shape) != (target_actions.shape[0],):
            raise ValueError("sample_weights must have shape [batch]")
        else:
            sample_weights = sample_weights.to(
                device=target_actions.device, dtype=target_actions.dtype
            )
        if bool((sample_weights < 0).any()) or float(sample_weights.sum()) <= 0.0:
            raise ValueError("sample_weights must be non-negative with a positive sum")

        loss_dtype = torch.float32 if self.config.flow_loss_fp32 else target_actions.dtype
        loss_sample_weights = sample_weights.to(dtype=loss_dtype)

        def weighted_mean(per_sample: Tensor) -> Tensor:
            # Task weights are normalized globally to E_dataset[w]=1.  Do not
            # renormalize inside a usually single-task shard batch, otherwise
            # inverse-frequency weighting cancels out exactly.
            return (per_sample.to(dtype=loss_dtype) * loss_sample_weights).mean()

        losses: dict[str, Tensor] = {}
        mask = valid_mask.to(target_actions.dtype)[:, :, None]
        sample_denominator = (
            mask.sum(dim=(1, 2)) * self.config.action_dim
        ).clamp_min(1.0)
        gripper_condition_actions: Tensor | None = None
        if stage in {"flow", "joint"}:
            noise = torch.randn_like(target_actions) if noise is None else noise
            if time is None:
                time = torch.distributions.Beta(1.5, 1.0).sample(
                    (target_actions.shape[0],)
                ).to(target_actions.device).clamp_(0.001, 0.999)
            time_view = time[:, None, None].to(target_actions.dtype)
            noisy_actions = (1.0 - time_view) * noise + time_view * target_actions
            prediction = self.flow_expert(context, state_token, noisy_actions, time)
            if not isinstance(prediction, Tensor):  # pragma: no cover - expert contract
                prediction = prediction[0]
            target_velocity = target_actions - noise
            flow_weights = mask.expand_as(target_actions).clone()
            if self.config.flow_prefix_weight > 1.0:
                flow_weights[:, : self.config.replan_action_steps] *= (
                    self.config.flow_prefix_weight
                )
            if self.config.flow_gripper_transition_weight > 1.0:
                if previous_actions is None or tuple(previous_actions.shape) != (
                    target_actions.shape[0],
                    self.config.action_dim,
                ):
                    raise ValueError(
                        "transition-weighted Flow requires previous_actions [batch, action_dim]"
                    )
                previous_gripper = torch.cat(
                    (previous_actions[:, None, -1], target_actions[:, :-1, -1]),
                    dim=1,
                )
                transitions = (
                    (target_actions[:, :, -1] > 0.0)
                    != (previous_gripper > 0.0)
                ) & valid_mask
                transition_scale = 1.0 + transitions.to(target_actions.dtype) * (
                    self.config.flow_gripper_transition_weight - 1.0
                )
                flow_weights[:, :, -1] *= transition_scale
            flow_weights_for_loss = flow_weights.to(dtype=loss_dtype)
            flow_denominator = flow_weights_for_loss.sum(dim=(1, 2)).clamp_min(1.0)
            flow_per_sample = (
                (
                    prediction.to(dtype=loss_dtype)
                    - target_velocity.to(dtype=loss_dtype)
                ).square()
                * flow_weights_for_loss
            ).sum(dim=(1, 2)) / flow_denominator
            losses["flow"] = weighted_mean(flow_per_sample)
            endpoint_prediction = noisy_actions + (1.0 - time_view) * prediction
            gripper_condition_actions = endpoint_prediction.detach()
            if self.config.flow_endpoint_loss_weight > 0.0:
                prefix = self.config.replan_action_steps
                endpoint_mask = mask[:, :prefix].expand(
                    -1, -1, self.config.action_dim
                )[:, :, :-1]
                endpoint_denominator = endpoint_mask.sum(dim=(1, 2)).clamp_min(1.0)
                endpoint_per_sample = (
                    (
                        endpoint_prediction[:, :prefix, :-1]
                        - target_actions[:, :prefix, :-1]
                    ).abs()
                    * endpoint_mask
                ).sum(dim=(1, 2)) / endpoint_denominator
                losses["flow_endpoint"] = (
                    self.config.flow_endpoint_loss_weight
                    * weighted_mean(endpoint_per_sample)
                )

        if stage == "gripper":
            if parent_actions is None or tuple(parent_actions.shape) != tuple(target_actions.shape):
                raise ValueError("gripper stage requires inference-matched parent_actions")
            gripper_condition_actions = parent_actions.detach()

        context_summary = context.mean(dim=1)
        if stage in {"direct", "instruction", "joint"}:
            if parent_actions is None or tuple(parent_actions.shape) != tuple(target_actions.shape):
                raise ValueError("residual stages require inference-matched parent_actions")
            base = parent_actions.detach()
            direct = self.direct_residual(base, context_summary, state_token[:, 0], language)
            stage_one = base + direct
            gripper_condition_actions = stage_one.detach()
            direct_target = target_actions - base
            residual_mask = mask.expand_as(target_actions).clone()
            residual_mask[:, :, -1] = 0.0
            residual_denominator = residual_mask.sum(dim=(1, 2)).clamp_min(1.0)
            direct_per_sample = (
                (direct - direct_target).abs() * residual_mask
            ).sum(dim=(1, 2)) / residual_denominator
            losses["direct"] = self.config.direct_loss_weight * weighted_mean(
                direct_per_sample
            )
            if stage in {"instruction", "joint"}:
                instruction = self.instruction_residual(
                    stage_one.detach(), context_summary, state_token[:, 0], language
                )
                instruction_target = target_actions - stage_one.detach()
                gripper_condition_actions = (stage_one + instruction).detach()
                instruction_per_sample = (
                    (instruction - instruction_target).abs() * residual_mask
                ).sum(dim=(1, 2)) / residual_denominator
                losses["instruction"] = self.config.instruction_loss_weight * weighted_mean(
                    instruction_per_sample
                )
                e0 = (base[:, :, :-1] - target_actions[:, :, :-1]).abs().mean(dim=-1)
                e1 = (stage_one[:, :, :-1] - target_actions[:, :, :-1]).abs().mean(dim=-1)
                e2 = (
                    stage_one[:, :, :-1]
                    + instruction[:, :, :-1]
                    - target_actions[:, :, :-1]
                ).abs().mean(dim=-1)
                valid = valid_mask.to(e0.dtype)
                monotonic_per_sample = (
                    (torch.relu(e1 - e0) + torch.relu(e2 - e1)) * valid
                ).sum(dim=1) / valid.sum(dim=1).clamp_min(1.0)
                losses["monotonic"] = self.config.monotonic_loss_weight * weighted_mean(
                    monotonic_per_sample
                )

        if self.config.unified_flow_gripper:
            return losses
        if self.config.gripper_action_conditioning:
            if (
                self.action_conditioned_gripper is None
                or previous_actions is None
                or tuple(previous_actions.shape)
                != (target_actions.shape[0], self.config.action_dim)
                or gripper_condition_actions is None
            ):
                raise ValueError(
                    "action-conditioned gripper requires previous and generated actions"
                )
            gripper_logits, gripper_transition_logits = (
                self.action_conditioned_gripper(
                    gripper_condition_actions,
                    context_summary,
                    state_token[:, 0],
                    language,
                    previous_actions,
                )
            )
        else:
            if self.gripper_event_head is None or self.gripper_transition_head is None:
                raise RuntimeError("legacy gripper heads are unavailable")
            gripper_logits = self.gripper_event_head(
                torch.cat((context_summary, state_token[:, 0]), dim=-1)
            )
            gripper_transition_logits = self.gripper_transition_head(
                torch.cat((context_summary, state_token[:, 0]), dim=-1)
            )
        gripper_target = target_actions[:, :, -1].gt(0.0)
        if self.config.gripper_action_conditioning:
            if previous_actions is None:  # pragma: no cover - checked above
                raise RuntimeError("previous actions disappeared")
            previous_closed = previous_actions[:, -1].gt(0.0)
            transition_target_bool = torch.cat(
                (
                    gripper_target[:, :1].ne(previous_closed[:, None]),
                    gripper_target[:, 1:].ne(gripper_target[:, :-1]),
                ),
                dim=1,
            )
            transition_target = transition_target_bool.to(
                gripper_transition_logits.dtype
            )
            event_weight = 1.0 + (
                self.config.gripper_event_change_weight - 1.0
            ) * transition_target
        else:
            transition_target_bool = gripper_target[:, 1:].ne(gripper_target[:, :-1])
            transition_target = transition_target_bool.to(
                gripper_transition_logits.dtype
            )
            event_weight = torch.ones_like(
                gripper_target, dtype=gripper_transition_logits.dtype
            )
        if self.config.gripper_control_mode == "event3":
            if not self.config.gripper_action_conditioning:
                raise RuntimeError("event3 gripper control requires action conditioning")
            if gripper_logits.shape != (*gripper_target.shape, 3):
                raise RuntimeError("event3 gripper logits must have shape [batch, horizon, 3]")
            # 0=HOLD, 1=CLOSE, 2=OPEN. Step zero compares against the
            # previous executed command; later steps compare within the chunk.
            event_target = torch.zeros_like(gripper_target, dtype=torch.long)
            event_target = torch.where(
                transition_target_bool & gripper_target,
                torch.ones_like(event_target),
                event_target,
            )
            event_target = torch.where(
                transition_target_bool & ~gripper_target,
                torch.full_like(event_target, 2),
                event_target,
            )
            class_weights = torch.as_tensor(
                self.config.gripper_event_class_weights,
                device=gripper_logits.device,
                dtype=gripper_logits.dtype,
            )
            event_loss = torch.nn.functional.cross_entropy(
                gripper_logits.reshape(-1, 3),
                event_target.reshape(-1),
                weight=class_weights,
                reduction="none",
            ).reshape_as(event_target)
            if self.config.gripper_event_focal_gamma > 0.0:
                event_probability = torch.softmax(gripper_logits, dim=-1).gather(
                    -1, event_target[..., None]
                ).squeeze(-1)
                event_loss = event_loss * (1.0 - event_probability).pow(
                    self.config.gripper_event_focal_gamma
                )
            event_valid = valid_mask.to(event_loss.dtype)
            event_per_sample = (event_loss * event_valid).sum(dim=1) / event_valid.sum(
                dim=1
            ).clamp_min(1.0)
            losses["gripper_event"] = self.config.gripper_event_loss_weight * weighted_mean(
                event_per_sample
            )
            state_aux_loss = torch.nn.functional.binary_cross_entropy_with_logits(
                gripper_transition_logits,
                gripper_target.to(gripper_transition_logits.dtype),
                reduction="none",
            )
            state_aux_per_sample = (
                state_aux_loss * event_valid
            ).sum(dim=1) / event_valid.sum(dim=1).clamp_min(1.0)
            losses["gripper_state_aux"] = (
                self.config.gripper_state_aux_loss_weight
                * weighted_mean(state_aux_per_sample)
            )
            return losses

        gripper_loss = torch.nn.functional.binary_cross_entropy_with_logits(
            gripper_logits, gripper_target.to(gripper_logits.dtype), reduction="none"
        ) * event_weight
        gripper_valid = valid_mask.to(gripper_loss.dtype)
        gripper_per_sample = (gripper_loss * gripper_valid).sum(dim=1) / gripper_valid.sum(
            dim=1
        ).clamp_min(1.0)
        losses["gripper"] = self.config.gripper_loss_weight * weighted_mean(
            gripper_per_sample
        )
        transition_valid = (
            valid_mask
            if self.config.gripper_action_conditioning
            else valid_mask[:, 1:] & valid_mask[:, :-1]
        )
        transition_loss = torch.nn.functional.binary_cross_entropy_with_logits(
            gripper_transition_logits,
            transition_target,
            reduction="none",
            pos_weight=torch.as_tensor(
                self.config.gripper_transition_positive_weight,
                device=gripper_transition_logits.device,
                dtype=gripper_transition_logits.dtype,
            ),
        )
        transition_valid_float = transition_valid.to(transition_loss.dtype)
        transition_per_sample = (
            transition_loss * transition_valid_float
        ).sum(dim=1) / transition_valid_float.sum(dim=1).clamp_min(1.0)
        losses["gripper_transition"] = (
            self.config.gripper_transition_loss_weight
            * weighted_mean(transition_per_sample)
        )
        return losses

    def predict_flow_velocity(
        self,
        visual: Tensor,
        semantic: Tensor,
        state: Tensor,
        noisy_actions: Tensor,
        time: Tensor,
        **context_kwargs: Tensor | bool | None,
    ) -> Tensor:
        """Expose the shared Flow field for paired observation consistency.

        This method does not create a second action definition: it calls the
        exact context encoder and Flow expert used by ``flow_loss_components``.
        It is intentionally separate so formal training can apply consistency
        only to a small paired subset without retaining a full-batch graph.
        """
        expected = (
            noisy_actions.ndim == 3
            and tuple(noisy_actions.shape[1:])
            == (self.config.action_horizon, self.config.action_dim)
            and tuple(time.shape) == (noisy_actions.shape[0],)
        )
        if not expected:
            raise ValueError("noisy_actions/time violate the Flow field contract")
        context, state_token, _language, _weights = self._encode_context(
            visual,
            semantic,
            state,
            **context_kwargs,
        )
        prediction = self.flow_expert(context, state_token, noisy_actions, time)
        if not isinstance(prediction, Tensor):  # pragma: no cover
            prediction = prediction[0]
        return prediction

    @torch.no_grad()
    def encode_outcome_context(
        self,
        visual: Tensor,
        semantic: Tensor,
        state: Tensor,
        *,
        auxiliary_visual: Tensor | None = None,
        history_visual: Tensor | None = None,
        history_auxiliary_visual: Tensor | None = None,
        history_semantic: Tensor | None = None,
        history_states: Tensor | None = None,
        previous_action: Tensor | None = None,
        history_previous_actions: Tensor | None = None,
        layer14_only: bool = False,
    ) -> Tensor:
        """Expose the exact task-agnostic context consumed by the Flow expert."""

        context, _state_token, _language, _weights = self._encode_context(
            visual,
            semantic,
            state,
            auxiliary_visual=auxiliary_visual,
            history_visual=history_visual,
            history_auxiliary_visual=history_auxiliary_visual,
            history_semantic=history_semantic,
            history_states=history_states,
            previous_action=previous_action,
            history_previous_actions=history_previous_actions,
            layer14_only=layer14_only,
        )
        return context

    def _integrate_flow_candidates(
        self,
        context: Tensor,
        state_token: Tensor,
        *,
        samples: int,
        steps: int,
        noise: Tensor | None,
    ) -> Tensor:
        if samples <= 0 or steps <= 0:
            raise ValueError("flow_samples and flow_steps must be positive")
        batch = context.shape[0]
        expected_noise = (
            batch,
            samples,
            self.config.action_horizon,
            self.config.action_dim,
        )
        if noise is None:
            actions = torch.randn(
                expected_noise,
                device=context.device,
                dtype=next(self.parameters()).dtype,
            )
        else:
            if tuple(noise.shape) != expected_noise:
                raise ValueError(f"noise must have shape {expected_noise}")
            actions = noise.to(device=context.device, dtype=next(self.parameters()).dtype)
        flat_actions = actions.flatten(0, 1)
        repeated_context = context[:, None].expand(-1, samples, -1, -1).flatten(0, 1)
        repeated_state = state_token[:, None].expand(-1, samples, -1, -1).flatten(0, 1)
        for index in range(steps):
            time = torch.full(
                (batch * samples,),
                index / steps,
                device=context.device,
                dtype=flat_actions.dtype,
            )
            velocity = self.flow_expert(
                repeated_context, repeated_state, flat_actions, time
            )
            if not isinstance(velocity, Tensor):  # pragma: no cover
                velocity = velocity[0]
            flat_actions = flat_actions + velocity / steps
        return flat_actions.reshape(expected_noise)

    @torch.no_grad()
    def sample_flow_candidates(
        self,
        visual: Tensor,
        semantic: Tensor,
        state: Tensor,
        *,
        auxiliary_visual: Tensor | None = None,
        history_visual: Tensor | None = None,
        history_auxiliary_visual: Tensor | None = None,
        history_semantic: Tensor | None = None,
        history_states: Tensor | None = None,
        previous_action: Tensor | None = None,
        history_previous_actions: Tensor | None = None,
        layer14_only: bool = False,
        flow_samples: int | None = None,
        flow_steps: int | None = None,
        noise: Tensor | None = None,
    ) -> Tensor:
        """Return whole Flow trajectories before median aggregation.

        The output is ``[batch, samples, horizon, action_dim]``.  Candidates
        are exposed for outcome-ranking diagnostics and planners; no task id,
        reward, success flag, or evaluation-state signal is accepted.
        """

        context, state_token, _language, _weights = self._encode_context(
            visual,
            semantic,
            state,
            auxiliary_visual=auxiliary_visual,
            history_visual=history_visual,
            history_auxiliary_visual=history_auxiliary_visual,
            history_semantic=history_semantic,
            history_states=history_states,
            previous_action=previous_action,
            history_previous_actions=history_previous_actions,
            layer14_only=layer14_only,
        )
        samples = self.config.flow_samples if flow_samples is None else flow_samples
        steps = self.config.flow_steps if flow_steps is None else flow_steps
        return self._integrate_flow_candidates(
            context, state_token, samples=samples, steps=steps, noise=noise
        )

    @torch.no_grad()
    def sample_action_candidates(
        self,
        visual: Tensor,
        semantic: Tensor,
        state: Tensor,
        *,
        auxiliary_visual: Tensor | None = None,
        history_visual: Tensor | None = None,
        history_auxiliary_visual: Tensor | None = None,
        history_semantic: Tensor | None = None,
        history_states: Tensor | None = None,
        previous_action: Tensor | None = None,
        history_previous_actions: Tensor | None = None,
        layer14_only: bool = False,
        flow_samples: int | None = None,
        flow_steps: int | None = None,
        apply_direct: bool = True,
        apply_instruction: bool = True,
        noise: Tensor | None = None,
    ) -> Tensor:
        """Return complete policy candidates plus the exact median baseline.

        Every raw Flow trajectory and the coordinate median pass through the
        same Direct and Instruction residual stack used by ``sample_actions``.
        The final candidate is therefore the actual deployed policy baseline,
        not an intermediate Flow tensor.  Candidate reranking is currently
        defined only for unified Flow arm/gripper policies so no stateful
        post-hoc gripper controller can be skipped accidentally.
        """

        if not self.config.unified_flow_gripper:
            raise ValueError(
                "complete action candidates require unified Flow gripper control"
            )
        context, state_token, language, _weights = self._encode_context(
            visual,
            semantic,
            state,
            auxiliary_visual=auxiliary_visual,
            history_visual=history_visual,
            history_auxiliary_visual=history_auxiliary_visual,
            history_semantic=history_semantic,
            history_states=history_states,
            previous_action=previous_action,
            history_previous_actions=history_previous_actions,
            layer14_only=layer14_only,
        )
        samples = self.config.flow_samples if flow_samples is None else flow_samples
        steps = self.config.flow_steps if flow_steps is None else flow_steps
        raw = self._integrate_flow_candidates(
            context, state_token, samples=samples, steps=steps, noise=noise
        )
        candidates = torch.cat((raw, raw.median(dim=1).values[:, None]), dim=1)
        batch, candidate_count, horizon, action_dim = candidates.shape
        flat = candidates.flatten(0, 1)
        context_summary = context.mean(dim=1)
        repeated_context = context_summary[:, None].expand(
            -1, candidate_count, -1
        ).flatten(0, 1)
        repeated_state = state_token[:, 0][:, None].expand(
            -1, candidate_count, -1
        ).flatten(0, 1)
        repeated_language = language[:, None].expand(
            -1, candidate_count, -1
        ).flatten(0, 1)
        if apply_direct:
            flat = flat + self.direct_residual(
                flat, repeated_context, repeated_state, repeated_language
            )
        if apply_instruction:
            flat = flat + self.instruction_residual(
                flat, repeated_context, repeated_state, repeated_language
            )
        return flat.reshape(batch, candidate_count, horizon, action_dim)

    @torch.no_grad()
    def sample_actions(
        self,
        visual: Tensor,
        semantic: Tensor,
        state: Tensor,
        *,
        auxiliary_visual: Tensor | None = None,
        language_tokens: Tensor | None = None,
        language_mask: Tensor | None = None,
        history_visual: Tensor | None = None,
        history_auxiliary_visual: Tensor | None = None,
        history_semantic: Tensor | None = None,
        history_states: Tensor | None = None,
        previous_action: Tensor | None = None,
        history_previous_actions: Tensor | None = None,
        layer14_only: bool = False,
        flow_samples: int | None = None,
        flow_steps: int | None = None,
        apply_direct: bool = True,
        apply_instruction: bool = True,
        apply_gripper_event: bool = True,
        previous_gripper_closed: Tensor | None = None,
        noise: Tensor | None = None,
        apply_recovery: bool = True,
        candidate_aggregation: str | None = None,
    ) -> Tensor:
        context, state_token, language, _weights = self._encode_context(
            visual,
            semantic,
            state,
            auxiliary_visual=auxiliary_visual,
            language_tokens=language_tokens,
            language_mask=language_mask,
            history_visual=history_visual,
            history_auxiliary_visual=history_auxiliary_visual,
            history_semantic=history_semantic,
            history_states=history_states,
            previous_action=previous_action,
            history_previous_actions=history_previous_actions,
            layer14_only=layer14_only,
        )
        samples = self.config.flow_samples if flow_samples is None else flow_samples
        steps = self.config.flow_steps if flow_steps is None else flow_steps
        candidates = self._integrate_flow_candidates(
            context, state_token, samples=samples, steps=steps, noise=noise
        )
        aggregation = self.config.candidate_aggregation if candidate_aggregation is None else candidate_aggregation
        base = aggregate_action_candidates(
            candidates,
            mode=aggregation,
            executed_prefix=self.config.replan_action_steps,
            prefix_weight=self.config.flow_prefix_weight,
        )
        batch = visual.shape[0]
        context_summary = context.mean(dim=1)
        if apply_direct:
            base = base + self.direct_residual(
                base, context_summary, state_token[:, 0], language
            )
        if apply_instruction:
            base = base + self.instruction_residual(
                base, context_summary, state_token[:, 0], language
            )
        if apply_gripper_event and not self.config.unified_flow_gripper:
            if self.config.gripper_action_conditioning:
                if self.action_conditioned_gripper is None or previous_action is None:
                    raise ValueError(
                        "action-conditioned gripper requires previous_action"
                    )
                event_logits, _auxiliary_logits = self.action_conditioned_gripper(
                    base,
                    context_summary,
                    state_token[:, 0],
                    language,
                    previous_action,
                )
                if self.config.gripper_control_mode == "event3":
                    event_probabilities = torch.softmax(event_logits, dim=-1)
                    probabilities = None
                else:
                    probabilities = event_logits.sigmoid()
                    event_probabilities = None
            else:
                if self.gripper_event_head is None:
                    raise RuntimeError("legacy gripper event head is unavailable")
                gripper_context = torch.cat(
                    (context_summary, state_token[:, 0]), dim=-1
                )
                probabilities = self.gripper_event_head(gripper_context).sigmoid()
                event_probabilities = None
            if previous_gripper_closed is None:
                if self.config.gripper_control_mode == "event3":
                    if previous_action is None:
                        raise ValueError("event3 gripper control requires previous_action")
                    closed_state = previous_action[:, -1].gt(0.0)
                else:
                    if probabilities is None:  # pragma: no cover - mode contract
                        raise RuntimeError("absolute gripper probabilities are unavailable")
                    closed_state = probabilities[:, 0].gt(0.5)
            else:
                if tuple(previous_gripper_closed.shape) != (batch,):
                    raise ValueError("previous_gripper_closed must have shape [batch]")
                closed_state = previous_gripper_closed.to(
                    device=base.device, dtype=torch.bool
                )
            closed_steps: list[Tensor] = []
            for step in range(self.config.action_horizon):
                if (
                    self.config.gripper_control_mode == "absolute"
                    and step == 0
                    and previous_gripper_closed is None
                ):
                    closed_steps.append(closed_state)
                    continue
                if self.config.gripper_control_mode == "event3":
                    if event_probabilities is None:  # pragma: no cover - mode contract
                        raise RuntimeError("event3 probabilities are unavailable")
                    confidence, event = event_probabilities[:, step].max(dim=-1)
                    confident = confidence.ge(
                        self.config.gripper_event_probability_threshold
                    )
                    request_close = confident & event.eq(1) & ~closed_state
                    request_open = confident & event.eq(2) & closed_state
                else:
                    if probabilities is None:  # pragma: no cover - mode contract
                        raise RuntimeError("absolute gripper probabilities are unavailable")
                    request_close = probabilities[:, step].gt(
                        self.config.gripper_close_threshold
                    )
                    request_open = probabilities[:, step].lt(
                        self.config.gripper_open_threshold
                    )
                next_state = torch.where(request_close, True, closed_state)
                next_state = torch.where(request_open, False, next_state)
                # The event head predicts the desired absolute gripper state.
                # Schmitt thresholds already provide temporal hysteresis.  The
                # transition head is an auxiliary training signal only: using
                # it as a hard switch gate caused a deterministic deadlock
                # because its sparse target describes A[t] -> A[t+1], not the
                # previous executed command -> A[t] boundary at replanning.
                closed_state = next_state
                closed_steps.append(closed_state)
            closed = torch.stack(closed_steps, dim=1)
            open_value = torch.as_tensor(
                self.config.gripper_open_value, device=base.device, dtype=base.dtype
            )
            close_value = torch.as_tensor(
                self.config.gripper_close_value, device=base.device, dtype=base.dtype
            )
            base[:, :, -1] = torch.where(closed, close_value, open_value)
        if apply_recovery and self.observation_feedback_recovery is not None:
            base, _residual, _gate = self.observation_feedback_recovery(
                context, state, base
            )
        return base

    def parameter_counts(self) -> dict[str, int]:
        total = sum(parameter.numel() for parameter in self.parameters())
        trainable = sum(
            parameter.numel() for parameter in self.parameters() if parameter.requires_grad
        )
        return {"robot_policy": total, "trainable": trainable}
