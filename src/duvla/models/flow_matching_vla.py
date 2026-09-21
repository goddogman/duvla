"""SmolVLA-style flow-matching action expert conditioned on frozen Qwen features.

The Qwen backbone is intentionally kept outside this module.  The training
boundary is a cached context tensor, robot state, and normalized action chunk;
this keeps the first full-data run feasible on an 8 GiB RTX 5060 while the
action expert follows SmolVLA's flow-matching objective.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

import torch
from torch import Tensor, nn

from .temporal_hierarchical import (
    PersistentProgressMemory,
    SubtaskProgressConditioner,
    TemporalHierarchicalConditioner,
    TemporalHierarchicalConfig,
)


@dataclass(frozen=True)
class FlowMatchingVLAConfig:
    """RTX 5060-compatible capacity matching the SmolVLA design choices."""

    state_dim: int = 8
    action_dim: int = 7
    vlm_feature_dim: int = 2048
    hidden_dim: int = 512
    action_horizon: int = 50
    max_context_tokens: int = 64
    expert_layers: int = 8
    expert_heads: int = 8
    dropout: float = 0.0
    num_flow_steps: int = 10
    self_attn_every_n_layers: int = 2
    time_sampling: str = "beta_1.5_1.0"
    feature_layer_norm: bool = False
    context_fusion_layers: int = 0
    context_fusion_zero_init: bool = False
    # Adapter-only experiments may keep the inherited context-fusion path
    # present for checkpoint compatibility while freezing its parameters.
    train_context_fusion_adapter: bool = True
    architecture_variant: str = "legacy"
    attention_pattern: str = "periodic_cross_self"
    ffn_type: str = "gelu"
    norm_type: str = "layer_norm"
    attention_bias: bool = True
    # Whether action tokens are decoded autoregressively inside a noisy Flow
    # chunk.  Disabling this mask lets all tokens in the *same generated
    # chunk* coordinate; it does not expose future observations or labels.
    causal_action_attention: bool = True
    state_projection_layers: int = 2
    state_conditioning: str = "expert"
    context_layout: str = "merged"
    camera_count: int = 1
    tokens_per_camera: int = 64
    qwen_hidden_layer: int = -1
    qwen_context_mode: str = "ordered"
    qwen_semantic_tokens: int = 0
    qwen_semantic_layer: int = -1
    qwen_grounding_tokens: int = 0
    min_period: float = 4e-3
    max_period: float = 4.0
    temporal_conditioning: bool = False
    history_length: int = 4
    phase_classes: int = 5
    phase_loss_weight: float = 0.1
    temporal_fusion: str = "append"
    temporal_gate_init: float = 0.02
    temporal_zero_init: bool = False
    train_temporal_gate: bool = True
    history_only_conditioning: bool = False
    selective_progress_gate: bool = False
    train_film_adapter: bool = True
    progress_conditioning: bool = False
    progress_classes: int = 4
    progress_loss_weight: float = 0.05
    train_progress_adapter: bool = True
    state_film_conditioning: bool = False
    visual_memory_conditioning: bool = False
    visual_memory_length: int = 2
    visual_memory_tokens_per_camera: int = 8
    gated_visual_memory: bool = False
    visual_memory_adapter_layers: int = 0
    visual_memory_summary_conditioning: bool = True
    deep_flow_conditioning: bool = False
    preserve_block_norm_conditioning: bool = False
    gripper_auxiliary: bool = False
    gripper_loss_weight: float = 0.05
    gripper_auxiliary_inference: bool = False
    gripper_auxiliary_inference_mix: float = 1.0
    gripper_only_adapter: bool = False
    direct_action_head: bool = False
    direct_action_loss_weight: float = 0.0
    direct_action_inference_mix: float = 0.0
    direct_action_attention_pool: bool = False
    train_temporal_adapters: bool = True
    task_conditioning: bool = False
    task_classes: int = 40
    task_conditioning_scale: float = 1.0
    task_router_pooling: str = "all_tokens"
    task_router_tail_tokens: int = 16
    train_task_adapter: bool = True
    train_direct_action_adapter: bool = True
    task_direct_residual_adapter: bool = False
    task_direct_residual_scale: float = 1.0
    # Optional per-task protection scales for transferred residual branches.
    # ``None`` preserves the historical single-scale behavior.  This is an
    # inference-time safety gate: it does not select a task or expose a
    # benchmark index to the policy.
    task_direct_residual_task_scales: tuple[float, ...] | list[float] | None = None
    train_task_direct_residual_adapter: bool = True
    # V52 turns the direct and task heads into additive corrections to the
    # frozen Flow prior.  The legacy V23/V51 path keeps the parallel blend.
    progressive_residual: bool = False
    progressive_action_conditioning: bool = False
    direct_residual_inference_scale: float = 1.0
    task_residual_inference_scale: float = 1.0
    monotonic_residual_loss_weight: float = 0.0
    monotonic_residual_margin: float = 0.0
    train_unified_head_adapter: bool = True
    unified_head_bank: bool = False
    unified_head_count: int = 6
    unified_action_expert_bank: bool = False
    # V53 keeps the complete V51 policy as the base and adds one shared
    # residual adapter for every task.  It is deliberately not indexed by
    # task id; task difficulty may only affect training weights.
    global_action_residual_adapter: bool = False
    global_action_residual_scale: float = 1.0
    # V54 replaces V53's always-on correction with one task-agnostic,
    # observation-dependent gated residual.  It operates after the complete
    # frozen V51 action path and never consumes a benchmark task index.
    state_adaptive_residual_adapter: bool = False
    state_adaptive_residual_scale: float = 1.0
    state_adaptive_gate_init: float = 0.02
    # V60 adapts frozen Qwen feature geometry with a zero-initialized low-rank
    # residual before the shared visual projection.
    visual_feature_adapter: bool = False
    visual_feature_adapter_bottleneck: int = 64
    train_visual_feature_adapter: bool = True
    state_feature_adapter: bool = False
    state_feature_adapter_bottleneck: int = 64
    train_state_feature_adapter: bool = True
    # V73 keeps the complete V51 action path frozen and reads the same ordered
    # Qwen cache through a separate, explicitly order-aware arm-only sidecar.
    # It consumes continuous visual/language features, never a benchmark id.
    ordered_relative_residual_adapter: bool = False
    ordered_relative_residual_scale: float = 0.25
    ordered_relative_semantic_tokens: int = 16
    # V74 keeps the complete V51 path frozen and consumes a separate exact
    # image-patch grid (optionally projected by train-only PCA) in a
    # task-agnostic geometry sidecar. It only predicts 6D arm residuals.
    spatial_grid_residual_adapter: bool = False
    spatial_grid_feature_dim: int = 64
    spatial_grid_residual_scale: float = 0.25
    spatial_grid_semantic_tokens: int = 16
    # V93 compresses a longer visual history into progress slots before the
    # existing zero-initialized gated Flow memory adapters.
    persistent_progress_memory: bool = False
    persistent_memory_slots: int = 4
    persistent_memory_heads: int = 4
    # V98-LaST: compact latent chain-of-thought tokens condition the frozen
    # V51/Flow action path.  The latent path is auxiliary and zero-gated at
    # initialization, so an untrained transfer is exactly the parent policy.
    latent_reasoning: bool = False
    latent_reasoning_tokens: int = 2
    latent_reasoning_loss_weight: float = 0.05
    latent_end_loss_weight: float = 0.01
    latent_gate_init: float = 0.01
    # V99-LaST residual bridge: the latent plan is converted directly into a
    # bounded correction of the frozen V51 action chunk.  The output and gate
    # are zero/low initialized, preserving V51 before warm-up training.
    latent_residual_adapter: bool = False
    latent_residual_scale: float = 0.25
    latent_residual_gate_init: float = 0.05
    latent_residual_loss_weight: float = 0.25
    latent_adaptive_horizon: bool = True
    train_latent_residual_adapter: bool = True

    @classmethod
    def smol_aligned(cls) -> FlowMatchingVLAConfig:
        """Return the RTX 5060 target that mirrors SmolVLA's key choices."""

        return cls(
            hidden_dim=768,
            max_context_tokens=128,
            expert_layers=12,
            expert_heads=12,
            feature_layer_norm=True,
            context_fusion_layers=2,
            architecture_variant="smol_aligned",
            attention_pattern="alternating",
            ffn_type="swiglu",
            norm_type="rms_norm",
            attention_bias=False,
            state_projection_layers=1,
            state_conditioning="prefix",
            context_layout="per_camera",
            camera_count=2,
            tokens_per_camera=64,
            qwen_hidden_layer=14,
        )

    @classmethod
    def v8_temporal(cls) -> FlowMatchingVLAConfig:
        """Return the V8 conditioner-enabled preset on top of V7's capacity."""

        return replace(
            cls.smol_aligned(),
            temporal_conditioning=True,
            history_length=4,
            phase_classes=5,
            phase_loss_weight=0.1,
        )

    @classmethod
    def v8_1_residual(cls) -> FlowMatchingVLAConfig:
        """Return the conservative V8.1 residual-conditioning variant.

        Temporal information is fused into the existing state token through a
        small gated adapter.  This keeps the V7 visual context length and
        positional table unchanged, so the new path cannot immediately
        disrupt the proven V7 cross-attention route.
        """

        return replace(
            cls.smol_aligned(),
            temporal_conditioning=True,
            history_length=4,
            phase_classes=5,
            phase_loss_weight=0.02,
            temporal_fusion="state_residual",
            temporal_gate_init=0.02,
        )

    @classmethod
    def v9_chunk8_film(cls) -> FlowMatchingVLAConfig:
        """Return the V9 low-risk V7-compatible FiLM/chunk-8 preset."""

        return replace(
            cls.smol_aligned(),
            action_horizon=8,
            state_film_conditioning=True,
            temporal_conditioning=False,
            temporal_fusion="append",
            phase_loss_weight=0.0,
        )

    @classmethod
    def v9_1_chunk16_history_film(cls) -> FlowMatchingVLAConfig:
        """Return the long-horizon residual-history V9.1 preset.

        V9-A improved short spatial/goal tasks but collapsed on LIBERO-10's
        multi-object sequences.  V9.1 keeps V7's visual context untouched,
        extends the action chunk to 16 steps, and injects a four-step
        state/action history through a zero-initialized residual before the
        existing zero-initialized FiLM adapter.
        """

        return replace(
            cls.smol_aligned(),
            action_horizon=16,
            temporal_conditioning=True,
            history_length=4,
            phase_classes=5,
            phase_loss_weight=0.0,
            temporal_fusion="state_residual",
            temporal_gate_init=0.02,
            temporal_zero_init=True,
            state_film_conditioning=True,
        )

    @classmethod
    def v9_2_selective_history_film(cls) -> FlowMatchingVLAConfig:
        """Return V9-A with a gated, history-only progress residual.

        The short horizon and FiLM path from V9-A are retained.  History is
        not appended to visual context and the unsupervised phase token is
        ignored; a context/state/history gate decides when the residual is
        useful for long-horizon multi-object tasks.
        """

        return replace(
            cls.v9_chunk8_film(),
            temporal_conditioning=True,
            history_length=4,
            phase_classes=5,
            phase_loss_weight=0.0,
            temporal_fusion="state_residual",
            temporal_gate_init=0.02,
            temporal_zero_init=True,
            history_only_conditioning=True,
            selective_progress_gate=True,
        )

    @classmethod
    def v9_3_preserve_v9a_history(cls) -> FlowMatchingVLAConfig:
        """Return V9.2 history correction while freezing the V9-A FiLM path."""

        return replace(
            cls.v9_2_selective_history_film(),
            temporal_gate_init=0.005,
            train_film_adapter=False,
        )

    @classmethod
    def v9_4_subtask_progress(cls) -> FlowMatchingVLAConfig:
        """Return V9.3 with supervised subtask progress conditioning."""

        return replace(
            cls.v9_3_preserve_v9a_history(),
            progress_conditioning=True,
            progress_classes=4,
            progress_loss_weight=0.05,
        )

    @classmethod
    def v93_persistent_progress_memory(cls) -> FlowMatchingVLAConfig:
        """V9 history upgraded to a longer visual progress-memory sidecar."""

        return replace(
            cls.v51_v23_tail_task_direct_residual(),
            # Keep V51's four-step state/action contract; only the visual
            # memory span is extended so the parent action path transfers
            # exactly and the new hypothesis is isolated to visual progress.
            history_length=4,
            temporal_conditioning=True,
            phase_classes=5,
            phase_loss_weight=0.0,
            temporal_fusion="state_residual",
            temporal_gate_init=0.005,
            temporal_zero_init=True,
            history_only_conditioning=True,
            selective_progress_gate=True,
            visual_memory_conditioning=True,
            visual_memory_length=16,
            visual_memory_tokens_per_camera=8,
            gated_visual_memory=True,
            visual_memory_adapter_layers=3,
            visual_memory_summary_conditioning=False,
            deep_flow_conditioning=False,
            preserve_block_norm_conditioning=False,
            persistent_progress_memory=True,
            persistent_memory_slots=4,
            persistent_memory_heads=4,
            train_temporal_adapters=False,
            train_task_adapter=False,
            train_progress_adapter=False,
            train_direct_action_adapter=False,
            train_film_adapter=False,
            train_task_direct_residual_adapter=False,
            train_unified_head_adapter=False,
        )

    @classmethod
    def v10_visual_memory(cls) -> FlowMatchingVLAConfig:
        """Return the zero-init V10 visual-memory/deep-flow preset."""

        return replace(
            cls.v9_3_preserve_v9a_history(),
            visual_memory_conditioning=True,
            visual_memory_length=2,
            visual_memory_tokens_per_camera=8,
            deep_flow_conditioning=True,
            gripper_auxiliary=True,
        )

    @classmethod
    def v10_1_gated_memory(cls) -> FlowMatchingVLAConfig:
        """Return the V9.3-equivalent gated visual-memory preset.

        Unlike V10, compressed history tokens are read through dedicated
        zero-gated residual cross-attention blocks.  The original visual
        context and trained block norms therefore remain unchanged at step 0.
        """

        return replace(
            cls.v9_3_preserve_v9a_history(),
            visual_memory_conditioning=True,
            visual_memory_length=2,
            visual_memory_tokens_per_camera=8,
            gated_visual_memory=True,
            visual_memory_adapter_layers=2,
            deep_flow_conditioning=True,
            preserve_block_norm_conditioning=True,
            gripper_auxiliary=False,
        )

    @classmethod
    def v10_2_memory_only(cls) -> FlowMatchingVLAConfig:
        """Return V10.1 with only the gated visual-memory path enabled."""

        return replace(
            cls.v10_1_gated_memory(),
            visual_memory_summary_conditioning=False,
            deep_flow_conditioning=False,
            preserve_block_norm_conditioning=False,
        )

    @classmethod
    def v11_oft_l1(cls) -> FlowMatchingVLAConfig:
        """Return an OFT-inspired continuous action-head ablation.

        The proven V9.3 conditioning path is retained while a direct action
        chunk head is trained with an L1 objective.  A conservative inference
        blend keeps the flow expert in the loop instead of replacing it.
        """

        return replace(
            cls.v9_3_preserve_v9a_history(),
            direct_action_head=True,
            direct_action_loss_weight=0.5,
            direct_action_inference_mix=0.5,
        )

    @classmethod
    def v11_1_oft_l1_frozen(cls) -> FlowMatchingVLAConfig:
        """Train only the direct L1 head while freezing V9.3 conditioning."""

        return replace(
            cls.v11_oft_l1(),
            direct_action_loss_weight=0.25,
            direct_action_inference_mix=0.25,
            train_temporal_adapters=False,
        )

    @classmethod
    def v12_long_chunk_l1(cls) -> FlowMatchingVLAConfig:
        """Use the original 50-step expert with conservative temporal/L1 heads."""

        return replace(
            cls.smol_aligned(),
            action_horizon=50,
            temporal_conditioning=True,
            history_length=4,
            phase_classes=5,
            phase_loss_weight=0.0,
            temporal_fusion="state_residual",
            temporal_gate_init=0.005,
            temporal_zero_init=True,
            history_only_conditioning=True,
            selective_progress_gate=True,
            state_film_conditioning=True,
            train_film_adapter=False,
            direct_action_head=True,
            direct_action_loss_weight=0.25,
            direct_action_inference_mix=0.25,
        )

    @classmethod
    def v13_attn_progress_l1(cls) -> FlowMatchingVLAConfig:
        """Use an 8-step L1 chunk with learned visual pooling and progress context.

        The 8-step horizon matches the LIBERO replan contract.  A learned
        state-conditioned query pools Qwen tokens instead of averaging them,
        while the existing temporal conditioner predicts coarse subtask
        progress for multi-object episodes.  The base Flow expert remains
        frozen so the change is a small, auditable adapter experiment.
        """

        return replace(
            cls.v11_1_oft_l1_frozen(),
            action_horizon=8,
            progress_conditioning=True,
            progress_classes=4,
            progress_loss_weight=0.05,
            direct_action_attention_pool=True,
        )

    @classmethod
    def v14_attn_l1(cls) -> FlowMatchingVLAConfig:
        """Use learned visual pooling without the progress-conditioner path."""

        return replace(
            cls.v11_1_oft_l1_frozen(),
            action_horizon=8,
            direct_action_attention_pool=True,
        )

    @classmethod
    def v18_gripper_aux_l1(cls) -> FlowMatchingVLAConfig:
        """Add SmolVLA-style binary gripper supervision to V14's action head.

        The auxiliary head predicts the open/close state for every action step;
        the continuous flow and direct L1 heads remain unchanged.
        """

        return replace(
            cls.v14_attn_l1(),
            gripper_auxiliary=True,
            gripper_loss_weight=0.10,
            gripper_auxiliary_inference=True,
        )

    @classmethod
    def v20_direct_strong(cls) -> FlowMatchingVLAConfig:
        """Use a pure supervised L1 action chunk for hard long-horizon tasks."""

        return replace(
            cls.v14_attn_l1(),
            direct_action_loss_weight=1.0,
            direct_action_inference_mix=1.0,
        )

    @classmethod
    def v22_task_conditioned_residual(cls) -> FlowMatchingVLAConfig:
        """Use one policy with a language-inferred task residual.

        The base V11.1 action expert and direct L1 head are retained.  A
        zero-initialized residual receives a task embedding selected from
        Qwen-feature prototypes, allowing multi-stage tasks to specialize
        without an external per-task checkpoint or evaluator-side routing.
        """

        return replace(
            cls.v14_attn_l1(),
            task_conditioning=True,
            task_classes=40,
            task_conditioning_scale=1.0,
            gripper_auxiliary=True,
            gripper_loss_weight=0.10,
            gripper_auxiliary_inference=True,
            direct_action_loss_weight=0.50,
            direct_action_inference_mix=0.50,
        )

    @classmethod
    def v22_1_task_residual_conservative(cls) -> FlowMatchingVLAConfig:
        """Keep V11.1's proven action blend while fitting only task residuals."""

        return replace(
            cls.v11_1_oft_l1_frozen(),
            task_conditioning=True,
            task_classes=40,
            task_conditioning_scale=1.0,
            train_direct_action_adapter=False,
        )

    @classmethod
    def v23_unified_head_bank(cls) -> FlowMatchingVLAConfig:
        """One checkpoint with a compact bank of audited direct action heads."""

        return replace(
            cls.v14_attn_l1(),
            task_conditioning=True,
            task_classes=40,
            unified_head_bank=True,
            unified_head_count=6,
            progress_conditioning=True,
            progress_classes=4,
            progress_loss_weight=0.05,
            direct_action_inference_mix=0.25,
            train_direct_action_adapter=False,
        )

    @classmethod
    def v30_unified_gripper_aux(cls) -> FlowMatchingVLAConfig:
        """V23 unified bank with an explicitly supervised binary gripper head.

        The Qwen context, language-conditioned direct heads, and temporal bank
        remain unchanged.  Only the per-action-step gripper auxiliary output is
        added, so this candidate isolates the grasp/release signal instead of
        changing the task map or evaluator protocol.
        """

        return replace(
            cls.v23_unified_head_bank(),
            gripper_auxiliary=True,
            gripper_loss_weight=0.10,
            gripper_auxiliary_inference=True,
            gripper_auxiliary_inference_mix=0.25,
            unified_action_expert_bank=False,
            gripper_only_adapter=True,
        )

    @classmethod
    def v32_oft_l1_hybrid(cls) -> FlowMatchingVLAConfig:
        """V23-compatible OFT-style continuous action-head ablation.

        Keep the single-checkpoint prototype-conditioned unified bank intact,
        increase the direct continuous L1 contribution, and train only the
        direct action adapters when ``--freeze-base-flow`` is used.
        """

        return replace(
            cls.v23_unified_head_bank(),
            direct_action_loss_weight=0.50,
            direct_action_inference_mix=0.50,
            train_direct_action_adapter=True,
        )

    @classmethod
    def v33_semantic_router_oft(cls) -> FlowMatchingVLAConfig:
        """Route the compact expert bank from explicit instruction tokens.

        The action path still consumes the complete two-camera hybrid context.
        Only task routing is restricted to final-layer instruction tokens, so
        changing object poses or camera pixels cannot dilute the language
        identity used to select an internal action head.
        """

        return replace(
            cls.v32_oft_l1_hybrid(),
            qwen_context_mode="hybrid_language",
            qwen_semantic_tokens=16,
            qwen_semantic_layer=-1,
            task_router_pooling="semantic_tokens",
        )

    @classmethod
    def v34_semantic_gripper(cls) -> FlowMatchingVLAConfig:
        """Add explicit binary gripper supervision to the semantic V33 path."""

        return replace(
            cls.v33_semantic_router_oft(),
            gripper_auxiliary=True,
            gripper_loss_weight=0.10,
            gripper_auxiliary_inference=True,
            gripper_auxiliary_inference_mix=0.25,
        )

    @classmethod
    def v35_semantic_gripper_conservative(cls) -> FlowMatchingVLAConfig:
        """Lower the direct-L1 blend while retaining semantic routing/gripper loss."""

        return replace(
            cls.v34_semantic_gripper(),
            direct_action_loss_weight=0.25,
            direct_action_inference_mix=0.25,
            gripper_auxiliary_inference_mix=0.10,
        )

    @classmethod
    def v36_semantic_gripper_direct(cls) -> FlowMatchingVLAConfig:
        """Increase direct-L1 inference for precise short action chunks."""

        return replace(
            cls.v34_semantic_gripper(),
            direct_action_loss_weight=0.75,
            direct_action_inference_mix=0.75,
            gripper_auxiliary_inference_mix=0.25,
        )

    @classmethod
    def v37_hybrid_router(cls) -> FlowMatchingVLAConfig:
        """Fuse semantic instruction and spatial visual summaries for routing."""

        return replace(
            cls.v32_oft_l1_hybrid(),
            qwen_context_mode="hybrid_language",
            qwen_semantic_tokens=16,
            qwen_semantic_layer=-1,
            task_router_pooling="semantic_visual_mix",
        )

    @classmethod
    def v39_ordered_bank_hybrid(cls) -> FlowMatchingVLAConfig:
        """Keep the V23 action bank while adapting its route to hybrid cache features.

        The V37/V38 experiments changed both the V23 action blend and the routing
        summary.  This preset restores V23's conservative direct-action blend and
        uses all hybrid-context tokens for the prototype contract, isolating the
        routing representation as the only intended change.
        """

        return replace(
            cls.v23_unified_head_bank(),
            qwen_context_mode="hybrid_language",
            qwen_semantic_tokens=16,
            qwen_semantic_layer=-1,
            task_router_pooling="all_tokens",
        )

    @classmethod
    def v41_ordered_oft_gripper(cls) -> FlowMatchingVLAConfig:
        """Use an OFT-style continuous action objective with gripper supervision.

        This keeps the audited six-head/task-conditioning path but makes the
        continuous L1 action stream the primary inference path and adds an
        explicit binary gripper auxiliary target for grasp/release transitions.
        """

        return replace(
            cls.v23_unified_head_bank(),
            direct_action_loss_weight=1.0,
            direct_action_inference_mix=1.0,
            train_direct_action_adapter=True,
            gripper_auxiliary=True,
            gripper_loss_weight=0.10,
            gripper_auxiliary_inference=True,
            gripper_auxiliary_inference_mix=0.25,
        )

    @classmethod
    def v42_frozen_v23_adapters(cls) -> FlowMatchingVLAConfig:
        """Preserve the V23 flow expert while adapting direct and gripper heads.

        V41's full-parameter update improved object pickup but regressed spatial
        and long-horizon suites.  This candidate keeps the audited V23 direct
        blend, adds the explicit gripper target, and is intended to be trained
        with ``--freeze-base-flow`` so the shared action expert cannot forget
        the V23 control distribution.
        """

        return replace(
            cls.v23_unified_head_bank(),
            direct_action_loss_weight=0.50,
            direct_action_inference_mix=0.25,
            train_direct_action_adapter=True,
            gripper_auxiliary=True,
            gripper_loss_weight=0.05,
            gripper_auxiliary_inference=True,
            gripper_auxiliary_inference_mix=0.10,
            gripper_only_adapter=False,
        )

    @classmethod
    def v43_frozen_v23_gripper_only(cls) -> FlowMatchingVLAConfig:
        """Keep the V23 policy path fixed and fit only gripper auxiliaries.

        V42's direct/task adapter updates regressed multi-object LIBERO-10
        tasks.  This variant deliberately leaves the V23 task residual,
        unified direct heads, flow expert, and progress adapter untouched;
        only the newly introduced binary gripper outputs are trainable.
        """

        return replace(
            cls.v23_unified_head_bank(),
            gripper_auxiliary=True,
            gripper_loss_weight=0.05,
            gripper_auxiliary_inference=True,
            gripper_auxiliary_inference_mix=0.10,
            gripper_only_adapter=True,
            train_direct_action_adapter=False,
            train_progress_adapter=False,
        )

    @classmethod
    def v45_v23_temporal_residual(cls) -> FlowMatchingVLAConfig:
        """Add zero-initialized history residuals without changing V23 heads.

        The strict audit's remaining failures are mostly multi-stage tasks.
        This variant supplies executed state/action history to a small
        per-head temporal conditioner while freezing V23's visual, Flow, task,
        direct-action, and progress paths.  The residual starts at zero, so
        its first-step behavior is the V23 policy and it cannot overwrite the
        proven action bank during initialization.
        """

        return replace(
            cls.v23_unified_head_bank(),
            temporal_conditioning=True,
            history_length=4,
            phase_classes=5,
            phase_loss_weight=0.02,
            temporal_fusion="state_residual",
            temporal_gate_init=0.02,
            temporal_zero_init=True,
            train_temporal_adapters=True,
            gripper_only_adapter=True,
            train_direct_action_adapter=False,
            train_progress_adapter=False,
        )

    @classmethod
    def v46_v23_frozen_temporal_gate(cls) -> FlowMatchingVLAConfig:
        """Use a tiny fixed temporal residual to preserve the V23 policy.

        V45's learned scalar gate became too large during full training and
        disturbed long-horizon multi-object behavior.  V46 keeps the same
        history adapter, but fixes its gate at 0.5% and removes the auxiliary
        phase objective, so the V23 action bank remains the dominant path.
        """

        return replace(
            cls.v23_unified_head_bank(),
            temporal_conditioning=True,
            history_length=4,
            phase_classes=5,
            phase_loss_weight=0.0,
            temporal_fusion="state_residual",
            temporal_gate_init=0.005,
            temporal_zero_init=True,
            train_temporal_adapters=True,
            train_temporal_gate=False,
            gripper_only_adapter=True,
            train_direct_action_adapter=False,
            train_progress_adapter=False,
        )

    @classmethod
    def v47_v23_target_direct_adapters(cls) -> FlowMatchingVLAConfig:
        """Adapt V23's direct action path without changing the Flow expert.

        This is intentionally a conservative, single-checkpoint experiment:
        the evaluator still infers the task from Qwen features, while the
        training driver may restrict gradients to the known training task IDs
        that failed the strict audit.  With ``--freeze-base-flow`` only the
        task residual and unified direct predictors are trainable; the proven
        V23 Flow path remains unchanged for all other tasks.
        """

        return replace(
            cls.v23_unified_head_bank(),
            direct_action_loss_weight=0.50,
            direct_action_inference_mix=0.25,
            train_direct_action_adapter=True,
        )

    @classmethod
    def v48_v23_target_task_residual(cls) -> FlowMatchingVLAConfig:
        """Fit only per-task residuals while retaining every shared V23 head.

        The V47 audit showed that updating shared unified direct predictors can
        transfer a hard-task correction to unrelated tasks.  This variant
        leaves the Flow and direct-action banks untouched and exposes only the
        task-conditioned residual to a filtered training stream.
        """

        return replace(
            cls.v23_unified_head_bank(),
            direct_action_loss_weight=0.25,
            direct_action_inference_mix=0.25,
            train_direct_action_adapter=False,
        )

    @classmethod
    def v49_v23_tail_task_residual(cls) -> FlowMatchingVLAConfig:
        """V48 with the same tail-token router used by strict evaluation."""

        return replace(
            cls.v48_v23_target_task_residual(),
            task_router_pooling="tail_tokens",
            task_router_tail_tokens=16,
        )

    @classmethod
    def v50_v23_tail_small_task_residual(cls) -> FlowMatchingVLAConfig:
        """Use a small task-residual scale to preserve V23's shared control."""

        return replace(
            cls.v49_v23_tail_task_residual(),
            task_conditioning_scale=0.10,
        )

    @classmethod
    def v51_v23_tail_task_direct_residual(cls) -> FlowMatchingVLAConfig:
        """Add zero-initialized per-task direct-action residual adapters.

        Unlike V47, no shared unified direct predictor is updated.  Each
        language-routed task owns a small residual branch, so adapting the
        four audited failures cannot overwrite another task that happens to
        share the same compact unified head.
        """

        return replace(
            cls.v23_unified_head_bank(),
            task_router_pooling="tail_tokens",
            task_router_tail_tokens=16,
            direct_action_loss_weight=0.50,
            direct_action_inference_mix=0.25,
            train_task_adapter=False,
            train_direct_action_adapter=False,
            task_direct_residual_adapter=True,
            task_direct_residual_scale=1.0,
            train_unified_head_adapter=False,
        )

    @classmethod
    def v52_progressive_residual(cls) -> FlowMatchingVLAConfig:
        """V23 action bank with stage-wise residual supervision.

        The Flow expert remains the action prior.  The direct head learns the
        residual to that prior and the optional task head learns the remaining
        residual.  All task residuals are enabled; task filtering is supplied
        by the training command only for an explicitly selected second stage.
        """

        return replace(
            cls.v23_unified_head_bank(),
            task_router_pooling="tail_tokens",
            task_router_tail_tokens=16,
            direct_action_loss_weight=0.50,
            direct_action_inference_mix=0.0,
            task_direct_residual_adapter=True,
            task_direct_residual_scale=1.0,
            progressive_residual=True,
            direct_residual_inference_scale=1.0,
            task_residual_inference_scale=1.0,
            monotonic_residual_loss_weight=0.05,
            monotonic_residual_margin=0.0,
            train_unified_head_adapter=False,
        )

    @classmethod
    def v52_progressive_direct(cls) -> FlowMatchingVLAConfig:
        """Stage B: train only the shared direct residual on all tasks."""

        return replace(
            cls.v52_progressive_residual(),
            task_direct_residual_adapter=False,
            monotonic_residual_loss_weight=0.0,
            train_unified_head_adapter=True,
        )

    @classmethod
    def v52_progressive_task(cls) -> FlowMatchingVLAConfig:
        """Stage D: add task residuals after the direct residual is frozen."""

        return replace(
            cls.v52_progressive_residual(),
            monotonic_residual_loss_weight=0.0,
        )

    @classmethod
    def v52_progressive_monotonic(cls) -> FlowMatchingVLAConfig:
        """Stage E: enforce non-degradation at both residual stages."""

        return cls.v52_progressive_residual()

    @classmethod
    def v53_global_residual(cls) -> FlowMatchingVLAConfig:
        """Keep V51 frozen and add one shared residual for all tasks.

        The existing V51 task-routed path is the frozen base.  The new adapter
        is task-agnostic and is trained from the exact base-action cache; no
        task id is consumed by this residual at inference.
        """

        return replace(
            cls.v51_v23_tail_task_direct_residual(),
            global_action_residual_adapter=True,
            global_action_residual_scale=1.0,
        )

    @classmethod
    def v54_state_adaptive_residual(cls) -> FlowMatchingVLAConfig:
        """Add a shared spatially grounded, state-adaptive residual to V51.

        The residual sees projected camera tokens, the state/history token,
        and the frozen V51 action chunk.  A per-step, per-dimension gate limits
        corrections, while a separate gripper logit avoids letting arm losses
        dominate grasp supervision.  Every output projection starts at zero,
        so an untrained V54 is exactly equivalent to V51.
        """

        return replace(
            cls.v51_v23_tail_task_direct_residual(),
            state_adaptive_residual_adapter=True,
            state_adaptive_residual_scale=1.0,
            state_adaptive_gate_init=0.02,
        )

    @classmethod
    def v56_v51_temporal_adapter(cls) -> FlowMatchingVLAConfig:
        """Keep V51's proven task residual and adapt only temporal features.

        V51 freezes the zero-initialized temporal state residual while fitting
        task-local direct residuals.  V56 exposes that existing temporal path
        as a shared, task-agnostic adapter; the task residual remains present
        and no benchmark task index is added to the model input.
        """

        return replace(
            cls.v51_v23_tail_task_direct_residual(),
            train_temporal_adapters=True,
            train_temporal_gate=True,
        )

    @classmethod
    def v60_v51_visual_feature_adapter(cls) -> FlowMatchingVLAConfig:
        """Adapt cached Qwen features without updating the 2.13B backbone."""

        return replace(
            cls.v51_v23_tail_task_direct_residual(),
            visual_feature_adapter=True,
            visual_feature_adapter_bottleneck=64,
            train_visual_feature_adapter=True,
        )

    @classmethod
    def v61_v51_state_feature_adapter(cls) -> FlowMatchingVLAConfig:
        """Adapt the normalized robot-state embedding with a zero residual."""

        return replace(
            cls.v51_v23_tail_task_direct_residual(),
            state_feature_adapter=True,
            state_feature_adapter_bottleneck=64,
            train_state_feature_adapter=True,
        )

    @classmethod
    def v62_v51_visual_state_adapters(cls) -> FlowMatchingVLAConfig:
        """Train only shared visual/state adapters while preserving V51 residuals."""

        return replace(
            cls.v51_v23_tail_task_direct_residual(),
            visual_feature_adapter=True,
            visual_feature_adapter_bottleneck=64,
            train_visual_feature_adapter=True,
            state_feature_adapter=True,
            state_feature_adapter_bottleneck=64,
            train_state_feature_adapter=True,
            train_task_direct_residual_adapter=False,
        )

    @classmethod
    def v66_v51_gripper_auxiliary(cls) -> FlowMatchingVLAConfig:
        """Add a conservative gripper classifier to the proven V51 path.

        The flow, direct-action, and task-residual branches remain frozen.  A
        zero-initialized auxiliary binary gripper head learns contact phase
        from the existing action-expert hidden state and contributes only a
        small inference blend.  It consumes no benchmark task index.
        """

        return replace(
            cls.v51_v23_tail_task_direct_residual(),
            gripper_auxiliary=True,
            gripper_loss_weight=0.05,
            gripper_auxiliary_inference=True,
            gripper_auxiliary_inference_mix=0.05,
            gripper_only_adapter=True,
        )

    @classmethod
    def v96_v51_gripper_gate(cls) -> FlowMatchingVLAConfig:
        """Use a very small gripper-only correction while preserving V51.

        V95 showed that a 20k-step, 5% auxiliary blend can harm the shared
        policy.  V96 isolates the same signal with a 1% inference blend and a
        lower supervised weight; the flow/direct/task-residual paths remain
        frozen and no benchmark task index is consumed.
        """

        return replace(
            cls.v51_v23_tail_task_direct_residual(),
            gripper_auxiliary=True,
            gripper_loss_weight=0.02,
            gripper_auxiliary_inference=True,
            gripper_auxiliary_inference_mix=0.01,
            gripper_only_adapter=True,
        )

    @classmethod
    def v97_v51_task_balanced_residual(cls) -> FlowMatchingVLAConfig:
        """Re-fit only V51's task residual bank with balanced all-task data.

        V51's published checkpoint was adapted with a four-task filter.  The
        400-state gate shows that its strong state-0 score does not transfer
        uniformly to the other official initial states.  V97 keeps the Qwen,
        Flow, shared direct heads, and normalization contract frozen, while
        allowing every language-routed task residual to see an equal stream of
        training examples.  No benchmark task index is consumed at inference.
        """

        return replace(
            cls.v51_v23_tail_task_direct_residual(),
            train_task_direct_residual_adapter=True,
            task_direct_residual_scale=1.0,
            gripper_auxiliary=False,
            gripper_only_adapter=False,
        )

    @classmethod
    def v98_last_residual(cls) -> FlowMatchingVLAConfig:
        """LaST-R1-inspired latent-reasoning residual on the V51 path.

        This is the offline, single-GPU first stage: compact latent CoT tokens
        are learned from demonstration action chunks and injected through a
        zero-initialized state residual.  LAPO-style online RL is deliberately
        a later stage and is not hidden in this preset.
        """

        return replace(
            cls.v51_v23_tail_task_direct_residual(),
            latent_reasoning=True,
            latent_reasoning_tokens=2,
            latent_reasoning_loss_weight=0.05,
            latent_end_loss_weight=0.01,
            latent_gate_init=0.01,
            train_temporal_adapters=False,
            train_temporal_gate=False,
            train_progress_adapter=False,
            train_context_fusion_adapter=False,
            # Keep the proven V51 task residual bank fixed for attribution.
            train_task_direct_residual_adapter=False,
            gripper_auxiliary=False,
            gripper_only_adapter=False,
        )

    @classmethod
    def v99_last_residual_action(cls) -> FlowMatchingVLAConfig:
        """LaST-R1 latent-to-action residual warm-up on the V51 policy.

        V98 only trained a latent auxiliary summary and a state fusion path.
        V99 makes the latent plan an auditable residual action adapter: a
        soft halting-weighted latent summary, the frozen parent action chunk,
        and the current visual/state context predict a bounded correction.
        This is still an offline warm-up; LAPO online reward optimization is
        intentionally a separate experiment and is not implied by this
        preset.
        """

        return replace(
            cls.v51_v23_tail_task_direct_residual(),
            latent_reasoning=True,
            latent_reasoning_tokens=4,
            latent_reasoning_loss_weight=0.02,
            latent_end_loss_weight=0.01,
            latent_gate_init=0.01,
            latent_residual_adapter=True,
            latent_residual_scale=0.25,
            latent_residual_gate_init=0.05,
            latent_residual_loss_weight=0.25,
            latent_adaptive_horizon=True,
            train_latent_residual_adapter=True,
            train_temporal_adapters=False,
            train_temporal_gate=False,
            train_progress_adapter=False,
            train_context_fusion_adapter=False,
            train_task_direct_residual_adapter=False,
            gripper_auxiliary=False,
            gripper_only_adapter=False,
        )

    @classmethod
    def v100_cac_latent_gate(cls) -> FlowMatchingVLAConfig:
        """Use a CAC-VLA-style latent condition inside zero-init Flow gates.

        V99 added a post-hoc action correction, which can overwrite a good
        parent chunk.  V100 keeps the V51 parent and latent warm-up, but sends
        the halting-weighted latent summary through the Flow expert's
        zero-initialized adaptive norms.  The latent plan therefore conditions
        denoising without an always-on additive correction; before training,
        the transferred policy remains behaviorally identical to V51.
        """

        return replace(
            cls.v51_v23_tail_task_direct_residual(),
            latent_reasoning=True,
            latent_reasoning_tokens=4,
            latent_reasoning_loss_weight=0.02,
            latent_end_loss_weight=0.01,
            latent_gate_init=0.01,
            latent_adaptive_horizon=True,
            deep_flow_conditioning=True,
            latent_residual_adapter=False,
            train_latent_residual_adapter=False,
            train_temporal_adapters=False,
            train_temporal_gate=False,
            train_progress_adapter=False,
            train_context_fusion_adapter=False,
            train_task_direct_residual_adapter=False,
            gripper_auxiliary=False,
            gripper_only_adapter=False,
        )

    @classmethod
    def v67_v51_state_film(cls) -> FlowMatchingVLAConfig:
        """Add a zero-initialized state-conditioned FiLM residual to V51."""

        return replace(
            cls.v51_v23_tail_task_direct_residual(),
            state_film_conditioning=True,
            train_film_adapter=True,
        )

    @classmethod
    def v68_v51_context_fusion(cls) -> FlowMatchingVLAConfig:
        """Learn a zero-init visual/state token fusion residual on V51."""

        return replace(
            cls.v51_v23_tail_task_direct_residual(),
            context_fusion_layers=1,
            context_fusion_zero_init=True,
            state_film_conditioning=True,
            train_film_adapter=False,
        )

    @classmethod
    def v70_grounded_language_residual(cls) -> FlowMatchingVLAConfig:
        """V51 residual policy with grounded Qwen context and robust routing.

        Grounded cache tokens are ordered as spatial, instruction-semantic,
        then goal-grounding tokens.  The action path sees all tokens while the
        task router combines semantic and spatial summaries, avoiding use of a
        benchmark task index.
        """

        return replace(
            cls.v51_v23_tail_task_direct_residual(),
            qwen_context_mode="grounded_language",
            qwen_semantic_tokens=8,
            qwen_semantic_layer=-1,
            qwen_grounding_tokens=8,
            task_router_pooling="semantic_visual_mix",
        )

    @classmethod
    def v71_ordered_robust_residual(cls) -> FlowMatchingVLAConfig:
        """V51 ordered-context path for state/feature augmentation experiments.

        Keep the audited V51 inference contract exactly unchanged.  Robustness
        is supplied by training-time state noise and token dropout, so an
        untrained V71 checkpoint is behaviorally identical to V51 and never
        depends on benchmark task indices.
        """

        return cls.v51_v23_tail_task_direct_residual()

    @classmethod
    def v72_ordered_visual_dropout_residual(cls) -> FlowMatchingVLAConfig:
        """V51 ordered-context path with visual-only training augmentation.

        This isolates feature dropout from V71's state-noise change; inference
        remains exactly the V51 contract and uses no benchmark task index.
        """

        return cls.v51_v23_tail_task_direct_residual()

    @classmethod
    def v73_ordered_relative_residual(cls) -> FlowMatchingVLAConfig:
        """Add an order-aware, continuous-language arm residual beside V51.

        V40 tokens are ordered prompt-token bins rather than a 2-D patch grid,
        so the module deliberately models relative token order and does not
        claim metric geometry.  Two continuous-language queries attend to the
        frozen ordered context, then predict a bounded 6-D arm correction.
        The seventh gripper dimension remains exactly V51.
        """

        return replace(
            cls.v51_v23_tail_task_direct_residual(),
            ordered_relative_residual_adapter=True,
            ordered_relative_residual_scale=0.25,
            ordered_relative_semantic_tokens=16,
        )

    @classmethod
    def v74_spatial_grid_residual(cls) -> FlowMatchingVLAConfig:
        """Add an exact Qwen image-grid sidecar beside frozen V51.

        The main ordered-context/Flow/direct/task-residual path remains the
        V51 contract. The task-agnostic sidecar consumes a separate
        ``[camera, 64 patch, PCA64]`` cache and only predicts arm residuals.
        Its output is zero-initialized, so an untrained V74 is exactly V51.
        """

        return replace(
            cls.v51_v23_tail_task_direct_residual(),
            spatial_grid_residual_adapter=True,
            spatial_grid_feature_dim=64,
            spatial_grid_residual_scale=0.25,
            spatial_grid_semantic_tokens=16,
        )

    def __post_init__(self) -> None:
        positive = (
            self.state_dim,
            self.action_dim,
            self.vlm_feature_dim,
            self.hidden_dim,
            self.action_horizon,
            self.max_context_tokens,
            self.expert_layers,
            self.expert_heads,
            self.num_flow_steps,
            self.self_attn_every_n_layers,
            self.state_projection_layers,
            self.camera_count,
            self.tokens_per_camera,
            self.history_length,
            self.phase_classes,
            self.visual_feature_adapter_bottleneck,
            self.state_feature_adapter_bottleneck,
            self.spatial_grid_feature_dim,
            self.latent_reasoning_tokens,
        )
        if any(value <= 0 for value in positive):
            raise ValueError("all flow-matching dimensions and step counts must be positive")
        if self.hidden_dim % self.expert_heads:
            raise ValueError("hidden_dim must be divisible by expert_heads")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if self.latent_reasoning_loss_weight < 0.0 or self.latent_end_loss_weight < 0.0:
            raise ValueError("latent reasoning loss weights must be non-negative")
        if not 0.0 < self.latent_gate_init < 1.0:
            raise ValueError("latent_gate_init must be in (0, 1)")
        if self.time_sampling not in {"uniform", "beta_1.5_1.0"}:
            raise ValueError("time_sampling must be uniform or beta_1.5_1.0")
        if self.context_fusion_layers < 0:
            raise ValueError("context_fusion_layers must be non-negative")
        if self.architecture_variant not in {"legacy", "smol_aligned"}:
            raise ValueError("architecture_variant must be legacy or smol_aligned")
        if self.attention_pattern not in {"periodic_cross_self", "alternating"}:
            raise ValueError("unsupported attention_pattern")
        if self.ffn_type not in {"gelu", "swiglu"}:
            raise ValueError("ffn_type must be gelu or swiglu")
        if self.norm_type not in {"layer_norm", "rms_norm"}:
            raise ValueError("norm_type must be layer_norm or rms_norm")
        if self.state_projection_layers not in {1, 2}:
            raise ValueError("state_projection_layers must be one or two")
        if self.state_conditioning not in {"expert", "prefix"}:
            raise ValueError("state_conditioning must be expert or prefix")
        if self.state_conditioning == "prefix" and self.context_fusion_layers == 0:
            raise ValueError("prefix state conditioning requires context fusion layers")
        if self.context_layout not in {"merged", "per_camera"}:
            raise ValueError("context_layout must be merged or per_camera")
        if (
            self.context_layout == "per_camera"
            and self.max_context_tokens != self.camera_count * self.tokens_per_camera
        ):
            raise ValueError("per-camera context size must equal camera_count * tokens_per_camera")
        if self.qwen_hidden_layer < -1:
            raise ValueError("qwen_hidden_layer must be -1 or a non-negative layer index")
        if self.qwen_context_mode not in {"ordered", "hybrid_language", "grounded_language"}:
            raise ValueError(
                "qwen_context_mode must be ordered, hybrid_language, or grounded_language"
            )
        if self.qwen_semantic_layer < -1:
            raise ValueError("qwen_semantic_layer must be -1 or a non-negative layer index")
        if self.qwen_context_mode == "ordered" and (
            self.qwen_semantic_tokens != 0 or self.qwen_grounding_tokens != 0
        ):
            raise ValueError("ordered Qwen context cannot declare semantic or grounding tokens")
        if self.qwen_context_mode == "hybrid_language" and not (
            0 < self.qwen_semantic_tokens < self.tokens_per_camera
            and self.qwen_grounding_tokens == 0
        ):
            raise ValueError("hybrid Qwen context requires semantic tokens and no grounding tokens")
        if self.qwen_context_mode == "grounded_language" and not (
            self.qwen_semantic_tokens > 0
            and self.qwen_grounding_tokens > 0
            and self.qwen_semantic_tokens + self.qwen_grounding_tokens < self.tokens_per_camera
        ):
            raise ValueError("grounded Qwen context requires spatial, semantic, and grounding tokens")
        if not 0.0 < self.min_period < self.max_period:
            raise ValueError("time periods must satisfy 0 < min_period < max_period")
        if not 0.0 <= self.phase_loss_weight:
            raise ValueError("phase_loss_weight must be non-negative")
        if self.temporal_fusion not in {"append", "state_residual"}:
            raise ValueError("temporal_fusion must be append or state_residual")
        if not 0.0 < self.temporal_gate_init <= 1.0:
            raise ValueError("temporal_gate_init must be in (0, 1]")
        if self.temporal_conditioning and self.phase_classes != 5:
            raise ValueError("the initial V8 phase contract requires exactly five phase classes")
        if not self.temporal_conditioning and self.temporal_fusion != "append":
            raise ValueError("temporal_fusion is only configurable with temporal_conditioning")
        if not self.temporal_conditioning and self.temporal_zero_init:
            raise ValueError("temporal_zero_init requires temporal_conditioning")
        if self.history_only_conditioning and not self.temporal_conditioning:
            raise ValueError("history_only_conditioning requires temporal_conditioning")
        if self.selective_progress_gate and not (
            self.temporal_conditioning and self.temporal_fusion == "state_residual"
        ):
            raise ValueError("selective_progress_gate requires state-residual temporal conditioning")
        if self.progress_conditioning and not self.history_only_conditioning:
            raise ValueError("progress_conditioning requires history_only_conditioning")
        if self.progress_classes != 4:
            raise ValueError("the initial progress contract requires four subtask classes")
        if self.progress_loss_weight < 0.0:
            raise ValueError("progress_loss_weight must be non-negative")
        if self.global_action_residual_scale < 0.0:
            raise ValueError("global_action_residual_scale must be non-negative")
        if self.state_adaptive_residual_scale < 0.0:
            raise ValueError("state_adaptive_residual_scale must be non-negative")
        if not 0.0 < self.state_adaptive_gate_init < 1.0:
            raise ValueError("state_adaptive_gate_init must be in (0, 1)")
        if self.ordered_relative_residual_scale < 0.0:
            raise ValueError("ordered_relative_residual_scale must be non-negative")
        if self.ordered_relative_residual_adapter and not (
            0 < self.ordered_relative_semantic_tokens <= self.tokens_per_camera
        ):
            raise ValueError(
                "ordered_relative_semantic_tokens must be in [1, tokens_per_camera]"
            )
        if self.ordered_relative_residual_adapter and self.context_layout != "per_camera":
            raise ValueError("ordered relative residual requires per-camera context")
        if self.spatial_grid_residual_scale < 0.0:
            raise ValueError("spatial_grid_residual_scale must be non-negative")
        if self.spatial_grid_residual_adapter and self.context_layout != "per_camera":
            raise ValueError("spatial grid residual requires per-camera V51 context")
        if self.spatial_grid_residual_adapter and not (
            0 < self.spatial_grid_semantic_tokens <= self.tokens_per_camera
        ):
            raise ValueError("spatial_grid_semantic_tokens must be in [1, tokens_per_camera]")
        if self.visual_memory_length <= 0 or self.visual_memory_tokens_per_camera <= 0:
            raise ValueError("visual memory dimensions must be positive")
        if self.visual_memory_conditioning and self.context_layout != "per_camera":
            raise ValueError("visual memory currently requires per-camera context")
        if self.visual_memory_conditioning and (
            self.tokens_per_camera % self.visual_memory_tokens_per_camera
        ):
            raise ValueError("visual memory tokens must divide tokens_per_camera")
        if self.visual_memory_adapter_layers < 0:
            raise ValueError("visual_memory_adapter_layers must be non-negative")
        if self.visual_memory_adapter_layers >= self.expert_layers:
            raise ValueError("visual memory adapter layers must be fewer than expert layers")
        if self.gated_visual_memory and not (
            self.visual_memory_conditioning
            and self.visual_memory_adapter_layers > 0
            and self.attention_pattern == "alternating"
        ):
            raise ValueError(
                "gated visual memory requires visual memory, adapter layers, and alternating attention"
            )
        if self.persistent_memory_slots <= 0 or self.persistent_memory_heads <= 0:
            raise ValueError("persistent memory slots and heads must be positive")
        if self.persistent_progress_memory and not self.gated_visual_memory:
            raise ValueError("persistent progress memory requires gated visual memory")
        if self.persistent_progress_memory and self.hidden_dim % self.persistent_memory_heads:
            raise ValueError("hidden_dim must be divisible by persistent memory heads")
        if self.preserve_block_norm_conditioning and not self.deep_flow_conditioning:
            raise ValueError("preserved block-norm conditioning requires deep Flow conditioning")
        if self.gripper_loss_weight < 0.0:
            raise ValueError("gripper_loss_weight must be non-negative")
        if not 0.0 <= self.gripper_auxiliary_inference_mix <= 1.0:
            raise ValueError("gripper_auxiliary_inference_mix must be in [0, 1]")
        if self.direct_action_loss_weight < 0.0:
            raise ValueError("direct_action_loss_weight must be non-negative")
        if not 0.0 <= self.direct_action_inference_mix <= 1.0:
            raise ValueError("direct_action_inference_mix must be in [0, 1]")
        if self.direct_action_inference_mix > 0.0 and not self.direct_action_head:
            raise ValueError("direct_action_inference_mix requires direct_action_head")
        if self.task_direct_residual_scale < 0.0:
            raise ValueError("task_direct_residual_scale must be non-negative")
        if self.task_direct_residual_task_scales is not None:
            if len(self.task_direct_residual_task_scales) != self.task_classes:
                raise ValueError(
                    "task_direct_residual_task_scales must contain one value per task"
                )
            if any(float(value) < 0.0 for value in self.task_direct_residual_task_scales):
                raise ValueError("task_direct_residual_task_scales must be non-negative")
        if self.task_direct_residual_adapter and not (
            self.task_conditioning and self.direct_action_head
        ):
            raise ValueError(
                "task direct residual adapters require task conditioning and a direct action head"
            )
        if self.task_classes <= 0:
            raise ValueError("task_classes must be positive")
        if self.task_conditioning_scale < 0.0:
            raise ValueError("task_conditioning_scale must be non-negative")
        if self.task_router_pooling not in {"all_tokens", "tail_tokens", "semantic_tokens", "semantic_visual_mix"}:
            raise ValueError("task_router_pooling must be all_tokens, tail_tokens, semantic_tokens, or semantic_visual_mix")
        if self.task_router_tail_tokens <= 0:
            raise ValueError("task_router_tail_tokens must be positive")
        if self.task_router_pooling == "tail_tokens" and not self.task_conditioning:
            raise ValueError("tail-token task routing requires task conditioning")
        if self.task_router_pooling in {"semantic_tokens", "semantic_visual_mix"} and not (
            self.task_conditioning
            and self.qwen_context_mode in {"hybrid_language", "grounded_language"}
            and self.qwen_semantic_tokens > 0
        ):
            raise ValueError(
                "semantic task routing requires task conditioning and explicit Qwen semantic tokens"
            )
        if self.unified_head_count <= 0:
            raise ValueError("unified_head_count must be positive")
        if self.unified_head_bank and not self.task_conditioning:
            raise ValueError("unified head bank requires task conditioning")
        if self.unified_action_expert_bank and not self.unified_head_bank:
            raise ValueError("unified action experts require a unified head bank")

    @property
    def extra_context_tokens(self) -> int:
        """Number of non-visual tokens appended to the Qwen context."""

        return 2 if self.temporal_conditioning and self.temporal_fusion == "append" else 0

    @property
    def action_context_tokens(self) -> int:
        """Maximum context visible to the action expert, including visual memory."""

        if not self.visual_memory_conditioning or self.gated_visual_memory:
            return self.max_context_tokens
        return self.max_context_tokens + (
            self.visual_memory_length
            * self.camera_count
            * self.visual_memory_tokens_per_camera
        )


def _make_norm(hidden_dim: int, norm_type: str) -> nn.Module:
    if norm_type == "rms_norm":
        return nn.RMSNorm(hidden_dim, eps=1e-6)
    return nn.LayerNorm(hidden_dim)


class _StateProjection(nn.Module):
    def __init__(
        self,
        state_dim: int,
        hidden_dim: int,
        *,
        layers: int,
        norm_type: str,
    ) -> None:
        super().__init__()
        if layers == 1:
            self.net = nn.Linear(state_dim, hidden_dim)
        else:
            input_norm: nn.Module = (
                nn.RMSNorm(state_dim, eps=1e-6)
                if norm_type == "rms_norm"
                else nn.LayerNorm(state_dim)
            )
            self.net = nn.Sequential(
                input_norm,
                nn.Linear(state_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )

    def forward(self, state: Tensor) -> Tensor:
        if state.ndim != 2:
            raise ValueError(f"state must have shape [batch, state_dim], got {tuple(state.shape)}")
        return self.net(state)


class _TimeProjection(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        *,
        min_period: float,
        max_period: float,
        smol_periods: bool,
    ) -> None:
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        half = hidden_dim // 2
        if smol_periods:
            fraction = torch.linspace(0.0, 1.0, half, dtype=torch.float32)
            periods = min_period * (max_period / min_period) ** fraction
            frequencies = 2.0 * math.pi / periods
        else:
            frequencies = torch.exp(torch.linspace(math.log(1.0), math.log(1000.0), half))
        self.register_buffer("frequencies", frequencies, persistent=False)

    def forward(self, time: Tensor) -> Tensor:
        if time.ndim != 1:
            raise ValueError(f"time must have shape [batch], got {tuple(time.shape)}")
        angles = time.float()[:, None] * self.frequencies[None, :]
        embedding = torch.cat((angles.sin(), angles.cos()), dim=-1)
        first = self.projection[0]
        if not isinstance(first, nn.Linear):  # pragma: no cover - construction invariant
            raise RuntimeError("time projection must start with a linear layer")
        return self.projection(embedding.to(dtype=first.weight.dtype))


class _FeedForward(nn.Module):
    def __init__(self, hidden_dim: int, dropout: float, *, ffn_type: str, norm_type: str) -> None:
        super().__init__()
        self.ffn_type = ffn_type
        if ffn_type == "gelu":
            self.net: nn.Module | None = nn.Sequential(
                _make_norm(hidden_dim, norm_type),
                nn.Linear(hidden_dim, 4 * hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(4 * hidden_dim, hidden_dim),
            )
            self.norm = self.gate = self.up = self.down = self.dropout = None
        else:
            intermediate = 256 * math.ceil((8 * hidden_dim / 3) / 256)
            self.net = None
            self.norm = _make_norm(hidden_dim, norm_type)
            self.gate = nn.Linear(hidden_dim, intermediate, bias=False)
            self.up = nn.Linear(hidden_dim, intermediate, bias=False)
            self.down = nn.Linear(intermediate, hidden_dim, bias=False)
            self.dropout = nn.Dropout(dropout)

    def forward(self, value: Tensor) -> Tensor:
        if self.net is not None:
            return value + self.net(value)
        if any(module is None for module in (self.norm, self.gate, self.up, self.down, self.dropout)):
            raise RuntimeError("SwiGLU feed-forward modules are incomplete")
        normalized = self.norm(value)
        gated = torch.nn.functional.silu(self.gate(normalized)) * self.up(normalized)
        return value + self.down(self.dropout(gated))


class _ZeroInitFiLM(nn.Module):
    """Condition action tokens without perturbing the V7 initialization."""

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.projection = nn.Linear(hidden_dim, 2 * hidden_dim)
        nn.init.zeros_(self.projection.weight)
        nn.init.zeros_(self.projection.bias)

    def forward(self, tokens: Tensor, condition: Tensor) -> Tensor:
        if condition.ndim != 2 or condition.shape[0] != tokens.shape[0]:
            raise ValueError("FiLM condition must have shape [batch, hidden_dim]")
        scale, bias = self.projection(condition).chunk(2, dim=-1)
        return tokens * (1.0 + scale[:, None, :]) + bias[:, None, :]


class _ZeroInitAdaRMS(nn.Module):
    """Apply a zero-init adaptive RMS/LayerNorm modulation."""

    def __init__(self, hidden_dim: int, norm_type: str) -> None:
        super().__init__()
        self.norm = _make_norm(hidden_dim, norm_type)
        self.projection = nn.Linear(hidden_dim, 2 * hidden_dim)
        nn.init.zeros_(self.projection.weight)
        nn.init.zeros_(self.projection.bias)

    def forward(self, tokens: Tensor, condition: Tensor) -> Tensor:
        if condition.ndim != 2 or condition.shape[0] != tokens.shape[0]:
            raise ValueError("AdaRMS condition must have shape [batch, hidden_dim]")
        scale, bias = self.projection(condition).chunk(2, dim=-1)
        normalized = self.norm(tokens)
        return normalized * (1.0 + scale[:, None, :]) + bias[:, None, :]


class _ZeroInitNormModulation(nn.Module):
    """Modulate an already-normalized tensor while preserving the base norm."""

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.projection = nn.Linear(hidden_dim, 2 * hidden_dim)
        nn.init.zeros_(self.projection.weight)
        nn.init.zeros_(self.projection.bias)

    def forward(self, normalized_tokens: Tensor, condition: Tensor) -> Tensor:
        if condition.ndim != 2 or condition.shape[0] != normalized_tokens.shape[0]:
            raise ValueError("norm condition must have shape [batch, hidden_dim]")
        scale, bias = self.projection(condition).chunk(2, dim=-1)
        return normalized_tokens * (1.0 + scale[:, None, :]) + bias[:, None, :]


class _GatedMemoryAdapter(nn.Module):
    """Read history tokens through an exactly zero-initialized residual gate."""

    def __init__(self, config: FlowMatchingVLAConfig) -> None:
        super().__init__()
        self.query_norm = _make_norm(config.hidden_dim, config.norm_type)
        self.memory_norm = _make_norm(config.hidden_dim, config.norm_type)
        self.attention = nn.MultiheadAttention(
            config.hidden_dim,
            config.expert_heads,
            dropout=config.dropout,
            bias=config.attention_bias,
            batch_first=True,
        )
        self.output = nn.Linear(config.hidden_dim, config.hidden_dim, bias=False)
        self.gate = nn.Parameter(torch.zeros((), dtype=torch.float32))

    def forward(self, action_tokens: Tensor, memory_tokens: Tensor) -> Tensor:
        if memory_tokens.ndim != 3 or memory_tokens.shape[0] != action_tokens.shape[0]:
            raise ValueError("memory_tokens must have shape [batch, memory_tokens, hidden_dim]")
        attended, _ = self.attention(
            self.query_norm(action_tokens),
            self.memory_norm(memory_tokens),
            self.memory_norm(memory_tokens),
            need_weights=False,
        )
        residual = self.output(attended)
        return action_tokens + self.gate.to(residual.dtype) * residual


class _CrossBlock(nn.Module):
    def __init__(self, config: FlowMatchingVLAConfig) -> None:
        super().__init__()
        self.norm = _make_norm(config.hidden_dim, config.norm_type)
        self.attention = nn.MultiheadAttention(
            config.hidden_dim,
            config.expert_heads,
            dropout=config.dropout,
            bias=config.attention_bias,
            batch_first=True,
        )
        self.feed_forward = _FeedForward(
            config.hidden_dim,
            config.dropout,
            ffn_type=config.ffn_type,
            norm_type=config.norm_type,
        )
        self.adaptive_norm = (
            _ZeroInitAdaRMS(config.hidden_dim, config.norm_type)
            if config.deep_flow_conditioning and not config.preserve_block_norm_conditioning
            else None
        )
        self.adaptive_modulation = (
            _ZeroInitNormModulation(config.hidden_dim)
            if config.preserve_block_norm_conditioning
            else None
        )

    def forward(
        self,
        action_tokens: Tensor,
        context: Tensor,
        condition: Tensor | None = None,
    ) -> Tensor:
        if self.adaptive_modulation is not None:
            if condition is None:
                raise ValueError("deep Flow conditioning requires a block condition")
            query = self.adaptive_modulation(self.norm(action_tokens), condition)
        elif self.adaptive_norm is not None:
            if condition is None:
                raise ValueError("deep Flow conditioning requires a block condition")
            query = self.adaptive_norm(action_tokens, condition)
        else:
            query = self.norm(action_tokens)
        attended, _ = self.attention(query, context, context, need_weights=False)
        return self.feed_forward(action_tokens + attended)


class _SelfBlock(nn.Module):
    def __init__(self, config: FlowMatchingVLAConfig) -> None:
        super().__init__()
        self.norm = _make_norm(config.hidden_dim, config.norm_type)
        self.attention = nn.MultiheadAttention(
            config.hidden_dim,
            config.expert_heads,
            dropout=config.dropout,
            bias=config.attention_bias,
            batch_first=True,
        )
        self.feed_forward = _FeedForward(
            config.hidden_dim,
            config.dropout,
            ffn_type=config.ffn_type,
            norm_type=config.norm_type,
        )
        self.adaptive_norm = (
            _ZeroInitAdaRMS(config.hidden_dim, config.norm_type)
            if config.deep_flow_conditioning and not config.preserve_block_norm_conditioning
            else None
        )
        self.adaptive_modulation = (
            _ZeroInitNormModulation(config.hidden_dim)
            if config.preserve_block_norm_conditioning
            else None
        )
        self.register_buffer(
            "causal_mask",
            torch.triu(
                torch.ones(config.action_horizon, config.action_horizon, dtype=torch.bool),
                diagonal=1,
            ),
            persistent=False,
        )
        self.causal_action_attention = config.causal_action_attention

    def forward(self, action_tokens: Tensor, condition: Tensor | None = None) -> Tensor:
        if self.adaptive_modulation is not None:
            if condition is None:
                raise ValueError("deep Flow conditioning requires a block condition")
            query = self.adaptive_modulation(self.norm(action_tokens), condition)
        elif self.adaptive_norm is not None:
            if condition is None:
                raise ValueError("deep Flow conditioning requires a block condition")
            query = self.adaptive_norm(action_tokens, condition)
        else:
            query = self.norm(action_tokens)
        length = action_tokens.shape[1]
        attention_mask = (
            self.causal_mask[:length, :length]
            if self.causal_action_attention
            else None
        )
        attended, _ = self.attention(
            query,
            query,
            query,
            attn_mask=attention_mask,
            need_weights=False,
        )
        return self.feed_forward(action_tokens + attended)


class _ContextFusionBlock(nn.Module):
    """Bidirectionally fuse frozen Qwen tokens with the robot-state token."""

    def __init__(self, config: FlowMatchingVLAConfig) -> None:
        super().__init__()
        self.norm = _make_norm(config.hidden_dim, config.norm_type)
        self.attention = nn.MultiheadAttention(
            config.hidden_dim,
            config.expert_heads,
            dropout=config.dropout,
            bias=config.attention_bias,
            batch_first=True,
        )
        self.feed_forward = _FeedForward(
            config.hidden_dim,
            config.dropout,
            ffn_type=config.ffn_type,
            norm_type=config.norm_type,
        )
        self.zero_init = bool(config.context_fusion_zero_init)
        self.residual_gate = nn.Parameter(torch.zeros(())) if self.zero_init else None

    def forward(self, tokens: Tensor) -> Tensor:
        query = self.norm(tokens)
        attended, _ = self.attention(query, query, query, need_weights=False)
        fused = self.feed_forward(tokens + attended)
        if self.residual_gate is None:
            return fused
        return tokens + self.residual_gate.to(fused.dtype) * (fused - tokens)


class FlowMatchingActionExpert(nn.Module):
    """Interleaved cross/self-attention expert predicting action velocity."""

    def __init__(self, config: FlowMatchingVLAConfig) -> None:
        super().__init__()
        self.action_input = nn.Linear(config.action_dim, config.hidden_dim)
        self.time_projection = _TimeProjection(
            config.hidden_dim,
            min_period=config.min_period,
            max_period=config.max_period,
            smol_periods=config.architecture_variant == "smol_aligned",
        )
        self.time_fusion = nn.Sequential(
            nn.Linear(2 * config.hidden_dim, config.hidden_dim),
            nn.SiLU(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
        )
        self.action_position = nn.Parameter(
            torch.randn(config.action_horizon, config.hidden_dim) * 0.02
        )
        self.context_position = nn.Parameter(
            torch.randn(
                config.action_context_tokens + config.extra_context_tokens + 1,
                config.hidden_dim,
            )
            * 0.02
        )
        cross_count = (
            config.expert_layers
            if config.attention_pattern == "periodic_cross_self"
            else (config.expert_layers + 1) // 2
        )
        self.cross_blocks = nn.ModuleList(_CrossBlock(config) for _ in range(cross_count))
        self.self_attn_every_n_layers = config.self_attn_every_n_layers
        self.attention_pattern = config.attention_pattern
        self.expert_layers = config.expert_layers
        self_count = (
            config.expert_layers // config.self_attn_every_n_layers
            if config.attention_pattern == "periodic_cross_self"
            else config.expert_layers // 2
        )
        self.self_blocks = nn.ModuleList(
            _SelfBlock(config) for _ in range(self_count)
        )
        self.memory_adapters = nn.ModuleList(
            _GatedMemoryAdapter(config)
            for _ in range(config.visual_memory_adapter_layers)
        )
        self.memory_adapter_points = tuple(
            round((index + 1) * config.expert_layers / (config.visual_memory_adapter_layers + 1))
            - 1
            for index in range(config.visual_memory_adapter_layers)
        )
        self.output = nn.Sequential(
            _make_norm(config.hidden_dim, config.norm_type),
            nn.Linear(config.hidden_dim, config.action_dim),
        )
        self.film_conditioner = (
            _ZeroInitFiLM(config.hidden_dim)
            if config.state_film_conditioning
            else None
        )
        self.gripper_output = (
            nn.Linear(config.hidden_dim, 1) if config.gripper_auxiliary else None
        )
        if self.gripper_output is not None and config.gripper_only_adapter:
            # Preserve the proven V51 continuous gripper path at initialization;
            # the auxiliary classifier is introduced as a zero residual.
            nn.init.zeros_(self.gripper_output.weight)
            nn.init.zeros_(self.gripper_output.bias)

    def forward(
        self,
        context: Tensor,
        state_token: Tensor,
        noisy_actions: Tensor,
        time: Tensor,
        *,
        film_condition: Tensor | None = None,
        deep_condition: Tensor | None = None,
        visual_memory: Tensor | None = None,
        return_hidden: bool = False,
    ) -> Tensor | tuple[Tensor, Tensor]:
        if context.ndim != 3:
            raise ValueError("context must have shape [batch, tokens, hidden]")
        if state_token.ndim != 3 or state_token.shape[1] != 1:
            raise ValueError("state_token must have shape [batch, 1, hidden]")
        if noisy_actions.ndim != 3:
            raise ValueError("noisy_actions must have shape [batch, horizon, action_dim]")
        if self.memory_adapters and visual_memory is None:
            raise ValueError("gated visual memory requires visual_memory")
        if not self.memory_adapters and visual_memory is not None:
            raise ValueError("visual_memory requires configured memory adapters")
        if context.shape[1] > self.context_position.shape[0] - 1:
            raise ValueError("context exceeds configured max_context_tokens")
        action_tokens = self.action_input(noisy_actions)
        time_token = self.time_projection(time)[:, None, :].expand_as(action_tokens)
        action_tokens = action_tokens + self.time_fusion(
            torch.cat((action_tokens, time_token), dim=-1)
        )
        action_tokens = action_tokens + self.action_position[None, :, :]
        if self.film_conditioner is not None:
            if film_condition is None:
                raise ValueError("state FiLM conditioning requires film_condition")
            action_tokens = self.film_conditioner(action_tokens, film_condition)
        context = context + self.context_position[: context.shape[1]][None, :, :]
        state_token = state_token + self.context_position[context.shape[1] : context.shape[1] + 1]
        context_with_state = torch.cat((context, state_token), dim=1)
        if self.attention_pattern == "alternating":
            cross_index = 0
            self_index = 0
            memory_index = 0
            for index in range(self.expert_layers):
                if index % 2 == 0:
                    action_tokens = self.cross_blocks[cross_index](
                        action_tokens, context_with_state, deep_condition
                    )
                    cross_index += 1
                else:
                    action_tokens = self.self_blocks[self_index](action_tokens, deep_condition)
                    self_index += 1
                if (
                    memory_index < len(self.memory_adapter_points)
                    and index == self.memory_adapter_points[memory_index]
                ):
                    if visual_memory is None:  # pragma: no cover - narrowed above
                        raise RuntimeError("visual memory is missing")
                    action_tokens = self.memory_adapters[memory_index](
                        action_tokens, visual_memory
                    )
                    memory_index += 1
        else:
            self_index = 0
            for index, cross in enumerate(self.cross_blocks):
                action_tokens = cross(action_tokens, context_with_state, deep_condition)
                if (
                    (index + 1) % self.self_attn_every_n_layers == 0
                    and self_index < len(self.self_blocks)
                ):
                    action_tokens = self.self_blocks[self_index](action_tokens, deep_condition)
                    self_index += 1
        velocity = self.output(action_tokens)
        if return_hidden:
            return velocity, action_tokens
        return velocity


class LatentReasoningAdapter(nn.Module):
    """Compact latent-CoT adapter used by the offline LaST-R1 stage.

    The adapter reads the current multimodal context and state, then emits a
    small set of action-space latent tokens.  They are supervision targets
    during offline training and are injected into the policy only through the
    parent-preserving zero-initialized fusion layer in ``FlowMatchingVLAPolicy``.
    """

    def __init__(self, config: FlowMatchingVLAConfig) -> None:
        super().__init__()
        self.queries = nn.Parameter(
            torch.randn(config.latent_reasoning_tokens, config.hidden_dim) * 0.02
        )
        self.attention = nn.MultiheadAttention(
            config.hidden_dim,
            config.expert_heads,
            bias=False,
            batch_first=True,
        )
        self.trunk = nn.Sequential(
            _make_norm(config.hidden_dim, config.norm_type),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.SiLU(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
        )
        self.action_projection = nn.Linear(config.hidden_dim, config.action_dim)
        self.end_projection = nn.Linear(config.hidden_dim, 1)

    def forward(
        self, context: Tensor, state_token: Tensor
    ) -> tuple[Tensor, Tensor, Tensor]:
        if context.ndim != 3 or state_token.ndim != 3 or state_token.shape[1] != 1:
            raise ValueError("latent reasoning expects context [B,T,H] and state [B,1,H]")
        query = self.queries[None, :, :].expand(context.shape[0], -1, -1)
        query = query + state_token
        source = torch.cat((context, state_token), dim=1)
        latent, _ = self.attention(query, source, source, need_weights=False)
        latent = latent + self.trunk(latent)
        summary = latent.mean(dim=1)
        action_summary = self.action_projection(latent)
        end_logits = self.end_projection(latent).squeeze(-1)
        return latent, action_summary, end_logits


def pool_task_routing_features(
    features: Tensor,
    *,
    pooling: str,
    semantic_tokens: int = 0,
    grounding_tokens: int = 0,
) -> Tensor:
    """Return one Qwen-space routing vector per sample.

    ``semantic_tokens`` uses the token layout emitted by
    :class:`QwenVLBackbone`: spatial, then instruction, then optional grounded
    goal tokens, independently for each camera.  Keeping this operation pure
    makes prototype construction and online inference share one exact
    contract.
    """

    if pooling == "all_tokens":
        if features.ndim == 4:
            return features.mean(dim=(1, 2))
        if features.ndim == 3:
            return features.mean(dim=1)
        if features.ndim == 2:
            return features
        raise ValueError("features must have rank 2, 3, or 4")
    if pooling == "tail_tokens":
        if features.ndim != 4 or semantic_tokens <= 0 or semantic_tokens > features.shape[2]:
            raise ValueError("tail-token routing requires rank-4 features and a valid tail length")
        return features[:, :, -semantic_tokens:, :].mean(dim=(1, 2))
    if pooling not in {"semantic_tokens", "semantic_visual_mix"}:
        raise ValueError("pooling must be all_tokens, tail_tokens, semantic_tokens, or semantic_visual_mix")
    if features.ndim != 4:
        raise ValueError("semantic routing requires rank-4 per-camera features")
    tokens_per_camera = features.shape[2]
    if semantic_tokens <= 0 or grounding_tokens < 0:
        raise ValueError("semantic routing requires positive semantic token count")
    semantic_stop = tokens_per_camera - grounding_tokens
    semantic_start = semantic_stop - semantic_tokens
    if semantic_start < 0 or semantic_stop > tokens_per_camera:
        raise ValueError("semantic/grounding token counts exceed tokens per camera")
    semantic = features[:, :, semantic_start:semantic_stop, :].mean(dim=(1, 2))
    if pooling == "semantic_tokens":
        return semantic
    spatial = features[:, :, :semantic_start, :].mean(dim=(1, 2))
    semantic = torch.nn.functional.normalize(semantic, dim=-1)
    spatial = torch.nn.functional.normalize(spatial, dim=-1)
    return (semantic + spatial).mul(0.5)


class FlowMatchingVLAPolicy(nn.Module):
    """Frozen-Qwen-conditioned SmolVLA-style flow-matching policy."""

    def __init__(self, config: FlowMatchingVLAConfig | None = None) -> None:
        super().__init__()
        self.config = config or FlowMatchingVLAConfig()
        self.feature_norm = (
            nn.LayerNorm(self.config.vlm_feature_dim)
            if self.config.feature_layer_norm
            else nn.Identity()
        )
        self.visual_feature_adapter = (
            nn.Sequential(
                nn.LayerNorm(self.config.vlm_feature_dim),
                nn.Linear(
                    self.config.vlm_feature_dim,
                    self.config.visual_feature_adapter_bottleneck,
                ),
                nn.SiLU(),
                nn.Linear(
                    self.config.visual_feature_adapter_bottleneck,
                    self.config.vlm_feature_dim,
                ),
            )
            if self.config.visual_feature_adapter
            else None
        )
        if self.visual_feature_adapter is not None:
            output = self.visual_feature_adapter[-1]
            if not isinstance(output, nn.Linear):  # pragma: no cover - defensive
                raise RuntimeError("visual feature adapter output must be linear")
            nn.init.zeros_(output.weight)
            nn.init.zeros_(output.bias)
        self.vlm_projection = nn.Linear(self.config.vlm_feature_dim, self.config.hidden_dim)
        self.state_projection = _StateProjection(
            self.config.state_dim,
            self.config.hidden_dim,
            layers=self.config.state_projection_layers,
            norm_type=self.config.norm_type,
        )
        self.state_feature_adapter = (
            nn.Sequential(
                nn.LayerNorm(self.config.state_dim),
                nn.Linear(self.config.state_dim, self.config.state_feature_adapter_bottleneck),
                nn.SiLU(),
                nn.Linear(self.config.state_feature_adapter_bottleneck, self.config.hidden_dim),
            )
            if self.config.state_feature_adapter
            else None
        )
        if self.state_feature_adapter is not None:
            output = self.state_feature_adapter[-1]
            if not isinstance(output, nn.Linear):  # pragma: no cover - defensive
                raise RuntimeError("state feature adapter output must be linear")
            nn.init.zeros_(output.weight)
            nn.init.zeros_(output.bias)
        self.camera_position = (
            nn.Parameter(torch.randn(self.config.camera_count, self.config.hidden_dim) * 0.02)
            if self.config.context_layout == "per_camera"
            else None
        )
        self.context_fusion_position = (
            nn.Parameter(
                torch.randn(self.config.max_context_tokens + 1, self.config.hidden_dim) * 0.02
            )
            if self.config.context_fusion_layers
            else None
        )
        self.context_fusion = nn.ModuleList(
            _ContextFusionBlock(self.config)
            for _ in range(self.config.context_fusion_layers)
        )
        self.visual_memory_position = (
            nn.Parameter(
                torch.randn(
                    self.config.visual_memory_length
                    * self.config.camera_count
                    * self.config.visual_memory_tokens_per_camera,
                    self.config.hidden_dim,
                )
                * 0.02
            )
            if self.config.visual_memory_conditioning
            else None
        )
        self.visual_memory_fusion = (
            nn.Sequential(
                _make_norm(self.config.hidden_dim, self.config.norm_type),
                nn.Linear(self.config.hidden_dim, self.config.hidden_dim),
            )
            if self.config.visual_memory_conditioning
            and self.config.visual_memory_summary_conditioning
            else None
        )
        if self.visual_memory_fusion is not None:
            linear = self.visual_memory_fusion[-1]
            if not isinstance(linear, nn.Linear):  # pragma: no cover - defensive
                raise RuntimeError("visual memory fusion output must be linear")
            nn.init.zeros_(linear.weight)
            nn.init.zeros_(linear.bias)
        self.persistent_memory_encoder = (
            PersistentProgressMemory(
                self.config.hidden_dim,
                slots=self.config.persistent_memory_slots,
                heads=self.config.persistent_memory_heads,
            )
            if self.config.persistent_progress_memory
            else None
        )
        self.persistent_state_action_encoder = (
            nn.Sequential(
                nn.LayerNorm(self.config.state_dim + self.config.action_dim),
                nn.Linear(
                    self.config.state_dim + self.config.action_dim,
                    self.config.hidden_dim,
                ),
                nn.SiLU(),
            )
            if self.config.persistent_progress_memory
            else None
        )
        self.temporal_conditioner = (
            TemporalHierarchicalConditioner(
                TemporalHierarchicalConfig(
                    state_dim=self.config.state_dim,
                    action_dim=self.config.action_dim,
                    hidden_dim=self.config.hidden_dim,
                    history_length=self.config.history_length,
                    phase_classes=self.config.phase_classes,
                )
            )
            if self.config.temporal_conditioning
            else None
        )
        self.unified_temporal_conditioners = (
            nn.ModuleList(
                TemporalHierarchicalConditioner(
                    TemporalHierarchicalConfig(
                        state_dim=self.config.state_dim,
                        action_dim=self.config.action_dim,
                        hidden_dim=self.config.hidden_dim,
                        history_length=self.config.history_length,
                        phase_classes=self.config.phase_classes,
                    )
                )
                for _ in range(self.config.unified_head_count)
            )
            if self.config.unified_head_bank and self.config.temporal_conditioning
            else None
        )
        self.temporal_state_fusion = (
            nn.Sequential(
                _make_norm(self.config.hidden_dim, self.config.norm_type),
                nn.Linear(self.config.hidden_dim, self.config.hidden_dim),
            )
            if self.config.temporal_conditioning
            and self.config.temporal_fusion == "state_residual"
            else None
        )
        if self.temporal_state_fusion is not None and self.config.temporal_zero_init:
            linear = self.temporal_state_fusion[-1]
            if not isinstance(linear, nn.Linear):  # pragma: no cover - defensive
                raise RuntimeError("temporal state fusion output must be linear")
            nn.init.zeros_(linear.weight)
            nn.init.zeros_(linear.bias)
        self.temporal_progress_gate = None
        if self.config.selective_progress_gate:
            self.temporal_progress_gate = nn.Sequential(
                nn.LayerNorm(3 * self.config.hidden_dim),
                nn.Linear(3 * self.config.hidden_dim, 1),
            )
            gate_linear = self.temporal_progress_gate[-1]
            if not isinstance(gate_linear, nn.Linear):  # pragma: no cover - defensive
                raise RuntimeError("progress gate output must be linear")
            nn.init.zeros_(gate_linear.weight)
            gate_linear.bias.data.fill_(
                math.log(self.config.temporal_gate_init)
                - math.log1p(-self.config.temporal_gate_init)
            )
        self.unified_temporal_state_fusion = (
            nn.ModuleList(
                nn.Sequential(
                    _make_norm(self.config.hidden_dim, self.config.norm_type),
                    nn.Linear(self.config.hidden_dim, self.config.hidden_dim),
                )
                for _ in range(self.config.unified_head_count)
            )
            if self.config.unified_head_bank
            and self.config.temporal_conditioning
            and self.config.temporal_fusion == "state_residual"
            else None
        )
        if self.unified_temporal_state_fusion is not None and self.config.temporal_zero_init:
            for fusion in self.unified_temporal_state_fusion:
                linear = fusion[-1]
                if not isinstance(linear, nn.Linear):  # pragma: no cover - defensive
                    raise RuntimeError("unified temporal fusion output must be linear")
                nn.init.zeros_(linear.weight)
                nn.init.zeros_(linear.bias)
        self.unified_temporal_progress_gate = (
            nn.ModuleList(
                nn.Sequential(
                    nn.LayerNorm(3 * self.config.hidden_dim),
                    nn.Linear(3 * self.config.hidden_dim, 1),
                )
                for _ in range(self.config.unified_head_count)
            )
            if self.config.unified_head_bank and self.config.selective_progress_gate
            else None
        )
        if self.unified_temporal_progress_gate is not None and self.config.temporal_zero_init:
            for gate in self.unified_temporal_progress_gate:
                linear = gate[-1]
                if not isinstance(linear, nn.Linear):  # pragma: no cover - defensive
                    raise RuntimeError("unified temporal gate output must be linear")
                nn.init.zeros_(linear.weight)
                nn.init.zeros_(linear.bias)
        self.subtask_progress_conditioner = (
            SubtaskProgressConditioner(
                self.config.hidden_dim,
                index_classes=self.config.progress_classes,
                cycle_classes=4,
            )
            if self.config.progress_conditioning
            else None
        )
        self.unified_subtask_progress_conditioners = (
            nn.ModuleList(
                SubtaskProgressConditioner(
                    self.config.hidden_dim,
                    index_classes=self.config.progress_classes,
                    cycle_classes=4,
                )
                for _ in range(self.config.unified_head_count)
            )
            if self.config.unified_head_bank and self.config.progress_conditioning
            else None
        )
        self.temporal_gate_logit = (
            nn.Parameter(
                torch.tensor(
                    math.log(self.config.temporal_gate_init)
                    - math.log1p(-self.config.temporal_gate_init),
                    dtype=torch.float32,
                )
            )
            if self.config.temporal_conditioning
            and self.config.temporal_fusion == "state_residual"
            else None
        )
        self.action_expert = FlowMatchingActionExpert(self.config)
        self.latent_reasoner = (
            LatentReasoningAdapter(self.config)
            if self.config.latent_reasoning
            else None
        )
        self.latent_state_fusion = (
            nn.Sequential(
                _make_norm(self.config.hidden_dim, self.config.norm_type),
                nn.Linear(self.config.hidden_dim, self.config.hidden_dim),
            )
            if self.config.latent_reasoning
            else None
        )
        if self.latent_state_fusion is not None:
            output = self.latent_state_fusion[-1]
            if not isinstance(output, nn.Linear):  # pragma: no cover - defensive
                raise RuntimeError("latent state fusion output must be linear")
            # The transferred parent policy is unchanged before LaST training.
            nn.init.zeros_(output.weight)
            nn.init.zeros_(output.bias)
        self.latent_residual_action_encoder = (
            nn.Sequential(
                nn.LayerNorm(self.config.action_horizon * self.config.action_dim),
                nn.Linear(
                    self.config.action_horizon * self.config.action_dim,
                    self.config.hidden_dim,
                ),
                nn.SiLU(),
            )
            if self.config.latent_residual_adapter
            else None
        )
        self.latent_residual_trunk = (
            nn.Sequential(
                _make_norm(4 * self.config.hidden_dim, self.config.norm_type),
                nn.Linear(4 * self.config.hidden_dim, self.config.hidden_dim),
                nn.SiLU(),
            )
            if self.config.latent_residual_adapter
            else None
        )
        self.latent_residual_output = (
            nn.Linear(
                self.config.hidden_dim,
                self.config.action_horizon * self.config.action_dim,
            )
            if self.config.latent_residual_adapter
            else None
        )
        self.latent_residual_gate = (
            nn.Linear(
                self.config.hidden_dim,
                self.config.action_horizon * self.config.action_dim,
            )
            if self.config.latent_residual_adapter
            else None
        )
        if self.config.latent_residual_adapter:
            if self.latent_residual_output is None or self.latent_residual_gate is None:
                raise RuntimeError("latent residual modules are incomplete")
            nn.init.zeros_(self.latent_residual_output.weight)
            nn.init.zeros_(self.latent_residual_output.bias)
            nn.init.zeros_(self.latent_residual_gate.weight)
            gate_bias = math.log(self.config.latent_residual_gate_init) - math.log1p(
                -self.config.latent_residual_gate_init
            )
            nn.init.constant_(self.latent_residual_gate.bias, gate_bias)
        self.unified_action_experts = (
            nn.ModuleList(
                FlowMatchingActionExpert(self.config)
                for _ in range(self.config.unified_head_count)
            )
            if self.config.unified_action_expert_bank
            else None
        )
        direct_input_dim = (3 if self.config.progressive_action_conditioning else 2) * self.config.hidden_dim
        self.residual_action_encoder = (
            nn.Sequential(
                nn.LayerNorm(self.config.action_horizon * self.config.action_dim),
                nn.Linear(
                    self.config.action_horizon * self.config.action_dim,
                    self.config.hidden_dim,
                ),
                nn.SiLU(),
            )
            if self.config.progressive_action_conditioning
            else None
        )
        self.direct_action_predictor = (
            nn.Sequential(
                nn.LayerNorm(direct_input_dim),
                nn.Linear(direct_input_dim, self.config.hidden_dim),
                nn.SiLU(),
                nn.Linear(
                    self.config.hidden_dim,
                    self.config.action_horizon * self.config.action_dim,
                ),
            )
            if self.config.direct_action_head
            else None
        )
        self.direct_action_attention = (
            nn.Linear(self.config.hidden_dim, self.config.hidden_dim, bias=False)
            if self.config.direct_action_attention_pool
            else None
        )
        if self.direct_action_attention is not None:
            nn.init.zeros_(self.direct_action_attention.weight)
        self.task_embedding = (
            nn.Embedding(self.config.task_classes, self.config.hidden_dim)
            if self.config.task_conditioning
            else None
        )
        self.task_state_fusion = (
            nn.Sequential(
                _make_norm(self.config.hidden_dim, self.config.norm_type),
                nn.Linear(self.config.hidden_dim, self.config.hidden_dim),
            )
            if self.config.task_conditioning
            else None
        )
        if self.task_state_fusion is not None:
            linear = self.task_state_fusion[-1]
            if not isinstance(linear, nn.Linear):  # pragma: no cover - defensive
                raise RuntimeError("task state fusion output must be linear")
            nn.init.zeros_(linear.weight)
            nn.init.zeros_(linear.bias)
        if self.config.task_conditioning:
            self.register_buffer(
                "task_prototypes",
                torch.zeros(self.config.task_classes, self.config.vlm_feature_dim),
                persistent=True,
            )
            self.register_buffer(
                "task_prototype_valid",
                torch.zeros(self.config.task_classes, dtype=torch.bool),
                persistent=True,
            )
        self.unified_direct_predictors = (
            nn.ModuleList(
                nn.Sequential(
                    nn.LayerNorm(direct_input_dim),
                    nn.Linear(direct_input_dim, self.config.hidden_dim),
                    nn.SiLU(),
                    nn.Linear(
                        self.config.hidden_dim,
                        self.config.action_horizon * self.config.action_dim,
                    ),
                )
                for _ in range(self.config.unified_head_count)
            )
            if self.config.unified_head_bank
            else None
        )
        self.unified_direct_attention = (
            nn.ModuleList(
                nn.Linear(self.config.hidden_dim, self.config.hidden_dim, bias=False)
                for _ in range(self.config.unified_head_count)
            )
            if self.config.unified_head_bank and self.config.direct_action_attention_pool
            else None
        )
        self.task_direct_action_residuals = (
            nn.ModuleList(
                nn.Sequential(
                    nn.LayerNorm(direct_input_dim),
                    nn.Linear(direct_input_dim, self.config.hidden_dim),
                    nn.SiLU(),
                    nn.Linear(
                        self.config.hidden_dim,
                        self.config.action_horizon * self.config.action_dim,
                    ),
                )
                for _ in range(self.config.task_classes)
            )
            if self.config.task_direct_residual_adapter
            else None
        )
        if self.task_direct_action_residuals is not None:
            for residual in self.task_direct_action_residuals:
                output = residual[-1]
                if not isinstance(output, nn.Linear):  # pragma: no cover - defensive
                    raise RuntimeError("task direct residual output must be linear")
                nn.init.zeros_(output.weight)
                nn.init.zeros_(output.bias)
        self.global_action_residual = (
            nn.Sequential(
                nn.LayerNorm(2 * self.config.hidden_dim + self.config.action_horizon * self.config.action_dim),
                nn.Linear(
                    2 * self.config.hidden_dim + self.config.action_horizon * self.config.action_dim,
                    self.config.hidden_dim,
                ),
                nn.SiLU(),
                nn.Linear(
                    self.config.hidden_dim,
                    self.config.action_horizon * self.config.action_dim,
                ),
            )
            if self.config.global_action_residual_adapter
            else None
        )
        if self.global_action_residual is not None:
            output = self.global_action_residual[-1]
            if not isinstance(output, nn.Linear):  # pragma: no cover - defensive
                raise RuntimeError("global residual output must be linear")
            # Zero initialization makes the V53 checkpoint exactly V51 before
            # any update, which is required for a fair regression gate.
            nn.init.zeros_(output.weight)
            nn.init.zeros_(output.bias)
        self.state_adaptive_action_encoder = (
            nn.Sequential(
                nn.LayerNorm(self.config.action_horizon * self.config.action_dim),
                nn.Linear(
                    self.config.action_horizon * self.config.action_dim,
                    self.config.hidden_dim,
                ),
                nn.SiLU(),
            )
            if self.config.state_adaptive_residual_adapter
            else None
        )
        self.state_adaptive_query = (
            nn.Sequential(
                nn.LayerNorm(2 * self.config.hidden_dim),
                nn.Linear(2 * self.config.hidden_dim, self.config.hidden_dim),
                nn.SiLU(),
            )
            if self.config.state_adaptive_residual_adapter
            else None
        )
        self.state_adaptive_attention = (
            nn.MultiheadAttention(
                self.config.hidden_dim,
                self.config.expert_heads,
                bias=False,
                batch_first=True,
            )
            if self.config.state_adaptive_residual_adapter
            else None
        )
        self.state_adaptive_trunk = (
            nn.Sequential(
                nn.LayerNorm(3 * self.config.hidden_dim),
                nn.Linear(3 * self.config.hidden_dim, self.config.hidden_dim),
                nn.SiLU(),
            )
            if self.config.state_adaptive_residual_adapter
            else None
        )
        arm_dimensions = self.config.action_horizon * (self.config.action_dim - 1)
        self.state_adaptive_arm_residual = (
            nn.Linear(self.config.hidden_dim, arm_dimensions)
            if self.config.state_adaptive_residual_adapter
            else None
        )
        self.state_adaptive_arm_gate = (
            nn.Linear(self.config.hidden_dim, arm_dimensions)
            if self.config.state_adaptive_residual_adapter
            else None
        )
        self.state_adaptive_gripper_logits = (
            nn.Linear(self.config.hidden_dim, self.config.action_horizon)
            if self.config.state_adaptive_residual_adapter
            else None
        )
        self.state_adaptive_gripper_gate = (
            nn.Linear(self.config.hidden_dim, self.config.action_horizon)
            if self.config.state_adaptive_residual_adapter
            else None
        )
        if self.config.state_adaptive_residual_adapter:
            residual_outputs = (
                self.state_adaptive_arm_residual,
                self.state_adaptive_gripper_logits,
            )
            gate_outputs = (
                self.state_adaptive_arm_gate,
                self.state_adaptive_gripper_gate,
            )
            if any(not isinstance(module, nn.Linear) for module in residual_outputs + gate_outputs):
                raise RuntimeError("state-adaptive residual modules are incomplete")
            for output in residual_outputs:
                nn.init.zeros_(output.weight)
                nn.init.zeros_(output.bias)
            gate_bias = math.log(self.config.state_adaptive_gate_init) - math.log1p(
                -self.config.state_adaptive_gate_init
            )
            for output in gate_outputs:
                nn.init.zeros_(output.weight)
                nn.init.constant_(output.bias, gate_bias)
        self.ordered_relative_action_encoder = (
            nn.Sequential(
                nn.LayerNorm(self.config.action_horizon * self.config.action_dim),
                nn.Linear(
                    self.config.action_horizon * self.config.action_dim,
                    self.config.hidden_dim,
                ),
                nn.SiLU(),
            )
            if self.config.ordered_relative_residual_adapter
            else None
        )
        ordered_query_dim = 3 * self.config.hidden_dim
        self.ordered_relative_source_query = (
            nn.Sequential(
                nn.LayerNorm(ordered_query_dim),
                nn.Linear(ordered_query_dim, self.config.hidden_dim),
                nn.SiLU(),
            )
            if self.config.ordered_relative_residual_adapter
            else None
        )
        self.ordered_relative_target_query = (
            nn.Sequential(
                nn.LayerNorm(ordered_query_dim),
                nn.Linear(ordered_query_dim, self.config.hidden_dim),
                nn.SiLU(),
            )
            if self.config.ordered_relative_residual_adapter
            else None
        )
        self.ordered_relative_attention = (
            nn.MultiheadAttention(
                self.config.hidden_dim,
                self.config.expert_heads,
                bias=False,
                batch_first=True,
            )
            if self.config.ordered_relative_residual_adapter
            else None
        )
        self.ordered_relative_trunk = (
            nn.Sequential(
                nn.LayerNorm(4 * self.config.hidden_dim),
                nn.Linear(4 * self.config.hidden_dim, self.config.hidden_dim),
                nn.SiLU(),
            )
            if self.config.ordered_relative_residual_adapter
            else None
        )
        self.ordered_relative_arm_output = (
            nn.Linear(
                self.config.hidden_dim,
                self.config.action_horizon * (self.config.action_dim - 1),
            )
            if self.config.ordered_relative_residual_adapter
            else None
        )
        if self.ordered_relative_arm_output is not None:
            # The untrained V73 policy is exactly V51; only the bounded arm
            # correction can become non-zero during sidecar training.
            nn.init.zeros_(self.ordered_relative_arm_output.weight)
            nn.init.zeros_(self.ordered_relative_arm_output.bias)
        self.spatial_grid_projection = (
            nn.Linear(self.config.spatial_grid_feature_dim, self.config.hidden_dim)
            if self.config.spatial_grid_residual_adapter
            else None
        )
        self.spatial_grid_camera_embedding = (
            nn.Parameter(torch.randn(self.config.camera_count, self.config.hidden_dim) * 0.02)
            if self.config.spatial_grid_residual_adapter
            else None
        )
        self.spatial_grid_action_encoder = (
            nn.Sequential(
                nn.LayerNorm(self.config.action_horizon * self.config.action_dim),
                nn.Linear(
                    self.config.action_horizon * self.config.action_dim,
                    self.config.hidden_dim,
                ),
                nn.SiLU(),
            )
            if self.config.spatial_grid_residual_adapter
            else None
        )
        spatial_query_dim = 3 * self.config.hidden_dim
        self.spatial_grid_source_query = (
            nn.Sequential(
                nn.LayerNorm(spatial_query_dim),
                nn.Linear(spatial_query_dim, self.config.hidden_dim),
                nn.SiLU(),
            )
            if self.config.spatial_grid_residual_adapter
            else None
        )
        self.spatial_grid_target_query = (
            nn.Sequential(
                nn.LayerNorm(spatial_query_dim),
                nn.Linear(spatial_query_dim, self.config.hidden_dim),
                nn.SiLU(),
            )
            if self.config.spatial_grid_residual_adapter
            else None
        )
        self.spatial_grid_attention = (
            nn.MultiheadAttention(
                self.config.hidden_dim,
                self.config.expert_heads,
                bias=False,
                batch_first=True,
            )
            if self.config.spatial_grid_residual_adapter
            else None
        )
        self.spatial_grid_trunk = (
            nn.Sequential(
                nn.LayerNorm(4 * self.config.hidden_dim),
                nn.Linear(4 * self.config.hidden_dim, self.config.hidden_dim),
                nn.SiLU(),
            )
            if self.config.spatial_grid_residual_adapter
            else None
        )
        self.spatial_grid_arm_output = (
            nn.Linear(
                self.config.hidden_dim,
                self.config.action_horizon * (self.config.action_dim - 1),
            )
            if self.config.spatial_grid_residual_adapter
            else None
        )
        if self.spatial_grid_arm_output is not None:
            # Zero-init is the parent-preservation invariant for V74.
            nn.init.zeros_(self.spatial_grid_arm_output.weight)
            nn.init.zeros_(self.spatial_grid_arm_output.bias)
        if self.config.progressive_residual:
            if self.direct_action_predictor is not None:
                output = self.direct_action_predictor[-1]
                if not isinstance(output, nn.Linear):  # pragma: no cover - defensive
                    raise RuntimeError("progressive direct output must be linear")
                nn.init.zeros_(output.weight)
                nn.init.zeros_(output.bias)
            if self.unified_direct_predictors is not None:
                for predictor in self.unified_direct_predictors:
                    output = predictor[-1]
                    if not isinstance(output, nn.Linear):  # pragma: no cover - defensive
                        raise RuntimeError("progressive unified output must be linear")
                    nn.init.zeros_(output.weight)
                    nn.init.zeros_(output.bias)
        if self.config.unified_head_bank:
            self.register_buffer(
                "unified_head_mix",
                torch.full((self.config.unified_head_count,), self.config.direct_action_inference_mix),
                persistent=True,
            )
            self.register_buffer(
                "task_head_map",
                torch.zeros(self.config.task_classes, dtype=torch.long),
                persistent=True,
            )

    def _infer_task_ids(self, features: Tensor) -> Tensor:
        """Infer a task id from language-conditioned Qwen feature prototypes."""

        if not self.config.task_conditioning:
            raise RuntimeError("task conditioning is not configured")
        summary = pool_task_routing_features(
            features,
            pooling=self.config.task_router_pooling,
            semantic_tokens=(
                self.config.task_router_tail_tokens
                if self.config.task_router_pooling == "tail_tokens"
                else self.config.qwen_semantic_tokens
            ),
            grounding_tokens=self.config.qwen_grounding_tokens,
        )
        if summary.shape[-1] != self.config.vlm_feature_dim:
            raise ValueError("task prototype feature dimension does not match features")
        # Prototype similarities are extremely close for related LIBERO
        # language instructions; always compare in fp32 so bf16/fp16 rounding
        # cannot send a sample to a different internal head.
        summary = summary.float()
        prototypes = self.task_prototypes.to(device=summary.device, dtype=torch.float32)
        valid = self.task_prototype_valid.to(device=summary.device)
        summary = torch.nn.functional.normalize(summary, dim=-1)
        prototypes = torch.nn.functional.normalize(prototypes, dim=-1)
        scores = summary @ prototypes.transpose(0, 1)
        if not bool(valid.all()):
            scores = scores.masked_fill(~valid[None, :], torch.finfo(scores.dtype).min)
        return scores.argmax(dim=-1)

    def _direct_action_pooled(
        self,
        context: Tensor,
        state_token: Tensor,
        task_ids: Tensor | None,
    ) -> Tensor:
        """Build the pooled visual/state input shared by direct action heads."""

        bank = self.unified_direct_predictors
        if bank is not None:
            if task_ids is None or self.unified_direct_attention is None:
                raise RuntimeError("unified head bank requires inferred task ids")
            head_ids = self.task_head_map.index_select(0, task_ids.long())
            pooled = torch.empty(
                context.shape[0], 2 * self.config.hidden_dim,
                device=context.device, dtype=context.dtype,
            )
            for head_index, attention in enumerate(self.unified_direct_attention):
                mask = head_ids == head_index
                if not bool(mask.any()):
                    continue
                selected_context = context[mask]
                selected_state = state_token[mask, 0]
                query = attention(selected_state)
                scores = torch.einsum("bth,bh->bt", selected_context, query)
                weights = torch.softmax(scores / math.sqrt(context.shape[-1]), dim=1)
                summary = torch.einsum("bt,bth->bh", weights, selected_context)
                pooled[mask] = torch.cat((summary, selected_state), dim=-1)
            return pooled
        if self.direct_action_attention is None:
            visual_summary = context.mean(dim=1)
        else:
            query = self.direct_action_attention(state_token[:, 0])
            scores = torch.einsum("bth,bh->bt", context, query)
            weights = torch.softmax(scores / math.sqrt(context.shape[-1]), dim=1)
            visual_summary = torch.einsum("bt,bth->bh", weights, context)
        return torch.cat((visual_summary, state_token[:, 0]), dim=-1)

    def _direct_action_prediction(
        self,
        context: Tensor,
        state_token: Tensor,
        task_ids: Tensor | None = None,
        base_actions: Tensor | None = None,
    ) -> Tensor:
        direct, task = self._direct_action_residual_components(
            context, state_token, task_ids, base_actions=base_actions
        )
        return direct + task

    def _global_action_residual_prediction(
        self,
        context: Tensor,
        state_token: Tensor,
        base_actions: Tensor,
    ) -> Tensor:
        """Predict one shared residual for all tasks from the current base action."""

        if self.global_action_residual is None:
            raise RuntimeError("global action residual adapter is not configured")
        expected = (context.shape[0], self.config.action_horizon, self.config.action_dim)
        if tuple(base_actions.shape) != expected:
            raise ValueError(f"base_actions must have shape {expected}")
        pooled = context.mean(dim=1)
        inputs = torch.cat(
            (pooled, state_token[:, 0], base_actions.reshape(base_actions.shape[0], -1)),
            dim=-1,
        )
        return self.global_action_residual(inputs).reshape(expected)

    def _latent_reasoning_summary(
        self, conditioning_outputs: dict[str, Tensor]
    ) -> Tensor:
        """Pool latent CoT slots with a differentiable adaptive halt rule.

        LaST-R1 learns a stop token in the RL loop.  During offline warm-up we
        use its probability as a soft halting distribution; this keeps the
        interface differentiable and makes the eventual LAPO rollout buffer
        contract explicit without claiming that offline imitation is RL.
        """

        latent = conditioning_outputs.get("latent_tokens")
        end_logits = conditioning_outputs.get("latent_end_logits")
        if latent is None or end_logits is None:
            raise RuntimeError("latent reasoning outputs are missing")
        if latent.ndim != 3 or end_logits.shape != latent.shape[:2]:
            raise ValueError("latent reasoning tensors have incompatible shapes")
        if not self.config.latent_adaptive_horizon:
            return latent.mean(dim=1)
        halt = end_logits.float().sigmoid().clamp(1e-4, 1.0 - 1e-4)
        survival = torch.cat(
            (torch.ones_like(halt[:, :1]), 1.0 - halt[:, :-1]), dim=1
        ).cumprod(dim=1)
        weights = survival * halt
        # If all stop probabilities are small, retain a stable non-zero
        # summary rather than silently dropping the residual branch.
        weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-4)
        return (latent * weights.to(dtype=latent.dtype).unsqueeze(-1)).sum(dim=1)

    def _latent_residual_prediction(
        self,
        context: Tensor,
        state_token: Tensor,
        conditioning_outputs: dict[str, Tensor],
        base_actions: Tensor,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        """Return LaST's latent-to-action residual and diagnostics."""

        modules = (
            self.latent_residual_action_encoder,
            self.latent_residual_trunk,
            self.latent_residual_output,
            self.latent_residual_gate,
        )
        if any(module is None for module in modules):
            raise RuntimeError("latent residual adapter is not configured")
        expected = (context.shape[0], self.config.action_horizon, self.config.action_dim)
        if tuple(base_actions.shape) != expected:
            raise ValueError(f"base_actions must have shape {expected}")
        latent_summary = self._latent_reasoning_summary(conditioning_outputs)
        action_token = self.latent_residual_action_encoder(
            base_actions.reshape(base_actions.shape[0], -1).to(context.dtype)
        )
        pooled = context.mean(dim=1)
        hidden = self.latent_residual_trunk(
            torch.cat((pooled, state_token[:, 0], latent_summary, action_token), dim=-1)
        )
        raw = self.latent_residual_output(hidden).reshape(expected)
        gate = self.latent_residual_gate(hidden).reshape(expected).sigmoid()
        correction = self.config.latent_residual_scale * gate * torch.tanh(raw)
        return correction, {
            "latent_summary": latent_summary,
            "latent_residual_raw": raw,
            "latent_residual_gate": gate,
            "latent_residual_correction": correction,
            "latent_halt_probability": conditioning_outputs["latent_end_logits"].sigmoid(),
        }

    def _state_adaptive_residual_components(
        self,
        context: Tensor,
        state_token: Tensor,
        base_actions: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Return shared arm/gripper corrections and their observation gates."""

        modules = (
            self.state_adaptive_action_encoder,
            self.state_adaptive_query,
            self.state_adaptive_attention,
            self.state_adaptive_trunk,
            self.state_adaptive_arm_residual,
            self.state_adaptive_arm_gate,
            self.state_adaptive_gripper_logits,
            self.state_adaptive_gripper_gate,
        )
        if any(module is None for module in modules):
            raise RuntimeError("state-adaptive residual adapter is not configured")
        expected = (context.shape[0], self.config.action_horizon, self.config.action_dim)
        if tuple(base_actions.shape) != expected:
            raise ValueError(f"base_actions must have shape {expected}")
        action_token = self.state_adaptive_action_encoder(
            base_actions.reshape(base_actions.shape[0], -1).to(context.dtype)
        )
        query = self.state_adaptive_query(
            torch.cat((state_token[:, 0], action_token), dim=-1)
        )
        spatial, _ = self.state_adaptive_attention(
            query[:, None, :], context, context, need_weights=False
        )
        hidden = self.state_adaptive_trunk(
            torch.cat((spatial[:, 0], state_token[:, 0], action_token), dim=-1)
        )
        arm_shape = (
            base_actions.shape[0],
            self.config.action_horizon,
            self.config.action_dim - 1,
        )
        arm_residual = self.state_adaptive_arm_residual(hidden).reshape(arm_shape)
        arm_gate = self.state_adaptive_arm_gate(hidden).reshape(arm_shape).sigmoid()
        gripper_logits = self.state_adaptive_gripper_logits(hidden)
        gripper_gate = self.state_adaptive_gripper_gate(hidden).sigmoid()
        return arm_residual, arm_gate, gripper_logits, gripper_gate

    def _apply_state_adaptive_residual(
        self,
        context: Tensor,
        state_token: Tensor,
        base_actions: Tensor,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        """Apply V54 while exposing components needed by its training losses."""

        arm_residual, arm_gate, gripper_logits, gripper_gate = (
            self._state_adaptive_residual_components(context, state_token, base_actions)
        )
        scale = self.config.state_adaptive_residual_scale
        corrected = base_actions.clone()
        arm_correction = scale * arm_gate * arm_residual
        gripper_correction = scale * gripper_gate * torch.tanh(gripper_logits)
        corrected[..., :-1] = corrected[..., :-1] + arm_correction
        corrected[..., -1] = corrected[..., -1] + gripper_correction
        return corrected, {
            "arm_residual": arm_residual,
            "arm_gate": arm_gate,
            "arm_correction": arm_correction,
            "gripper_logits": gripper_logits,
            "gripper_gate": gripper_gate,
            "gripper_correction": gripper_correction,
        }

    def _ordered_relative_position(self, context: Tensor) -> Tensor:
        """Return fixed 1-D token-order features for every camera.

        The cache does not retain a 2-D patch grid.  This encoding therefore
        exposes only the order that is actually present in the V40 contract.
        """

        expected_tokens = self.config.camera_count * self.config.tokens_per_camera
        if context.ndim != 3 or context.shape[1] != expected_tokens:
            raise ValueError(
                "ordered relative context must have camera_count * tokens_per_camera tokens"
            )
        half = self.config.hidden_dim // 2
        positions = torch.arange(
            self.config.tokens_per_camera, device=context.device, dtype=torch.float32
        )
        frequencies = torch.exp(
            -math.log(10_000.0)
            * torch.arange(half, device=context.device, dtype=torch.float32)
            / max(half - 1, 1)
        )
        angles = positions[:, None] * frequencies[None, :]
        encoding = torch.cat((angles.sin(), angles.cos()), dim=-1)
        if encoding.shape[-1] < self.config.hidden_dim:
            encoding = nn.functional.pad(
                encoding, (0, self.config.hidden_dim - encoding.shape[-1])
            )
        encoding = encoding[None, :, :].expand(self.config.camera_count, -1, -1)
        return encoding.reshape(expected_tokens, self.config.hidden_dim).to(context.dtype)

    def _ordered_relative_residual_components(
        self,
        context: Tensor,
        state_token: Tensor,
        base_actions: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Predict a bounded arm residual from continuous ordered context."""

        modules = (
            self.ordered_relative_action_encoder,
            self.ordered_relative_source_query,
            self.ordered_relative_target_query,
            self.ordered_relative_attention,
            self.ordered_relative_trunk,
            self.ordered_relative_arm_output,
        )
        if any(module is None for module in modules):
            raise RuntimeError("ordered relative residual adapter is not configured")
        expected = (context.shape[0], self.config.action_horizon, self.config.action_dim)
        if tuple(base_actions.shape) != expected:
            raise ValueError(f"base_actions must have shape {expected}")
        camera_context = context.reshape(
            context.shape[0],
            self.config.camera_count,
            self.config.tokens_per_camera,
            self.config.hidden_dim,
        )
        semantic = camera_context[
            :, :, -self.config.ordered_relative_semantic_tokens :, :
        ].mean(dim=(1, 2))
        action_token = self.ordered_relative_action_encoder(
            base_actions.reshape(base_actions.shape[0], -1).to(context.dtype)
        )
        query_input = torch.cat((semantic, state_token[:, 0], action_token), dim=-1)
        source_query = self.ordered_relative_source_query(query_input)
        target_query = self.ordered_relative_target_query(query_input)
        queries = torch.stack((source_query, target_query), dim=1)
        ordered_context = context + self._ordered_relative_position(context)[None, :, :]
        attended, _ = self.ordered_relative_attention(
            queries, ordered_context, ordered_context, need_weights=False
        )
        hidden = self.ordered_relative_trunk(
            torch.cat(
                (attended[:, 0], attended[:, 1], state_token[:, 0], action_token), dim=-1
            )
        )
        shape = (
            base_actions.shape[0],
            self.config.action_horizon,
            self.config.action_dim - 1,
        )
        raw = self.ordered_relative_arm_output(hidden).reshape(shape)
        correction = self.config.ordered_relative_residual_scale * torch.tanh(raw)
        return raw, correction

    def _apply_ordered_relative_residual(
        self,
        context: Tensor,
        state_token: Tensor,
        base_actions: Tensor,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        """Apply the V73 arm-only sidecar while preserving V51 gripper."""

        raw, correction = self._ordered_relative_residual_components(
            context, state_token, base_actions
        )
        corrected = base_actions.clone()
        corrected[..., :-1] = corrected[..., :-1] + correction
        return corrected, {"raw_arm_residual": raw, "arm_correction": correction}

    def _spatial_grid_position(self, *, device: torch.device, dtype: torch.dtype) -> Tensor:
        """Return fixed 2-D sinusoidal positions for the 8x8 patch grid."""

        tokens = self.config.tokens_per_camera
        side = math.isqrt(tokens)
        if side * side != tokens:
            raise ValueError("spatial grid token count must be a square")
        half = max(self.config.hidden_dim // 4, 1)
        positions = torch.arange(side, device=device, dtype=torch.float32)
        frequencies = torch.exp(
            -math.log(10_000.0)
            * torch.arange(half, device=device, dtype=torch.float32)
            / max(half - 1, 1)
        )
        angles = positions[:, None] * frequencies[None, :]
        one_axis = torch.cat((angles.sin(), angles.cos()), dim=-1)
        y = one_axis[:, None, :].expand(side, side, -1)
        x = one_axis[None, :, :].expand(side, side, -1)
        encoding = torch.cat((x, y), dim=-1).reshape(tokens, -1)
        if encoding.shape[-1] < self.config.hidden_dim:
            encoding = nn.functional.pad(encoding, (0, self.config.hidden_dim - encoding.shape[-1]))
        return encoding[:, : self.config.hidden_dim].to(dtype=dtype)

    def _spatial_grid_residual_components(
        self,
        spatial_features: Tensor,
        context: Tensor,
        state_token: Tensor,
        base_actions: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Predict a bounded arm residual from exact per-camera patch tokens."""

        modules = (
            self.spatial_grid_projection,
            self.spatial_grid_camera_embedding,
            self.spatial_grid_action_encoder,
            self.spatial_grid_source_query,
            self.spatial_grid_target_query,
            self.spatial_grid_attention,
            self.spatial_grid_trunk,
            self.spatial_grid_arm_output,
        )
        if any(module is None for module in modules):
            raise RuntimeError("spatial grid residual adapter is not configured")
        expected_actions = (self.config.action_horizon, self.config.action_dim)
        if tuple(base_actions.shape[1:]) != expected_actions:
            raise ValueError(f"base_actions must have shape [batch, {expected_actions[0]}, {expected_actions[1]}]")
        expected_features = (
            spatial_features.shape[0],
            self.config.camera_count,
            self.config.tokens_per_camera,
            self.config.spatial_grid_feature_dim,
        )
        if tuple(spatial_features.shape) != expected_features:
            raise ValueError(f"spatial_features must have shape {expected_features}")
        expected_context_tokens = self.config.camera_count * self.config.tokens_per_camera
        if tuple(context.shape) != (
            spatial_features.shape[0],
            expected_context_tokens,
            self.config.hidden_dim,
        ):
            raise ValueError("context must be the projected V51 per-camera context")
        projected = self.spatial_grid_projection(
            spatial_features.to(dtype=self.vlm_projection.weight.dtype)
        )
        projected = projected + self._spatial_grid_position(
            device=projected.device, dtype=projected.dtype
        )[None, None, :, :]
        projected = projected + self.spatial_grid_camera_embedding[None, :, None, :]
        grid_context = projected.reshape(projected.shape[0], expected_context_tokens, -1)
        semantic_context = context.reshape(
            context.shape[0], self.config.camera_count, self.config.tokens_per_camera, -1
        )[:, :, -self.config.spatial_grid_semantic_tokens :, :].mean(dim=(1, 2))
        action_token = self.spatial_grid_action_encoder(
            base_actions.reshape(base_actions.shape[0], -1).to(context.dtype)
        )
        query_input = torch.cat((semantic_context, state_token[:, 0], action_token), dim=-1)
        source_query = self.spatial_grid_source_query(query_input)
        target_query = self.spatial_grid_target_query(query_input)
        queries = torch.stack((source_query, target_query), dim=1)
        attended, _ = self.spatial_grid_attention(
            queries, grid_context, grid_context, need_weights=False
        )
        hidden = self.spatial_grid_trunk(
            torch.cat((attended[:, 0], attended[:, 1], state_token[:, 0], action_token), dim=-1)
        )
        shape = (
            base_actions.shape[0],
            self.config.action_horizon,
            self.config.action_dim - 1,
        )
        raw = self.spatial_grid_arm_output(hidden).reshape(shape)
        correction = self.config.spatial_grid_residual_scale * torch.tanh(raw)
        return raw, correction

    def _apply_spatial_grid_residual(
        self,
        spatial_features: Tensor,
        context: Tensor,
        state_token: Tensor,
        base_actions: Tensor,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        """Apply V74 while preserving the V51 gripper dimension exactly."""

        raw, correction = self._spatial_grid_residual_components(
            spatial_features, context, state_token, base_actions
        )
        corrected = base_actions.clone()
        corrected[..., :-1] = corrected[..., :-1] + correction
        return corrected, {"raw_arm_residual": raw, "arm_correction": correction}

    def _direct_action_residual_components(
        self,
        context: Tensor,
        state_token: Tensor,
        task_ids: Tensor | None = None,
        *,
        base_actions: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Return direct and task corrections separately.

        V52 conditions both residual heads on the sampled Flow action.  This
        keeps a stochastic Flow prior and makes the supervised residual target
        well-defined for every sampled action chunk.
        """
        if self.direct_action_predictor is None:
            raise RuntimeError("direct action head is not configured")
        if self.config.progressive_action_conditioning:
            if base_actions is None:
                raise ValueError("progressive residual heads require base_actions")
            expected = (context.shape[0], self.config.action_horizon, self.config.action_dim)
            if tuple(base_actions.shape) != expected:
                raise ValueError(f"base_actions must have shape {expected}")
            if self.residual_action_encoder is None:  # pragma: no cover - defensive
                raise RuntimeError("progressive action encoder is missing")
        bank = self.unified_direct_predictors
        pooled = self._direct_action_pooled(context, state_token, task_ids)
        if self.config.progressive_action_conditioning:
            action_token = self.residual_action_encoder(
                base_actions.reshape(base_actions.shape[0], -1).to(pooled.dtype)
            )
            pooled = torch.cat((pooled, action_token), dim=-1)
        if bank is not None:
            if task_ids is None or self.unified_direct_attention is None:
                raise RuntimeError("unified head bank requires inferred task ids")
            head_ids = self.task_head_map.index_select(0, task_ids.long())
            result = torch.empty(
                context.shape[0], self.config.action_horizon, self.config.action_dim,
                device=context.device, dtype=context.dtype,
            )
            for head_index, predictor in enumerate(bank):
                mask = head_ids == head_index
                if not bool(mask.any()):
                    continue
                values = predictor(pooled[mask]).reshape(
                    int(mask.sum()), self.config.action_horizon, self.config.action_dim
                )
                result[mask] = values
        else:
            result = self.direct_action_predictor(pooled).reshape(
                pooled.shape[0], self.config.action_horizon, self.config.action_dim
            )
        task_result = torch.zeros_like(result)
        if self.task_direct_action_residuals is not None:
            if task_ids is None:
                raise RuntimeError("task direct residuals require inferred task ids")
            if self.config.task_direct_residual_task_scales is None:
                task_scales = torch.full(
                    (task_ids.shape[0],),
                    self.config.task_direct_residual_scale,
                    device=task_ids.device,
                    dtype=result.dtype,
                )
            else:
                configured_scales = torch.as_tensor(
                    self.config.task_direct_residual_task_scales,
                    device=task_ids.device,
                    dtype=result.dtype,
                )
                task_scales = configured_scales.index_select(0, task_ids.long())
            for task_index, residual in enumerate(self.task_direct_action_residuals):
                mask = task_ids.long() == task_index
                if not bool(mask.any()):
                    continue
                values = residual(pooled[mask]).reshape(
                    int(mask.sum()), self.config.action_horizon, self.config.action_dim
                )
                task_result[mask] = task_scales[mask, None, None] * values
        return result, task_result

    def _prepare(self, features: Tensor, state: Tensor) -> tuple[Tensor, Tensor]:
        context: Tensor | None = None
        if self.visual_feature_adapter is not None:
            features = features + self.visual_feature_adapter(
                features.to(self.vlm_projection.weight.dtype)
            )
        if features.ndim == 2:
            features = features[:, None, :]
        elif features.ndim == 4:
            if features.shape[-1] != self.config.vlm_feature_dim:
                raise ValueError("features must end with vlm_feature_dim")
            if self.config.context_layout == "per_camera":
                expected = (self.config.camera_count, self.config.tokens_per_camera)
                if tuple(features.shape[1:3]) != expected:
                    raise ValueError(
                        f"per-camera features must have shape [batch, {expected[0]}, {expected[1]}, feature_dim]"
                    )
                features = features.to(self.vlm_projection.weight.dtype)
                context = self.vlm_projection(self.feature_norm(features))
                if self.camera_position is None:
                    raise RuntimeError("per-camera context is missing camera position embeddings")
                context = context + self.camera_position[None, :, None, :]
                context = context.reshape(context.shape[0], -1, context.shape[-1])
            else:
                features = features.reshape(features.shape[0], -1, features.shape[-1])
        if context is None:
            if self.config.context_layout == "per_camera":
                raise ValueError("per-camera context requires rank-4 camera-separated features")
            if features.ndim != 3 or features.shape[-1] != self.config.vlm_feature_dim:
                raise ValueError("features must have shape [batch, tokens, vlm_feature_dim]")
            features = features.to(self.vlm_projection.weight.dtype)
            context = self.vlm_projection(self.feature_norm(features))
        if context.shape[1] > self.config.max_context_tokens:
            raise ValueError("features exceed max_context_tokens")
        if state.ndim == 3:
            state = state[:, -1]
        if state.ndim != 2 or state.shape[-1] != self.config.state_dim:
            raise ValueError("state must have shape [batch, state_dim]")
        state_value = state.to(self.vlm_projection.weight.dtype)
        state_token = self.state_projection(state_value)[:, None, :]
        if self.state_feature_adapter is not None:
            state_token = state_token + self.state_feature_adapter(state_value)[:, None, :]
        if self.context_fusion:
            tokens = torch.cat((context, state_token), dim=1)
            if self.context_fusion_position is None:  # defensive type narrowing
                raise RuntimeError("context fusion position is missing")
            tokens = tokens + self.context_fusion_position[: tokens.shape[1]][None, :, :]
            for block in self.context_fusion:
                tokens = block(tokens)
            context, state_token = tokens[:, :-1], tokens[:, -1:]
        return context, state_token

    def _prepare_visual_memory(self, visual_history: Tensor) -> Tensor:
        """Project and compress previous per-camera Qwen features into memory tokens."""

        if not self.config.visual_memory_conditioning:
            raise ValueError("visual history requires visual_memory_conditioning=True")
        expected_prefix = (
            visual_history.shape[0],
            self.config.visual_memory_length,
            self.config.camera_count,
            self.config.tokens_per_camera,
        )
        if visual_history.ndim != 5 or tuple(visual_history.shape[:4]) != expected_prefix:
            raise ValueError(
                "visual_history must have shape "
                f"[batch, {self.config.visual_memory_length}, {self.config.camera_count}, "
                f"{self.config.tokens_per_camera}, feature_dim]"
            )
        if visual_history.shape[-1] != self.config.vlm_feature_dim:
            raise ValueError("visual_history must end with vlm_feature_dim")
        features = visual_history.to(self.vlm_projection.weight.dtype)
        projected = self.vlm_projection(self.feature_norm(features))
        group = self.config.tokens_per_camera // self.config.visual_memory_tokens_per_camera
        projected = projected.reshape(
            visual_history.shape[0],
            self.config.visual_memory_length,
            self.config.camera_count,
            self.config.visual_memory_tokens_per_camera,
            group,
            self.config.hidden_dim,
        ).mean(dim=4)
        if self.camera_position is not None:
            projected = projected + self.camera_position[None, None, :, None, :]
        memory = projected.reshape(visual_history.shape[0], -1, self.config.hidden_dim)
        if self.visual_memory_position is not None:
            memory = memory + self.visual_memory_position[None, :, :]
        return memory

    def _prepare_persistent_visual_memory(
        self,
        visual_history: Tensor,
        current_context: Tensor,
        history_states: Tensor | None = None,
        previous_actions: Tensor | None = None,
    ) -> Tensor:
        """Compress historical visual features and the current context."""

        if self.persistent_memory_encoder is None:
            raise RuntimeError("persistent progress memory encoder is missing")
        if visual_history.ndim != 5 or visual_history.shape[-1] != self.config.vlm_feature_dim:
            raise ValueError(
                "persistent visual history must have shape "
                "[batch, history, camera, tokens, feature_dim]"
            )
        projected = self.vlm_projection(
            self.feature_norm(visual_history.to(dtype=current_context.dtype))
        )
        historical = projected.mean(dim=(2, 3))
        current = current_context.mean(dim=1, keepdim=False)[:, None, :]
        sequence_parts = [historical, current]
        if history_states is not None or previous_actions is not None:
            if (
                history_states is None
                or previous_actions is None
                or history_states.ndim != 3
                or previous_actions.ndim != 3
                or history_states.shape[:2] != previous_actions.shape[:2]
                or history_states.shape[-1] != self.config.state_dim
                or previous_actions.shape[-1] != self.config.action_dim
            ):
                raise ValueError(
                    "persistent state/action history must have aligned rank-3 shapes"
                )
            if self.persistent_state_action_encoder is None:  # pragma: no cover
                raise RuntimeError("persistent state/action encoder is missing")
            state_action = self.persistent_state_action_encoder(
                torch.cat((history_states, previous_actions), dim=-1).to(current_context.dtype)
            ).mean(dim=1, keepdim=True)
            sequence_parts.append(state_action)
        sequence = torch.cat(sequence_parts, dim=1)
        return self.persistent_memory_encoder(sequence)

    def _prepare_action_inputs(
        self,
        features: Tensor,
        state: Tensor,
        *,
        history_states: Tensor | None = None,
        previous_actions: Tensor | None = None,
        visual_history: Tensor | None = None,
        phase_labels: Tensor | None = None,
        subtask_index: Tensor | None = None,
        subtask_progress: Tensor | None = None,
        cycle_state: Tensor | None = None,
        task_index: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, dict[str, Tensor]]:
        task_ids: Tensor | None = None
        if self.config.task_conditioning:
            task_ids = self._infer_task_ids(features) if task_index is None else task_index.long()
        context, state_token = self._prepare(features, state)
        conditioning_outputs: dict[str, Tensor] = {}
        if task_ids is not None:
            if self.task_embedding is None or self.task_state_fusion is None:
                raise RuntimeError("task conditioning modules are missing")
            task_embedding = self.task_embedding(task_ids)
            task_residual = self.task_state_fusion(task_embedding)
            state_token = state_token + self.config.task_conditioning_scale * task_residual[:, None, :]
            conditioning_outputs["task_ids"] = task_ids
        if self.config.visual_memory_conditioning:
            if visual_history is None:
                raise ValueError("V10 visual memory requires visual_history")
            if self.config.persistent_progress_memory:
                memory = self._prepare_persistent_visual_memory(
                    visual_history.to(device=context.device),
                    context,
                    history_states=history_states,
                    previous_actions=previous_actions,
                )
                memory_summary = None
            else:
                memory = self._prepare_visual_memory(
                    visual_history.to(device=context.device)
                )
                memory_summary = memory.mean(dim=1)
            if self.config.gated_visual_memory:
                conditioning_outputs["visual_memory_tokens"] = memory
            else:
                context = torch.cat((context, memory), dim=1)
            if self.visual_memory_fusion is not None and memory_summary is not None:
                state_token = state_token + self.visual_memory_fusion(memory_summary)[:, None, :]
                conditioning_outputs["visual_memory_summary"] = memory_summary
        elif visual_history is not None:
            raise ValueError("visual_history requires visual_memory_conditioning=True")
        if self.temporal_conditioner is None:
            if any(
                value is not None
                for value in (
                    history_states,
                    previous_actions,
                    phase_labels,
                    subtask_index,
                    subtask_progress,
                    cycle_state,
                )
            ):
                raise ValueError("temporal inputs require temporal_conditioning=True")
            return self._apply_latent_reasoning(
                context, state_token, conditioning_outputs
            )
        if history_states is None or previous_actions is None:
            raise ValueError(
                "V8 temporal conditioning requires history_states and previous_actions"
            )
        history_states = history_states.to(dtype=context.dtype, device=context.device)
        previous_actions = previous_actions.to(dtype=context.dtype, device=context.device)
        phase_labels_device = None if phase_labels is None else phase_labels.to(device=context.device)
        if self.unified_temporal_conditioners is None or task_ids is None:
            temporal_tokens, phase_logits = self.temporal_conditioner(
                history_states,
                previous_actions,
                phase_labels_device,
            )
        else:
            temporal_tokens = torch.empty(
                history_states.shape[0], 2, self.config.hidden_dim,
                device=context.device, dtype=context.dtype,
            )
            phase_logits = torch.empty(
                history_states.shape[0], self.config.phase_classes,
                device=context.device, dtype=context.dtype,
            )
            head_ids = self.task_head_map.index_select(0, task_ids.long())
            for head_index, conditioner in enumerate(self.unified_temporal_conditioners):
                mask = head_ids == head_index
                if not bool(mask.any()):
                    continue
                selected_labels = None if phase_labels_device is None else phase_labels_device[mask]
                values, logits = conditioner(history_states[mask], previous_actions[mask], selected_labels)
                temporal_tokens[mask] = values
                phase_logits[mask] = logits
        conditioning_outputs["phase_logits"] = phase_logits
        if self.config.temporal_fusion == "append":
            context = torch.cat((context, temporal_tokens), dim=1)
            if context.shape[1] > self.config.max_context_tokens + self.config.extra_context_tokens:
                raise ValueError("visual and temporal context exceed configured token budget")
        else:
            if self.temporal_state_fusion is None or self.temporal_gate_logit is None:
                raise RuntimeError("state-residual temporal fusion modules are missing")
            temporal_summary = (
                temporal_tokens[:, :1]
                if self.config.history_only_conditioning
                else temporal_tokens.mean(dim=1, keepdim=True)
            )
            temporal_summary = temporal_summary[:, 0]
            progress_conditioner = self.subtask_progress_conditioner
            if self.unified_subtask_progress_conditioners is not None and task_ids is not None:
                head_ids = self.task_head_map.index_select(0, task_ids.long())
                # A batch normally contains one task, but selecting per head
                # keeps the unified policy well-defined for mixed-task batches.
                if bool((head_ids == head_ids[0]).all()):
                    progress_conditioner = self.unified_subtask_progress_conditioners[int(head_ids[0])]
                else:
                    progress_conditioner = None
            if progress_conditioner is not None:
                pooled = context.mean(dim=1)
                progress_embedding, index_logits, progress_prediction, cycle_logits = (
                    progress_conditioner(
                        pooled,
                        state_token[:, 0],
                        temporal_summary,
                        subtask_index=None
                        if subtask_index is None
                        else subtask_index.to(device=context.device),
                        subtask_progress=None
                        if subtask_progress is None
                        else subtask_progress.to(device=context.device),
                        cycle_state=None
                        if cycle_state is None
                        else cycle_state.to(device=context.device),
                    )
                )
                temporal_summary = temporal_summary + progress_embedding
                conditioning_outputs.update(
                    {
                        "subtask_index_logits": index_logits,
                        "subtask_progress_prediction": progress_prediction,
                        "cycle_state_logits": cycle_logits,
                    }
                )
            pooled = context.mean(dim=1)
            gate_input = torch.cat((pooled, state_token[:, 0], temporal_summary), dim=-1)
            if self.unified_temporal_state_fusion is None or task_ids is None:
                residual = self.temporal_state_fusion(temporal_summary)[:, None, :]
            else:
                residual_flat = torch.empty_like(temporal_summary)
                head_ids = self.task_head_map.index_select(0, task_ids.long())
                for head_index, fusion in enumerate(self.unified_temporal_state_fusion):
                    mask = head_ids == head_index
                    if bool(mask.any()):
                        residual_flat[mask] = fusion(temporal_summary[mask])
                residual = residual_flat[:, None, :]
            if self.unified_temporal_progress_gate is not None and task_ids is not None:
                gate_flat = torch.empty(
                    temporal_summary.shape[0], 1, device=context.device, dtype=context.dtype
                )
                head_ids = self.task_head_map.index_select(0, task_ids.long())
                for head_index, gate_module in enumerate(self.unified_temporal_progress_gate):
                    mask = head_ids == head_index
                    if bool(mask.any()):
                        gate_flat[mask] = gate_module(gate_input[mask]).sigmoid()
                gate = gate_flat[:, None, :]
            elif self.temporal_progress_gate is not None:
                gate = self.temporal_progress_gate(gate_input).sigmoid()[:, None, :]
            else:
                if self.temporal_gate_logit is None:
                    raise RuntimeError("temporal gate is missing")
                gate = self.temporal_gate_logit.sigmoid().to(dtype=residual.dtype)
                gate = gate.reshape(1, 1, 1)
            state_token = state_token + gate * residual
        return self._apply_latent_reasoning(
            context, state_token, conditioning_outputs
        )

    def _apply_latent_reasoning(
        self,
        context: Tensor,
        state_token: Tensor,
        conditioning_outputs: dict[str, Tensor],
    ) -> tuple[Tensor, Tensor, dict[str, Tensor]]:
        """Compute latent CoT and inject it through a parent-preserving gate."""

        if self.latent_reasoner is None:
            return context, state_token, conditioning_outputs
        if self.latent_state_fusion is None:  # pragma: no cover - defensive
            raise RuntimeError("latent state fusion is missing")
        latent, latent_actions, latent_end = self.latent_reasoner(context, state_token)
        state_token = state_token + self.latent_state_fusion(latent.mean(dim=1))[:, None, :]
        conditioning_outputs.update(
            {
                "latent_tokens": latent,
                "latent_action_summary": latent_actions,
                "latent_end_logits": latent_end,
            }
        )
        return context, state_token, conditioning_outputs

    def _film_condition(self, context: Tensor, state_token: Tensor) -> Tensor | None:
        if not self.config.state_film_conditioning:
            return None
        return context.mean(dim=1) + state_token[:, 0]

    def _memory_tokens(self, conditioning_outputs: dict[str, Tensor]) -> Tensor | None:
        if not self.config.gated_visual_memory:
            return None
        memory = conditioning_outputs.get("visual_memory_tokens")
        if memory is None:  # pragma: no cover - defensive
            raise RuntimeError("gated visual memory tokens are missing")
        return memory

    def _deep_flow_condition(
        self,
        context: Tensor,
        state_token: Tensor,
        time: Tensor,
        conditioning_outputs: dict[str, Tensor],
    ) -> Tensor | None:
        if not self.config.deep_flow_conditioning:
            return None
        condition = self.action_expert.time_projection(time)
        condition = condition + context.mean(dim=1) + state_token[:, 0]
        # V100 uses the halting-weighted latent plan as a condition for the
        # expert's zero-initialized adaptive norms.  This is deliberately
        # inside the denoising path (rather than a post-hoc action addition),
        # so the parent V51 trajectory is preserved at transfer time.
        if self.config.latent_reasoning and "latent_tokens" in conditioning_outputs:
            condition = condition + self._latent_reasoning_summary(conditioning_outputs)
        memory_summary = conditioning_outputs.get("visual_memory_summary")
        if memory_summary is not None:
            condition = condition + memory_summary
        return condition

    def velocity(
        self,
        features: Tensor,
        state: Tensor,
        noisy_actions: Tensor,
        time: Tensor,
        *,
        history_states: Tensor | None = None,
        previous_actions: Tensor | None = None,
        visual_history: Tensor | None = None,
        phase_labels: Tensor | None = None,
        subtask_index: Tensor | None = None,
        subtask_progress: Tensor | None = None,
        cycle_state: Tensor | None = None,
    ) -> Tensor:
        context, state_token, conditioning_outputs = self._prepare_action_inputs(
            features,
            state,
            history_states=history_states,
            previous_actions=previous_actions,
            visual_history=visual_history,
            phase_labels=phase_labels,
            subtask_index=subtask_index,
            subtask_progress=subtask_progress,
            cycle_state=cycle_state,
        )
        return self.action_expert(
            context,
            state_token,
            noisy_actions,
            time,
            film_condition=self._film_condition(context, state_token),
            deep_condition=self._deep_flow_condition(
                context, state_token, time, conditioning_outputs
            ),
            visual_memory=self._memory_tokens(conditioning_outputs),
        )

    def flow_loss_components(
        self,
        features: Tensor,
        state: Tensor,
        target_actions: Tensor,
        valid_mask: Tensor,
        *,
        noise: Tensor | None = None,
        time: Tensor | None = None,
        history_states: Tensor | None = None,
        previous_actions: Tensor | None = None,
        visual_history: Tensor | None = None,
        phase_labels: Tensor | None = None,
        subtask_index: Tensor | None = None,
        subtask_progress: Tensor | None = None,
        cycle_state: Tensor | None = None,
        gripper_labels: Tensor | None = None,
        flow_actions: Tensor | None = None,
    ) -> Tensor:
        expected = (
            target_actions.ndim == 3
            and target_actions.shape[1] == self.config.action_horizon
            and target_actions.shape[2] == self.config.action_dim
        )
        if not expected or valid_mask.shape != target_actions.shape[:2]:
            raise ValueError("target_actions and valid_mask have incompatible shapes")
        noise = torch.randn_like(target_actions) if noise is None else noise
        if time is None:
            if self.config.time_sampling == "beta_1.5_1.0":
                time = torch.distributions.Beta(1.5, 1.0).sample((target_actions.shape[0],)).to(
                    target_actions.device
                )
                time = time.clamp_(0.001, 0.999)
            else:
                time = torch.rand(target_actions.shape[0], device=target_actions.device)
        time_view = time[:, None, None].to(dtype=target_actions.dtype)
        noisy_actions = (1.0 - time_view) * noise + time_view * target_actions
        target_velocity = target_actions - noise
        gripper_logits: Tensor | None = None
        if (
            self.temporal_conditioner is None
            and self.direct_action_predictor is None
            and visual_history is None
        ):
            prediction = self.velocity(features, state, noisy_actions, time)
            context, state_token, conditioning_outputs = self._prepare_action_inputs(
                features, state
            )
        else:
            context, state_token, conditioning_outputs = self._prepare_action_inputs(
                features,
                state,
                history_states=history_states,
                previous_actions=previous_actions,
                visual_history=visual_history,
                phase_labels=phase_labels,
                subtask_index=subtask_index,
                subtask_progress=subtask_progress,
                cycle_state=cycle_state,
            )
            experts = self.unified_action_experts
            task_ids = conditioning_outputs.get("task_ids")
            if experts is None or task_ids is None:
                result = self.action_expert(
                    context,
                    state_token,
                    noisy_actions,
                    time,
                    film_condition=self._film_condition(context, state_token),
                    deep_condition=self._deep_flow_condition(
                        context, state_token, time, conditioning_outputs
                    ),
                    visual_memory=self._memory_tokens(conditioning_outputs),
                    return_hidden=self.config.gripper_auxiliary,
                )
                if isinstance(result, tuple):
                    prediction, action_hidden = result
                    if self.action_expert.gripper_output is None:  # pragma: no cover
                        raise RuntimeError("gripper output head is missing")
                    gripper_logits = self.action_expert.gripper_output(action_hidden).squeeze(-1)
                else:
                    prediction = result
            else:
                head_ids = self.task_head_map.index_select(0, task_ids.long())
                prediction = torch.empty_like(noisy_actions)
                if self.config.gripper_auxiliary:
                    gripper_logits = torch.empty(
                        noisy_actions.shape[0], noisy_actions.shape[1],
                        device=noisy_actions.device, dtype=noisy_actions.dtype,
                    )
                for head_index, expert in enumerate(experts):
                    selected = head_ids == head_index
                    if not bool(selected.any()):
                        continue
                    result = expert(
                        context[selected],
                        state_token[selected],
                        noisy_actions[selected],
                        time[selected],
                        film_condition=self._film_condition(context[selected], state_token[selected]),
                        deep_condition=self._deep_flow_condition(
                            context[selected], state_token[selected], time[selected], conditioning_outputs
                        ),
                        visual_memory=(
                            None if self._memory_tokens(conditioning_outputs) is None
                            else self._memory_tokens(conditioning_outputs)[selected]
                        ),
                        return_hidden=self.config.gripper_auxiliary,
                    )
                    if isinstance(result, tuple):
                        values, action_hidden = result
                        if expert.gripper_output is None:  # pragma: no cover
                            raise RuntimeError("unified gripper output head is missing")
                        gripper_logits[selected] = expert.gripper_output(action_hidden).squeeze(-1)
                    else:
                        values = result
                    prediction[selected] = values
        loss = (prediction - target_velocity).square()
        mask = valid_mask.to(loss.dtype)[:, :, None]
        denominator = mask.sum() * target_actions.shape[-1]
        if denominator.item() == 0:
            raise ValueError("valid_mask must contain at least one valid action")
        components: dict[str, Tensor] = {
            "flow": (loss * mask).sum() / denominator,
        }
        if self.latent_reasoner is not None:
            latent_action_summary = conditioning_outputs.get("latent_action_summary")
            latent_end_logits = conditioning_outputs.get("latent_end_logits")
            if latent_action_summary is None or latent_end_logits is None:
                raise RuntimeError("latent reasoning outputs are missing")
            token_count = latent_action_summary.shape[1]
            action_targets: list[Tensor] = []
            valid_targets: list[Tensor] = []
            for token_index in range(token_count):
                start = (token_index * target_actions.shape[1]) // token_count
                stop = ((token_index + 1) * target_actions.shape[1]) // token_count
                stop = max(stop, start + 1)
                action_targets.append(target_actions[:, start:stop].mean(dim=1))
                valid_targets.append(valid_mask[:, start:stop].any(dim=1))
            coarse_target = torch.stack(action_targets, dim=1)
            coarse_valid = torch.stack(valid_targets, dim=1)
            coarse_error = (latent_action_summary - coarse_target).abs()
            coarse_mask = coarse_valid.to(coarse_error.dtype)[:, :, None]
            coarse_denom = coarse_mask.sum() * target_actions.shape[-1]
            components["latent_action"] = self.config.latent_reasoning_loss_weight * (
                coarse_error * coarse_mask
            ).sum() / coarse_denom.clamp_min(1.0)
            if self.config.latent_end_loss_weight > 0.0:
                last_valid = coarse_valid.to(torch.long).sum(dim=1).clamp_min(1) - 1
                end_target = torch.zeros_like(latent_end_logits)
                end_target.scatter_(1, last_valid[:, None], 1.0)
                end_loss = nn.functional.binary_cross_entropy_with_logits(
                    latent_end_logits,
                    end_target.to(dtype=latent_end_logits.dtype),
                )
                components["latent_end"] = self.config.latent_end_loss_weight * end_loss
        if self.direct_action_predictor is not None and self.config.direct_action_loss_weight > 0.0:
            task_ids = conditioning_outputs.get("task_ids")
            if self.config.progressive_residual:
                if flow_actions is None:
                    # A cheap one-step clean-action estimate keeps smoke tests
                    # usable.  Production V52 runs may pass cached full-flow
                    # teacher actions for an exact residual target.
                    flow_base = (
                        noisy_actions + (1.0 - time_view) * prediction.detach()
                    ).detach()
                else:
                    flow_base = flow_actions.to(device=target_actions.device, dtype=target_actions.dtype).detach()
                if tuple(flow_base.shape) != tuple(target_actions.shape):
                    raise ValueError("flow_actions must match target_actions shape")
                direct_residual, task_residual = self._direct_action_residual_components(
                    context,
                    state_token,
                    task_ids,
                    base_actions=flow_base,
                )
                direct_target = target_actions - flow_base
                stage_one = flow_base + direct_residual
                task_target = target_actions - stage_one.detach()
                direct_loss = (direct_residual - direct_target).abs()
                components["direct_residual"] = self.config.direct_action_loss_weight * (
                    direct_loss * mask
                ).sum() / denominator
                task_loss = (task_residual - task_target).abs()
                components["task_residual"] = self.config.direct_action_loss_weight * (
                    task_loss * mask
                ).sum() / denominator
                if self.config.monotonic_residual_loss_weight > 0.0:
                    e0 = (flow_base - target_actions).abs().mean(dim=-1)
                    e1 = (stage_one - target_actions).abs().mean(dim=-1)
                    e2 = (stage_one + task_residual - target_actions).abs().mean(dim=-1)
                    mono = (
                        torch.relu(e1 - e0 + self.config.monotonic_residual_margin)
                        + torch.relu(e2 - e1 + self.config.monotonic_residual_margin)
                    )
                    valid = valid_mask.to(mono.dtype)
                    components["monotonic"] = self.config.monotonic_residual_loss_weight * (
                        mono * valid
                    ).sum() / valid.sum().clamp_min(1.0)
            else:
                direct_prediction = self._direct_action_prediction(
                    context, state_token, task_ids
                )
                direct_loss = (direct_prediction - target_actions).abs()
                components["direct_action"] = self.config.direct_action_loss_weight * (
                    direct_loss * mask
                ).sum() / denominator
        if self.config.latent_residual_adapter:
            if flow_actions is None:
                parent_actions = (
                    noisy_actions + (1.0 - time_view) * prediction.detach()
                ).detach()
            else:
                parent_actions = flow_actions.to(
                    device=target_actions.device, dtype=target_actions.dtype
                ).detach()
            if tuple(parent_actions.shape) != tuple(target_actions.shape):
                raise ValueError("flow_actions must match target_actions shape")
            # The residual is trained against the complete inherited V51
            # action path when its direct/task heads are available.  Their
            # parameters stay frozen in the V99 preset, isolating the new
            # latent-to-action hypothesis.
            if (
                self.config.progressive_residual
                and self.direct_action_predictor is not None
                and self.config.direct_action_loss_weight > 0.0
            ):
                direct_residual, task_residual = self._direct_action_residual_components(
                    context,
                    state_token,
                    conditioning_outputs.get("task_ids"),
                    base_actions=parent_actions,
                )
                parent_actions = parent_actions + direct_residual.detach() + task_residual.detach()
            elif (
                self.direct_action_predictor is not None
                and self.config.direct_action_loss_weight > 0.0
            ):
                direct_prediction = self._direct_action_prediction(
                    context, state_token, conditioning_outputs.get("task_ids")
                ).detach()
                parent_actions = (
                    (1.0 - self.config.direct_action_inference_mix) * parent_actions
                    + self.config.direct_action_inference_mix * direct_prediction
                )
            latent_correction, _ = self._latent_residual_prediction(
                context, state_token, conditioning_outputs, parent_actions
            )
            latent_target = target_actions - parent_actions
            latent_loss = (latent_correction - latent_target).abs()
            components["latent_residual"] = self.config.latent_residual_loss_weight * (
                latent_loss * mask
            ).sum() / denominator
        flow_loss = components["flow"]
        phase_logits = conditioning_outputs.get("phase_logits")
        if phase_logits is not None and phase_labels is not None:
            components["phase"] = self.config.phase_loss_weight * nn.functional.cross_entropy(
                phase_logits,
                phase_labels.to(device=phase_logits.device, dtype=torch.long),
            )
        index_logits = conditioning_outputs.get("subtask_index_logits")
        progress_prediction = conditioning_outputs.get("subtask_progress_prediction")
        cycle_logits = conditioning_outputs.get("cycle_state_logits")
        if index_logits is not None and subtask_index is not None:
            components["subtask_index"] = self.config.progress_loss_weight * nn.functional.cross_entropy(
                index_logits,
                subtask_index.to(device=index_logits.device, dtype=torch.long),
            )
        if progress_prediction is not None and subtask_progress is not None:
            components["subtask_progress"] = self.config.progress_loss_weight * nn.functional.mse_loss(
                progress_prediction,
                subtask_progress.to(device=progress_prediction.device, dtype=progress_prediction.dtype),
            )
        if cycle_logits is not None and cycle_state is not None:
            components["cycle_state"] = self.config.progress_loss_weight * nn.functional.cross_entropy(
                cycle_logits,
                cycle_state.to(device=cycle_logits.device, dtype=torch.long),
            )
        if self.config.gripper_auxiliary:
            if gripper_labels is None:
                raise ValueError("V10 gripper auxiliary loss requires gripper_labels")
            if tuple(gripper_labels.shape) != tuple(valid_mask.shape):
                raise ValueError("gripper_labels must have shape [batch, action_horizon]")
            if gripper_logits is None:  # pragma: no cover - defensive
                raise RuntimeError("gripper logits are missing")
            gripper_loss = nn.functional.binary_cross_entropy_with_logits(
                gripper_logits,
                gripper_labels.to(device=gripper_logits.device, dtype=gripper_logits.dtype),
                reduction="none",
            )
            components["gripper"] = self.config.gripper_loss_weight * (
                gripper_loss * valid_mask.to(gripper_loss.dtype)
            ).sum() / valid_mask.to(gripper_loss.dtype).sum().clamp_min(1.0)
        return components

    def flow_loss(self, *args: object, **kwargs: object) -> Tensor:
        """Return the weighted sum while keeping individual loss components available."""

        components = self.flow_loss_components(*args, **kwargs)
        return sum(components.values())

    @torch.no_grad()
    def sample_actions(
        self,
        features: Tensor,
        state: Tensor,
        *,
        spatial_grid_features: Tensor | None = None,
        steps: int | None = None,
        noise: Tensor | None = None,
        history_states: Tensor | None = None,
        previous_actions: Tensor | None = None,
        visual_history: Tensor | None = None,
        phase_labels: Tensor | None = None,
        subtask_index: Tensor | None = None,
        subtask_progress: Tensor | None = None,
        cycle_state: Tensor | None = None,
        task_index: Tensor | None = None,
    ) -> Tensor:
        steps = self.config.num_flow_steps if steps is None else steps
        if steps <= 0:
            raise ValueError("steps must be positive")
        batch = features.shape[0]
        device = features.device
        dtype = self.vlm_projection.weight.dtype
        expected_shape = (batch, self.config.action_horizon, self.config.action_dim)
        if noise is None:
            actions = torch.randn(expected_shape, device=device, dtype=dtype)
        else:
            if tuple(noise.shape) != expected_shape:
                raise ValueError(f"noise must have shape {expected_shape}")
            actions = noise.to(device=device, dtype=dtype).clone()
        ordered_relative_inputs: tuple[Tensor, Tensor] | None = None
        spatial_grid_inputs: tuple[Tensor, Tensor] | None = None
        if self.config.ordered_relative_residual_adapter:
            # Keep the sidecar task-agnostic: it receives only continuous
            # frozen-Qwen features and normalized robot state.  The V51 main
            # path below may still use its language-inferred internal route.
            ordered_relative_inputs = self._prepare(features, state)
        if self.config.spatial_grid_residual_adapter:
            # Keep the V74 sidecar contract identical to its offline training
            # path: it reads the frozen visual/state context before task and
            # temporal conditioning.  The main V51 action path may still use
            # language-inferred task conditioning below, but the shared
            # geometry correction never receives a task id or history.
            spatial_grid_inputs = self._prepare(features, state)
        context, state_token, conditioning_outputs = self._prepare_action_inputs(
            features,
            state,
            history_states=history_states,
            previous_actions=previous_actions,
            visual_history=visual_history,
            phase_labels=phase_labels,
            subtask_index=subtask_index,
            subtask_progress=subtask_progress,
            cycle_state=cycle_state,
            task_index=task_index,
        )
        action_expert = self.action_expert
        if self.unified_action_experts is not None:
            task_ids = conditioning_outputs.get("task_ids")
            if task_ids is None:
                raise RuntimeError("unified action experts require task-conditioned inference")
            head_ids = self.task_head_map.index_select(0, task_ids.long())
            if not bool((head_ids == head_ids[0]).all()):
                raise RuntimeError(
                    "unified action-expert bank currently requires one task head per inference batch"
                )
            action_expert = self.unified_action_experts[int(head_ids[0])]
        gripper_logits: Tensor | None = None
        for index in range(steps):
            t_value = index / steps
            time = torch.full((batch,), t_value, device=device, dtype=dtype)
            expert_output = action_expert(
                context,
                state_token,
                actions,
                time,
                film_condition=self._film_condition(context, state_token),
                deep_condition=self._deep_flow_condition(
                    context, state_token, time, conditioning_outputs
                ),
                visual_memory=self._memory_tokens(conditioning_outputs),
                return_hidden=self.config.gripper_auxiliary
                and self.config.gripper_auxiliary_inference,
            )
            if isinstance(expert_output, tuple):
                velocity, action_hidden = expert_output
                if action_expert.gripper_output is None:  # pragma: no cover - defensive
                    raise RuntimeError("gripper output head is missing")
                gripper_logits = action_expert.gripper_output(action_hidden).squeeze(-1)
            else:
                velocity = expert_output
            actions = actions + velocity / steps
        if self.config.progressive_residual:
            direct_residual, task_residual = self._direct_action_residual_components(
                context,
                state_token,
                conditioning_outputs.get("task_ids"),
                base_actions=actions.detach(),
            )
            actions = (
                actions
                + self.config.direct_residual_inference_scale * direct_residual
                + self.config.task_residual_inference_scale * task_residual
            )
        elif (self.direct_action_predictor is not None or self.unified_direct_predictors is not None):
            direct_actions = self._direct_action_prediction(
                context, state_token, conditioning_outputs.get("task_ids")
            )
            if self.unified_direct_predictors is not None:
                head_ids = self.task_head_map.index_select(
                    0, conditioning_outputs["task_ids"].long()
                )
                mix = self.unified_head_mix.index_select(0, head_ids).to(actions.dtype)
                mix = mix[:, None, None]
            else:
                mix = torch.full(
                    (actions.shape[0], 1, 1),
                    self.config.direct_action_inference_mix,
                    device=actions.device,
                    dtype=actions.dtype,
                )
            actions = (1.0 - mix) * actions + mix * direct_actions
        if self.config.latent_residual_adapter:
            latent_correction, _ = self._latent_residual_prediction(
                context,
                state_token,
                conditioning_outputs,
                actions.detach(),
            )
            actions = actions + latent_correction
        if self.global_action_residual is not None:
            residual = self._global_action_residual_prediction(
                context, state_token, actions.detach()
            )
            actions = actions + self.config.global_action_residual_scale * residual
        if self.config.state_adaptive_residual_adapter:
            actions, _ = self._apply_state_adaptive_residual(
                context, state_token, actions.detach()
            )
        if self.config.ordered_relative_residual_adapter:
            if ordered_relative_inputs is None:  # pragma: no cover - narrowed above
                raise RuntimeError("ordered relative inputs are missing")
            actions, _ = self._apply_ordered_relative_residual(
                ordered_relative_inputs[0], ordered_relative_inputs[1], actions.detach()
            )
        if self.config.spatial_grid_residual_adapter:
            if spatial_grid_features is None or spatial_grid_inputs is None:
                raise ValueError("V74 spatial_grid_features are required for sampling")
            actions, _ = self._apply_spatial_grid_residual(
                spatial_grid_features,
                spatial_grid_inputs[0],
                spatial_grid_inputs[1],
                actions.detach(),
            )
        if self.config.gripper_auxiliary and self.config.gripper_auxiliary_inference:
            if gripper_logits is None:  # pragma: no cover - defensive
                raise RuntimeError("gripper auxiliary logits are missing")
            actions = actions.clone()
            auxiliary_action = gripper_logits.sigmoid().mul(2.0).sub(1.0)
            mix = self.config.gripper_auxiliary_inference_mix
            actions[..., -1] = (
                (1.0 - mix) * actions[..., -1] + mix * auxiliary_action
            ).clamp(-1.0, 1.0)
        return actions
