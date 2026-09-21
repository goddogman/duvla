#!/usr/bin/env python3
"""Train one formally defined Duvla V2.1 stage from schema-7 caches."""

from __future__ import annotations

import argparse
import copy
from collections import Counter
from dataclasses import asdict, replace
from datetime import datetime, timezone
import json
from itertools import zip_longest
import math
from pathlib import Path
import random
from typing import Any

import numpy as np
import torch

from duvla.models.duvla_v2_1 import DuvlaV21Config, DuvlaV21Policy
from duvla.models.duvla_v3_21 import DuvlaV321Config, DuvlaV321Policy
from duvla.models.duvla_v3_22 import DuvlaV322Config, DuvlaV322Policy
from duvla.training.feature_cache import atomic_write_json
from duvla.training.loss_curve import (
    append_loss_point,
    truncate_loss_points,
    write_loss_curve_artifacts,
)
from duvla.training.v2_1_data import (
    deterministic_flow_noise,
    iter_v2_1_batches,
    load_long_horizon_action_sidecar,
    prefetch_v2_1_batches,
    validate_v2_1_manifest,
)


# Exact train-split action-chunk counts for schema-7/horizon=8:
# 1,470,770 non-transitions and 21,980 transitions.
V2_1_GRIPPER_TRANSITION_POSITIVE_WEIGHT = 1470770 / 21980
# Train-only count after adding the replan boundary transition at chunk step 0:
# 1,686,296 non-transitions and 25,120 transitions.
V2_1_R1B_GRIPPER_TRANSITION_POSITIVE_WEIGHT = 1686296 / 25120


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-cache", type=Path, required=True)
    parser.add_argument("--validation-cache", type=Path)
    parser.add_argument("--multilayer-aux-cache", type=Path)
    parser.add_argument("--validation-multilayer-aux-cache", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--stage", choices=("flow", "gripper", "direct", "instruction", "joint"), default="flow"
    )
    parser.add_argument("--variant", choices=("multilayer", "layer14"), default="multilayer")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--micro-batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=16)
    parser.add_argument("--prefetch-batches", type=int, choices=(0, 1), default=1)
    parser.add_argument("--shard-shuffle-block-size", type=int, default=4)
    parser.add_argument("--mmap-shards", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--parent-checkpoint", type=Path)
    parser.add_argument("--parent-actions", type=Path)
    parser.add_argument("--action-sidecar", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--log-every", type=int, default=5000)
    parser.add_argument("--metrics-every", type=int, default=100)
    parser.add_argument(
        "--resume-every",
        type=int,
        default=500,
        help="atomically save model/optimizer/scheduler state every N optimizer steps",
    )
    parser.add_argument("--milestone-epochs", default="1,3,6,8,10,15,20,25,30")
    parser.add_argument("--max-train-samples", type=int, default=0, help="smoke-test limit only")
    parser.add_argument("--smoke-model", action="store_true", help="tiny CPU/GPU contract test model")
    candidate = parser.add_mutually_exclusive_group()
    candidate.add_argument(
        "--r1b-action-coupling",
        action="store_true",
        help="enable stride-2 action history and action-conditioned gripper architecture",
    )
    candidate.add_argument(
        "--r1c-causal-gripper",
        action="store_true",
        help="use stride-2 visual history and previous-gripper-only conditioning without arm action teacher forcing",
    )
    candidate.add_argument(
        "--r1d-event-gripper",
        action="store_true",
        help="use causal three-class HOLD/CLOSE/OPEN gripper events aligned with inference",
    )
    candidate.add_argument(
        "--r1e-event-recalibration",
        action="store_true",
        help="freeze an R1d parent and retrain the event head on inference-matched Flow actions",
    )
    candidate.add_argument(
        "--v2-2-unified-flow",
        action="store_true",
        help="train V2.2 G0 with gripper generated directly by the unified 7-D Flow",
    )
    candidate.add_argument(
        "--v2-3-long-horizon-flow",
        action="store_true",
        help="train V2.3 with a long-horizon sidecar and unified 7-D Flow",
    )
    candidate.add_argument(
        "--v2-4-execution-aligned-flow",
        action="store_true",
        help="train V2.4 unified Flow with stride-2 history and executed-prefix/event weighting",
    )
    candidate.add_argument(
        "--v2-5-highres-flow",
        action="store_true",
        help="train V2.5 with exact layer-14 8x8 spatial tokens and multi-layer semantics",
    )
    candidate.add_argument(
        "--v2-6-dense-interleaved-flow",
        action="store_true",
        help="train V2.6 with full-rank Qwen projection and alternating cross/self Flow",
    )
    candidate.add_argument(
        "--v2-7-stable-gripper-flow",
        action="store_true",
        help="train V2.7 with the V2.6 dense path but without transition-overweighted gripper Flow",
    )
    candidate.add_argument(
        "--v3-19-update-sufficient-flow",
        action="store_true",
        help="train V3.19 with the V2.6 dense path using a smaller effective batch",
    )
    candidate.add_argument(
        "--v3-21-parallel-action-query",
        action="store_true",
        help="train the fresh V3.21 parallel continuous action-query generator",
    )
    candidate.add_argument(
        "--v3-22-prior-residual-flow",
        action="store_true",
        help="train arm Flow residuals around a frozen V3.21 action-query prior",
    )
    candidate.add_argument(
        "--v3-23-multilayer-spatial-flow",
        action="store_true",
        help="train dense Flow on four Qwen spatial layers at 4x4 per camera",
    )
    candidate.add_argument(
        "--v3-24-highres-multilayer-residual",
        action="store_true",
        help="train a bounded multi-layer 4x4 residual on the frozen V2.6 8x8 anchor",
    )
    candidate.add_argument(
        "--v3-27-official2000-flow",
        action="store_true",
        help="train the V2.6 dense Flow architecture from scratch on all 2,000 official demonstrations",
    )
    candidate.add_argument(
        "--v3-28-fp32-amp-flow",
        action="store_true",
        help="train V3.28 with FP32 parameters/AdamW state and BF16 autocast on the official-2000 cache",
    )
    return parser.parse_args()


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _atomic_torch_save(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def _candidate_identity(args: argparse.Namespace) -> tuple[str, str, str]:
    if args.v3_28_fp32_amp_flow:
        return "V3.28", "duvla_v3_28", "v3.28"
    if args.v3_27_official2000_flow:
        return "V3.27", "duvla_v3_27", "v3.27"
    if args.v3_24_highres_multilayer_residual:
        return "V3.24", "duvla_v3_24", "v3.24"
    if args.v3_23_multilayer_spatial_flow:
        return "V3.23", "duvla_v3_23", "v3.23"
    if args.v3_22_prior_residual_flow:
        return "V3.22", "duvla_v3_22", "v3.22"
    if args.v3_21_parallel_action_query:
        return "V3.21", "duvla_v3_21", "v3.21"
    if args.v3_19_update_sufficient_flow:
        return "V3.19", "duvla_v3_19", "v3.19"
    if args.v2_7_stable_gripper_flow:
        return "V2.7", "duvla_v2_7", "v2.7"
    if args.v2_6_dense_interleaved_flow:
        return "V2.6", "duvla_v2_6", "v2.6"
    if args.v2_5_highres_flow:
        return "V2.5", "duvla_v2_5", "v2.5"
    if args.v2_4_execution_aligned_flow:
        return "V2.4", "duvla_v2_4", "v2.4"
    if args.v2_3_long_horizon_flow:
        return "V2.3", "duvla_v2_3", "v2.3"
    if args.v2_2_unified_flow:
        return "V2.2", "duvla_v2_2", "v2.2"
    return "V2.1", "duvla_v2_1", "v2.1"


def _v3_21_model_config(
    manifest: dict[str, object], *, smoke: bool
) -> DuvlaV321Config:
    action_mean = manifest.get("action_mean")
    action_std = manifest.get("action_std")
    if not isinstance(action_mean, list) or not isinstance(action_std, list):
        raise ValueError("cache manifest lacks action normalization")
    if float(action_std[-1]) <= 0.0:
        raise ValueError("gripper action standard deviation must be positive")
    config = DuvlaV321Config(
        gripper_open_value=(-1.0 - float(action_mean[-1])) / float(action_std[-1]),
        gripper_close_value=(1.0 - float(action_mean[-1])) / float(action_std[-1]),
    )
    if smoke:
        config = replace(
            config,
            hidden_dim=96,
            decoder_layers=2,
            decoder_heads=4,
        )
    return config


def _model_config(
    manifest: dict[str, object],
    *,
    smoke: bool,
    r1b_action_coupling: bool = False,
    r1c_causal_gripper: bool = False,
    r1d_event_gripper: bool = False,
    r1e_event_recalibration: bool = False,
    v2_2_unified_flow: bool = False,
    v2_3_long_horizon_flow: bool = False,
    v2_4_execution_aligned_flow: bool = False,
    v2_5_highres_flow: bool = False,
    v2_6_dense_interleaved_flow: bool = False,
    v2_7_stable_gripper_flow: bool = False,
    v3_19_update_sufficient_flow: bool = False,
    v3_23_multilayer_spatial_flow: bool = False,
    v3_24_highres_multilayer_residual: bool = False,
    v3_27_official2000_flow: bool = False,
    v3_28_fp32_amp_flow: bool = False,
    action_horizon: int = 8,
) -> DuvlaV21Config:
    if sum(
        (
            r1b_action_coupling,
            r1c_causal_gripper,
            r1d_event_gripper,
            r1e_event_recalibration,
            v2_2_unified_flow,
            v2_3_long_horizon_flow,
            v2_4_execution_aligned_flow,
            v2_5_highres_flow,
            v2_6_dense_interleaved_flow,
            v2_7_stable_gripper_flow,
            v3_19_update_sufficient_flow,
            v3_23_multilayer_spatial_flow,
            v3_24_highres_multilayer_residual,
            v3_27_official2000_flow,
            v3_28_fp32_amp_flow,
        )
    ) > 1:
        raise ValueError("R1b, R1c, R1d, and R1e recipes are mutually exclusive")
    event_aligned = r1d_event_gripper or r1e_event_recalibration
    action_mean = manifest.get("action_mean")
    action_std = manifest.get("action_std")
    if not isinstance(action_mean, list) or not isinstance(action_std, list):
        raise ValueError("cache manifest lacks action normalization")
    gripper_mean = float(action_mean[-1])
    gripper_std = float(action_std[-1])
    if gripper_std <= 0.0:
        raise ValueError("gripper action standard deviation must be positive")
    config = DuvlaV21Config(
        gripper_open_value=(-1.0 - gripper_mean) / gripper_std,
        gripper_close_value=(1.0 - gripper_mean) / gripper_std,
        gripper_transition_positive_weight=(
            1.0
            if event_aligned
            else V2_1_R1B_GRIPPER_TRANSITION_POSITIVE_WEIGHT
            if r1b_action_coupling or r1c_causal_gripper
            else V2_1_GRIPPER_TRANSITION_POSITIVE_WEIGHT
        ),
        history_stride=(
            2
            if r1b_action_coupling
            or r1c_causal_gripper
            or event_aligned
            or v2_4_execution_aligned_flow
            or v2_5_highres_flow
            or v2_6_dense_interleaved_flow
            or v2_7_stable_gripper_flow
            or v3_19_update_sufficient_flow
            or v3_23_multilayer_spatial_flow
            or v3_24_highres_multilayer_residual
            or v3_27_official2000_flow
            or v3_28_fp32_amp_flow
            else 1
        ),
        action_history_conditioning=r1b_action_coupling,
        gripper_action_conditioning=(
            r1b_action_coupling or r1c_causal_gripper or event_aligned
        ),
        gripper_previous_action_mode=(
            "gripper_only" if r1c_causal_gripper or event_aligned else "full"
        ),
        gripper_control_mode="event3" if event_aligned else "absolute",
        gripper_event_class_weights=(
            (1.0, 3.0, 3.0)
            if r1e_event_recalibration
            else (1.0, 10.0, 10.0)
        ),
        gripper_event_focal_gamma=(
            1.0 if r1e_event_recalibration else 2.0 if r1d_event_gripper else 0.0
        ),
        gripper_event_loss_weight=0.05,
        gripper_state_aux_loss_weight=0.01,
        gripper_event_probability_threshold=0.5,
        gripper_event_change_weight=(
            8.0 if r1b_action_coupling or r1c_causal_gripper else 1.0
        ),
        flow_endpoint_loss_weight=(
            0.02
            if r1c_causal_gripper or event_aligned
            else 0.1
            if r1b_action_coupling
            else 0.0
        ),
        flow_prefix_weight=(
            4.0
            if v2_4_execution_aligned_flow
            or v2_5_highres_flow
            or v2_6_dense_interleaved_flow
            or v2_7_stable_gripper_flow
            or v3_19_update_sufficient_flow
            or v3_23_multilayer_spatial_flow
            or v3_24_highres_multilayer_residual
            or v3_27_official2000_flow
            or v3_28_fp32_amp_flow
            else 1.0
        ),
        flow_gripper_transition_weight=(
            10.0
            if v2_4_execution_aligned_flow
            or v2_5_highres_flow
            or v2_6_dense_interleaved_flow
            or v3_19_update_sufficient_flow
            or v3_23_multilayer_spatial_flow
            or v3_24_highres_multilayer_residual
            or v3_27_official2000_flow
            or v3_28_fp32_amp_flow
            else 1.0
        ),
        unified_flow_gripper=(
            v2_2_unified_flow
            or v2_3_long_horizon_flow
            or v2_4_execution_aligned_flow
            or v2_5_highres_flow
            or v2_6_dense_interleaved_flow
            or v2_7_stable_gripper_flow
            or v3_19_update_sufficient_flow
            or v3_23_multilayer_spatial_flow
            or v3_24_highres_multilayer_residual
            or v3_27_official2000_flow
            or v3_28_fp32_amp_flow
        ),
        spatial_tokens=(
            64
            if v2_5_highres_flow
            or v2_6_dense_interleaved_flow
            or v2_7_stable_gripper_flow
            or v3_19_update_sufficient_flow
            or v3_24_highres_multilayer_residual
            or v3_27_official2000_flow
            or v3_28_fp32_amp_flow
            else 16
        ),
        highres_layer14_visual=(
            v2_5_highres_flow
            or v2_6_dense_interleaved_flow
            or v2_7_stable_gripper_flow
            or v3_19_update_sufficient_flow
            or v3_24_highres_multilayer_residual
            or v3_27_official2000_flow
            or v3_28_fp32_amp_flow
        ),
        dense_interleaved_flow=(
            v2_6_dense_interleaved_flow
            or v2_7_stable_gripper_flow
            or v3_19_update_sufficient_flow
            or v3_23_multilayer_spatial_flow
            or v3_24_highres_multilayer_residual
            or v3_27_official2000_flow
            or v3_28_fp32_amp_flow
        ),
        dense_multilayer_spatial_flow=v3_23_multilayer_spatial_flow,
        multilayer_spatial_residual=v3_24_highres_multilayer_residual,
        context_layers=(
            0
            if v2_6_dense_interleaved_flow
            or v2_7_stable_gripper_flow
            or v3_19_update_sufficient_flow
            or v3_23_multilayer_spatial_flow
            or v3_24_highres_multilayer_residual
            or v3_27_official2000_flow
            or v3_28_fp32_amp_flow
            else 4
        ),
        flow_loss_fp32=v3_28_fp32_amp_flow,
        action_horizon=action_horizon,
    )
    if smoke:
        config = replace(
            config,
            adapter_rank=16,
            hidden_dim=96,
            expert_layers=2,
            expert_heads=4,
            context_layers=(
                0
                if v2_6_dense_interleaved_flow
                or v2_7_stable_gripper_flow
                or v3_19_update_sufficient_flow
                or v3_23_multilayer_spatial_flow
                or v3_24_highres_multilayer_residual
                or v3_27_official2000_flow
                or v3_28_fp32_amp_flow
                else 1
            ),
            residual_layers=1,
            flow_steps=2,
            flow_samples=2,
        )
    return config


def _configure_stage(
    policy: DuvlaV21Policy | DuvlaV321Policy | DuvlaV322Policy, stage: str
) -> tuple[str, ...]:
    if isinstance(policy, DuvlaV322Policy):
        if stage != "flow":
            raise ValueError("V3.22 is trained only through its residual_flow stage")
        policy.requires_grad_(False)
        policy.residual_flow.requires_grad_(True)
        return ("residual_flow",)
    if isinstance(policy, DuvlaV321Policy):
        if stage != "flow":
            raise ValueError("V3.21 is trained only through its unified generator stage")
        policy.requires_grad_(True)
        return ("fusion", "history_position", "action_queries", "decoder", "heads")
    policy.requires_grad_(False)
    if policy.config.multilayer_spatial_residual:
        if stage != "flow":
            raise ValueError("V3.24 is trained only through its spatial residual stage")
        module = policy.fusion.multilayer_spatial_residual
        if module is None:  # pragma: no cover - configuration contract
            raise RuntimeError("V3.24 multi-layer spatial residual is unavailable")
        module.requires_grad_(True)
        return ("fusion.multilayer_spatial_residual",)
    trainable_groups: list[str] = []
    if stage == "flow":
        names = ["fusion", "context_blocks", "flow_expert"]
        if policy.config.unified_flow_gripper:
            pass
        elif policy.config.gripper_action_conditioning:
            if policy.previous_action_projection is not None:
                names.append("previous_action_projection")
            names.append("action_conditioned_gripper")
        else:
            names.extend(("gripper_event_head", "gripper_transition_head"))
        for name in names:
            module = getattr(policy, name)
            if module is None:  # pragma: no cover - guarded by config
                raise RuntimeError(f"trainable module {name} is unavailable")
            module.requires_grad_(True)
            trainable_groups.append(name)
        policy.history_position.requires_grad_(True)
        trainable_groups.append("history_position")
    elif stage == "direct":
        policy.direct_residual.requires_grad_(True)
        trainable_groups.append("direct_residual")
    elif stage == "gripper":
        if policy.action_conditioned_gripper is None:
            raise RuntimeError("gripper stage requires an action-conditioned head")
        policy.action_conditioned_gripper.requires_grad_(True)
        trainable_groups.append("action_conditioned_gripper")
    elif stage == "instruction":
        policy.instruction_residual.requires_grad_(True)
        trainable_groups.append("instruction_residual")
    else:
        policy.fusion.requires_grad_(True)
        policy.direct_residual.gate.requires_grad_(True)
        policy.instruction_residual.gate.requires_grad_(True)
        gripper_groups: tuple[str, ...]
        if policy.config.unified_flow_gripper:
            gripper_groups = ()
        elif policy.config.gripper_action_conditioning:
            if policy.action_conditioned_gripper is None:
                raise RuntimeError("action-conditioned gripper module is unavailable")
            gripper_group_names: list[str] = []
            if policy.previous_action_projection is not None:
                policy.previous_action_projection.requires_grad_(True)
                gripper_group_names.append("previous_action_projection")
            policy.action_conditioned_gripper.requires_grad_(True)
            gripper_group_names.append("action_conditioned_gripper")
            gripper_groups = tuple(gripper_group_names)
        else:
            if policy.gripper_event_head is None or policy.gripper_transition_head is None:
                raise RuntimeError("legacy gripper modules are unavailable")
            policy.gripper_event_head.requires_grad_(True)
            policy.gripper_transition_head.requires_grad_(True)
            gripper_groups = ("gripper_event_head", "gripper_transition_head")
        trainable_groups.extend(
            (
                "fusion",
                "direct_residual.gate",
                "instruction_residual.gate",
                *gripper_groups,
            )
        )
    return tuple(trainable_groups)


def _load_parent_actions(
    path: Path | None,
    *,
    cache_signature: object,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    if path is None:
        return None
    payload = torch.load(path, map_location="cpu", weights_only=True)
    signature = payload.get("signature")
    if (
        not isinstance(signature, dict)
        or signature.get("format") != "duvla_v2_1_parent_actions"
        or signature.get("stage") != "flow"
        or signature.get("cache_signature") != cache_signature
        or signature.get("benchmark_task_index") is not False
    ):
        raise ValueError("parent actions are not the strict Flow-stage cache for this dataset")
    actions = payload.get("actions")
    available = payload.get("available")
    if not isinstance(actions, torch.Tensor) or actions.ndim != 3:
        raise ValueError("parent action cache must contain rank-three actions")
    if not isinstance(available, torch.Tensor) or available.shape != actions.shape[:1]:
        raise ValueError("parent action cache must contain a matching available mask")
    return actions, available.to(dtype=torch.bool)


def _parent_batch(
    store: tuple[torch.Tensor, torch.Tensor] | None,
    indices: torch.Tensor,
) -> torch.Tensor | None:
    if store is None:
        return None
    actions, available = store
    if int(indices.max()) >= actions.shape[0] or not bool(available.index_select(0, indices).all()):
        raise ValueError("parent action cache does not cover the requested dataset indices")
    return actions.index_select(0, indices)


def _task_weights(manifest: dict[str, object]) -> torch.Tensor:
    raw = manifest.get("selected_task_sample_counts")
    if not isinstance(raw, dict) or not raw:
        raise ValueError("schema-7 manifest lacks selected_task_sample_counts")
    counts = {int(key): int(value) for key, value in raw.items()}
    if any(value <= 0 for value in counts.values()):
        raise ValueError("every selected task must contain training samples")
    task_count = max(counts) + 1
    total = sum(counts.values())
    weights = torch.ones(task_count, dtype=torch.float32)
    for task, count in counts.items():
        weights[task] = total / (len(counts) * count)
    return weights


def _move(batch: dict[str, torch.Tensor], device: torch.device, dtype: torch.dtype) -> dict[str, torch.Tensor]:
    result: dict[str, torch.Tensor] = {}
    for key, value in batch.items():
        if key in {
            "visual",
            "semantic",
            "states",
            "previous_actions",
            "actions",
            "history_visual",
            "history_semantic",
            "history_states",
            "history_previous_actions",
            "auxiliary_visual",
            "history_auxiliary_visual",
        }:
            result[key] = value.to(device=device, dtype=dtype, non_blocking=True)
        else:
            result[key] = value.to(device=device, non_blocking=True)
    return result


def _aligned_auxiliary_batches(
    primary: object, auxiliary: object
) -> object:
    """Attach a schema-7 multi-layer view to matching schema-8 batches."""

    for base, extra in zip_longest(primary, auxiliary):
        if base is None or extra is None:
            raise RuntimeError("primary and auxiliary caches contain different batch counts")
        for key in ("dataset_indices", "episode_indices", "task_indices"):
            if not torch.equal(base[key], extra[key]):
                raise RuntimeError(f"primary/auxiliary cache rows differ for {key}")
        if not torch.equal(base["valid_mask"], extra["valid_mask"]):
            raise RuntimeError("primary/auxiliary valid masks differ")
        base["auxiliary_visual"] = extra["visual"]
        base["history_auxiliary_visual"] = extra["history_visual"]
        yield base


def _rescale_partial_accumulation(
    parameters: list[torch.nn.Parameter],
    *,
    completed_micro_steps: int,
    target_micro_steps: int,
) -> None:
    if not 0 < completed_micro_steps <= target_micro_steps:
        raise ValueError("partial accumulation count is outside the target interval")
    if completed_micro_steps == target_micro_steps:
        return
    scale = target_micro_steps / completed_micro_steps
    for parameter in parameters:
        if parameter.grad is not None:
            parameter.grad.mul_(scale)


def _assert_fp32_training_state(
    parameters: list[torch.nn.Parameter], optimizer: torch.optim.Optimizer
) -> None:
    if any(parameter.dtype != torch.float32 for parameter in parameters):
        raise RuntimeError("V3.28 requires every trainable parameter to remain FP32")
    for parameter in parameters:
        if parameter.grad is not None and parameter.grad.dtype != torch.float32:
            raise RuntimeError("V3.28 requires FP32 accumulated gradients")
        for value in optimizer.state.get(parameter, {}).values():
            if isinstance(value, torch.Tensor) and value.is_floating_point():
                if value.dtype != torch.float32:
                    raise RuntimeError("V3.28 requires FP32 AdamW state")


def _checkpoint_payload(
    policy: DuvlaV21Policy | DuvlaV321Policy | DuvlaV322Policy,
    config: DuvlaV21Config | DuvlaV321Config | DuvlaV322Config,
    *,
    args: argparse.Namespace,
    epoch: int,
    optimizer_step: int,
    effective_samples: int,
    trainable_groups: tuple[str, ...],
    manifest: dict[str, object],
    validation_losses: dict[str, float] | None = None,
) -> dict[str, object]:
    version, checkpoint_format, _artifact_version = _candidate_identity(args)
    return {
        "format": checkpoint_format,
        "version": version,
        "stage": args.stage,
        "variant": args.variant,
        "r1b_action_coupling": args.r1b_action_coupling,
        "r1c_causal_gripper": args.r1c_causal_gripper,
        "r1d_event_gripper": args.r1d_event_gripper,
        "r1e_event_recalibration": args.r1e_event_recalibration,
        "v2_2_unified_flow": args.v2_2_unified_flow,
        "v2_3_long_horizon_flow": args.v2_3_long_horizon_flow,
        "v2_4_execution_aligned_flow": args.v2_4_execution_aligned_flow,
        "v2_5_highres_flow": args.v2_5_highres_flow,
        "v2_6_dense_interleaved_flow": args.v2_6_dense_interleaved_flow,
        "v2_7_stable_gripper_flow": args.v2_7_stable_gripper_flow,
        "v3_19_update_sufficient_flow": args.v3_19_update_sufficient_flow,
        "v3_21_parallel_action_query": args.v3_21_parallel_action_query,
        "v3_22_prior_residual_flow": args.v3_22_prior_residual_flow,
        "v3_23_multilayer_spatial_flow": args.v3_23_multilayer_spatial_flow,
        "v3_24_highres_multilayer_residual": args.v3_24_highres_multilayer_residual,
        "v3_27_official2000_flow": args.v3_27_official2000_flow,
        "v3_28_fp32_amp_flow": args.v3_28_fp32_amp_flow,
        "training_precision": (
            "fp32_parameters_adamw_bf16_autocast_fp32_loss"
            if args.v3_28_fp32_amp_flow
            else "full_bfloat16_cuda"
        ),
        "action_parameterization": (
            "parallel_prior_plus_arm_residual_flow"
            if args.v3_22_prior_residual_flow
            else
            "parallel_continuous_query"
            if args.v3_21_parallel_action_query
            else "conditional_flow_matching"
        ),
        "epoch": epoch,
        "optimizer_step": optimizer_step,
        "effective_samples": effective_samples,
        "effective_epochs": effective_samples / int(manifest["selected_frames"]),
        "model_config": asdict(config),
        "model_state_dict": policy.state_dict(),
        "trainable_groups": trainable_groups,
        "benchmark_task_index": False,
        "task_routing": "natural_language",
        "cache_signature": manifest.get("cache_signature"),
        "validation_losses": validation_losses,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }


@torch.no_grad()
def _validate_v3_21(
    policy: DuvlaV321Policy,
    cache: Path,
    *,
    config: DuvlaV321Config,
    device: torch.device,
    dtype: torch.dtype,
    batch_size: int,
    weights_by_task: torch.Tensor,
) -> dict[str, float]:
    """Evaluate the deterministic V3.21 objective on episode-disjoint data."""
    policy.eval()
    totals: Counter[str] = Counter()
    sample_count = 0
    for batch in iter_v2_1_batches(
        cache,
        batch_size=batch_size,
        history_length=config.history_length,
        history_stride=config.history_stride,
        seed=0,
        epoch=0,
        shuffle=False,
        shard_shuffle_block_size=1,
        mmap_shards=True,
    ):
        cpu_tasks = batch["task_indices"]
        moved = _move(batch, device, dtype)
        sample_weights = weights_by_task.index_select(0, cpu_tasks).to(
            device=device, dtype=dtype
        )
        losses = policy.flow_loss_components(
            moved["visual"],
            moved["semantic"],
            moved["states"],
            moved["actions"],
            moved["valid_mask"],
            history_visual=moved.get("history_visual"),
            history_semantic=moved.get("history_semantic"),
            history_states=moved.get("history_states"),
            previous_actions=moved.get("previous_actions"),
            stage="flow",
            sample_weights=sample_weights,
        )
        current = int(cpu_tasks.shape[0])
        sample_count += current
        for name, value in losses.items():
            totals[name] += float(value) * current
    result = {name: value / max(sample_count, 1) for name, value in totals.items()}
    result["total"] = sum(result.values())
    result["samples"] = float(sample_count)
    return result


def main() -> None:
    args = parse_args()
    if (
        args.epochs <= 0
        or args.micro_batch_size <= 0
        or args.gradient_accumulation_steps <= 0
        or args.learning_rate <= 0.0
        or args.shard_shuffle_block_size <= 0
        or args.log_every <= 0
        or args.metrics_every <= 0
        or args.resume_every <= 0
        or args.weight_decay < 0.0
        or not 0.0 <= args.warmup_ratio < 1.0
        or args.max_train_samples < 0
    ):
        raise SystemExit("training sizes/rates are invalid")
    if args.stage in {"gripper", "direct", "instruction"} and (
        args.parent_checkpoint is None or args.parent_actions is None
    ):
        raise SystemExit(
            "gripper/direct/instruction stages require --parent-checkpoint and Flow --parent-actions"
        )
    if args.r1e_event_recalibration != (args.stage == "gripper"):
        raise SystemExit("R1e event recalibration must use --stage gripper, and gripper stage must use R1e")
    if args.stage == "joint" and args.parent_checkpoint is None:
        raise SystemExit("joint calibration requires --parent-checkpoint")
    if (
        args.stage == "flow"
        and args.parent_checkpoint is not None
        and not (args.v3_22_prior_residual_flow or args.v3_24_highres_multilayer_residual)
    ):
        raise SystemExit("R1 Flow must use a fresh initialization, not --parent-checkpoint")
    if args.v3_22_prior_residual_flow and args.parent_checkpoint is None:
        raise SystemExit("V3.22 requires its frozen V3.21 parent checkpoint")
    if args.v3_24_highres_multilayer_residual and args.parent_checkpoint is None:
        raise SystemExit("V3.24 requires the frozen V2.6 parent checkpoint")
    if args.v3_24_highres_multilayer_residual != (args.multilayer_aux_cache is not None):
        raise SystemExit("V3.24 requires --multilayer-aux-cache, reserved for that recipe")
    if args.v3_24_highres_multilayer_residual != (
        args.validation_multilayer_aux_cache is not None
    ):
        raise SystemExit(
            "V3.24 requires --validation-multilayer-aux-cache, reserved for that recipe"
        )
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but unavailable")
    _seed_everything(args.seed)
    torch.set_float32_matmul_precision("high")
    manifest = validate_v2_1_manifest(args.train_cache)
    official2000_recipe = args.v3_27_official2000_flow or args.v3_28_fp32_amp_flow
    if official2000_recipe and not args.smoke_model and not (
        manifest.get("candidate") == "Duvla V3.27"
        and manifest.get("split") == "all_demonstrations_train"
        and manifest.get("all_demonstrations") is True
        and manifest.get("selected_frames") == 338_575
        and manifest.get("total_episodes") == 2000
        and manifest.get("train_episodes") == 2000
        and manifest.get("validation_episodes") == 0
        and manifest.get("zero_arm_rows_retained") is True
        and manifest.get("uses_evaluation_initial_states") is False
    ):
        raise SystemExit("V3.27 requires the exact official-2000 all-demonstration cache")
    highres_cache = manifest.get("schema_version") == 8
    highres_recipe = (
        args.v2_5_highres_flow
        or args.v2_6_dense_interleaved_flow
        or args.v2_7_stable_gripper_flow
        or args.v3_19_update_sufficient_flow
        or args.v3_21_parallel_action_query
        or args.v3_22_prior_residual_flow
        or args.v3_24_highres_multilayer_residual
        or args.v3_27_official2000_flow
        or args.v3_28_fp32_amp_flow
    )
    if highres_cache != highres_recipe:
        raise SystemExit(
            "schema-8 high-resolution cache requires exactly one high-resolution V2 recipe"
        )
    auxiliary_manifest: dict[str, object] | None = None
    if args.multilayer_aux_cache is not None:
        auxiliary_manifest = validate_v2_1_manifest(args.multilayer_aux_cache)
        if (
            auxiliary_manifest.get("schema_version") != 7
            or auxiliary_manifest.get("qwen_context_mode")
            != "multilayer_spatial_semantic"
        ):
            raise SystemExit("V3.24 auxiliary cache must contain four-layer 4x4 features")
        for key in (
            "dataset_root",
            "seed",
            "split",
            "validation_fraction",
            "horizon",
            "selected_frames",
            "state_mean",
            "state_std",
            "action_mean",
            "action_std",
        ):
            if auxiliary_manifest.get(key) != manifest.get(key):
                raise SystemExit(f"V3.24 primary/auxiliary cache mismatch for {key}")
    action_sidecar = None
    if args.v2_3_long_horizon_flow:
        if args.action_sidecar is None:
            raise SystemExit("V2.3 long-horizon Flow requires --action-sidecar")
        action_sidecar = load_long_horizon_action_sidecar(
            args.action_sidecar, cache_manifest=manifest
        )
    elif args.action_sidecar is not None:
        raise SystemExit("--action-sidecar is reserved for V2.3 long-horizon Flow")
    if args.validation_cache is not None:
        validation_manifest = validate_v2_1_manifest(args.validation_cache)
        train_signature = manifest.get("cache_signature", {})
        validation_signature = validation_manifest.get("cache_signature", {})
        for key in ("seed", "validation_fraction", "state_mean", "state_std", "action_mean", "action_std"):
            if not isinstance(train_signature, dict) or not isinstance(validation_signature, dict) or train_signature.get(key) != validation_signature.get(key):
                raise SystemExit(f"train/validation cache mismatch for {key}")
        if args.validation_multilayer_aux_cache is not None:
            validation_auxiliary_manifest = validate_v2_1_manifest(
                args.validation_multilayer_aux_cache
            )
            if (
                validation_auxiliary_manifest.get("schema_version") != 7
                or validation_auxiliary_manifest.get("split") != "validation"
            ):
                raise SystemExit("V3.24 validation auxiliary cache is invalid")
            for key in (
                "dataset_root",
                "seed",
                "split",
                "validation_fraction",
                "horizon",
                "selected_frames",
                "state_mean",
                "state_std",
                "action_mean",
                "action_std",
            ):
                if validation_auxiliary_manifest.get(key) != validation_manifest.get(key):
                    raise SystemExit(
                        f"V3.24 validation primary/auxiliary mismatch for {key}"
                    )
    config: DuvlaV21Config | DuvlaV321Config | DuvlaV322Config
    if args.v3_22_prior_residual_flow:
        parent_payload = torch.load(
            args.parent_checkpoint, map_location="cpu", weights_only=True
        )
        if (parent_payload.get("format"), parent_payload.get("version")) != (
            "duvla_v3_21",
            "V3.21",
        ):
            raise SystemExit("V3.22 parent must be a Duvla V3.21 checkpoint")
        parent_config_raw = parent_payload.get("model_config")
        if not isinstance(parent_config_raw, dict):
            raise SystemExit("V3.22 parent lacks model_config")
        prior_config = DuvlaV321Config(**parent_config_raw)
        config = DuvlaV322Config(
            prior=prior_config,
            residual_hidden_dim=(96 if args.smoke_model else prior_config.hidden_dim),
            residual_layers=(2 if args.smoke_model else 12),
            residual_heads=(4 if args.smoke_model else 12),
            flow_steps=(2 if args.smoke_model else 10),
            flow_samples=(2 if args.smoke_model else 5),
        )
        if args.smoke_model and prior_config.hidden_dim != 96:
            raise SystemExit("V3.22 smoke-model requires a smoke-sized V3.21 parent")
        policy = DuvlaV322Policy(config)
        policy.prior.load_state_dict(parent_payload["model_state_dict"], strict=True)
    elif args.v3_21_parallel_action_query:
        if int(manifest["horizon"]) != 8:
            raise SystemExit("V3.21 requires the schema-8 horizon-8 cache")
        config = _v3_21_model_config(manifest, smoke=args.smoke_model)
        policy: DuvlaV21Policy | DuvlaV321Policy | DuvlaV322Policy = DuvlaV321Policy(config)
    else:
        config = _model_config(
            manifest,
            smoke=args.smoke_model,
            r1b_action_coupling=args.r1b_action_coupling,
            r1c_causal_gripper=args.r1c_causal_gripper,
            r1d_event_gripper=args.r1d_event_gripper,
            r1e_event_recalibration=args.r1e_event_recalibration,
            v2_2_unified_flow=args.v2_2_unified_flow,
            v2_3_long_horizon_flow=args.v2_3_long_horizon_flow,
            v2_4_execution_aligned_flow=args.v2_4_execution_aligned_flow,
            v2_5_highres_flow=args.v2_5_highres_flow,
            v2_6_dense_interleaved_flow=args.v2_6_dense_interleaved_flow,
            v2_7_stable_gripper_flow=args.v2_7_stable_gripper_flow,
            v3_19_update_sufficient_flow=args.v3_19_update_sufficient_flow,
            v3_23_multilayer_spatial_flow=args.v3_23_multilayer_spatial_flow,
            v3_24_highres_multilayer_residual=args.v3_24_highres_multilayer_residual,
            v3_27_official2000_flow=args.v3_27_official2000_flow,
            v3_28_fp32_amp_flow=args.v3_28_fp32_amp_flow,
            action_horizon=(
                action_sidecar.horizon
                if action_sidecar is not None
                else int(manifest["horizon"])
            ),
        )
        policy = DuvlaV21Policy(config)
    fresh_gripper_state = (
        copy.deepcopy(policy.action_conditioned_gripper.state_dict())
        if args.r1e_event_recalibration and policy.action_conditioned_gripper is not None
        else None
    )
    if args.parent_checkpoint is not None and not args.v3_22_prior_residual_flow:
        checkpoint = torch.load(args.parent_checkpoint, map_location="cpu", weights_only=True)
        if args.v3_24_highres_multilayer_residual:
            if (checkpoint.get("format"), checkpoint.get("version")) != (
                "duvla_v2_6",
                "V2.6",
            ):
                raise SystemExit("V3.24 parent must be the audited V2.6 checkpoint")
            missing, unexpected = policy.load_state_dict(
                checkpoint["model_state_dict"], strict=False
            )
            expected_missing = {
                name
                for name in policy.state_dict()
                if name.startswith("fusion.multilayer_spatial_residual.")
            }
            if set(missing) != expected_missing or unexpected:
                raise SystemExit(
                    "V3.24 parent transfer changed parameters outside the new residual"
                )
        else:
            _version, expected_format, _artifact_version = _candidate_identity(args)
            if checkpoint.get("format") != expected_format:
                raise SystemExit(f"parent checkpoint is not {expected_format}")
            policy.load_state_dict(checkpoint["model_state_dict"], strict=True)
        if fresh_gripper_state is not None:
            if policy.action_conditioned_gripper is None:  # pragma: no cover - config contract
                raise RuntimeError("R1e gripper head disappeared")
            policy.action_conditioned_gripper.load_state_dict(fresh_gripper_state, strict=True)
    trainable_groups = _configure_stage(policy, args.stage)
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    parameter_dtype = (
        torch.float32 if args.v3_28_fp32_amp_flow else dtype
    )
    policy.to(device=device, dtype=parameter_dtype)
    trainable_parameters = [parameter for parameter in policy.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        fused=device.type == "cuda",
    )
    if args.v3_28_fp32_amp_flow:
        _assert_fp32_training_state(trainable_parameters, optimizer)
    selected_frames = int(manifest["selected_frames"])
    planned_samples = selected_frames if args.max_train_samples == 0 else min(
        selected_frames, args.max_train_samples
    )
    micro_steps_per_epoch = math.ceil(planned_samples / args.micro_batch_size)
    optimizer_steps_per_epoch = math.ceil(
        micro_steps_per_epoch / args.gradient_accumulation_steps
    )
    total_optimizer_steps = optimizer_steps_per_epoch * args.epochs
    warmup_steps = max(1, round(total_optimizer_steps * args.warmup_ratio))

    def lr_factor(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(total_optimizer_steps - warmup_steps, 1)
        return 0.5 * (1.0 + math.cos(math.pi * min(max(progress, 0.0), 1.0)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_factor)
    parent_store = _load_parent_actions(
        args.parent_actions,
        cache_signature=manifest.get("cache_signature"),
    )
    weights_by_task = _task_weights(manifest)
    milestones = {int(value) for value in args.milestone_epochs.split(",") if value.strip()}
    args.output.mkdir(parents=True, exist_ok=True)
    version, _checkpoint_format, artifact_version = _candidate_identity(args)
    run_signature = {
        "version": version,
        "stage": args.stage,
        "variant": args.variant,
        "r1b_action_coupling": args.r1b_action_coupling,
        "r1c_causal_gripper": args.r1c_causal_gripper,
        "r1d_event_gripper": args.r1d_event_gripper,
        "r1e_event_recalibration": args.r1e_event_recalibration,
        "v2_2_unified_flow": args.v2_2_unified_flow,
        "v2_3_long_horizon_flow": args.v2_3_long_horizon_flow,
        "v2_4_execution_aligned_flow": args.v2_4_execution_aligned_flow,
        "v2_5_highres_flow": args.v2_5_highres_flow,
        "v2_6_dense_interleaved_flow": args.v2_6_dense_interleaved_flow,
        "v2_7_stable_gripper_flow": args.v2_7_stable_gripper_flow,
        "v3_19_update_sufficient_flow": args.v3_19_update_sufficient_flow,
        "v3_21_parallel_action_query": args.v3_21_parallel_action_query,
        "v3_22_prior_residual_flow": args.v3_22_prior_residual_flow,
        "v3_23_multilayer_spatial_flow": args.v3_23_multilayer_spatial_flow,
        "v3_24_highres_multilayer_residual": args.v3_24_highres_multilayer_residual,
        "v3_27_official2000_flow": args.v3_27_official2000_flow,
        "v3_28_fp32_amp_flow": args.v3_28_fp32_amp_flow,
        "training_precision": (
            "fp32_parameters_adamw_bf16_autocast_fp32_loss"
            if args.v3_28_fp32_amp_flow
            else "full_bfloat16_cuda"
        ),
        "epochs": args.epochs,
        "micro_batch_size": args.micro_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "prefetch_batches": args.prefetch_batches,
        "pin_memory": device.type == "cuda" and args.prefetch_batches > 0,
        "shard_shuffle_block_size": args.shard_shuffle_block_size,
        "mmap_shards": args.mmap_shards,
        "metrics_every": args.metrics_every,
        "resume_every": args.resume_every,
        "effective_batch_size": args.micro_batch_size * args.gradient_accumulation_steps,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "seed": args.seed,
        "train_cache": str(args.train_cache.resolve()),
        "validation_cache": str(args.validation_cache.resolve()) if args.validation_cache else None,
        "multilayer_aux_cache": (
            str(args.multilayer_aux_cache.resolve()) if args.multilayer_aux_cache else None
        ),
        "validation_multilayer_aux_cache": (
            str(args.validation_multilayer_aux_cache.resolve())
            if args.validation_multilayer_aux_cache
            else None
        ),
        "parent_checkpoint": str(args.parent_checkpoint.resolve()) if args.parent_checkpoint else None,
        "parent_actions": str(args.parent_actions.resolve()) if args.parent_actions else None,
        "action_sidecar": str(args.action_sidecar.resolve()) if args.action_sidecar else None,
        "model_config": asdict(config),
        "trainable_groups": trainable_groups,
        "benchmark_task_index": False,
        "parameter_dtype": str(parameter_dtype).removeprefix("torch."),
        "autocast_dtype": (
            "bfloat16" if args.v3_28_fp32_amp_flow and device.type == "cuda" else None
        ),
        "loss_reduction_dtype": (
            "float32" if args.v3_28_fp32_amp_flow else str(dtype).removeprefix("torch.")
        ),
    }
    state_path = args.output / "training_state.json"
    curve_path = args.output / "loss_curve.jsonl"
    if state_path.exists() and not args.resume:
        raise SystemExit(f"output already contains training state: {state_path}; use --resume")
    start_epoch = 0
    resume_samples_in_epoch = 0
    resume_running: dict[str, float] = {}
    optimizer_step = 0
    effective_samples = 0
    if args.resume:
        resume_path = args.output / "resume.pt"
        if not resume_path.is_file():
            raise SystemExit("--resume requested but resume.pt is missing")
        resume = torch.load(resume_path, map_location=device, weights_only=True)
        if resume.get("run_signature") != run_signature:
            raise SystemExit("resume run signature mismatch")
        policy.load_state_dict(resume["model_state_dict"], strict=True)
        optimizer.load_state_dict(resume["optimizer_state_dict"])
        scheduler.load_state_dict(resume["scheduler_state_dict"])
        start_epoch = int(resume.get("epoch_index", resume["epoch"]))
        resume_samples_in_epoch = int(resume.get("samples_in_epoch", 0))
        resume_running = {
            str(key): float(value)
            for key, value in dict(resume.get("running_losses", {})).items()
        }
        optimizer_step = int(resume["optimizer_step"])
        effective_samples = int(resume["effective_samples"])
        truncate_loss_points(curve_path, max_optimizer_step=optimizer_step)

    counts = policy.parameter_counts()
    counts["trainable"] = sum(parameter.numel() for parameter in trainable_parameters)
    atomic_write_json(
        args.output / "run_manifest.json",
        {
            **run_signature,
            "parameter_counts": counts,
            "selected_frames": selected_frames,
            "planned_optimizer_steps": total_optimizer_steps,
            "minimum_formal_epochs": 6,
            "formal": args.epochs >= 6 and args.max_train_samples == 0 and not args.smoke_model,
            "started_at": datetime.now(timezone.utc).isoformat(),
        },
    )
    log_path = args.output / "metrics.jsonl"
    validation_log_path = args.output / "validation_metrics.jsonl"
    best_validation_total = float("inf")

    def save_resume(
        *,
        epoch_index: int,
        samples_in_epoch: int,
        running_losses: Counter[str],
    ) -> None:
        _atomic_torch_save(
            args.output / "resume.pt",
            {
                "run_signature": run_signature,
                "epoch": epoch_index,
                "epoch_index": epoch_index,
                "samples_in_epoch": samples_in_epoch,
                "running_losses": dict(running_losses),
                "optimizer_step": optimizer_step,
                "effective_samples": effective_samples,
                "model_state_dict": policy.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
            },
        )

    optimizer.zero_grad(set_to_none=True)
    for epoch in range(start_epoch, args.epochs):
        policy.train()
        task_exposure: Counter[int] = Counter()
        unique_rows: set[int] = set()
        unique_episodes: set[int] = set()
        accumulation = 0
        samples_this_epoch = 0
        curve_samples = 0
        curve_running: Counter[str] = Counter()
        running: Counter[str] = Counter(
            resume_running if epoch == start_epoch else {}
        )
        skip_samples = resume_samples_in_epoch if epoch == start_epoch else 0
        raw_batches = iter_v2_1_batches(
            args.train_cache,
            batch_size=args.micro_batch_size,
            history_length=config.history_length,
            seed=args.seed,
            epoch=epoch,
            shuffle=True,
            shard_shuffle_block_size=args.shard_shuffle_block_size,
            mmap_shards=args.mmap_shards,
            history_stride=config.history_stride,
            action_sidecar=action_sidecar,
        )
        if args.multilayer_aux_cache is not None:
            auxiliary_batches = iter_v2_1_batches(
                args.multilayer_aux_cache,
                batch_size=args.micro_batch_size,
                history_length=config.history_length,
                history_stride=config.history_stride,
                seed=args.seed,
                epoch=epoch,
                shuffle=True,
                shard_shuffle_block_size=args.shard_shuffle_block_size,
                mmap_shards=args.mmap_shards,
            )
            raw_batches = _aligned_auxiliary_batches(raw_batches, auxiliary_batches)
        batches = (
            prefetch_v2_1_batches(raw_batches, pin_memory=device.type == "cuda")
            if args.prefetch_batches
            else raw_batches
        )
        try:
            for batch in batches:
                if args.max_train_samples and samples_this_epoch >= args.max_train_samples:
                    break
                if args.max_train_samples and samples_this_epoch + batch["visual"].shape[0] > args.max_train_samples:
                    keep = args.max_train_samples - samples_this_epoch
                    batch = {key: value[:keep] for key, value in batch.items()}
                cpu_indices = batch["dataset_indices"].clone()
                cpu_tasks = batch["task_indices"].clone()
                cpu_episodes = batch["episode_indices"].clone()
                batch_size = int(cpu_indices.shape[0])
                if samples_this_epoch < skip_samples:
                    if samples_this_epoch + batch_size > skip_samples:
                        raise RuntimeError("resume position is not aligned to a deterministic batch boundary")
                    samples_this_epoch += batch_size
                    unique_rows.update(int(value) for value in cpu_indices)
                    unique_episodes.update(int(value) for value in cpu_episodes)
                    task_exposure.update(int(value) for value in cpu_tasks)
                    continue
                parent = _parent_batch(parent_store, cpu_indices)
                moved = _move(batch, device, dtype)
                parent_device = None if parent is None else parent.to(device=device, dtype=dtype)
                if args.stage == "joint":
                    joint_noise = deterministic_flow_noise(
                        cpu_indices,
                        seed=args.seed,
                        samples=config.flow_samples,
                        horizon=config.action_horizon,
                        action_dim=config.action_dim,
                    ).to(device=device, dtype=dtype)
                    with torch.no_grad():
                        parent_device = policy.sample_actions(
                            moved["visual"],
                            moved["semantic"],
                            moved["states"],
                            history_visual=moved.get("history_visual"),
                            history_semantic=moved.get("history_semantic"),
                            history_states=moved.get("history_states"),
                            previous_action=moved.get("previous_actions"),
                            history_previous_actions=moved.get(
                                "history_previous_actions"
                            ),
                            layer14_only=args.variant == "layer14",
                            flow_samples=config.flow_samples,
                            flow_steps=config.flow_steps,
                            apply_direct=False,
                            apply_instruction=False,
                            apply_gripper_event=False,
                            noise=joint_noise,
                        )
                sample_weights = weights_by_task.index_select(0, cpu_tasks).to(device=device, dtype=dtype)
                with torch.autocast(
                    device_type=device.type,
                    dtype=torch.bfloat16,
                    enabled=args.v3_28_fp32_amp_flow and device.type == "cuda",
                ):
                    losses = policy.flow_loss_components(
                        moved["visual"],
                        moved["semantic"],
                        moved["states"],
                        moved["actions"],
                        moved["valid_mask"],
                        auxiliary_visual=moved.get("auxiliary_visual"),
                        history_visual=moved.get("history_visual"),
                        history_auxiliary_visual=moved.get("history_auxiliary_visual"),
                        history_semantic=moved.get("history_semantic"),
                        history_states=moved.get("history_states"),
                        previous_actions=moved.get("previous_actions"),
                        history_previous_actions=moved.get(
                            "history_previous_actions"
                        ),
                        layer14_only=args.variant == "layer14",
                        parent_actions=parent_device,
                        stage=args.stage,
                        sample_weights=sample_weights,
                    )
                if args.v3_28_fp32_amp_flow and any(
                    value.dtype != torch.float32 for value in losses.values()
                ):
                    raise RuntimeError("V3.28 losses must be reduced in FP32")
                total_loss = sum(losses.values()) / args.gradient_accumulation_steps
                total_loss.backward()
                accumulation += 1
                samples_this_epoch += batch_size
                effective_samples += batch_size
                unique_rows.update(int(value) for value in cpu_indices)
                unique_episodes.update(int(value) for value in cpu_episodes)
                task_exposure.update(int(value) for value in cpu_tasks)
                for name, value in losses.items():
                    weighted_value = float(value.detach()) * batch_size
                    running[name] += weighted_value
                    curve_running[name] += weighted_value
                curve_samples += batch_size
                should_step = accumulation == args.gradient_accumulation_steps
                if should_step:
                    torch.nn.utils.clip_grad_norm_(trainable_parameters, args.max_grad_norm)
                    optimizer.step()
                    if args.v3_28_fp32_amp_flow and optimizer_step == 0:
                        _assert_fp32_training_state(trainable_parameters, optimizer)
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
                    accumulation = 0
                    optimizer_step += 1
                    if optimizer_step % args.metrics_every == 0:
                        append_loss_point(
                            curve_path,
                            {
                                "optimizer_step": optimizer_step,
                                "effective_epochs": effective_samples / selected_frames,
                                "learning_rate": optimizer.param_groups[0]["lr"],
                                "window_samples": curve_samples,
                                "losses": {
                                    name: value / max(curve_samples, 1)
                                    for name, value in curve_running.items()
                                },
                                "at": datetime.now(timezone.utc).isoformat(),
                            },
                        )
                        curve_running.clear()
                        curve_samples = 0
                    if optimizer_step % args.log_every == 0:
                        record = {
                            "type": "optimizer_progress",
                            "optimizer_step": optimizer_step,
                            "planned_optimizer_steps": total_optimizer_steps,
                            "effective_samples": effective_samples,
                            "effective_epochs": effective_samples / selected_frames,
                            "learning_rate": optimizer.param_groups[0]["lr"],
                            "losses": {name: value / max(samples_this_epoch, 1) for name, value in running.items()},
                            "peak_vram_gib": (
                                torch.cuda.max_memory_allocated() / 1024**3 if device.type == "cuda" else 0.0
                            ),
                            "at": datetime.now(timezone.utc).isoformat(),
                        }
                        with log_path.open("a") as handle:
                            handle.write(json.dumps(record) + "\n")
                        print(json.dumps(record), flush=True)
                    if optimizer_step % args.resume_every == 0:
                        save_resume(
                            epoch_index=epoch,
                            samples_in_epoch=samples_this_epoch,
                            running_losses=running,
                        )
                        atomic_write_json(
                            state_path,
                            {
                                "state": "running",
                                "epoch": epoch,
                                "epochs": args.epochs,
                                "optimizer_step": optimizer_step,
                                "effective_samples": effective_samples,
                                "effective_epochs": effective_samples / selected_frames,
                                "updated_at": datetime.now(timezone.utc).isoformat(),
                            },
                        )
        finally:
            close = getattr(batches, "close", None)
            if close is not None:
                close()
        if accumulation:
            _rescale_partial_accumulation(
                trainable_parameters,
                completed_micro_steps=accumulation,
                target_micro_steps=args.gradient_accumulation_steps,
            )
            torch.nn.utils.clip_grad_norm_(trainable_parameters, args.max_grad_norm)
            optimizer.step()
            if args.v3_28_fp32_amp_flow and optimizer_step == 0:
                _assert_fp32_training_state(trainable_parameters, optimizer)
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            optimizer_step += 1
        if curve_samples:
            append_loss_point(
                curve_path,
                {
                    "optimizer_step": optimizer_step,
                    "effective_epochs": effective_samples / selected_frames,
                    "learning_rate": optimizer.param_groups[0]["lr"],
                    "window_samples": curve_samples,
                    "losses": {
                        name: value / max(curve_samples, 1)
                        for name, value in curve_running.items()
                    },
                    "at": datetime.now(timezone.utc).isoformat(),
                },
            )
        epoch_number = epoch + 1
        epoch_record: dict[str, Any] = {
            "type": "epoch_complete",
            "epoch": epoch_number,
            "optimizer_step": optimizer_step,
            "effective_samples": effective_samples,
            "effective_epochs": effective_samples / selected_frames,
            "unique_rows": len(unique_rows),
            "unique_row_coverage": len(unique_rows) / selected_frames,
            "unique_episodes": len(unique_episodes),
            "task_exposure": dict(sorted(task_exposure.items())),
            "losses": {name: value / max(samples_this_epoch, 1) for name, value in running.items()},
            "peak_vram_gib": (
                torch.cuda.max_memory_allocated() / 1024**3 if device.type == "cuda" else 0.0
            ),
            "at": datetime.now(timezone.utc).isoformat(),
        }
        with log_path.open("a") as handle:
            handle.write(json.dumps(epoch_record) + "\n")
        print(json.dumps(epoch_record), flush=True)
        checkpoint_epoch = epoch_number in milestones or epoch_number == args.epochs
        validation_losses: dict[str, float] | None = None
        if (
            checkpoint_epoch
            and isinstance(policy, DuvlaV321Policy)
            and args.validation_cache is not None
        ):
            validation_losses = _validate_v3_21(
                policy,
                args.validation_cache,
                config=config,
                device=device,
                dtype=dtype,
                batch_size=args.micro_batch_size,
                weights_by_task=weights_by_task,
            )
            validation_record = {
                "type": "validation",
                "epoch": epoch_number,
                "optimizer_step": optimizer_step,
                "losses": validation_losses,
                "at": datetime.now(timezone.utc).isoformat(),
            }
            with validation_log_path.open("a") as handle:
                handle.write(json.dumps(validation_record) + "\n")
            print(json.dumps(validation_record), flush=True)
        checkpoint_payload = _checkpoint_payload(
            policy,
            config,
            args=args,
            epoch=epoch_number,
            optimizer_step=optimizer_step,
            effective_samples=effective_samples,
            trainable_groups=trainable_groups,
            manifest=manifest,
            validation_losses=validation_losses,
        )
        if checkpoint_epoch:
            _atomic_torch_save(
                args.output
                / f"duvla-{artifact_version}-{args.variant}-{args.stage}-{epoch_number}e.pt",
                checkpoint_payload,
            )
            if (
                validation_losses is not None
                and validation_losses["total"] < best_validation_total
            ):
                best_validation_total = validation_losses["total"]
                _atomic_torch_save(args.output / "best-validation.pt", checkpoint_payload)
        curve_title = (
            f"Duvla {version} "
            f"{args.variant} {args.stage} loss through {epoch_number}E"
        )
        write_loss_curve_artifacts(
            curve_path,
            csv_path=args.output / "loss_curve.csv",
            svg_path=args.output / "loss_curve.svg",
            title=curve_title,
        )
        if checkpoint_epoch:
            write_loss_curve_artifacts(
                curve_path,
                csv_path=args.output / f"loss_curve_{epoch_number}e.csv",
                svg_path=args.output / f"loss_curve_{epoch_number}e.svg",
                title=curve_title,
            )
        save_resume(
            epoch_index=epoch_number,
            samples_in_epoch=0,
            running_losses=Counter(),
        )
        atomic_write_json(
            state_path,
            {
                "state": "completed" if epoch_number == args.epochs else "running",
                "epoch": epoch_number,
                "epochs": args.epochs,
                "optimizer_step": optimizer_step,
                "effective_samples": effective_samples,
                "effective_epochs": effective_samples / selected_frames,
                "updated_at": datetime.now(timezone.utc).isoformat(),
            },
        )
        resume_samples_in_epoch = 0
        resume_running = {}


if __name__ == "__main__":
    main()
