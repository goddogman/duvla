#!/usr/bin/env python3
"""Natural-language LIBERO evaluator with DuVLA V3.31 public defaults."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import random

os.environ.setdefault("MUJOCO_GL", "osmesa")

import numpy as np
import torch
from PIL import Image

from duvla.evaluation.flow_contract import StandardizationStats, flow_normalized_action_to_env
from duvla.evaluation.libero_contract import (
    libero_dummy_action,
    libero_standard_max_steps,
    observation_to_state,
    orient_libero_view,
)
from duvla.evaluation.libero_rollout import compose_views
from duvla.evaluation.initial_states import load_libero_initial_states
from duvla.data import file_sha256
from duvla.models.duvla_v2_1 import (
    DuvlaV21Config,
    DuvlaV21Policy,
    pool_multilayer_spatial_grid,
)
from duvla.models.duvla_v3_21 import DuvlaV321Config, DuvlaV321Policy
from duvla.models.duvla_v3_22 import DuvlaV322Config, DuvlaV322Policy
from duvla.models.duvla_v3_26 import DuvlaV326Config, DuvlaV326Policy
from duvla.evaluation.execution_feedback import ExecutionFeedbackTracker


BASELINE_FALLBACK_VERIFIER_FORMATS = frozenset(
    {
        "duvla_v2_9_outcome_verifier",
        "duvla_v2_10_outcome_verifier",
        "duvla_v3_1_p3_outcome_verifier",
        "duvla_v3_1_p3b_multihorizon_verifier",
        "duvla_v3_1_p3c_relational_verifier",
        "duvla_v3_1_p3d_grounded_verifier",
        "duvla_v3_14_outcome_ranker",
        "duvla_v3_15_outcome_ranker",
        "duvla_v3_16_outcome_ranker",
        "duvla_v3_17_outcome_ranker",
        "duvla_v3_18_outcome_ensemble",
        "duvla_v3_20_outcome_ranker",
    }
)
from duvla.models.action_intent import (
    ActionIntentConfig,
    StructuredActionIntentReasoner,
)
from duvla.models.duvla_pwr import (
    DuvlaV31AnchoredPolicy,
    DuvlaV31AnchoredPolicyConfig,
    DuvlaV31Policy,
    DuvlaV31PolicyConfig,
    PWRActionExpertConfig,
    PWRPlannerConfig,
    PWRProgressiveResidualConfig,
)
from duvla.models.outcome_verifier import (
    ActionConditionedOutcomeEnsemble,
    ActionConditionedOutcomeVerifier,
    OutcomeVerifierConfig,
    arrange_v2_policy_context,
    select_with_baseline_fallback,
)
from duvla.models.qwen_backbone import QwenBackboneConfig, QwenVLBackbone
from duvla.models.qwen_lora import QwenLoraSpec, attach_qwen_lora


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--outcome-verifier",
        type=Path,
        help="optional task-index-free V2.8/V2.9 physical outcome verifier",
    )
    parser.add_argument(
        "--action-intent",
        type=Path,
        help="formal V3.13 structured candidate generator required by V3.14+ rankers",
    )
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--suite", choices=("libero_10", "libero_spatial", "libero_object", "libero_goal"), required=True)
    parser.add_argument("--task-indices", default="0-9")
    parser.add_argument("--state-manifest", type=Path)
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--all-init-states", action="store_true")
    parser.add_argument("--init-state-start", type=int, default=0)
    parser.add_argument("--init-state-count", type=int)
    parser.add_argument(
        "--max-steps",
        type=int,
        help="episode budget; defaults to the standard budget for the selected suite",
    )
    parser.add_argument("--action-steps", type=int, default=2)
    parser.add_argument("--settle-steps", type=int, default=10)
    parser.add_argument("--camera-size", type=int, default=128)
    parser.add_argument("--v329-development-20e", action="store_true")
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--flow-seed", type=int, default=23)
    parser.add_argument("--flow-samples", type=int, default=5)
    parser.add_argument(
        "--candidate-aggregation",
        choices=("coordinate_median", "trajectory_medoid"),
        help="optional same-checkpoint aggregation ablation; defaults to checkpoint config",
    )
    parser.add_argument("--trace-dir", type=Path, required=True)
    parser.add_argument("--video-dir", type=Path)
    parser.add_argument("--save-video", action="store_true")
    parser.add_argument("--no-flip-views", action="store_true")
    parser.add_argument("--state-clip", type=float)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--stop-after-failures",
        type=int,
        help=(
            "stop cleanly after this suite accumulates more than N failures; "
            "the partial summary remains complete=false"
        ),
    )
    return parser.parse_args()


def _task_ids(spec: str, task_count: int) -> tuple[int, ...]:
    if spec.strip().casefold() == "all":
        return tuple(range(task_count))
    result: set[int] = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start, stop = (int(value) for value in part.split("-", 1))
            if stop < start:
                raise ValueError("task range must be ascending")
            result.update(range(start, stop + 1))
        else:
            result.add(int(part))
    if not result or min(result) < 0 or max(result) >= task_count:
        raise ValueError("task indices are outside the selected LIBERO suite")
    return tuple(sorted(result))


def _atomic_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def _load_state_manifest(
    path: Path,
    *,
    suite_name: str,
    task_ids: tuple[int, ...],
    task_count: int,
) -> tuple[dict[int, tuple[int, ...]], str]:
    payload = json.loads(path.read_text())
    if payload.get("schema_version") != 1:
        raise ValueError("state manifest schema_version must be 1")
    suites = payload.get("suites")
    if not isinstance(suites, dict) or not isinstance(suites.get(suite_name), dict):
        raise ValueError(f"state manifest has no suite {suite_name}")
    suite = suites[suite_name]
    result: dict[int, tuple[int, ...]] = {}
    for task_id in task_ids:
        values = suite.get(str(task_id))
        if not isinstance(values, list) or not values:
            raise ValueError(f"state manifest lacks {suite_name} task {task_id}")
        indices = tuple(int(value) for value in values)
        if len(indices) != len(set(indices)) or min(indices) < 0:
            raise ValueError("state manifest contains invalid or duplicate state ids")
        result[task_id] = indices
    if set(int(key) for key in suite) != set(range(task_count)):
        raise ValueError("state manifest must cover every task in the suite")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return result, digest


def _load_policy(
    checkpoint_path: Path,
    train_manifest: dict[str, object],
    device: torch.device,
) -> tuple[
    DuvlaV21Policy | DuvlaV321Policy | DuvlaV322Policy | DuvlaV326Policy | DuvlaV31Policy | DuvlaV31AnchoredPolicy,
    dict[str, object],
]:
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    identity = (payload.get("format"), payload.get("version"))
    if identity not in {
        ("duvla_v2_1", "V2.1"),
        ("duvla_v2_2", "V2.2"),
        ("duvla_v2_3", "V2.3"),
        ("duvla_v2_4", "V2.4"),
        ("duvla_v2_5", "V2.5"),
        ("duvla_v2_6", "V2.6"),
        ("duvla_v2_7", "V2.7"),
        ("duvla_v3_19", "V3.19"),
        ("duvla_v3_21", "V3.21"),
        ("duvla_v3_22", "V3.22"),
        ("duvla_v3_23", "V3.23"),
        ("duvla_v3_24", "V3.24"),
        ("duvla_v3_25", "V3.25"),
        ("duvla_v3_26", "V3.26"),
        ("duvla_v3_27", "V3.27"),
        ("duvla_v3_28", "V3.28"),
        ("duvla_v3_29", "V3.29"),
        ("duvla_v3_30", "V3.30"),
        ("duvla_v3_31", "V3.31"),
        ("duvla_v3_32", "V3.32"),
        ("duvla_v3_1", "V3.1-P1"),
        ("duvla_v3_1", "V3.1-P1b"),
        ("duvla_v3_1", "V3.1-P1c"),
        ("duvla_v3_1", "V3.1-P2D"),
        ("duvla_v3_1", "V3.1-P2P"),
    }:
        raise ValueError("checkpoint is not a supported Duvla V2 policy")
    if payload.get("benchmark_task_index") is not False or payload.get("task_routing") != "natural_language":
        raise ValueError("strict V2.1 evaluation rejects task-index-routed checkpoints")
    if payload.get("cache_signature") != train_manifest.get("cache_signature"):
        raise ValueError("checkpoint and train cache signatures differ")
    config_raw = payload.get("model_config")
    if not isinstance(config_raw, dict):
        raise ValueError("checkpoint lacks model_config")
    if identity == ("duvla_v3_26", "V3.26"):
        policy = DuvlaV326Policy(DuvlaV326Config(**config_raw))
    elif identity == ("duvla_v3_22", "V3.22"):
        prior_raw = config_raw.get("prior")
        if not isinstance(prior_raw, dict):
            raise ValueError("V3.22 checkpoint lacks nested prior config")
        scalar_config = {name: value for name, value in config_raw.items() if name != "prior"}
        policy = DuvlaV322Policy(
            DuvlaV322Config(prior=DuvlaV321Config(**prior_raw), **scalar_config)
        )
    elif identity == ("duvla_v3_21", "V3.21"):
        policy = DuvlaV321Policy(DuvlaV321Config(**config_raw))
    elif identity in {
        ("duvla_v3_1", "V3.1-P1"),
        ("duvla_v3_1", "V3.1-P1b"),
    }:
        planner_raw = config_raw.get("planner")
        expert_raw = config_raw.get("expert")
        if not isinstance(planner_raw, dict) or not isinstance(expert_raw, dict):
            raise ValueError("V3.1 checkpoint lacks nested planner/expert config")
        scalar_config = {
            name: value
            for name, value in config_raw.items()
            if name not in {"planner", "expert"}
        }
        policy = DuvlaV31Policy(
            DuvlaV31PolicyConfig(
                planner=PWRPlannerConfig(**planner_raw),
                expert=PWRActionExpertConfig(**expert_raw),
                **scalar_config,
            )
        )
    elif identity in {
        ("duvla_v3_1", "V3.1-P1c"),
        ("duvla_v3_1", "V3.1-P2D"),
        ("duvla_v3_1", "V3.1-P2P"),
    }:
        base_raw = config_raw.get("base")
        planner_raw = config_raw.get("planner")
        residual_raw = config_raw.get("residual")
        if not all(
            isinstance(value, dict)
            for value in (base_raw, planner_raw, residual_raw)
        ):
            raise ValueError("anchored V3.1 checkpoint lacks nested model configs")
        scalar_config = {
            name: value
            for name, value in config_raw.items()
            if name not in {"base", "planner", "residual"}
        }
        policy = DuvlaV31AnchoredPolicy(
            DuvlaV31AnchoredPolicyConfig(
                base=DuvlaV21Config(**base_raw),
                planner=PWRPlannerConfig(**planner_raw),
                residual=PWRProgressiveResidualConfig(**residual_raw),
                **scalar_config,
            )
        )
    else:
        policy = DuvlaV21Policy(DuvlaV21Config(**config_raw))
    policy.load_state_dict(payload["model_state_dict"], strict=True)
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    return policy.to(device=device, dtype=dtype).eval(), payload


def _load_outcome_verifier(
    path: Path,
    *,
    policy_checkpoint: Path,
    policy: DuvlaV21Policy,
    device: torch.device,
    action_intent_checkpoint: Path | None = None,
) -> tuple[
    ActionConditionedOutcomeVerifier | ActionConditionedOutcomeEnsemble,
    dict[str, object],
]:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    verifier_format = payload.get("format")
    if verifier_format not in {
        "duvla_v2_8_outcome_verifier",
        "duvla_v2_9_outcome_verifier",
        "duvla_v2_10_outcome_verifier",
        "duvla_v3_1_p3_outcome_verifier",
        "duvla_v3_1_p3b_multihorizon_verifier",
        "duvla_v3_1_p3c_relational_verifier",
        "duvla_v3_1_p3d_grounded_verifier",
        "duvla_v3_14_outcome_ranker",
        "duvla_v3_15_outcome_ranker",
        "duvla_v3_16_outcome_ranker",
        "duvla_v3_17_outcome_ranker",
        "duvla_v3_18_outcome_ensemble",
        "duvla_v3_20_outcome_ranker",
    }:
        raise ValueError("outcome verifier checkpoint is not a supported Duvla verifier")
    if payload.get("benchmark_task_index") is not False or payload.get("task_routing") != "natural_language":
        raise ValueError("strict evaluation rejects a routed outcome verifier")
    train_manifest = payload.get("train_manifest")
    if not isinstance(train_manifest, dict):
        raise ValueError("V2.8 verifier lacks its branch-cache manifest")
    structured_ranker_formats = {
        "duvla_v3_14_outcome_ranker",
        "duvla_v3_15_outcome_ranker",
        "duvla_v3_16_outcome_ranker",
        "duvla_v3_17_outcome_ranker",
        "duvla_v3_18_outcome_ensemble",
    }
    if verifier_format in structured_ranker_formats:
        if action_intent_checkpoint is None:
            raise ValueError("structured V3 rankers require their V3.13 action-intent checkpoint")
        if (
            train_manifest.get("candidate_generator_sha256")
            != file_sha256(action_intent_checkpoint)
            or train_manifest.get("candidate_generator_format")
            != "duvla_v3_action_intent_reasoner"
            or train_manifest.get("candidate_generator_version") != "V3.13"
            or train_manifest.get("candidate_parent_sha256")
            != file_sha256(policy_checkpoint)
        ):
            raise ValueError("structured V3 ranker candidate lineage is inconsistent")
    else:
        if train_manifest.get("candidate_generator_sha256") != file_sha256(policy_checkpoint):
            raise ValueError("V2.8 verifier was trained for a different parent policy")
        if train_manifest.get("candidate_generator_format") != "duvla_v2_6":
            raise ValueError("V2.8 verifier requires a V2.6 candidate generator")
    raw_config = payload.get("config")
    if not isinstance(raw_config, dict):
        raise ValueError("V2.8 verifier lacks model config")
    config = OutcomeVerifierConfig(**raw_config)
    expected_tokens = policy.config.spatial_tokens + (
        policy.config.context_tokens
        - policy.config.camera_count * policy.config.spatial_tokens
    )
    if (
        config.context_dim != policy.config.hidden_dim
        or config.action_horizon != policy.config.action_horizon
        or config.action_dim != policy.config.action_dim
        or train_manifest.get("context_tokens_per_camera") != expected_tokens
    ):
        raise ValueError("V2.8 verifier and parent policy tensor contracts differ")
    if verifier_format == "duvla_v3_18_outcome_ensemble":
        states = payload.get("member_model_state_dicts")
        member_count = payload.get("member_count")
        if not isinstance(states, list) or member_count != len(states):
            raise ValueError("V3.18 ensemble member metadata is incomplete")
        members: list[ActionConditionedOutcomeVerifier] = []
        for state in states:
            if not isinstance(state, dict):
                raise ValueError("V3.18 ensemble contains an invalid member state")
            member = ActionConditionedOutcomeVerifier(config)
            member.load_state_dict(state, strict=True)
            members.append(member)
        verifier = ActionConditionedOutcomeEnsemble(
            members,
            uncertainty_penalty=float(payload["uncertainty_penalty"]),
            minimum_consensus=int(payload["minimum_consensus"]),
        )
    else:
        verifier = ActionConditionedOutcomeVerifier(config)
        verifier.load_state_dict(payload["model_state_dict"], strict=True)
    return verifier.to(device=device).eval(), payload


def _load_action_intent(
    path: Path,
    *,
    policy_checkpoint: Path,
    policy: DuvlaV21Policy,
    device: torch.device,
) -> tuple[StructuredActionIntentReasoner, dict[str, object]]:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if (
        payload.get("format") != "duvla_v3_action_intent_reasoner"
        or payload.get("version") != "V3.13"
        or payload.get("task_routing") != "natural_language"
        or payload.get("benchmark_task_index") is not False
        or payload.get("uses_reward") is not False
        or payload.get("uses_success") is not False
        or payload.get("uses_evaluation_initial_states") is not False
        or payload.get("parent_checkpoint_sha256") != file_sha256(policy_checkpoint)
    ):
        raise ValueError("strict evaluation rejects an unsafe V3.13 checkpoint")
    raw_config = payload.get("model_config")
    state = payload.get("model_state_dict")
    if not isinstance(raw_config, dict) or not isinstance(state, dict):
        raise ValueError("V3.13 checkpoint lacks model config/state")
    codebook = state.get("codebook")
    if not isinstance(codebook, torch.Tensor):
        raise ValueError("V3.13 checkpoint lacks its action codebook")
    config = ActionIntentConfig(**raw_config)
    if (
        config.context_dim != policy.config.hidden_dim
        or config.state_dim != policy.config.state_dim
        or config.action_horizon != policy.config.action_horizon
        or config.action_dim != policy.config.action_dim
        or config.executed_steps != policy.config.replan_action_steps
    ):
        raise ValueError("V3.13 and V2.6 tensor contracts differ")
    reasoner = StructuredActionIntentReasoner(config, codebook.float())
    reasoner.load_state_dict(state, strict=True)
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    return reasoner.to(device=device, dtype=dtype).eval(), payload


def _episode_complete(path: Path, *, task_id: int, episode: int) -> dict[str, object] | None:
    if not path.is_file():
        return None
    payload = json.loads(path.read_text())
    summary = payload.get("summary")
    if not isinstance(summary, dict):
        return None
    if summary.get("task_id") != task_id or summary.get("episode") != episode:
        return None
    return summary


def _failure_budget_exhausted(
    *, episode_count: int, success_count: int, maximum_failures: int | None
) -> bool:
    if episode_count < 0 or success_count < 0 or success_count > episode_count:
        raise ValueError("episode/success counts are inconsistent")
    if maximum_failures is None:
        return False
    if maximum_failures < 0:
        raise ValueError("maximum_failures must be non-negative")
    return episode_count - success_count > maximum_failures


def _checkpoint_stage(checkpoint: dict[str, object]) -> str:
    """Resolve the deployed action stage without weakening legacy contracts."""
    if checkpoint.get("format") in {"duvla_v3_25", "duvla_v3_26"}:
        stage = checkpoint.get("stage", "flow")
    else:
        if "stage" not in checkpoint:
            raise ValueError("checkpoint lacks its action stage")
        stage = checkpoint["stage"]
    value = str(stage)
    if value not in {"flow", "direct", "instruction", "joint"}:
        raise ValueError(f"unsupported checkpoint action stage: {value}")
    return value


def _validate_v329_deployment(checkpoint: dict[str, object], args: argparse.Namespace) -> None:
    if checkpoint.get('format') != 'duvla_v3_29':
        return
    expected_epoch = 20 if args.v329_development_20e else 30
    if (checkpoint.get('formal') is not True or checkpoint.get('checkpoint_complete') is not True
            or checkpoint.get('epoch') != expected_epoch
            or float(checkpoint.get('effective_epochs',0)) < expected_epoch
            or checkpoint.get('planned_epochs') != 30 or checkpoint.get('action_offset') != 1):
        raise ValueError('V3.29 requires a completed 20E development or 30E formal checkpoint')
    if expected_epoch==30 and checkpoint.get('training_complete') is not True:
        raise ValueError('V3.29 30E training is incomplete')
    if any(checkpoint.get(k) is not False for k in ('uses_reward','uses_success','uses_evaluation_initial_states')):
        raise ValueError('V3.29 provenance missing')
    contract = checkpoint.get('data_contract',{})
    if not contract.get('complete') or contract.get('train_demonstrations') != 2000 or not checkpoint.get('sidecar_sha256'):
        raise ValueError('V3.29 full training sidecar not verified')
    if (args.camera_size,args.fps,args.action_steps,args.flow_samples,args.flow_seed) != (128,20,2,5,23):
        raise ValueError('V3.29 requires 128 camera / 20Hz / K5 / action2 / seed23')
    if (args.no_flip_views or args.state_clip is not None or args.outcome_verifier is not None
            or args.action_intent is not None or getattr(args, 'candidate_aggregation', None) is not None):
        raise ValueError('V3.29 cannot override the frozen observation/action contract')


def _validate_v330_deployment(checkpoint: dict[str, object], args: argparse.Namespace) -> None:
    if checkpoint.get('format') != 'duvla_v3_30':
        return
    if (checkpoint.get('formal') is not True or checkpoint.get('checkpoint_complete') is not True
            or checkpoint.get('epoch') != 30 or float(checkpoint.get('effective_epochs', 0)) < 30
            or checkpoint.get('planned_epochs') != 30 or checkpoint.get('training_complete') is not True
            or checkpoint.get('action_offset') != 1):
        raise ValueError('V3.30 requires a completed 30E formal checkpoint')
    if any(checkpoint.get(k) is not False for k in ('uses_reward','uses_success','uses_evaluation_initial_states')):
        raise ValueError('V3.30 provenance missing')
    contract = checkpoint.get('data_contract', {})
    config = checkpoint.get('model_config', {})
    if (not contract.get('complete') or contract.get('train_demonstrations') != 2000
            or not checkpoint.get('sidecar_sha256')):
        raise ValueError('V3.30 full training sidecar not verified')
    if (config.get('causal_action_attention') is not False
            or config.get('ordered_language_bridge_mode') != 'zero_output'
            or config.get('candidate_aggregation') != 'trajectory_medoid'):
        raise ValueError('V3.30 architecture contract mismatch')
    if (args.camera_size,args.fps,args.action_steps,args.flow_samples,args.flow_seed) != (128,20,2,5,23):
        raise ValueError('V3.30 requires 128 camera / 20Hz / K5 / action2 / seed23')
    if args.no_flip_views or args.state_clip is not None or args.outcome_verifier is not None or args.action_intent is not None:
        raise ValueError('V3.30 cannot override the frozen observation/action contract')


def _validate_v331_deployment(checkpoint: dict[str, object], args: argparse.Namespace) -> None:
    if checkpoint.get("format") != "duvla_v3_31":
        return
    config = checkpoint.get("model_config", {})
    augmentation = checkpoint.get("augmentation_contract", {})
    if (
        checkpoint.get("formal") is not True
        or checkpoint.get("checkpoint_complete") is not True
        or checkpoint.get("training_complete") is not True
        or checkpoint.get("epoch") != 30
        or float(checkpoint.get("effective_epochs", 0)) < 30
        or checkpoint.get("planned_epochs") != 30
        or checkpoint.get("action_offset") != 1
    ):
        raise ValueError("V3.31 requires its completed formal 30E checkpoint")
    if any(
        checkpoint.get(name) is not False
        for name in ("uses_reward", "uses_success", "uses_evaluation_initial_states")
    ):
        raise ValueError("V3.31 provenance is incomplete")
    if (
        not isinstance(config, dict)
        or config.get("ordered_language_bridge") is not True
        or int(config.get("language_max_tokens", 0)) != 256
        or config.get("cross_camera_fusion") is not True
        or config.get("causal_action_attention") is not True
        or config.get("candidate_aggregation") != "coordinate_median"
    ):
        raise ValueError("V3.31 architecture contract mismatch")
    if (
        not isinstance(augmentation, dict)
        or augmentation.get("formal") is not True
        or augmentation.get("complete") is not True
        or int(augmentation.get("selected_rows", 0)) != 10000
        or not checkpoint.get("augmentation_sidecar_sha256")
    ):
        raise ValueError("V3.31 formal augmentation contract mismatch")
    if (args.camera_size, args.fps, args.action_steps, args.flow_samples, args.flow_seed) != (128, 20, 2, 5, 23):
        raise ValueError("V3.31 requires 128 camera / 20Hz / K5 / action2 / seed23")
    if (
        args.no_flip_views
        or args.state_clip is not None
        or args.outcome_verifier is not None
        or args.action_intent is not None
        or getattr(args, "candidate_aggregation", None) is not None
    ):
        raise ValueError("V3.31 cannot override its frozen observation/action contract")


def _validate_v332_deployment(checkpoint: dict[str, object], args: argparse.Namespace) -> None:
    if checkpoint.get("format") != "duvla_v3_32":
        return
    _validate_v331_deployment({**checkpoint, "format": "duvla_v3_31", "version": "V3.31", "epoch": 30, "effective_epochs": 30.0, "planned_epochs": 30}, args)
    raw = checkpoint.get("qwen_lora_spec")
    state = checkpoint.get("qwen_lora_state_dict")
    if (
        checkpoint.get("formal") is not True
        or checkpoint.get("checkpoint_complete") is not True
        or checkpoint.get("training_complete") is not True
        or checkpoint.get("epoch") != 6
        or float(checkpoint.get("effective_epochs", 0)) < 6
        or checkpoint.get("planned_epochs") != 6
        or checkpoint.get("history_feature_contract")
        != "frozen_Qwen_history_plus_online_LoRA_current"
        or not isinstance(raw, dict)
        or not isinstance(state, dict)
        or int(checkpoint.get("qwen_lora_trainable_parameters", 0)) != 229376
    ):
        raise ValueError("V3.32 requires its completed formal 6E LoRA checkpoint")
    spec = QwenLoraSpec(**raw)
    if spec != QwenLoraSpec():
        raise ValueError("V3.32 LoRA architecture contract mismatch")


def _state_bound_seed(flow_seed: int, task_id: int, init_state_id: int) -> int:
    """Seed a physical initial state independently of manifest ordering."""
    return flow_seed + task_id * 100000 + init_state_id * 1000


def _validate_v326_deployment(checkpoint: dict[str, object], args: argparse.Namespace) -> None:
    """An architecture-only checkpoint must never silently enter a formal gate."""
    if checkpoint.get("format") != "duvla_v3_26":
        return
    epochs = checkpoint.get("effective_epochs", 0.0)
    valid_epochs = (
        isinstance(epochs, (int, float)) and not isinstance(epochs, bool)
        and math.isfinite(epochs) and epochs >= 6.0
    )
    if (
        checkpoint.get("training_complete") is not True
        or not valid_epochs
        or checkpoint.get("feedback_contract") != "actual_executed_prefix2_v1"
    ):
        raise ValueError("V3.26 architecture is not a trained, qualified feedback checkpoint")
    if any(
        checkpoint.get(name) is not False
        for name in ("uses_reward", "uses_success", "uses_evaluation_initial_states")
    ):
        raise ValueError("V3.26 checkpoint lacks reward-free training provenance")
    if _checkpoint_stage(checkpoint) != "flow":
        raise ValueError("V3.26 only deploys unified Flow with execution feedback")
    if args.fps != 20 or args.action_steps != 2:
        raise ValueError("V3.26 feedback requires 20Hz and exactly two executed steps")
    if args.outcome_verifier is not None or args.action_intent is not None:
        raise ValueError("V3.26 cannot bypass feedback via a legacy candidate verifier")
    if args.no_flip_views or args.state_clip is not None:
        raise ValueError("V3.26 must preserve its camera and state-normalization contract")


def _execution_prefix(
    predicted: np.ndarray,
    action_stats: StandardizationStats,
    *,
    count: int,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[np.ndarray, torch.Tensor, torch.Tensor]:
    """Forecast the commands actually sent to env.step, including physical clipping."""
    if count not in {1, 2} or predicted.ndim != 2 or predicted.shape[1] != 7:
        raise ValueError("execution feedback requires a one/two-step 7D prefix")
    if predicted.shape[0] < count or not np.isfinite(predicted[:count]).all():
        raise ValueError("execution prefix is too short or non-finite")
    commands = flow_normalized_action_to_env(predicted[:count], action_stats)
    normalized = torch.zeros((1, 2, 7), device=device, dtype=dtype)
    normalized[:, :count] = torch.as_tensor(
        action_stats.normalize(commands), device=device, dtype=dtype
    )[None]
    mask = torch.zeros((1, 2), device=device, dtype=torch.bool)
    mask[:, :count] = True
    return commands, normalized, mask


def main() -> None:
    args = parse_args()
    if args.max_steps is None:
        args.max_steps = libero_standard_max_steps(args.suite)
    if args.resume and args.overwrite:
        raise SystemExit("--resume and --overwrite are mutually exclusive")
    if args.save_video and args.video_dir is None:
        raise SystemExit("--save-video requires --video-dir")
    if args.action_intent is not None and args.outcome_verifier is None:
        raise SystemExit("V3.13 candidates may only deploy with an audited V3.14 ranker")
    if min(args.episodes, args.max_steps, args.action_steps, args.settle_steps, args.camera_size, args.fps, args.flow_samples) <= 0:
        raise SystemExit("episode, rollout, camera and Flow values must be positive")
    if args.state_manifest is not None and not args.all_init_states:
        raise SystemExit("--state-manifest requires --all-init-states")
    if args.state_clip is not None and args.state_clip <= 0:
        raise SystemExit("state-clip must be positive")
    if args.stop_after_failures is not None and args.stop_after_failures < 0:
        raise SystemExit("stop-after-failures must be non-negative")
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but unavailable")
    train_manifest = json.loads(args.train_manifest.read_text())
    if train_manifest.get("schema_version") not in {7, 8}:
        raise SystemExit("Duvla V2 evaluation requires a schema-7/8 train manifest")
    state_stats = StandardizationStats.from_manifest(train_manifest, "state")
    action_stats = StandardizationStats.from_manifest(train_manifest, "action")
    normalized_noop_action = action_stats.normalize(
        np.asarray([0.0] * (action_stats.dim - 1) + [-1.0], dtype=np.float32)
    )
    policy, checkpoint = _load_policy(args.checkpoint, train_manifest, device)
    _validate_v329_deployment(checkpoint, args)
    _validate_v330_deployment(checkpoint, args)
    _validate_v331_deployment(checkpoint, args)
    _validate_v332_deployment(checkpoint, args)
    _validate_v326_deployment(checkpoint, args)
    execution_feedback_enabled = checkpoint.get("format") == "duvla_v3_26"
    if state_stats.dim != policy.config.state_dim or action_stats.dim != policy.config.action_dim:
        raise SystemExit("checkpoint and normalization dimensions differ")
    stage = _checkpoint_stage(checkpoint)
    apply_direct = stage in {"direct", "instruction", "joint"}
    apply_instruction = stage in {"instruction", "joint"}
    apply_gripper_event = not policy.config.unified_flow_gripper
    outcome_verifier: (
        ActionConditionedOutcomeVerifier | ActionConditionedOutcomeEnsemble | None
    ) = None
    outcome_checkpoint: dict[str, object] | None = None
    action_intent: StructuredActionIntentReasoner | None = None
    action_intent_checkpoint: dict[str, object] | None = None
    if args.action_intent is not None:
        if checkpoint.get("format") != "duvla_v2_6" or stage != "flow":
            raise SystemExit("V3.13 candidate generation requires the exact V2.6 Flow parent")
        action_intent, action_intent_checkpoint = _load_action_intent(
            args.action_intent,
            policy_checkpoint=args.checkpoint,
            policy=policy,
            device=device,
        )
    if args.outcome_verifier is not None:
        if checkpoint.get("format") != "duvla_v2_6" or stage != "flow":
            raise SystemExit("V2.8 outcome selection currently requires a V2.6 Flow checkpoint")
        outcome_verifier, outcome_checkpoint = _load_outcome_verifier(
            args.outcome_verifier,
            policy_checkpoint=args.checkpoint,
            policy=policy,
            device=device,
            action_intent_checkpoint=args.action_intent,
        )
        is_structured_ranker = outcome_checkpoint.get("format") in {
            "duvla_v3_14_outcome_ranker",
            "duvla_v3_15_outcome_ranker",
            "duvla_v3_16_outcome_ranker",
            "duvla_v3_17_outcome_ranker",
            "duvla_v3_18_outcome_ensemble",
        }
        if is_structured_ranker != (action_intent is not None):
            raise SystemExit("structured V3 ranker and V3.13 candidate generator must be used together")

    try:
        import huggingface_hub
        import transformers
        from libero.libero import benchmark
        from libero.libero.envs import OffScreenRenderEnv
        if args.save_video:
            import imageio.v2 as imageio
        else:
            imageio = None
    except ImportError as exc:  # pragma: no cover - evaluation environment only
        raise SystemExit("run in the dedicated LIBERO evaluation environment") from exc
    if huggingface_hub.__version__ != "0.36.2" or transformers.__version__ != "4.57.6":
        raise SystemExit("LIBERO evaluator requires huggingface_hub==0.36.2 and transformers==4.57.6")

    suite = benchmark.get_benchmark(args.suite)(task_order_index=0)
    task_ids = _task_ids(args.task_indices, suite.get_num_tasks())
    selection: dict[int, tuple[int, ...]] | None = None
    state_digest: str | None = None
    if args.state_manifest is not None:
        selection, state_digest = _load_state_manifest(
            args.state_manifest,
            suite_name=args.suite,
            task_ids=task_ids,
            task_count=suite.get_num_tasks(),
        )
    if checkpoint.get("format") == "duvla_v3_32":
        backbone = QwenVLBackbone(QwenBackboneConfig(freeze=False)).load(device=device)
        spec = QwenLoraSpec(**checkpoint["qwen_lora_spec"])
        attach_qwen_lora(backbone, spec)
        from peft import set_peft_model_state_dict

        loaded_lora = set_peft_model_state_dict(
            backbone.model, checkpoint["qwen_lora_state_dict"]
        )
        if getattr(loaded_lora, "unexpected_keys", None):
            raise ValueError(f"V3.32 LoRA load mismatch: {loaded_lora.unexpected_keys}")
        backbone.model.eval()
    else:
        backbone = QwenVLBackbone().load(device=device)
    args.trace_dir.mkdir(parents=True, exist_ok=True)
    if args.video_dir is not None:
        args.video_dir.mkdir(parents=True, exist_ok=True)
    run_config: dict[str, object] = {
        "format": (
            "duvla_v3_20_strict_libero"
            if outcome_checkpoint is not None
            and outcome_checkpoint.get("format") == "duvla_v3_20_outcome_ranker"
            else "duvla_v3_14_strict_libero"
            if action_intent is not None
            else "duvla_v2_8_strict_libero"
            if outcome_verifier is not None
            else
            "duvla_v3_19_strict_libero"
            if checkpoint.get("format") == "duvla_v3_19"
            else "duvla_v3_32_strict_libero"
            if checkpoint.get("format") == "duvla_v3_32"
            else "duvla_v3_31_strict_libero"
            if checkpoint.get("format") == "duvla_v3_31"
            else "duvla_v3_23_strict_libero"
            if checkpoint.get("format") == "duvla_v3_23"
            else "duvla_v3_25_strict_libero"
            if checkpoint.get("format") == "duvla_v3_25"
            else "duvla_v2_7_strict_libero"
            if checkpoint.get("format") == "duvla_v2_7"
            else "duvla_v2_6_strict_libero"
            if checkpoint.get("format") == "duvla_v2_6"
            else "duvla_v3_27_strict_libero"
            if checkpoint.get("format") == "duvla_v3_27"
            else "duvla_v3_28_strict_libero"
            if checkpoint.get("format") == "duvla_v3_28"
            else "duvla_v2_5_strict_libero"
            if checkpoint.get("format") == "duvla_v2_5"
            else "duvla_v2_4_strict_libero"
            if checkpoint.get("format") == "duvla_v2_4"
            else "duvla_v2_3_strict_libero"
            if checkpoint.get("format") == "duvla_v2_3"
            else "duvla_v2_2_strict_libero"
            if checkpoint.get("format") == "duvla_v2_2"
            else "duvla_v3_1_strict_libero"
            if checkpoint.get("format") == "duvla_v3_1"
            else "duvla_v2_1_strict_libero"
        ),
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": file_sha256(args.checkpoint),
        "outcome_verifier": (
            str(args.outcome_verifier.resolve())
            if args.outcome_verifier is not None
            else None
        ),
        "outcome_verifier_version": (
            outcome_checkpoint.get("version")
            if outcome_checkpoint is not None
            else None
        ),
        "action_intent": (
            str(args.action_intent.resolve()) if args.action_intent is not None else None
        ),
        "action_intent_version": (
            action_intent_checkpoint.get("version")
            if action_intent_checkpoint is not None
            else None
        ),
        "suite": args.suite,
        "task_indices": list(task_ids),
        "state_manifest": str(args.state_manifest.resolve()) if args.state_manifest else None,
        "state_manifest_sha256": state_digest,
        "max_steps": args.max_steps,
        "control_frequency_hz": args.fps,
        "action_steps": args.action_steps,
        "settle_steps": args.settle_steps,
        "flow_seed": args.flow_seed,
        "flow_samples": args.flow_samples,
        "candidate_aggregation": args.candidate_aggregation or policy.config.candidate_aggregation,
        "flow_ensemble": (
            "v3.13_structured_top5_plus_base_then_outcome_ranker"
            if action_intent is not None
            else "physical_outcome_verifier_whole_trajectory"
            if outcome_verifier is not None
            else "parallel_continuous_query"
            if checkpoint.get("format") == "duvla_v3_21"
            else "parallel_prior_plus_arm_residual_flow_median"
            if checkpoint.get("format") == "duvla_v3_22"
            else "whole_trajectory_medoid"
            if (args.candidate_aggregation or policy.config.candidate_aggregation) == "trajectory_medoid"
            else "median"
        ),
        "task_routing": "natural_language",
        "benchmark_task_index": False,
        "qwen_hidden_layers": [12, 14, 18, -1],
        "spatial_representation": (
            "layer14_8x8_plus_layers12_14_18_final_4x4_bounded_residual"
            if policy.config.multilayer_spatial_residual
            else "layer14_8x8_plus_layers12_14_18_final_semantic"
            if policy.config.highres_layer14_visual
            else "four_layer_4x4"
        ),
        "history_stride": policy.config.history_stride,
        "action_history_conditioning": policy.config.action_history_conditioning,
        "gripper_action_conditioning": policy.config.gripper_action_conditioning,
        "camera_orientation": "flip_180" if not args.no_flip_views else "native",
        "save_video": args.save_video,
        "stage": stage,
        "apply_direct": apply_direct,
        "apply_instruction": apply_instruction,
        "apply_gripper_event": apply_gripper_event,
        "gripper_control": (
            "joint_continuous_flow"
            if policy.config.unified_flow_gripper
            else "three_class_event_fsm"
            if policy.config.gripper_control_mode == "event3"
            else "event_schmitt_hysteresis"
        ),
        "gripper_control_mode": policy.config.gripper_control_mode,
        "gripper_transition_head": (
            "state_auxiliary_only"
            if policy.config.gripper_control_mode == "event3"
            else "auxiliary_training_only"
        ),
    }
    if checkpoint.get('format') == 'duvla_v3_29':
        run_config.update(format='duvla_v3_29_strict_libero', camera_size=args.camera_size,
            action_offset=1, sidecar_sha256=checkpoint['sidecar_sha256'],
            ordered_language_bridge=policy.config.ordered_language_bridge,
            checkpoint_role='development_20e' if args.v329_development_20e else 'final_30e')
    elif checkpoint.get('format') == 'duvla_v3_30':
        run_config.update(
            format='duvla_v3_30_strict_libero', camera_size=args.camera_size,
            action_offset=1, sidecar_sha256=checkpoint['sidecar_sha256'],
            ordered_language_bridge=policy.config.ordered_language_bridge,
            action_attention='bidirectional_chunk', checkpoint_role='final_30e',
            seed_binding='task_and_init_state_id',
        )
    elif checkpoint.get("format") == "duvla_v3_31":
        run_config.update(
            format="duvla_v3_31_strict_libero",
            camera_size=args.camera_size,
            action_offset=1,
            sidecar_sha256=checkpoint["sidecar_sha256"],
            augmentation_sidecar_sha256=checkpoint["augmentation_sidecar_sha256"],
            ordered_language_bridge=True,
            language_max_tokens=policy.config.language_max_tokens,
            cross_camera_fusion=True,
            checkpoint_role="final_30e",
            seed_binding="task_and_init_state_id",
        )
    elif checkpoint.get("format") == "duvla_v3_32":
        run_config.update(
            format="duvla_v3_32_strict_libero",
            camera_size=args.camera_size,
            action_offset=1,
            sidecar_sha256=checkpoint["sidecar_sha256"],
            augmentation_sidecar_sha256=checkpoint["augmentation_sidecar_sha256"],
            ordered_language_bridge=True,
            language_max_tokens=policy.config.language_max_tokens,
            cross_camera_fusion=True,
            qwen_lora_spec=checkpoint["qwen_lora_spec"],
            qwen_lora_trainable_parameters=checkpoint["qwen_lora_trainable_parameters"],
            history_feature_contract=checkpoint["history_feature_contract"],
            checkpoint_role="final_6e_lora",
            seed_binding="task_and_init_state_id",
        )
    config_path = args.trace_dir / "run_config.json"
    if execution_feedback_enabled:
        run_config["execution_feedback"] = {
            "contract": "actual_executed_prefix2_v1",
            "input": "actual_clipped_actions+observed_delta+previous_forecast_error",
            "residual_location": "Flow_velocity_before_5_sample_median",
            "forecast": "two_step_state_and_dual_camera_spatial_summary_delta",
            "reset_per_episode": True,
        }
    if config_path.exists() and not args.overwrite:
        if json.loads(config_path.read_text()) != run_config:
            raise SystemExit("trace directory belongs to a different evaluation run")
    _atomic_json(config_path, run_config)

    results: list[dict[str, object]] = []
    summary_path = args.trace_dir / "summary.json"
    if args.resume and summary_path.is_file():
        existing = json.loads(summary_path.read_text()).get("episodes", [])
        if isinstance(existing, list):
            results.extend(item for item in existing if isinstance(item, dict))
    completed_keys = {(int(item["task_id"]), int(item["episode"])) for item in results}
    flip_views = not args.no_flip_views
    for task_id in task_ids:
        task = suite.get_task(task_id)
        init_states = load_libero_initial_states(suite, task_id)
        if selection is not None:
            episode_state_pairs = tuple(enumerate(selection[task_id]))
        elif args.all_init_states:
            stop = len(init_states) if args.init_state_count is None else args.init_state_start + args.init_state_count
            if args.init_state_start < 0 or stop > len(init_states):
                raise SystemExit("requested init-state interval is out of range")
            episode_state_pairs = tuple((state_id, state_id) for state_id in range(args.init_state_start, stop))
        else:
            episode_state_pairs = tuple((episode, episode % len(init_states)) for episode in range(args.episodes))
        for episode, init_state_id in episode_state_pairs:
            trace_path = args.trace_dir / f"task-{task_id:02d}-episode-{episode:03d}.json"
            if args.resume and (task_id, episode) in completed_keys:
                continue
            if trace_path.exists() and not args.overwrite:
                complete = _episode_complete(trace_path, task_id=task_id, episode=episode)
                if complete is None:
                    raise SystemExit(f"incomplete existing trace: {trace_path}")
                results.append(complete)
                completed_keys.add((task_id, episode))
                continue
            video_path = None if args.video_dir is None else args.video_dir / f"task-{task_id:02d}-episode-{episode:03d}.mp4"
            env = OffScreenRenderEnv(
                bddl_file_name=suite.get_task_bddl_file_path(task_id),
                camera_names=["agentview", "robot0_eye_in_hand"],
                camera_heights=args.camera_size,
                camera_widths=args.camera_size,
                control_freq=args.fps,
                horizon=args.max_steps + args.settle_steps,
            )
            episode_seed = _state_bound_seed(args.flow_seed, task_id, init_state_id)
            env.seed(episode_seed)
            rewards: list[float] = []
            env_actions: list[list[float]] = []
            normalized_actions: list[list[float]] = []
            feedback_records: list[dict[str, object]] = []
            feedback_tracker = (
                ExecutionFeedbackTracker(action_stats)
                if execution_feedback_enabled else None
            )
            selected_candidates: list[int] = []
            feature_history: list[
                tuple[
                    torch.Tensor,
                    torch.Tensor | None,
                    torch.Tensor,
                    torch.Tensor,
                    torch.Tensor,
                ]
            ] = []
            previous_gripper_closed = torch.zeros(1, dtype=torch.bool, device=device)
            previous_action_tensor = torch.from_numpy(
                normalized_noop_action[None].astype(np.float32)
            ).to(device=device, dtype=next(policy.parameters()).dtype)
            flow_generator = torch.Generator(device="cpu")
            flow_generator.manual_seed(episode_seed)
            writer = None
            frames = 0
            try:
                observation = env.reset()
                observation = env.set_init_state(init_states[init_state_id])
                for _ in range(args.settle_steps):
                    observation, _, done, _ = env.step(libero_dummy_action())
                    if done:
                        break
                if args.save_video:
                    if imageio is None or video_path is None:  # pragma: no cover
                        raise RuntimeError("video writer is unavailable")
                    writer = imageio.get_writer(video_path, fps=args.fps, codec="libx264", macro_block_size=1)
                while len(rewards) < args.max_steps:
                    agent = orient_libero_view(observation["agentview_image"], flip_180=flip_views)
                    wrist = orient_libero_view(observation["robot0_eye_in_hand_image"], flip_180=flip_views)
                    if writer is not None:
                        writer.append_data(compose_views(agent, wrist))
                    frames += 1
                    state = state_stats.normalize(observation_to_state(observation))
                    if args.state_clip is not None:
                        state = np.clip(state, -args.state_clip, args.state_clip)
                    state_tensor = torch.from_numpy(state[None].astype(np.float32)).to(device=device, dtype=next(policy.parameters()).dtype)
                    # Training consumes the FP32 bridge residual under BF16
                    # autocast.  Keep the same boundary online; this also
                    # prevents dtype mismatches after the V3.30 FP32 add.
                    with torch.inference_mode(), torch.autocast(
                        device_type=device.type,
                        dtype=torch.bfloat16,
                        enabled=device.type == "cuda",
                    ):
                        camera_inputs = (
                            backbone.prepare_view_inputs(Image.fromarray(agent), task.language),
                            backbone.prepare_view_inputs(Image.fromarray(wrist), task.language),
                        )
                        language_kwargs = {}
                        base_history_visual_all = None
                        base_history_semantic = None
                        if checkpoint.get("format") in {"duvla_v3_31", "duvla_v3_32"}:
                            visual_all, semantic, language_tokens, language_mask = backbone.forward_v331_context(
                                camera_inputs,
                                [task.language],
                                max_tokens=policy.config.language_max_tokens,
                            )
                            language_kwargs = {'language_tokens':language_tokens, 'language_mask':language_mask}
                            if checkpoint.get("format") == "duvla_v3_32":
                                with backbone.model.disable_adapter():
                                    base_history_visual_all, base_history_semantic, _, _ = backbone.forward_v331_context(
                                        camera_inputs,
                                        [task.language],
                                        max_tokens=policy.config.language_max_tokens,
                                    )
                        elif getattr(policy.config, 'ordered_language_bridge', False):
                            visual_all, semantic, language_tokens, language_mask = backbone.forward_v329_context(camera_inputs, [task.language])
                            language_kwargs = {'language_tokens':language_tokens, 'language_mask':language_mask}
                        else:
                            visual_all, semantic = backbone.forward_multiview_multilayer_spatial_semantic_context(
                                camera_inputs, [task.language], hidden_layers=(12,14,18,-1),
                                expected_grid=(8,8), output_grid=(8,8) if policy.config.highres_layer14_visual else (4,4))
                        auxiliary_visual = None
                        if policy.config.highres_layer14_visual:
                            if policy.config.multilayer_spatial_residual:
                                auxiliary_visual = pool_multilayer_spatial_grid(
                                    visual_all, output_side=4
                                )
                            visual = visual_all[:, 1:2]
                        else:
                            visual = visual_all
                        visual = visual.to(dtype=next(policy.parameters()).dtype)
                        if auxiliary_visual is not None:
                            auxiliary_visual = auxiliary_visual.to(
                                dtype=next(policy.parameters()).dtype
                            )
                        semantic = semantic.to(dtype=next(policy.parameters()).dtype)
                        history_visual_current = (
                            base_history_visual_all[:, 1:2].to(dtype=visual.dtype)
                            if base_history_visual_all is not None
                            else visual
                        )
                        history_semantic_current = (
                            base_history_semantic.to(dtype=semantic.dtype)
                            if base_history_semantic is not None
                            else semantic
                        )
                        history_auxiliary_current = (
                            pool_multilayer_spatial_grid(base_history_visual_all, output_side=4).to(
                                dtype=visual.dtype
                            )
                            if base_history_visual_all is not None
                            and policy.config.multilayer_spatial_residual
                            else auxiliary_visual
                        )
                        history_count = policy.config.history_length - 1
                        past = feature_history[-history_count:]
                        padded = [
                            (
                                history_visual_current,
                                history_auxiliary_current,
                                history_semantic_current,
                                state_tensor,
                                previous_action_tensor,
                            )
                        ] * (history_count - len(past)) + past
                        history_visual = torch.stack([item[0] for item in padded], dim=1)
                        history_auxiliary_visual = (
                            torch.stack([item[1] for item in padded], dim=1)
                            if history_auxiliary_current is not None
                            else None
                        )
                        history_semantic = torch.stack([item[2] for item in padded], dim=1)
                        history_states = torch.stack([item[3] for item in padded], dim=1)
                        history_previous_actions = torch.stack(
                            [item[4] for item in padded], dim=1
                        )
                        noise = torch.randn(
                            1,
                            args.flow_samples,
                            policy.config.action_horizon,
                            policy.config.action_dim,
                            generator=flow_generator,
                            dtype=torch.float32,
                        ).to(device=device, dtype=next(policy.parameters()).dtype)
                        feedback_context = None
                        feedback_kwargs: dict[str, object] = {}
                        if feedback_tracker is not None:
                            feedback_context = policy.encode_outcome_context(
                                visual, semantic, state_tensor,
                                history_visual=history_visual,
                                history_semantic=history_semantic,
                                history_states=history_states,
                                previous_action=previous_action_tensor,
                                history_previous_actions=history_previous_actions,
                            )
                            feedback_kwargs["execution_feedback"] = feedback_tracker.get_feedback()
                            feedback_records.append(
                                {"decision": frames - 1, **feedback_tracker.diagnostics(
                                    feedback_context, state_tensor
                                )}
                            )
                        if outcome_verifier is None:
                            predicted = policy.sample_actions(
                                visual,
                                semantic,
                                state_tensor,
                                auxiliary_visual=auxiliary_visual,
                                history_visual=history_visual,
                                history_auxiliary_visual=history_auxiliary_visual,
                                history_semantic=history_semantic,
                                history_states=history_states,
                                previous_action=previous_action_tensor,
                                history_previous_actions=history_previous_actions,
                                flow_samples=args.flow_samples,
                                apply_direct=apply_direct,
                                apply_instruction=apply_instruction,
                                apply_gripper_event=apply_gripper_event,
                                previous_gripper_closed=previous_gripper_closed,
                                noise=noise,
                                candidate_aggregation=args.candidate_aggregation,
                                **feedback_kwargs,
                                **language_kwargs,
                            )[0].float().cpu().numpy()
                        else:
                            candidate_pool = policy.sample_action_candidates(
                                visual,
                                semantic,
                                state_tensor,
                                auxiliary_visual=auxiliary_visual,
                                history_visual=history_visual,
                                history_auxiliary_visual=history_auxiliary_visual,
                                history_semantic=history_semantic,
                                history_states=history_states,
                                previous_action=previous_action_tensor,
                                history_previous_actions=history_previous_actions,
                                flow_samples=args.flow_samples,
                                apply_direct=apply_direct,
                                apply_instruction=apply_instruction,
                                noise=noise,
                            )
                            policy_context = policy.encode_outcome_context(
                                visual,
                                semantic,
                                state_tensor,
                                auxiliary_visual=auxiliary_visual,
                                history_visual=history_visual,
                                history_auxiliary_visual=history_auxiliary_visual,
                                history_semantic=history_semantic,
                                history_states=history_states,
                                previous_action=previous_action_tensor,
                                history_previous_actions=history_previous_actions,
                            )
                            if action_intent is not None:
                                train_branch_manifest = outcome_checkpoint["train_manifest"]
                                top_k = int(train_branch_manifest["baseline_candidate_index"])
                                expected_candidates = int(
                                    train_branch_manifest["candidate_count"]
                                )
                                candidate_pool, _intent_indices, _intent_scores = (
                                    action_intent.candidates(
                                        policy_context,
                                        state_tensor,
                                        candidate_pool[:, -1],
                                        top_k=top_k,
                                    )
                                )
                                if candidate_pool.shape[1] != expected_candidates:
                                    raise RuntimeError("structured V3 online candidate count drifted")
                            verifier_context = arrange_v2_policy_context(
                                policy_context,
                                camera_count=policy.config.camera_count,
                                spatial_tokens_per_camera=policy.config.spatial_tokens,
                            )
                            candidate_mask = torch.ones(
                                candidate_pool.shape[:2],
                                dtype=torch.bool,
                                device=device,
                            )
                            verifier_output = outcome_verifier(
                                verifier_context,
                                state_tensor,
                                candidate_pool,
                                candidate_mask=candidate_mask,
                            )
                            if (
                                outcome_checkpoint.get("format")
                                in BASELINE_FALLBACK_VERIFIER_FORMATS
                            ):
                                selected_tensor = select_with_baseline_fallback(
                                    verifier_output.score,
                                    candidate_mask,
                                    baseline_index=int(
                                        outcome_checkpoint["train_manifest"][
                                            "baseline_candidate_index"
                                        ]
                                    ),
                                    minimum_advantage=float(
                                        outcome_checkpoint["selection_margin"]
                                    ),
                                )
                            else:
                                selected_tensor = verifier_output.selected_index
                            selected = int(selected_tensor[0])
                            selected_candidates.append(selected)
                            predicted = candidate_pool[0, selected].float().cpu().numpy()
                    feature_history.append(
                        (
                            history_visual_current.detach(),
                            (
                                None
                                if history_auxiliary_current is None
                                else history_auxiliary_current.detach()
                            ),
                            history_semantic_current.detach(),
                            state_tensor.detach(),
                            previous_action_tensor.detach(),
                        )
                    )
                    del feature_history[:-history_count]
                    forecast = None
                    planned_commands = None
                    if feedback_tracker is not None:
                        planned_commands, forecast_actions, forecast_mask = _execution_prefix(
                            predicted, action_stats,
                            count=min(args.action_steps, args.max_steps - len(rewards)),
                            device=device, dtype=state_tensor.dtype,
                        )
                        with torch.inference_mode():
                            forecast = policy.forecast_execution(
                                feedback_context, state_tensor, forecast_actions, forecast_mask
                            )
                    execution_start = len(env_actions)
                    done = False
                    for action_index, normalized in enumerate(predicted[: args.action_steps]):
                        action = (
                            planned_commands[action_index]
                            if planned_commands is not None
                            else flow_normalized_action_to_env(normalized, action_stats)
                        )
                        observation, reward, done, _ = env.step(action)
                        normalized_actions.append(normalized.astype(np.float32).tolist())
                        env_actions.append(action.tolist())
                        rewards.append(float(reward))
                        if done or len(rewards) >= args.max_steps:
                            break
                    if feedback_tracker is not None:
                        executed = env_actions[execution_start:]
                        actual_commands = torch.zeros((1, 2, 7), device=device, dtype=torch.float32)
                        actual_mask = torch.zeros((1, 2), device=device, dtype=torch.bool)
                        actual_commands[:, :len(executed)] = torch.as_tensor(
                            executed, device=device, dtype=torch.float32
                        )[None]
                        actual_mask[:, :len(executed)] = True
                        feedback_tracker.record_execution(
                            feedback_context, state_tensor, actual_commands, actual_mask,
                            forecast,
                        )
                    if normalized_actions:
                        previous_action_tensor = torch.as_tensor(
                            normalized_actions[-1],
                            device=device,
                            dtype=next(policy.parameters()).dtype,
                        )[None, :]
                    if len(predicted[: args.action_steps]):
                        previous_gripper_closed.fill_(
                            bool(predicted[min(args.action_steps, len(predicted)) - 1, -1] > 0.0)
                        )
                    if done:
                        break
                success = bool(env.check_success())
            finally:
                if writer is not None:
                    writer.close()
                env.close()
            result: dict[str, object] = {
                "task_id": task_id,
                "task_language": task.language,
                "episode": episode,
                "init_state_id": init_state_id,
                "env_seed": episode_seed,
                "flow_noise_seed": episode_seed,
                "success": success,
                "frames": frames,
                "executed_actions": len(env_actions),
                "reward_sum": float(sum(rewards)),
                "checkpoint": str(args.checkpoint.resolve()),
                "outcome_verifier": (
                    str(args.outcome_verifier.resolve())
                    if args.outcome_verifier is not None
                    else None
                ),
                "selected_candidate_histogram": {
                    str(index): selected_candidates.count(index)
                    for index in sorted(set(selected_candidates))
                },
                "benchmark_task_index": False,
            }
            trace: dict[str, object] = {
                "summary": result, "normalized_actions": normalized_actions,
                "env_actions": env_actions,
            }
            if execution_feedback_enabled:
                trace["execution_feedback"] = feedback_records
            _atomic_json(trace_path, trace)
            results.append(result)
            completed_keys.add((task_id, episode))
            successes = sum(bool(item["success"]) for item in results)
            _atomic_json(
                summary_path,
                {
                    "run_config": run_config,
                    "aggregate": {
                        "episode_count": len(results),
                        "success_count": successes,
                        "success_rate": successes / len(results),
                    },
                    "episodes": results,
                    "complete": False,
                },
            )
            print(json.dumps(result, ensure_ascii=False), flush=True)
            failures = len(results) - successes
            if _failure_budget_exhausted(
                episode_count=len(results),
                success_count=successes,
                maximum_failures=args.stop_after_failures,
            ):
                print(
                    json.dumps(
                        {
                            "type": "failure_budget_exhausted",
                            "episode_count": len(results),
                            "success_count": successes,
                            "failure_count": failures,
                            "stop_after_failures": args.stop_after_failures,
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
                return
    successes = sum(bool(item["success"]) for item in results)
    aggregate = {
        "episode_count": len(results),
        "success_count": successes,
        "success_rate": successes / len(results) if results else 0.0,
    }
    _atomic_json(
        summary_path,
        {"run_config": run_config, "aggregate": aggregate, "episodes": results, "complete": True},
    )
    print(json.dumps(aggregate), flush=True)


if __name__ == "__main__":
    main()
