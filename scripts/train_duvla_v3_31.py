#!/usr/bin/env python3
"""Train the single preregistered V3.31 joint adaptation candidate."""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict, replace
from hashlib import sha256
import json
from itertools import repeat
import math
from pathlib import Path
import time

import torch

from duvla.models.duvla_v2_1 import DuvlaV21Config, DuvlaV21Policy
from duvla.training.feature_cache import atomic_write_json
from duvla.training.loss_curve import (
    append_loss_point,
    truncate_loss_points,
    write_loss_curve_artifacts,
)
from duvla.training.resource_budget import require_disk_budget
from duvla.training.v2_1_data import iter_v2_1_batches, prefetch_v2_1_batches
from duvla.training.v3_29_data import V329Sidecar
from duvla.training.v3_31_data import V331AugmentSidecar
from train_duvla_v2_1 import (
    _assert_fp32_training_state,
    _atomic_torch_save,
    _configure_stage,
    _model_config,
    _seed_everything,
    _task_weights,
)


VERSION = "V3.31"
PARENT_SHA256 = "600c8c46aa63d7f6a1f38cbaf7b9fc7b5c98601736a1861b0dd9f50dee598227"
RESUME_COMPATIBLE_TRAINER_HASHES = {
    # Formal 1E checkpoint before the artifact-only learning-rate export and
    # resumed-parity guard fixes. Training math and optimizer state are equal.
    "17c06642a4fbcdbb8695e9335226b501cb0756b41977b7974c567b52dce7d60f",
}


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def make_policy(base: dict[str, object], parent: dict[str, object], seed: int) -> tuple[DuvlaV21Policy, object, tuple[str, ...]]:
    if (parent.get("format"), parent.get("version")) != ("duvla_v3_29", "V3.29"):
        raise ValueError("V3.31 requires the fixed V3.29 parent")
    _seed_everything(seed)
    config = replace(
        _model_config(base, smoke=False, v3_28_fp32_amp_flow=True),
        ordered_language_bridge=True,
        ordered_language_bridge_mode="scalar_gate",
        language_max_tokens=256,
        cross_camera_fusion=True,
        causal_action_attention=True,
        candidate_aggregation="coordinate_median",
    )
    policy = DuvlaV21Policy(config)
    loaded = policy.load_state_dict(parent["model_state_dict"], strict=False)
    if loaded.unexpected_keys or not loaded.missing_keys:
        raise ValueError(f"V3.31 parent load mismatch: {loaded}")
    if any(not name.startswith("cross_camera_bridge.") for name in loaded.missing_keys):
        raise ValueError(f"V3.31 has non-camera parent gaps: {loaded.missing_keys}")
    groups = list(_configure_stage(policy, "flow"))
    if policy.language_bridge is None or policy.cross_camera_bridge is None:
        raise RuntimeError("V3.31 joint modules are unavailable")
    policy.language_bridge.requires_grad_(True)
    policy.cross_camera_bridge.requires_grad_(True)
    groups.extend(("language_bridge", "cross_camera_bridge"))
    return policy, config, tuple(groups)


def _select(values: torch.Tensor, slots: torch.Tensor) -> torch.Tensor:
    return values.index_select(0, slots.to(values.device))


def validate_parent(parent: dict, base: dict, language_sha256: str) -> None:
    """Allow a newly trained parent while retaining lineage and full-budget checks."""
    if (parent.get('format'), parent.get('version')) != ('duvla_v3_29', 'V3.29'):
        raise ValueError('V3.31 requires a V3.29 parent')
    if (parent.get('formal') is not True or parent.get('training_complete') is not True
            or parent.get('checkpoint_complete') is not True or parent.get('epoch') != 30
            or float(parent.get('effective_epochs', 0)) < 30 or parent.get('action_offset') != 1):
        raise ValueError('parent must be a completed formal 30E V3.29 checkpoint')
    if any(parent.get(k) is not False for k in
           ('uses_reward', 'uses_success', 'uses_evaluation_initial_states', 'benchmark_task_index')):
        raise ValueError('parent training provenance mismatch')
    if parent.get('cache_signature') != base.get('cache_signature'):
        raise ValueError('V3.31 parent/base cache mismatch')
    if parent.get('sidecar_sha256') != language_sha256:
        raise ValueError('V3.31 parent/language sidecar mismatch')
    # 历史V3.29保存于这两个显式字段加入前，必须按加载器同一dataclass默认值恢复。
    config = DuvlaV21Config(**parent.get('model_config', {}))
    if (config.ordered_language_bridge is not True
            or config.causal_action_attention is not True
            or config.candidate_aggregation != 'coordinate_median'):
        raise ValueError('parent architecture is not the V3.29 language recipe')


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-cache", type=Path, required=True)
    parser.add_argument("--language-sidecar", type=Path, required=True)
    parser.add_argument("--augment-sidecar", type=Path, required=True)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--parent-sha256", default=PARENT_SHA256,
                        help="Expected SHA256 of your completed V3.29; defaults to the historical parent")
    parser.add_argument("--check-inputs-only", action="store_true",
                        help="CPU-only lineage check; no training, output creation or GPU allocation")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=192)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--parent-learning-rate", type=float, default=5e-5)
    parser.add_argument("--new-learning-rate", type=float, default=2e-4)
    parser.add_argument("--augment-loss-weight", type=float, default=0.25)
    parser.add_argument("--consistency-weight", type=float, default=0.05)
    parser.add_argument("--smoke-updates", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if min(args.batch_size, args.epochs) <= 0 or args.smoke_updates < 0:
        raise ValueError("invalid V3.31 training budget")
    if min(args.parent_learning_rate, args.new_learning_rate) <= 0:
        raise ValueError("learning rates must be positive")
    if min(args.augment_loss_weight, args.consistency_weight) <= 0:
        raise ValueError("joint loss weights must be positive")
    if not args.smoke_updates and args.epochs != 30:
        raise ValueError("formal V3.31 must be planned from the start for 30E")
    if _file_sha256(args.parent) != args.parent_sha256:
        raise ValueError("V3.31 parent checkpoint SHA256 mismatch")
    base = json.loads((args.base_cache / "manifest.json").read_text())
    original = V329Sidecar(args.language_sidecar, args.base_cache)
    augmented = V331AugmentSidecar(
        args.augment_sidecar,
        args.base_cache,
        args.language_sidecar,
        allow_smoke=bool(args.smoke_updates),
    )
    parent = torch.load(args.parent, map_location="cpu", weights_only=True)
    validate_parent(parent, base, original.signature)
    if args.check_inputs_only:
        print(json.dumps({'input_contracts': 'passed', 'parent_sha256': args.parent_sha256,
                          'training_started': False}, ensure_ascii=False))
        return
    args.output.mkdir(parents=True, exist_ok=True)
    if not args.smoke_updates:
        require_disk_budget(args.output, minimum_gib=20)
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise ValueError("V3.31 training requires the preflighted GPU")
    policy, config, groups = make_policy(base, parent, args.seed)
    policy.to(device=device, dtype=torch.float32)
    new_parameters = list(policy.cross_camera_bridge.parameters())
    new_ids = {id(parameter) for parameter in new_parameters}
    parent_parameters = [
        parameter for parameter in policy.parameters()
        if parameter.requires_grad and id(parameter) not in new_ids
    ]
    optimizer = torch.optim.AdamW(
        [
            {"params": parent_parameters, "lr": args.parent_learning_rate},
            {"params": new_parameters, "lr": args.new_learning_rate},
        ],
        weight_decay=0.01,
        fused=True,
    )
    all_parameters = parent_parameters + new_parameters
    _assert_fp32_training_state(all_parameters, optimizer)
    eligible = int(original.manifest["eligible_rows"])
    per_epoch = math.ceil(eligible / args.batch_size)
    formal_total_steps = per_epoch * args.epochs
    total_steps = args.smoke_updates or formal_total_steps
    warmup = max(1, round(total_steps * 0.03))

    def factor(step: int) -> float:
        if step < warmup:
            return (step + 1) / warmup
        fraction = min(1.0, (step - warmup) / max(1, total_steps - warmup))
        return 0.5 * (1 + math.cos(math.pi * fraction))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, factor)
    code_files = (
        "scripts/train_duvla_v3_31.py",
        "src/duvla/models/duvla_v2_1.py",
        "src/duvla/models/ordered_language_bridge.py",
        "src/duvla/models/robust_camera_fusion.py",
        "src/duvla/training/v3_29_data.py",
        "src/duvla/training/v3_31_data.py",
    )
    signature = {
        "version": VERSION,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "seed": args.seed,
        "parent_sha256": args.parent_sha256,
        "base_signature": base["cache_signature"],
        "language_sidecar_sha256": original.signature,
        "augment_sidecar_sha256": augmented.signature,
        "model_config": asdict(config),
        "augment_loss_weight": args.augment_loss_weight,
        "consistency_weight": args.consistency_weight,
        "smoke_updates": args.smoke_updates,
        "code_hashes": {path: sha256(Path(path).read_bytes()).hexdigest() for path in code_files},
    }
    resume_path = args.output / "resume.pt"
    curve_path = args.output / "loss_curve.jsonl"
    if (args.output / "run_manifest.json").exists() and not args.resume:
        raise ValueError("existing V3.31 run requires explicit --resume")
    epoch_start = samples_to_skip = step = original_exposures = augmented_exposures = 0
    saved_rng: dict[str, object] | None = None
    resume_compatibility_fix: dict[str, str] | None = None
    if args.resume:
        saved = torch.load(resume_path, map_location="cpu", weights_only=True)
        saved_signature = saved["run_signature"]
        if saved_signature != signature:
            saved_copy = json.loads(json.dumps(saved_signature))
            current_copy = json.loads(json.dumps(signature))
            code_path = "scripts/train_duvla_v3_31.py"
            saved_hash = saved_copy["code_hashes"].get(code_path)
            current_hash = current_copy["code_hashes"].get(code_path)
            if saved_hash not in RESUME_COMPATIBLE_TRAINER_HASHES:
                raise ValueError("V3.31 resume signature mismatch")
            saved_copy["code_hashes"][code_path] = current_hash
            if saved_copy != current_copy:
                raise ValueError("V3.31 resume differs beyond the audited artifact/parity fix")
            resume_compatibility_fix = {
                "previous_trainer_sha256": str(saved_hash),
                "current_trainer_sha256": str(current_hash),
                "scope": "loss_curve_group_lr_export_and_skip_step0_parent_parity_on_resume",
            }
        policy.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        scheduler.load_state_dict(saved["scheduler"])
        epoch_start = int(saved["epoch_index"])
        samples_to_skip = int(saved["samples_in_epoch"])
        step = int(saved["step"])
        original_exposures = int(saved["original_exposures"])
        augmented_exposures = int(saved["augmented_exposures"])
        saved_rng = saved
        truncate_loss_points(curve_path, max_optimizer_step=step)
    atomic_write_json(
        args.output / "run_manifest.json",
        {
            **signature,
            "formal": not bool(args.smoke_updates),
            "trainable_groups": groups,
            "selected_original_rows": eligible,
            "selected_augmented_rows": augmented.manifest["selected_rows"],
            "planned_optimizer_steps": total_steps,
            "formal_planned_optimizer_steps": formal_total_steps,
            "parameter_counts": {
                "total": sum(parameter.numel() for parameter in policy.parameters()),
                "trainable": sum(parameter.numel() for parameter in all_parameters),
                "new_cross_camera": sum(parameter.numel() for parameter in new_parameters),
            },
            "uses_evaluation_initial_states": False,
            "uses_reward": False,
            "uses_success": False,
            "benchmark_task_index": False,
            "resume_compatibility_fix": resume_compatibility_fix,
        },
    )

    def save(epoch_index: int, samples_in_epoch: int) -> None:
        _atomic_torch_save(
            resume_path,
            {
                "run_signature": signature,
                "model": policy.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "epoch_index": epoch_index,
                "samples_in_epoch": samples_in_epoch,
                "step": step,
                "original_exposures": original_exposures,
                "augmented_exposures": augmented_exposures,
                "torch_rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state(),
            },
        )

    def checkpoint(epoch: int) -> dict[str, object]:
        return {
            "format": "duvla_v3_31",
            "version": VERSION,
            "stage": "flow",
            "epoch": epoch,
            "optimizer_step": step,
            "effective_samples": original_exposures,
            "effective_epochs": original_exposures / eligible,
            "augmented_exposures": augmented_exposures,
            "planned_epochs": 30,
            "training_complete": epoch == 30,
            "checkpoint_complete": True,
            "formal": not bool(args.smoke_updates),
            "model_config": asdict(config),
            "model_state_dict": policy.state_dict(),
            "cache_signature": base["cache_signature"],
            "sidecar_sha256": original.signature,
            "augmentation_sidecar_sha256": augmented.signature,
            "data_contract": original.manifest,
            "augmentation_contract": augmented.manifest,
            "parent_checkpoint_sha256": args.parent_sha256,
            "action_offset": 1,
            "camera_size": 128,
            "candidate_aggregation": "coordinate_median",
            "task_routing": "natural_language",
            "benchmark_task_index": False,
            "uses_evaluation_initial_states": False,
            "uses_reward": False,
            "uses_success": False,
            "trainable_groups": groups,
            "run_signature": signature,
        }

    if saved_rng is not None:
        torch.set_rng_state(saved_rng["torch_rng"])
        torch.cuda.set_rng_state(saved_rng["cuda_rng"])
    else:
        _seed_everything(args.seed + 1)
    task_weights = _task_weights(base)
    started = time.monotonic()
    initial_original = original_exposures
    window: Counter[str] = Counter()
    window_rows = 0
    milestones = {1, 3, 6, 8, 10, 15, 20, 25, 30}
    parity_checked = saved_rng is not None
    first_smoke_loss: float | None = None
    final_smoke_loss: float | None = None
    for epoch in range(epoch_start, args.epochs):
        if not args.smoke_updates:
            require_disk_budget(args.output, minimum_gib=12)
        policy.train()
        samples = 0
        tasks: Counter[int] = Counter()
        episodes: set[int] = set()
        unique: set[int] = set()
        iterator = iter_v2_1_batches(
            args.base_cache,
            batch_size=args.batch_size,
            history_length=4,
            history_stride=2,
            seed=args.seed,
            epoch=epoch,
            shuffle=not bool(args.smoke_updates),
            shard_shuffle_block_size=4,
            mmap_shards=True,
            row_transform=original.transform,
        )
        batches = prefetch_v2_1_batches(iterator, pin_memory=True)
        try:
            if args.smoke_updates:
                first_batch = next(batches)
                batch_stream = repeat(first_batch, args.smoke_updates)
            else:
                batch_stream = batches
            for batch in batch_stream:
                count = len(batch["dataset_indices"])
                tasks.update(batch["task_indices"].tolist())
                episodes.update(batch["episode_indices"].tolist())
                unique.update(batch["dataset_indices"].tolist())
                if epoch == epoch_start and samples < samples_to_skip:
                    samples += count
                    if samples > samples_to_skip:
                        raise ValueError("V3.31 resume cursor is not on a batch boundary")
                    continue
                paired = augmented.fetch(batch)
                moved = {
                    name: value.to(
                        device=device,
                        dtype=torch.bfloat16 if value.is_floating_point() else value.dtype,
                        non_blocking=True,
                    )
                    for name, value in batch.items()
                }
                context = {
                    name: moved[name]
                    for name in (
                        "history_visual",
                        "history_semantic",
                        "history_states",
                        "previous_actions",
                        "language_tokens",
                        "language_mask",
                    )
                }
                if not parity_checked:
                    policy.eval()
                    sample_context = dict(context)
                    sample_context["previous_action"] = sample_context.pop("previous_actions")
                    noise = torch.randn(
                        count, 1, config.action_horizon, config.action_dim,
                        device=device, dtype=torch.bfloat16,
                    )
                    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                        enabled = policy.sample_actions(
                            moved["visual"], moved["semantic"], moved["states"],
                            flow_samples=1, noise=noise, apply_direct=False,
                            apply_instruction=False, apply_gripper_event=False, **sample_context,
                        )
                        bridge = policy.cross_camera_bridge
                        policy.cross_camera_bridge = None
                        disabled = policy.sample_actions(
                            moved["visual"], moved["semantic"], moved["states"],
                            flow_samples=1, noise=noise, apply_direct=False,
                            apply_instruction=False, apply_gripper_event=False, **sample_context,
                        )
                        policy.cross_camera_bridge = bridge
                    if not torch.equal(enabled, disabled):
                        raise RuntimeError("V3.31 zero-initialized parent parity failed")
                    parity_checked = True
                    policy.train()
                optimizer.zero_grad(set_to_none=True)
                weights = task_weights.index_select(0, batch["task_indices"]).to(
                    device=device, dtype=torch.bfloat16
                )
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    original_noise = torch.zeros_like(moved["actions"]) if args.smoke_updates else None
                    original_time = (
                        torch.full((count,), 0.5, device=device)
                        if args.smoke_updates else None
                    )
                    losses = policy.flow_loss_components(
                        moved["visual"], moved["semantic"], moved["states"],
                        moved["actions"], moved["valid_mask"],
                        sample_weights=weights, noise=original_noise,
                        time=original_time, **context,
                    )
                    if paired is not None:
                        slots = paired["slots"].to(device)
                        pair = {
                            name: value.to(
                                device=device,
                                dtype=torch.bfloat16 if value.is_floating_point() else value.dtype,
                                non_blocking=True,
                            )
                            for name, value in paired.items() if name != "slots"
                        }
                        pair_context = {
                            "history_visual": _select(moved["history_visual"], slots),
                            "history_semantic": _select(moved["history_semantic"], slots),
                            "history_states": _select(moved["history_states"], slots),
                            "previous_actions": _select(moved["previous_actions"], slots),
                            "language_tokens": pair["language_tokens"],
                            "language_mask": pair["language_mask"],
                        }
                        target = _select(moved["actions"], slots)
                        valid = _select(moved["valid_mask"], slots)
                        pair_weights = _select(weights, slots)
                        pair_noise = (
                            torch.zeros_like(target)
                            if args.smoke_updates else torch.randn_like(target)
                        )
                        pair_time = (
                            torch.full((len(slots),), 0.5, device=device)
                            if args.smoke_updates
                            else torch.distributions.Beta(1.5, 1.0).sample(
                                (len(slots),)
                            ).to(device).clamp_(0.001, 0.999)
                        )
                        pair_noisy = (
                            (1.0 - pair_time[:, None, None]) * pair_noise
                            + pair_time[:, None, None] * target
                        )
                        augmented_losses = policy.flow_loss_components(
                            pair["visual"], pair["semantic"], _select(moved["states"], slots),
                            target, valid, sample_weights=pair_weights,
                            noise=pair_noise, time=pair_time, **pair_context,
                        )
                        with torch.no_grad():
                            original_velocity = policy.predict_flow_velocity(
                                _select(moved["visual"], slots),
                                _select(moved["semantic"], slots),
                                _select(moved["states"], slots),
                                pair_noisy,
                                pair_time,
                                history_visual=pair_context["history_visual"],
                                history_semantic=pair_context["history_semantic"],
                                history_states=pair_context["history_states"],
                                previous_action=pair_context["previous_actions"],
                                language_tokens=_select(moved["language_tokens"], slots),
                                language_mask=_select(moved["language_mask"], slots),
                            )
                        augmented_velocity = policy.predict_flow_velocity(
                            pair["visual"], pair["semantic"], _select(moved["states"], slots),
                            pair_noisy, pair_time,
                            history_visual=pair_context["history_visual"],
                            history_semantic=pair_context["history_semantic"],
                            history_states=pair_context["history_states"],
                            previous_action=pair_context["previous_actions"],
                            language_tokens=pair_context["language_tokens"],
                            language_mask=pair_context["language_mask"],
                        )
                        mask = valid[:, :, None].expand_as(augmented_velocity)
                        consistency = (
                            (augmented_velocity.float() - original_velocity.float()).square()
                            * mask
                        ).sum() / mask.sum().clamp_min(1)
                        losses["augmented_flow"] = (
                            args.augment_loss_weight * augmented_losses["flow"]
                        )
                        losses["paired_consistency"] = args.consistency_weight * consistency
                        augmented_exposures += len(slots)
                    total_loss = sum(losses.values())
                if args.smoke_updates:
                    current_smoke_loss = float(total_loss.detach())
                    if first_smoke_loss is None:
                        first_smoke_loss = current_smoke_loss
                    final_smoke_loss = current_smoke_loss
                if any(not torch.isfinite(value).all() for value in losses.values()):
                    raise RuntimeError("nonfinite V3.31 loss")
                total_loss.backward()
                if step == 0:
                    if policy.cross_camera_bridge.output.weight.grad is None:
                        raise RuntimeError("V3.31 cross-camera bridge has no gradient")
                    if policy.language_bridge.gate.grad is None:
                        raise RuntimeError("V3.31 language bridge has no gradient")
                norm = torch.nn.utils.clip_grad_norm_(all_parameters, 1.0, error_if_nonfinite=True)
                optimizer.step(); scheduler.step()
                step += 1; samples += count; original_exposures += count
                losses = {**losses, "total": total_loss, "grad_norm": norm}
                for name, value in losses.items():
                    window[name] += float(value.detach()) * count
                window_rows += count
                if step % 100 == 0 or (args.smoke_updates and step >= args.smoke_updates) or samples == eligible:
                    point = {
                        "optimizer_step": step,
                        "effective_epochs": original_exposures / eligible,
                        "augmented_exposures": augmented_exposures,
                        "learning_rates": [group["lr"] for group in optimizer.param_groups],
                        "losses": {name: value / window_rows for name, value in window.items()},
                    }
                    append_loss_point(curve_path, point)
                    window.clear(); window_rows = 0
                    atomic_write_json(
                        args.output / "training_state.json",
                        {
                            "阶段": "V3.31联合训练",
                            "state": "running",
                            "optimizer_step": step,
                            "planned_optimizer_steps": total_steps,
                            "effective_epochs": original_exposures / eligible,
                            "augmented_exposures": augmented_exposures,
                            "samples_per_second": (original_exposures - initial_original) / max(time.monotonic() - started, 1e-6),
                            "peak_gpu_gib": torch.cuda.max_memory_allocated() / 2**30,
                            "latest_loss": point["losses"],
                        },
                    )
                if step % 500 == 0:
                    if not args.smoke_updates:
                        require_disk_budget(args.output, minimum_gib=12)
                    save(epoch, samples)
                if step % 5000 == 0:
                    print(f"V3.31 {step}/{total_steps} updates, {original_exposures/eligible:.2f}E", flush=True)
                if args.smoke_updates and step >= args.smoke_updates:
                    if args.smoke_updates > 1 and (
                        first_smoke_loss is None
                        or final_smoke_loss is None
                        or final_smoke_loss >= 0.98 * first_smoke_loss
                    ):
                        raise RuntimeError(
                            f"V3.31 repeated-batch fit failed: {first_smoke_loss} -> {final_smoke_loss}"
                        )
                    save(epoch, samples)
                    atomic_write_json(
                        args.output / "smoke.json",
                        {
                            "passed": True,
                            "kind": "single_update_wiring" if args.smoke_updates == 1 else "repeated_batch_fit",
                            "updates": step,
                            "parent_parity": parity_checked,
                            "peak_gpu_gib": torch.cuda.max_memory_allocated() / 2**30,
                            "augmented_exposures": augmented_exposures,
                            "repeated_batch_initial_loss": first_smoke_loss,
                            "repeated_batch_final_loss": final_smoke_loss,
                            "relative_loss_change": final_smoke_loss / first_smoke_loss - 1.0,
                            "seconds": time.monotonic() - started,
                        },
                    )
                    atomic_write_json(args.output / 'training_state.json', {
                        'state': 'smoke_completed', 'formal': False, 'optimizer_step': step,
                        'effective_epochs': original_exposures / eligible,
                        'augmented_exposures': augmented_exposures,
                    })
                    return
        finally:
            batches.close()
        if samples != eligible or len(unique) != eligible or len(episodes) != 2000 or len(tasks) != 40:
            raise RuntimeError(
                f"incomplete V3.31 epoch: {samples}/{eligible}, episodes={len(episodes)}, tasks={len(tasks)}"
            )
        atomic_write_json(
            args.output / f"coverage_{epoch+1}e.json",
            {
                "original_unique_rows": len(unique),
                "episodes": len(episodes),
                "tasks": dict(tasks),
                "augmented_exposures_total": augmented_exposures,
            },
        )
        save(epoch + 1, 0)
        if epoch + 1 in milestones:
            _atomic_torch_save(args.output / f"duvla-v3.31-{epoch+1}e.pt", checkpoint(epoch + 1))
            write_loss_curve_artifacts(
                curve_path,
                csv_path=args.output / f"loss_curve_{epoch+1}e.csv",
                svg_path=args.output / f"loss_curve_{epoch+1}e.svg",
                title=f"Duvla V3.31 {epoch+1}E",
            )
        write_loss_curve_artifacts(
            curve_path,
            csv_path=args.output / "loss_curve.csv",
            svg_path=args.output / "loss_curve.svg",
            title="Duvla V3.31",
        )
        samples_to_skip = 0
    atomic_write_json(
        args.output / "training_state.json",
        {
            "state": "completed",
            "optimizer_step": step,
            "effective_epochs": original_exposures / eligible,
            "augmented_exposures": augmented_exposures,
            "planned_optimizer_steps": total_steps,
        },
    )


if __name__ == "__main__":
    main()
