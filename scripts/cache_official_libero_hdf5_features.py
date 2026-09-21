#!/usr/bin/env python3
"""Cache frozen-Qwen features from all 2,000 official LIBERO demonstrations.

V3.27 deliberately has no demonstration validation split.  All training
demonstrations are used for the final base-policy fit; official evaluation
initial states, rewards, success flags, and simulator states are never read.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
import json
import math
from pathlib import Path

import h5py
import numpy as np
import torch
from PIL import Image

from duvla.evaluation.libero_contract import (
    libero_global_task_index,
    orient_libero_view,
    resolve_benchmark_task_index,
)
from duvla.models import QwenVLBackbone
from duvla.training.feature_cache import (
    FeatureShard,
    atomic_write_json,
    prepare_cache_resume,
    save_feature_shard,
)
from duvla.training.libero_branch import hdf5_control_frequency


OFFICIAL_REPOSITORY = "yifengzhu-hf/LIBERO-datasets"
OFFICIAL_REVISION = "f13aa24a3da8c43c7225569f28c562979fa0e35a"
SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")
HIDDEN_LAYERS = (12, 14, 18, -1)


@dataclass(frozen=True)
class TaskSource:
    path: Path
    suite: str
    suite_task_index: int
    global_task_index: int
    instruction: str


@dataclass(frozen=True)
class DemoSource:
    task: TaskSource
    demo_key: str
    episode_index: int
    dataset_start: int
    frames: int


@dataclass(frozen=True)
class DatasetAudit:
    tasks: tuple[TaskSource, ...]
    demos: tuple[DemoSource, ...]
    total_frames: int
    zero_arm_rows: int
    state_mean: np.ndarray
    state_std: np.ndarray
    action_mean: np.ndarray
    action_std: np.ndarray
    source_contract_sha256: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--horizon", type=int, default=8)
    parser.add_argument("--shard-size", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--log-every-shards", type=int, default=100)
    parser.add_argument("--limit", type=int, default=0, help="smoke-test rows only")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def _demo_number(key: str) -> int:
    try:
        prefix, value = key.rsplit("_", 1)
        if prefix != "demo":
            raise ValueError
        return int(value)
    except ValueError as exc:
        raise ValueError(f"invalid LIBERO demonstration key: {key!r}") from exc


def _normalise_text(value: str) -> str:
    return " ".join(value.split()).casefold()


def _task_sources(root: Path) -> tuple[TaskSource, ...]:
    try:
        from libero.libero import benchmark
    except ImportError as exc:  # pragma: no cover - integration environment only
        raise RuntimeError("the LIBERO package is required to audit official HDF5 tasks") from exc

    discovered: list[TaskSource] = []
    for suite_name in SUITES:
        directory = root / suite_name
        paths = sorted(directory.glob("*_demo.hdf5"))
        if len(paths) != 10:
            raise ValueError(f"{suite_name} must contain exactly 10 HDF5 task files")
        suite = benchmark.get_benchmark(suite_name)(task_order_index=0)
        instructions = tuple(
            str(suite.get_task(index).language) for index in range(suite.get_num_tasks())
        )
        if len(instructions) != 10:
            raise ValueError(f"{suite_name} benchmark must contain exactly 10 tasks")
        used_local_indices: set[int] = set()
        for path in paths:
            with h5py.File(path, "r") as source:
                data = source.get("data")
                if data is None:
                    raise ValueError(f"HDF5 file has no data group: {path}")
                problem_info = json.loads(str(data.attrs["problem_info"]))
                instruction = str(problem_info["language_instruction"])
                local_index = resolve_benchmark_task_index(instruction, instructions)
                if local_index in used_local_indices:
                    raise ValueError(f"duplicate task language in {suite_name}: {instruction}")
                used_local_indices.add(local_index)
                if hdf5_control_frequency(data.attrs["env_args"]) != 20:
                    raise ValueError(f"official source is not 20 Hz: {path}")
                if int(data.attrs["num_demos"]) != 50 or len(data) != 50:
                    raise ValueError(f"official task must contain 50 demonstrations: {path}")
            discovered.append(
                TaskSource(
                    path=path,
                    suite=suite_name,
                    suite_task_index=local_index,
                    global_task_index=libero_global_task_index(suite_name, local_index),
                    instruction=instruction,
                )
            )
    discovered.sort(key=lambda item: item.global_task_index)
    indices = [item.global_task_index for item in discovered]
    if indices != list(range(40)):
        raise ValueError(f"global task mapping is not the exact range 0..39: {indices}")
    if len({_normalise_text(item.instruction) for item in discovered}) != 40:
        raise ValueError("official task instructions are not unique")
    return tuple(discovered)


def _robot_states(observation: h5py.Group) -> np.ndarray:
    if "ee_states" not in observation or "gripper_states" not in observation:
        raise ValueError("official observation lacks ee_states/gripper_states")
    ee = np.asarray(observation["ee_states"], dtype=np.float32)
    gripper = np.asarray(observation["gripper_states"], dtype=np.float32)
    if ee.ndim != 2 or ee.shape[1] != 6 or gripper.shape != (ee.shape[0], 2):
        raise ValueError("official robot state must be [N,6]+[N,2]")
    result = np.concatenate((ee, gripper), axis=1)
    if not np.isfinite(result).all():
        raise ValueError("official robot state contains non-finite values")
    return result


def _safe_std(values: np.ndarray) -> np.ndarray:
    result = np.asarray(values, dtype=np.float64)
    return np.where(result == 0.0, 1.0, result).astype(np.float32)


def _dataset_audit(root: Path) -> DatasetAudit:
    tasks = _task_sources(root)
    demos: list[DemoSource] = []
    state_sum = np.zeros(8, dtype=np.float64)
    state_square_sum = np.zeros(8, dtype=np.float64)
    action_sum = np.zeros(7, dtype=np.float64)
    action_square_sum = np.zeros(7, dtype=np.float64)
    total_frames = 0
    zero_arm_rows = 0
    episode_index = 0
    contract_rows: list[dict[str, object]] = []
    for task in tasks:
        with h5py.File(task.path, "r") as source:
            data = source["data"]
            for demo_key in sorted(data.keys(), key=_demo_number):
                demo = data[demo_key]
                actions = np.asarray(demo["actions"], dtype=np.float32)
                states = _robot_states(demo["obs"])
                frames = int(actions.shape[0])
                if actions.shape != (frames, 7) or states.shape != (frames, 8):
                    raise ValueError(f"action/state contract mismatch: {task.path}/{demo_key}")
                if not np.isfinite(actions).all():
                    raise ValueError(f"non-finite actions: {task.path}/{demo_key}")
                for camera in ("agentview_rgb", "eye_in_hand_rgb"):
                    if demo["obs"][camera].shape != (frames, 128, 128, 3):
                        raise ValueError(f"camera contract mismatch: {task.path}/{demo_key}/{camera}")
                demos.append(
                    DemoSource(
                        task=task,
                        demo_key=str(demo_key),
                        episode_index=episode_index,
                        dataset_start=total_frames,
                        frames=frames,
                    )
                )
                state64 = states.astype(np.float64)
                action64 = actions.astype(np.float64)
                state_sum += state64.sum(axis=0)
                state_square_sum += np.square(state64).sum(axis=0)
                action_sum += action64.sum(axis=0)
                action_square_sum += np.square(action64).sum(axis=0)
                zero_arm_rows += int(np.count_nonzero(np.all(actions[:, :6] == 0.0, axis=1)))
                contract_rows.append(
                    {
                        "task": task.global_task_index,
                        "file": str(task.path.relative_to(root)),
                        "demo": str(demo_key),
                        "frames": frames,
                    }
                )
                total_frames += frames
                episode_index += 1
    if len(tasks) != 40 or len(demos) != 2000 or total_frames != 338_575:
        raise ValueError(
            f"official dataset mismatch: tasks={len(tasks)}, demos={len(demos)}, "
            f"frames={total_frames}"
        )
    state_mean = state_sum / total_frames
    action_mean = action_sum / total_frames
    state_std = np.sqrt(np.maximum(state_square_sum / total_frames - state_mean**2, 0.0))
    action_std = np.sqrt(np.maximum(action_square_sum / total_frames - action_mean**2, 0.0))
    contract = json.dumps(contract_rows, sort_keys=True, separators=(",", ":")).encode()
    return DatasetAudit(
        tasks=tasks,
        demos=tuple(demos),
        total_frames=total_frames,
        zero_arm_rows=zero_arm_rows,
        state_mean=state_mean.astype(np.float32),
        state_std=_safe_std(state_std),
        action_mean=action_mean.astype(np.float32),
        action_std=_safe_std(action_std),
        source_contract_sha256=sha256(contract).hexdigest(),
    )


def build_action_chunks(
    actions: np.ndarray,
    *,
    mean: np.ndarray,
    std: np.ndarray,
    horizon: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Standardise future chunks without crossing a demonstration boundary."""

    values = np.asarray(actions, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] != 7 or horizon <= 0:
        raise ValueError("actions must have shape [N,7] and horizon must be positive")
    if mean.shape != (7,) or std.shape != (7,) or np.any(std <= 0.0):
        raise ValueError("action statistics must have shape [7] and positive std")
    chunks = np.zeros((values.shape[0], horizon, 7), dtype=np.float32)
    mask = np.zeros((values.shape[0], horizon), dtype=np.bool_)
    normalized = (values - mean) / std
    for offset in range(horizon):
        count = max(values.shape[0] - offset, 0)
        if count:
            chunks[:count, offset] = normalized[offset:]
            mask[:count, offset] = True
    return chunks, mask


def _indices_sha256(count: int) -> str:
    encoded = ",".join(str(index) for index in range(count)).encode()
    return sha256(encoded).hexdigest()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def main() -> None:
    args = parse_args()
    if min(args.horizon, args.shard_size, args.batch_size, args.log_every_shards) <= 0:
        raise SystemExit("horizon, shard/batch sizes and logging interval must be positive")
    if args.limit < 0:
        raise SystemExit("limit cannot be negative")
    root = args.dataset_root.resolve()
    audit = _dataset_audit(root)
    selected_frames = min(audit.total_frames, args.limit) if args.limit else audit.total_frames
    formal = args.limit == 0
    if formal and audit.zero_arm_rows != 760:
        raise SystemExit(
            f"expected 760 zero-arm rows in official source, found {audit.zero_arm_rows}"
        )
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but unavailable")
    backbone = QwenVLBackbone()
    signature: dict[str, object] = {
        "version": 1,
        "candidate": "Duvla V3.27",
        "dataset_root": str(root),
        "source_repository": OFFICIAL_REPOSITORY,
        "source_revision": OFFICIAL_REVISION,
        "source_contract_sha256": audit.source_contract_sha256,
        "split": "all_demonstrations_train" if formal else "official_demonstration_smoke",
        "all_demonstrations": formal,
        "smoke_cache": not formal,
        "zero_arm_rows_retained": True,
        "horizon": args.horizon,
        "selected_frames": selected_frames,
        "selected_indices_sha256": _indices_sha256(selected_frames),
        "state_mean": audit.state_mean.tolist(),
        "state_std": audit.state_std.tolist(),
        "action_mean": audit.action_mean.tolist(),
        "action_std": audit.action_std.tolist(),
        "qwen_model_path": str(backbone.config.model_path),
        "qwen_dtype": "bfloat16" if backbone.config.use_bfloat16 else "float32",
        "qwen_hidden_layers": list(HIDDEN_LAYERS),
        "qwen_context_mode": "highres_layer14_multilayer_semantic",
        "context_layout": "per_camera",
        "tokens_per_camera": 64,
        "shard_size": args.shard_size,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    selected_indices = range(selected_frames)
    try:
        resume_plan = prepare_cache_resume(
            args.output,
            signature=signature,
            selected_indices=selected_indices,
            shard_size=args.shard_size,
            resume=args.resume,
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise SystemExit(str(exc)) from exc
    if resume_plan.complete:
        print(f"cache already complete and signature verified: {args.output}", flush=True)
        return
    completed_count = len(resume_plan.completed_indices)
    shard_number = resume_plan.next_shard_number
    atomic_write_json(
        args.output / "cache_state.json",
        {
            "state": "running",
            "cache_signature": signature,
            "completed_samples": completed_count,
            "selected_frames": selected_frames,
            "completed_shards": shard_number,
            "updated_at": _utc_now(),
        },
    )
    print(
        f"official2000_audit tasks=40 episodes=2000 frames={audit.total_frames} "
        f"zero_arm_rows_retained={audit.zero_arm_rows} selected={selected_frames}",
        flush=True,
    )
    backbone.load(device=device)
    buffer: list[
        tuple[int, int, int, int, str, Image.Image, Image.Image, np.ndarray, np.ndarray, np.ndarray]
    ] = []

    def flush() -> None:
        nonlocal buffer, shard_number, completed_count
        if not buffer:
            return
        feature_parts: list[torch.Tensor] = []
        semantic_parts: list[torch.Tensor] = []
        for start in range(0, len(buffer), args.batch_size):
            batch = buffer[start : start + args.batch_size]
            instructions = [item[4] for item in batch]
            agent_inputs = backbone.prepare_view_batch_inputs(
                [(item[5], item[4]) for item in batch]
            )
            wrist_inputs = backbone.prepare_view_batch_inputs(
                [(item[6], item[4]) for item in batch]
            )
            with torch.inference_mode():
                visual, semantic = backbone.forward_multiview_multilayer_spatial_semantic_context(
                    (agent_inputs, wrist_inputs),
                    instructions,
                    hidden_layers=HIDDEN_LAYERS,
                    expected_grid=(8, 8),
                    output_grid=(8, 8),
                )
            feature_parts.append(visual[:, HIDDEN_LAYERS.index(14) : HIDDEN_LAYERS.index(14) + 1].cpu())
            semantic_parts.append(semantic.cpu())
        save_feature_shard(
            args.output / f"shard-{shard_number:06d}.pt",
            FeatureShard(
                features=torch.cat(feature_parts),
                semantic_features=torch.cat(semantic_parts),
                states=torch.from_numpy(np.stack([item[7] for item in buffer])),
                actions=torch.from_numpy(np.stack([item[8] for item in buffer])),
                valid_mask=torch.from_numpy(np.stack([item[9] for item in buffer])),
                dataset_indices=tuple(item[0] for item in buffer),
                episode_indices=torch.tensor([item[1] for item in buffer], dtype=torch.long),
                frame_indices=torch.tensor([item[2] for item in buffer], dtype=torch.long),
                task_indices=torch.tensor([item[3] for item in buffer], dtype=torch.long),
            ),
        )
        completed_count += len(buffer)
        shard_number += 1
        buffer = []
        atomic_write_json(
            args.output / "cache_state.json",
            {
                "state": "running",
                "cache_signature": signature,
                "completed_samples": completed_count,
                "selected_frames": selected_frames,
                "completed_shards": shard_number,
                "updated_at": _utc_now(),
            },
        )
        if shard_number % args.log_every_shards == 0 or completed_count == selected_frames:
            print(
                f"cache_progress version=V3.27 shards={shard_number} "
                f"samples={completed_count}/{selected_frames} "
                f"percent={100.0 * completed_count / selected_frames:.1f}",
                flush=True,
            )

    for source_meta in audit.demos:
        demo_end = source_meta.dataset_start + source_meta.frames
        if source_meta.dataset_start >= selected_frames:
            break
        if demo_end <= completed_count:
            continue
        with h5py.File(source_meta.task.path, "r") as source:
            demo = source["data"][source_meta.demo_key]
            actions = np.asarray(demo["actions"], dtype=np.float32)
            states = _robot_states(demo["obs"])
            chunks, masks = build_action_chunks(
                actions,
                mean=audit.action_mean,
                std=audit.action_std,
                horizon=args.horizon,
            )
            normalized_states = (states - audit.state_mean) / audit.state_std
            local_start = max(completed_count - source_meta.dataset_start, 0)
            local_stop = min(source_meta.frames, selected_frames - source_meta.dataset_start)
            for frame_index in range(local_start, local_stop):
                dataset_index = source_meta.dataset_start + frame_index
                agent = Image.fromarray(
                    orient_libero_view(demo["obs"]["agentview_rgb"][frame_index])
                )
                wrist = Image.fromarray(
                    orient_libero_view(demo["obs"]["eye_in_hand_rgb"][frame_index])
                )
                buffer.append(
                    (
                        dataset_index,
                        source_meta.episode_index,
                        frame_index,
                        source_meta.task.global_task_index,
                        source_meta.task.instruction,
                        agent,
                        wrist,
                        normalized_states[frame_index].astype(np.float32),
                        chunks[frame_index],
                        masks[frame_index],
                    )
                )
                if len(buffer) == args.shard_size:
                    flush()
    flush()
    if completed_count != selected_frames:
        raise SystemExit(
            f"cache stopped at {completed_count} rows, expected {selected_frames}"
        )
    manifest: dict[str, object] = {
        "candidate": "Duvla V3.27",
        "dataset_root": str(root),
        "source_repository": OFFICIAL_REPOSITORY,
        "source_revision": OFFICIAL_REVISION,
        "source_contract_sha256": audit.source_contract_sha256,
        "source_control_frequency_hz": 20,
        "source_camera_resolution": [128, 128],
        "seed": 17,
        "split": "all_demonstrations_train" if formal else "official_demonstration_smoke",
        "validation_fraction": 0.0,
        "all_demonstrations": formal,
        "smoke_cache": not formal,
        "uses_evaluation_initial_states": False,
        "uses_rewards": False,
        "uses_success": False,
        "zero_arm_rows_retained": True,
        "zero_arm_rows": audit.zero_arm_rows,
        "horizon": args.horizon,
        "context_tokens": 128,
        "tokens_per_camera": 64,
        "camera_count": 2,
        "context_layout": "per_camera",
        "shard_size": args.shard_size,
        "batch_size": args.batch_size,
        "total_frames": audit.total_frames,
        "total_episodes": 2000,
        "total_tasks": 40,
        "selected_frames": selected_frames,
        "selected_task_sample_counts": {
            str(task.global_task_index): sum(
                demo.frames for demo in audit.demos if demo.task == task
            )
            for task in audit.tasks
        },
        "selected_task_indices": list(range(40)),
        "selected_task_count": 40,
        "train_episodes": 2000 if formal else None,
        "validation_episodes": 0 if formal else None,
        "feature_dim": backbone.output_dim,
        "action_dim": 7,
        "normalization": "all-demonstrations per-dimension mean/std for state and action",
        "state_mean": audit.state_mean.tolist(),
        "state_std": audit.state_std.tolist(),
        "action_mean": audit.action_mean.tolist(),
        "action_std": audit.action_std.tolist(),
        "schema_version": 8,
        "qwen_model_path": str(backbone.config.model_path),
        "qwen_local_files_only": backbone.config.local_files_only,
        "qwen_dtype": "bfloat16" if backbone.config.use_bfloat16 else "float32",
        "qwen_hidden_layer": 14,
        "qwen_hidden_layers": list(HIDDEN_LAYERS),
        "qwen_hidden_state": "hidden_states[14] visual; hidden_states[12,14,18,-1] semantic",
        "qwen_context_mode": "highres_layer14_multilayer_semantic",
        "qwen_semantic_tokens": 1,
        "qwen_semantic_layer": -1,
        "qwen_grounding_tokens": 0,
        "spatial_grid_height": 8,
        "spatial_grid_width": 8,
        "context_reduction": "per_camera_layer14_exact_image_grid_8x8_plus_multilayer_post_instruction_summary",
        "camera_keys": ["agentview_rgb", "eye_in_hand_rgb"],
        "camera_orientation": "official HDF5 views flipped 180 degrees to match runtime contract",
        "action_semantics": "native LIBERO 7D action; gripper -1=open, +1=close",
        "cache_signature": signature,
    }
    atomic_write_json(args.output / "manifest.json", manifest)
    atomic_write_json(
        args.output / "cache_state.json",
        {
            "state": "completed",
            "cache_signature": signature,
            "completed_samples": selected_frames,
            "selected_frames": selected_frames,
            "completed_shards": math.ceil(selected_frames / args.shard_size),
            "updated_at": _utc_now(),
        },
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
