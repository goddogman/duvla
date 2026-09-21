#!/usr/bin/env python3
"""Create the bounded V3.31 paired photometric/long-language sidecar."""
from __future__ import annotations

import argparse
from collections import defaultdict
from hashlib import sha256
import json
from pathlib import Path
import random
import time

import h5py
import numpy as np
from PIL import Image, ImageEnhance
import torch

from cache_official_libero_hdf5_features import _dataset_audit
from duvla.evaluation.libero_contract import orient_libero_view
from duvla.models.qwen_backbone import QwenVLBackbone
from duvla.training.feature_cache import atomic_write_json
from duvla.training.resource_budget import require_disk_budget


WRAPPERS = (
    "Please carefully complete exactly this robot manipulation task and no other task: {instruction}. Use the visual observations to finish the stated instruction.",
    "Your only objective is the following robot manipulation instruction: {instruction}. Carefully use both camera observations and complete that exact objective.",
    "Follow this manipulation request precisely while avoiding unrelated actions: {instruction}. Continue until the requested physical outcome is complete.",
)


def _selected_rows(audit: object, per_task: int) -> list[tuple[int, object, int]]:
    groups: dict[int, list[object]] = defaultdict(list)
    for demo in audit.demos:
        groups[int(demo.task.global_task_index)].append(demo)
    if set(groups) != set(range(40)):
        raise ValueError("official data must cover global task indices 0..39")
    result: list[tuple[int, object, int]] = []
    for task_index in range(40):
        demos = sorted(groups[task_index], key=lambda item: item.episode_index)
        base, remainder = divmod(per_task, len(demos))
        for number, demo in enumerate(demos):
            allocation = base + (1 if number < remainder else 0)
            if allocation <= 0:
                continue
            if demo.frames < 2:
                raise ValueError("demonstration has no next-action target")
            frames = np.linspace(0, demo.frames - 2, allocation, dtype=np.int64)
            if len(set(int(value) for value in frames)) != allocation:
                raise ValueError("demonstration too short for unique phase samples")
            result.extend(
                (demo.dataset_start + int(frame), demo, int(frame)) for frame in frames
            )
    result.sort(key=lambda item: item[0])
    if len(result) != 40 * per_task or len({row[0] for row in result}) != len(result):
        raise RuntimeError("V3.31 sampler did not produce the exact unique row budget")
    return result


def _augment(image: np.ndarray, *, seed: int) -> Image.Image:
    rng = random.Random(seed)
    output = Image.fromarray(orient_libero_view(image))
    output = ImageEnhance.Brightness(output).enhance(rng.uniform(0.90, 1.10))
    output = ImageEnhance.Contrast(output).enhance(rng.uniform(0.90, 1.10))
    output = ImageEnhance.Color(output).enhance(rng.uniform(0.92, 1.08))
    return output


def _atomic_save(path: Path, payload: dict[str, object]) -> None:
    temporary = path.with_suffix(".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-cache", type=Path, required=True)
    parser.add_argument("--language-sidecar", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rows-per-task", type=int, default=250)
    parser.add_argument("--shard-size", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--max-cache-gib", type=float, default=10.0)
    parser.add_argument("--limit", type=int, default=0, help="isolated smoke cache rows")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if min(args.rows_per_task, args.shard_size, args.batch_size, args.max_cache_gib) <= 0 or args.max_tokens < 32:
        raise ValueError("invalid V3.31 cache dimensions")
    if args.limit < 0:
        raise ValueError("limit must be nonnegative")
    base_path = args.base_cache / "manifest.json"
    language_path = args.language_sidecar / "manifest.json"
    base = json.loads(base_path.read_text())
    language_manifest = json.loads(language_path.read_text())
    audit = _dataset_audit(args.dataset_root)
    if base.get("source_contract_sha256") != audit.source_contract_sha256:
        raise ValueError("V3.31 source/base contract mismatch")
    if not base.get("all_demonstrations") or language_manifest.get("complete") is not True:
        raise ValueError("V3.31 requires the complete all-demonstration V3.29 parent data")
    selected = _selected_rows(audit, args.rows_per_task)
    formal = args.limit == 0
    if args.limit:
        selected = selected[: args.limit]
    selected_indices = [row[0] for row in selected]
    contract = {
        "version": "V3.31",
        "format": "v331_sparse_paired_photometric_long_language",
        "formal": formal,
        "base_manifest_sha256": sha256(base_path.read_bytes()).hexdigest(),
        "language_manifest_sha256": sha256(language_path.read_bytes()).hexdigest(),
        "source_contract_sha256": audit.source_contract_sha256,
        "rows_per_task": args.rows_per_task,
        "selected_rows": len(selected),
        "selected_indices": selected_indices,
        "shard_size": args.shard_size,
        "batch_size": args.batch_size,
        "max_tokens": args.max_tokens,
        "max_cache_gib": args.max_cache_gib,
        "instruction_wrappers": list(WRAPPERS),
        "augmentation": {
            "brightness": [0.90, 1.10],
            "contrast": [0.90, 1.10],
            "color": [0.92, 1.08],
            "geometry_changed": False,
        },
        "train_demonstrations": 2000,
        "uses_evaluation_initial_states": False,
        "uses_rewards_or_success": False,
        "action_offset": 1,
    }
    contract_sha = sha256(
        json.dumps(contract, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    contract["contract_sha256"] = contract_sha
    args.output.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output / "manifest.json"
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text())
        comparable = {key: previous.get(key) for key in contract}
        if comparable != contract:
            raise ValueError("existing V3.31 cache contract differs")
        if previous.get("complete"):
            print("V3.31增强缓存已完成，无需重复")
            return
        if not args.resume:
            raise ValueError("partial V3.31 cache requires --resume")
    elif any(args.output.glob("shard-*.pt")):
        raise ValueError("V3.31 output has shards without a manifest")
    if formal:
        require_disk_budget(args.output, minimum_gib=20)
    backbone = QwenVLBackbone().load(device=torch.device("cuda"))
    contract["model_path"] = str(backbone.config.model_path)
    total_shards = (len(selected) + args.shard_size - 1) // args.shard_size
    opened: h5py.File | None = None
    opened_path: Path | None = None
    started = time.monotonic()
    completed = 0
    written_bytes = sum(path.stat().st_size for path in args.output.glob("shard-*.pt"))
    try:
        for shard_id in range(total_shards):
            path = args.output / f"shard-{shard_id:06d}.pt"
            begin = shard_id * args.shard_size
            rows = selected[begin : begin + args.shard_size]
            if path.exists():
                payload = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
                if payload.get("contract_sha256") != contract_sha:
                    raise ValueError(f"invalid existing V3.31 shard: {path}")
                if payload["dataset_indices"].tolist() != [row[0] for row in rows]:
                    raise ValueError(f"V3.31 shard indices changed: {path}")
                completed += 1
                continue
            visuals, semantics, languages = [], [], []
            offsets, token_ids = [0], []
            task_indices, episode_indices, frame_indices = [], [], []
            for batch_start in range(0, len(rows), args.batch_size):
                source_rows = rows[batch_start : batch_start + args.batch_size]
                images_by_camera = [[], []]
                instructions = []
                for global_index, demo, frame in source_rows:
                    if opened_path != demo.task.path:
                        if opened is not None:
                            opened.close()
                        opened = h5py.File(demo.task.path, "r")
                        opened_path = demo.task.path
                    obs = opened["data"][demo.demo_key]["obs"]
                    for camera, key in enumerate(("agentview_rgb", "eye_in_hand_rgb")):
                        images_by_camera[camera].append(
                            _augment(obs[key][frame], seed=global_index * 17 + camera * 1000003)
                        )
                    wrapper = WRAPPERS[global_index % len(WRAPPERS)]
                    instructions.append(wrapper.format(instruction=demo.task.instruction))
                    task_indices.append(int(demo.task.global_task_index))
                    episode_indices.append(int(demo.episode_index))
                    frame_indices.append(frame)
                camera_inputs = [
                    backbone.prepare_view_batch_inputs(list(zip(images, instructions)))
                    for images in images_by_camera
                ]
                visual_all, semantic, words, mask = backbone.forward_v331_context(
                    camera_inputs, instructions, max_tokens=args.max_tokens
                )
                visual = visual_all[:, 1:2]
                for row, instruction in enumerate(instructions):
                    length = int(mask[row, 0].sum())
                    visuals.append(visual[row].cpu().to(torch.bfloat16).contiguous())
                    semantics.append(semantic[row].cpu().to(torch.bfloat16).contiguous())
                    languages.append(words[row, :, :length].transpose(0, 1).cpu().to(torch.bfloat16).contiguous())
                    offsets.append(offsets[-1] + length)
                    token_ids.append(backbone.processor.tokenizer.encode(instruction, add_special_tokens=False))
            payload = {
                "contract_sha256": contract_sha,
                "dataset_indices": torch.tensor([row[0] for row in rows], dtype=torch.long),
                "task_indices": torch.tensor(task_indices, dtype=torch.long),
                "episode_indices": torch.tensor(episode_indices, dtype=torch.long),
                "frame_indices": torch.tensor(frame_indices, dtype=torch.long),
                "visual": torch.stack(visuals),
                "semantic": torch.stack(semantics),
                "language": torch.cat(languages),
                "language_offsets": torch.tensor(offsets, dtype=torch.long),
                "token_ids": token_ids,
            }
            _atomic_save(path, payload)
            written_bytes += path.stat().st_size
            if written_bytes > args.max_cache_gib * 2**30:
                raise RuntimeError(
                    f"V3.31缓存超过硬上限：{written_bytes/2**30:.2f}GiB > {args.max_cache_gib:.2f}GiB"
                )
            completed += 1
            atomic_write_json(
                manifest_path,
                {**contract, "complete": False, "completed_shards": completed, "total_shards": total_shards},
            )
            atomic_write_json(
                args.output / "progress.json",
                {
                    "阶段": "V3.31限额配对增强缓存",
                    "completed_shards": completed,
                    "total_shards": total_shards,
                    "elapsed_seconds": time.monotonic() - started,
                    "cache_gib": written_bytes / 2**30,
                },
            )
            if completed % 25 == 0 or completed == total_shards:
                print(f"V3.31缓存 {completed}/{total_shards}", flush=True)
                if formal:
                    require_disk_budget(args.output, minimum_gib=12)
    finally:
        if opened is not None:
            opened.close()
    token_rows = 0
    long_language_rows = 0
    maximum_tokens = 0
    for shard_id in range(total_shards):
        payload = torch.load(
            args.output / f"shard-{shard_id:06d}.pt",
            map_location="cpu",
            weights_only=True,
            mmap=True,
        )
        lengths = payload["language_offsets"][1:] - payload["language_offsets"][:-1]
        token_rows += int(lengths.numel())
        long_language_rows += int((lengths > 32).sum())
        maximum_tokens = max(maximum_tokens, int(lengths.max()))
    if token_rows != len(selected) or long_language_rows < len(selected) // 4:
        raise RuntimeError(
            f"V3.31长语言覆盖不足：{long_language_rows}/{token_rows}，至少需要25%"
        )
    atomic_write_json(
        manifest_path,
        {**contract, "complete": True, "completed_shards": completed, "total_shards": total_shards,
         "cache_gib": written_bytes / 2**30, "long_language_rows": long_language_rows,
         "maximum_tokens_observed": maximum_tokens},
    )


if __name__ == "__main__":
    main()
