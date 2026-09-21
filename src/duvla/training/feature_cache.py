"""Atomic, shard-based storage for frozen-Qwen action-training features."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import torch
from torch import Tensor


@dataclass(frozen=True)
class FeatureShard:
    features: Tensor
    states: Tensor
    actions: Tensor
    valid_mask: Tensor
    dataset_indices: tuple[int, ...]
    semantic_features: Tensor | None = None
    episode_indices: Tensor | None = None
    frame_indices: Tensor | None = None
    task_indices: Tensor | None = None


@dataclass(frozen=True)
class CacheResumePlan:
    """Validated position from which an interrupted cache build may continue."""

    next_shard_number: int
    completed_indices: tuple[int, ...]
    complete: bool


def atomic_write_json(path: str | Path, payload: Mapping[str, object]) -> None:
    """Atomically publish a JSON status/manifest file."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(destination)


def save_feature_shard(path: str | Path, shard: FeatureShard) -> None:
    """Write one shard through a temporary file, then atomically publish it."""

    if shard.features.ndim not in (2, 3, 4, 5) or shard.states.ndim != 2 or shard.actions.ndim != 3:
        raise ValueError("features, states, and actions have invalid ranks")
    if shard.features.shape[0] != shard.states.shape[0]:
        raise ValueError("features and states must have the same batch size")
    if shard.valid_mask.shape != shard.actions.shape[:2]:
        raise ValueError("valid_mask must match [batch, horizon]")
    if len(shard.dataset_indices) != shard.features.shape[0]:
        raise ValueError("dataset_indices must match the feature batch")
    if shard.semantic_features is not None:
        if shard.semantic_features.ndim != 4:
            raise ValueError("semantic_features must have shape [batch, layers, tokens, dim]")
        if shard.semantic_features.shape[0] != shard.features.shape[0]:
            raise ValueError("semantic_features and features must have the same batch size")
        if shard.features.ndim != 5:
            raise ValueError("semantic_features require multilayer camera features")
        if shard.features.shape[1] not in {1, shard.semantic_features.shape[1]}:
            raise ValueError(
                "visual layer count must be one or match semantic layer count"
            )
        if shard.semantic_features.shape[-1] != shard.features.shape[-1]:
            raise ValueError("semantic_features and features must have the same feature dim")
    for name, values in (
        ("episode_indices", shard.episode_indices),
        ("frame_indices", shard.frame_indices),
        ("task_indices", shard.task_indices),
    ):
        if values is not None and (values.ndim != 1 or values.shape[0] != shard.features.shape[0]):
            raise ValueError(f"{name} must have shape [batch]")
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    payload: dict[str, object] = {
            "features": shard.features.cpu(),
            "states": shard.states.cpu(),
            "actions": shard.actions.cpu(),
            "valid_mask": shard.valid_mask.cpu(),
            "dataset_indices": shard.dataset_indices,
        }
    if shard.semantic_features is not None:
        payload["semantic_features"] = shard.semantic_features.cpu()
    for name, values in (
        ("episode_indices", shard.episode_indices),
        ("frame_indices", shard.frame_indices),
        ("task_indices", shard.task_indices),
    ):
        if values is not None:
            payload[name] = values.cpu()
    torch.save(payload, temporary)
    temporary.replace(destination)


def load_feature_shard(path: str | Path, *, mmap: bool = False) -> FeatureShard:
    """Load a previously published CPU feature shard."""

    payload = torch.load(Path(path), map_location="cpu", weights_only=True, mmap=mmap)
    return FeatureShard(
        features=payload["features"],
        states=payload["states"],
        actions=payload["actions"],
        valid_mask=payload["valid_mask"],
        dataset_indices=tuple(int(index) for index in payload["dataset_indices"]),
        semantic_features=payload.get("semantic_features"),
        episode_indices=payload.get("episode_indices"),
        frame_indices=payload.get("frame_indices"),
        task_indices=payload.get("task_indices"),
    )


def list_feature_shards(directory: str | Path) -> tuple[Path, ...]:
    return tuple(sorted(Path(directory).glob("shard-*.pt")))


def prepare_cache_resume(
    directory: str | Path,
    *,
    signature: Mapping[str, object],
    selected_indices: Sequence[int],
    shard_size: int,
    resume: bool,
) -> CacheResumePlan:
    """Validate existing shards and return a deterministic resume position.

    Existing samples must be an exact prefix of the current deterministic
    selection.  This prevents a changed split, normalization, camera layout,
    or model setting from being silently mixed into the same cache.
    """

    root = Path(directory)
    paths = list_feature_shards(root)
    state_path = root / "cache_state.json"
    manifest_path = root / "manifest.json"
    if (paths or state_path.exists() or manifest_path.exists()) and not resume:
        raise ValueError(
            f"cache output is not empty: {root}; pass --resume only when continuing "
            "the same configuration"
        )

    metadata_path = manifest_path if manifest_path.exists() else state_path
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text())
        if metadata.get("cache_signature") != dict(signature):
            raise ValueError(f"cache resume signature mismatch: {metadata_path}")
    elif paths:
        raise ValueError("feature shards exist without cache_state.json or manifest.json")

    expected_names = [f"shard-{index:06d}.pt" for index in range(len(paths))]
    if [path.name for path in paths] != expected_names:
        raise ValueError("feature shard numbering is not contiguous from shard-000000.pt")

    completed: list[int] = []
    shard_lengths: list[int] = []
    for path in paths:
        shard = load_feature_shard(path)
        completed.extend(shard.dataset_indices)
        shard_lengths.append(len(shard.dataset_indices))
    if len(completed) != len(set(completed)):
        raise ValueError("duplicate dataset indices in existing feature shards")
    selected = tuple(int(index) for index in selected_indices)
    completed_tuple = tuple(completed)
    if completed_tuple != selected[: len(completed_tuple)]:
        raise ValueError("existing feature shards are not an exact prefix of the selected rows")
    if len(completed_tuple) > len(selected):
        raise ValueError("existing feature shards contain more rows than the current selection")
    if shard_lengths and any(length != shard_size for length in shard_lengths[:-1]):
        raise ValueError("a non-final feature shard has an unexpected sample count")
    if shard_lengths and len(completed_tuple) < len(selected) and shard_lengths[-1] != shard_size:
        raise ValueError("an incomplete final shard cannot be extended safely")

    complete = manifest_path.exists()
    if complete and completed_tuple != selected:
        raise ValueError("completed cache manifest does not match all selected rows")
    return CacheResumePlan(
        next_shard_number=len(paths),
        completed_indices=completed_tuple,
        complete=complete,
    )
