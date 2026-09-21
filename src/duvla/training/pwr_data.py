"""Episode-safe future targets for DuVLA-PWR planner training."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Iterator

import torch
from torch import Tensor

from duvla.training.feature_cache import (
    FeatureShard,
    list_feature_shards,
    load_feature_shard,
)
from duvla.training.v2_1_data import iter_v2_1_batches, validate_v2_1_shard


PWR_CACHE_COMPATIBILITY_FIELDS = (
    "dataset_root",
    "seed",
    "validation_fraction",
    "horizon",
    "context_tokens_per_camera",
    "context_layout",
    "qwen_hidden_layer",
    "qwen_hidden_layers",
    "qwen_context_mode",
    "qwen_semantic_tokens",
    "qwen_semantic_layer",
    "qwen_grounding_tokens",
    "spatial_projector",
    "spatial_projector_sha256",
    "spatial_projected_dim",
    "qwen_model_path",
    "qwen_dtype",
    "task_indices",
    "limit_per_task",
    "state_mean",
    "state_std",
    "action_mean",
    "action_std",
)


def validate_pwr_cache_compatibility(
    reference: dict[str, object],
    candidate: dict[str, object],
) -> None:
    """Require one feature/action contract while allowing train/val identity."""

    mismatches = [
        field
        for field in PWR_CACHE_COMPATIBILITY_FIELDS
        if reference.get(field) != candidate.get(field)
    ]
    if mismatches:
        raise ValueError(
            "PWR cache contract mismatch: " + ", ".join(sorted(mismatches))
        )


@dataclass(frozen=True)
class _FutureLocation:
    path_index: int
    row: int
    episode_index: int
    frame_index: int


@dataclass(frozen=True)
class PWRTeacherActions:
    """Compact training-only behavior targets from one frozen policy."""

    dataset_indices: Tensor
    actions: Tensor
    source_cache_signature: dict[str, object]
    teacher_checkpoint_sha256: str

    def select(self, indices: Tensor) -> Tensor:
        requested = indices.to(device="cpu", dtype=torch.long)
        positions = torch.searchsorted(self.dataset_indices, requested)
        if bool((positions >= self.dataset_indices.numel()).any()):
            raise ValueError("teacher sidecar does not cover requested dataset indices")
        matched = self.dataset_indices.index_select(0, positions)
        if not torch.equal(matched, requested):
            raise ValueError("teacher sidecar does not cover requested dataset indices")
        return self.actions.index_select(0, positions)


def load_pwr_teacher_actions(
    directory: str | Path,
    *,
    cache_manifest: dict[str, object],
) -> PWRTeacherActions:
    """Load and strictly bind a completed V3.1 teacher-action sidecar."""

    root = Path(directory)
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        raise ValueError(f"teacher sidecar is incomplete: {manifest_path}")
    manifest = json.loads(manifest_path.read_text())
    if (
        manifest.get("format") != "duvla_v3_1_teacher_actions"
        or int(manifest.get("schema_version", 0)) != 1
    ):
        raise ValueError("unsupported PWR teacher sidecar format")
    if manifest.get("source_cache_signature") != cache_manifest.get("cache_signature"):
        raise ValueError("teacher sidecar source cache signature mismatch")
    if manifest.get("task_index_scope") != "training_teacher_only":
        raise ValueError("teacher sidecar has an unsafe task-index scope")
    paths = sorted(root.glob("shard-*.pt"))
    if len(paths) != int(manifest.get("shard_count", -1)):
        raise ValueError("teacher sidecar shard count differs from its manifest")
    index_parts: list[Tensor] = []
    action_parts: list[Tensor] = []
    for path in paths:
        payload = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
        indices = payload.get("dataset_indices")
        actions = payload.get("teacher_actions")
        if (
            payload.get("format") != "duvla_v3_1_teacher_action_shard"
            or not isinstance(indices, Tensor)
            or indices.ndim != 1
            or indices.dtype != torch.long
            or not isinstance(actions, Tensor)
            or actions.ndim != 3
            or actions.shape[0] != indices.numel()
            or tuple(actions.shape[1:])
            != (int(manifest["action_horizon"]), int(manifest["action_dim"]))
        ):
            raise ValueError(f"invalid teacher action shard: {path}")
        index_parts.append(indices)
        action_parts.append(actions)
    if not index_parts:
        raise ValueError("teacher sidecar contains no shards")
    indices = torch.cat(index_parts).long()
    actions = torch.cat(action_parts)
    if indices.numel() != int(manifest.get("row_count", -1)):
        raise ValueError("teacher sidecar row count differs from its manifest")
    if indices.numel() > 1 and not bool((indices[1:] > indices[:-1]).all()):
        raise ValueError("teacher sidecar indices must be strictly increasing")
    if not bool(torch.isfinite(actions).all()):
        raise ValueError("teacher sidecar actions contain non-finite values")
    return PWRTeacherActions(
        dataset_indices=indices,
        actions=actions,
        source_cache_signature=dict(manifest["source_cache_signature"]),
        teacher_checkpoint_sha256=str(manifest["teacher_checkpoint_sha256"]),
    )


@dataclass(frozen=True)
class PWRBaseActions:
    """Task-index-free frozen base actions used by progressive residual training."""

    dataset_indices: Tensor
    actions: Tensor
    source_cache_signature: dict[str, object]
    base_checkpoint_sha256: str

    def select(self, indices: Tensor) -> Tensor:
        requested = indices.to(device="cpu", dtype=torch.long)
        positions = torch.searchsorted(self.dataset_indices, requested)
        if bool((positions >= self.dataset_indices.numel()).any()):
            raise ValueError("base sidecar does not cover requested dataset indices")
        matched = self.dataset_indices.index_select(0, positions)
        if not torch.equal(matched, requested):
            raise ValueError("base sidecar does not cover requested dataset indices")
        return self.actions.index_select(0, positions)


def load_pwr_base_actions(
    directory: str | Path,
    *,
    cache_manifest: dict[str, object],
) -> PWRBaseActions:
    """Load a completed natural-language P1c base-action sidecar."""

    root = Path(directory)
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        raise ValueError(f"base sidecar is incomplete: {manifest_path}")
    manifest = json.loads(manifest_path.read_text())
    if (
        manifest.get("format") != "duvla_v3_1_base_actions"
        or int(manifest.get("schema_version", 0)) != 1
    ):
        raise ValueError("unsupported PWR base sidecar format")
    if manifest.get("source_cache_signature") != cache_manifest.get("cache_signature"):
        raise ValueError("base sidecar source cache signature mismatch")
    if (
        manifest.get("task_routing") != "natural_language"
        or manifest.get("benchmark_task_index") is not False
    ):
        raise ValueError("base sidecar has an unsafe routing contract")
    paths = sorted(root.glob("shard-*.pt"))
    if len(paths) != int(manifest.get("shard_count", -1)):
        raise ValueError("base sidecar shard count differs from its manifest")
    index_parts: list[Tensor] = []
    action_parts: list[Tensor] = []
    for path in paths:
        payload = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
        indices = payload.get("dataset_indices")
        actions = payload.get("base_actions")
        if (
            payload.get("format") != "duvla_v3_1_base_action_shard"
            or not isinstance(indices, Tensor)
            or indices.ndim != 1
            or indices.dtype != torch.long
            or not isinstance(actions, Tensor)
            or actions.ndim != 3
            or actions.shape[0] != indices.numel()
            or tuple(actions.shape[1:])
            != (int(manifest["action_horizon"]), int(manifest["action_dim"]))
        ):
            raise ValueError(f"invalid base action shard: {path}")
        index_parts.append(indices)
        action_parts.append(actions)
    if not index_parts:
        raise ValueError("base sidecar contains no shards")
    indices = torch.cat(index_parts).long()
    actions = torch.cat(action_parts)
    if indices.numel() != int(manifest.get("row_count", -1)):
        raise ValueError("base sidecar row count differs from its manifest")
    if indices.numel() > 1 and not bool((indices[1:] > indices[:-1]).all()):
        raise ValueError("base sidecar indices must be strictly increasing")
    if not bool(torch.isfinite(actions).all()):
        raise ValueError("base sidecar actions contain non-finite values")
    return PWRBaseActions(
        dataset_indices=indices,
        actions=actions,
        source_cache_signature=dict(manifest["source_cache_signature"]),
        base_checkpoint_sha256=str(manifest["base_checkpoint_sha256"]),
    )


class FutureFeatureStore:
    """Resolve t+offset frozen-Qwen rows without crossing episode boundaries."""

    def __init__(self, root: str | Path, *, maximum_open_shards: int = 12) -> None:
        if maximum_open_shards <= 0:
            raise ValueError("maximum_open_shards must be positive")
        self.paths = list_feature_shards(root)
        if not self.paths:
            raise ValueError(f"feature cache contains no shards: {root}")
        self.maximum_open_shards = maximum_open_shards
        self.locations: dict[int, _FutureLocation] = {}
        self._loaded: OrderedDict[int, FeatureShard] = OrderedDict()
        for path_index, path in enumerate(self.paths):
            shard = load_feature_shard(path, mmap=True)
            validate_v2_1_shard(shard)
            if shard.episode_indices is None or shard.frame_indices is None:
                raise ValueError("future targets require episode/frame metadata")
            for row, dataset_index in enumerate(shard.dataset_indices):
                index = int(dataset_index)
                if index in self.locations:
                    raise ValueError("feature cache contains duplicate dataset indices")
                self.locations[index] = _FutureLocation(
                    path_index=path_index,
                    row=row,
                    episode_index=int(shard.episode_indices[row]),
                    frame_index=int(shard.frame_indices[row]),
                )

    def _shard(self, path_index: int) -> FeatureShard:
        shard = self._loaded.pop(path_index, None)
        if shard is None:
            shard = load_feature_shard(self.paths[path_index], mmap=True)
            validate_v2_1_shard(shard)
        self._loaded[path_index] = shard
        while len(self._loaded) > self.maximum_open_shards:
            self._loaded.popitem(last=False)
        return shard

    def future(
        self,
        dataset_index: int,
        *,
        episode_index: int,
        frame_index: int,
        offset: int,
    ) -> tuple[Tensor, Tensor] | None:
        if offset <= 0:
            raise ValueError("future offset must be positive")
        location = self.locations.get(dataset_index + offset)
        if location is None:
            return None
        if (
            location.episode_index != episode_index
            or location.frame_index != frame_index + offset
        ):
            return None
        shard = self._shard(location.path_index)
        return shard.features[location.row], shard.states[location.row]


def attach_pwr_future_targets(
    batch: dict[str, Tensor],
    store: FutureFeatureStore,
    *,
    offsets: tuple[int, ...] = (2, 4, 8),
) -> dict[str, Tensor]:
    """Attach future visual/state targets while masking episode tails."""

    if not offsets or any(offset <= 0 for offset in offsets):
        raise ValueError("future offsets must be non-empty and positive")
    required = {"visual", "states", "dataset_indices", "episode_indices"}
    if not required.issubset(batch):
        raise ValueError("batch lacks fields required for PWR future targets")
    frame_indices = batch.get("frame_indices")
    if frame_indices is None:
        # The standard iterator deliberately did not need frame ids in older
        # policies.  Dataset rows are contiguous within each episode, so use
        # the store's audited current location rather than infer silently.
        resolved = [store.locations[int(index)] for index in batch["dataset_indices"]]
        frame_indices = torch.tensor(
            [location.frame_index for location in resolved], dtype=torch.long
        )
    future_visual: list[Tensor] = []
    future_states: list[Tensor] = []
    future_mask: list[Tensor] = []
    for row, raw_index in enumerate(batch["dataset_indices"]):
        index = int(raw_index)
        episode = int(batch["episode_indices"][row])
        frame = int(frame_indices[row])
        row_visual: list[Tensor] = []
        row_states: list[Tensor] = []
        row_mask: list[bool] = []
        for offset in offsets:
            target = store.future(
                index,
                episode_index=episode,
                frame_index=frame,
                offset=offset,
            )
            if target is None:
                row_visual.append(batch["visual"][row])
                row_states.append(batch["states"][row])
                row_mask.append(False)
            else:
                visual, state = target
                row_visual.append(visual)
                row_states.append(state)
                row_mask.append(True)
        future_visual.append(torch.stack(row_visual))
        future_states.append(torch.stack(row_states))
        future_mask.append(torch.tensor(row_mask, dtype=torch.bool))
    result = dict(batch)
    result["future_visual"] = torch.stack(future_visual)
    result["future_states"] = torch.stack(future_states)
    result["future_mask"] = torch.stack(future_mask)
    result["frame_indices"] = frame_indices
    return result


def iter_pwr_batches(
    cache_root: str | Path,
    *,
    future_store: FutureFeatureStore,
    batch_size: int,
    history_length: int,
    history_stride: int,
    seed: int,
    epoch: int,
    shuffle: bool,
    shard_shuffle_block_size: int = 4,
    mmap_shards: bool = True,
    offsets: tuple[int, ...] = (2, 4, 8),
) -> Iterator[dict[str, Tensor]]:
    for batch in iter_v2_1_batches(
        cache_root,
        batch_size=batch_size,
        history_length=history_length,
        history_stride=history_stride,
        seed=seed,
        epoch=epoch,
        shuffle=shuffle,
        shard_shuffle_block_size=shard_shuffle_block_size,
        mmap_shards=mmap_shards,
    ):
        yield attach_pwr_future_targets(batch, future_store, offsets=offsets)
