"""Streaming schema-7 cache batches for Duvla V2.1."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
import json
from pathlib import Path
import random

import torch
from torch import Tensor

from duvla.training.feature_cache import FeatureShard, list_feature_shards, load_feature_shard


@dataclass(frozen=True)
class LongHorizonActionSidecar:
    """Compact action targets aligned to the existing frozen-Qwen cache rows."""

    dataset_indices: Tensor
    actions: Tensor
    valid_mask: Tensor
    horizon: int
    source_cache_signature: dict[str, object]

    def select(self, indices: Tensor) -> tuple[Tensor, Tensor]:
        requested = indices.to(device="cpu", dtype=torch.long)
        positions = torch.searchsorted(self.dataset_indices, requested)
        if bool((positions >= self.dataset_indices.numel()).any()):
            raise ValueError("action sidecar does not cover requested dataset indices")
        matched = self.dataset_indices.index_select(0, positions)
        if not torch.equal(matched, requested):
            raise ValueError("action sidecar does not cover requested dataset indices")
        return (
            self.actions.index_select(0, positions),
            self.valid_mask.index_select(0, positions),
        )


def load_long_horizon_action_sidecar(
    path: str | Path,
    *,
    cache_manifest: dict[str, object],
    mmap: bool = True,
) -> LongHorizonActionSidecar:
    """Load and strictly bind a long-horizon sidecar to one feature cache."""

    payload = torch.load(Path(path), map_location="cpu", weights_only=True, mmap=mmap)
    if payload.get("format") != "duvla_v2_long_horizon_actions":
        raise ValueError("invalid long-horizon action sidecar format")
    if payload.get("schema_version") != 1:
        raise ValueError("unsupported long-horizon action sidecar schema")
    source_signature = payload.get("source_cache_signature")
    if source_signature != cache_manifest.get("cache_signature"):
        raise ValueError("long-horizon action sidecar cache signature mismatch")
    indices = payload.get("dataset_indices")
    actions = payload.get("actions")
    mask = payload.get("valid_mask")
    horizon = payload.get("horizon")
    expected_rows = int(cache_manifest.get("selected_frames", -1))
    if not isinstance(indices, Tensor) or indices.ndim != 1 or indices.dtype != torch.long:
        raise ValueError("sidecar dataset_indices must be an int64 vector")
    if not isinstance(actions, Tensor) or actions.ndim != 3 or not actions.is_floating_point():
        raise ValueError("sidecar actions must be a floating [rows, horizon, action] tensor")
    if not isinstance(mask, Tensor) or mask.dtype != torch.bool:
        raise ValueError("sidecar valid_mask must be boolean")
    if not isinstance(horizon, int) or horizon <= 0:
        raise ValueError("sidecar horizon must be positive")
    if actions.shape[:2] != mask.shape or actions.shape[1] != horizon:
        raise ValueError("sidecar action and mask shapes disagree")
    if actions.shape[0] != indices.numel() or actions.shape[0] != expected_rows:
        raise ValueError("sidecar row count does not match the feature cache")
    if indices.numel() > 1 and not bool((indices[1:] > indices[:-1]).all()):
        raise ValueError("sidecar dataset indices must be strictly increasing")
    if not bool(torch.isfinite(actions).all()):
        raise ValueError("sidecar actions contain non-finite values")
    return LongHorizonActionSidecar(
        dataset_indices=indices,
        actions=actions,
        valid_mask=mask,
        horizon=horizon,
        source_cache_signature=dict(source_signature),
    )


def assemble_long_horizon_actions(
    current_actions: Tensor,
    dataset_indices: Tensor,
    episode_indices: Tensor,
    frame_indices: Tensor,
    *,
    horizon: int,
    padding_action: Tensor,
) -> tuple[Tensor, Tensor]:
    """Assemble future chunks while enforcing global, episode and frame continuity."""

    if (
        current_actions.ndim != 2
        or dataset_indices.ndim != 1
        or episode_indices.ndim != 1
        or frame_indices.ndim != 1
        or padding_action.ndim != 1
        or horizon <= 0
    ):
        raise ValueError("long-horizon action assembly received invalid ranks")
    row_count, action_dim = current_actions.shape
    if not (
        dataset_indices.numel()
        == episode_indices.numel()
        == frame_indices.numel()
        == row_count
        and padding_action.numel() == action_dim
    ):
        raise ValueError("long-horizon action assembly metadata shapes disagree")
    if row_count == 0:
        raise ValueError("long-horizon action assembly requires rows")
    if row_count > 1 and not bool((dataset_indices[1:] > dataset_indices[:-1]).all()):
        raise ValueError("dataset indices must be strictly increasing")
    actions = padding_action.to(dtype=current_actions.dtype).view(
        1, 1, action_dim
    ).expand(row_count, horizon, action_dim).clone()
    valid_mask = torch.zeros(row_count, horizon, dtype=torch.bool)
    rows = torch.arange(row_count)
    for offset in range(horizon):
        source = rows + offset
        within = source < row_count
        safe_source = source.clamp_max(row_count - 1)
        valid = (
            within
            & (episode_indices.index_select(0, safe_source) == episode_indices)
            & (frame_indices.index_select(0, safe_source) == frame_indices + offset)
            & (dataset_indices.index_select(0, safe_source) == dataset_indices + offset)
        )
        valid_rows = rows[valid]
        valid_sources = source[valid]
        actions[valid_rows, offset] = current_actions.index_select(0, valid_sources)
        valid_mask[valid_rows, offset] = True
    return actions, valid_mask


def build_long_horizon_action_payload(
    directory: str | Path,
    *,
    horizon: int,
) -> dict[str, object]:
    """Reconstruct longer normalized action chunks without recomputing Qwen features."""

    if horizon <= 0:
        raise ValueError("action horizon must be positive")
    manifest = validate_v2_1_manifest(directory)
    first_actions: list[Tensor] = []
    dataset_indices: list[int] = []
    episode_indices: list[Tensor] = []
    frame_indices: list[Tensor] = []
    action_dim: int | None = None
    for path in list_feature_shards(directory):
        shard = load_feature_shard(path, mmap=True)
        validate_v2_1_shard(shard)
        if action_dim is None:
            action_dim = int(shard.actions.shape[-1])
        elif action_dim != int(shard.actions.shape[-1]):
            raise ValueError("feature cache action dimensions differ across shards")
        if not bool(shard.valid_mask[:, 0].all()):
            raise ValueError("every cached row must contain a valid current action")
        first_actions.append(shard.actions[:, 0].to(dtype=torch.float32))
        dataset_indices.extend(int(value) for value in shard.dataset_indices)
        if shard.episode_indices is None or shard.frame_indices is None:  # pragma: no cover
            raise RuntimeError("validated shard lost episode/frame metadata")
        episode_indices.append(shard.episode_indices.to(dtype=torch.long))
        frame_indices.append(shard.frame_indices.to(dtype=torch.long))
    if not first_actions or action_dim is None:
        raise ValueError("V2 cache has no feature shards")
    index_tensor = torch.tensor(dataset_indices, dtype=torch.long)
    if index_tensor.numel() > 1 and not bool((index_tensor[1:] > index_tensor[:-1]).all()):
        raise ValueError("feature cache rows must be strictly ordered for sidecar construction")
    current_actions = torch.cat(first_actions, dim=0)
    episodes = torch.cat(episode_indices, dim=0)
    frames = torch.cat(frame_indices, dim=0)
    row_count = index_tensor.numel()
    if row_count != int(manifest["selected_frames"]):
        raise ValueError("feature shard rows do not match the manifest")
    mean = manifest.get("action_mean")
    std = manifest.get("action_std")
    if not isinstance(mean, list) or not isinstance(std, list):
        raise ValueError("cache manifest lacks action normalization")
    padding = -torch.tensor(mean, dtype=torch.float32) / torch.tensor(std, dtype=torch.float32)
    actions, valid_mask = assemble_long_horizon_actions(
        current_actions,
        index_tensor,
        episodes,
        frames,
        horizon=horizon,
        padding_action=padding,
    )
    return {
        "format": "duvla_v2_long_horizon_actions",
        "schema_version": 1,
        "horizon": horizon,
        "action_dim": action_dim,
        "selected_frames": row_count,
        "source_cache_signature": manifest.get("cache_signature"),
        "dataset_indices": index_tensor,
        "actions": actions,
        "valid_mask": valid_mask,
    }


def _next_batch(
    iterator: Iterator[dict[str, Tensor]],
    *,
    pin_memory: bool,
) -> dict[str, Tensor]:
    batch = next(iterator)
    if pin_memory:
        batch = {name: value.pin_memory() for name, value in batch.items()}
    return batch


def prefetch_v2_1_batches(
    batches: Iterable[dict[str, Tensor]],
    *,
    pin_memory: bool,
) -> Iterator[dict[str, Tensor]]:
    """Load and optionally pin the next CPU batch while the GPU consumes this one."""

    iterator = iter(batches)
    executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="duvla-v2-1-loader")
    future: Future[dict[str, Tensor]] | None = executor.submit(
        _next_batch, iterator, pin_memory=pin_memory
    )
    try:
        while future is not None:
            try:
                batch = future.result()
            except StopIteration:
                break
            future = executor.submit(_next_batch, iterator, pin_memory=pin_memory)
            yield batch
    finally:
        if future is not None:
            future.cancel()
        executor.shutdown(wait=True, cancel_futures=True)


def deterministic_flow_noise(
    indices: Tensor,
    *,
    seed: int,
    samples: int,
    horizon: int,
    action_dim: int,
) -> Tensor:
    """Create order-independent Flow noise keyed by global dataset index."""

    if indices.ndim != 1 or min(samples, horizon, action_dim) <= 0 or seed < 0:
        raise ValueError("noise indices/counts or seed are invalid")
    values: list[Tensor] = []
    for index in indices.tolist():
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed + int(index) * 1009)
        values.append(
            torch.randn(samples, horizon, action_dim, generator=generator)
        )
    return torch.stack(values)


def validate_v2_1_manifest(directory: str | Path) -> dict[str, object]:
    root = Path(directory)
    path = root / "manifest.json"
    if not path.is_file():
        raise ValueError(f"missing cache manifest: {path}")
    manifest = json.loads(path.read_text())
    identity = (manifest.get("schema_version"), manifest.get("qwen_context_mode"))
    if identity not in {
        (7, "multilayer_spatial_semantic"),
        (8, "highres_layer14_multilayer_semantic"),
    }:
        raise ValueError("Duvla V2 cache must use schema 7 or high-resolution schema 8")
    if manifest.get("context_layout") != "per_camera":
        raise ValueError("Duvla V2 cache manifest requires context_layout='per_camera'")
    normalization = manifest.get("normalization")
    if normalization == "train-split per-dimension mean/std for state and action":
        pass
    elif normalization == "all-demonstrations per-dimension mean/std for state and action":
        final_refit = (
            manifest.get("split") == "all_demonstrations_train"
            and manifest.get("validation_fraction") == 0.0
            and manifest.get("all_demonstrations") is True
            and manifest.get("train_episodes") == manifest.get("total_episodes")
            and manifest.get("validation_episodes") == 0
            and manifest.get("uses_evaluation_initial_states") is False
        )
        explicit_smoke = (
            manifest.get("split") == "official_demonstration_smoke"
            and manifest.get("smoke_cache") is True
            and manifest.get("all_demonstrations") is False
            and manifest.get("uses_evaluation_initial_states") is False
        )
        if not (final_refit or explicit_smoke):
            raise ValueError("all-demonstration normalization requires an explicit final-refit manifest")
    else:
        raise ValueError("V2.1 cache has an unsupported normalization contract")
    return manifest


def validate_v2_1_shard(shard: FeatureShard) -> None:
    if shard.features.ndim != 5:
        raise ValueError("V2.1 visual cache must have rank five")
    if shard.semantic_features is None or shard.semantic_features.ndim != 4:
        raise ValueError("V2.1 cache requires semantic_features")
    if shard.episode_indices is None or shard.frame_indices is None or shard.task_indices is None:
        raise ValueError("V2.1 cache requires episode/frame/task metadata")
    if shard.features.shape[1] not in {1, shard.semantic_features.shape[1]}:
        raise ValueError("visual layer count must be one or match semantic layers")
    if shard.features.shape[-1] != shard.semantic_features.shape[-1]:
        raise ValueError("visual and semantic feature dimensions differ")


def _history_source(
    current: FeatureShard,
    previous: Sequence[FeatureShard],
) -> dict[int, tuple[FeatureShard, int]]:
    result: dict[int, tuple[FeatureShard, int]] = {}
    for previous_shard in previous:
        for offset, index in enumerate(previous_shard.dataset_indices):
            result[int(index)] = (previous_shard, offset)
    for offset, index in enumerate(current.dataset_indices):
        result[int(index)] = (current, offset)
    return result


def _metadata_at(shard: FeatureShard, name: str, offset: int) -> int:
    values = getattr(shard, name)
    if not isinstance(values, Tensor):
        raise ValueError(f"V2.1 shard is missing {name}")
    return int(values[offset])


def _normalized_noop_action(manifest: dict[str, object], action_dim: int) -> Tensor:
    """Return the normalized LIBERO settle action without using validation statistics."""

    mean = manifest.get("action_mean")
    std = manifest.get("action_std")
    if not (
        isinstance(mean, list)
        and isinstance(std, list)
        and len(mean) == action_dim
        and len(std) == action_dim
        and all(float(value) > 0.0 for value in std)
    ):
        # Tiny synthetic test manifests intentionally omit numeric statistics.
        return torch.zeros(action_dim, dtype=torch.float32)
    native = torch.zeros(action_dim, dtype=torch.float32)
    native[-1] = -1.0
    return (native - torch.tensor(mean, dtype=torch.float32)) / torch.tensor(
        std, dtype=torch.float32
    )


def _previous_action_at(
    lookup: dict[int, tuple[FeatureShard, int]],
    shard: FeatureShard,
    offset: int,
    *,
    noop_action: Tensor,
) -> Tensor:
    dataset_index = int(shard.dataset_indices[offset])
    episode = _metadata_at(shard, "episode_indices", offset)
    frame = _metadata_at(shard, "frame_indices", offset)
    candidate = lookup.get(dataset_index - 1)
    if candidate is None:
        return noop_action
    source, source_offset = candidate
    if (
        _metadata_at(source, "episode_indices", source_offset) != episode
        or _metadata_at(source, "frame_indices", source_offset) != frame - 1
        or not bool(source.valid_mask[source_offset, 0])
    ):
        return noop_action
    return source.actions[source_offset, 0]


def iter_v2_1_batches(
    directory: str | Path,
    *,
    batch_size: int,
    history_length: int,
    seed: int,
    epoch: int,
    shuffle: bool = True,
    shard_shuffle_block_size: int = 1,
    mmap_shards: bool = False,
    history_stride: int = 1,
    action_sidecar: LongHorizonActionSidecar | None = None,
    row_transform: Callable[[dict[str, Tensor]], dict[str, Tensor]] | None = None,
) -> Iterator[dict[str, Tensor]]:
    """Yield one full cache traversal and assemble batches across shard boundaries.

    History lookup uses global dataset indices and verifies both episode id and
    frame continuity.  Missing history at episode starts is filled by the
    current frame, never by a preceding episode.  Cache shards are a storage
    detail: a requested batch may span several shards instead of being silently
    capped by the shard row count.
    """

    if (
        batch_size <= 0
        or history_length <= 0
        or history_stride <= 0
        or shard_shuffle_block_size <= 0
        or seed < 0
        or epoch < 0
    ):
        raise ValueError("batch/history sizes must be positive and seed/epoch non-negative")
    manifest = validate_v2_1_manifest(directory)
    paths = list(list_feature_shards(directory))
    if not paths:
        raise ValueError("V2.1 cache has no feature shards")
    canonical_index = {path: index for index, path in enumerate(paths)}
    generator = random.Random(seed + epoch)
    if shard_shuffle_block_size > 1:
        path_groups = [
            list(paths[start : start + shard_shuffle_block_size])
            for start in range(0, len(paths), shard_shuffle_block_size)
        ]
        if shuffle:
            generator.shuffle(path_groups)
            for group in path_groups:
                generator.shuffle(group)
    else:
        order = list(paths)
        if shuffle:
            generator.shuffle(order)
        path_groups = [[path] for path in order]
    history_count = history_length - 1
    pending: dict[str, list[Tensor]] = {}
    pending_rows = 0

    def append_and_drain(batch: dict[str, Tensor]) -> Iterator[dict[str, Tensor]]:
        nonlocal pending, pending_rows
        if pending and set(pending) != set(batch):
            raise RuntimeError("V2.1 batches changed tensor fields during one traversal")
        for name, value in batch.items():
            pending.setdefault(name, []).append(value)
        pending_rows += int(batch["dataset_indices"].shape[0])
        while pending_rows >= batch_size:
            merged = {name: torch.cat(values, dim=0) for name, values in pending.items()}
            yield {name: value[:batch_size] for name, value in merged.items()}
            remainder = {name: value[batch_size:] for name, value in merged.items()}
            pending_rows -= batch_size
            pending = {
                name: [value]
                for name, value in remainder.items()
                if value.shape[0] > 0
            }

    for path_group in path_groups:
        shard_cache: dict[Path, FeatureShard] = {}

        def cached_shard(path: Path) -> FeatureShard:
            shard = shard_cache.get(path)
            if shard is None:
                shard = load_feature_shard(path, mmap=mmap_shards)
                validate_v2_1_shard(shard)
                shard_cache[path] = shard
            return shard

        for path in path_group:
            shard = cached_shard(path)
            path_index = canonical_index[path]
            required_lookback = history_count * history_stride + 1
            previous_shards: list[FeatureShard] = []
            available_lookback = 0
            previous_index = path_index - 1
            while previous_index >= 0 and available_lookback < required_lookback:
                previous_shard = cached_shard(paths[previous_index])
                previous_shards.append(previous_shard)
                available_lookback += len(previous_shard.dataset_indices)
                previous_index -= 1
            lookup = _history_source(shard, tuple(reversed(previous_shards)))
            noop_action = _normalized_noop_action(
                manifest, int(shard.actions.shape[-1])
            ).to(dtype=shard.actions.dtype)
            sample_order = list(range(len(shard.dataset_indices)))
            if shuffle:
                generator.shuffle(sample_order)
            for start in range(0, len(sample_order), batch_size):
                offsets = sample_order[start : start + batch_size]
                index_tensor = torch.tensor(offsets, dtype=torch.long)
                history_visual: list[Tensor] = []
                history_semantic: list[Tensor] = []
                history_states: list[Tensor] = []
                history_previous_actions: list[Tensor] = []
                previous_actions: list[Tensor] = []
                for offset in offsets:
                    dataset_index = int(shard.dataset_indices[offset])
                    episode = _metadata_at(shard, "episode_indices", offset)
                    frame = _metadata_at(shard, "frame_indices", offset)
                    sample_visual: list[Tensor] = []
                    sample_semantic: list[Tensor] = []
                    sample_states: list[Tensor] = []
                    sample_previous_actions: list[Tensor] = []
                    current_previous_action = _previous_action_at(
                        lookup, shard, offset, noop_action=noop_action
                    )
                    previous_actions.append(current_previous_action)
                    for distance in range(history_count, 0, -1):
                        frame_distance = distance * history_stride
                        candidate = lookup.get(dataset_index - frame_distance)
                        if candidate is None:
                            source, source_offset = shard, offset
                        else:
                            source, source_offset = candidate
                            same_episode = _metadata_at(source, "episode_indices", source_offset) == episode
                            consecutive = (
                                _metadata_at(source, "frame_indices", source_offset)
                                == frame - frame_distance
                            )
                            if not (same_episode and consecutive):
                                source, source_offset = shard, offset
                        if source.semantic_features is None:  # pragma: no cover - validated
                            raise RuntimeError("semantic features disappeared after validation")
                        sample_visual.append(source.features[source_offset])
                        sample_semantic.append(source.semantic_features[source_offset])
                        sample_states.append(source.states[source_offset])
                        if source is shard and source_offset == offset:
                            sample_previous_actions.append(current_previous_action)
                        else:
                            sample_previous_actions.append(
                                _previous_action_at(
                                    lookup,
                                    source,
                                    source_offset,
                                    noop_action=noop_action,
                                )
                            )
                    if history_count:
                        history_visual.append(torch.stack(sample_visual))
                        history_semantic.append(torch.stack(sample_semantic))
                        history_states.append(torch.stack(sample_states))
                        history_previous_actions.append(
                            torch.stack(sample_previous_actions)
                        )
                if shard.semantic_features is None:  # pragma: no cover - validated
                    raise RuntimeError("semantic features disappeared after validation")
                batch: dict[str, Tensor] = {
                    "visual": shard.features.index_select(0, index_tensor),
                    "semantic": shard.semantic_features.index_select(0, index_tensor),
                    "states": shard.states.index_select(0, index_tensor),
                    "previous_actions": torch.stack(previous_actions),
                    "actions": shard.actions.index_select(0, index_tensor),
                    "valid_mask": shard.valid_mask.index_select(0, index_tensor),
                    "dataset_indices": torch.tensor(
                        [shard.dataset_indices[offset] for offset in offsets], dtype=torch.long
                    ),
                    "task_indices": shard.task_indices.index_select(0, index_tensor),
                    "episode_indices": shard.episode_indices.index_select(0, index_tensor),
                }
                if action_sidecar is not None:
                    sidecar_actions, sidecar_mask = action_sidecar.select(
                        batch["dataset_indices"]
                    )
                    batch["actions"] = sidecar_actions
                    batch["valid_mask"] = sidecar_mask
                if history_count:
                    batch["history_visual"] = torch.stack(history_visual)
                    batch["history_semantic"] = torch.stack(history_semantic)
                    batch["history_states"] = torch.stack(history_states)
                    batch["history_previous_actions"] = torch.stack(
                        history_previous_actions
                    )
                if row_transform is not None:
                    batch = row_transform(batch)
                if batch['dataset_indices'].numel():
                    yield from append_and_drain(batch)
    if pending_rows:
        yield {name: torch.cat(values, dim=0) for name, values in pending.items()}
