"""Deterministic episode splits and train-only action statistics."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Mapping, Sequence

from .contracts import MinMaxStats
from .libero_v3 import LiberoV3Dataset


@dataclass(frozen=True)
class EpisodeSplit:
    """Disjoint episode IDs for an episode-level train/validation split."""

    train_episodes: tuple[int, ...]
    validation_episodes: tuple[int, ...]
    seed: int
    validation_fraction: float

    def __post_init__(self) -> None:
        if set(self.train_episodes) & set(self.validation_episodes):
            raise ValueError("train and validation episodes must be disjoint")


def split_episodes(
    episode_to_task: Mapping[int, int], *, validation_fraction: float = 0.2, seed: int = 0
) -> EpisodeSplit:
    """Split episodes within each task, preserving every task in both splits."""

    if not episode_to_task:
        raise ValueError("episode_to_task cannot be empty")
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be between zero and one")
    rng = random.Random(seed)
    train: list[int] = []
    validation: list[int] = []
    by_task: dict[int, list[int]] = {}
    for episode, task in episode_to_task.items():
        if episode < 0 or task < 0:
            raise ValueError("episode and task indices must be non-negative")
        by_task.setdefault(task, []).append(episode)
    for task in sorted(by_task):
        episodes = sorted(by_task[task])
        if len(episodes) < 2:
            raise ValueError(f"task {task} needs at least two episodes for a split")
        rng.shuffle(episodes)
        validation_count = max(1, round(len(episodes) * validation_fraction))
        validation_count = min(validation_count, len(episodes) - 1)
        validation.extend(episodes[:validation_count])
        train.extend(episodes[validation_count:])
    return EpisodeSplit(
        train_episodes=tuple(sorted(train)),
        validation_episodes=tuple(sorted(validation)),
        seed=seed,
        validation_fraction=validation_fraction,
    )


def frame_indices_for_episodes(
    dataset: LiberoV3Dataset, episodes: Sequence[int]
) -> tuple[int, ...]:
    """Map episode IDs to row positions without crossing episode boundaries."""

    allowed = set(episodes)
    return tuple(
        index for index, episode in enumerate(dataset.frame_episode_indices) if episode in allowed
    )


def fit_action_stats(dataset: LiberoV3Dataset, frame_indices: Sequence[int]) -> MinMaxStats:
    """Fit raw action min/max statistics from explicitly supplied train rows."""

    if not frame_indices:
        raise ValueError("frame_indices cannot be empty")
    actions = []
    for index in frame_indices:
        sample = dataset[index]
        if not sample.action_mask[0]:
            raise ValueError(f"frame {index} has no valid first action")
        actions.append(sample.action_chunk[0])
    return MinMaxStats.from_actions(actions)
