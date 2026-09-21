"""Deterministic observation-history windows for cached VLA features."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor


@dataclass(frozen=True)
class TemporalSidecar:
    """CPU tensors aligned by ``dataset_indices`` for V8 conditioning."""

    dataset_indices: Tensor
    history_states: Tensor
    previous_actions: Tensor
    phase_labels: Tensor
    subtask_index: Tensor | None = None
    subtask_progress: Tensor | None = None
    cycle_state: Tensor | None = None
    visual_history_indices: Tensor | None = None

    def validate(
        self,
        *,
        state_dim: int = 8,
        action_dim: int = 7,
        history: int | None = None,
    ) -> None:
        if self.dataset_indices.ndim != 1:
            raise ValueError("dataset_indices must have shape [samples]")
        samples = self.dataset_indices.shape[0]
        expected_history = (
            int(self.history_states.shape[1])
            if history is None and self.history_states.ndim >= 2
            else history
        )
        if expected_history is None or self.history_states.shape != (
            samples,
            expected_history,
            state_dim,
        ):
            raise ValueError("history_states has an invalid shape")
        if self.previous_actions.shape != (samples, expected_history, action_dim):
            raise ValueError("previous_actions has an invalid shape")
        if self.phase_labels.shape != (samples,):
            raise ValueError("phase_labels must have shape [samples]")
        progress_values = (self.subtask_index, self.subtask_progress, self.cycle_state)
        if any(value is not None for value in progress_values) and not all(
            value is not None for value in progress_values
        ):
            raise ValueError("subtask progress fields must be provided together")
        if self.dataset_indices.dtype not in (torch.int32, torch.int64):
            raise ValueError("dataset_indices must be an integer tensor")
        if self.phase_labels.dtype not in (torch.int32, torch.int64):
            raise ValueError("phase_labels must be an integer tensor")
        if self.subtask_index is not None:
            if self.subtask_index.shape != (samples,) or self.cycle_state is None or self.cycle_state.shape != (samples,):
                raise ValueError("subtask_index and cycle_state must have shape [samples]")
            if self.subtask_progress is None or self.subtask_progress.shape != (samples,):
                raise ValueError("subtask_progress must have shape [samples]")
            if self.subtask_index.dtype not in (torch.int32, torch.int64):
                raise ValueError("subtask_index must be an integer tensor")
            if self.cycle_state.dtype not in (torch.int32, torch.int64):
                raise ValueError("cycle_state must be an integer tensor")
            if self.subtask_progress.dtype not in (torch.float16, torch.float32, torch.float64):
                raise ValueError("subtask_progress must be a floating tensor")
            if bool(((self.subtask_index < 0) | (self.subtask_index >= 4)).any()):
                raise ValueError("subtask_index must be in [0, 3]")
            if bool(((self.cycle_state < 0) | (self.cycle_state >= 4)).any()):
                raise ValueError("cycle_state must be in [0, 3]")
            if not torch.isfinite(self.subtask_progress.float()).all() or bool(
                ((self.subtask_progress < 0.0) | (self.subtask_progress > 1.0)).any()
            ):
                raise ValueError("subtask_progress must be finite and in [0, 1]")
        if self.visual_history_indices is not None:
            if (
                self.visual_history_indices.ndim != 2
                or self.visual_history_indices.shape[0] != samples
                or self.visual_history_indices.shape[1] <= 0
            ):
                raise ValueError(
                    "visual_history_indices must have shape [samples, history]"
                )
            if self.visual_history_indices.dtype not in (torch.int32, torch.int64):
                raise ValueError("visual_history_indices must be an integer tensor")
            if bool((self.visual_history_indices < 0).any()):
                raise ValueError("visual_history_indices cannot contain negative indices")
        if samples == 0 or torch.unique(self.dataset_indices).numel() != samples:
            raise ValueError("dataset_indices must be non-empty and unique")
        if bool(((self.phase_labels < 0) | (self.phase_labels >= 5)).any()):
            raise ValueError("phase_labels must be in [0, 4]")
        for name, value in (
            ("history_states", self.history_states),
            ("previous_actions", self.previous_actions),
        ):
            if not torch.isfinite(value.float()).all():
                raise ValueError(f"{name} contains non-finite values")


@dataclass(frozen=True)
class TaskIndexSidecar:
    """Task labels aligned with the raw ``dataset_indices`` in a feature cache.

    The labels are used only to fit language-conditioned task prototypes.  The
    deployed policy predicts the task from its Qwen features and does not
    receive a benchmark task id from the evaluator.
    """

    dataset_indices: Tensor
    task_indices: Tensor
    episode_indices: Tensor | None = None

    def validate(self, *, task_classes: int = 40) -> None:
        if self.dataset_indices.ndim != 1 or self.task_indices.ndim != 1:
            raise ValueError("task sidecar tensors must have shape [samples]")
        if self.dataset_indices.shape != self.task_indices.shape:
            raise ValueError("task sidecar tensors must have matching shapes")
        if self.dataset_indices.dtype not in (torch.int32, torch.int64):
            raise ValueError("dataset_indices must be an integer tensor")
        if self.task_indices.dtype not in (torch.int32, torch.int64):
            raise ValueError("task_indices must be an integer tensor")
        if self.dataset_indices.numel() == 0:
            raise ValueError("task sidecar cannot be empty")
        if torch.unique(self.dataset_indices).numel() != self.dataset_indices.numel():
            raise ValueError("dataset_indices must be unique")
        if bool(((self.task_indices < 0) | (self.task_indices >= task_classes)).any()):
            raise ValueError(f"task_indices must be in [0, {task_classes - 1}]")
        if self.episode_indices is not None:
            if self.episode_indices.ndim != 1 or self.episode_indices.shape != self.dataset_indices.shape:
                raise ValueError("episode_indices must have shape [samples]")
            if self.episode_indices.dtype not in (torch.int32, torch.int64):
                raise ValueError("episode_indices must be an integer tensor")
            if bool((self.episode_indices < 0).any()):
                raise ValueError("episode_indices cannot be negative")


def save_temporal_sidecar(path: str | Path, sidecar: TemporalSidecar) -> None:
    """Atomically save a validated CPU sidecar."""

    sidecar.validate(history=int(sidecar.history_states.shape[1]))
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    torch.save(
        {
            "dataset_indices": sidecar.dataset_indices.cpu(),
            "history_states": sidecar.history_states.cpu(),
            "previous_actions": sidecar.previous_actions.cpu(),
            "phase_labels": sidecar.phase_labels.cpu(),
            "subtask_index": None if sidecar.subtask_index is None else sidecar.subtask_index.cpu(),
            "subtask_progress": None if sidecar.subtask_progress is None else sidecar.subtask_progress.cpu(),
            "cycle_state": None if sidecar.cycle_state is None else sidecar.cycle_state.cpu(),
            "visual_history_indices": (
                None
                if sidecar.visual_history_indices is None
                else sidecar.visual_history_indices.cpu()
            ),
        },
        temporary,
    )
    temporary.replace(destination)


def load_temporal_sidecar(path: str | Path) -> TemporalSidecar:
    """Load and validate a CPU sidecar."""

    payload = torch.load(Path(path), map_location="cpu", weights_only=True)
    sidecar = TemporalSidecar(
        dataset_indices=payload["dataset_indices"],
        history_states=payload["history_states"],
        previous_actions=payload["previous_actions"],
        phase_labels=payload["phase_labels"],
        subtask_index=payload.get("subtask_index"),
        subtask_progress=payload.get("subtask_progress"),
        cycle_state=payload.get("cycle_state"),
        visual_history_indices=payload.get("visual_history_indices"),
    )
    sidecar.validate(history=int(sidecar.history_states.shape[1]))
    return sidecar


def save_task_index_sidecar(path: str | Path, sidecar: TaskIndexSidecar) -> None:
    """Atomically save task labels aligned with a feature-cache split."""

    sidecar.validate()
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    torch.save(
        {
            "dataset_indices": sidecar.dataset_indices.cpu(),
            "task_indices": sidecar.task_indices.cpu(),
            "episode_indices": (
                None if sidecar.episode_indices is None else sidecar.episode_indices.cpu()
            ),
        },
        temporary,
    )
    temporary.replace(destination)


def load_task_index_sidecar(path: str | Path) -> TaskIndexSidecar:
    """Load and validate a task-index sidecar."""

    payload = torch.load(Path(path), map_location="cpu", weights_only=True)
    sidecar = TaskIndexSidecar(
        dataset_indices=payload["dataset_indices"],
        task_indices=payload["task_indices"],
        episode_indices=payload.get("episode_indices"),
    )
    sidecar.validate()
    return sidecar


def build_history_indices(
    dataset_indices: Sequence[int],
    frame_episode_indices: Sequence[int],
    history: int,
) -> Tensor:
    """Return cache-row indices for oldest-to-current episode windows.

    Windows never cross an episode boundary.  At an episode start, the first
    available row is repeated so every sample has a fixed shape.  The returned
    tensor indexes rows in the original ``dataset_indices`` order.
    """

    if history <= 0:
        raise ValueError("history must be positive")
    if not dataset_indices:
        raise ValueError("dataset_indices cannot be empty")
    if len(set(dataset_indices)) != len(dataset_indices):
        raise ValueError("dataset_indices must be unique")
    if any(index < 0 or index >= len(frame_episode_indices) for index in dataset_indices):
        raise ValueError("dataset_indices contain an out-of-range dataset row")
    row_for_dataset_index = {int(index): row for row, index in enumerate(dataset_indices)}
    windows: list[list[int]] = []
    for current_row, dataset_index in enumerate(dataset_indices):
        episode = frame_episode_indices[dataset_index]
        rows = [current_row]
        previous = int(dataset_index) - 1
        while len(rows) < history:
            previous_row = row_for_dataset_index.get(previous)
            if previous_row is None or frame_episode_indices[previous] != episode:
                break
            rows.append(previous_row)
            previous -= 1
        rows.reverse()
        rows = [rows[0]] * (history - len(rows)) + rows
        windows.append(rows)
    return torch.tensor(windows, dtype=torch.long)


def build_strided_past_indices(
    dataset_indices: Sequence[int],
    frame_episode_indices: Sequence[int],
    *,
    history: int,
    stride: int,
) -> Tensor:
    """Return oldest-to-newest past rows at a fixed control stride.

    The current row is excluded whenever enough history exists.  Missing
    history at an episode boundary is padded with the first selected row from
    that episode.  Because complete episodes belong to exactly one split,
    every returned row remains inside the provided split.
    """

    if history <= 0 or stride <= 0:
        raise ValueError("history and stride must be positive")
    if not dataset_indices:
        raise ValueError("dataset_indices cannot be empty")
    if len(set(dataset_indices)) != len(dataset_indices):
        raise ValueError("dataset_indices must be unique")
    if any(index < 0 or index >= len(frame_episode_indices) for index in dataset_indices):
        raise ValueError("dataset_indices contain an out-of-range dataset row")
    row_for_dataset_index = {int(index): row for row, index in enumerate(dataset_indices)}
    episode_first_row: dict[int, int] = {}
    for row, dataset_index in enumerate(dataset_indices):
        episode_first_row.setdefault(int(frame_episode_indices[dataset_index]), row)
    windows: list[list[int]] = []
    for dataset_index in dataset_indices:
        episode = int(frame_episode_indices[dataset_index])
        first_row = episode_first_row[episode]
        rows: list[int] = []
        for distance in range(history, 0, -1):
            candidate = int(dataset_index) - distance * stride
            candidate_row = row_for_dataset_index.get(candidate)
            if (
                candidate_row is None
                or int(frame_episode_indices[candidate]) != episode
            ):
                candidate_row = first_row
            rows.append(candidate_row)
        windows.append(rows)
    return torch.tensor(windows, dtype=torch.long)
