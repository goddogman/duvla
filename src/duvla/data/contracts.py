"""Explicit, dependency-light contracts for LIBERO v3 frame data.

The module deliberately does not assign semantic names to the eight state
values or seven action values.  Those meanings must be established from the
dataset conversion and simulator before a policy uses them.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from numbers import Integral, Real
from typing import Mapping, Sequence

STATE_DIM = 8
ACTION_DIM = 7
STATE_KEY = "observation.state"
ACTION_KEY = "action"


class ContractError(ValueError):
    """Raised when a sample violates the current data contract."""


def _vector(value: object, *, name: str, dimension: int) -> tuple[float, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ContractError(f"{name} must be a numeric sequence")
    if len(value) != dimension:
        raise ContractError(f"{name} must have dimension {dimension}, got {len(value)}")
    result: list[float] = []
    for item in value:
        if not isinstance(item, Real) or isinstance(item, bool) or not isfinite(float(item)):
            raise ContractError(f"{name} must contain finite real numbers")
        result.append(float(item))
    return tuple(result)


def _index(value: object, *, name: str) -> int:
    if not isinstance(value, Integral) or isinstance(value, bool) or int(value) < 0:
        raise ContractError(f"{name} must be a non-negative integer")
    return int(value)


@dataclass(frozen=True)
class Frame:
    """Validated scalar/vector fields needed by an action policy."""

    state: tuple[float, ...]
    action: tuple[float, ...]
    timestamp: float
    frame_index: int
    episode_index: int
    index: int
    task_index: int


def validate_frame(record: Mapping[str, object]) -> Frame:
    """Validate and convert one LIBERO v3 Parquet-like record."""

    required = {
        STATE_KEY,
        ACTION_KEY,
        "timestamp",
        "frame_index",
        "episode_index",
        "index",
        "task_index",
    }
    missing = sorted(required.difference(record))
    if missing:
        raise ContractError(f"missing required fields: {', '.join(missing)}")
    timestamp = record["timestamp"]
    if not isinstance(timestamp, Real) or isinstance(timestamp, bool) or not isfinite(float(timestamp)):
        raise ContractError("timestamp must be a finite real number")
    return Frame(
        state=_vector(record[STATE_KEY], name=STATE_KEY, dimension=STATE_DIM),
        action=_vector(record[ACTION_KEY], name=ACTION_KEY, dimension=ACTION_DIM),
        timestamp=float(timestamp),
        frame_index=_index(record["frame_index"], name="frame_index"),
        episode_index=_index(record["episode_index"], name="episode_index"),
        index=_index(record["index"], name="index"),
        task_index=_index(record["task_index"], name="task_index"),
    )


def validate_episode(records: Sequence[Mapping[str, object]]) -> tuple[Frame, ...]:
    """Validate temporal and episode continuity without changing values."""

    if not records:
        raise ContractError("an episode must contain at least one frame")
    frames = tuple(validate_frame(record) for record in records)
    episode = frames[0].episode_index
    if any(frame.episode_index != episode for frame in frames):
        raise ContractError("all frames in an episode must share episode_index")
    for previous, current in zip(frames, frames[1:]):
        if current.frame_index != previous.frame_index + 1:
            raise ContractError("frame_index must increase by one within an episode")
        if current.index != previous.index + 1:
            raise ContractError("global index must increase by one within an episode")
        if current.timestamp <= previous.timestamp:
            raise ContractError("timestamps must be strictly increasing")
    return frames


def action_chunk(
    actions: Sequence[Sequence[Real]], *, start: int, horizon: int
) -> tuple[tuple[tuple[float, ...], ...], tuple[bool, ...]]:
    """Return a fixed-size action chunk and a mask for the episode tail.

    Tail positions are zero-padded and *must* be excluded from the loss using
    the returned mask.  No action semantics or normalization is applied here.
    """

    if start < 0 or horizon <= 0:
        raise ContractError("start must be non-negative and horizon must be positive")
    chunk: list[tuple[float, ...]] = []
    mask: list[bool] = []
    for offset in range(horizon):
        position = start + offset
        if position < len(actions):
            chunk.append(_vector(actions[position], name="action", dimension=ACTION_DIM))
            mask.append(True)
        else:
            chunk.append((0.0,) * ACTION_DIM)
            mask.append(False)
    return tuple(chunk), tuple(mask)


@dataclass(frozen=True)
class MinMaxStats:
    """Explicit per-dimension min/max statistics for a later policy choice."""

    minimum: tuple[float, ...]
    maximum: tuple[float, ...]

    @classmethod
    def from_actions(cls, actions: Sequence[Sequence[Real]]) -> MinMaxStats:
        if not actions:
            raise ContractError("cannot fit normalization statistics on empty actions")
        vectors = tuple(_vector(action, name="action", dimension=ACTION_DIM) for action in actions)
        return cls(
            minimum=tuple(min(vector[index] for vector in vectors) for index in range(ACTION_DIM)),
            maximum=tuple(max(vector[index] for vector in vectors) for index in range(ACTION_DIM)),
        )

    def normalize(self, action: Sequence[Real]) -> tuple[float, ...]:
        vector = _vector(action, name="action", dimension=ACTION_DIM)
        result: list[float] = []
        for value, minimum, maximum in zip(vector, self.minimum, self.maximum):
            if maximum == minimum:
                result.append(0.0)
            else:
                result.append(2.0 * (value - minimum) / (maximum - minimum) - 1.0)
        return tuple(result)

    def denormalize(self, normalized: Sequence[Real]) -> tuple[float, ...]:
        vector = _vector(normalized, name="normalized action", dimension=ACTION_DIM)
        return tuple(
            minimum + (value + 1.0) * 0.5 * (maximum - minimum)
            for value, minimum, maximum in zip(vector, self.minimum, self.maximum)
        )
