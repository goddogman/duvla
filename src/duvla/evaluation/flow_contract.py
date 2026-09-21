"""Normalization and action-boundary contracts for the full-data Flow policy."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class StandardizationStats:
    """Per-dimension train-split standardization statistics."""

    mean: tuple[float, ...]
    std: tuple[float, ...]

    @classmethod
    def from_manifest(cls, manifest: Mapping[str, object], prefix: str) -> "StandardizationStats":
        mean = manifest.get(f"{prefix}_mean")
        std = manifest.get(f"{prefix}_std")
        if not isinstance(mean, Sequence) or isinstance(mean, (str, bytes)):
            raise ValueError(f"manifest is missing {prefix}_mean")
        if not isinstance(std, Sequence) or isinstance(std, (str, bytes)):
            raise ValueError(f"manifest is missing {prefix}_std")
        if len(mean) != len(std) or not mean:
            raise ValueError(f"invalid {prefix} normalization dimensions")
        values = cls(tuple(float(value) for value in mean), tuple(float(value) for value in std))
        if any(not np.isfinite(value) or value <= 0.0 for value in values.std):
            raise ValueError(f"{prefix} standard deviations must be finite and positive")
        return values

    @property
    def dim(self) -> int:
        return len(self.mean)

    def normalize(self, values: np.ndarray) -> np.ndarray:
        array = np.asarray(values, dtype=np.float32)
        if array.shape[-1] != self.dim:
            raise ValueError(f"expected final dimension {self.dim}, got {array.shape[-1]}")
        return (array - np.asarray(self.mean, dtype=np.float32)) / np.asarray(self.std, dtype=np.float32)

    def denormalize(self, values: np.ndarray) -> np.ndarray:
        array = np.asarray(values, dtype=np.float32)
        if array.shape[-1] != self.dim:
            raise ValueError(f"expected final dimension {self.dim}, got {array.shape[-1]}")
        return array * np.asarray(self.std, dtype=np.float32) + np.asarray(self.mean, dtype=np.float32)


def flow_normalized_action_to_env(
    normalized_action: np.ndarray,
    action_stats: StandardizationStats,
) -> np.ndarray:
    """Convert model output to LIBERO action units without changing gripper semantics.

    The dataset stores the seventh action dimension natively as -1=open and
    +1=close.  Only physical command bounds are clipped; no 0/1 remapping or
    sign inversion is applied here.
    """

    raw = action_stats.denormalize(normalized_action)
    if action_stats.dim != 7:
        raise ValueError("LIBERO action contract requires seven dimensions")
    result = raw.copy()
    result[..., :6] = np.clip(result[..., :6], -1.0, 1.0)
    result[..., 6] = np.clip(result[..., 6], -1.0, 1.0)
    return result.astype(np.float32, copy=False)
