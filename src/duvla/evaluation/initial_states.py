"""Load LIBERO's numeric initial states with PyTorch's restricted unpickler."""

from __future__ import annotations

from importlib import import_module
from pathlib import Path
from typing import Any

import numpy as np
import torch


def load_initial_state_file(path: str | Path) -> np.ndarray:
    """Read a finite [episodes, simulator_state] array without changing torch defaults.

    Official LIBERO files contain NumPy arrays pickled by older NumPy versions.
    Allow only their reconstruction types for this load; never fall back to
    unrestricted pickle or change the global torch.load function.
    """
    core = import_module(
        "numpy._core.multiarray" if hasattr(np, "_core") else "numpy.core.multiarray"
    )
    allowed = [
        bytes,
        np.ndarray,
        np.dtype,
        (core._reconstruct, "numpy.core.multiarray._reconstruct"),
        (core._reconstruct, "numpy._core.multiarray._reconstruct"),
        type(np.dtype("float32")),
        type(np.dtype("float64")),
    ]
    existing = torch.serialization.get_safe_globals()
    with torch.serialization.safe_globals([item for item in allowed if item not in existing]):
        raw = torch.load(Path(path), map_location="cpu", weights_only=True)
    if isinstance(raw, torch.Tensor):
        raw = raw.numpy()
    states = np.asarray(raw)
    if states.ndim != 2 or min(states.shape) == 0:
        raise ValueError("LIBERO initial states must be a nonempty [episodes, state] array")
    if states.dtype not in (np.dtype("float32"), np.dtype("float64")):
        raise ValueError("LIBERO initial states must contain float32 or float64 values")
    if not np.isfinite(states).all():
        raise ValueError("LIBERO initial states contain non-finite values")
    return states


def load_libero_initial_states(
    task_suite: Any, task_index: int, *, init_states_root: str | Path | None = None
) -> np.ndarray:
    """Resolve a suite task's official initial-state file beneath the asset root."""
    if init_states_root is None:
        from libero.libero import get_libero_path

        init_states_root = get_libero_path("init_states")
    root = Path(init_states_root).resolve()
    task = task_suite.get_task(task_index)
    path = (root / task.problem_folder / task.init_states_file).resolve()
    if not path.is_relative_to(root):
        raise ValueError("LIBERO initial-state path escapes its configured asset root")
    return load_initial_state_file(path)
