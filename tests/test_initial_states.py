from __future__ import annotations

import pickle
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from duvla.evaluation.initial_states import load_initial_state_file, load_libero_initial_states


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_numpy_states_preserve_values_and_restore_allowlist(tmp_path: Path, dtype: type) -> None:
    states = np.arange(50 * 47, dtype=dtype).reshape(50, 47)
    path = tmp_path / "task.pruned_init"
    torch.save(states, path)
    before = torch.serialization.get_safe_globals()
    actual = load_initial_state_file(path)
    np.testing.assert_array_equal(actual, states)
    assert actual.dtype == states.dtype
    assert torch.serialization.get_safe_globals() == before


@pytest.mark.parametrize("states", [np.zeros(4), np.empty((0, 47)), np.full((2, 3), np.nan)])
def test_rejects_invalid_state_arrays(tmp_path: Path, states: np.ndarray) -> None:
    path = tmp_path / "task.pruned_init"
    torch.save(states, path)
    with pytest.raises(ValueError, match="initial states"):
        load_initial_state_file(path)


def _forbidden_payload() -> np.ndarray:
    raise AssertionError("the restricted unpickler must not execute this function")


class _UntrustedState:
    def __reduce__(self) -> tuple:
        return _forbidden_payload, ()


def test_does_not_fall_back_to_unrestricted_pickle(tmp_path: Path) -> None:
    path = tmp_path / "task.pruned_init"
    torch.save(_UntrustedState(), path)
    before = torch.serialization.get_safe_globals()
    with pytest.raises(pickle.UnpicklingError):
        load_initial_state_file(path)
    assert torch.serialization.get_safe_globals() == before


def test_preserves_existing_numpy_allowlist(tmp_path: Path) -> None:
    path = tmp_path / "task.pruned_init"
    torch.save(np.ones((2, 3)), path)
    with torch.serialization.safe_globals([np.ndarray]):
        before = torch.serialization.get_safe_globals()
        load_initial_state_file(path)
        assert torch.serialization.get_safe_globals() == before


def test_suite_loader_uses_configured_root_and_preserves_order(tmp_path: Path) -> None:
    states = np.arange(6, dtype=np.float64).reshape(2, 3)
    folder = tmp_path / "libero_10"
    folder.mkdir()
    torch.save(states, folder / "task.pruned_init")
    task = SimpleNamespace(problem_folder="libero_10", init_states_file="task.pruned_init")
    suite = SimpleNamespace(get_task=lambda task_index: task)
    np.testing.assert_array_equal(
        load_libero_initial_states(suite, 0, init_states_root=tmp_path), states
    )
    task.problem_folder = "../outside"
    with pytest.raises(ValueError, match="escapes"):
        load_libero_initial_states(suite, 0, init_states_root=tmp_path)


def test_public_evaluation_defaults_match_v331(monkeypatch: pytest.MonkeyPatch) -> None:
    import evaluate_duvla_v2_1 as evaluate

    monkeypatch.setattr("sys.argv", [
        "evaluate", "--checkpoint", "policy.pt", "--train-manifest", "train_manifest.json",
        "--suite", "libero_10", "--trace-dir", "output",
    ])
    args = evaluate.parse_args()
    assert (args.camera_size, args.fps, args.flow_samples, args.action_steps, args.flow_seed) == (
        128, 20, 5, 2, 23
    )
