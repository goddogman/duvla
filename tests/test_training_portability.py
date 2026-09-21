from pathlib import Path
from types import SimpleNamespace

import pytest

import duvla.training.resource_budget as budget
from scripts.train_duvla_v3_31 import validate_parent


def test_native_linux_does_not_require_windows(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(budget, '_is_wsl', lambda: False)
    monkeypatch.setattr(budget.shutil, 'disk_usage', lambda _: SimpleNamespace(free=100 * 2**30))
    assert budget.require_disk_budget(Path('.'), minimum_gib=20)['host_check'].startswith('not_applicable')


def test_wsl_requires_explicit_safe_drive_and_keeps_reserve(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(budget, '_is_wsl', lambda: True)
    monkeypatch.setattr(budget.shutil, 'disk_usage', lambda _: SimpleNamespace(free=100 * 2**30))
    monkeypatch.delenv('DUVLA_WSL_HOST_DRIVE', raising=False)
    with pytest.raises(RuntimeError, match='DUVLA_WSL_HOST_DRIVE'):
        budget.require_disk_budget(Path('.'), minimum_gib=20)
    monkeypatch.setenv('DUVLA_WSL_HOST_DRIVE', 'E;echo bad')
    with pytest.raises(RuntimeError):
        budget.require_disk_budget(Path('.'), minimum_gib=20)
    monkeypatch.setenv('DUVLA_WSL_HOST_DRIVE', 'E')
    monkeypatch.setattr(Path, 'exists', lambda _: True)
    calls = []
    def shell(args, **kwargs):
        calls.append(args)
        return SimpleNamespace(stdout=str(30 * 2**30))
    monkeypatch.setattr(budget.subprocess, 'run', shell)
    assert budget.require_disk_budget(Path('.'), minimum_gib=20)['windows_host_drive'] == 'E'
    assert 'Get-PSDrive -Name E' in calls[0][-1]
    monkeypatch.setattr(budget.subprocess, 'run', lambda *a, **k: SimpleNamespace(stdout=str(2**30)))
    with pytest.raises(RuntimeError, match='安全余量'):
        budget.require_disk_budget(Path('.'), minimum_gib=20)


def test_new_parent_requires_full_budget_and_matching_lineage() -> None:
    parent = {
        'format': 'duvla_v3_29', 'version': 'V3.29', 'formal': True,
        'training_complete': True, 'checkpoint_complete': True,
        'epoch': 30, 'effective_epochs': 30., 'action_offset': 1,
        'uses_reward': False, 'uses_success': False,
        'uses_evaluation_initial_states': False, 'benchmark_task_index': False,
        'cache_signature': {'source': 'new-local-cache'}, 'sidecar_sha256': 'new-sidecar',
        'model_config': {'ordered_language_bridge': True, 'causal_action_attention': True,
                         'candidate_aggregation': 'coordinate_median'},
    }
    base = {'cache_signature': parent['cache_signature']}
    validate_parent(parent, base, 'new-sidecar')
    for change in ({'formal': False}, {'epoch': 10}, {'effective_epochs': 0.1},
                   {'uses_success': True}, {'benchmark_task_index': True},
                   {'sidecar_sha256': 'wrong'}, {'cache_signature': {'source': 'wrong'}}):
        with pytest.raises(ValueError):
            validate_parent({**parent, **change}, base, 'new-sidecar')
