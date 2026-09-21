from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from duvla.models.duvla_v2_1 import DuvlaV21Config, DuvlaV21Policy
from duvla.models.ordered_language_bridge import OrderedLanguageBridge
from duvla.models.robust_camera_fusion import RobustCrossCameraFusion
from duvla.training.v3_31_data import V331AugmentSidecar
from scripts.evaluate_duvla_v2_1 import _validate_v331_deployment


def test_variable_language_preserves_parent_positions_and_masks_padding() -> None:
    torch.manual_seed(3)
    parent = OrderedLanguageBridge(8, 24, 4, max_tokens=32)
    torch.manual_seed(3)
    extended = OrderedLanguageBridge(8, 24, 4, max_tokens=96)
    assert parent.state_dict().keys() == extended.state_dict().keys()
    for name, value in parent.state_dict().items():
        torch.testing.assert_close(value, extended.state_dict()[name], rtol=0, atol=0)
    with torch.no_grad():
        parent.gate.fill_(0.2)
        extended.gate.fill_(0.2)
    spatial = torch.randn(2, 16, 24)
    language = torch.randn(2, 2, 32, 8)
    mask = torch.arange(32)[None, None].expand(2, 2, -1) < 17
    torch.testing.assert_close(
        parent(spatial, language, mask), extended(spatial, language, mask), rtol=0, atol=0
    )
    long_language = torch.randn(2, 2, 70, 8)
    long_mask = torch.arange(70)[None, None].expand(2, 2, -1) < 51
    changed_padding = long_language.clone()
    changed_padding[~long_mask] = 1e4
    torch.testing.assert_close(
        extended(spatial, long_language, long_mask),
        extended(spatial, changed_padding, long_mask),
    )
    with pytest.raises(ValueError, match="exceeds"):
        extended(spatial, torch.randn(2, 2, 97, 8), torch.ones(2, 2, 97, dtype=torch.bool))


def test_cross_camera_fusion_is_identity_then_receives_gradient() -> None:
    bridge = RobustCrossCameraFusion(24, 4)
    spatial = torch.randn(2, 18, 24)
    semantic = torch.randn(2, 24)
    state = torch.randn(2, 24)
    assert torch.equal(bridge(spatial, semantic, state), spatial)
    bridge(spatial, semantic, state).square().mean().backward()
    assert bridge.output.weight.grad is not None
    assert float(bridge.output.weight.grad.abs().sum()) > 0


def test_v331_policy_parent_parity_and_flow_field_gradient() -> None:
    base = DuvlaV21Config(
        feature_dim=32,
        adapter_rank=8,
        hidden_dim=24,
        expert_layers=2,
        expert_heads=4,
        context_layers=1,
        residual_layers=1,
        spatial_tokens=4,
        history_length=2,
        flow_steps=2,
        flow_samples=2,
        ordered_language_bridge=True,
    )
    torch.manual_seed(5)
    parent = DuvlaV21Policy(base)
    torch.manual_seed(5)
    candidate = DuvlaV21Policy(
        replace(base, language_max_tokens=96, cross_camera_fusion=True)
    )
    missing = candidate.load_state_dict(parent.state_dict(), strict=False)
    assert not missing.unexpected_keys
    assert missing.missing_keys
    assert all(name.startswith("cross_camera_bridge.") for name in missing.missing_keys)
    batch = 2
    common = {
        "visual": torch.randn(batch, 4, 2, 4, 32),
        "semantic": torch.randn(batch, 4, 1, 32),
        "state": torch.randn(batch, 8),
        "history_visual": torch.randn(batch, 1, 4, 2, 4, 32),
        "history_semantic": torch.randn(batch, 1, 4, 1, 32),
        "history_states": torch.randn(batch, 1, 8),
    }
    language = torch.randn(batch, 2, 32, 32)
    mask = torch.ones(batch, 2, 32, dtype=torch.bool)
    noise = torch.randn(batch, 2, 8, 7)
    parent_actions = parent.sample_actions(
        **common, language_tokens=language, language_mask=mask, noise=noise,
        apply_direct=False, apply_instruction=False, apply_gripper_event=False,
    )
    candidate_actions = candidate.sample_actions(
        **common, language_tokens=language, language_mask=mask, noise=noise,
        apply_direct=False, apply_instruction=False, apply_gripper_event=False,
    )
    torch.testing.assert_close(parent_actions, candidate_actions, rtol=0, atol=0)

    noisy = torch.randn(batch, 8, 7)
    time = torch.full((batch,), 0.4)
    velocity = candidate.predict_flow_velocity(
        common["visual"], common["semantic"], common["state"], noisy, time,
        history_visual=common["history_visual"],
        history_semantic=common["history_semantic"],
        history_states=common["history_states"],
        language_tokens=torch.randn(batch, 2, 70, 32),
        language_mask=torch.ones(batch, 2, 70, dtype=torch.bool),
    )
    assert velocity.shape == noisy.shape
    velocity.square().mean().backward()
    assert candidate.cross_camera_bridge is not None
    assert candidate.cross_camera_bridge.output.weight.grad is not None


def test_sparse_augment_sidecar_fetches_only_selected_rows(tmp_path) -> None:
    import hashlib
    import json

    base = tmp_path / "base"
    language = tmp_path / "language"
    augment = tmp_path / "augment"
    base.mkdir(); language.mkdir(); augment.mkdir()
    (base / "manifest.json").write_text("{}")
    (language / "manifest.json").write_text('{"complete": true}')
    manifest = {
        "formal": True,
        "complete": True,
        "uses_evaluation_initial_states": False,
        "base_manifest_sha256": hashlib.sha256(b"{}").hexdigest(),
        "language_manifest_sha256": hashlib.sha256(b'{"complete": true}').hexdigest(),
        "selected_indices": [2, 7],
        "shard_size": 2,
        "contract_sha256": "contract",
    }
    (augment / "manifest.json").write_text(json.dumps(manifest))
    torch.save(
        {
            "contract_sha256": "contract",
            "dataset_indices": torch.tensor([2, 7]),
            "visual": torch.randn(2, 1, 2, 4, 8),
            "semantic": torch.randn(2, 4, 1, 8),
            "language": torch.randn(9, 2, 8),
            "language_offsets": torch.tensor([0, 4, 9]),
        },
        augment / "shard-000000.pt",
    )
    store = V331AugmentSidecar(augment, base, language)
    result = store.fetch({"dataset_indices": torch.tensor([1, 2, 5, 7])})
    assert result is not None
    assert result["slots"].tolist() == [1, 3]
    assert result["visual"].shape == (2, 1, 2, 4, 8)
    assert result["language_tokens"].shape == (2, 2, 5, 8)
    assert result["language_mask"].sum(dim=(1, 2)).tolist() == [8, 10]


def test_v331_formal_deployment_contract() -> None:
    from types import SimpleNamespace

    checkpoint = {
        "format": "duvla_v3_31",
        "formal": True,
        "checkpoint_complete": True,
        "training_complete": True,
        "epoch": 30,
        "effective_epochs": 30.0,
        "planned_epochs": 30,
        "action_offset": 1,
        "uses_reward": False,
        "uses_success": False,
        "uses_evaluation_initial_states": False,
        "model_config": {
            "ordered_language_bridge": True,
            "language_max_tokens": 256,
            "cross_camera_fusion": True,
            "causal_action_attention": True,
            "candidate_aggregation": "coordinate_median",
        },
        "augmentation_contract": {
            "formal": True,
            "complete": True,
            "selected_rows": 10000,
        },
        "augmentation_sidecar_sha256": "test",
    }
    args = SimpleNamespace(
        camera_size=128, fps=20, action_steps=2, flow_samples=5, flow_seed=23,
        no_flip_views=False, state_clip=None, outcome_verifier=None,
        action_intent=None, candidate_aggregation=None,
    )
    _validate_v331_deployment(checkpoint, args)
    with pytest.raises(ValueError, match="architecture"):
        _validate_v331_deployment(
            {**checkpoint, "model_config": {**checkpoint["model_config"], "cross_camera_fusion": False}},
            args,
        )
