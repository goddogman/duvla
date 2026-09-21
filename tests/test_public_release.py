from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import pytest
import torch

from scripts.prepare_public_release import (
    LIMITS, SUITES, audit_evaluation, digest, export_model, export_source, public_metadata,
)
from duvla.models.qwen_backbone import QwenBackboneConfig


def make_evaluation(root: Path, checkpoint_hash: str) -> None:
    for suite in SUITES:
        p = root / suite
        p.mkdir(parents=True)
        config = {
            "suite": suite, "checkpoint_sha256": checkpoint_hash,
            "benchmark_task_index": False, "task_routing": "natural_language",
            "control_frequency_hz": 20, "action_steps": 2, "flow_samples": 5,
            "flow_seed": 23, "camera_size": 128, "settle_steps": 10,
            "max_steps": LIMITS[suite], "candidate_aggregation": "coordinate_median",
        }
        rows = [{"task_id": t, "init_state_id": s, "success": True,
                 "env_seed": t * 100000 + s * 1000 + 23,
                 "flow_noise_seed": t * 100000 + s * 1000 + 23}
                for t in range(10) for s in range(50)]
        (p / "summary.json").write_text(json.dumps({
            "complete": True, "run_config": config, "episodes": rows,
            "aggregate": {"episode_count": 500, "success_count": 500, "success_rate": 1.0},
        }))


def test_release_audit_rejects_duplicate_states(tmp_path: Path) -> None:
    make_evaluation(tmp_path, "sha")
    assert audit_evaluation(tmp_path, "sha")["success_count"] == 2000
    p = tmp_path / SUITES[0] / "summary.json"
    data = json.loads(p.read_text())
    data["episodes"][-1] = data["episodes"][0]
    p.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="唯一"):
        audit_evaluation(tmp_path, "sha")


def test_release_audit_rejects_wrong_checkpoint_and_protocol(tmp_path: Path) -> None:
    make_evaluation(tmp_path, "sha")
    with pytest.raises(ValueError, match="协议"):
        audit_evaluation(tmp_path, "different")
    p = tmp_path / SUITES[0] / "summary.json"
    data = json.loads(p.read_text())
    data["run_config"]["benchmark_task_index"] = True
    p.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="协议"):
        audit_evaluation(tmp_path, "sha")


def test_export_preserves_weights_normalization_and_original(tmp_path: Path) -> None:
    source = tmp_path / "original.pt"
    manifest = tmp_path / "manifest.json"
    template = tmp_path / "card.md"
    signature = {"dataset_root": "/home/private-user/data", "action_mean": [0.2]}
    weights = {"weight": torch.arange(8).reshape(2, 4).float()}
    torch.save({"version": "V3.31", "training_complete": True, "benchmark_task_index": False,
                "task_routing": "natural_language", "cache_signature": signature,
                "model_config": {"action_dim": 7}, "model_state_dict": weights}, source)
    manifest.write_text(json.dumps({"schema_version": 7, "cache_signature": signature,
                                    "action_mean": [0.2], "action_std": [0.7]}))
    original_sha = digest(source)
    template.write_text("# {{VERSION}}")
    evaluation = tmp_path / "evaluation"
    make_evaluation(evaluation, original_sha)
    destination = tmp_path / "model"
    export_model(source, manifest, evaluation, destination, template)
    exported = torch.load(destination / "policy.pt", weights_only=True)
    metadata = json.loads((destination / "train_manifest.json").read_text())
    assert torch.equal(exported["model_state_dict"]["weight"], weights["weight"])
    assert exported["cache_signature"] == metadata["cache_signature"]
    assert metadata["action_mean"] == [0.2] and metadata["action_std"] == [0.7]
    assert "private-user" not in json.dumps(metadata)
    assert digest(source) == original_sha
    assert json.loads((destination / "provenance.json").read_text())["tensor_equality_verified"]
    with pytest.raises(FileExistsError):
        export_model(source, manifest, evaluation, destination, template)


def test_source_allowlist_omits_local_artifacts_and_includes_helpers(tmp_path: Path) -> None:
    root = tmp_path / "research"
    (root / "scripts").mkdir(parents=True)
    (root / "scripts/main.py").write_text("from helper import foo\n")
    (root / "scripts/helper.py").write_text("foo = 1\n")
    (root / "AGENTS.local.md").write_text("private")
    (root / ".env").write_text("fake secret")
    manifest = tmp_path / "allow.json"
    manifest.write_text(json.dumps({"include": ["scripts/main.py"]}))
    destination = tmp_path / "source"
    export_source(root, manifest, destination)
    assert (destination / "scripts/helper.py").is_file()
    assert not (destination / "AGENTS.local.md").exists()
    assert not (destination / ".env").exists()


def test_backbone_path_can_be_relocated(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("DUVLA_QWEN_PATH", str(tmp_path))
    assert QwenBackboneConfig().model_path == tmp_path
    assert QwenBackboneConfig().local_files_only is True
    assert QwenBackboneConfig(model_path=tmp_path / "explicit").model_path == tmp_path / "explicit"


def test_metadata_redaction_keeps_tensors_and_hashes() -> None:
    tensor = torch.ones(2)
    result = public_metadata({"path": "/home/a/model", "sha256": "abc123", "w": tensor})
    assert result["w"] is tensor
    assert result["sha256"] == "abc123"
    assert result["path"] == "${LOCAL_HOME}/model"


def test_public_source_relocates_without_author_workspace(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    destination = tmp_path / "arbitrary clone name with spaces"
    export_source(root, root / "release/source_manifest.json", destination)
    assert (destination / "src/duvla/__init__.py").is_file()
    assert not (destination / "src/qwen_vla").exists()
    for folder in ("src", "scripts"):
        for path in (destination / folder).rglob("*.py"):
            content = path.read_text()
            assert "/home/yj-dsw" not in content, path
            assert "qwen_vla" not in content, path
    # -I 忽略外部PYTHONPATH；只注入新目录，避免意外导入原工作区的包。
    result = subprocess.run(
        [sys.executable, "-I", "-c",
         "import sys; from pathlib import Path; "
         "sys.path.insert(0, str(Path.cwd() / 'src')); "
         "import duvla; from duvla.models.duvla_v2_1 import DuvlaV21Policy; "
         "assert Path(duvla.__file__).is_relative_to(Path.cwd()); "
         "print(duvla.__name__)"],
        cwd=destination, capture_output=True, text=True, timeout=60, check=True,
    )
    assert result.stdout.strip() == "duvla"
