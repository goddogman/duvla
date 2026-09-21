#!/usr/bin/env python3
"""构建只写新目录的本地发布候选；不联网、不提交、不上传、不删除。"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path
import re
import shutil


SUITES = ("libero_10", "libero_spatial", "libero_object", "libero_goal")
LIMITS = dict(zip(SUITES, (520, 280, 280, 300)))


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def public_metadata(value: object) -> object:
    """只规范本机路径标记；张量、数值、来源hash保持不变。"""
    if isinstance(value, dict):
        return {key: public_metadata(item) for key, item in value.items()}
    if isinstance(value, list):
        return [public_metadata(item) for item in value]
    if isinstance(value, tuple):
        return tuple(public_metadata(item) for item in value)
    if isinstance(value, Path):
        return public_metadata(str(value))
    if isinstance(value, str):
        return re.sub(r"/home/[^/\s]+", "${LOCAL_HOME}", value)
    return value


def write_json(path: Path, payload: object) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def audit_evaluation(root: Path, checkpoint_sha256: str) -> dict:
    """校验40×50唯一状态、协议、原权重hash和原始逐集计数。"""
    suites = {}
    all_episodes = []
    for suite in SUITES:
        d = json.loads((root / suite / "summary.json").read_text())
        cfg = d["run_config"]
        expected = {
            "suite": suite, "checkpoint_sha256": checkpoint_sha256,
            "benchmark_task_index": False, "task_routing": "natural_language",
            "control_frequency_hz": 20, "action_steps": 2, "flow_samples": 5,
            "flow_seed": 23, "camera_size": 128, "settle_steps": 10,
            "max_steps": LIMITS[suite], "candidate_aggregation": "coordinate_median",
        }
        if d.get("complete") is not True or any(cfg.get(k) != v for k, v in expected.items()):
            raise ValueError(f"{suite}: 完整性、checkpoint或测评协议不匹配")
        episodes = d["episodes"]
        keys = [(r["task_id"], r["init_state_id"]) for r in episodes]
        if len(keys) != 500 or set(keys) != {(t, s) for t in range(10) for s in range(50)}:
            raise ValueError(f"{suite}: 必须是500个唯一的task/state组合")
        if any(type(r["success"]) is not bool for r in episodes):
            raise ValueError(f"{suite}: success必须为布尔值")
        success = sum(r["success"] for r in episodes)
        aggregate = {"episode_count": 500, "success_count": success, "success_rate": success / 500}
        if any(d["aggregate"].get(k) != v for k, v in aggregate.items()):
            raise ValueError(f"{suite}: 汇总与逐集结果不一致")
        suites[suite] = aggregate
        for row in episodes:
            all_episodes.append({"suite": suite, **{
                key: row[key] for key in ("task_id", "init_state_id", "env_seed", "flow_noise_seed", "success")
            }})
    success = sum(d["success_count"] for d in suites.values())
    return {"complete": True, "episode_count": 2000, "success_count": success,
            "success_rate": success / 2000, "original_checkpoint_sha256": checkpoint_sha256,
            "role": "完整开发测评，非独立盲测", "suites": suites, "episodes": all_episodes}


def inventory(destination: Path, kind: str, extra: dict | None = None) -> dict:
    files = {str(p.relative_to(destination)): {"bytes": p.stat().st_size, "sha256": digest(p)}
             for p in sorted(destination.rglob("*")) if p.is_file()}
    result = {"kind": kind, "publication_status": "local_draft_not_uploaded",
              "files": files, **(extra or {})}
    write_json(destination / "bundle_manifest.json", result)
    return {"kind": kind, "files": len(files), "bytes": sum(d["bytes"] for d in files.values()),
            "destination": str(destination)}


def export_model(checkpoint: Path, manifest: Path, evaluation: Path, destination: Path, template: Path) -> dict:
    import torch

    if destination.exists():
        raise FileExistsError(f"拒绝覆盖已有目录: {destination}")
    original_sha = digest(checkpoint)
    results = audit_evaluation(evaluation, original_sha)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    data = json.loads(manifest.read_text())
    if (payload.get("version") != "V3.31"
            or payload.get("training_complete") is not True
            or payload.get("benchmark_task_index") is not False
            or payload.get("task_routing") != "natural_language"
            or payload.get("cache_signature") != data.get("cache_signature")):
        raise ValueError("权重版本、训练完整性或归一化manifest不匹配")
    clean = public_metadata(payload)
    clean_data = public_metadata(data)
    destination.mkdir(parents=True, exist_ok=False)
    torch.save(clean, destination / "policy.pt")
    restored = torch.load(destination / "policy.pt", map_location="cpu", weights_only=True)
    for field in ("model_state_dict", "qwen_lora_state_dict"):
        if field not in payload:
            continue
        if restored[field].keys() != payload[field].keys():
            raise ValueError("导出权重键发生变化")
        for key, tensor in payload[field].items():
            if not torch.equal(tensor, restored[field][key]):
                raise ValueError(f"导出权重张量发生变化: {field}.{key}")
    if restored["cache_signature"] != clean_data["cache_signature"]:
        raise ValueError("公开权重与manifest签名不匹配")
    write_json(destination / "train_manifest.json", clean_data)
    write_json(destination / "model_config.json", clean["model_config"])
    write_json(destination / "evaluation.json", results)
    write_json(destination / "provenance.json", {
        "original_checkpoint_sha256": original_sha,
        "exported_checkpoint_sha256": digest(destination / "policy.pt"),
        "original_train_manifest_sha256": digest(manifest),
        "exported_train_manifest_sha256": digest(destination / "train_manifest.json"),
        "tensor_equality_verified": True,
        "metadata_change": "仅将/home/<user>替换为${LOCAL_HOME}，不用于运行时路径解析",
        "base_model": "Qwen/Qwen3-VL-2B-Instruct",
        "backbone_revision": "89644892e4d85e24eaac8bacfd4f463576704203",
        "backbone_included": False, "training_cache_required_for_inference": False,
        "original_evaluation_uses_original_checkpoint_hash": True,
        "license_status": "MIT；第三方来源核验与作者GitHub署名待补，本包尚未上传",
    })
    card = template.read_text().replace("{{VERSION}}", payload["version"])
    (destination / "README.md").write_text(card)
    shutil.copy2(Path(__file__).resolve().parents[1] / 'LICENSE', destination / 'LICENSE')
    for name in ("loss_curve.jsonl", "loss_curve.csv", "loss_curve.svg"):
        p = checkpoint.parent / name
        if p.is_file():
            shutil.copyfile(p, destination / name)
    return inventory(destination, "huggingface_model", {"version": payload["version"]})


def source_files(root: Path, manifest: dict) -> set[Path]:
    selected = set()
    for pattern in manifest["include"]:
        for p in root.glob(pattern):
            if p.is_symlink() or not p.is_file():
                continue
            relative = p.relative_to(root)
            if any(relative.match(pattern) for pattern in manifest.get("exclude", [])):
                continue
            selected.add(relative)
    # 训练入口复用历史脚本辅助函数；将实际Python脚本依赖闭包一并导出。
    queue = list(selected)
    while queue:
        relative = queue.pop()
        if relative.suffix != ".py":
            continue
        for node in ast.walk(ast.parse((root / relative).read_text())):
            modules = []
            if isinstance(node, ast.ImportFrom) and node.module:
                modules = [node.module]
            elif isinstance(node, ast.Import):
                modules = [alias.name for alias in node.names]
            for module in modules:
                name = module.removeprefix("scripts.").split(".")[0]
                dependency = Path("scripts") / f"{name}.py"
                if (root / dependency).is_file() and dependency not in selected:
                    selected.add(dependency)
                    queue.append(dependency)
    return selected


def export_source(root: Path, manifest: Path, destination: Path) -> dict:
    if destination.exists():
        raise FileExistsError(f"拒绝覆盖已有目录: {destination}")
    selected = source_files(root, json.loads(manifest.read_text()))
    if not selected:
        raise ValueError("公开文件允许列表为空")
    for relative in selected:
        if (root / relative).stat().st_size > 10 * 2**20:
            raise ValueError(f"代码包拒绝超过10MiB的文件: {relative}")
    destination.mkdir(parents=True, exist_ok=False)
    for relative in sorted(selected):
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(root / relative, target)
    return inventory(destination, "github_source", {
        "license_status": "MIT", "clean_machine_reproduction": "尚未完成",
        "privacy_review": "允许列表不包含私人日志/凭据；代码中的历史本地路径仍须人工审阅",
    })


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="kind", required=True)
    source = sub.add_parser("source")
    source.add_argument("--root", type=Path, default=Path.cwd())
    source.add_argument("--manifest", type=Path, default=Path("release/source_manifest.json"))
    source.add_argument("--destination", type=Path, required=True)
    model = sub.add_parser("model")
    model.add_argument("--checkpoint", type=Path, required=True)
    model.add_argument("--train-manifest", type=Path, required=True)
    model.add_argument("--evaluation-root", type=Path, required=True)
    model.add_argument("--template", type=Path, default=Path("release/model_card_template.md"))
    model.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args()
    if args.kind == "source":
        result = export_source(args.root, args.manifest, args.destination)
    else:
        result = export_model(args.checkpoint, args.train_manifest, args.evaluation_root,
                              args.destination, args.template)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
