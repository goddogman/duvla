"""LIBERO-Plus 元数据和固定清单契约；仅标准库，不导入模拟器或执行评测。

上游格式来源：sylvestf/LIBERO-plus 的 task_classification.json 和
libero_suite_task_map.py。上游分类 id 从 1 开始，必须与任务表交叉验证；
language 不在分类文件中，不能从含扰动后缀的名称猜测。元数据许可单独记录，
不把 Hugging Face 资产页的 MIT 标记自动应用于 GitHub 代码和分类文件。
"""

from __future__ import annotations

import ast
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import PurePosixPath
import re
from typing import Literal


SOURCE_REPO = "https://github.com/sylvestf/LIBERO-plus"
METADATA_RELPATH = "libero/libero/benchmark/task_classification.json"
TASK_MAP_RELPATH = "libero/libero/benchmark/libero_suite_task_map.py"
SUITES = ("libero_10", "libero_spatial", "libero_object", "libero_goal")
CATEGORIES = (
    "Camera Viewpoints", "Robot Initial States", "Language Instructions",
    "Light Conditions", "Background Textures", "Sensor Noise", "Objects Layout",
)
FULL_VARIANT_COUNT = 10_030
SUBSET_VARIANT_COUNT = 700
SCHEMA_VERSION = 1
Scope = Literal["development_subset", "full_benchmark"]
_NAME = re.compile(r"[A-Za-z0-9_-]+\Z")


def _integer(value: object, name: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} 必须是 >= {minimum} 的整数")
    return value


def _digest(value: object, name: str, length: int) -> str:
    if not isinstance(value, str) or re.fullmatch(rf"[0-9a-f]{{{length}}}", value) is None:
        raise ValueError(f"{name} 必须是 {length} 位小写十六进制完整标识")
    return value


@dataclass(frozen=True)
class LiberoPlusVariant:
    suite: str
    classification_id: int
    benchmark_task_index: int
    name: str
    category: str
    difficulty_level: int | None
    language: None = None

    def __post_init__(self) -> None:
        if self.suite not in SUITES or self.category not in CATEGORIES:
            raise ValueError("未知的官方 suite/category")
        _integer(self.classification_id, "classification_id", 1)
        _integer(self.benchmark_task_index, "benchmark_task_index")
        if self.benchmark_task_index != self.classification_id - 1:
            raise ValueError("classification_id 与零起点 benchmark_task_index 不一致")
        if not isinstance(self.name, str) or _NAME.fullmatch(self.name) is None:
            raise ValueError("name 必须是安全的单一文件名 stem，禁止路径穿越")
        if self.difficulty_level is not None:
            if _integer(self.difficulty_level, "difficulty_level", 1) > 5:
                raise ValueError("difficulty_level 只允许 1..5 或 null")
        if self.language is not None:
            raise ValueError("分类元数据不包含语言，必须留空并由运行时官方 BDDL loader 解析")

    @property
    def variant_key(self) -> str:
        return f"{self.suite}/{self.name}"

    @property
    def bddl_relpath(self) -> str:
        # 某些 _view_ 名称是官方 wrapper 的逻辑入口，并非独立实体文件。
        return str(PurePosixPath(self.suite, f"{self.name}.bddl"))


def parse_task_map_source(source: str) -> dict[str, tuple[str, ...]]:
    """仅解析上游常量字典，不执行下载的 Python 源码。"""
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        raise ValueError("task_map schema：上游 Python 文件语法无效") from exc
    assignments = [
        node for node in tree.body
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "libero_task_map"
                for target in node.targets)
    ]
    if len(assignments) != 1:
        raise ValueError("task_map schema：必须且只能有一个 libero_task_map 常量赋值")
    try:
        value = ast.literal_eval(assignments[0].value)
    except (ValueError, TypeError, SyntaxError) as exc:
        raise ValueError("task_map 必须是字面量，拒绝执行动态表达式") from exc
    if not isinstance(value, dict) or not set(SUITES).issubset(value):
        raise ValueError("task_map 缺少四个官方 suite")
    result = {}
    for suite in SUITES:
        names = value[suite]
        if not isinstance(names, list) or not names:
            raise ValueError("task_map suite 必须是非空名称列表")
        if any(not isinstance(name, str) or _NAME.fullmatch(name) is None for name in names):
            raise ValueError("task_map 包含不安全的任务名称")
        if len(set(names)) != len(names):
            raise ValueError("task_map 包含 duplicate 名称")
        result[suite] = tuple(names)
    return result


def parse_official_metadata(
    payload: object, *, task_names_by_suite: Mapping[str, Sequence[str]],
) -> tuple[LiberoPlusVariant, ...]:
    """核对真实四字段 schema、连续 id、名称、类别及难度；不猜测缺失字段。"""
    if not isinstance(payload, dict) or set(payload) != set(SUITES):
        raise ValueError("metadata schema 必须恰好包含四个官方 suite")
    records = []
    for suite in SUITES:
        rows = payload[suite]
        if not isinstance(rows, list) or not rows:
            raise ValueError("metadata suite 必须是非空列表")
        names = task_names_by_suite.get(suite)
        if (not isinstance(names, (list, tuple)) or len(names) != len(rows)
                or any(not isinstance(name, str) for name in names)):
            raise ValueError("metadata 与官方 task_map 数量/类型不一致")
        if len(set(names)) != len(names):
            raise ValueError("task_map 包含 duplicate 名称")
        ids: set[int] = set()
        row_names: set[str] = set()
        for row in rows:
            if not isinstance(row, dict) or set(row) != {"id", "name", "category", "difficulty_level"}:
                raise ValueError("metadata schema 字段与已核对的官方四字段格式不一致")
            identifier = _integer(row["id"], "id", 1)
            record = LiberoPlusVariant(
                suite=suite, classification_id=identifier,
                benchmark_task_index=identifier - 1, name=row["name"],
                category=row["category"], difficulty_level=row["difficulty_level"],
            )
            if identifier in ids or record.name in row_names:
                raise ValueError("metadata 包含 duplicate id/name")
            if identifier > len(names) or names[identifier - 1] != record.name:
                raise ValueError("metadata id/name 与官方 task_map 位置不一致")
            records.append(record)
            ids.add(identifier)
            row_names.add(record.name)
        if ids != set(range(1, len(rows) + 1)):
            raise ValueError("metadata id 必须连续覆盖完整官方任务表")
    if {record.category for record in records} != set(CATEGORIES):
        raise ValueError("metadata 未覆盖七类扰动")
    return tuple(sorted(records, key=lambda record: record.variant_key))


def _hash(namespace: str, seed: int, key: str) -> str:
    _integer(seed, "seed")
    return hashlib.sha256(f"libero_plus:{namespace}:{seed}:{key}".encode()).hexdigest()


def variant_rollout_seed(variant_key: str, seed: int = 17) -> int:
    """与变体身份绑定；与清单顺序、已完成数和进程分配无关。"""
    return int(_hash("rollout", seed, variant_key)[:16], 16) % (2**31 - 1)


def stratified_subset(
    records: Sequence[LiberoPlusVariant], *, selection_seed: int = 17,
) -> tuple[LiberoPlusVariant, ...]:
    """固定 7×100；每类别四 suite 各 25，再按可用难度含 unknown 水位均衡。"""
    _integer(selection_seed, "selection_seed")
    if len({record.variant_key for record in records}) != len(records):
        raise ValueError("records 包含 duplicate variant")
    result = []
    for category in CATEGORIES:
        for suite in SUITES:
            groups: dict[int | None, list[LiberoPlusVariant]] = defaultdict(list)
            for record in records:
                if record.category == category and record.suite == suite:
                    groups[record.difficulty_level].append(record)
            if sum(map(len, groups.values())) < 25:
                raise ValueError(f"分层配额不足：{category}/{suite} 少于 25 条")
            for group in groups.values():
                group.sort(key=lambda record: _hash("selection", selection_seed, record.variant_key))
            levels = sorted(groups, key=lambda level: _hash(
                "stratum", selection_seed, f"{category}/{suite}/{level}",
            ))
            selected: Counter[int | None] = Counter()
            for _ in range(25):
                available = [level for level in levels if selected[level] < len(groups[level])]
                level = min(available, key=lambda item: (selected[item], levels.index(item)))
                result.append(groups[level][selected[level]])
                selected[level] += 1
    return tuple(sorted(result, key=lambda record: (SUITES.index(record.suite), record.variant_key)))


def canonical_manifest_sha256(payload: Mapping[str, object]) -> str:
    """自身 digest 字段不参与哈希；不加入生成时间，保证重复准备字节稳定。"""
    document = {key: value for key, value in payload.items() if key != "manifest_sha256"}
    return hashlib.sha256(json.dumps(
        document, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
    ).encode()).hexdigest()


def build_manifest(
    records: Sequence[LiberoPlusVariant], *, scope: Scope, source_commit: str,
    metadata_sha256: str, task_map_sha256: str, selection_seed: int = 17,
    rollout_seed: int = 17,
    source_license: str = "未核实：官方代码仓库未见许可证文件；不得套用资产许可证",
) -> dict[str, object]:
    if scope not in ("development_subset", "full_benchmark"):
        raise ValueError("未知 scope")
    _digest(source_commit, "source_commit", 40)
    _digest(metadata_sha256, "metadata_sha256", 64)
    _digest(task_map_sha256, "task_map_sha256", 64)
    _integer(selection_seed, "selection_seed")
    _integer(rollout_seed, "rollout_seed")
    if not isinstance(source_license, str) or not source_license.strip():
        raise ValueError("source_license 必须明确说明许可状态")
    if len(records) != FULL_VARIANT_COUNT:
        raise ValueError("必须从完整的 10030 条官方分类生成清单，拒绝截断来源")
    if len({record.variant_key for record in records}) != len(records):
        raise ValueError("records 包含 duplicate variant")
    if scope == "development_subset":
        selected = stratified_subset(records, selection_seed=selection_seed)
    else:
        selected = tuple(sorted(records, key=lambda record: (SUITES.index(record.suite), record.variant_key)))
    entries = [
        {
            **asdict(record), "variant_key": record.variant_key,
            "bddl_relpath": record.bddl_relpath,
            "rollout_seed": variant_rollout_seed(record.variant_key, rollout_seed),
        }
        for record in selected
    ]
    payload: dict[str, object] = {
        "schema_version": SCHEMA_VERSION, "benchmark": "libero_plus", "scope": scope,
        "official_full_benchmark": scope == "full_benchmark",
        "comparison_note": (
            "完整覆盖固定官方来源中的 10,030 个变体，每变体一次"
            if scope == "full_benchmark"
            else "700 为预注册 development 子集，不是官方完整 LIBERO-Plus 榜单结果"
        ),
        "trials_per_variant": 1, "source_variant_count": len(records),
        "variant_count": len(selected), "selection_seed": selection_seed,
        "rollout_seed": rollout_seed,
        "selection_policy": "category_100_suite_25_available_level_waterfill_v1"
        if scope == "development_subset" else "all_official_variants_v1",
        "source": {
            "repository": SOURCE_REPO, "commit": source_commit,
            "metadata_relpath": METADATA_RELPATH, "metadata_sha256": metadata_sha256,
            "task_map_relpath": TASK_MAP_RELPATH, "task_map_sha256": task_map_sha256,
            "license_status": source_license,
        },
        "runtime_contract": {
            "readiness": "metadata_only_not_evaluation_ready",
            "language_resolution": "runtime_official_bddl_loader_required",
            "logical_bddl_entrypoint": True,
            "task_order_index": 0,
            "init_state_selection": "trial_index_0_official_loader_not_random_state_resampling",
            "task_metadata_for_environment_only": True,
        },
        "category_counts": dict(sorted(Counter(record.category for record in selected).items())),
        "suite_counts": dict(sorted(Counter(record.suite for record in selected).items())),
        "entries": entries,
    }
    payload["manifest_sha256"] = canonical_manifest_sha256(payload)
    validate_manifest(payload)
    return payload


def validate_manifest(payload: object) -> dict[str, object]:
    """加载时拒绝范围伪装、重复项、改写种子及未知 schema。不启动评测。"""
    if not isinstance(payload, dict) or type(payload.get("schema_version")) is not int:
        raise ValueError("manifest schema 无效")
    if payload.get("schema_version") != SCHEMA_VERSION or payload.get("benchmark") != "libero_plus":
        raise ValueError("manifest schema_version/benchmark 不匹配")
    scope = payload.get("scope")
    if scope not in ("development_subset", "full_benchmark"):
        raise ValueError("manifest scope 无效")
    full = scope == "full_benchmark"
    if payload.get("official_full_benchmark") is not full:
        raise ValueError("禁止把 development 子集伪装成 official 完整榜单")
    if type(payload.get("trials_per_variant")) is not int or payload["trials_per_variant"] != 1:
        raise ValueError("trials_per_variant 必须为 1")
    if _integer(payload.get("source_variant_count"), "source_variant_count") != FULL_VARIANT_COUNT:
        raise ValueError("source_variant_count 必须为 10030")
    expected = FULL_VARIANT_COUNT if full else SUBSET_VARIANT_COUNT
    entries = payload.get("entries")
    if (not isinstance(entries, list) or len(entries) != expected
            or _integer(payload.get("variant_count"), "variant_count") != expected):
        raise ValueError("scope 对应的清单数量必须为 700/10030")
    source = payload.get("source")
    if not isinstance(source, dict) or source.get("repository") != SOURCE_REPO:
        raise ValueError("manifest 缺少官方来源")
    for field, length in (("commit", 40), ("metadata_sha256", 64), ("task_map_sha256", 64)):
        _digest(source.get(field), field, length)
    if source.get("metadata_relpath") != METADATA_RELPATH or source.get("task_map_relpath") != TASK_MAP_RELPATH:
        raise ValueError("manifest 上游元数据路径不匹配")
    if not isinstance(source.get("license_status"), str) or not source["license_status"].strip():
        raise ValueError("manifest 缺少许可状态")
    _integer(payload.get("selection_seed"), "selection_seed")
    rollout_seed = _integer(payload.get("rollout_seed"), "rollout_seed")
    expected_policy = "all_official_variants_v1" if full else "category_100_suite_25_available_level_waterfill_v1"
    if payload.get("selection_policy") != expected_policy:
        raise ValueError("selection_policy 与 scope 不匹配")
    contract = payload.get("runtime_contract")
    if not isinstance(contract, dict) or contract != {
        "readiness": "metadata_only_not_evaluation_ready",
        "language_resolution": "runtime_official_bddl_loader_required",
        "logical_bddl_entrypoint": True, "task_order_index": 0,
        "init_state_selection": "trial_index_0_official_loader_not_random_state_resampling",
        "task_metadata_for_environment_only": True,
    }:
        raise ValueError("runtime_contract 被修改或错误声称评测已就绪")
    records = []
    indices: set[tuple[str, int]] = set()
    keys: set[str] = set()
    expected_fields = set(LiberoPlusVariant.__dataclass_fields__) | {"variant_key", "bddl_relpath", "rollout_seed"}
    for entry in entries:
        if not isinstance(entry, dict) or set(entry) != expected_fields:
            raise ValueError("manifest entry schema 不匹配")
        record = LiberoPlusVariant(**{field: entry[field] for field in LiberoPlusVariant.__dataclass_fields__})
        if entry["variant_key"] != record.variant_key or entry["bddl_relpath"] != record.bddl_relpath:
            raise ValueError("variant_key/bddl_relpath 不匹配或存在路径穿越")
        if (record.suite, record.benchmark_task_index) in indices or record.variant_key in keys:
            raise ValueError("manifest 包含 duplicate variant/index")
        if type(entry["rollout_seed"]) is not int or entry["rollout_seed"] != variant_rollout_seed(record.variant_key, rollout_seed):
            raise ValueError("rollout_seed 必须按稳定 variant_key 生成")
        records.append(record)
        keys.add(record.variant_key)
        indices.add((record.suite, record.benchmark_task_index))
    categories = Counter(record.category for record in records)
    suites = Counter(record.suite for record in records)
    if dict(categories) != payload.get("category_counts") or dict(suites) != payload.get("suite_counts"):
        raise ValueError("分类数量汇总与 entries 不一致")
    if set(categories) != set(CATEGORIES) or set(suites) != set(SUITES):
        raise ValueError("清单未覆盖全部七类/四套")
    if not full:
        cells = Counter((record.category, record.suite) for record in records)
        if set(categories.values()) != {100} or set(cells.values()) != {25} or len(cells) != 28:
            raise ValueError("development 分层配额必须每类别 100、每类别/suite 25")
    else:
        for suite in SUITES:
            if {record.benchmark_task_index for record in records if record.suite == suite} != set(range(suites[suite])):
                raise ValueError("完整清单任务索引必须连续覆盖")
    _digest(payload.get("manifest_sha256"), "manifest_sha256", 64)
    if payload["manifest_sha256"] != canonical_manifest_sha256(payload):
        raise ValueError("manifest_sha256 不匹配，固定清单被修改")
    return payload
