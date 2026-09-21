"""LIBERO-Plus子进程边界：只读来源审计、显式导入和观测白名单。

顶层仅依赖标准库；导入本模块不导入NumPy、Torch、LIBERO或渲染后端。
bootstrap仅可在新建worker使用，不负责安装、下载、渲染或加载策略。
"""

from __future__ import annotations

from collections.abc import Mapping
import hashlib
import importlib
from importlib.machinery import ModuleSpec
import json
import os
from pathlib import Path, PurePosixPath
import re
import subprocess
import sys
from types import ModuleType
from typing import Any


PINNED_COMMIT = "4976dc30028e805ff8094b55501d532c48fec182"
PINNED_ASSET_SHA256 = "96764a4bfbdaea98d4411598caeab235458318fe0f549611b93d1a323027b3cf"
PINNED_ASSET_REVISION = "dd2bd61b7d9a6fef1abc52d606e983b41886a149"
RUNTIME_ASSET_LINK = "libero/libero/assets"
OBSERVATION_KEYS = (
    "agentview_image", "robot0_eye_in_hand_image",
    "robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos",
)


def _git(source: Path, *args: str) -> bytes:
    command = [
        "git", "--no-optional-locks", "-c", "core.fsmonitor=false",
        "-C", str(source), *args,
    ]
    try:
        result = subprocess.run(command, capture_output=True, check=False, timeout=15)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise ValueError(f"无法只读核验Plus Git来源：{exc}") from exc
    if result.returncode:
        raise ValueError(f"Plus Git来源核验失败：{os.fsdecode(result.stderr).strip()}")
    return result.stdout


def _runtime_asset_link_audit(source: Path) -> dict[str, Any]:
    """唯一获准的运行时链接，引用既有完整解压报告，不重写上游源码。"""
    link = source / RUNTIME_ASSET_LINK
    target = source.parent / "assets"
    if not link.is_symlink() or not target.is_dir() or target.is_symlink():
        raise ValueError("仅允许源码assets符号链接指向同一runtime的真实assets目录")
    if link.resolve() != target:
        raise ValueError("源码assets链接目标不等于source.parent/assets")
    report = target / ".duvla_asset_extraction.json"
    if report.is_symlink() or not report.is_file() or report.stat().st_size > 64 * 1024:
        raise ValueError("assets链接缺少有效的小型完整解压报告")
    with report.open("rb") as stream:
        content = stream.read(64 * 1024 + 1)
    if len(content) > 64 * 1024:
        raise ValueError("assets解压报告超过64KiB预算")
    try:
        payload = json.loads(content)
    except (ValueError, UnicodeError) as exc:
        raise ValueError("assets解压报告不是有效JSON") from exc
    lock = payload.get("lock") if isinstance(payload, dict) else None
    if (
        not isinstance(payload, dict) or payload.get("status") != "complete"
        or not isinstance(lock, dict)
        or lock.get("archive_sha256") != PINNED_ASSET_SHA256
        or lock.get("revision") != PINNED_ASSET_REVISION
    ):
        raise ValueError("assets解压报告尚未完成或archive hash/revision不匹配")
    return {
        "relative_path": RUNTIME_ASSET_LINK, "target": str(target),
        "extraction_report": str(report),
        "extraction_report_sha256": hashlib.sha256(content).hexdigest(),
        "archive_sha256": PINNED_ASSET_SHA256, "revision": PINNED_ASSET_REVISION,
        "extraction_status": "complete", "full_asset_tree_rehashed_this_check": False,
    }


def validate_pinned_source(
    source_root: str | Path, *, expected_commit: str = PINNED_COMMIT,
) -> dict[str, Any]:
    """核验tracked源码clean；另列运行时__pycache__及唯一已审计资产链接。"""
    source = Path(source_root).resolve(strict=True)
    if not source.is_dir() or not re.fullmatch(r"[0-9a-f]{40}", expected_commit):
        raise ValueError("Plus来源必须是目录，commit必须是完整40位SHA")
    root = Path(os.fsdecode(_git(source, "rev-parse", "--show-toplevel")).strip()).resolve()
    if root != source:
        raise ValueError("Plus源码路径必须是Git仓库根，不能误用父仓库内的普通目录")
    actual_commit = os.fsdecode(_git(source, "rev-parse", "HEAD")).strip()
    if actual_commit != expected_commit:
        raise ValueError(f"Plus commit不匹配：{actual_commit}")
    ignored: list[str] = []
    asset_link: dict[str, Any] | None = None
    changes: list[str] = []
    for record in _git(source, "status", "--porcelain=v1", "-z", "--untracked-files=all").split(b"\0"):
        if not record:
            continue
        status = record[:2]
        relative = os.fsdecode(record[3:])
        components = PurePosixPath(relative).parts
        candidate = source / relative
        if status == b"??" and relative == RUNTIME_ASSET_LINK:
            asset_link = _runtime_asset_link_audit(source)
        elif (
            status == b"??" and "__pycache__" in components[:-1]
            and not candidate.is_symlink()
            and candidate.resolve().is_relative_to(source)
        ):
            ignored.append(relative)
        else:
            changes.append(os.fsdecode(record))
    if changes:
        raise ValueError(f"Plus源码存在未授权修改或未跟踪文件：{changes[:10]}")
    if not (source / "libero/libero/__init__.py").is_file():
        raise ValueError("Plus源码缺少libero/libero/__init__.py")
    return {
        "source_root": str(source), "commit": actual_commit,
        "tracked_source_clean": True, "ignored_untracked_pycache": sorted(ignored),
        "allowed_runtime_asset_link": asset_link,
        "worktree_has_allowed_untracked": bool(ignored or asset_link),
    }


def validate_runtime_config(
    config_path: str | Path, *, runtime_root: str | Path,
) -> dict[str, Any]:
    """仅核验独立配置文件边界，不把其存在当作资产路径已核验。"""
    runtime = Path(runtime_root).resolve(strict=True)
    config = Path(config_path).resolve(strict=True)
    protected = (Path.home() / ".libero").resolve()
    if (
        not runtime.is_dir() or not config.is_relative_to(runtime)
        or config.is_relative_to(protected) or config.name != "config.yaml"
        or not config.is_file()
    ):
        raise ValueError("Plus config.yaml必须已存在于独立runtime内，禁止使用全局.libero")
    if config.stat().st_size > 64 * 1024:
        raise ValueError("Plus runtime config.yaml超过64KiB预算")
    content = config.read_bytes()
    if not content.strip():
        raise ValueError("Plus runtime config.yaml不能为空")
    return {
        "config_path": str(config), "config_sha256": hashlib.sha256(content).hexdigest(),
        "config_boundary_verified": True, "asset_paths_verified": False,
    }


def validate_libero_module_origins(source_root: str | Path) -> dict[str, Any]:
    """检查当前worker所有libero.*来源，不接受旧site-packages的混合导入。"""
    boundary = Path(source_root).resolve(strict=True) / "libero"
    if "libero" not in sys.modules or "libero.libero" not in sys.modules:
        raise ValueError("Plus命名空间尚未完整导入")
    checked: dict[str, list[str]] = {}
    for name, module in tuple(sys.modules.items()):
        if name != "libero" and not name.startswith("libero."):
            continue
        if not isinstance(module, ModuleType):
            raise ValueError(f"无效或未完成的Plus模块：{name}")
        origins: list[str] = []
        module_file = getattr(module, "__file__", None)
        if module_file:
            origins.append(str(module_file))
        module_spec = getattr(module, "__spec__", None)
        if module_spec is not None and module_spec.origin:
            origins.append(str(module_spec.origin))
        origins.extend(str(path) for path in getattr(module, "__path__", ()))
        if not origins:
            raise ValueError(f"Plus模块没有可核验来源：{name}")
        if any(not Path(path).resolve().is_relative_to(boundary) for path in origins):
            raise ValueError(f"Plus模块来源越界/混入现有LIBERO：{name} {origins}")
        checked[name] = sorted(set(str(Path(path).resolve()) for path in origins))
    return {"module_origins_verified": True, "module_origins": checked}


def bootstrap_libero(
    source_root: str | Path, config_path: str | Path, *, runtime_root: str | Path,
    allow_numpy2_float_alias: bool = False, expected_commit: str = PINNED_COMMIT,
) -> dict[str, Any]:
    """只在fresh worker显式绑定Plus；失败后应终止worker，不在原进程切包。"""
    if type(allow_numpy2_float_alias) is not bool:
        raise ValueError("NumPy兼容补丁必须显式使用布尔开关")
    if any(name == "libero" or name.startswith("libero.") for name in sys.modules):
        raise ValueError("禁止在已加载LIBERO的进程中bootstrap；请启动fresh worker")
    source_audit = validate_pinned_source(source_root, expected_commit=expected_commit)
    config_audit = validate_runtime_config(config_path, runtime_root=runtime_root)
    source = Path(source_audit["source_root"])
    os.environ["LIBERO_CONFIG_PATH"] = str(Path(config_audit["config_path"]).parent)
    patches: list[dict[str, str]] = []
    if allow_numpy2_float_alias:
        numpy = importlib.import_module("numpy")
        if "float_" not in vars(numpy):
            major = str(numpy.__version__).split(".", 1)[0]
            if major != "2":
                raise ValueError("仅审计过NumPy2的float_→float64兼容")
            numpy.float_ = numpy.float64
            patches.append({
                "patch": "numpy.float_=numpy.float64", "numpy_version": str(numpy.__version__),
                "scope": "current_worker_process_only", "authorization": "explicit_opt_in",
            })
    namespace = ModuleType("libero")
    namespace.__package__ = "libero"
    namespace.__path__ = [str(source / "libero")]
    namespace.__spec__ = ModuleSpec("libero", loader=None, is_package=True)
    namespace.__spec__.submodule_search_locations = list(namespace.__path__)
    sys.modules["libero"] = namespace
    importlib.import_module("libero.libero")
    origin_audit = validate_libero_module_origins(source)
    return {
        **source_audit, **config_audit, **origin_audit, "compatibility_patches": patches,
        "bootstrap_complete": True, "renderer_verified": False, "policy_verified": False,
    }


def make_policy_request(
    observation: Mapping[str, object], language: str, *, image_size: int | None = None,
) -> dict[str, Any]:
    """拷贝白名单原始观测，不做翻转/归一化，避免与策略桥重复转换。

返回{language, observation}；reward、success、task id及物体真值一律不转发。
相机翻转、四元数转轴角、state标准化留在单一策略adapter。
"""
    numpy = importlib.import_module("numpy")
    if not isinstance(observation, Mapping):
        raise ValueError("观测必须为mapping")
    if not isinstance(language, str) or not language.strip():
        raise ValueError("策略语言必须为非空字符串")
    if image_size is not None and (type(image_size) is not int or image_size <= 0):
        raise ValueError("image_size必须为正整数或None")
    missing = [key for key in OBSERVATION_KEYS if key not in observation]
    if missing:
        raise ValueError(f"观测缺少白名单字段：{missing}")
    selected: dict[str, Any] = {}
    camera_shape: tuple[int, ...] | None = None
    for key in OBSERVATION_KEYS[:2]:
        value = numpy.asarray(observation[key])
        if value.dtype != numpy.uint8 or value.ndim != 3 or value.shape[-1] != 3 or min(value.shape[:2]) <= 0:
            raise ValueError(f"{key}必须为非空uint8 RGB HWC")
        if image_size is not None and value.shape != (image_size, image_size, 3):
            raise ValueError(f"{key}分辨率与已锁定图像协议不一致")
        if camera_shape is not None and value.shape != camera_shape:
            raise ValueError("两路相机分辨率必须一致")
        camera_shape = value.shape
        selected[key] = value.copy()
    for key, size in zip(OBSERVATION_KEYS[2:], (3, 4, 2), strict=True):
        value = numpy.asarray(observation[key])
        if value.shape != (size,) or value.dtype.kind != "f" or not numpy.isfinite(value).all():
            raise ValueError(f"{key}必须为有限浮点数组，shape=({size},)")
        with numpy.errstate(over="ignore", invalid="ignore"):
            copied = value.astype(numpy.float32, copy=True)
        if not numpy.isfinite(copied).all():
            raise ValueError(f"{key}超出float32表示范围")
        if key == "robot0_eef_quat" and float(numpy.linalg.norm(copied.astype(numpy.float64))) <= 1e-12:
            raise ValueError("机器人四元数模长必须为正")
        selected[key] = copied
    return {"language": language, "observation": selected}


def zero_action() -> Any:
    """LIBERO7D无位移指令，夹爪保持打开；不声称物理上绝对无效果。"""
    numpy = importlib.import_module("numpy")
    return numpy.asarray([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0], dtype=numpy.float32)
