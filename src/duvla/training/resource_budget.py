"""Check both WSL filesystem and its Windows backing drive before large writes."""
from __future__ import annotations

import shutil
import subprocess
import os
import platform
import re
from pathlib import Path


def _is_wsl() -> bool:
    return bool(os.environ.get('WSL_DISTRO_NAME')) or 'microsoft' in platform.release().lower()


def require_disk_budget(path: Path, *, minimum_gib: float, minimum_host_gib: float = 12) -> dict[str, object]:
    # ext4 free blocks can reuse already allocated VHDX space. Do not require
    # Windows to hold the entire guest write budget a second time. Host reserve
    # remains mandatory: reuse is not a guarantee against further VHDX growth.
    if minimum_gib <= 0 or minimum_host_gib <= 0:
        raise ValueError('disk budgets must be positive')
    required = minimum_gib * 2**30
    available = shutil.disk_usage(path).free
    if available < required:
        raise RuntimeError(f'WSL可用空间不足：{available/2**30:.1f}GiB，需要{minimum_gib:.1f}GiB')
    result_info = {'wsl_free_gib': available/2**30, 'required_wsl_gib': minimum_gib,
                   'required_host_reserve_gib': minimum_host_gib}
    if not _is_wsl():
        return {**result_info, 'host_check': 'not_applicable_native_system'}
    drive = os.environ.get('DUVLA_WSL_HOST_DRIVE', '').upper()
    if not re.fullmatch('[A-Z]', drive):
        raise RuntimeError('WSL下须设置DUVLA_WSL_HOST_DRIVE为承载VHDX的实际盘符，如D；不自动猜测')
    powershell = Path(os.environ.get('DUVLA_POWERSHELL',
        '/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe'))
    if not powershell.exists():
        raise RuntimeError('无法核验WSL宿主磁盘；检查DUVLA_POWERSHELL及Windows互操作能力')
    result = subprocess.run([str(powershell),'-NoProfile','-Command',
        f'[Console]::Write((Get-PSDrive -Name {drive}).Free)'],capture_output=True,text=True,timeout=30,check=True)
    host = int(result.stdout.strip())
    if host < minimum_host_gib * 2**30:
        raise RuntimeError(f'Windows {drive}盘仅剩{host/2**30:.1f}GiB，低于宿主安全余量{minimum_host_gib:.1f}GiB；暂停写入并检查VHDX增长')
    return {**result_info, 'windows_host_drive': drive, 'host_check': 'verified',
            f'windows_{drive.lower()}_free_gib':host/2**30, 'windows_host_free_gib':host/2**30}
