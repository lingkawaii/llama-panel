"""整机电源档位：powerprofilesctl 只读采集与切换。

纯标准库、2 秒超时、非 Linux 或无工具时优雅降级（返回 None），绝不阻塞接口。
"""
from __future__ import annotations

import platform
import re
import shutil
import subprocess

# 严格白名单：仅允许这三个档位
PROFILES = {"power-saver", "balanced", "performance"}

_TIMEOUT = 2.0  # 秒（只读采集）
_SET_TIMEOUT = 5.0  # 秒（set 子命令）

# powerprofilesctl list 的块格式：每个 profile 独占一个块，激活项行首带 `*`
_PROFILE_HEADER = re.compile(r"^\*?\s*([A-Za-z][A-Za-z0-9_-]*):\s*$")
# 兼容简单列表格式：`- name` / `- * name`
_BULLET = re.compile(r"^\s*-\s+\*?\s*([A-Za-z][A-Za-z0-9_-]*)\s*\*?\s*$")


def parse_power_profiles(list_text: str) -> tuple[str | None, list[str]]:
    """解析 ``powerprofilesctl list`` 输出，返回 ``(current, available)``。

    - 识别带 ``*`` 的当前档位与所有可用档位；
    - 兼容 daemon 块格式（``* performance:`` 行首星号）与简单列表（``- name``）；
    - 非法文本（无识别到的档位）返回 ``(None, [])``。
    """
    available: list[str] = []
    current: str | None = None

    def _add(name: str, star: bool) -> None:
        nonlocal current
        if name not in available:
            available.append(name)
        if star:
            current = name

    for raw in list_text.splitlines():
        line = raw.rstrip()
        m = _PROFILE_HEADER.match(line)
        if m:
            _add(m.group(1), line.lstrip().startswith("*"))
            continue
        m = _BULLET.match(line)
        if m:
            _add(m.group(1), "*" in line)
    return current, available


def _run_tuple(cmd: list[str]) -> tuple[int, str, str]:
    """执行命令，返回 (returncode, stdout, stderr)；失败返回 (-1, "", str(e))。"""
    try:
        out = subprocess.run(cmd, capture_output=True, timeout=_TIMEOUT)
    except (OSError, subprocess.SubprocessError) as e:
        return -1, "", str(e)
    return (
        out.returncode,
        (out.stdout or b"").decode("utf-8", "replace"),
        (out.stderr or b"").decode("utf-8", "replace"),
    )


def get_power_status(
    runner=None,
    is_linux: bool | None = None,
    ctl: str | None = None,
) -> dict | None:
    """采集电源状态；非 Linux 或无 powerprofilesctl 返回 None。

    返回 ``{"current": str | None, "available": list[str]}``。

    注入点（便于测试）：
    - ``runner``: ``(cmd) -> (rc, stdout, stderr)``（默认 subprocess，2s 超时）
    - ``is_linux``: 覆盖平台检测
    - ``ctl``: 覆盖 ``shutil.which("powerprofilesctl")`` 结果
    """
    if is_linux is None:
        is_linux = platform.system() == "Linux"
    if not is_linux:
        return None
    if ctl is None:
        ctl = shutil.which("powerprofilesctl")
    if not ctl:
        return None

    run = runner or _run_tuple
    rc, out, _err = run(["powerprofilesctl", "list"])
    current: str | None = None
    available: list[str] = []
    if rc == 0:
        current, available = parse_power_profiles(out)
    return {"current": current, "available": available}


def set_power_profile(profile: str) -> dict:
    """切换电源档位（严格白名单校验）。

    成功: ``{"ok": True, "profile": profile}``
    失败: ``{"ok": False, "error": "..."}``
    """
    if profile not in PROFILES:
        return {"ok": False, "error": f"未知电源档位: {profile!r}"}
    try:
        out = subprocess.run(
            ["powerprofilesctl", "set", profile],
            capture_output=True,
            timeout=_SET_TIMEOUT,
        )
    except (OSError, subprocess.SubprocessError) as e:
        return {"ok": False, "error": f"powerprofilesctl 不可用: {e}"}
    if out.returncode != 0:
        err = (out.stderr or out.stdout or b"").decode("utf-8", "replace").strip()
        return {"ok": False, "error": err or f"powerprofilesctl set 返回 {out.returncode}"}
    return {"ok": True, "profile": profile}
