"""整机省电：powerprofilesctl 电源档位 + S3 整机休眠。

纯标准库、2 秒超时、非 Linux 或无工具时优雅降级（返回 None），绝不阻塞接口。
"""
from __future__ import annotations

import platform
import re
import shutil
import subprocess
import threading
import time

# 严格白名单：仅允许这三个档位
PROFILES = {"power-saver", "balanced", "performance"}

_TIMEOUT = 2.0  # 秒（只读采集）
_SET_TIMEOUT = 5.0  # 秒（set 子命令）
_SUSPEND_TIMEOUT = 10.0  # 秒（systemctl suspend）

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
    sys_state_path: str = "/sys/power/state",
    is_linux: bool | None = None,
    ctl: str | None = None,
) -> dict | None:
    """采集电源状态；非 Linux 或（无 powerprofilesctl 且无 /sys/power/state）返回 None。

    返回 ``{"current": str | None, "available": list[str], "can_suspend": bool}``。
    ``can_suspend`` 仅在 Linux 且 ``/sys/power/state`` 含 ``mem`` 时为 True。

    注入点（便于测试）：
    - ``runner``: ``(cmd) -> (rc, stdout, stderr)``（默认 subprocess，2s 超时）
    - ``sys_state_path``: ``/sys/power/state`` 路径
    - ``is_linux``: 覆盖平台检测
    - ``ctl``: 覆盖 ``shutil.which("powerprofilesctl")`` 结果
    """
    if is_linux is None:
        is_linux = platform.system() == "Linux"
    if not is_linux:
        return None

    run = runner or _run_tuple

    can_suspend = False
    sys_state_exists = False
    try:
        with open(sys_state_path, "r", encoding="utf-8") as f:
            sys_state_exists = True
            can_suspend = "mem" in f.read().split()
    except (OSError, ValueError):
        sys_state_exists = False

    current: str | None = None
    available: list[str] = []
    if ctl is None:
        ctl = shutil.which("powerprofilesctl")
    if ctl:
        rc, out, _err = run(["powerprofilesctl", "list"])
        if rc == 0:
            current, available = parse_power_profiles(out)

    if not ctl and not sys_state_exists:
        return None
    return {"current": current, "available": available, "can_suspend": can_suspend}


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


def trigger_suspend(delay: float = 0.6) -> None:
    """后台守护线程延迟 ``delay`` 秒后执行 ``systemctl suspend``。

    延迟确保 HTTP 200 先返回给浏览器，再进入 S3 休眠。
    """

    def _do() -> None:
        if delay > 0:
            time.sleep(delay)
        try:
            subprocess.run(
                ["systemctl", "suspend"], capture_output=True, timeout=_SUSPEND_TIMEOUT
            )
        except (OSError, subprocess.SubprocessError):
            pass

    threading.Thread(target=_do, daemon=True).start()
