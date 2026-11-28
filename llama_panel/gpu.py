"""跨平台只读 GPU/VRAM 采集：优先 rocm-smi，回退 nvidia-smi。

采集失败（无工具、超时、解析失败）一律返回 None，绝不影响 API 响应。
"""
from __future__ import annotations

import glob
import json
import os
import shutil
import subprocess

_TIMEOUT = 2.0  # 秒


def _num(value) -> float | None:
    """把 rocm-smi 的字符串数值（可能带 %、空格）转成 float。"""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    s = str(value).strip().replace("%", "").replace(",", "")
    if not s:
        return None
    try:
        return float(s)
    except ValueError:
        return None


def _first(d: dict, *keys: str):
    for k in keys:
        if k in d:
            return d[k]
    return None


def _build(
    name: str,
    vram_used: int,
    vram_total: int,
    vram_percent: float | None,
    gpu_util: float | None,
    temp_c: float | None,
    power_w: float | None,
) -> dict:
    if vram_percent is None and vram_total > 0:
        vram_percent = vram_used / vram_total * 100
    return {
        "name": name,
        "vram_used_bytes": int(vram_used),
        "vram_total_bytes": int(vram_total),
        "vram_percent": vram_percent,
        "gpu_util_percent": gpu_util,
        "temp_c": temp_c,
        "power_w": power_w,
    }


def parse_rocm_smi_json(text: str) -> dict | None:
    """解析 rocm-smi --json 输出（单卡 dict 或多卡 {cardN: {...}} 结构）。"""
    try:
        data = json.loads(text)
    except (ValueError, TypeError):
        return None
    if not isinstance(data, dict):
        return None
    # 单卡：字段直接在顶层；多卡：按 cardN / 数字键嵌套
    if "Card series" in data or "GPU use (%)" in data:
        cards = [data]
    else:
        cards = [
            v
            for v in data.values()
            if isinstance(v, dict) and ("Card series" in v or "GPU use (%)" in v)
        ]
        if not cards:
            return None
    c = cards[0]
    total = _num(_first(c, "VRAM Total Memory (B)"))
    used = _num(_first(c, "VRAM Total Used Memory (B)"))
    if total is None or used is None:
        return None
    return _build(
        name=str(_first(c, "Card series") or "Unknown GPU"),
        vram_used=used,
        vram_total=total,
        vram_percent=_num(_first(c, "GPU memory use (%)")),
        gpu_util=_num(_first(c, "GPU use (%)")),
        temp_c=_num(_first(c, "Temperature (C)", "Temperature (Sensor edge) (C)")),
        power_w=_num(
            _first(c, "Graphics Package Power (W)", "Average Graphics Package Power (W)")
        ),
    )


def parse_nvidia_smi_csv(text: str) -> dict | None:
    """解析 nvidia-smi --query-gpu=... --format=csv,noheader,nounits 输出（取第一行）。"""
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if not lines:
        return None
    parts = [p.strip() for p in lines[0].split(",")]
    if len(parts) < 6:
        return None
    name, mem_used, mem_total, util, temp, power = parts[:6]
    used = _num(mem_used)
    total = _num(mem_total)
    if used is None or total is None or total <= 0:
        return None
    return _build(
        name=name or "Unknown GPU",
        vram_used=used * 1024 * 1024,
        vram_total=total * 1024 * 1024,
        vram_percent=None,  # 由 used/total 计算
        gpu_util=_num(util),
        temp_c=_num(temp),
        power_w=_num(power),
    )


_MEMINFO_KEYS = ("MemTotal", "MemAvailable", "MemFree", "SwapTotal", "SwapFree")


def parse_proc_meminfo(text: str) -> dict | None:
    """解析 /proc/meminfo 文本（数值单位 kB，转字节）。

    MemAvailable 缺失时回退 MemFree；MemTotal <= 0 或解析失败返回 None。
    """
    if not text:
        return None
    kv: dict[str, int] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or ":" not in line:
            continue
        key, _, rest = line.partition(":")
        key = key.strip()
        if key not in _MEMINFO_KEYS:
            continue
        parts = rest.split()
        if not parts:
            continue
        try:
            kv[key] = int(parts[0])
        except ValueError:
            continue
    total = kv.get("MemTotal")
    if total is None or total <= 0:
        return None
    avail = kv.get("MemAvailable")
    if avail is None:
        avail = kv.get("MemFree", 0)
    swap_total = kv.get("SwapTotal", 0)
    swap_free = kv.get("SwapFree", 0)
    used = max(0, total - avail)
    return {
        "used_bytes": used * 1024,
        "total_bytes": total * 1024,
        "percent": round(used * 100.0 / total, 1),
        "swap_used_bytes": max(0, swap_total - swap_free) * 1024,
        "swap_total_bytes": swap_total * 1024,
    }


def get_host_memory(meminfo_path: str = "/proc/meminfo") -> dict | None:
    """读取 /proc/meminfo 返回整机内存用量；文件不存在或异常（如非 Linux）返回 None。"""
    try:
        with open(meminfo_path) as f:
            return parse_proc_meminfo(f.read())
    except OSError:
        return None


def _find_rocm_smi() -> str | None:
    p = shutil.which("rocm-smi")
    if p:
        return p
    home = os.path.expanduser("~/bin/rocm-smi")
    if os.path.isfile(home):
        return home
    return None


def _run(cmd: list[str]) -> str | None:
    try:
        out = subprocess.run(cmd, capture_output=True, timeout=_TIMEOUT)
    except (OSError, subprocess.SubprocessError):
        return None
    if out.returncode != 0:
        return None
    return out.stdout.decode("utf-8", "replace")


def _read_watts(path: str) -> int | None:
    """读微瓦值文件并换算为整数瓦特；失败返回 None。"""
    try:
        with open(path) as f:
            return int(int(f.read().strip()) / 1_000_000)
    except (OSError, ValueError):
        return None


def _find_hwmon_dir(hwmon_dir: str | None = None) -> str | None:
    """定位含 power1_cap/min/max 的 hwmon 目录。"""
    if hwmon_dir is not None:
        if all(os.path.isfile(os.path.join(hwmon_dir, k))
               for k in ("power1_cap", "power1_cap_min", "power1_cap_max")):
            return hwmon_dir
        return None
    for pattern in ("/sys/class/drm/card*/device/hwmon/hwmon*",):
        for d in sorted(glob.glob(pattern)):
            if not os.path.isdir(d):
                continue
            if all(os.path.isfile(os.path.join(d, k))
                   for k in ("power1_cap", "power1_cap_min", "power1_cap_max")):
                return d
    return None


def get_gpu_power_cap(hwmon_dir: str | None = None) -> dict | None:
    """读取 GPU 硬件功耗墙（power1_cap）当前值与可写范围。

    无可用 hwmon 时返回 None。
    """
    d = _find_hwmon_dir(hwmon_dir)
    if d is None:
        return None
    cur = _read_watts(os.path.join(d, "power1_cap"))
    mn = _read_watts(os.path.join(d, "power1_cap_min"))
    mx = _read_watts(os.path.join(d, "power1_cap_max"))
    if cur is None or mn is None or mx is None:
        return None
    cap_path = os.path.join(d, "power1_cap")
    return {
        "current_w": cur,
        "min_w": mn,
        "max_w": mx,
        "default_w": _read_watts(os.path.join(d, "power1_cap_default")),
        "writable": os.access(cap_path, os.W_OK),
    }


def set_gpu_power_cap(watts: int, hwmon_dir: str | None = None) -> dict:
    """设置 GPU 硬件功耗墙（瓦特）。越界/非法/无权限返回 ok False。"""
    if not isinstance(watts, int) or isinstance(watts, bool):
        return {"ok": False, "error": "watts 须为整数"}
    cap = get_gpu_power_cap(hwmon_dir)
    if cap is None:
        return {"ok": False, "error": "未找到可用的 GPU 功耗墙（power1_cap）"}
    if not (cap["min_w"] <= watts <= cap["max_w"]):
        return {"ok": False, "error":
                f"watts 越界，允许范围 {cap['min_w']}..{cap['max_w']}"}
    d = _find_hwmon_dir(hwmon_dir)
    cap_path = os.path.join(d, "power1_cap")
    if not os.access(cap_path, os.W_OK):
        return {"ok": False, "error": "无权限写入 power1_cap"}
    try:
        with open(cap_path, "w") as f:
            f.write(str(watts * 1_000_000))
    except OSError as e:
        return {"ok": False, "error": f"写入 power1_cap 失败: {e}"}
    return {"ok": True, "watts": watts}


def get_gpu_stats(runner=None) -> dict | None:
    """采集第一张 GPU 的只读状态；不可用时返回 None。

    runner: 可选的 (cmd) -> str|None 注入点（默认 subprocess，超时 2s），便于测试。
    """
    run = runner or _run
    try:
        rocm = _find_rocm_smi()
        if rocm:
            text = run([
                rocm, "--showproductname", "--showuse", "--showtemp",
                "--showpower", "--showmeminfo", "vram", "--json",
            ])
            if text:
                stats = parse_rocm_smi_json(text)
                if stats:
                    stats["power_cap"] = get_gpu_power_cap()
                    return stats
        text = run([
            "nvidia-smi",
            "--query-gpu=name,memory.used,memory.total,utilization.gpu,"
            "temperature.gpu,power.draw",
            "--format=csv,noheader,nounits",
        ])
        if text:
            stats = parse_nvidia_smi_csv(text)
            if stats:
                stats["power_cap"] = get_gpu_power_cap()
                return stats
    except Exception:
        return None
    return None
