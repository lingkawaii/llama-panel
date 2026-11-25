"""跨平台只读 GPU/VRAM 采集：优先 rocm-smi，回退 nvidia-smi。

采集失败（无工具、超时、解析失败）一律返回 None，绝不影响 API 响应。
"""
from __future__ import annotations

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
                    return stats
        text = run([
            "nvidia-smi",
            "--query-gpu=name,memory.used,memory.total,utilization.gpu,"
            "temperature.gpu,power.draw",
            "--format=csv,noheader,nounits",
        ])
        if text:
            return parse_nvidia_smi_csv(text)
    except Exception:
        return None
    return None
