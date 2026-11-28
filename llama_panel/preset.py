"""models.ini 预设文件读写：在线修改空闲自动卸载时长与单模型参数（纯函数）。"""
from __future__ import annotations

import math
import re

_STAR_SECTION = re.compile(r"^\[\*\][ \t]*$", re.MULTILINE)
_SLEEP_LINE = re.compile(r"^sleep-idle-seconds[ \t]*=.*$", re.MULTILINE)
_VERSION_LINE = re.compile(r"^version[ \t]*=.*$", re.MULTILINE)
_NEXT_SECTION = re.compile(r"^\[", re.MULTILINE)


def update_sleep_idle_in_ini(text: str, seconds: int) -> str:
    """在 ``[*]`` 节下更新或插入 ``sleep-idle-seconds = <val>``。

    - ``seconds <= 0`` 写入 ``-1``（关闭自动休眠）；
    - 已有 ``[*]`` 节：节内已有该行则原地替换，否则追加到节末尾；
    - 无 ``[*]`` 节：在 ``version = 1`` 行后（无则在文件开头）插入
      ``[*]`` 节；完整保留其余节与注释不变。
    """
    val = -1 if seconds <= 0 else int(seconds)
    line = f"sleep-idle-seconds = {val}"

    m = _STAR_SECTION.search(text)
    if m:
        # 定位 [*] 节范围：到下一个 [section] 行或文件末尾
        rest = text[m.end():]
        nxt = _NEXT_SECTION.search(rest)
        section_end = m.end() + nxt.start() if nxt else len(text)
        section = text[m.end():section_end]
        if _SLEEP_LINE.search(section):
            new_section = _SLEEP_LINE.sub(line, section, count=1)
        else:
            # 追加到节末尾（保证换行分隔）
            prefix = "" if section.endswith("\n") or not section else "\n"
            new_section = section + prefix + line + "\n"
        return text[: m.end()] + new_section + text[section_end:]

    vm = _VERSION_LINE.search(text)
    if vm:
        pos = vm.end()
        # version 行后补空行再插入 [*] 节
        return text[:pos] + f"\n\n[*]\n{line}\n" + text[pos:]
    return f"[*]\n{line}\n" + text


# 单模型可调参数白名单
MODEL_PARAM_KEYS = ("temp", "ctx-size", "reasoning-budget", "reasoning-effort")
_REASONING_EFFORTS = ("low", "medium", "high", "xhigh")


def _find_model_section(text: str, model_id: str) -> tuple[int, int] | None:
    """定位 ``[<model_id>]`` 节，返回 (节头结束位置, 节结束位置)；节不存在返回 None。"""
    m = re.search(rf"^\[{re.escape(model_id)}\][ \t]*$", text, re.MULTILINE)
    if not m:
        return None
    rest = text[m.end():]
    nxt = _NEXT_SECTION.search(rest)
    end = m.end() + nxt.start() if nxt else len(text)
    return m.end(), end


def read_model_params_in_ini(text: str, model_id: str) -> dict | None:
    """读取 ``[<model_id>]`` 节内白名单 4 键的当前值（字符串形式）。

    节不存在返回 None；节内缺失的键对应值为 None。
    """
    found = _find_model_section(text, model_id)
    if found is None:
        return None
    section = text[found[0]:found[1]]
    params: dict[str, str | None] = {}
    for key in MODEL_PARAM_KEYS:
        m = re.search(rf"^[ \t]*{re.escape(key)}[ \t]*=[ \t]*(.*)$",
                      section, re.MULTILINE)
        params[key] = m.group(1).strip() if m else None
    return params


def _validate_model_param(key: str, raw) -> str | None:
    """校验单个参数值，返回写入文件的字符串；None/空串表示删除该行。

    校验失败抛 ValueError。
    """
    if raw is None or (isinstance(raw, str) and raw.strip() == ""):
        return None
    if key == "temp":
        if isinstance(raw, bool) or not isinstance(raw, (int, float, str)):
            raise ValueError("temp 须为 0.0..2.0 的浮点数")
        try:
            v = float(raw)
        except (TypeError, ValueError):
            raise ValueError("temp 须为 0.0..2.0 的浮点数")
        if not math.isfinite(v) or not (0.0 <= v <= 2.0):
            raise ValueError("temp 须为 0.0..2.0 的浮点数")
        return str(v)
    if key == "ctx-size":
        if isinstance(raw, bool) or not isinstance(raw, int):
            raise ValueError("ctx-size 须为 512..1048576 的整数")
        if not (512 <= raw <= 1048576):
            raise ValueError("ctx-size 须为 512..1048576 的整数")
        return str(raw)
    if key == "reasoning-budget":
        if isinstance(raw, bool) or not isinstance(raw, int):
            raise ValueError("reasoning-budget 须为 >= -1 的整数（或空值移除）")
        if raw < -1:
            raise ValueError("reasoning-budget 须为 >= -1 的整数（或空值移除）")
        return str(raw)
    if key == "reasoning-effort":
        if not isinstance(raw, str) or raw not in _REASONING_EFFORTS:
            raise ValueError(
                f"reasoning-effort 须为 {'|'.join(_REASONING_EFFORTS)}（或空值移除）")
        return raw
    raise ValueError(f"非法参数键: {key!r}（仅允许 {', '.join(MODEL_PARAM_KEYS)}）")


def update_model_params_in_ini(text: str, model_id: str, updates: dict) -> str:
    """更新 ``[<model_id>]`` 节内白名单参数，仅修改该节，绝不破坏注释和其他节。

    - 节不存在或含白名单外键 → ValueError；
    - 各键取值校验见 _validate_model_param，失败抛 ValueError；
    - 值非空：节内已有该行则原地替换，否则追加到节末尾；
      值为 None/空串：删除该行（不存在则不变）。
    """
    found = _find_model_section(text, model_id)
    if found is None:
        raise ValueError(f"models.ini 中不存在 [{model_id}] 节")
    for key in updates:
        if key not in MODEL_PARAM_KEYS:
            raise ValueError(f"非法参数键: {key!r}（仅允许 {', '.join(MODEL_PARAM_KEYS)}）")

    start, end = found
    section = text[start:end]
    for key, raw in updates.items():
        val = _validate_model_param(key, raw)
        pat = re.compile(rf"^[ \t]*{re.escape(key)}[ \t]*=[^\n]*\n?", re.MULTILINE)
        if val is None:
            section = pat.sub("", section, count=1)
        elif pat.search(section):
            section = pat.sub(lambda _m: f"{key} = {val}\n", section, count=1)
        else:
            prefix = "" if section.endswith("\n") or not section else "\n"
            section += prefix + f"{key} = {val}\n"
    return text[:start] + section + text[end:]
