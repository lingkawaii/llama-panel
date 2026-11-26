"""models.ini 预设文件读写：在线修改空闲自动卸载时长（纯函数）。"""
from __future__ import annotations

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
