"""`/v1/*` 透明代理的内存会话追踪（纯标准库、线程安全、仅内存不落盘）。

TraceStore 维护最近 maxlen 条 chat 会话记录：
- start_trace：POST /v1/chat/completions 时创建记录；
- feed_sse_chunk：流式 SSE 逐块累加 reasoning/content/tool_calls 与 timings/usage；
- finish_trace：流结束/非流式响应完成时收尾（status/elapsed_s/error）。
"""
from __future__ import annotations

import threading
import time
from collections import deque

_TRUNC_MARK = "…[truncated]"


def _num(v) -> float | None:
    """取数值（int/float，排除 bool）；否则 None。"""
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        return None
    return v


class TraceStore:
    """内存会话追踪存储：deque(maxlen) + Lock，自增 id，仅存内存不落盘。"""

    def __init__(self, maxlen: int = 20, max_text: int = 16000):
        self.maxlen = maxlen
        self.max_text = max_text
        self._store: deque[dict] = deque(maxlen=maxlen)
        self._lock = threading.Lock()
        self._next_id = 1

    # ---------- 内部工具 ----------

    def _cap(self, text: str) -> str:
        """超过 max_text 时截断并保留末尾标志。"""
        if len(text) > self.max_text:
            return text[: self.max_text] + _TRUNC_MARK
        return text

    def _find(self, trace_id: int) -> dict | None:
        for rec in self._store:
            if rec["id"] == trace_id:
                return rec
        return None

    def _extract_metrics(self, obj: dict, rec: dict) -> None:
        """从 timings / usage 提取 prompt_n / predicted_n / 每秒速度（timings 优先）。"""
        timings = obj.get("timings")
        if isinstance(timings, dict):
            for key in ("prompt_n", "predicted_n", "prompt_per_second", "predicted_per_second"):
                v = _num(timings.get(key))
                if v is not None:
                    rec[key] = v
        usage = obj.get("usage")
        if isinstance(usage, dict):
            if rec["prompt_n"] is None:
                v = _num(usage.get("prompt_n", usage.get("prompt_tokens")))
                if v is not None:
                    rec["prompt_n"] = v
            if rec["predicted_n"] is None:
                v = _num(usage.get("predicted_n", usage.get("completion_tokens")))
                if v is not None:
                    rec["predicted_n"] = v

    def _merge_delta(self, rec: dict, delta: dict) -> None:
        """累加流式 choices[0].delta：reasoning_content / content / tool_calls。"""
        rz = delta.get("reasoning_content")
        if isinstance(rz, str):
            rec["reasoning"] = self._cap(rec["reasoning"] + rz)
        c = delta.get("content")
        if isinstance(c, str):
            rec["content"] = self._cap(rec["content"] + c)
        tcs = delta.get("tool_calls")
        if isinstance(tcs, list):
            calls = rec["tool_calls"]
            for tc in tcs:
                if not isinstance(tc, dict):
                    continue
                idx = tc.get("index")
                if not isinstance(idx, int) or isinstance(idx, bool) or idx < 0:
                    idx = 0
                while len(calls) <= idx:
                    calls.append({"index": len(calls),
                                 "function": {"name": "", "arguments": ""}})
                fn = tc.get("function")
                if not isinstance(fn, dict):
                    continue
                name = fn.get("name")
                if isinstance(name, str):
                    calls[idx]["function"]["name"] += name
                args = fn.get("arguments")
                if isinstance(args, str):
                    calls[idx]["function"]["arguments"] += args

    # ---------- 公开 API ----------

    def start_trace(self, req_json: dict) -> dict:
        """创建一条 chat 会话记录（仅用于 POST /v1/chat/completions），返回记录。"""
        messages = req_json.get("messages") if isinstance(req_json, dict) else None
        if not isinstance(messages, list):
            messages = []
        last_user = ""
        for m in reversed(messages):
            if isinstance(m, dict) and m.get("role") == "user":
                c = m.get("content")
                last_user = c if isinstance(c, str) else (str(c) if c is not None else "")
                break
        preview = []
        for m in messages[-6:]:
            if not isinstance(m, dict):
                continue
            c = m.get("content")
            text = c if isinstance(c, str) else (str(c) if c is not None else "")
            preview.append({"role": m.get("role"), "text": text[:1500]})
        rec = {
            "id": 0,
            "ts": time.time(),
            "model": req_json.get("model") if isinstance(req_json, dict) else None,
            "stream": bool(req_json.get("stream")) if isinstance(req_json, dict) else False,
            "status": "streaming",
            "elapsed_s": 0.0,
            "messages_count": len(messages),
            "last_user": last_user[:4000],
            "messages_preview": preview,
            "reasoning": "",
            "content": "",
            "tool_calls": [],
            "prompt_n": None,
            "predicted_n": None,
            "prompt_per_second": None,
            "predicted_per_second": None,
            "error": None,
        }
        with self._lock:
            rec["id"] = self._next_id
            self._next_id += 1
            self._store.append(rec)
        return rec

    def feed_sse_chunk(self, trace_id: int, obj: dict) -> None:
        """实时累加流式 SSE 事件（choices[0].delta 与 timings/usage）。"""
        if not isinstance(obj, dict):
            return
        with self._lock:
            rec = self._find(trace_id)
            if rec is None:
                return
            choices = obj.get("choices")
            if isinstance(choices, list) and choices and isinstance(choices[0], dict):
                delta = choices[0].get("delta")
                if isinstance(delta, dict):
                    self._merge_delta(rec, delta)
            self._extract_metrics(obj, rec)

    def finish_trace(self, trace_id: int, elapsed_s: float,
                     resp_json: dict | None = None, error: str | None = None) -> None:
        """收尾：非流式时从 resp_json 提取结果；设置 status / elapsed_s / error。"""
        with self._lock:
            rec = self._find(trace_id)
            if rec is None:
                return
            if isinstance(resp_json, dict):
                choices = resp_json.get("choices")
                if isinstance(choices, list) and choices and isinstance(choices[0], dict):
                    msg = choices[0].get("message")
                    if isinstance(msg, dict):
                        c = msg.get("content")
                        rec["content"] = self._cap(c if isinstance(c, str) else "")
                        rz = msg.get("reasoning_content")
                        rec["reasoning"] = self._cap(rz if isinstance(rz, str) else "")
                        tcs = msg.get("tool_calls")
                        if isinstance(tcs, list):
                            calls = []
                            for i, tc in enumerate(tcs):
                                if not isinstance(tc, dict):
                                    continue
                                fn = tc.get("function")
                                fn = fn if isinstance(fn, dict) else {}
                                calls.append({
                                    "index": tc.get("index", i),
                                    "function": {
                                        "name": fn.get("name", ""),
                                        "arguments": fn.get("arguments", ""),
                                    },
                                })
                            if calls:
                                rec["tool_calls"] = calls
                self._extract_metrics(resp_json, rec)
            rec["status"] = "error" if error else "ok"
            rec["elapsed_s"] = elapsed_s
            rec["error"] = error

    def list_traces(self, limit: int = 20) -> list[dict]:
        """按最新在前返回浅拷贝列表。"""
        with self._lock:
            items = list(self._store)
        if limit > 0:
            items = items[-limit:]
        else:
            items = []
        return [dict(r) for r in reversed(items)]

    def clear(self) -> None:
        with self._lock:
            self._store.clear()
