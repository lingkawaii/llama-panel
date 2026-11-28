"""Flask 应用：llama.cpp 监控与控制面板 API。"""
from __future__ import annotations

import json
import math
import os
import subprocess
import threading
import time
from urllib.parse import urlparse

import requests

from flask import Flask, Response, jsonify, request, send_from_directory

from .client import LlamaClient, parse_model_state, parse_models_list, enrich_slots
from .proxy import TraceStore
from .gpu import get_gpu_stats, get_host_memory, set_gpu_power_cap
from .preset import (
    update_sleep_idle_in_ini, read_model_params_in_ini, update_model_params_in_ini,
)
from .power import PROFILES, get_power_status, set_power_profile
from .process import ServerProcess, find_pid_on_port, kill_pid

_STATIC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "static")
_LOCAL_HOSTS = {"127.0.0.1", "localhost", "::1"}


def _systemctl(args: list[str]) -> tuple[bool, str]:
    """执行 systemctl 子命令，返回 (是否成功, 输出文本)。"""
    try:
        out = subprocess.run(
            ["systemctl", *args], capture_output=True, timeout=10
        )
    except (OSError, subprocess.SubprocessError) as e:
        return False, f"systemctl 不可用: {e}"
    text = (out.stdout or b"").decode("utf-8", "replace").strip()
    err = (out.stderr or b"").decode("utf-8", "replace").strip()
    if out.returncode != 0:
        return False, (err or text or f"systemctl {args[0]} 返回 {out.returncode}")
    return True, text


def parse_systemd_uptime(active_enter_ts_usec: str, now_ts: float | None = None) -> float | None:
    """把 ActiveEnterTimestampMonotonic 的微秒值解析为运行时长（秒）。

    now_ts 为单调时钟秒数（默认 time.monotonic()）。
    输入无效、非正、或结果 <= 0 时返回 None。
    """
    try:
        usec = float(str(active_enter_ts_usec).strip())
    except (TypeError, ValueError):
        return None
    if not math.isfinite(usec) or usec <= 0:
        return None
    now = time.monotonic() if now_ts is None else now_ts
    up = (now * 1e6 - usec) / 1e6
    return up if up > 0 else None


class SystemdService:
    """通过 systemd 管理 llama-server 单元（非 root 无免密权限时操作会失败）。"""

    def __init__(self, unit: str):
        self.unit = unit

    def status(self) -> dict:
        ok, text = _systemctl(["is-active", self.unit])
        active = ok and text == "active"
        pid = None
        uptime = None
        if active:
            ok2, pid_text = _systemctl(["show", "-p", "MainPID", "--value", self.unit])
            if ok2 and pid_text.isdigit():
                pid = int(pid_text)
            ok3, up_text = _systemctl(
                ["show", "-p", "ActiveEnterTimestampMonotonic", "--value", self.unit]
            )
            if ok3:
                uptime = parse_systemd_uptime(up_text)
        return {"running": active, "external": False, "pid": pid,
                "uptime": uptime, "command": [f"systemd:{self.unit}"]}

    def _call(self, action: str) -> tuple[bool, str]:
        return _systemctl([action, self.unit])

    def start(self) -> None:
        ok, text = self._call("start")
        if not ok:
            raise RuntimeError(f"systemctl start {self.unit} 失败: {text}")

    def stop(self) -> None:
        ok, text = self._call("stop")
        if not ok:
            raise RuntimeError(f"systemctl stop {self.unit} 失败: {text}")

    def restart(self) -> None:
        ok, text = self._call("restart")
        if not ok:
            raise RuntimeError(f"systemctl restart {self.unit} 失败: {text}")


def create_app(
    server_url: str,
    server_command: list[str],
    log_path: str,
    systemd_unit: str | None = None,
    models_preset: str | None = None,
) -> Flask:
    app = Flask(__name__, static_folder=None)
    client = LlamaClient(server_url)
    proc = ServerProcess(server_command, log_path=log_path)
    svc = SystemdService(systemd_unit) if systemd_unit else None
    _host = urlparse(server_url).hostname or ""
    _port = urlparse(server_url).port

    # 并发防护：Flask 默认多线程。状态轮询与卸载并发时，status 可能在
    # 模型卸载中/刚卸载后向 Router 发 ?model= 的 /slots /metrics 请求，
    # 触发 --models-autoload 误唤醒。model_lock 串行化状态查询与
    # 加载/卸载；卸载后 /api/status 在 UNLOAD_COOLDOWN_SEC 秒冷却期内
    # 跳过 /slots /metrics（time.monotonic() 秒，nonlocal 维护）。
    model_lock = threading.Lock()
    unload_cooldown_until = 0.0
    UNLOAD_COOLDOWN_SEC = 5.0

    # /v1/* 透明代理的内存会话追踪：独立锁（TraceStore 内部），
    # 绝不占用 model_lock；仅存内存不落盘。
    trace_store = TraceStore(maxlen=20)

    # 槽位速度内存缓存：键为 (slot_id, task_id)，/api/status 每次解析日志时
    # 合并进 enrich_slots，防止 proxy_request 刷屏把真实 timing 行冲出日志
    # 窗口后丢失槽位速度。子进程重启后旧 task_id 会串味，故在服务停止/
    # 重启、模型加载/卸载及预设修改重启时清空。
    slot_timing_cache: dict[tuple[int, int], dict] = {}

    # 真实推理活跃追踪（Router 模式）：/api/status 轮询本身会重置
    # llama-server 的 --sleep-idle-seconds 空闲计时器，故只有「真实活动」
    # （槽位处理中 / token 数增长）才能作为空闲计时基准；无活动超过
    # sleep_idle_seconds 时由面板主动卸载。均在 model_lock 保护下，
    # 时间基准为 time.monotonic()。
    idle_track_model: str | None = None
    last_inference_ts: float = 0.0
    last_token_total: float | None = None

    def _external_pid() -> int | None:
        """面板未管理但 server 健康时，按端口发现本机外部进程 PID。"""
        if _host not in _LOCAL_HOSTS or _port is None:
            return None
        return find_pid_on_port(_port)

    def _process_info() -> dict:
        if svc is not None:
            return svc.status()
        process = {
            "pid": proc.pid,
            "running": proc.is_running(),
            "uptime": proc.uptime(),
            "command": proc.command,
            "external": False,
        }
        if not process["running"] and client.health():
            ext = _external_pid()
            if ext is not None:
                process.update(running=True, external=True, pid=ext)
        return process

    @app.get("/")
    def index():
        return send_from_directory(_STATIC, "index.html")

    def _metric_value(metrics: dict[str, float] | None, name: str) -> float | None:
        """按指标名取值（key 可能带 {...} 标签）；metrics 为空或缺失返回 None。"""
        if not metrics:
            return None
        for k, v in metrics.items():
            if k.split("{", 1)[0] == name:
                return v
        return None

    @app.get("/api/status")
    def api_status():
        nonlocal unload_cooldown_until, idle_track_model, last_inference_ts, last_token_total
        # 持锁完成「查模型状态 → 查 slots/metrics」整段，
        # 避免卸载在中间插入、对卸载中/刚卸载的模型发 ?model= 请求误唤醒。
        with model_lock:
            models = client.models()
            ms = parse_model_state(models)
            # Router 模式防误唤醒：仅 loaded 且不在卸载冷却期内才能请求
            # /slots 与 /metrics；unloaded 或冷却期内请求它们会触发
            # --models-autoload 误唤醒模型。
            if ms["router_mode"]:
                in_cooldown = time.monotonic() < unload_cooldown_until
                if ms["state"] == "loaded" and ms["model_id"] and not in_cooldown:
                    slots = client.slots(ms["model_id"])
                    slots = enrich_slots(slots, proc.tail(300), slot_timing_cache)
                    metrics = client.metrics(ms["model_id"])
                else:
                    slots, metrics = [], None
            else:
                slots = client.slots()
                slots = enrich_slots(slots, proc.tail(300), slot_timing_cache)
                metrics = client.metrics()
            # 真实推理活跃追踪：纯轮询不延长空闲计时，仅槽位处理中或
            # token 数增长算活动；无活动超过 sleep_idle_seconds 时由面板
            # 主动卸载（llama-server 自身空闲计时器被轮询不断重置，永不触发）。
            if ms["router_mode"]:
                now = time.monotonic()
                if ms["state"] == "loaded" and ms["model_id"] and not in_cooldown:
                    pt = _metric_value(metrics, "llamacpp:prompt_tokens_total")
                    tp = _metric_value(metrics, "llamacpp:tokens_predicted_total")
                    tok_total = (pt + tp) if (pt is not None and tp is not None) else None
                    active = (
                        idle_track_model != ms["model_id"]
                        or any(s.get("is_processing") for s in slots)
                        or (tok_total is not None and last_token_total is not None
                            and tok_total > last_token_total)
                    )
                    if active:
                        idle_track_model = ms["model_id"]
                        last_inference_ts = now
                        last_token_total = tok_total
                    else:
                        sleep_idle = ms["sleep_idle_seconds"]
                        if (isinstance(sleep_idle, int) and sleep_idle > 0
                                and (now - last_inference_ts) >= sleep_idle):
                            res = client.unload_model(ms["model_id"])
                            if res.get("ok"):
                                unload_cooldown_until = now + UNLOAD_COOLDOWN_SEC
                                slot_timing_cache.clear()
                                idle_track_model = None
                                last_token_total = None
                                slots, metrics = [], None
                                models = client.models()
                                ms = parse_model_state(models)
                elif ms["state"] != "loaded":
                    idle_track_model = None
                    last_token_total = None
        models_list = parse_models_list(models)
        power_status = get_power_status()
        return jsonify(
            {
                "ts": time.time(),
                "server": {
                    "base_url": server_url,
                    "health": client.health(),
                    "models": models,
                    "models_list": models_list,
                    "model_state": ms,
                    "preset_editable": bool(models_preset),
                    "process": _process_info(),
                },
                "slots": slots,
                "metrics": metrics,
                "gpu": get_gpu_stats(),
                "memory": get_host_memory(),
                "power": power_status,
            }
        )

    def _model_id_from_body() -> str | None:
        data = request.get_json(silent=True)
        if isinstance(data, dict):
            model = data.get("model")
            if isinstance(model, str) and model:
                return model
        return None

    def _resolve_model() -> str | None:
        """body 指定 model 优先；缺省时取当前 /v1/models 的 model_id。"""
        model = _model_id_from_body()
        if model:
            return model
        return parse_model_state(client.models()).get("model_id")

    @app.post("/api/model/load")
    def api_model_load():
        """Router 模式手动加载模型（唤醒）。body 可选 {"model": "..."}。"""
        nonlocal unload_cooldown_until, idle_track_model, last_token_total
        model = _resolve_model()
        if not model:
            return jsonify({"ok": False, "error": "未指定 model，且当前无法从 /v1/models 获取 model_id"}), 400
        with model_lock:
            # 主动加载：立即恢复 /api/status 的 /slots /metrics 查询；
            # 重置活跃追踪，空闲计时自下次 status 首次观测重新起算。
            unload_cooldown_until = 0.0
            idle_track_model = None
            last_token_total = None
            result = client.load_model(model)
        if result.get("ok"):
            slot_timing_cache.clear()
        return jsonify(result), (200 if result.get("ok") else 502)

    @app.post("/api/model/unload")
    def api_model_unload():
        """Router 模式手动卸载模型释放显存。body 可选 {"model": "..."}。"""
        nonlocal unload_cooldown_until, idle_track_model, last_token_total
        model = _resolve_model()
        if not model:
            return jsonify({"ok": False, "error": "未指定 model，且当前无法从 /v1/models 获取 model_id"}), 400
        with model_lock:
            # 先进入冷却期：卸载期间 status 轮询不再请求 /slots /metrics；
            # 同时重置活跃追踪。
            unload_cooldown_until = time.monotonic() + UNLOAD_COOLDOWN_SEC
            idle_track_model = None
            last_token_total = None
            result = client.unload_model(model)
            if result.get("ok"):
                unload_cooldown_until = time.monotonic() + UNLOAD_COOLDOWN_SEC
            else:
                unload_cooldown_until = 0.0
        if result.get("ok"):
            slot_timing_cache.clear()
        return jsonify(result), (200 if result.get("ok") else 502)

    @app.get("/api/logs")
    def api_logs():
        n = request.args.get("n", 200, type=int)
        return jsonify({"lines": proc.tail(n)})

    @app.post("/api/probe")
    def api_probe():
        """一键快速测速：极短 chat 请求，返回耗时与 prefill/decode 速度。"""
        nonlocal unload_cooldown_until
        model = None
        data = request.get_json(silent=True)
        if isinstance(data, dict):
            model = data.get("model")
        # 主动测速允许唤醒模型：清零冷却期，/api/status 立即恢复查询。
        # 只在锁内清零，不在锁内执行耗时的 probe，避免阻塞 /api/status。
        with model_lock:
            unload_cooldown_until = 0.0
        result = client.probe(model)
        return jsonify(result), (200 if result.get("ok") else 502)

    @app.post("/api/chat")
    def api_chat():
        """快捷对话试跑：向上游发单轮 chat 请求，返回回复与速度指标。

        body: {"model": str|None, "prompt": str, "max_tokens": int (默认 512, 上限 4096)}
        空 prompt → 400；上游异常/非 200 → ok False。
        """
        data = request.get_json(silent=True)
        if not isinstance(data, dict):
            return jsonify({"ok": False, "error": "请求体须为 JSON 对象"}), 400
        prompt = data.get("prompt")
        if not isinstance(prompt, str) or not prompt.strip():
            return jsonify({"ok": False, "error": "提示词不能为空"}), 400
        model = data.get("model")
        if not isinstance(model, str) or not model:
            model = None
        max_tokens = data.get("max_tokens", 512)
        if isinstance(max_tokens, bool) or not isinstance(max_tokens, int):
            try:
                max_tokens = int(max_tokens)
            except (TypeError, ValueError):
                max_tokens = 512
        max_tokens = max(16, min(max_tokens, 4096))
        payload = {
            "messages": [{"role": "user", "content": prompt.strip()}],
            "max_tokens": max_tokens,
            "stream": False,
        }
        if model:
            payload["model"] = model
        try:
            start = time.monotonic()
            r = requests.post(
                server_url.rstrip("/") + "/v1/chat/completions",
                json=payload, timeout=120,
            )
            dt = time.monotonic() - start
            if r.status_code != 200:
                return jsonify({"ok": False, "error": f"上游返回 HTTP {r.status_code}"})
            body = r.json()
        except requests.RequestException as e:
            return jsonify({"ok": False, "error": f"请求失败: {e}"})
        except ValueError:
            return jsonify({"ok": False, "error": "响应不是合法 JSON"})
        if not isinstance(body, dict):
            return jsonify({"ok": False, "error": "响应不是合法 JSON"})
        content = ""
        reasoning = ""
        choices = body.get("choices")
        if isinstance(choices, list) and choices and isinstance(choices[0], dict):
            msg = choices[0].get("message")
            if isinstance(msg, dict):
                c = msg.get("content")
                content = c if isinstance(c, str) else ""
                rz = msg.get("reasoning_content")
                reasoning = rz if isinstance(rz, str) else ""
        timings = body.get("timings")
        if not isinstance(timings, dict):
            timings = {}
        return jsonify({
            "ok": True,
            "model": body.get("model"),
            "content": content,
            "reasoning": reasoning,
            "elapsed_s": round(dt, 2),
            "prompt_n": timings.get("prompt_n"),
            "predicted_n": timings.get("predicted_n"),
            "prompt_per_second": timings.get("prompt_per_second"),
            "predicted_per_second": timings.get("predicted_per_second"),
        })

    @app.post("/api/chat/stream")
    def api_chat_stream():
        """快捷对话流式试跑：SSE 逐块推送 reasoning/content/done 事件。

        body 同 /api/chat；空 prompt → 400。
        上游非 text/event-stream（如 mock 普通 JSON）时解析完整响应，
        一次性发出 content（+reasoning）后发 done。
        """
        data = request.get_json(silent=True)
        if not isinstance(data, dict):
            return jsonify({"ok": False, "error": "请求体须为 JSON 对象"}), 400
        prompt = data.get("prompt")
        if not isinstance(prompt, str) or not prompt.strip():
            return jsonify({"ok": False, "error": "提示词不能为空"}), 400
        model = data.get("model")
        if not isinstance(model, str) or not model:
            model = None
        max_tokens = data.get("max_tokens", 512)
        if isinstance(max_tokens, bool) or not isinstance(max_tokens, int):
            try:
                max_tokens = int(max_tokens)
            except (TypeError, ValueError):
                max_tokens = 512
        max_tokens = max(16, min(max_tokens, 4096))
        payload = {
            "messages": [{"role": "user", "content": prompt.strip()}],
            "max_tokens": max_tokens,
            "stream": True,
        }
        if model:
            payload["model"] = model

        def sse(obj: dict) -> str:
            return "data: " + json.dumps(obj, ensure_ascii=False) + "\n\n"

        def generate():
            start = time.monotonic()
            timings: dict = {}
            try:
                r = requests.post(
                    server_url.rstrip("/") + "/v1/chat/completions",
                    json=payload, stream=True, timeout=120,
                )
                if r.status_code != 200:
                    yield sse({"type": "error", "error": f"上游返回 HTTP {r.status_code}"})
                    return
                r.encoding = "utf-8"
                ctype = r.headers.get("Content-Type", "")
                if "text/event-stream" not in ctype:
                    # 兼容上游返回普通非流式 JSON 的 mock：一次性发 content + done
                    try:
                        body = r.json()
                    except ValueError:
                        yield sse({"type": "error", "error": "响应不是合法 JSON"})
                        return
                    if not isinstance(body, dict):
                        yield sse({"type": "error", "error": "响应不是合法 JSON"})
                        return
                    content = ""
                    reasoning = ""
                    choices = body.get("choices")
                    if isinstance(choices, list) and choices and isinstance(choices[0], dict):
                        msg = choices[0].get("message")
                        if isinstance(msg, dict):
                            c = msg.get("content")
                            content = c if isinstance(c, str) else ""
                            rz = msg.get("reasoning_content")
                            reasoning = rz if isinstance(rz, str) else ""
                    t = body.get("timings")
                    if isinstance(t, dict):
                        timings = t
                    if reasoning:
                        yield sse({"type": "reasoning", "delta": reasoning})
                    if content:
                        yield sse({"type": "content", "delta": content})
                else:
                    for raw in r.iter_lines(decode_unicode=False):
                        line = raw.decode("utf-8", errors="replace")
                        if not line or not line.startswith("data:"):
                            continue
                        data_str = line[5:].strip()
                        if data_str == "[DONE]":
                            break
                        try:
                            obj = json.loads(data_str)
                        except ValueError:
                            continue
                        if not isinstance(obj, dict):
                            continue
                        choices = obj.get("choices")
                        if isinstance(choices, list) and choices and isinstance(choices[0], dict):
                            delta = choices[0].get("delta")
                            if isinstance(delta, dict):
                                rz = delta.get("reasoning_content")
                                if isinstance(rz, str):
                                    yield sse({"type": "reasoning", "delta": rz})
                                c = delta.get("content")
                                if isinstance(c, str):
                                    yield sse({"type": "content", "delta": c})
                        t = obj.get("timings")
                        if isinstance(t, dict):
                            timings = t
                dt = time.monotonic() - start
                yield sse({
                    "type": "done",
                    "elapsed_s": round(dt, 2),
                    "prompt_n": timings.get("prompt_n"),
                    "predicted_n": timings.get("predicted_n"),
                    "prompt_per_second": timings.get("prompt_per_second"),
                    "predicted_per_second": timings.get("predicted_per_second"),
                })
            except requests.RequestException as e:
                yield sse({"type": "error", "error": f"请求失败: {e}"})

        resp = Response(generate(), mimetype="text/event-stream; charset=utf-8")
        resp.headers["Cache-Control"] = "no-cache"
        resp.headers["X-Accel-Buffering"] = "no"
        return resp

    # hop-by-hop 请求头：转发时剔除，由 requests 按上游连接重新生成。
    _HOP_BY_HOP = {"host", "content-length", "transfer-encoding", "connection"}

    @app.route("/v1/<path:subpath>", methods=["GET", "POST", "OPTIONS", "DELETE"])
    def v1_proxy(subpath):
        """`/v1/*` 零损耗透明流式反向代理。

        原字节转发查询参数与请求体到上游 `{server_url}/v1/{subpath}`；
        上游 SSE 时逐块原字节立即 yield 给客户端（同时内存解析喂 trace_store），
        非流式时读完整字节原样返回。仅 POST /v1/chat/completions 写 trace。
        """
        headers = {k: v for k, v in request.headers.items()
                   if k.lower() not in _HOP_BY_HOP}
        url = f"{server_url.rstrip('/')}/v1/{subpath}"
        body = request.get_data()
        trace = None
        if request.method == "POST" and subpath == "chat/completions":
            req_json = request.get_json(silent=True)
            if isinstance(req_json, dict):
                trace = trace_store.start_trace(req_json)
        start = time.monotonic()
        try:
            upstream = requests.request(
                request.method, url, params=request.args, headers=headers,
                data=body, stream=True, timeout=(10, 600),
            )
        except requests.RequestException as e:
            if trace is not None:
                trace_store.finish_trace(
                    trace["id"], time.monotonic() - start, error=f"上游请求失败: {e}")
            return jsonify({"ok": False, "error": f"上游请求失败: {e}"}), 502
        ctype = upstream.headers.get("Content-Type", "")
        if trace is not None and "text/event-stream" in ctype:
            trace_id = trace["id"]
            status_code = upstream.status_code

            def generate():
                buf = b""
                error = None
                if status_code != 200:
                    error = f"上游返回 HTTP {status_code}"
                try:
                    for chunk in upstream.iter_content(chunk_size=None):
                        if not chunk:
                            continue
                        # 原字节立即转发给客户端（零损耗）
                        yield chunk
                        # 内存中按行缓冲 UTF-8 解码，解析 data: {...} 喂 trace
                        buf += chunk
                        while b"\n" in buf:
                            line, buf = buf.split(b"\n", 1)
                            text = line.decode("utf-8", errors="replace").strip()
                            if not text.startswith("data:"):
                                continue
                            data_str = text[5:].strip()
                            if data_str == "[DONE]":
                                continue
                            try:
                                obj = json.loads(data_str)
                            except ValueError:
                                continue
                            if isinstance(obj, dict):
                                trace_store.feed_sse_chunk(trace_id, obj)
                except (requests.RequestException, OSError) as e:
                    error = f"上游流中断: {e}"
                finally:
                    try:
                        upstream.close()
                    except OSError:
                        pass
                    trace_store.finish_trace(
                        trace_id, time.monotonic() - start, error=error)

            resp = Response(generate(), status=status_code,
                            mimetype=ctype or "text/event-stream")
            resp.headers["Cache-Control"] = "no-cache"
            resp.headers["X-Accel-Buffering"] = "no"
            return resp
        # 非流式：读取完整字节原样返回
        raw = upstream.content
        upstream.close()
        if trace is not None:
            resp_json = None
            try:
                parsed = json.loads(raw.decode("utf-8", "replace"))
                if isinstance(parsed, dict):
                    resp_json = parsed
            except ValueError:
                resp_json = None
            err = None if upstream.status_code == 200 \
                else f"上游返回 HTTP {upstream.status_code}"
            trace_store.finish_trace(
                trace["id"], time.monotonic() - start, resp_json=resp_json, error=err)
        return Response(raw, status=upstream.status_code,
                        mimetype=ctype or "application/octet-stream")

    @app.get("/api/traces")
    def api_traces():
        """最近 chat 会话追踪列表（最新在前）。?limit=20。"""
        limit = request.args.get("limit", 20, type=int)
        if limit is None or limit < 1:
            limit = 20
        return jsonify({"ok": True, "traces": trace_store.list_traces(limit)})

    @app.post("/api/traces/clear")
    def api_traces_clear():
        trace_store.clear()
        return jsonify({"ok": True})

    @app.post("/api/server/start")
    def api_server_start():
        if svc is not None:
            try:
                svc.start()
            except RuntimeError as e:
                return jsonify({"ok": False, "error": str(e)}), 500
            return jsonify({"ok": True})
        try:
            pid = proc.start()
        except RuntimeError as e:
            return jsonify({"ok": False, "error": str(e)}), 409
        except OSError as e:
            return jsonify({"ok": False, "error": str(e)}), 500
        return jsonify({"ok": True, "pid": pid})

    @app.post("/api/server/stop")
    def api_server_stop():
        nonlocal idle_track_model, last_token_total
        if svc is not None:
            try:
                svc.stop()
            except RuntimeError as e:
                return jsonify({"ok": False, "error": str(e)}), 500
            with model_lock:
                idle_track_model = None
                last_token_total = None
            slot_timing_cache.clear()
            return jsonify({"ok": True})
        if proc.is_running():
            proc.stop()
        else:
            ext = _external_pid()
            if ext is not None:
                kill_pid(ext)
        with model_lock:
            idle_track_model = None
            last_token_total = None
        slot_timing_cache.clear()
        return jsonify({"ok": True})

    @app.post("/api/server/restart")
    def api_server_restart():
        nonlocal idle_track_model, last_token_total
        if svc is not None:
            try:
                svc.restart()
            except RuntimeError as e:
                return jsonify({"ok": False, "error": str(e)}), 500
            with model_lock:
                idle_track_model = None
                last_token_total = None
            slot_timing_cache.clear()
            return jsonify({"ok": True})
        # 非 systemd：先停后启
        if proc.is_running():
            proc.stop()
        else:
            ext = _external_pid()
            if ext is not None:
                kill_pid(ext)
        try:
            pid = proc.start()
        except RuntimeError as e:
            return jsonify({"ok": False, "error": str(e)}), 409
        except OSError as e:
            return jsonify({"ok": False, "error": str(e)}), 500
        with model_lock:
            idle_track_model = None
            last_token_total = None
        slot_timing_cache.clear()
        return jsonify({"ok": True, "pid": pid})

    @app.post("/api/preset/sleep-idle")
    def api_preset_sleep_idle():
        """在线修改 models.ini 的 sleep-idle-seconds（空闲自动卸载时长）。

        body: {"seconds": int, "restart": bool}
        - 未配置 --models-preset 返回 400；
        - seconds 须为 -1（或 0，按 -1 处理）或 60..86400，否则 400；
        - restart 为 True 且服务受面板/systemd 托管时触发重启使配置生效。
        """
        if not models_preset:
            return jsonify({"ok": False, "error": "未配置 models 预设文件（--models-preset）"}), 400
        data = request.get_json(silent=True)
        if not isinstance(data, dict):
            return jsonify({"ok": False, "error": "请求体须为 JSON 对象"}), 400
        seconds = data.get("seconds")
        restart = bool(data.get("restart"))
        if isinstance(seconds, bool) or not isinstance(seconds, int):
            return jsonify({"ok": False, "error": "seconds 须为整数"}), 400
        if seconds <= 0:
            seconds = -1
        elif not (60 <= seconds <= 86400):
            return jsonify({"ok": False, "error": "seconds 须为 -1 或 60..86400"}), 400
        try:
            with open(models_preset, "r", encoding="utf-8") as f:
                text = f.read()
        except OSError as e:
            return jsonify({"ok": False, "error": f"读取预设文件失败: {e}"}), 500
        new_text = update_sleep_idle_in_ini(text, seconds)
        try:
            with open(models_preset, "w", encoding="utf-8") as f:
                f.write(new_text)
        except OSError as e:
            return jsonify({"ok": False, "error": f"写入预设文件失败: {e}"}), 500
        result = {"ok": True, "seconds": seconds}
        if restart:
            restarted = False
            try:
                if svc is not None:
                    svc.restart()
                    restarted = True
                elif proc.is_running():
                    proc.stop()
                    proc.start()
                    restarted = True
                else:
                    ext = _external_pid()
                    if ext is not None:
                        kill_pid(ext)
                        restarted = True
            except (RuntimeError, OSError) as e:
                return jsonify({"ok": False, "error": f"重启失败: {e}"}), 500
            result["restarted"] = restarted
            if restarted:
                slot_timing_cache.clear()
        return jsonify(result)

    @app.get("/api/preset/model/<path:model_id>")
    def api_preset_model_get(model_id):
        """读取 models.ini 中指定模型节的当前参数（白名单 4 键）。"""
        if not models_preset:
            return jsonify({"ok": False, "error": "未配置 models 预设文件（--models-preset）"}), 400
        if not os.path.exists(models_preset):
            return jsonify({"ok": False, "error": f"预设文件不存在: {models_preset}"}), 404
        try:
            with open(models_preset, "r", encoding="utf-8") as f:
                text = f.read()
        except OSError as e:
            return jsonify({"ok": False, "error": f"读取预设文件失败: {e}"}), 500
        params = read_model_params_in_ini(text, model_id)
        if params is None:
            return jsonify({"ok": False, "error": f"预设文件中不存在 [{model_id}] 节"}), 404
        return jsonify({"ok": True, "model": model_id, "params": params})

    @app.post("/api/preset/model/<path:model_id>")
    def api_preset_model_post(model_id):
        """在线微调 models.ini 中指定模型的参数并（可选）重启服务。

        body: {"params": {"temp"?, "ctx-size"?, "reasoning-budget"?, "reasoning-effort"?},
               "restart": bool (默认 True)}
        - 未配置 --models-preset → 400；预设文件不存在 → 404；
        - 节不存在 / 非法键 / 值越界 → 400；
        - restart 为 True 时清空槽位速度缓存与空闲追踪状态并重启服务。
        """
        nonlocal idle_track_model, last_token_total
        if not models_preset:
            return jsonify({"ok": False, "error": "未配置 models 预设文件（--models-preset）"}), 400
        if not os.path.exists(models_preset):
            return jsonify({"ok": False, "error": f"预设文件不存在: {models_preset}"}), 404
        data = request.get_json(silent=True)
        if not isinstance(data, dict):
            return jsonify({"ok": False, "error": "请求体须为 JSON 对象"}), 400
        params = data.get("params")
        if not isinstance(params, dict):
            return jsonify({"ok": False, "error": "params 须为 JSON 对象"}), 400
        restart = bool(data.get("restart", True))
        try:
            with open(models_preset, "r", encoding="utf-8") as f:
                text = f.read()
        except OSError as e:
            return jsonify({"ok": False, "error": f"读取预设文件失败: {e}"}), 500
        try:
            new_text = update_model_params_in_ini(text, model_id, params)
        except ValueError as e:
            return jsonify({"ok": False, "error": str(e)}), 400
        try:
            with open(models_preset, "w", encoding="utf-8") as f:
                f.write(new_text)
        except OSError as e:
            return jsonify({"ok": False, "error": f"写入预设文件失败: {e}"}), 500
        result = {"ok": True, "model": model_id}
        if restart:
            try:
                if svc is not None:
                    svc.restart()
                    result["restarted"] = True
                elif proc.is_running():
                    proc.stop()
                    proc.start()
                    result["restarted"] = True
                else:
                    ext = _external_pid()
                    if ext is not None:
                        kill_pid(ext)
                        result["restarted"] = True
                    else:
                        result["restarted"] = False
            except (RuntimeError, OSError) as e:
                return jsonify({"ok": False, "error": f"重启失败: {e}"}), 500
            if result["restarted"]:
                slot_timing_cache.clear()
                with model_lock:
                    idle_track_model = None
                    last_token_total = None
        return jsonify(result)

    @app.post("/api/gpu/power-cap")
    def api_gpu_power_cap():
        """设置 GPU 硬件功耗墙。body: {"watts": int}"""
        data = request.get_json(silent=True)
        watts = data.get("watts") if isinstance(data, dict) else None
        if not isinstance(watts, int) or isinstance(watts, bool):
            return jsonify({"ok": False, "error": "watts 须为整数"}), 400
        res = set_gpu_power_cap(watts)
        if res.get("ok"):
            return jsonify({"ok": True, "watts": watts}), 200
        # 越界/非法 → 400；不可用/权限不足 → 500
        err = res.get("error", "")
        if "越界" in err or "整数" in err:
            return jsonify(res), 400
        return jsonify(res), 500

    @app.post("/api/power/profile")
    def api_power_profile():
        """切换整机电源档位。body: {"profile": "power-saver"|"balanced"|"performance"}"""
        data = request.get_json(silent=True)
        profile = data.get("profile") if isinstance(data, dict) else None
        if profile not in PROFILES:
            return jsonify({"ok": False, "error": f"非法电源档位: {profile!r}"}), 400
        res = set_power_profile(profile)
        return jsonify(res), (200 if res["ok"] else 502)

    return app
