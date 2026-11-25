"""Flask 应用：llama.cpp 监控与控制面板 API。"""
from __future__ import annotations

import math
import os
import subprocess
import threading
import time
from urllib.parse import urlparse

from flask import Flask, jsonify, request, send_from_directory

from .client import LlamaClient, parse_model_state
from .gpu import get_gpu_stats
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

    @app.get("/api/status")
    def api_status():
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
                    metrics = client.metrics(ms["model_id"])
                else:
                    slots, metrics = [], None
            else:
                slots = client.slots()
                metrics = client.metrics()
        return jsonify(
            {
                "ts": time.time(),
                "server": {
                    "base_url": server_url,
                    "health": client.health(),
                    "models": models,
                    "model_state": ms,
                    "process": _process_info(),
                },
                "slots": slots,
                "metrics": metrics,
                "gpu": get_gpu_stats(),
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
        nonlocal unload_cooldown_until
        model = _resolve_model()
        if not model:
            return jsonify({"ok": False, "error": "未指定 model，且当前无法从 /v1/models 获取 model_id"}), 400
        with model_lock:
            # 主动加载：立即恢复 /api/status 的 /slots /metrics 查询。
            unload_cooldown_until = 0.0
            result = client.load_model(model)
        return jsonify(result), (200 if result.get("ok") else 502)

    @app.post("/api/model/unload")
    def api_model_unload():
        """Router 模式手动卸载模型释放显存。body 可选 {"model": "..."}。"""
        nonlocal unload_cooldown_until
        model = _resolve_model()
        if not model:
            return jsonify({"ok": False, "error": "未指定 model，且当前无法从 /v1/models 获取 model_id"}), 400
        with model_lock:
            # 先进入冷却期：卸载期间 status 轮询不再请求 /slots /metrics。
            unload_cooldown_until = time.monotonic() + UNLOAD_COOLDOWN_SEC
            result = client.unload_model(model)
            if result.get("ok"):
                unload_cooldown_until = time.monotonic() + UNLOAD_COOLDOWN_SEC
            else:
                unload_cooldown_until = 0.0
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
        if svc is not None:
            try:
                svc.stop()
            except RuntimeError as e:
                return jsonify({"ok": False, "error": str(e)}), 500
            return jsonify({"ok": True})
        if proc.is_running():
            proc.stop()
        else:
            ext = _external_pid()
            if ext is not None:
                kill_pid(ext)
        return jsonify({"ok": True})

    @app.post("/api/server/restart")
    def api_server_restart():
        if svc is not None:
            try:
                svc.restart()
            except RuntimeError as e:
                return jsonify({"ok": False, "error": str(e)}), 500
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
        return jsonify({"ok": True, "pid": pid})

    return app
