"""Flask 应用：llama.cpp 监控与控制面板 API。"""
from __future__ import annotations

import os
import subprocess
import time
from urllib.parse import urlparse

from flask import Flask, jsonify, request, send_from_directory

from .client import LlamaClient
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


class SystemdService:
    """通过 systemd 管理 llama-server 单元（非 root 无免密权限时操作会失败）。"""

    def __init__(self, unit: str):
        self.unit = unit

    def status(self) -> dict:
        ok, text = _systemctl(["is-active", self.unit])
        active = ok and text == "active"
        pid = None
        if active:
            ok2, pid_text = _systemctl(["show", "-p", "MainPID", "--value", self.unit])
            if ok2 and pid_text.isdigit():
                pid = int(pid_text)
        return {"running": active, "external": False, "pid": pid,
                "uptime": None, "command": [f"systemd:{self.unit}"]}

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
        return jsonify(
            {
                "ts": time.time(),
                "server": {
                    "base_url": server_url,
                    "health": client.health(),
                    "models": client.models(),
                    "process": _process_info(),
                },
                "slots": client.slots(),
                "metrics": client.metrics(),
                "gpu": get_gpu_stats(),
            }
        )

    @app.get("/api/logs")
    def api_logs():
        n = request.args.get("n", 200, type=int)
        return jsonify({"lines": proc.tail(n)})

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
