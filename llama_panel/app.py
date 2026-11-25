"""Flask 应用：llama.cpp 监控与控制面板 API。"""
from __future__ import annotations

import os
import time
from urllib.parse import urlparse

from flask import Flask, jsonify, request, send_from_directory

from .client import LlamaClient
from .process import ServerProcess, find_pid_on_port, kill_pid

_STATIC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "static")
_LOCAL_HOSTS = {"127.0.0.1", "localhost", "::1"}


def create_app(server_url: str, server_command: list[str], log_path: str) -> Flask:
    app = Flask(__name__, static_folder=None)
    client = LlamaClient(server_url)
    proc = ServerProcess(server_command, log_path=log_path)
    _host = urlparse(server_url).hostname or ""
    _port = urlparse(server_url).port

    def _external_pid() -> int | None:
        """面板未管理但 server 健康时，按端口发现本机外部进程 PID。"""
        if _host not in _LOCAL_HOSTS or _port is None:
            return None
        return find_pid_on_port(_port)

    @app.get("/")
    def index():
        return send_from_directory(_STATIC, "index.html")

    @app.get("/api/status")
    def api_status():
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
        return jsonify(
            {
                "ts": time.time(),
                "server": {
                    "base_url": server_url,
                    "health": client.health(),
                    "models": client.models(),
                    "process": process,
                },
                "slots": client.slots(),
                "metrics": client.metrics(),
            }
        )

    @app.get("/api/logs")
    def api_logs():
        n = request.args.get("n", 200, type=int)
        return jsonify({"lines": proc.tail(n)})

    @app.post("/api/server/start")
    def api_server_start():
        try:
            pid = proc.start()
        except RuntimeError as e:
            return jsonify({"ok": False, "error": str(e)}), 409
        except OSError as e:
            return jsonify({"ok": False, "error": str(e)}), 500
        return jsonify({"ok": True, "pid": pid})

    @app.post("/api/server/stop")
    def api_server_stop():
        if proc.is_running():
            proc.stop()
        else:
            ext = _external_pid()
            if ext is not None:
                kill_pid(ext)
        return jsonify({"ok": True})

    return app
