"""Flask 应用：llama.cpp 监控与控制面板 API。"""
from __future__ import annotations

import os
import time

from flask import Flask, jsonify, request, send_from_directory

from .client import LlamaClient
from .process import ServerProcess

_STATIC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "static")


def create_app(server_url: str, server_command: list[str], log_path: str) -> Flask:
    app = Flask(__name__, static_folder=None)
    client = LlamaClient(server_url)
    proc = ServerProcess(server_command, log_path=log_path)

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
                    "process": {
                        "pid": proc.pid,
                        "running": proc.is_running(),
                        "uptime": proc.uptime(),
                        "command": proc.command,
                    },
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
        proc.stop()
        return jsonify({"ok": True})

    return app
