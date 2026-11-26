"""Flask 应用：llama.cpp 监控与控制面板 API。"""
from __future__ import annotations

import math
import os
import subprocess
import threading
import time
from urllib.parse import urlparse

from flask import Flask, jsonify, request, send_from_directory

from .client import LlamaClient, parse_model_state, parse_models_list, enrich_slots
from .gpu import get_gpu_stats
from .preset import update_sleep_idle_in_ini
from .power import PROFILES, get_power_status, set_power_profile, trigger_suspend
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

    # 整机省电联动：auto_power_saver 开启时，Router 模式状态跃迁
    # （全部 unloaded ↔ 有模型 loaded）自动切 power-saver / balanced；
    # last_auto_power_state 记录已应用档位，避免每次轮询重复执行。
    auto_power_saver = False
    last_auto_power_state: str | None = None

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

    def _auto_target(states: list[str]) -> str | None:
        """按模型状态集合决定省电目标档位。

        有模型 loaded → balanced；全部 unloaded → power-saver；
        其余（loading 中）→ None（不切换）。
        """
        if any(s == "loaded" for s in states):
            return "balanced"
        if states and all(s == "unloaded" for s in states):
            return "power-saver"
        return None

    def _apply_auto_power(target: str | None, available: list[str]) -> None:
        """仅在目标与已应用档位不同、且目标在可用列表中时才切换。"""
        nonlocal last_auto_power_state
        if not target or not available:
            return
        if target == last_auto_power_state or target not in available:
            return
        set_power_profile(target)
        last_auto_power_state = target

    def _auto_power_from_models() -> None:
        """手动 load/unload 成功后按最新模型集合联动省电档位。"""
        if not auto_power_saver:
            return
        models = client.models()
        ms = parse_model_state(models)
        if not ms["router_mode"]:
            return
        ps = get_power_status()
        if ps is None:
            return
        _apply_auto_power(
            _auto_target([m["state"] for m in parse_models_list(models)]),
            ps["available"],
        )

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
                    slots = enrich_slots(slots, proc.tail(150))
                    metrics = client.metrics(ms["model_id"])
                else:
                    slots, metrics = [], None
            else:
                slots = client.slots()
                slots = enrich_slots(slots, proc.tail(150))
                metrics = client.metrics()
        models_list = parse_models_list(models)
        # 自动省电联动：auto 开启且 Router 模式，检测状态跃迁自动切档
        power_status = get_power_status()
        if auto_power_saver and ms["router_mode"] and power_status is not None:
            _apply_auto_power(
                _auto_target([m["state"] for m in models_list]),
                power_status["available"],
            )
        if power_status is not None:
            power_status["auto_power_saver"] = auto_power_saver
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
        nonlocal unload_cooldown_until
        model = _resolve_model()
        if not model:
            return jsonify({"ok": False, "error": "未指定 model，且当前无法从 /v1/models 获取 model_id"}), 400
        with model_lock:
            # 主动加载：立即恢复 /api/status 的 /slots /metrics 查询。
            unload_cooldown_until = 0.0
            result = client.load_model(model)
        if result.get("ok"):
            _auto_power_from_models()
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
        if result.get("ok"):
            _auto_power_from_models()
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
        return jsonify(result)

    @app.post("/api/power/profile")
    def api_power_profile():
        """切换整机电源档位。body: {"profile": "power-saver"|"balanced"|"performance"}"""
        data = request.get_json(silent=True)
        profile = data.get("profile") if isinstance(data, dict) else None
        if profile not in PROFILES:
            return jsonify({"ok": False, "error": f"非法电源档位: {profile!r}"}), 400
        res = set_power_profile(profile)
        return jsonify(res), (200 if res["ok"] else 502)

    @app.post("/api/power/auto")
    def api_power_auto():
        """开关「模型休眠自动切省电」。body: {"enabled": bool}"""
        nonlocal auto_power_saver, last_auto_power_state
        data = request.get_json(silent=True)
        enabled = data.get("enabled") if isinstance(data, dict) else None
        if not isinstance(enabled, bool):
            return jsonify({"ok": False, "error": "enabled 须为布尔值"}), 400
        auto_power_saver = enabled
        last_auto_power_state = None
        power_status = get_power_status()
        if enabled and power_status is not None:
            # 开启时：无已加载模型立即切 power-saver，有则切 balanced
            models = client.models()
            ms = parse_model_state(models)
            states = [m["state"] for m in parse_models_list(models)]
            target = (
                _auto_target(states)
                if ms["router_mode"]
                else ("balanced" if states and states[0] == "loaded" else "power-saver")
            )
            _apply_auto_power(target, power_status["available"])
        if power_status is not None:
            power_status["auto_power_saver"] = auto_power_saver
        return jsonify({"ok": True, "enabled": auto_power_saver, "power": power_status})

    @app.post("/api/power/suspend")
    def api_power_suspend():
        """S3 整机休眠：先安全卸载已加载模型释放显存，再延迟触发休眠。"""
        nonlocal unload_cooldown_until
        ps = get_power_status()
        if ps is None or not ps.get("can_suspend"):
            return jsonify({"ok": False, "error": "当前环境不支持整机休眠（S3）"}), 400
        models = client.models()
        loaded = [m for m in parse_models_list(models) if m["state"] == "loaded" and m["id"]]
        if loaded:
            with model_lock:
                # 先进入冷却期：卸载期间 status 轮询不再请求 /slots /metrics
                unload_cooldown_until = time.monotonic() + UNLOAD_COOLDOWN_SEC
                for m in loaded:
                    client.unload_model(m["id"])
        trigger_suspend(0.8)
        return jsonify({"ok": True})

    return app
