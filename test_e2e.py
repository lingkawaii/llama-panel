"""端到端测试：外部进程识别 + 启停/重启 + GPU 字段与解析单测。"""
import json
import os
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable
PANEL_PORT = 5001
FAKE_PORT = 8080
PID_FILE = os.path.join(HERE, "fake_llama.pid")


def get(path):
    # 面板单线程 Flask：server 死后各 client 调用走满 3s 超时，
    # 一次 status 最坏 ~15s，请求超时必须大于它，否则会排队堆积
    with urllib.request.urlopen(f"http://127.0.0.1:{PANEL_PORT}{path}", timeout=25) as r:
        return json.loads(r.read())


def post(path):
    req = urllib.request.Request(f"http://127.0.0.1:{PANEL_PORT}{path}", method="POST")
    with urllib.request.urlopen(req, timeout=25) as r:
        return json.loads(r.read())


def post_raw(path):
    """POST，返回 (状态码, JSON 体)；非 2xx 不抛异常。"""
    req = urllib.request.Request(f"http://127.0.0.1:{PANEL_PORT}{path}", method="POST")
    try:
        with urllib.request.urlopen(req, timeout=25) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def post_json(path, obj):
    """POST JSON body，返回 (状态码, JSON 体)；非 2xx 不抛异常。"""
    req = urllib.request.Request(
        f"http://127.0.0.1:{PANEL_PORT}{path}",
        data=json.dumps(obj).encode(),
        headers={"Content-Type": "application/json"}, method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=25) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def fake_get(path):
    """直接请求 fake server（绕过面板），用于读取 /debug 等测试钩子。"""
    with urllib.request.urlopen(f"http://127.0.0.1:{FAKE_PORT}{path}", timeout=10) as r:
        return json.loads(r.read())


def wait_status(predicate, what, timeout=40):
    deadline = time.time() + timeout
    last = None
    last_err = None
    while time.time() < deadline:
        try:
            last = get("/api/status")
            if predicate(last):
                return last
        except Exception as e:
            last_err = f"{type(e).__name__}: {e}"
        time.sleep(0.3)
    raise AssertionError(f"timeout waiting for {what}; last={last}; last_err={last_err}")


def port_in_use(port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(("127.0.0.1", port)) == 0


def wait_port_free(port, timeout=10):
    deadline = time.time() + timeout
    while port_in_use(port):
        assert time.time() < deadline, f"端口 {port} 未释放"
        time.sleep(0.2)


def read_pid_file():
    if not os.path.exists(PID_FILE):
        return None
    with open(PID_FILE) as f:
        t = f.read().strip()
    return int(t) if t.isdigit() else None


def wait_pid_file(expect=None, expect_diff=None, what="", timeout=15):
    deadline = time.time() + timeout
    while time.time() < deadline:
        pid = read_pid_file()
        if pid is not None and (expect is None or pid == expect) \
                and (expect_diff is None or pid != expect_diff):
            return pid
        time.sleep(0.1)
    raise AssertionError(f"timeout waiting for {what}; pid={read_pid_file()}")


def test_gpu_parsers():
    """纯函数单测：rocm-smi JSON（单卡/多卡/字段变体）与 nvidia-smi CSV。"""
    from llama_panel import gpu

    rocm_single = json.dumps({
        "Card series": "Radeon RX 7900 XTX",
        "GPU use (%)": "12",
        "VRAM Total Memory (B)": "24746393600",
        "VRAM Total Used Memory (B)": "12373196800",
        "GPU memory use (%)": "50",
        "Temperature (C)": "65",
        "Graphics Package Power (W)": "180.5",
    })
    s = gpu.parse_rocm_smi_json(rocm_single)
    assert s is not None, "单卡 rocm JSON 应解析成功"
    assert s["name"] == "Radeon RX 7900 XTX"
    assert s["vram_used_bytes"] == 12373196800
    assert s["vram_total_bytes"] == 24746393600
    assert s["vram_percent"] == 50.0
    assert s["gpu_util_percent"] == 12.0
    assert s["temp_c"] == 65.0
    assert s["power_w"] == 180.5

    # 多卡：取第一张
    rocm_multi = json.dumps({
        "card0": json.loads(rocm_single),
        "card1": {
            "Card series": "Radeon RX 6600",
            "GPU use (%)": "0",
            "VRAM Total Memory (B)": "8053063680",
            "VRAM Total Used Memory (B)": "0",
            "GPU memory use (%)": "0",
            "Temperature (C)": "40",
            "Graphics Package Power (W)": "10",
        },
    })
    s = gpu.parse_rocm_smi_json(rocm_multi)
    assert s is not None and s["name"] == "Radeon RX 7900 XTX", "多卡应取第一张"

    # 字段变体：Sensor edge 温度 / Average 功耗 / 缺省百分比（由用量计算）
    rocm_variant = json.dumps({
        "Card series": "Radeon RX 9070",
        "GPU use (%)": "0",
        "VRAM Total Memory (B)": "1000",
        "VRAM Total Used Memory (B)": "100",
        "Temperature (Sensor edge) (C)": "55",
        "Average Graphics Package Power (W)": "10",
    })
    s = gpu.parse_rocm_smi_json(rocm_variant)
    assert s is not None
    assert s["temp_c"] == 55.0
    assert s["power_w"] == 10.0
    assert s["vram_percent"] == 10.0, "缺省百分比应由 used/total 计算"

    # 非法输入
    assert gpu.parse_rocm_smi_json("not json") is None
    assert gpu.parse_rocm_smi_json(json.dumps({"foo": "bar"})) is None
    assert gpu.parse_rocm_smi_json(json.dumps({"Card series": "x"})) is None

    nv = "NVIDIA GeForce RTX 4090, 8000, 24564, 95, 70, 300.5\n"
    s = gpu.parse_nvidia_smi_csv(nv)
    assert s is not None, "nvidia CSV 应解析成功"
    assert s["name"] == "NVIDIA GeForce RTX 4090"
    assert s["vram_used_bytes"] == 8000 * 1024 * 1024
    assert s["vram_total_bytes"] == 24564 * 1024 * 1024
    assert abs(s["vram_percent"] - 8000 / 24564 * 100) < 1e-6
    assert s["gpu_util_percent"] == 95.0
    assert s["temp_c"] == 70.0
    assert s["power_w"] == 300.5

    # 多行取第一行；非法输入
    assert gpu.parse_nvidia_smi_csv("a, 1, 2, 3, 4, 5\nb, 9, 9, 9, 9, 9\n")["name"] == "a"
    assert gpu.parse_nvidia_smi_csv("") is None
    assert gpu.parse_nvidia_smi_csv("a,b\n") is None
    assert gpu.parse_nvidia_smi_csv("a, x, y, z, w, v\n") is None

    # get_gpu_stats 注入 runner（不依赖本机 GPU 工具）
    def runner(cmd):
        if "nvidia-smi" in cmd[0]:
            return nv
        return rocm_single
    s = gpu.get_gpu_stats(runner=runner)
    assert s is not None and s["name"] in ("Radeon RX 7900 XTX", "NVIDIA GeForce RTX 4090")
    s = gpu.get_gpu_stats(runner=lambda cmd: None)
    assert s is None, "两个工具都无输出时应返回 None"
    print("A) gpu parser unit tests passed")


def test_parse_systemd_uptime():
    """纯函数单测：ActiveEnterTimestampMonotonic 微秒值 → 运行时长秒数。"""
    from llama_panel.app import parse_systemd_uptime

    # 固定 now_ts：1e6 微秒 = 1s，now=10s → 9s
    assert parse_systemd_uptime("1000000", now_ts=10.0) == 9.0
    # 空白 / 浮点字符串
    assert abs(parse_systemd_uptime(" 2000000.5 ", now_ts=10.0) - 8.0) < 1e-6
    # 无效输入
    assert parse_systemd_uptime("") is None
    assert parse_systemd_uptime("0") is None
    assert parse_systemd_uptime("abc") is None
    assert parse_systemd_uptime("-5") is None
    # 未来时间戳（结果 <= 0）
    assert parse_systemd_uptime("99999999999999", now_ts=10.0) is None
    # now_ts=None 走 time.monotonic()
    usec = int((time.monotonic() - 5) * 1e6)
    up = parse_systemd_uptime(str(usec))
    assert up is not None and 4.9 < up < 5.5, f"monotonic 分支异常: {up}"
    print("B) parse_systemd_uptime unit tests passed")


def test_probe_ok():
    """端到端：server 运行时 POST /api/probe 返回 200 + timings 解析。"""
    status, d = post_raw("/api/probe")
    assert status == 200 and d["ok"] is True, f"probe 应成功: {status} {d}"
    assert d["wall_ms"] > 0
    assert d["prompt_tps"] == 820.0
    assert d["gen_tps"] == 58.4
    assert d["prompt_n"] == 128
    assert d["predicted_n"] == 10
    assert abs(d["spec_accept_percent"] - 70.0) < 1e-6
    print("probe ok:", d)


def test_probe_down():
    """端到端：server 停止后 POST /api/probe 返回 502 + ok False。"""
    status, d = post_raw("/api/probe")
    assert status == 502 and d["ok"] is False, f"probe 应 502: {status} {d}"
    assert d.get("error"), "缺少 error 信息"
    print("probe down:", status, d)


def test_parse_model_state():
    """纯函数单测：parse_model_state（Router 模式感知）。"""
    from llama_panel.client import parse_model_state

    # None / 空 → 非 Router、无模型
    assert parse_model_state(None) == {
        "router_mode": False, "model_id": None, "state": "none",
        "sleep_idle_seconds": None,
    }
    assert parse_model_state({})["state"] == "none"
    assert parse_model_state({"data": []})["state"] == "none"

    # 非 Router 单模型：有 model_id → loaded
    assert parse_model_state({"data": [{"id": "fake-model"}]}) == {
        "router_mode": False, "model_id": "fake-model",
        "state": "loaded", "sleep_idle_seconds": None,
    }

    # Router 模式：value/args 解析
    payload = {"data": [{
        "id": "qwen3.8-27b",
        "status": {
            "value": "loaded",
            "args": ["--models-preset", "m.ini", "--models-autoload",
                     "--sleep-idle-seconds", "600"],
            "preset": "m.ini",
        },
        "meta": {"n_ctx": 32768},
    }]}
    r = parse_model_state(payload)
    assert r["router_mode"] is True
    assert r["model_id"] == "qwen3.8-27b"
    assert r["state"] == "loaded"
    assert r["sleep_idle_seconds"] == 600

    # unloaded / loading 状态（无 meta）
    payload["data"][0]["status"]["value"] = "unloaded"
    del payload["data"][0]["meta"]
    assert parse_model_state(payload)["state"] == "unloaded"
    payload["data"][0]["status"]["value"] = "loading"
    assert parse_model_state(payload)["state"] == "loading"

    # sleep_idle_seconds：preset 文本正则解析
    payload["data"][0]["status"]["args"] = []
    payload["data"][0]["status"]["preset"] = "m.ini (sleep-idle-seconds = 1200)"
    assert parse_model_state(payload)["sleep_idle_seconds"] == 1200

    # sleep_idle_seconds：未配置 / <= 0 / 缺值 / 非数字 → None
    status = payload["data"][0]["status"]
    status["args"] = ["--sleep-idle-seconds", "0"]
    status["preset"] = "m.ini"
    assert parse_model_state(payload)["sleep_idle_seconds"] is None
    status["args"] = ["--sleep-idle-seconds"]
    assert parse_model_state(payload)["sleep_idle_seconds"] is None
    status["args"] = ["--sleep-idle-seconds", "abc"]
    assert parse_model_state(payload)["sleep_idle_seconds"] is None
    status["args"] = ["--other", "5"]
    assert parse_model_state(payload)["sleep_idle_seconds"] is None

    # status 非 dict / 缺 value → 非 Router
    assert parse_model_state({"data": [{"id": "a", "status": "loaded"}]})["router_mode"] is False
    assert parse_model_state({"data": [{"id": "a", "status": {}}]})["router_mode"] is False
    assert parse_model_state({"data": [{"status": {"value": "loaded"}}]}) == {
        "router_mode": True, "model_id": None, "state": "loaded",
        "sleep_idle_seconds": None,
    }
    print("A) parse_model_state 单测通过")


def test_model_routes():
    """单测：/api/model/load|unload 无 model_id 返回 400，上游失败返回 502。"""
    from llama_panel.app import create_app
    from llama_panel.client import LlamaClient

    orig = (LlamaClient.models, LlamaClient.load_model, LlamaClient.unload_model)
    try:
        LlamaClient.models = lambda self: {"data": []}
        LlamaClient.load_model = lambda self, model, timeout=30.0: {"ok": True, "model": model}
        LlamaClient.unload_model = lambda self, model, timeout=15.0: {"ok": True, "model": model}
        app = create_app("http://127.0.0.1:9", ["fake"],
                         log_path=os.path.join(HERE, "test_model_routes.log"))
        c = app.test_client()

        # 无 body 且无 model_id → 400
        r = c.post("/api/model/load")
        assert r.status_code == 400, f"无 model_id 应 400: {r.status_code} {r.get_json()}"
        r = c.post("/api/model/unload")
        assert r.status_code == 400, f"无 model_id 应 400: {r.status_code} {r.get_json()}"

        # body 指定 model → 200
        r = c.post("/api/model/load", json={"model": "x"})
        assert r.status_code == 200 and r.get_json() == {"ok": True, "model": "x"}
        r = c.post("/api/model/unload", json={"model": "x"})
        assert r.status_code == 200 and r.get_json() == {"ok": True, "model": "x"}

        # 上游失败 → 502
        LlamaClient.models = lambda self: {"data": [{"id": "x"}]}
        LlamaClient.load_model = lambda self, model, timeout=30.0: {"ok": False, "error": "boom"}
        r = c.post("/api/model/load")
        assert r.status_code == 502 and r.get_json()["ok"] is False
        LlamaClient.unload_model = lambda self, model, timeout=15.0: {"ok": False, "error": "boom"}
        r = c.post("/api/model/unload")
        assert r.status_code == 502 and r.get_json()["ok"] is False
    finally:
        LlamaClient.models, LlamaClient.load_model, LlamaClient.unload_model = orig
    print("B) model load/unload 路由单测通过")


def test_router_e2e():
    """Router 模式端到端：unloaded 不误唤醒，load/unload API 可用。"""
    # 9) 启动 --router 模式 fake server（初始状态 unloaded）
    if os.path.exists(PID_FILE):
        os.remove(PID_FILE)
    fake_log = open(os.path.join(HERE, "test_fake.log"), "ab")
    fake2 = subprocess.Popen([PY, "fake_llama.py", "--router"], cwd=HERE,
                             stdout=fake_log, stderr=fake_log)
    router_pid = wait_pid_file(what="router fake pid")
    print("9) router fake server 已启动, real pid:", router_pid)

    # 9a) unloaded：/api/status 必须返回 slots=[] metrics=None
    s = wait_status(
        lambda d: (d["server"].get("model_state") or {}).get("router_mode") is True
        and (d["server"].get("model_state") or {}).get("state") == "unloaded",
        "router unloaded status",
    )
    ms = s["server"]["model_state"]
    assert ms["model_id"] == "qwen3.8-27b", ms
    assert ms["sleep_idle_seconds"] == 600, ms
    assert s["slots"] == [], f"unloaded 不应请求 /slots: {s['slots']}"
    assert s["metrics"] is None, f"unloaded 不应请求 /metrics: {s['metrics']}"
    print("9a) unloaded status ok:", ms)

    # 9b) 多次轮询确认绝不误触发 autoload
    for _ in range(3):
        get("/api/status")
        time.sleep(0.2)
    dbg = fake_get("/debug")
    assert dbg["state"] == "unloaded", f"unloaded 被误唤醒: {dbg}"
    assert dbg["slots_misfires"] == 0 and dbg["metrics_misfires"] == 0, f"误触发 autoload: {dbg}"
    print("9b) 无误唤醒:", dbg)

    # 9c) POST /api/model/load（无 body，自动取 model_id）→ 200，随后 loaded
    code, d = post_raw("/api/model/load")
    assert code == 200 and d.get("ok") is True, f"load 应成功: {code} {d}"
    s = wait_status(
        lambda d: (d["server"].get("model_state") or {}).get("state") == "loaded",
        "router loaded",
    )
    assert s["slots"], f"loaded 应返回 slots: {s['slots']}"
    assert s["metrics"], f"loaded 应返回 metrics: {s['metrics']}"
    assert fake_get("/debug")["state"] == "loaded"
    print("9c) loaded ok:", s["server"]["model_state"])

    # 9d) POST /api/model/unload → 200，回到 unloaded
    code, d = post_raw("/api/model/unload")
    assert code == 200 and d.get("ok") is True, f"unload 应成功: {code} {d}"
    s = wait_status(
        lambda d: (d["server"].get("model_state") or {}).get("state") == "unloaded",
        "router unloaded after unload",
    )
    assert s["slots"] == [] and s["metrics"] is None, s
    dbg = fake_get("/debug")
    assert dbg["state"] == "unloaded", dbg
    print("9d) unloaded ok:", dbg)
    return fake2


def test_router_cooldown():
    """冷却期/并发防误唤醒：卸载后 /api/status 不得触发 autoload（冷却期 + 并发）。"""
    # 1) 确保 loaded
    post_raw("/api/model/load")
    wait_status(
        lambda d: (d["server"].get("model_state") or {}).get("state") == "loaded",
        "cooldown test: loaded",
    )

    # 2) 冷却期：卸载后立即模拟 stale loaded 视图（Router 状态滞后），
    #    /api/status 必须跳过 /slots /metrics
    code, d = post_raw("/api/model/unload")
    assert code == 200 and d.get("ok") is True, f"unload 应成功: {code} {d}"
    fake_get("/debug/stale_loaded")
    s = get("/api/status")
    ms = s["server"]["model_state"]
    assert ms["state"] == "loaded", f"stale 视图应报 loaded: {ms}"
    assert s["slots"] == [] and s["metrics"] is None, \
        f"冷却期内不应请求 /slots /metrics: slots={s['slots']} metrics={s['metrics']}"
    dbg = fake_get("/debug")
    assert dbg["state"] == "unloaded", f"冷却期内被误唤醒: {dbg}"
    assert dbg["slots_misfires"] == 0 and dbg["metrics_misfires"] == 0, f"误触发 autoload: {dbg}"
    fake_get("/debug/stale_loaded?clear=1")
    print("10a) 冷却期跳过 /slots /metrics:", dbg)

    # 3) 并发：重新加载后，8 线程各轮询 5 次 /api/status，主线程中途发起 unload
    post_raw("/api/model/load")
    wait_status(
        lambda d: (d["server"].get("model_state") or {}).get("state") == "loaded",
        "cooldown test: reloaded",
    )
    errors = []

    def poll():
        for _ in range(5):
            try:
                get("/api/status")
            except Exception as e:
                errors.append(repr(e))
            time.sleep(0.05)

    threads = [threading.Thread(target=poll) for _ in range(8)]
    for t in threads:
        t.start()
    time.sleep(0.2)
    t_unload = time.monotonic()
    code, d = post_raw("/api/model/unload")
    assert code == 200 and d.get("ok") is True, f"unload 应成功: {code} {d}"
    for t in threads:
        t.join()
    assert not errors, f"并发 status 请求出错: {errors}"
    wait_status(
        lambda d: (d["server"].get("model_state") or {}).get("state") == "unloaded",
        "cooldown test: unloaded",
    )
    dbg = fake_get("/debug")
    assert dbg["state"] == "unloaded", f"并发期间被误唤醒: {dbg}"
    assert dbg["slots_misfires"] == 0 and dbg["metrics_misfires"] == 0, f"误触发 autoload: {dbg}"
    print("10b) 并发 unload + status 无误唤醒:", dbg)

    # 4) 冷却期结束（自第 3 步 unload 起 5s）：status 恢复 /slots /metrics 查询
    #    （stale 视图仍报 loaded，查询会真实唤醒模型，属预期行为）
    elapsed = time.monotonic() - t_unload
    if elapsed < 5.5:
        time.sleep(5.5 - elapsed)
    fake_get("/debug/stale_loaded")
    s = get("/api/status")
    assert s["slots"], f"冷却期后应查询 /slots: {s['slots']}"
    assert s["metrics"], f"冷却期后应查询 /metrics: {s['metrics']}"
    fake_get("/debug/stale_loaded?clear=1")
    dbg = fake_get("/debug")
    assert dbg["state"] == "loaded", f"冷却期后查询应唤醒模型: {dbg}"
    print("10c) 冷却期后恢复指标查询:", dbg)


def main():
    test_gpu_parsers()
    test_parse_systemd_uptime()
    test_parse_model_state()
    test_model_routes()

    for p in (FAKE_PORT, PANEL_PORT):
        assert not port_in_use(p), f"端口 {p} 已被占用，先清理残留进程"
    if os.path.exists(PID_FILE):
        os.remove(PID_FILE)
    fake_log = open(os.path.join(HERE, "test_fake.log"), "wb")
    panel_log = open(os.path.join(HERE, "test_panel.log"), "wb")
    fake = subprocess.Popen([PY, "fake_llama.py"], cwd=HERE,
                            stdout=fake_log, stderr=fake_log)
    panel = subprocess.Popen(
        [PY, "main.py", "--port", str(PANEL_PORT),
         "--server-cmd", f"{PY} fake_llama.py"],
        cwd=HERE, stdout=panel_log, stderr=panel_log)
    fake2 = None
    try:
        # 0) 假 server 的真实 pid（本环境 Popen.pid 可能与子进程实际 pid 不同）
        deadline = time.time() + 10
        while not os.path.exists(PID_FILE):
            assert time.time() < deadline, "fake server 未启动"
            time.sleep(0.1)
        fake_pid = read_pid_file()
        print("0) fake server real pid:", fake_pid)

        # 1) 面板识别外部 server
        s = wait_status(
            lambda d: d["server"]["process"]["running"] and d["server"]["process"]["external"],
            "external detection",
        )
        p = s["server"]["process"]
        print("1) external detected:", p)
        assert p["pid"] == fake_pid, f"pid mismatch: panel={p['pid']} fake={fake_pid}"
        assert p["external"] is True
        assert p["running"] is True

        # 2) /api/status 必须包含 gpu 字段（无 GPU 工具时为 null）
        assert "gpu" in s, "status 缺少 gpu 字段"
        g = s["gpu"]
        assert g is None or isinstance(g, dict), f"gpu 字段类型异常: {g!r}"
        if g is not None:
            for k in ("name", "vram_used_bytes", "vram_total_bytes",
                      "vram_percent", "gpu_util_percent", "temp_c", "power_w"):
                assert k in g, f"gpu 缺少字段 {k}"
        print("2) status gpu field:", g)

        # 3) 停止外部 server
        r = post("/api/server/stop")
        print("3) stop response:", r)
        assert r["ok"] is True

        # 4) 面板报告未运行
        s2 = wait_status(
            lambda d: not d["server"]["process"]["running"],
            "server stopped",
        )
        print("4) after stop:", s2["server"]["process"])
        assert s2["server"]["process"]["running"] is False

        # 5) 端口释放（假 server 确实退出）
        wait_port_free(FAKE_PORT)
        print("5) fake server exited")

        # 6) 面板启动 server
        if os.path.exists(PID_FILE):
            os.remove(PID_FILE)
        r = post("/api/server/start")
        print("6) start response:", r)
        assert r["ok"] is True
        s3 = wait_status(
            lambda d: d["server"]["process"]["running"]
            and not d["server"]["process"]["external"],
            "panel-managed start",
        )
        assert s3["server"]["process"]["pid"] is not None
        managed_pid = wait_pid_file(what="managed start pid")
        old_panel_pid = s3["server"]["process"]["pid"]
        print("6) panel started server, real pid:", managed_pid)

        # 6b) server 运行时一键测速
        test_probe_ok()

        # 7) 重启：先停后启，实际进程 pid 必须变化
        r = post("/api/server/restart")
        print("7) restart response:", r)
        assert r["ok"] is True
        new_pid = wait_pid_file(expect_diff=managed_pid, what="restart pid change")
        s4 = wait_status(
            lambda d: d["server"]["process"]["running"]
            and d["server"]["process"]["pid"] != old_panel_pid,
            "restart status",
        )
        assert s4["server"]["process"]["external"] is False
        print("7) restarted, new real pid:", new_pid)

        # 8) 停止
        r = post("/api/server/stop")
        print("8) stop response:", r)
        assert r["ok"] is True
        s5 = wait_status(
            lambda d: not d["server"]["process"]["running"],
            "final stop",
        )
        assert s5["server"]["process"]["running"] is False
        wait_port_free(FAKE_PORT)
        print("8) stopped and port free")

        # 8b) server 停止后测速应 502
        test_probe_down()

        # 9) Router 模式端到端
        fake2 = test_router_e2e()

        # 10) 冷却期/并发防误唤醒
        test_router_cooldown()

        print("\nALL E2E CHECKS PASSED")
    finally:
        for proc in (panel, fake2, fake):
            if proc is None:
                continue
            try:
                proc.kill()
                proc.wait(timeout=5)
            except (subprocess.TimeoutExpired, OSError):
                pass
        # 清理可能残留的真实进程（本环境 Popen.kill 未必杀到实际子进程）
        for port in (FAKE_PORT, PANEL_PORT):
            if port_in_use(port):
                from llama_panel.process import find_pid_on_port, kill_pid
                ext = find_pid_on_port(port)
                if ext:
                    print("cleanup: killing leftover pid", ext, "on port", port)
                    kill_pid(ext)
        wait_port_free(FAKE_PORT, timeout=5)
        fake_log.close()
        panel_log.close()
        for f in (PID_FILE, os.path.join(HERE, "test_fake.log"), os.path.join(HERE, "test_panel.log")):
            try:
                os.remove(f)
            except OSError:
                pass


if __name__ == "__main__":
    main()
