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

    # 多模型 Router：优先选第一个 loaded（第 2 个模型），其次 loading
    multi = {"data": [
        {"id": "model-a", "status": {"value": "unloaded", "args": [], "preset": "m.ini"}},
        {"id": "model-b", "status": {"value": "loaded",
                                     "args": ["--sleep-idle-seconds", "600"],
                                     "preset": "m.ini"}},
    ]}
    r = parse_model_state(multi)
    assert r["router_mode"] is True
    assert r["model_id"] == "model-b", r
    assert r["state"] == "loaded", r
    assert r["sleep_idle_seconds"] == 600, r

    # 无 loaded 但有 loading → 选 loading
    multi["data"][1]["status"]["value"] = "loading"
    assert parse_model_state(multi)["model_id"] == "model-b"
    assert parse_model_state(multi)["state"] == "loading"

    # 全 unloaded → 兜底 data[0]
    multi["data"][1]["status"]["value"] = "unloaded"
    r = parse_model_state(multi)
    assert r["model_id"] == "model-a" and r["state"] == "unloaded", r
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


def test_parse_models_list():
    """纯函数单测：parse_models_list（多模型摘要 + modalities 默认值）。"""
    from llama_panel.client import parse_models_list

    # 非 Router 单模型：state loaded，modalities 默认 ["text"]
    r = parse_models_list({"data": [{"id": "fake-model"}]})
    assert r == [{"id": "fake-model", "state": "loaded",
                  "sleep_idle_seconds": None, "meta": {}, "modalities": ["text"]}], r

    # Router 多模型：state 来自 status.value，sleep_idle 来自 args
    payload = {"data": [
        {"id": "a", "status": {"value": "unloaded",
                               "args": ["--sleep-idle-seconds", "600"],
                               "preset": "m.ini"}},
        {"id": "b", "status": {"value": "loaded",
                               "args": ["--sleep-idle-seconds", "600"],
                               "preset": "m.ini"},
         "meta": {"n_ctx": 4096,
                  "architecture": {"input_modalities": ["text", "image"]}}},
    ]}
    r = parse_models_list(payload)
    assert len(r) == 2, r
    assert r[0]["id"] == "a" and r[0]["state"] == "unloaded", r
    assert r[0]["sleep_idle_seconds"] == 600, r
    assert r[0]["modalities"] == ["text"], r
    assert r[1]["id"] == "b" and r[1]["state"] == "loaded", r
    assert r[1]["modalities"] == ["text", "image"], r
    assert r[1]["meta"]["n_ctx"] == 4096, r

    # 空 / 非法输入
    assert parse_models_list(None) == []
    assert parse_models_list({}) == []
    assert parse_models_list({"data": []}) == []
    print("parse_models_list 单测通过")


def test_update_sleep_idle_in_ini():
    """纯函数单测：update_sleep_idle_in_ini（改值/关闭/无 [*] 节插入）。"""
    from llama_panel.preset import update_sleep_idle_in_ini

    # 已有 [*] 节：原地替换，保留其余行
    ini = "version = 1\n\n[*]\nsleep-idle-seconds = 600\nmodel-a = /path/a.gguf\n"
    out = update_sleep_idle_in_ini(ini, 300)
    assert "sleep-idle-seconds = 300" in out, out
    assert "sleep-idle-seconds = 600" not in out, out
    assert "model-a = /path/a.gguf" in out, out

    # <= 0 → 写入 -1（关闭自动休眠）
    out = update_sleep_idle_in_ini(ini, 0)
    assert "sleep-idle-seconds = -1" in out, out

    # [*] 节内无该行：追加到节末尾
    ini_b = "version = 1\n\n[*]\nmodel-a = /path/a.gguf\n"
    out = update_sleep_idle_in_ini(ini_b, 900)
    assert "sleep-idle-seconds = 900" in out, out
    assert "model-a = /path/a.gguf" in out, out

    # 无 [*] 节：在 version = 1 行后插入，保留其余节
    ini2 = "version = 1\n\n[other]\nfoo = bar\n"
    out = update_sleep_idle_in_ini(ini2, 900)
    assert out.index("[*]") > out.index("version = 1"), out
    assert "sleep-idle-seconds = 900" in out, out
    assert "[other]" in out and "foo = bar" in out, out

    # 无 version 行：插入到文件开头
    ini3 = "[other]\nfoo = bar\n"
    out = update_sleep_idle_in_ini(ini3, 120)
    assert out.startswith("[*]"), out
    assert "sleep-idle-seconds = 120" in out, out
    print("update_sleep_idle_in_ini 单测通过")


def test_parse_power_profiles():
    """纯函数单测：parse_power_profiles（daemon 块 / 简单列表 / 非法文本）。"""
    from llama_panel.power import parse_power_profiles

    # daemon 块格式：激活项行首带 *
    text = (
        "  power-saver:\n"
        "  CpuDriver: intel_pstate\n"
        "* balanced:\n"
        "  CpuDriver: intel_pstate\n"
        "  performance:\n"
    )
    current, available = parse_power_profiles(text)
    assert current == "balanced", (current, available)
    assert set(available) == {"power-saver", "balanced", "performance"}, available

    # 两档位块
    cur2, avail2 = parse_power_profiles("* performance:\n  x: y\n  power-saver:\n")
    assert cur2 == "performance", (cur2, avail2)
    assert set(avail2) == {"performance", "power-saver"}, avail2

    # 简单列表格式：- name / - * name
    cur3, avail3 = parse_power_profiles("- power-saver\n- * balanced\n- performance\n")
    assert cur3 == "balanced", (cur3, avail3)
    assert set(avail3) == {"power-saver", "balanced", "performance"}, avail3

    # 非法文本（无识别到的档位）
    assert parse_power_profiles("no profiles here") == (None, [])
    assert parse_power_profiles("") == (None, [])
    print("parse_power_profiles 单测通过")


def test_preset_sleep_idle_e2e():
    """单测：/api/preset/sleep-idle（在线修改 models.ini 预设文件）。"""
    from llama_panel.app import create_app

    tmp = os.path.join(HERE, "test_preset.ini")
    with open(tmp, "w", encoding="utf-8") as f:
        f.write("version = 1\n\n[*]\nsleep-idle-seconds = 600\nmodel = /path/a.gguf\n")
    try:
        app = create_app("http://127.0.0.1:9", ["fake"],
                         log_path=os.path.join(HERE, "test_preset.log"),
                         models_preset=tmp)
        c = app.test_client()

        # 正常修改 → 200，文件更新
        r = c.post("/api/preset/sleep-idle", json={"seconds": 300, "restart": False})
        assert r.status_code == 200 and r.get_json()["ok"] is True, r.get_json()
        with open(tmp, encoding="utf-8") as f:
            assert "sleep-idle-seconds = 300" in f.read()

        # 0 → 按 -1 处理（关闭）
        r = c.post("/api/preset/sleep-idle", json={"seconds": 0, "restart": False})
        assert r.status_code == 200 and r.get_json()["seconds"] == -1, r.get_json()
        with open(tmp, encoding="utf-8") as f:
            assert "sleep-idle-seconds = -1" in f.read()

        # 越界 / 布尔 → 400
        r = c.post("/api/preset/sleep-idle", json={"seconds": 30, "restart": False})
        assert r.status_code == 400, r.status_code
        r = c.post("/api/preset/sleep-idle", json={"seconds": 99999, "restart": False})
        assert r.status_code == 400, r.status_code
        r = c.post("/api/preset/sleep-idle", json={"seconds": True, "restart": False})
        assert r.status_code == 400, r.status_code
    finally:
        for f in (tmp, os.path.join(HERE, "test_preset.log")):
            try:
                os.remove(f)
            except OSError:
                pass

    # 未配置 --models-preset → 400
    app2 = create_app("http://127.0.0.1:9", ["fake"],
                      log_path=os.path.join(HERE, "test_preset2.log"))
    c2 = app2.test_client()
    r = c2.post("/api/preset/sleep-idle", json={"seconds": 300, "restart": False})
    assert r.status_code == 400, r.status_code
    try:
        os.remove(os.path.join(HERE, "test_preset2.log"))
    except OSError:
        pass
    print("preset sleep-idle 路由单测通过")


def test_power_auto_e2e():
    """单测：/api/power/auto（模型休眠自动切省电联动）。"""
    import llama_panel.app as appmod
    from llama_panel.app import create_app
    from llama_panel.client import LlamaClient

    calls = []
    orig = (appmod.get_power_status, appmod.set_power_profile, LlamaClient.models)
    try:
        appmod.get_power_status = lambda: {
            "current": "performance",
            "available": ["power-saver", "balanced", "performance"],
            "can_suspend": False,
        }
        appmod.set_power_profile = lambda profile: (
            calls.append(profile), {"ok": True, "profile": profile})[1]
        # 无已加载模型 → 开启时应切 power-saver
        LlamaClient.models = lambda self: {"data": [
            {"id": "model-a", "status": {"value": "unloaded",
                                         "args": [], "preset": "m.ini"}},
        ]}
        app = create_app("http://127.0.0.1:9", ["fake"],
                         log_path=os.path.join(HERE, "test_power.log"))
        c = app.test_client()

        r = c.post("/api/power/auto", json={"enabled": True})
        assert r.status_code == 200 and r.get_json()["ok"] is True, r.get_json()
        assert r.get_json()["enabled"] is True, r.get_json()
        assert calls == ["power-saver"], calls

        # 关闭 → ok，enabled False
        r = c.post("/api/power/auto", json={"enabled": False})
        assert r.status_code == 200 and r.get_json()["enabled"] is False, r.get_json()

        # enabled 非布尔 → 400
        r = c.post("/api/power/auto", json={"enabled": "yes"})
        assert r.status_code == 400, r.status_code
    finally:
        appmod.get_power_status, appmod.set_power_profile, LlamaClient.models = orig
        try:
            os.remove(os.path.join(HERE, "test_power.log"))
        except OSError:
            pass
    print("power auto 路由单测通过")


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


def test_router_idle_unload(fake2):
    """空闲自动卸载：Router 模式下 /api/status 轮询不重置空闲计时器。

    杀掉传入的单模型 fake，启动 sleep-idle 2s 的 router fake，
    返回新 fake 进程供 main() 回收。
    - 无真实活动：持续轮询超过 sleep_idle 后面板自动卸载，且不误唤醒；
    - token 增长：重置空闲计时，不卸载。
    """
    if fake2 is not None:
        fake2.kill()
        try:
            fake2.wait(timeout=5)
        except (subprocess.TimeoutExpired, OSError):
            pass
    wait_port_free(FAKE_PORT)
    if os.path.exists(PID_FILE):
        os.remove(PID_FILE)
    fake_log = open(os.path.join(HERE, "test_fake_idle.log"), "ab")
    fake4 = subprocess.Popen(
        [PY, "fake_llama.py", "--router", "--sleep-idle", "2"],
        cwd=HERE, stdout=fake_log, stderr=fake_log)
    wait_pid_file(what="idle fake pid")
    print("12) idle-unload fake server 已启动")

    # 12a) 加载模型；首次 status 观测建立空闲计时基准，
    #      此后纯轮询不算活动：超过 sleep_idle（2s）面板应自动卸载
    code, d = post_raw("/api/model/load")
    assert code == 200 and d.get("ok") is True, f"load 应成功: {code} {d}"
    s = wait_status(
        lambda d: (d["server"].get("model_state") or {}).get("state") == "loaded",
        "idle test: loaded",
    )
    assert s["server"]["model_state"]["sleep_idle_seconds"] == 2, s["server"]["model_state"]
    assert s["slots"] and s["metrics"], s
    s = wait_status(
        lambda d: (d["server"].get("model_state") or {}).get("state") == "unloaded",
        "idle auto-unload",
        timeout=20,
    )
    assert s["slots"] == [] and s["metrics"] is None, s
    dbg = fake_get("/debug")
    assert dbg["state"] == "unloaded", dbg
    assert dbg["slots_misfires"] == 0 and dbg["metrics_misfires"] == 0, \
        f"卸载过程误唤醒: {dbg}"
    print("12a) 空闲自动卸载 ok:", dbg)

    # 12b) token 增长重置空闲计时：重新加载后每次轮询前递增 token 累计值，
    #      全程（> sleep_idle）模型不得被卸载
    code, d = post_raw("/api/model/load")
    assert code == 200 and d.get("ok") is True, f"load 应成功: {code} {d}"
    wait_status(
        lambda d: (d["server"].get("model_state") or {}).get("state") == "loaded",
        "idle test: reloaded",
    )
    prompt = 2000
    for _ in range(5):
        fake_get(f"/debug/metrics?prompt={prompt}&predicted=600")
        prompt += 100
        s = get("/api/status")
        assert (s["server"]["model_state"] or {}).get("state") == "loaded", \
            f"token 增长应重置空闲计时，模型不应被卸载: {s['server']['model_state']}"
    time.sleep(1.0)
    dbg = fake_get("/debug")
    assert dbg["state"] == "loaded", f"token 增长期间模型应仍 loaded: {dbg}"
    assert dbg["slots_misfires"] == 0 and dbg["metrics_misfires"] == 0, dbg
    print("12b) token 增长不卸载 ok:", dbg)
    return fake4


def test_multi_model_e2e(fake2):
    """多模型 Router 端到端：每模型独立 load/unload，status 指向已加载模型。

    杀掉传入的 router fake，启动双模型 fake（model-a / model-b），
    返回新 fake 进程供 main() 的 finally 回收。
    """
    if fake2 is not None:
        fake2.kill()
        try:
            fake2.wait(timeout=5)
        except (subprocess.TimeoutExpired, OSError):
            pass
    wait_port_free(FAKE_PORT)
    if os.path.exists(PID_FILE):
        os.remove(PID_FILE)
    fake_log = open(os.path.join(HERE, "test_fake_mm.log"), "ab")
    fake3 = subprocess.Popen(
        [PY, "fake_llama.py", "--router", "--model", "model-a", "--model", "model-b"],
        cwd=HERE, stdout=fake_log, stderr=fake_log)
    wait_pid_file(what="multi-model fake pid")
    print("11) 多模型 fake server 已启动")

    # 11a) models_list 两项，均 unloaded；不触发 /slots /metrics
    s = wait_status(
        lambda d: len((d["server"].get("models_list") or [])) == 2
        and (d["server"].get("model_state") or {}).get("state") == "unloaded",
        "multi-model initial status",
    )
    ml = s["server"]["models_list"]
    assert {m["id"] for m in ml} == {"model-a", "model-b"}, ml
    assert all(m["state"] == "unloaded" for m in ml), ml
    assert s["slots"] == [] and s["metrics"] is None, s
    print("11a) 初始双模型 unloaded ok")

    # 11b) 加载 model-b → status 指向 model-b，slots/metrics 非空，无误唤醒
    code, d = post_json("/api/model/load", {"model": "model-b"})
    assert code == 200 and d.get("ok") is True, f"load model-b 应成功: {code} {d}"
    s = wait_status(
        lambda d: (d["server"].get("model_state") or {}).get("model_id") == "model-b"
        and (d["server"].get("model_state") or {}).get("state") == "loaded",
        "model-b loaded",
    )
    ms = s["server"]["model_state"]
    assert ms["model_id"] == "model-b", ms
    assert ms["state"] == "loaded", ms
    assert s["slots"], f"loaded 应返回 slots: {s['slots']}"
    assert s["metrics"], f"loaded 应返回 metrics: {s['metrics']}"
    by_id = {m["id"]: m for m in s["server"]["models_list"]}
    assert by_id["model-b"]["state"] == "loaded", by_id
    assert by_id["model-a"]["state"] == "unloaded", by_id
    dbg = fake_get("/debug")
    assert dbg["models"]["model-b"] == "loaded" and dbg["models"]["model-a"] == "unloaded", dbg
    assert dbg["slots_misfires"] == 0 and dbg["metrics_misfires"] == 0, dbg
    print("11b) model-b loaded ok:", ms)

    # 11c) 卸载 model-b → 回落到 model-a / unloaded
    code, d = post_json("/api/model/unload", {"model": "model-b"})
    assert code == 200 and d.get("ok") is True, f"unload model-b 应成功: {code} {d}"
    s = wait_status(
        lambda d: (d["server"].get("model_state") or {}).get("state") == "unloaded",
        "model-b unloaded",
    )
    ms = s["server"]["model_state"]
    assert ms["model_id"] == "model-a", ms
    assert ms["state"] == "unloaded", ms
    assert s["slots"] == [] and s["metrics"] is None, s
    dbg = fake_get("/debug")
    assert dbg["models"]["model-b"] == "unloaded", dbg
    print("11c) model-b unloaded ok:", ms)
    return fake3


def test_parse_slot_timings():
    """纯函数单测：parse_slot_timings_from_logs + enrich_slots。

    覆盖：推理中 next_token[0].n_decoded 优先、空闲时日志回填 n_decoded、
    速度指标回填、兼容旧版 speed 不覆盖、真实实时行（n_gen/tg）解析、
    enrich_slots 不就地污染输入 speed、timings_cache 在日志被刷屏后保留速度。
    """
    from llama_panel.client import (
        parse_slot_timings_from_logs, enrich_slots
    )

    log = [
        "slot print_timing: id  0 | task 201 | task 332 tg = 110.11 t/s, tg_3s = 110.44 t/s",
        "slot print_timing: id  0 | task 201 | prompt eval time =      93.72 ms /    22 tokens (    4.26 ms per token,   234.73 tokens per second)",
        "slot print_timing: id  0 | task 201 |        eval time =    3608.79 ms /   400 tokens (    9.04 ms per token,   110.56 tokens per second)",
        "slot print_timing: id  1 | task 305 | task 100 tg = 95.00 t/s, tg_3s = 94.00 t/s",
        "slot print_timing: id  1 | task 305 | prompt eval time =     200.00 ms /    50 tokens (    4.00 ms per token,   250.00 tokens per second)",
        "slot print_timing: id  1 | task 305 |        eval time =    1052.63 ms /   100 tokens (   10.53 ms per token,   100.00 tokens per second)",
    ]
    timings = parse_slot_timings_from_logs(log)
    assert timings == {
        (0, 201): {"decoded_n": 400, "decoded_n_per_second": 110.56,
                   "prompt_n": 22, "prompt_n_per_second": 234.73},
        (1, 305): {"decoded_n": 100, "decoded_n_per_second": 100.00,
                   "prompt_n": 50, "prompt_n_per_second": 250.00},
    }
    # 空日志
    assert parse_slot_timings_from_logs([]) == {}
    # 空 slots
    assert enrich_slots([]) == []
    # 推理中：next_token[0].n_decoded 优先于日志
    slots = [
        {"id": 0, "is_processing": True, "n_ctx": 4096,
         "n_prompt_tokens": 22, "next_token": [{"n_decoded": 50}], "id_task": 201},
    ]
    enriched = enrich_slots(slots, log)
    assert enriched[0]["n_decoded"] == 50
    assert enriched[0]["speed"]["prompt_n_per_second"] == 234.73
    assert enriched[0]["speed"]["decoded_n_per_second"] == 110.56
    # 空闲时 n_decoded=0，从日志回填
    slots2 = [
        {"id": 1, "is_processing": False, "n_ctx": 4096,
         "n_prompt_tokens": 50, "n_decoded": 0, "id_task": 305},
    ]
    enriched2 = enrich_slots(slots2, log)
    assert enriched2[0]["n_decoded"] == 100
    assert enriched2[0]["speed"]["prompt_n_per_second"] == 250.00
    assert enriched2[0]["speed"]["decoded_n_per_second"] == 100.00
    # 兼容旧版：上游已有 speed 不覆盖
    slots3 = [
        {"id": 2, "is_processing": False, "n_ctx": 4096,
         "n_prompt_tokens": 10, "n_decoded": 5, "id_task": 0,
         "speed": {"prompt_n_per_second": 999.0, "decoded_n_per_second": 999.0}},
    ]
    enriched3 = enrich_slots(slots3, [])
    assert enriched3[0]["speed"]["prompt_n_per_second"] == 999.0
    assert enriched3[0]["speed"]["decoded_n_per_second"] == 999.0
    # 真实实时行（llama-server 推理进行中）解析：n_gen / tg
    real_line = ("[53521] 2.23.942.905 I slot print_timing: id  0 | task 201 | "
                 "n_gen =    332, tg = 110.11 t/s, tg_3s = 110.44 t/s")
    real = parse_slot_timings_from_logs([real_line])
    assert real[(0, 201)]["decoded_n"] == 332
    assert real[(0, 201)]["decoded_n_per_second"] == 110.11
    # enrich_slots 不就地污染输入 slot["speed"]
    input_slot = {
        "id": 0, "is_processing": False, "n_ctx": 4096,
        "n_prompt_tokens": 22, "n_decoded": 0, "id_task": 201,
        "speed": {"prompt_n_per_second": 1.0},
    }
    original_speed = dict(input_slot["speed"])
    out = enrich_slots([input_slot], [real_line])
    assert input_slot["speed"] == original_speed  # 入参未被就地修改
    assert "decoded_n_per_second" not in input_slot["speed"]
    assert out[0]["n_decoded"] == 332
    assert out[0]["speed"]["prompt_n_per_second"] == 1.0  # 已有值不覆盖
    assert out[0]["speed"]["decoded_n_per_second"] == 110.11  # 新值回填
    # 日志被无关行刷满后，仍可从 timings_cache 保留 (0, 201) 的速度与 decoded_n
    cache: dict[tuple[int, int], dict] = {}
    enrich_slots([{"id": 0, "id_task": 201, "n_decoded": 0}], [real_line], cache)
    flushed = [
        f"proxy_request: slot {i % 4} received chat request" for i in range(300)
    ]
    kept = enrich_slots([{"id": 0, "id_task": 201, "n_decoded": 0}], flushed, cache)
    assert kept[0]["n_decoded"] == 332
    assert kept[0]["speed"]["decoded_n_per_second"] == 110.11
    print("C) slot timings parsing & enrichment unit tests passed")
def main():
    test_gpu_parsers()
    test_parse_systemd_uptime()
    test_parse_model_state()
    test_parse_models_list()
    test_update_sleep_idle_in_ini()
    test_parse_power_profiles()
    test_parse_slot_timings()
    test_preset_sleep_idle_e2e()
    test_power_auto_e2e()

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
    fake3 = None
    fake4 = None
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

        # 12) 空闲自动卸载 / 真实活动追踪
        fake4 = test_router_idle_unload(fake2)

        # 11) 多模型 Router 切换
        fake3 = test_multi_model_e2e(fake4)

        print("\nALL E2E CHECKS PASSED")
    finally:
        for proc in (panel, fake3, fake2, fake4, fake):
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
        for f in (PID_FILE, os.path.join(HERE, "test_fake.log"),
                  os.path.join(HERE, "test_fake_idle.log"), os.path.join(HERE, "test_panel.log")):
            try:
                os.remove(f)
            except OSError:
                pass


if __name__ == "__main__":
    main()
