"""端到端测试：外部进程识别 + 启停/重启 + GPU 字段与解析单测。"""
import json
import os
import socket
import subprocess
import sys
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


def main():
    test_gpu_parsers()
    test_parse_systemd_uptime()

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

        print("\nALL E2E CHECKS PASSED")
    finally:
        for proc in (panel, fake):
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
