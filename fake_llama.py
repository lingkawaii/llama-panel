"""测试用假 llama-server：提供 /health /v1/models /slots /v1/chat/completions，/metrics 返回 501。

--router 模式：模拟 llama-server Router 模式
（--models-preset ... --models-autoload --sleep-idle-seconds 600）：
- /v1/models 的 data[0] 带 status（value/args/preset），仅 loaded 时带 meta；初始状态 unloaded
- /slots 与 /metrics 必须带 ?model=（否则 400）；unloaded 期间被调用会记录误唤醒计数并模拟 autoload 唤醒
- POST /models/load 与 /models/unload 切换状态，返回 {"success": true}
- /v1/chat/completions 收到请求时自动从 unloaded 切为 loaded（模拟 autoload 唤醒）
- GET /debug 返回内部状态（state 与误唤醒计数），供测试断言
"""
import json
import os
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import parse_qs, urlparse

HERE = os.path.dirname(os.path.abspath(__file__))
ROUTER = "--router" in sys.argv[1:]
MODEL_ID = "qwen3.8-27b"
SLEEP_IDLE = 600

METRICS_TEXT = (
    "# HELP llamacpp:predicted_tokens_seconds Prompt processing speed\n"
    "llamacpp:predicted_tokens_seconds 58.4\n"
    "# HELP llamacpp:prompt_tokens_seconds Prompt processing speed\n"
    "llamacpp:prompt_tokens_seconds 820.0\n"
    "llamacpp:prompt_tokens_total 1234\n"
    "llamacpp:tokens_predicted_total 567\n"
)

CHAT_BODY = {
    "id": "chatcmpl-fake",
    "object": "chat.completion",
    "model": MODEL_ID if ROUTER else "fake-model",
    "choices": [{
        "index": 0,
        "message": {"role": "assistant", "content": "OK 1 2 3 4 5"},
        "finish_reason": "stop",
    }],
    "timings": {
        "prompt_n": 128,
        "prompt_per_second": 820.0,
        "predicted_n": 10,
        "predicted_per_second": 58.4,
        "draft_n": 40,
        "draft_n_accepted": 28,
    },
}


# 单线程 HTTPServer：模块级共享状态（每个连接是独立 handler 实例，类/实例属性不可靠）
STATE = {
    "value": "unloaded" if ROUTER else "loaded",
    "slots_misfires": 0,   # unloaded 期间 /slots 被调用次数（误唤醒）
    "metrics_misfires": 0,  # unloaded 期间 /metrics 被调用次数（误唤醒）
}


class H(BaseHTTPRequestHandler):

    def _send_json(self, code, obj):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_text(self, code, text, ctype="text/plain"):
        body = text.encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _model_param(self, query):
        return (query.get("model") or [None])[0]

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path
        query = parse_qs(parsed.query)
        if path == "/health":
            self._send_json(200, {"status": "ok"})
        elif path == "/v1/models":
            if ROUTER:
                item = {
                    "id": MODEL_ID,
                    "object": "model",
                    "status": {
                        "value": STATE["value"],
                        "args": ["--models-preset", "models.ini",
                                 "--models-autoload",
                                 "--sleep-idle-seconds", str(SLEEP_IDLE)],
                        "preset": "models.ini",
                    },
                }
                if STATE["value"] == "loaded":
                    item["meta"] = {"n_ctx": 32768, "n_params": 27000000000,
                                   "size": 17000000000, "ftype": "Q4_K_M"}
                self._send_json(200, {"data": [item]})
            else:
                self._send_json(200, {"data": [{"id": "fake-model", "object": "model"}]})
        elif path == "/slots":
            if ROUTER:
                if self._model_param(query) is None:
                    self._send_json(400, {"error": "missing required parameter: model"})
                    return
                if STATE["value"] != "loaded":
                    # 模拟 --models-autoload：unloaded 时被查询即误唤醒
                    STATE["slots_misfires"] += 1
                    STATE["value"] = "loaded"
            self._send_json(200, [{"id": 0, "is_processing": False, "n_ctx": 4096,
                                   "n_prompt_tokens": 123}])
        elif path == "/metrics":
            if ROUTER:
                if self._model_param(query) is None:
                    self._send_json(400, {"error": "missing required parameter: model"})
                    return
                if STATE["value"] != "loaded":
                    STATE["metrics_misfires"] += 1
                    STATE["value"] = "loaded"
                self._send_text(200, METRICS_TEXT)
            else:
                self.send_response(501)
                self.end_headers()
        elif path == "/debug" and ROUTER:
            self._send_json(200, {"state": STATE["value"],
                                  "slots_misfires": STATE["slots_misfires"],
                                  "metrics_misfires": STATE["metrics_misfires"]})
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length)
        if self.path == "/v1/chat/completions":
            if ROUTER and STATE["value"] == "unloaded":
                STATE["value"] = "loaded"  # 模拟 --models-autoload 唤醒
            self._send_json(200, CHAT_BODY)
        elif ROUTER and self.path in ("/models/load", "/models/unload"):
            try:
                payload = json.loads(raw) if raw else {}
            except ValueError:
                payload = {}
            if not isinstance(payload, dict) or not payload.get("model"):
                self._send_json(400, {"success": False, "error": "missing model"})
                return
            STATE["value"] = "loaded" if self.path == "/models/load" else "unloaded"
            self._send_json(200, {"success": True})
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, *a):
        pass


server = HTTPServer(("127.0.0.1", 8080), H)
with open(os.path.join(HERE, "fake_llama.pid"), "w") as f:
    f.write(str(os.getpid()))
print("fake server ready, pid", os.getpid(), "router", ROUTER, flush=True)
server.serve_forever()
