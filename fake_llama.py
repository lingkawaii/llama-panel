"""测试用假 llama-server：提供 /health /v1/models /slots /v1/chat/completions，/metrics 返回 501。

--router 模式：模拟 llama-server Router 模式
（--models-preset ... --models-autoload --sleep-idle-seconds 600）：
- /v1/models 的 data 每项带 status（value/args/preset），仅 loaded 时带 meta；初始状态 unloaded
- 多模型：--model 可多次指定，每个模型独立 load/unload 状态（模拟 --models-max 1）
- /slots 与 /metrics 必须带 ?model=（否则 400）；unloaded 期间被调用会记录误唤醒计数并模拟 autoload 唤醒
- POST /models/load 与 /models/unload 切换对应模型状态（body {"model": id}），返回 {"success": true}
- /v1/chat/completions 收到请求时自动从 unloaded 切为 loaded（模拟 autoload 唤醒）
- GET /debug 返回内部状态（state=首个模型状态、models=每模型状态、误唤醒计数），供测试断言
- GET /debug/stale_loaded（带 ?clear=1 清除）：实际 unloaded 时 /v1/models
  仍报 loaded（模拟 Router 状态滞后），供冷却期测试使用
"""
import json
import os
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import parse_qs, urlparse

HERE = os.path.dirname(os.path.abspath(__file__))
ROUTER = "--router" in sys.argv[1:]

# 解析 --model（可多次）；未指定时默认单模型 qwen3.8-27b
MODELS: list[str] = []
_argv = sys.argv[1:]
_i = 0
while _i < len(_argv):
    if _argv[_i] == "--model" and _i + 1 < len(_argv):
        MODELS.append(_argv[_i + 1])
        _i += 2
    else:
        _i += 1
if not MODELS:
    MODELS = ["qwen3.8-27b"]
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
    "model": MODELS[0] if ROUTER else "fake-model",
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
    # 每模型独立状态：id -> "loaded" | "unloaded"
    "models": {m: ("unloaded" if ROUTER else "loaded") for m in MODELS},
    "slots_misfires": 0,   # unloaded 期间 /slots 被调用次数（误唤醒）
    "metrics_misfires": 0,  # unloaded 期间 /metrics 被调用次数（误唤醒）
    "stale_loaded": False,  # /v1/models 报 loaded 即使实际 unloaded（模拟 Router 状态滞后）
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
                data = []
                for m in MODELS:
                    value = "loaded" if STATE["stale_loaded"] else STATE["models"][m]
                    item = {
                        "id": m,
                        "object": "model",
                        "status": {
                            "value": value,
                            "args": ["--models-preset", "models.ini",
                                     "--models-autoload",
                                     "--sleep-idle-seconds", str(SLEEP_IDLE)],
                            "preset": "models.ini",
                        },
                    }
                    if value == "loaded":
                        item["meta"] = {"n_ctx": 32768, "n_params": 27000000000,
                                       "size": 17000000000, "ftype": "Q4_K_M"}
                    data.append(item)
                self._send_json(200, {"data": data})
            else:
                self._send_json(200, {"data": [{"id": "fake-model", "object": "model"}]})
        elif path == "/slots":
            if ROUTER:
                model = self._model_param(query)
                if model is None:
                    self._send_json(400, {"error": "missing required parameter: model"})
                    return
                if model not in STATE["models"]:
                    self._send_json(404, {"error": f"unknown model {model}"})
                    return
                if STATE["models"][model] != "loaded":
                    # 模拟 --models-autoload：unloaded 时被查询即误唤醒
                    STATE["slots_misfires"] += 1
                    STATE["models"][model] = "loaded"
            self._send_json(200, [{"id": 0, "is_processing": False, "n_ctx": 4096,
                                   "n_prompt_tokens": 123}])
        elif path == "/metrics":
            if ROUTER:
                model = self._model_param(query)
                if model is None:
                    self._send_json(400, {"error": "missing required parameter: model"})
                    return
                if model not in STATE["models"]:
                    self._send_json(404, {"error": f"unknown model {model}"})
                    return
                if STATE["models"][model] != "loaded":
                    STATE["metrics_misfires"] += 1
                    STATE["models"][model] = "loaded"
                self._send_text(200, METRICS_TEXT)
            else:
                self.send_response(501)
                self.end_headers()
        elif path == "/debug" and ROUTER:
            self._send_json(200, {
                "state": STATE["models"][MODELS[0]],
                "models": dict(STATE["models"]),
                "slots_misfires": STATE["slots_misfires"],
                "metrics_misfires": STATE["metrics_misfires"],
            })
        elif path == "/debug/stale_loaded" and ROUTER:
            STATE["stale_loaded"] = "clear" not in query
            self._send_json(200, {"stale_loaded": STATE["stale_loaded"]})
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length)
        if self.path == "/v1/chat/completions":
            if ROUTER:
                # 模拟 --models-autoload 唤醒：优先唤醒请求的模型，否则首个
                model = MODELS[0]
                try:
                    body = json.loads(raw) if raw else {}
                    if isinstance(body, dict) and body.get("model") in STATE["models"]:
                        model = body["model"]
                except ValueError:
                    pass
                if STATE["models"][model] == "unloaded":
                    STATE["models"][model] = "loaded"
            self._send_json(200, CHAT_BODY)
        elif ROUTER and self.path in ("/models/load", "/models/unload"):
            try:
                payload = json.loads(raw) if raw else {}
            except ValueError:
                payload = {}
            model = payload.get("model") if isinstance(payload, dict) else None
            if not model or model not in STATE["models"]:
                self._send_json(400, {"success": False, "error": "missing or unknown model"})
                return
            STATE["models"][model] = "loaded" if self.path == "/models/load" else "unloaded"
            self._send_json(200, {"success": True})
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, *a):
        pass


server = HTTPServer(("127.0.0.1", 8080), H)
with open(os.path.join(HERE, "fake_llama.pid"), "w") as f:
    f.write(str(os.getpid()))
print("fake server ready, pid", os.getpid(), "router", ROUTER, "models", MODELS, flush=True)
server.serve_forever()
