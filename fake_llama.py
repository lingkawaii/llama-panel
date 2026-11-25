"""测试用假 llama-server：提供 /health /v1/models /slots，/metrics 返回 501。"""
import json
import os
from http.server import BaseHTTPRequestHandler, HTTPServer

HERE = os.path.dirname(os.path.abspath(__file__))


class H(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/health":
            body = json.dumps({"status": "ok"}).encode()
        elif self.path == "/v1/models":
            body = json.dumps({"data": [{"id": "fake-model", "object": "model"}]}).encode()
        elif self.path == "/slots":
            body = json.dumps([{"id": 0, "is_processing": False, "n_ctx": 4096,
                                "n_prompt_tokens": 123}]).encode()
        elif self.path == "/metrics":
            self.send_response(501)
            self.end_headers()
            return
        else:
            self.send_response(404)
            self.end_headers()
            return
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *a):
        pass


server = HTTPServer(("127.0.0.1", 8080), H)
with open(os.path.join(HERE, "fake_llama.pid"), "w") as f:
    f.write(str(os.getpid()))
print("fake server ready, pid", os.getpid(), flush=True)
server.serve_forever()
