"""llama-server HTTP 客户端与 Prometheus 文本解析。"""
from __future__ import annotations

import re
import time

import requests

_METRIC_LABELED = re.compile(
    r"^([a-zA-Z_:][a-zA-Z0-9_:]*)\{([^}]*)\}\s+(-?[\d.eE+]+|NaN|[-+]?Inf(?:inity)?)$"
)
_METRIC_PLAIN = re.compile(
    r"^([a-zA-Z_:][a-zA-Z0-9_:]*)\s+(-?[\d.eE+]+|NaN|[-+]?Inf(?:inity)?)$"
)


def parse_prometheus(text: str) -> dict[str, float]:
    """解析 Prometheus 文本格式为 {name 或 name{labels}: value}。

    通用解析：不依赖具体指标名，兼容 llamacpp:* 与旧版 llama_server_* 命名。
    """
    out: dict[str, float] = {}
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        m = _METRIC_LABELED.match(line)
        if m:
            name, labels, value = m.groups()
        else:
            m = _METRIC_PLAIN.match(line)
            if not m:
                continue
            name, value = m.groups()
            labels = ""
        try:
            v = float(value)
        except ValueError:
            continue
        out[f"{name}{{{labels}}}" if labels else name] = v
    return out


_SLEEP_ARG_RE = re.compile(r"sleep-idle-seconds\s*=\s*(\d+)")


def _sleep_idle_seconds(status: dict) -> int | None:
    """从 Router 模式 status 中解析 --sleep-idle-seconds 秒数。

    优先取 args 列表（"--sleep-idle-seconds" 的下一项），
    否则在 preset 文本中正则匹配；未配置或 <= 0 返回 None。
    """
    args = status.get("args")
    if isinstance(args, list):
        for i, a in enumerate(args):
            if a == "--sleep-idle-seconds" and i + 1 < len(args):
                try:
                    v = int(str(args[i + 1]).strip())
                except (TypeError, ValueError):
                    return None
                return v if v > 0 else None
    preset = status.get("preset")
    if isinstance(preset, str):
        m = _SLEEP_ARG_RE.search(preset)
        if m:
            v = int(m.group(1))
            return v if v > 0 else None
    return None


def parse_model_state(models_payload: dict | None) -> dict:
    """从 GET /v1/models 响应解析模型状态（Router 模式感知）。

    返回 {"router_mode": bool, "model_id": str | None,
          "state": str, "sleep_idle_seconds": int | None}

    - router_mode: data[0].status 是含 "value" 的 dict（Router 模式特征）
    - state: Router 模式取 status["value"]（loaded/unloaded/loading）；
      非 Router 单模型模式有 model_id 则为 "loaded"，否则 "none"
    - sleep_idle_seconds: 仅 Router 模式从 status 解析，未配置为 None
    """
    data = models_payload.get("data") if isinstance(models_payload, dict) else None
    first = data[0] if isinstance(data, list) and data and isinstance(data[0], dict) else {}
    model_id = first.get("id")
    status = first.get("status")
    router_mode = isinstance(status, dict) and "value" in status
    if router_mode:
        value = status.get("value")
        state = value if isinstance(value, str) and value else "unknown"
        sleep_idle = _sleep_idle_seconds(status)
    else:
        state = "loaded" if model_id else "none"
        sleep_idle = None
    return {
        "router_mode": router_mode,
        "model_id": model_id,
        "state": state,
        "sleep_idle_seconds": sleep_idle,
    }


class LlamaClient:
    """对 llama-server 的只读查询封装。所有方法在失败时返回 None/[] 而非抛异常。"""

    def __init__(self, base_url: str, timeout: float = 3.0):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._session = requests.Session()

    def _get(self, path: str, params: dict | None = None):
        return self._session.get(self.base_url + path, params=params, timeout=self.timeout)

    def health(self) -> dict | None:
        try:
            r = self._get("/health")
            r.raise_for_status()
            return r.json()
        except (requests.RequestException, ValueError):
            return None

    def models(self) -> dict | None:
        """GET /v1/models，返回 OpenAI 兼容的模型信息。"""
        try:
            r = self._get("/v1/models")
            r.raise_for_status()
            return r.json()
        except (requests.RequestException, ValueError):
            return None

    def probe(self, model: str | None = None, timeout: float = 30.0) -> dict:
        """发送极短 chat 请求测速，返回 timings 解析结果。

        成功: {"ok": True, "wall_ms", "prompt_tps", "gen_tps",
               "prompt_n", "predicted_n", "spec_accept_percent"}
        失败: {"ok": False, "error": "..."}
        """
        if model is None:
            models = self.models()
            data = models.get("data") if isinstance(models, dict) else None
            if isinstance(data, list) and data and isinstance(data[0], dict):
                model = data[0].get("id") or "default"
            else:
                model = "default"
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": "Reply OK and count 1 to 5."}],
            "max_tokens": 32,
            "temperature": 0.0,
            "stream": False,
        }
        try:
            start = time.perf_counter()
            r = self._session.post(
                self.base_url + "/v1/chat/completions", json=payload, timeout=timeout
            )
            wall_ms = (time.perf_counter() - start) * 1000
            r.raise_for_status()
            data = r.json()
        except requests.RequestException as e:
            return {"ok": False, "error": f"请求失败: {e}"}
        except ValueError:
            return {"ok": False, "error": "响应不是合法 JSON"}
        timings = data.get("timings") if isinstance(data, dict) else None
        if not isinstance(timings, dict):
            timings = {}
        draft_n = timings.get("draft_n")
        draft_accepted = timings.get("draft_n_accepted")
        spec = None
        if isinstance(draft_n, (int, float)) and draft_n > 0 \
                and isinstance(draft_accepted, (int, float)):
            spec = draft_accepted / draft_n * 100
        return {
            "ok": True,
            "wall_ms": round(wall_ms, 1),
            "prompt_tps": timings.get("prompt_per_second"),
            "gen_tps": timings.get("predicted_per_second"),
            "prompt_n": timings.get("prompt_n"),
            "predicted_n": timings.get("predicted_n"),
            "spec_accept_percent": spec,
        }

    def slots(self, model: str | None = None) -> list[dict]:
        """GET /slots，返回槽位状态列表（服务端 --no-slots 时为空）。

        Router 模式必须带 model 参数，否则服务端返回 400。
        """
        params = {"model": model} if model else None
        try:
            r = self._get("/slots", params=params)
            r.raise_for_status()
            data = r.json()
            return data if isinstance(data, list) else []
        except (requests.RequestException, ValueError):
            return []

    def metrics(self, model: str | None = None) -> dict[str, float] | None:
        """GET /metrics。服务端未开 --metrics 时返回 None（HTTP 501）。

        Router 模式必须带 model 参数，否则服务端返回 400。
        """
        params = {"model": model} if model else None
        try:
            r = self._get("/metrics", params=params)
            if r.status_code == 501:
                return None
            r.raise_for_status()
            return parse_prometheus(r.text)
        except requests.RequestException:
            return None

    def load_model(self, model: str, timeout: float = 30.0) -> dict:
        """POST /models/load，Router 模式手动加载模型（唤醒）。

        成功: {"ok": True, "model": model}
        失败: {"ok": False, "error": "..."}
        """
        return self._model_action("/models/load", model, timeout)

    def unload_model(self, model: str, timeout: float = 15.0) -> dict:
        """POST /models/unload，Router 模式手动卸载模型释放显存。

        成功: {"ok": True, "model": model}
        失败: {"ok": False, "error": "..."}
        """
        return self._model_action("/models/unload", model, timeout)

    def _model_action(self, path: str, model: str, timeout: float) -> dict:
        try:
            r = self._session.post(
                self.base_url + path, json={"model": model}, timeout=timeout
            )
            r.raise_for_status()
            return {"ok": True, "model": model}
        except requests.RequestException as e:
            return {"ok": False, "error": f"请求失败: {e}"}
