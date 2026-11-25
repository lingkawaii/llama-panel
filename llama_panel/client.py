"""llama-server HTTP 客户端与 Prometheus 文本解析。"""
from __future__ import annotations

import re

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


class LlamaClient:
    """对 llama-server 的只读查询封装。所有方法在失败时返回 None/[] 而非抛异常。"""

    def __init__(self, base_url: str, timeout: float = 3.0):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._session = requests.Session()

    def _get(self, path: str):
        return self._session.get(self.base_url + path, timeout=self.timeout)

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

    def slots(self) -> list[dict]:
        """GET /slots，返回槽位状态列表（服务端 --no-slots 时为空）。"""
        try:
            r = self._get("/slots")
            r.raise_for_status()
            data = r.json()
            return data if isinstance(data, list) else []
        except (requests.RequestException, ValueError):
            return []

    def metrics(self) -> dict[str, float] | None:
        """GET /metrics。服务端未开 --metrics 时返回 None（HTTP 501）。"""
        try:
            r = self._get("/metrics")
            if r.status_code == 501:
                return None
            r.raise_for_status()
            return parse_prometheus(r.text)
        except requests.RequestException:
            return None
