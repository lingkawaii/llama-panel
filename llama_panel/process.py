"""llama-server 进程管理：启动、停止、日志读取。"""
from __future__ import annotations

import os
import subprocess
import time


class ServerProcess:
    """管理一个 llama-server 子进程，stdout/stderr 追加写入日志文件。"""

    def __init__(
        self,
        command: list[str],
        cwd: str | None = None,
        log_path: str = "llama-server.log",
    ):
        self.command = command
        self.cwd = cwd
        self.log_path = log_path
        self._proc: subprocess.Popen | None = None
        self._log_file = None
        self.started_at: float | None = None

    @property
    def pid(self) -> int | None:
        if self._proc is not None and self._proc.poll() is None:
            return self._proc.pid
        return None

    def is_running(self) -> bool:
        return self.pid is not None

    def uptime(self) -> float | None:
        if self.started_at is None or not self.is_running():
            return None
        return time.time() - self.started_at

    def start(self) -> int:
        if self.is_running():
            raise RuntimeError("server already running")
        self._log_file = open(self.log_path, "ab", buffering=0)
        self._proc = subprocess.Popen(
            self.command,
            cwd=self.cwd,
            stdout=self._log_file,
            stderr=subprocess.STDOUT,
        )
        self.started_at = time.time()
        return self._proc.pid

    def stop(self, timeout: float = 10.0) -> None:
        proc = self._proc
        if proc is None or proc.poll() is not None:
            self._cleanup()
            return
        proc.terminate()
        try:
            proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=5)
        self._cleanup()

    def _cleanup(self) -> None:
        if self._log_file is not None:
            self._log_file.close()
            self._log_file = None
        self._proc = None
        self.started_at = None

    def tail(self, n: int = 200) -> list[str]:
        """返回日志末尾 n 行（只读最近 64KB）。"""
        if not os.path.exists(self.log_path):
            return []
        with open(self.log_path, "rb") as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            f.seek(max(0, size - 64 * 1024))
            data = f.read()
        return data.decode("utf-8", "replace").splitlines()[-n:]
