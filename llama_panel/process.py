"""llama-server 进程管理：启动、停止、日志读取、外部进程发现。"""
from __future__ import annotations

import os
import re
import signal
import subprocess
import time


def _local_port(addr: str) -> int | None:
    """从 "0.0.0.0:8080" / "[::1]:8080" / "*:8080" 提取端口。"""
    if ":" not in addr:
        return None
    port = addr.rsplit(":", 1)[-1]
    return int(port) if port.isdigit() else None


def find_pid_on_port(port: int) -> int | None:
    """查找监听指定端口的进程 PID（面板未管理的外部 server）。

    Windows: netstat -ano；POSIX: ss -ltnp，回退 lsof。
    """
    if os.name == "nt":
        try:
            out = subprocess.run(
                ["netstat", "-ano"], capture_output=True, timeout=5
            ).stdout or b""
        except (OSError, subprocess.SubprocessError):
            return None
        for line in out.decode("utf-8", "replace").splitlines():
            parts = line.split()
            # TCP 行: Proto Local Foreign State PID；本地端口 == 监听端口
            # PID 0 是 TIME_WAIT 等内核态行，跳过
            if (
                len(parts) >= 5
                and _local_port(parts[1]) == port
                and parts[4].isdigit()
                and parts[4] != "0"
            ):
                return int(parts[4])
        return None

    try:
        out = subprocess.run(
            ["ss", "-ltnp"], capture_output=True, timeout=5
        ).stdout or b""
        for line in out.decode("utf-8", "replace").splitlines():
            fields = line.split()
            if (
                fields
                and fields[0] == "LISTEN"
                and len(fields) >= 4
                and _local_port(fields[3]) == port
            ):
                m = re.search(r"pid=(\d+)", line)
                if m:
                    return int(m.group(1))
    except (OSError, subprocess.SubprocessError):
        pass
    try:
        out = subprocess.run(
            ["lsof", f"-iTCP:{port}", "-sTCP:LISTEN", "-t"],
            capture_output=True, timeout=5,
        ).stdout or b""
        pids = [int(x) for x in out.decode("utf-8", "replace").split() if x.isdigit()]
        if pids:
            return pids[0]
    except (OSError, subprocess.SubprocessError):
        pass
    return None


def kill_pid(pid: int, timeout: float = 10.0) -> None:
    """终止外部进程：POSIX SIGTERM→SIGKILL，Windows taskkill /T /F。"""
    if pid <= 0:
        return
    if os.name == "nt":
        try:
            subprocess.run(
                ["taskkill", "/PID", str(pid), "/T", "/F"],
                capture_output=True, timeout=15,
            )
        except (OSError, subprocess.SubprocessError):
            pass
        return
    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return
        time.sleep(0.1)
    try:
        os.kill(pid, signal.SIGKILL)
    except ProcessLookupError:
        pass


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
