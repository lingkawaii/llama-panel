"""llama.cpp 监控与控制面板入口。"""
from __future__ import annotations

import argparse

from llama_panel.app import create_app


def main() -> None:
    p = argparse.ArgumentParser(description="llama.cpp monitoring & control panel")
    p.add_argument("--host", default="127.0.0.1", help="面板监听地址 (默认 127.0.0.1)")
    p.add_argument("--port", type=int, default=5000, help="面板监听端口 (默认 5000)")
    p.add_argument(
        "--server-url",
        default="http://127.0.0.1:8080",
        help="llama-server 地址 (默认 http://127.0.0.1:8080)",
    )
    p.add_argument(
        "--server-cmd",
        default="llama-server",
        help="启动 llama-server 的命令，空格分隔 (默认 llama-server)",
    )
    p.add_argument("--log", default="llama-server.log", help="llama-server 日志文件")
    args = p.parse_args()

    app = create_app(args.server_url, args.server_cmd.split(), args.log)
    app.run(host=args.host, port=args.port)


if __name__ == "__main__":
    main()
