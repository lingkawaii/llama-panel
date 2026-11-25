# llama.cpp 监控与控制面板

基于 Flask 的本地面板：监控 llama-server 的运行状态、模型信息、槽位与 Prometheus 指标，并可一键启动/停止 llama-server 进程。

## 功能

- **服务控制**：启动 / 停止 llama-server 子进程，显示 PID 与运行时长，日志落盘到 `llama-server.log`
- **模型信息**：来自 `GET /v1/models`（模型 ID、架构、上下文长度）
- **槽位状态**：来自 `GET /slots`（处理中/空闲、提示词与生成 tokens、实时速度）
- **指标看板**：解析 `GET /metrics` 的 Prometheus 文本（`llamacpp:*` 指标），展示生成速度、提示词速度、请求数、上下文高水位、spec 解码接受率等，并列出全量指标
- **外部进程识别**：server 不是面板启动的也没关系——面板会按端口自动发现本机监听进程（Windows `netstat`，Linux `ss`/`lsof`），显示"运行中（外部）· PID"，并可直接停止
- 单页深色界面，自动轮询刷新（状态 2s / 日志 5s），无前端依赖

## 环境

```bash
python -m venv .venv
.venv/Scripts/activate        # Windows
pip install -r requirements.txt
```

## 使用

启动面板（默认监听 `http://127.0.0.1:5000`）：

```bash
python main.py
```

常用参数：

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--host` | `127.0.0.1` | 面板监听地址 |
| `--port` | `5000` | 面板监听端口 |
| `--server-url` | `http://127.0.0.1:8080` | llama-server 地址 |
| `--server-cmd` | `llama-server` | 面板启动 llama-server 的命令（空格分隔） |
| `--log` | `llama-server.log` | llama-server 日志文件 |

示例——让面板管理一个带指标的 llama-server：

```bash
python main.py --server-cmd "llama-server -m models/qwen3-8b-q4.gguf --metrics"
```

> 注意：llama-server 需加 `--metrics` 启动，`/metrics` 端点才会返回数据（否则返回 501，面板显示"无指标"）。

## API

| 方法 | 路径 | 说明 |
|---|---|---|
| GET | `/` | 面板页面 |
| GET | `/api/status` | 聚合状态：health / models / 进程 / 槽位 / 指标 |
| GET | `/api/logs?n=200` | llama-server 日志末尾 n 行 |
| POST | `/api/server/start` | 启动 llama-server（已在运行返回 409） |
| POST | `/api/server/stop` | 停止 llama-server（外部启动的按端口发现的 PID 终止） |

## 结构

```
main.py              # 入口（参数解析）
llama_panel/
  app.py             # Flask 应用与 API
  client.py          # llama-server HTTP 客户端 + Prometheus 解析
  process.py         # 进程管理 + 按端口发现/终止外部进程
static/index.html    # 单页面板
fake_llama.py        # 测试用假 llama-server
test_e2e.py          # 端到端测试（外部进程识别 + 停止）
```
