# llama.cpp 监控与控制面板

基于 Flask 的本地面板：监控 llama-server 的运行状态、模型信息、GPU/VRAM、槽位与 Prometheus 指标，并可一键启动/停止/重启 llama-server 进程（支持 systemd 托管）。

## 功能

- **服务控制**：启动 / 停止 / 重启 llama-server 子进程，显示 PID 与运行时长，日志落盘到 `llama-server.log`；停止/重启需二次确认，请求进行中按钮自动禁用
- **systemd 托管**：`--systemd-unit` 指定 systemd 单元后，状态查询与启停/重启全部走 `systemctl`（非 root 无免密权限时返回 500 + 明确错误）；运行时长取自 `ActiveEnterTimestampMonotonic`（单调时钟，真实 uptime）
- **模型信息**：来自 `GET /v1/models`（模型 ID、架构、上下文长度）
- **显卡 / VRAM**：只读采集 GPU 型号、显存用量（带进度条，>85% 警告色、>95% 错误色）、GPU 利用率、温度、功耗；优先 `rocm-smi`（PATH 或 `~/bin/rocm-smi`），回退 `nvidia-smi`，2 秒超时，无 GPU 工具时优雅隐藏
- **槽位状态**：来自 `GET /slots`（处理中/空闲、提示词与生成 tokens、实时速度）
- **指标看板**：解析 `GET /metrics` 的 Prometheus 文本（`llamacpp:*` 指标），4×2 对称网格展示生成速度、提示词速度、spec 接受率、KV 缓存命中率、上下文高水位、处理中/排队请求、提示词/生成 tokens 累计；全量指标折叠收纳
- **快速测速**：模型卡内一键测速按钮，向 `/v1/chat/completions` 发极短请求，返回耗时 / Prefill / Decode 速度 / MTP 接受率（`POST /api/probe`，失败返回 502 并内联提示）
- **Router 模式适配**：自动识别 `--models-preset ... --models-autoload` 的 Router 模式（`/v1/models` 返回 `status` 字段），模型卡显示状态徽章（已加载/休眠中（已卸载）/加载中…）与自动休眠策略（如"空闲 10m 自动卸载"）；**防误唤醒**：仅 `loaded` 状态才请求 `/slots` 与 `/metrics`（Router 模式下这两端点必须带 `?model=`，`unloaded` 时被查询会触发 autoload 误唤醒），休眠时表格显示友好提示；支持**一键加载/卸载模型**（`POST /api/model/load` / `POST /api/model/unload`，对应上游 `POST /models/load` / `POST /models/unload`），卸载前二次确认，操作期间按钮禁用、失败内联红字提示
- **实时趋势**：纯原生 SVG 双图（生成/提示词速度 + VRAM 占用/GPU 温度），保留最近 60 个采样点（约 2 分钟），无 GPU 数据时第二张图隐藏、第一张撑满整行
- **日志查看器**：关键词搜索、告警/错误过滤、测速行过滤、100/200/500 行切换、暂停/恢复滚屏、一键复制当前过滤结果
- **外部进程识别**：server 不是面板启动的也没关系——面板会按端口自动发现本机监听进程（Windows `netstat`，Linux `ss`/`lsof`），显示"运行中（外部）· PID"，并可直接停止
- 单页深色界面，自动轮询刷新（状态 2s / 日志 5s），页头显示最后刷新时间并可手动刷新，无前端依赖，移动端自适应

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
| `--systemd-unit` | 无 | 用 systemd 单元管理 server（如 `llama-server.service`），启停/重启走 `systemctl` 而非面板拉起子进程 |

示例——让面板管理一个带指标的 llama-server：

```bash
python main.py --server-cmd "llama-server -m models/qwen3-8b-q4.gguf --metrics"
```

示例——server 由 systemd 托管（单元文件 `ExecStart=... llama-server --metrics`）：

```bash
python main.py --systemd-unit llama-server.service --log /var/log/llama-server.log
```

> 注意 1：llama-server 需加 `--metrics` 启动，`/metrics` 端点才会返回数据（否则返回 501，面板显示"无指标"）。
> 注意 2：`--systemd-unit` 模式下，面板用户需要有权执行 `systemctl start/stop/restart <unit>`（root 或配置了免密 polkit 规则），否则操作返回 500 并在页面内联提示原因。
> 注意 3：Router 模式（`--models-preset ... --models-autoload --sleep-idle-seconds 600`）下，闲置超时的模型会自动卸载释放显存，来请求自动重新加载；面板会自动识别该模式，休眠时不轮询 `/slots`/`/metrics`（避免误唤醒），并可在模型卡一键加载/卸载。

示例——Router 模式（多模型预设 + 自动加载/休眠）：

```bash
python main.py --server-cmd "llama-server --models-preset models.ini --models-autoload --sleep-idle-seconds 600"
```

## API

| 方法 | 路径 | 说明 |
|---|---|---|
| GET | `/` | 面板页面 |
| GET | `/api/status` | 聚合状态：health / models / model_state / 进程 / 槽位 / 指标 / gpu（Router 模式 `unloaded` 时 `slots=[]`、`metrics=null`，不请求上游） |
| GET | `/api/logs?n=200` | llama-server 日志末尾 n 行 |
| POST | `/api/model/load` | Router 模式加载模型（上游 `POST /models/load`）；body 可选 `{"model": "..."}`，缺省取当前 model_id；无 model_id 400，上游失败 502 |
| POST | `/api/model/unload` | Router 模式卸载模型释放显存（上游 `POST /models/unload`）；参数与错误码同上 |
| POST | `/api/server/start` | 启动 llama-server（已在运行返回 409；systemd 模式走 `systemctl start`） |
| POST | `/api/server/stop` | 停止 llama-server（外部启动的按端口发现的 PID 终止；systemd 模式走 `systemctl stop`） |
| POST | `/api/server/restart` | 重启 llama-server（面板模式先停后启；systemd 模式走 `systemctl restart`） |
| POST | `/api/probe` | 一键快速测速：极短 chat 请求，返回耗时 / Prefill / Decode / MTP 接受率；失败 502 |

## 结构

```
main.py              # 入口（参数解析）
llama_panel/
  app.py             # Flask 应用与 API（含 systemd 托管）
  client.py          # llama-server HTTP 客户端 + Prometheus 解析 + 快速测速 probe + Router 状态解析/加载卸载
  gpu.py             # GPU/VRAM 只读采集（rocm-smi / nvidia-smi）+ 纯函数解析
  process.py         # 进程管理 + 按端口发现/终止外部进程
static/index.html    # 单页面板
fake_llama.py        # 测试用假 llama-server（--router 模拟 Router 模式：status 字段、?model= 必填、/models/load|unload、autoload 唤醒与误唤醒计数）
test_e2e.py          # 端到端测试（外部进程识别 + 启停/重启 + 快速测速 + Router 模式 load/unload + GPU 解析、systemd uptime 与 parse_model_state 单测）
```

## 测试

```bash
.venv/Scripts/python.exe test_e2e.py
```
