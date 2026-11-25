# llama.cpp 监控与控制面板 (`llama-panel`)

基于 Flask 的轻量级本地 `llama-server` 监控与控制面板（零前端依赖、仅依赖 `Flask` 与 `requests`）：实时监控服务状态、模型信息、GPU/VRAM、槽位与 Prometheus 指标，支持 `systemd` 托管启停、**Router 模式自动休眠/防误唤醒/一键卸载加载**，以及一键快速测速。

## 核心功能

- **服务控制与 `systemd` 托管**：
  - 支持直接托管 `llama-server` 子进程，或通过 `--systemd-unit` 接入 `systemd` 单元（`systemctl start/stop/restart`）；
  - 真实运行时长 `uptime` 取自 `ActiveEnterTimestampMonotonic`（单调时钟）；
  - 外部进程自动发现：即使服务不是面板拉起的，也会按端口自动识别本机进程（Windows `netstat` / Linux `ss`、`lsof`）；
  - **页面内暗色确认弹窗**：所有危险操作（停止、重启、卸载模型）均使用页面内模态框确认（支持遮罩与 `Esc` 取消），避免原生 `window.confirm()` 在 IDE/Electron 内置 WebView 中被静默拦截。
- **Router 模式与显存省电（防误唤醒）**：
  - 自动识别 `--models-preset ... --models-autoload --sleep-idle-seconds <N>` 的 Router 模式，展示模型状态徽章（`已加载` / `休眠中（已卸载）` / `加载中…`）与空闲自动卸载策略（如 `空闲 10m 自动卸载`）；
  - 支持一键 **「卸载模型」** 释放显存与 **「加载模型」** 手动唤醒（`POST /api/model/unload`、`/api/model/load`）；
  - **并发防误唤醒机制**：后端通过 `threading.Lock` 串行化状态轮询与加载/卸载，并在卸载后启用 **5 秒冷却期**，冷却期及 `unloaded` 状态下绝不向 Router 发送 `?model=` 的 `/slots` 或 `/metrics` 请求，彻底防止多标签页轮询误触发 `--models-autoload`。
- **模型信息与一键测速**：
  - 展示模型 ID、上下文窗口 `n_ctx`、量化格式 `ftype`、参数量与文件大小；
  - 模型卡内置 **「快速测速」** 按钮（`POST /api/probe`），发送极短对话探针并返回总耗时、Prefill、Decode 速度与 MTP 投机解码接受率。
- **显卡 / VRAM 实时监控**：
  - 只读采集显卡型号、VRAM 占用与颜色分级进度条（`>85%` 琥珀色、`>95%` 红色）、GPU 利用率、温度与功耗；
  - 优先探测 `rocm-smi`（`PATH` 或 `~/bin/rocm-smi`），自动回退 `nvidia-smi`，2 秒超时保护，无 GPU 环境优雅降级。
- **指标看板与原生 SVG 趋势图**：
  - 4×2 对称网格卡片（移动端自动切换 2 列）：生成速度、提示词速度、spec 接受率、KV 缓存命中率、上下文高水位、处理中/排队请求、提示词/生成 tokens 累计；全量 Prometheus 指标默认折叠收纳；
  - 纯原生 `<svg>` 实时双趋势图（吞吐趋势 + 显卡 VRAM/温度趋势，近 2 分钟 60 个采样点，零外部 CDN，离线可用）。
- **槽位与运维日志查看器**：
  - 槽位表展示并发槽位状态、上下文及吞吐速度（移动端单行横向滚动不折行）；
  - 日志区支持关键词搜索（忽略大小写）、`全部` / `告警/错误` / `测速` 快捷过滤、`100 / 200 / 500` 行切换、暂停/恢复滚屏与一键复制。

## 安装依赖

```bash
python -m venv .venv

# Windows (PowerShell)
.venv\Scripts\Activate.ps1
# Linux / macOS
source .venv/bin/activate

pip install -r requirements.txt
```

## 快速启动

```bash
python main.py
```

默认监听 `http://127.0.0.1:5000`，常用参数如下：

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--host` | `127.0.0.1` | 面板监听地址（局域网访问填 `0.0.0.0`） |
| `--port` | `5000` | 面板监听端口 |
| `--server-url` | `http://127.0.0.1:8080` | `llama-server` 地址 |
| `--server-cmd` | `llama-server` | 面板直接拉起 `llama-server` 的命令（空格分隔） |
| `--log` | `llama-server.log` | `llama-server` 日志文件路径（支持直接读取外部/systemd 日志） |
| `--systemd-unit` | 无 | 指定 `systemd` 单元名（如 `llama-server`），启停与重启走 `systemctl` |

### 启动示例

```bash
# 1. 面板直接管理单模型 llama-server
python main.py --server-cmd "llama-server -m models/qwen.gguf --metrics"

# 2. 面板接入 systemd 托管的 llama-server（局域网可访问）
python main.py --host 0.0.0.0 --port 5000 --systemd-unit llama-server --log /home/ling/llama/server.log

# 3. Router 模式（空闲 10 分钟自动卸载模型，来请求自动唤醒）
python main.py --server-cmd "llama-server --models-preset models.ini --models-autoload --sleep-idle-seconds 600 --metrics"
```

> 提示：`llama-server` 需启用 `--metrics`（或 `models.ini` 中配置 `metrics = true`），`/metrics` 才会返回数据。

## Linux 生产部署参考（Ubuntu + systemd + Router）

### 1. Router 模式 `models.ini` 示例

```ini
version = 1

[*]
sleep-idle-seconds = 600
metrics = true

[qwen3.8-27b]
model = /home/ling/models/llama/your-model.gguf
n-gpu-layers = 999
ctx-size = 81920
cache-type-k = q8_0
cache-type-v = q4_0
parallel = 2
kv-unified = true
flash-attn = on
```

配合启动脚本：

```bash
exec ./llama-server \
  --models-preset /home/ling/llama/models.ini \
  --models-max 1 \
  --models-autoload \
  --sleep-idle-seconds 600 \
  --metrics \
  --host 0.0.0.0 --port 8080
```

### 2. 配置 `llama-panel.service` 开机自启

`/etc/systemd/system/llama-panel.service`：

```ini
[Unit]
Description=llama.cpp Monitoring and Control Panel
After=network-online.target llama-server.service
Wants=network-online.target

[Service]
Type=simple
User=ling
Group=ling
WorkingDirectory=/home/ling/llama/llama-panel
Environment=PATH=/home/ling/bin:/usr/local/bin:/usr/bin:/bin
ExecStart=/home/ling/llama/llama-panel/.venv/bin/python /home/ling/llama/llama-panel/main.py --host 0.0.0.0 --port 5000 --systemd-unit llama-server --log /home/ling/llama/server.log
Restart=on-failure
RestartSec=3

[Install]
WantedBy=multi-user.target
```

启用服务：

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now llama-panel.service
```

### 3. 最小权限 Polkit 规则（允许非 root 用户在网页一键启停）

`/etc/polkit-1/rules.d/50-llama-panel.rules`（仅放行指定用户管理 `llama-server.service` 单个单元）：

```javascript
polkit.addRule(function(action, subject) {
    if (action.id === "org.freedesktop.systemd1.manage-units" &&
        (action.lookup("unit") === "llama-server.service" || action.lookup("unit") === "llama-server") &&
        subject.user === "ling") {
        return polkit.Result.YES;
    }
});
```

## HTTP API

| 方法 | 路径 | 说明 |
|---|---|---|
| GET | `/` | 面板单页前端 |
| GET | `/api/status` | 聚合状态：`health` / `models` / `model_state` / `process` / `slots` / `metrics` / `gpu`（Router `unloaded` 或卸载冷却期内不请求上游 `/slots` 与 `/metrics`） |
| GET | `/api/logs?n=200` | 返回日志文件末尾 `n` 行 |
| POST | `/api/model/load` | Router 模式手动加载模型（body 可选 `{"model": "..."}`，缺省取当前模型；无模型 400，上游失败 502） |
| POST | `/api/model/unload` | Router 模式手动卸载模型并启动 5 秒防误唤醒冷却期（参数与状态码同上） |
| POST | `/api/server/start` | 启动 `llama-server`（运行中 409；`systemd` 模式走 `systemctl start`） |
| POST | `/api/server/stop` | 停止 `llama-server`（外部进程按端口发现 PID 终止；`systemd` 模式走 `systemctl stop`） |
| POST | `/api/server/restart` | 重启 `llama-server`（面板模式先停后启；`systemd` 模式走 `systemctl restart`） |
| POST | `/api/probe` | 一键快速测速探针，返回耗时、Prefill/Decode 速度与 MTP 接受率（失败 502） |

## 项目结构

```text
main.py              # CLI 入口与参数解析
llama_panel/
  app.py             # Flask 应用、API 路由、systemd 托管与卸载并发防护（锁 + 冷却期）
  client.py          # llama-server HTTP 客户端、Prometheus 解析、快速测速与 Router 控制
  gpu.py             # GPU/VRAM 只读采集（rocm-smi / nvidia-smi）与纯函数解析
  process.py         # 子进程管理与跨平台外部进程发现/终止
static/index.html    # 零依赖深色单页面板（SVG 趋势图、日志查看器、页面内确认弹窗）
fake_llama.py        # 模拟 llama-server（支持 --router 模式、误唤醒计数与 stale_loaded 钩子）
test_e2e.py          # 单元测试与端到端测试（GPU/uptime/Router 解析、启停重启、探针、8 线程并发防误唤醒）
```

## 运行测试

```bash
# Windows
.venv\Scripts\python.exe test_e2e.py

# Linux / macOS
.venv/bin/python test_e2e.py
```
