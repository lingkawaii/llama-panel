# llama.cpp 监控与控制面板 (`llama-panel`)

基于 Flask 的轻量级本地 `llama-server` 监控与控制面板（零前端依赖，仅依赖 `Flask` 与 `requests`）：Bento 暗色面板聚合服务状态、模型、GPU/VRAM、槽位与 Prometheus 指标，支持 systemd 托管、Router 多模型切换与快捷试跑。

## 核心功能

- **Bento 暗色面板**：4×2 网格卡片 + 原生 SVG 双趋势图（吞吐 / VRAM·温度，近 2 分钟采样），零外部 CDN、离线可用；危险操作均用页面内模态框确认
- **systemd 托管**：`--systemd-unit` 接入 systemd 单元（`systemctl start/stop/restart`）；uptime 取自 `ActiveEnterTimestampMonotonic`；外部进程按端口自动发现
- **Router 多模型切换**：识别 `--models-preset --models-autoload` 模式，多模型列表展示、一键加载/卸载、在线修改 `sleep-idle-seconds` 空闲自动卸载时长；`threading.Lock` 串行化 + 卸载后 5 秒冷却期，防止轮询误触发 `--models-autoload`
- **快捷试跑**：`POST /api/chat` 单轮对话试跑，返回回复与 prefill/decode 速度；`POST /api/probe` 极短探针测速
- **GPU/VRAM 与真实均值吞吐指标**：`rocm-smi` 优先、`nvidia-smi` 回退，2 秒超时优雅降级；槽位速度从日志 timing 行回填缓存，防刷屏丢失
- **彩色日志查看器**：关键词搜索、`全部/告警/错误/测速` 过滤、行数切换、暂停滚屏与一键复制
- **Linux CPU 电源档位**：接入 `powerprofilesctl`，读取并切换 `power-saver / balanced / performance`（非 Linux 自动隐藏）

## 安装

```bash
python -m venv .venv
pip install -r requirements.txt
```

## 快速启动

```bash
python main.py
```

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--host` | `127.0.0.1` | 面板监听地址（局域网访问填 `0.0.0.0`） |
| `--port` | `5000` | 面板监听端口 |
| `--server-url` | `http://127.0.0.1:8080` | `llama-server` 地址 |
| `--server-cmd` | `llama-server` | 面板直接拉起 `llama-server` 的命令（空格分隔） |
| `--log` | `llama-server.log` | `llama-server` 日志文件路径（支持外部/systemd 日志） |
| `--systemd-unit` | 无 | systemd 单元名，启停/重启走 `systemctl` |
| `--models-preset` | 无 | Router 模式 `models.ini` 路径，开启在线修改空闲自动卸载时长 |

```bash
# 面板直接管理单模型
python main.py --server-cmd "llama-server -m model.gguf --metrics"

# Router 模式：空闲 10 分钟自动卸载模型，来请求自动唤醒
python main.py --server-cmd "llama-server --models-preset models.ini --models-autoload --sleep-idle-seconds 600 --metrics"
```

> 提示：`llama-server` 需启用 `--metrics`（或 `models.ini` 中配置 `metrics = true`），`/metrics` 才会返回数据。

## Linux 部署参考（systemd + Polkit）

`/etc/systemd/system/llama-panel.service`（将 `<user>` 与 `/path/to/...` 替换为实际值）：

```ini
[Unit]
Description=llama.cpp Monitoring and Control Panel
After=network-online.target llama-server.service
Wants=network-online.target

[Service]
Type=simple
User=<user>
WorkingDirectory=/path/to/llama-panel
ExecStart=/path/to/llama-panel/.venv/bin/python /path/to/llama-panel/main.py --host 0.0.0.0 --port 5000 --systemd-unit llama-server --log /var/log/llama-server.log
Restart=on-failure
RestartSec=3

[Install]
WantedBy=multi-user.target
```

最小权限 Polkit 规则 `/etc/polkit-1/rules.d/50-llama-panel.rules`（允许非 root 用户启停服务、切换电源档位）：

```javascript
polkit.addRule(function(action, subject) {
    if (subject.user !== "<user>") return;
    if (action.id === "org.freedesktop.systemd1.manage-units" &&
        (action.lookup("unit") === "llama-server.service" || action.lookup("unit") === "llama-server")) {
        return polkit.Result.YES;
    }
    if (action.id === "org.freedesktop.UPower.PowerProfiles.switch-profile") {
        return polkit.Result.YES;
    }
});
```

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now llama-panel.service
```

## HTTP API

| 方法 | 路径 | 说明 |
|---|---|---|
| GET | `/` | 面板单页前端 |
| GET | `/api/status` | 聚合状态：`server` / `slots` / `metrics` / `gpu` / `power` |
| GET | `/api/logs?n=200` | 日志文件末尾 `n` 行 |
| POST | `/api/chat` | 快捷对话试跑：`{"prompt": "...", "model": "..."(可选), "max_tokens": 512}`，返回回复与速度指标 |
| POST | `/api/model/load` | Router 模式手动加载/切换模型（body 可选 `{"model": "..."}`） |
| POST | `/api/model/unload` | Router 模式手动卸载模型，启动 5 秒防误唤醒冷却期 |
| POST | `/api/preset/sleep-idle` | 在线修改 `sleep-idle-seconds`（`{"seconds": 600, "restart": true}`） |
| POST | `/api/power/profile` | 切换 CPU 电源档位（`{"profile": "power-saver"\|"balanced"\|"performance"}`） |
| POST | `/api/server/start` | 启动 `llama-server`（systemd 模式走 `systemctl start`） |
| POST | `/api/server/stop` | 停止 `llama-server`（外部进程按端口发现 PID 终止） |
| POST | `/api/server/restart` | 重启 `llama-server`（systemd 模式走 `systemctl restart`） |
| POST | `/api/probe` | 一键快速测速探针，返回耗时、Prefill/Decode 与 MTP 接受率 |

## 运行测试

```bash
.venv/Scripts/python.exe test_e2e.py   # Windows
.venv/bin/python test_e2e.py           # Linux / macOS
```

## 项目结构

```text
main.py            # CLI 入口与参数解析
llama_panel/       # app（路由/systemd）/ client（llama-server 客户端）/ gpu / power / preset / process
static/index.html  # 零依赖深色单页面板
fake_llama.py      # 模拟 llama-server（单/多模型 Router）
test_e2e.py        # 单元测试与端到端测试
```
