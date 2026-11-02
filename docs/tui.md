# 多 Agent 终端工作台

在交互终端运行 `hiro-agent`（不带子命令）会打开紫色全屏工作台，也可显式使用 `hiro-agent tui`。左侧的 Agent 对应持久 Bot；每条新消息通过统一 Task 入口提交，右侧按 Bot 汇总最近 30 个任务的对话。普通工具、审批、预算和恢复规则沿用持久内核。任务仍由本地单 worker 依次执行；界面可自由切换，不会增加执行并发或中断正在运行的任务。

## 启动

```bash
uv run hiro-agent --workspace /path/to/project
hiro-agent tui --workspace /path/to/project
hiro-agent tui -w /path/to/project --config writer.json --config reviewer.json --agent writer
hiro-agent chat --workspace /path/to/project
```

工作区需要已经存在。首次打开没有 Bot 的工作区时，程序会创建稳定 ID 为 `default` 的 Agent，沿用旧配置、`HIRO_MODEL` / `HIRO_BASE_URL`、工具权限及 `--subagents` 设置。已有 Bot 的工作区直接使用保存的配置和权限，不会因启动时的默认值或环境变量自动覆盖。

`tui --config` 可重复指定；多个配置均需有稳定的 `bot.id`。单个不含 `bot` 的旧配置可导入为 `default`。也可在界面中按 `F7` 或点击「导入 Agent」读取带 `bot.id` 的配置文件。显式导入相同 ID 按已有 BotStore 规则更新配置与权限版本，切换 Agent 本身不会更改配置。

离线体验：

```bash
mkdir -p /tmp/hiro-tui-demo
hiro-agent tui --demo -w /tmp/hiro-tui-demo
```

`--demo`（亦可写 `--mock`）添加 `demo-a` 和 `demo-b` 两个脚本化演示 Agent，不覆盖已有同 ID 配置。它们演示读取 `hiro-demo.txt`、请求批准写入和验证，不理解任意编程任务。写入仍需批准；真实模型使用各 Bot 保存的完整配置和凭据环境变量。

TUI 需要 stdin 和 stdout 都连接交互式 Linux 终端，支持鼠标点击与滚轮。不带子命令时，重定向或管道会继续使用传统聊天；`chat`、`resume` 和根命令的 `--resume SESSION_ID` 也继续进入传统聊天。

## 切换和交互

| 操作 | 行为 |
| --- | --- |
| `/agent ID` 或 `/agent 完整名称` | 切换 Agent；重名时使用完整 ID，未知目标保留当前 Agent |
| `/agent` | 打开左侧 Agent 列表；传统聊天中列出可用 Agent |
| `/chat` | 切换到当前 Agent 的传统聊天 |
| `/tui` | 切换到全屏工作台；已在 TUI 时保持当前界面 |
| `/help` | 查看命令帮助；输入 `/` 或命令前缀后按 Tab 补全 |
| 点击左侧 Agent | 立即切换对话，保留当前未发送草稿 |
| F2，↑/↓，Enter | 聚焦左侧列表、选择 Agent、回到输入框 |
| Alt+↑ / Alt+↓ | 在列表中直接切换 Agent |
| 点击搜索框输入 | 按名称、ID 或模型筛选；Tab/Shift+Tab 切换焦点 |
| Enter | 在主输入框发送任务 |
| Alt+Enter / Ctrl+J | 在主输入框换行 |
| PgUp / PgDn / 滚轮 | 浏览对话；Ctrl+End 回到最新消息并恢复跟随 |
| F3 / 查看并处理 | 查看完整待批准工具与参数，或回答问题 |
| F6 | 连接或启动独立 worker |
| F7 / 导入 Agent | 导入或更新 Bot JSON 配置 |
| F8 | 从当前 Agent 的活动任务中选择一个取消 |
| Esc | 关闭弹窗，或从 Agent 列表回到输入框 |
| Ctrl+C | 清空当前输入，不取消后台任务 |
| Ctrl+Q / 空输入时 Ctrl+D | 退出 TUI，不停止 worker |

侧栏显示最近消息、任务状态、待处理请求数量；其他 Agent 出现新活动时显示圆点提示。未发送的消息草稿按 Bot 分开保存在本次界面会话中，问题答复草稿按请求分开保存。弹窗打开期间不会切换到其他 Agent；提交、答复和取消均绑定打开时的 Bot 与目标任务/请求，后台刷新不会改变操作归属。

在同一次启动中，可以反复用 `/chat`、`/tui` 切换界面，并在传统聊天中用 `/agent ID或完整名称` 切换 Agent。每个 Agent 保留各自的传统聊天 Session，返回 TUI 后保留各自的草稿；重复进入当前界面的命令不会另开会话。导航命令不会发送模型任务。

不带 Bot 身份的旧版聊天首次切到 TUI 时，会保留原配置和权限建立对应 Agent；若工作区已有 `default`，会为这段聊天分配独立身份，避免混入已有 Agent 的指令或权限。

TUI 显示持久 Task 的汇总，传统聊天显示对应 Session 的历史，两种历史分别保留，不自动合并。切换界面不会把 worker 的任务 Session 当作传统会话恢复，也不会移动或重放已提交任务。重新启动后，传统会话仍可通过 `resume` 或 `/resume` 显式恢复。

审批弹窗先展示工具名与完整参数，可滚动查看；「批准一次」只答复该持久交互，执行端仍验证当前权限版本和精确操作。多个待处理请求可点击「下一条」。拒绝与问题答案同样持久保存，worker 重启后可继续处理。

窄于 78 列时，列表和对话分别显示，使用 F2 打开列表、Enter/Esc 返回对话。宽终端同时显示两栏。

## 后台任务和恢复

默认检查工作区 worker 锁：已有 worker 时直接连接，没有时启动独立进程。退出或切换 TUI 不会取消已提交任务。仅连接、不自动启动：

```bash
hiro-agent --no-start-worker -w /path/to/project
hiro-agent tui --no-start-worker -w /path/to/project
```

worker 离线时仍可保存任务和答复，启动后恢复处理；启动失败会显示提示，进程输出保存在工作区 `.hiro-agent/worker.log`。单 worker 的既有 OS 锁防止多个界面启动重复执行器。手工管理 worker 仍使用 [内核使用指南](kernel/USAGE.md)中的命令；TUI 的退出键只关闭界面。

消息区刷新已经完整落盘的内容，不伪造未收到的模型文字。显示层对活动日志取只读快照，只解析完整行，不取得执行锁，也不修复、恢复或截断日志。缓存仅用于显示；提交、答复、取消始终重新通过服务校验。读取损坏日志会显示错误并保留可用的任务内容。

结果为 `needs_reconciliation` 时，仍需使用 `kernel resolve` 记录人工核实说明；TUI 不会自动重试未知副作用。历史记录、任务和权限的持久语义见 [接口契约](kernel/CONTRACTS.md)。

![实际 TUI 渲染示例](images/tui-preview.png)
