# hiro-agent

一个用 Python 手写的、支持会话恢复与只读子 Agent 的 Coding Agent Harness，通过 CLI 在本地项目中读取代码、修改文件、执行验证。

参考 [mini-dsh](https://github.com/huangjunsen0406/mini-dsh) 对核心运行机制的拆解，Hiro 将模型、工具、权限、会话与 Agent Loop 组织成可替换的服务，适合阅读源码和扩展实验。

核心由六部分组成：

1. **Context / Plugin / Service**：服务注册、依赖检查与资源生命周期。
2. **Session Event Log**：用 JSONL 记录事件，从日志投影模型消息并恢复中断任务。
3. **Tool Registry / Executor**：统一管理工具 schema、参数校验、审批、执行与结果记录。
4. **Model Adapter**：接入兼容 Chat Completions 的文本模型，或使用离线 Mock。
5. **Agent Loop**：模型请求 → 工具调用 → 结果回传 → 最终回答。
6. **Subagents**：按需并行委派只读探索与审查，共享主任务预算。

默认运行单 Agent；普通工具顺序执行，模型非流式调用。详细配置与恢复语义见 [使用与运行机制](docs/guide.md)。

## 环境

- Linux
- Python `>= 3.13`
- [uv](https://docs.astral.sh/uv/)（仓库 CI 使用 `0.9.15`）

## 快速开始

```bash
git clone https://github.com/curealb/hiro-agent.git
cd hiro-agent
uv python install 3.13
uv sync --locked --managed-python
uv run hiro-agent --help
```

### 无需 API Key 的演示

```bash
mkdir -p /tmp/hiro-demo
uv run hiro-agent --mock --workspace /tmp/hiro-demo
```

输入任意任务，Mock 会执行固定流程：

```text
读取 hiro-demo.txt → 请求批准写入 Hello from Hiro!
                  → 请求批准运行 Python 验证 → 回答结果
```

写入和命令各输入一次 `y` 批准；拒绝后演示停止。Mock 用于验证运行链路，不理解任意编程需求；请使用允许覆盖 `hiro-demo.txt` 的演示目录。

### 接入真实模型

选择支持 Chat Completions 文本和 function tools 的服务，在当前 shell 导出配置：

```bash
export HIRO_BASE_URL='https://your-provider.example/v1'
export HIRO_MODEL='your-tool-calling-model'
export HIRO_API_KEY='your-key'

uv run hiro-agent --workspace /path/to/your/project
```

`HIRO_BASE_URL` 填服务的 API 基址，程序会追加 `/chat/completions`。程序不会自动加载 `.env`，变量示例见 [.env.example](.env.example)。本地无鉴权服务也需要设置占位 API Key。

使用 JSON 配置启动：

```bash
uv run hiro-agent chat --config configs/baseline.json --workspace /path/to/your/project
```

完整配置见 [configs/baseline.json](configs/baseline.json)；`HIRO_BASE_URL`、`HIRO_MODEL` 可覆盖对应配置，`--mock` 强制使用 Mock。

## CLI

| 输入 | 行为 |
| --- | --- |
| 普通任务描述 | 在当前 Session 创建新 Run |
| `/help` | 显示帮助 |
| `/continue` | 继续最近未完成的任务，使用新 Run 和新预算 |
| `/tasks` | 查看当前 Session 最近一个 Run 的委派任务 |
| `/exit` 或空闲时 Ctrl+D | 退出 |
| 空闲时 Ctrl+C | 清空当前输入 |
| 执行或审批时 Ctrl+C | 取消当前 Run，清理后返回输入提示 |

修改文件和执行命令前会展示操作并请求批准。仅 `y` / `yes` 授权本次操作，回车、`n` 或 EOF 均表示拒绝。

查询与恢复会话：

```bash
uv run hiro-agent sessions --workspace /path/to/your/project
uv run hiro-agent tasks --session SESSION_ID --workspace /path/to/your/project
uv run hiro-agent chat --resume SESSION_ID --workspace /path/to/your/project
```

恢复后等待用户输入；用 `/continue` 继续任务。已记录的旧工具调用不会自动重跑，结果未知的操作需要先核实。

## 只读子 Agent

通过 `--subagents` 开启委派，模型按任务需要选择角色：

| 角色 | 职责 | 可用能力 |
| --- | --- | --- |
| `explorer` | 定位实现、调用关系和测试 | 列举、读取、搜索文件 |
| `reviewer` | 静态审查代码并提出验证建议 | 列举、读取、搜索文件 |

```bash
# 真实模型
uv run hiro-agent --subagents --workspace /path/to/your/project

# 离线演示：读取 / 写入 / 验证 → 并行探索与审查 → 汇总
mkdir -p /tmp/hiro-subagents-demo
uv run hiro-agent --mock --subagents --workspace /tmp/hiro-subagents-demo
```

每个子任务使用独立 Session，只接收任务描述和角色提示，返回带文件证据的 JSON 报告。默认并发为 2，可配置为 1–8；仅支持一层委派，子 Agent 不能写文件、运行命令或继续委派。

主任务默认每 Run 最多请求模型 64 次、受理工具 128 次、运行 600 秒，父子共享额度；每个子任务另有 16 次模型请求、32 次工具受理的局部上限。角色模型、预算与暂停恢复示例见[详细指南](docs/guide.md#只读子-agent)。

## 工具

| 工具 | 用途 |
| --- | --- |
| `list_files` | 列举工作区文件 |
| `read_file` | 按行读取文本 |
| `search_text` | 搜索纯文本，返回匹配位置 |
| `write_file` | 创建或覆盖文件，需批准 |
| `edit_file` | 精确匹配替换，需批准 |
| `run_command` | 执行程序及参数，需批准 |
| `delegate_task` | 启用子 Agent 后，委派探索或审查任务 |

文件工具检查工作区边界、符号链接并限制输出大小。命令通过可执行文件和参数运行，不隐式调用 shell；取消或超时会终止并回收所属进程组。

批准的命令拥有当前操作系统用户的权限，工作目录并非 OS 沙箱。运行不可信代码应使用独立隔离环境。

## 核心设计

```text
CLI
 └─ Agent Loop
     ├─ Context ── Session Event Log ── JSONL / 恢复
     ├─ Model Adapter ── Chat Completions / Mock
     └─ Tool Executor ── 校验 / 预算 / 审批 / 记录
         ├─ 文件与命令工具
         └─ delegate_task
             ├─ explorer ── 独立 Session / 只读工具
             └─ reviewer ── 独立 Session / 只读工具

Context / Plugin / Service 负责上述服务的装配和生命周期
```

Agent Loop 通过服务契约调用模型、工具与会话。替换某个服务时，通过插件注册新实现；工具执行仍经过统一 Executor。

会话保存在工作区的 `.hiro-agent/sessions/`。工具开始和结果事件落盘后才继续推进；恢复根据调用 ID 配对结果，避免盲目重放旧操作。日志会包含任务、工具参数及输出，在目标项目中应将 `.hiro-agent/` 加入 `.gitignore`。

## 从哪里开始读源码

按下面的顺序，可先理解主循环，再查看执行与恢复机制：

```text
src/hiro_agent/
├── cli.py          # 入口、输入、展示与审批交互
├── config.py       # 配置与默认值
├── contracts.py    # 数据结构与服务 Protocol
├── core.py         # 服务容器、插件依赖与生命周期
├── bootstrap.py    # 将实现装配为运行时
├── agent.py        # Agent Loop 与运行状态机
├── models.py       # 兼容接口与固定脚本 Mock
├── context.py      # 历史消息投影与任务处理入口
├── tools.py        # 工具注册、调度与执行记录
├── permissions.py  # 审批策略
├── execution.py    # 文件、子进程执行与取消
├── session.py      # JSONL 日志、会话锁与恢复
├── budget.py       # 父子共享预算与指标
├── subagents.py    # 只读角色、并行委派与恢复
└── tasks.py        # 委派任务查询
```

随后阅读 [tests/](tests/) 中对应模块的行为测试。[详细指南](docs/guide.md) 保留了插件清理、日志写入顺序、未知工具结果与委派恢复的说明。

## 当前边界

- 保留完整历史，尚无自动摘要或上下文裁剪。
- 仅支持文本、非流式调用；网络请求不自动重试。
- 子 Agent 只读，共用工作区，无文件系统快照。
- 恢复不回滚文件，也不保证外部副作用恰好发生一次。
- 尚无 MCP 集成、HTTP API、TUI / Web UI、Benchmark 或可写子 Agent。

## 开发与验证

```bash
uv sync --locked
uv run --locked ruff check .
uv run --locked ruff format --check .
uv run --locked mypy src
uv run --locked pytest -q
uv build
```

测试使用临时工作区、Mock 模型和 HTTP MockTransport，无需真实 API Key。覆盖模型适配、工具审批、预算、取消、子进程回收、并行委派、会话锁与日志恢复；CI 执行相同检查。

## 参考

- [mini-dsh](https://github.com/huangjunsen0406/mini-dsh)：README 的组织方式与 Agent Harness 核心机制的学习参考。
