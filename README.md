# hiro-agent

一个用 Python 手写的 Coding Agent Harness，通过 CLI 在本地项目中读取代码、修改文件、执行验证。

参考 [mini-dsh](https://github.com/huangjunsen0406/mini-dsh) 对核心运行机制的拆解，**用一个可以跑起来的项目理解 Agent Harness**。Hiro 自行实现服务容器与插件生命周期，核心保留五条清晰的链路：

1. **Context / Plugin / Service**：注册服务 → 检查依赖 → 管理资源生命周期。
2. **Session Event Log**：记录事件 → 派生模型消息 → 保存与恢复会话。
3. **Tool Registry / Executor**：注册工具 → 导出 schema → 校验、审批与执行。
4. **Model Adapter**：统一模型接口 → Chat Completions 兼容服务 / 离线 Mock。
5. **Agent Loop**：模型 → 工具 → 模型 → 最终回答。

在这条主链路上，Hiro 还实现了运行预算、JSONL 会话恢复和可选的只读子 Agent。默认运行单 Agent，普通工具顺序执行，模型非流式调用；适合阅读源码、学习实现和扩展实验。

完整配置与恢复细节见 [使用与运行机制](docs/guide.md)。

## 环境

* Linux

* Python `>= 3.12`

* [uv](https://docs.astral.sh/uv/)（仓库 CI 使用 `0.9.15`）

## 快速开始

```bash
git clone https://github.com/curealb/hiro-agent.git
cd hiro-agent
uv python install 3.12
uv sync --locked --managed-python
uv run hiro-agent --help
```

### 先跑通：无需 API Key

```bash
mkdir -p /tmp/hiro-demo
uv run hiro-agent --mock --workspace /tmp/hiro-demo
```

输入任意任务，Mock 会演示完整的「模型 → 工具 → 模型」链路：

```text
用户输入
  → 读取 hiro-demo.txt
  → 请求批准写入 Hello from Hiro!
  → 请求批准运行 Python 验证
  → 回答结果
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

完整配置见 [configs/baseline.json](configs/baseline.json)；`HIRO_BASE_URL`、`HIRO_MODEL` 可覆盖对应配置，`--mock` 强制使用 Mock。真实模式没有默认模型名，需要显式设置 `HIRO_MODEL` 或配置中的 `model.model`。

## CLI

```text
/help
/continue
/tasks
/exit
```

| 输入                  | 行为                          |
| ------------------- | --------------------------- |
| 普通任务描述              | 在当前 Session 创建新 Run         |
| `/help`             | 显示帮助                        |
| `/continue`         | 继续最近未完成的任务，使用新 Run 和新预算     |
| `/tasks`            | 查看当前 Session 最近一个 Run 的委派任务 |
| `/exit` 或空闲时 Ctrl+D | 退出                          |
| 空闲时 Ctrl+C          | 清空当前输入                      |
| 执行或审批时 Ctrl+C       | 取消当前 Run，清理后返回输入提示          |

修改文件和执行命令前会展示操作并请求批准。仅 `y` / `yes` 授权本次操作，回车、`n` 或 EOF 均表示拒绝。

## 会话与恢复

Session 保存跨轮次的对话与事件；Run 是一次任务执行，拥有自己的状态和预算。同一个 Session 可以包含多个 Run。

会话保存在工作区的 `.hiro-agent/sessions/`。查询与恢复：

```bash
uv run hiro-agent sessions --workspace /path/to/your/project
uv run hiro-agent tasks --session SESSION_ID --workspace /path/to/your/project
uv run hiro-agent chat --resume SESSION_ID --workspace /path/to/your/project
```

恢复后等待用户输入；用 `/continue` 继续任务，或直接输入新任务并保留历史。`tasks` 查询整个会话的委派记录，可追加 `--run RUN_ID` 筛选。

工具开始和结果事件落盘后才继续推进；恢复根据调用 ID 配对结果。旧工具调用不会自动重跑，结果未知的操作需要先核实。日志包含任务、工具参数及输出，在目标项目中应将 `.hiro-agent/` 加入 `.gitignore`。

## Agent Loop 什么时候结束？

理解主循环时，可以先忽略预算、审批与日志细节，只看这段伪代码：

```python
while True:
    messages = derive_messages(session.events)
    response = await model(messages, tool_schemas)
    record_assistant(response)

    if not response.tool_calls:
        return response.content

    await execute_tools(response.tool_calls)
```

工具结果写回会话，再派生下一轮模型消息。模型正常返回且不再请求工具时，当前 Run 完成；具体实现见 [agent.py](src/hiro_agent/agent.py)。

Hiro 还为循环设置了可配置的预算，防止错误的模型或工具链一直运行：

| 每个 Run 的预算        | 默认值   |
| ----------------- | ----- |
| 模型请求次数            | 64    |
| 工具受理次数            | 128   |
| 总运行时间（包含模型、工具和审批） | 600 秒 |

预算耗尽、上下文超限或模型输出被长度限制截断时暂停；截断响应中的工具不会执行。Ctrl+C 取消当前 Run，模型请求或主循环异常会结束当前 Run 并记录失败。暂停后可用 `/continue` 在新 Run 中继续，并获得新预算。参数错误与权限拒绝也计入工具预算。

## 只读子 Agent

通过 `--subagents` 开启委派，模型按任务需要选择角色：

| 角色         | 职责            | 可用能力       |
| ---------- | ------------- | ---------- |
| `explorer` | 定位实现、调用关系和测试  | 列举、读取、搜索文件 |
| `reviewer` | 静态审查代码并提出验证建议 | 列举、读取、搜索文件 |

```bash
# 真实模型
uv run hiro-agent --subagents --workspace /path/to/your/project

# 离线演示：读取 / 写入 / 验证 → 并行探索与审查 → 汇总
mkdir -p /tmp/hiro-subagents-demo
uv run hiro-agent --mock --subagents --workspace /tmp/hiro-subagents-demo
```

每个子任务使用独立 Session，只接收任务描述和角色提示，返回带文件证据的 JSON 报告。默认并发为 2，可配置为 1–8；同一轮相邻的委派可以并行，普通工具仍按顺序执行。仅支持一层委派，子 Agent 不能写文件、运行命令或继续委派。

父子共享主任务预算，每个子任务另有 16 次模型请求、32 次工具受理的局部上限。角色模型、预算与暂停恢复示例见 [详细指南](docs/guide.md#只读子-agent)。

## 工具

| 工具              | 用途                    |
| --------------- | --------------------- |
| `list_files`    | 列举工作区文件               |
| `read_file`     | 按行读取文本                |
| `search_text`   | 搜索纯文本，返回匹配位置          |
| `write_file`    | 创建或覆盖文件，需批准           |
| `edit_file`     | 精确匹配替换，需批准            |
| `run_command`   | 执行程序及参数，需批准           |
| `delegate_task` | 启用子 Agent 后，委派探索或审查任务 |

文件工具检查工作区边界、符号链接并限制输出大小。命令通过可执行文件和参数运行，不隐式调用 shell；取消或超时会终止并回收所属进程组。

批准的命令拥有当前操作系统用户的权限，工作目录并非 OS 沙箱。运行不可信代码应使用独立隔离环境。

## 最重要的心智模型

```text
                 Context / Plugin / Service
                 服务注册、依赖与资源生命周期
                            │
      ┌──────────┬──────────┼──────────┬──────────┐
      ▼          ▼          ▼          ▼          ▼
   Session     Model      Tools    Permission   Agent
   事件日志    兼容接口    注册与执行   审批策略     主循环
   JSONL       Mock         │
                       ┌────┴─────────────┐
                       ▼                  ▼
                    文件 / 命令       delegate_task（可选）
                                      ├─ explorer
                                      └─ reviewer

运行链路：CLI → Agent Loop → 历史消息 → Model → Tools → 下一轮
```

**Agent Loop 通过统一的服务契约调用模型、工具与会话。** 文件读写和命令执行的具体实现留在工具层；子 Agent 也通过 `delegate_task` 接入这条工具链路。

服务容器的 `Context` 位于 [core.py](src/hiro_agent/core.py)，负责装配与生命周期；[context.py](src/hiro_agent/context.py) 中的上下文服务负责历史消息投影与任务预处理。理解这两个职责，再阅读循环会更清楚。

替换模型、工具或会话实现时，实现对应 Protocol，再通过插件注册服务；工具调用经过统一 Executor 完成校验、预算、审批与结果记录。当前插件是项目内的服务装配机制，尚未接入 MCP。

## 当前边界

* 保留完整历史，尚无自动摘要或上下文裁剪。

* 仅支持文本、非流式调用；网络请求不自动重试。

* 子 Agent 只读，共用工作区，无文件系统快照。

* 恢复不回滚文件，也不保证外部副作用恰好发生一次。

* 尚无 MCP 集成、HTTP API、TUI / Web UI、Benchmark 或可写子 Agent。

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
