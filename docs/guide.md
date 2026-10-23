# 使用与运行机制

[返回项目首页](../README.md)

一个 Python + uv 管理的、CLI 优先的 Coding Harness。主 Agent 可按需并行委派只读子 Agent，普通工具顺序执行、模型非流式调用；通过服务契约与插件生命周期拆分模型、执行、权限和会话。当前支持 Linux / Python 3.13。

## 快速开始

```bash
uv python install 3.13
uv sync --locked --managed-python
uv run hiro-agent --help
```

无需 API Key 的演示：

```bash
mkdir -p /tmp/hiro-demo
uv run hiro-agent --mock --workspace /tmp/hiro-demo
```

输入任意任务后，Mock 将依次读取 `hiro-demo.txt`、请求批准写入 `Hello from Hiro!`、请求批准运行 Python 验证，然后回答结果。写入与命令各输入 `y` 批准一次；拒绝会被记录，Mock 会停止演示。Mock 是固定脚本，不理解任意编程需求。选择一个可覆盖 `hiro-demo.txt` 的演示目录。

也可以使用 `uv run hiro-agent chat --config configs/mock.json --workspace /tmp/hiro-demo`。

## 只读子 Agent

默认保持单 Agent。用 `--subagents` 或配置 `subagents.enabled: true` 启用委派：

```bash
uv run hiro-agent --subagents --workspace /path/to/your/project

# 无需 API Key：主 Agent 读取/写入/验证 → 并行探索与审查 → 汇总报告
mkdir -p /tmp/hiro-subagents-demo
uv run hiro-agent --mock --subagents --workspace /tmp/hiro-subagents-demo
# 或使用完整示例配置
uv run hiro-agent chat --config configs/mock-subagents.json --workspace /tmp/hiro-subagents-demo
```

Mock 输入任意任务，写入和命令各批准一次；验证文件后，同一轮声明独立的探索与审查任务，终端显示各任务的调用 ID、角色、子会话与状态，最后汇总两份报告。真实模型自行决定何时调用 `delegate_task`，简单任务可直接完成。

观察暂停后恢复：用 `configs/mock-subagents-paused.json` 启动演示，文件验证后会耗尽三次模型请求并暂停，输入 `/exit`。随后执行 `sessions` 查到父 Session，使用正常配置重新打开并输入 `/continue`；已提交的读取、写入和命令不会重跑，并行委派在新 Run 完成：

```bash
uv run hiro-agent chat --config configs/mock-subagents-paused.json --workspace /tmp/hiro-subagents-demo
uv run hiro-agent sessions --workspace /tmp/hiro-subagents-demo
uv run hiro-agent chat --config configs/mock-subagents.json --resume SESSION_ID --workspace /tmp/hiro-subagents-demo
```

| 角色 | 用途 | 权限 |
| --- | --- | --- |
| `explorer` | 定位实现、调用关系和测试，返回文件证据 | 读取、列举、搜索 |
| `reviewer` | 静态审查代码，指出问题和验证建议 | 读取、列举、搜索 |

委派参数只有 `role` 和 `task`；任务中应说明目标、路径、必要背景或待审查的 diff。每个子任务使用独立 Session，只接收该任务与角色提示，不复制父会话历史。最终报告必须是 JSON，包含 `summary`、`findings`（`path`、可选 `line`、`description`）和 `suggestions`；格式错误、超限或未完成会返回结构化失败，不能当作审查通过。

只支持一层、只读委派。同一轮相邻的 `delegate_task` 调用有界并发，`subagents.max_concurrency` 默认 2，允许 1–8；设为 1 恢复串行。普通工具构成顺序边界，例如 `[委派 A, 委派 B, 写文件, 委派 C]` 会先等待 A、B 全部结算，再写文件，最后执行 C。排队任务不会提前记录开始或创建子会话；主 Agent 等整轮调用结算后才再次请求模型。汇总时应保留各报告的文件证据与分歧。

子 Agent 不拥有写入、命令或继续委派的工具；即使模型猜出这些工具名称，执行器也会拒绝。子会话共用工作区，不是文件系统快照或 OS 沙箱。修改、测试和审批仍由主 Agent 执行；不提供后台任务、独立子任务取消或可写子代理。

角色可分别配置模型和追加指令；未设置 `model` 或设为 `null` 时共享主模型。提供角色 `model` 时独立解析完整 `ModelConfig`，省略字段使用该类型的默认值，不从主模型逐项合并。示例：

```json
{
  "subagents": {
    "enabled": true,
    "max_concurrency": 2,
    "roles": {
      "explorer": {
        "model": {
          "adapter": "compatible",
          "base_url": "https://your-provider.example/v1",
          "model": "your-explorer-model",
          "api_key_env": "HIRO_EXPLORER_API_KEY",
          "request_timeout_seconds": 120,
          "extra_body": {}
        },
        "instructions": "优先定位调用链和相关测试。"
      },
      "reviewer": {"model": null, "instructions": "优先报告可复现缺陷。"}
    }
  }
}
```

追加指令不能覆盖只读、禁止递归委派和 JSON 报告约束。启用委派时会在启动阶段校验两个角色的模型配置，初始化失败会释放已创建客户端和会话锁。客户端由运行时统一持有，子任务完成不会关闭共享模型；日志只增加 `adapter:model` 标识。主模型为 Mock（包括 `--mock`）时，所有角色强制使用 Mock，不读取真实密钥。

父子累计共享当前 Run 的 64 次模型请求、128 次工具受理和 600 秒总预算，委派工具本身也计入工具预算，排队时间计入根截止时间。每个子任务另有 16 次模型请求和 32 次工具受理的局部上限，配置可调；子任务不会获得新的主任务额度或截止时间。主 Run 指标包含子任务，子日志保留局部指标，两者不应再次相加。单个子任务模型失败、报告无效或局部额度用尽会返回结构化结果，其他任务继续。根预算耗尽会暂停父任务；Ctrl+C、超时或持久化失败停止派发，等待在途任务清理。持久化失败优先记录；其他中断保留最初原因，根预算对应暂停，用户取消对应取消。

父日志在启动子任务前持久化委派关系；子结果先写入子日志，再写入父日志缓存，最后配对父工具结果。恢复能复用已经持久化的完成报告；缺失、未开始或中断的子任务不会自动重跑。子日志损坏、关联不一致或仍被其他进程占用时会报错。通过父 Session 执行 `/continue`；子 Session 不能独立续跑，`sessions` 默认只列主会话。即使恢复时未启用 `--subagents`，仍会结算旧委派。

## 真实模型

选择支持 Chat Completions 文本与 function tools 的模型，设置环境变量后启动：

```bash
export HIRO_BASE_URL='https://your-provider.example/v1'
export HIRO_MODEL='your-tool-calling-model'
export HIRO_API_KEY='your-key'
uv run hiro-agent --workspace /path/to/your/project
```

接口地址在 `base_url` 后追加 `/chat/completions`；不需要把完整 endpoint 放进 `HIRO_BASE_URL`。本地无鉴权兼容服务也需要设置一个占位 API Key。程序不自动读取 `.env`，参考 `.env.example` 将变量导出到当前 shell。

`configs/baseline.json` 展示全部配置。CLI 的 `--mock` 优先选择 Mock；`HIRO_BASE_URL`、`HIRO_MODEL` 覆盖配置中的对应路由，密钥从 `api_key_env` 指定的环境变量读取。没有模型名称或密钥时会明确报错。

`model.extra_body` 可配置供应商支持的可选请求参数，如 `max_completion_tokens`；部分兼容服务仍使用 `max_tokens`，应按所选服务设置。它不能覆盖模型、消息、工具、stream、n 或密钥。首版不透传供应商专用推理消息字段、不支持图片与流式输出；需要特殊回传字段的模型应增加专用 Adapter。

网络请求不自动重试。请求错误结束当前 Run，历史保留；上下文超限和输出长度截断暂停，截断响应中的工具不会执行。当前 Context 保留完整历史，不自动摘要或裁剪旧消息。

## 聊天、取消与续跑

| 操作 | 行为 |
| --- | --- |
| 普通输入 | 在当前 Session 创建一个新 Run |
| `/help` | 显示帮助 |
| `/continue` | 继续最近未完成任务，新 Run、新预算，不重复原始任务 |
| `/tasks` | 空闲时查看当前 Session 最近一个 Run 的委派任务 |
| `/exit` 或空闲 Ctrl+D | 退出并关闭会话 |
| 空闲 Ctrl+C | 清空输入，继续聊天 |
| 执行或审批时 Ctrl+C | 取消当前 Run，等待工具清理后返回输入提示 |

运行期间只接受当前审批，不排队接收新任务。审批显示文件路径、写入内容或精确替换参数，以及完整的可执行文件和参数；回车、`n` 或 EOF 表示拒绝。只有明确输入 `y` / `yes` 才授权一次操作。

```bash
uv run hiro-agent sessions --workspace /path/to/your/project
uv run hiro-agent tasks --session SESSION_ID --workspace /path/to/your/project
uv run hiro-agent tasks --session SESSION_ID --run RUN_ID --workspace /path/to/your/project
uv run hiro-agent chat --resume SESSION_ID --workspace /path/to/your/project
```

`tasks` 默认显示该主 Session 的全部委派，可按 Run 筛选；展示任务简述、角色、模型标识、等待/运行/终态、子会话、局部模型与工具次数、报告摘要。查询只读取已有日志，不调用模型、不修复记录、不启动恢复；日志损坏或被占用时明确报不可读取。运行期间通过事件持续输出进度，不提供终端任务面板或输入队列。

打开会话后先恢复日志并等待输入，不会自动启动模型。使用 `/continue` 明确续跑，或输入新任务保留历史。一个会话同时只允许一个进程持有锁、一个 Run 正在运行。

预算默认每 Run 64 次模型请求、128 次工具受理、600 秒。参数错误与权限拒绝也消耗工具预算；总时间包含模型、工具和审批。续跑会显式创建新预算，历史各 Run 指标保留。`usage_missing` 区分供应商没有返回 usage 的请求，token 计数不伪造估算值。

## 工具与执行边界

| 工具 | 参数与行为 |
| --- | --- |
| `list_files` | `path`、`recursive`、`max_depth`、`max_results` |
| `read_file` | `path`、`start_line`、`num_lines`，有文件体积与输出上限 |
| `search_text` | `query`、`path`、`case_sensitive`、`max_matches`；纯文本匹配 |
| `write_file` | `path`、`content`、`create_parents`；创建或覆盖，必须批准 |
| `edit_file` | `path`、`old_text`、`new_text`、`expected_matches`；精确匹配，默认一次 |
| `run_command` | `executable`、`args`、`cwd`、`timeout_seconds`；必须批准 |

文件工具检查工作区边界与符号链接，不允许直接访问运行日志或改写 Git 元数据。遍历跳过 `.git`、`.venv`、`node_modules` 等目录。输出有界并标记截断，文件读取可使用行号继续获取；超大文件明确拒绝。

命令不隐式调用 shell，标准输入关闭，环境变量采用白名单并移除主模型和所有角色配置的密钥变量。命令超时或取消时对所属进程组发送终止信号，必要时强制结束并回收。stdout/stderr 共享输出上限，返回各自的有界预览。

这是受审批控制的本地开发执行器。批准的命令仍具有当前操作系统用户权限；工作目录不是操作系统沙箱。命令可以访问工作区外文件、网络或用户凭据文件。请在受信任项目中使用，运行不可信代码需要独立隔离环境。进程硬崩溃后，旧命令可能仍存活；恢复时会提示未知结果，需要先核实。

## 架构与恢复保证

代码位于 `src/hiro_agent/`，职责按小模块组织：

- `contracts.py` 定义 Pydantic 数据与 Protocol 服务；`core.py` 提供服务注册、插件依赖与生命周期，`bootstrap.py` 装配实现。
- `agent.py` 通过公共服务驱动状态机；`models.py` 提供兼容接口、脚本 Mock；`context.py` 提供完整历史投影入口和原样任务处理。
- `tools.py` 负责统一校验、预算、执行与结果记录；`permissions.py` 决定审批；`execution.py` 传播取消。
- `session.py` 提供内存 / JSONL 实现、模型历史投影和恢复；`cli.py` 负责输入、展示与审批交互。
- `subagents.py` 提供 `SubagentService` 的原生实现、只读角色、父子生命周期和跨会话恢复；`budget.py` 汇总父子额度与指标。

启动按 Session → Model → Permission → Tools → Context → 可选 Subagents → Agent 装配。插件初始化失败会撤销本插件注册并释放已登记资源；退出先停止 Agent，再逆序释放。普通 EventBus 监听器仅观察运行，不决定权限或持久化。

会话存放于工作区的 `.hiro-agent/sessions/`，该目录应加入目标项目的 `.gitignore`。Hiro 自身仓库已忽略它；程序不会自动修改目标项目的 Git 配置。日志包含任务、模型输入、工具参数与输出，应按项目内容妥善保管；程序不会把模型鉴权头或 API Key 配置写进日志。

持久化顺序为：

```text
assistant.message 持久化
  → tool.called → 参数 / 预算 / 审批
  → tool.started 写入并 fsync
  → 执行工具
  → tool.result 写入并 fsync
  → 下一顺序工具、委派组或模型请求
```

每次追加通过单写入队列完成，返回成功意味着对应写入已 fsync。并发委派允许不同调用乱序完成，仍按调用 ID 配对；下一次模型请求前整轮调用必须配对完整。追加失败会停止后续执行。恢复式日志读取只修复明确未完成的尾记录，中间损坏会报错；历史查询完全只读。旧配置和旧 JSONL 无需迁移。

恢复将完整 Assistant 消息中的每个调用配对：没有开始记录的补 `TOOL_NOT_STARTED`；已开始却没有结果的补 `TOOL_OUTCOME_UNKNOWN`。这些结果写入原 Run，恢复本身可重复执行。未结束的模型请求与 Run 也会补充中断状态。恢复不会自动重新执行旧工具，不保证外部副作用恰好一次，也不回滚文件。

添加能力时实现对应 Protocol，通过插件 `provide` 注册；资源清理用 `on_dispose` 登记。工具只实现能力，不直接改聊天历史；控制流必须经过统一 Executor 的 `execute_many` 调度和单次调用校验。需要内部运行上下文的工具通过 `register_contextual` 注册，该上下文由执行器注入，不能由模型参数指定。摘要、任务优化、Benchmark、可写子 Agent 和 HTTP API 尚未包含。

## 开发与验证

```bash
uv add <package>
uv add --dev <package>
uv sync --locked
uv run --locked pytest -q
uv run --locked ruff check .
uv run --locked ruff format --check .
uv run --locked mypy src
uv build
```

提交依赖变更时一并提交 `pyproject.toml` 与 `uv.lock`。测试使用临时工作区、Mock 模型和 HTTP MockTransport，不需要网络或 API Key；覆盖插件失败清理、拒绝执行、非法参数、预算、取消、子进程回收、会话锁、日志损坏和崩溃恢复。真实模型可按上面的配置在测试项目中执行一次读取、修改与验证作为独立冒烟测试。
