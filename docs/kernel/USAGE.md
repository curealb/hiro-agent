# 持久 Bot / Task 使用方法

B3/C4/D 集成说明。命令从项目的 `hiro-agent` 入口运行，也可使用 `.venv/bin/python -m hiro_agent.cli`。下面所有命令都指定同一个工作区；SQLite 位于其 `.hiro-agent/kernel.sqlite3`，原始执行日志位于 `.hiro-agent/sessions/`。持久命令输出 JSON，便于提取 Bot、Task、Run、交互和定时计划 ID。

## 保存 Bot，提交任务，启动独立 worker

在 `bot.json` 保存完整配置，例如：

```json
{
  "bot": {
    "id": "writer",
    "name": "报告助手",
    "instructions": "需要澄清时使用 ask_user。所有写入必须等待审批。",
    "allowed_tools": ["list_files", "read_file", "search_text", "write_file", "edit_file", "ask_user", "read_skill"]
  },
  "model": {
    "adapter": "compatible",
    "base_url": "https://your-provider.example/v1",
    "model": "your-model-id",
    "api_key_env": "HIRO_API_KEY"
  },
  "permissions": "ask-write",
  "budget": {"max_model_requests": 64, "max_tool_calls": 128, "timeout_seconds": 600}
}
```

凭据由 worker 的环境变量 `HIRO_API_KEY` 提供，不写入 SQLite。离线检查可使用 `"model": {"adapter": "mock"}`；内置 mock 面向原有简单演示，集成测试使用注入的确定性模型验证审批/提问，不需要网络。

```sh
hiro-agent kernel bot-save bot.json -w /absolute/workspace
hiro-agent kernel bots -w /absolute/workspace
hiro-agent kernel bot writer -w /absolute/workspace
hiro-agent kernel submit writer '检查项目，生成报告并保存到 report.md' --idempotency-key report-001 -w /absolute/workspace
```

`submit` 保存 Task 后立即退出。相同 Bot 和幂等键的相同请求返回原任务，冲突文本报错。省略幂等键每次新建任务。每个 Task 有独立 Session，每次恢复执行有新的 Run ID。

另一个终端独立运行：

```sh
hiro-agent kernel worker -w /absolute/workspace --poll-interval 0.5
```

worker 持续处理队列，完全独立于提交 CLI。一个工作区只允许一个 worker；`.hiro-agent/worker.lock` 是 OS 进程锁，第二个 worker 会失败，进程退出释放锁。不要删除锁文件。`--once` 在启动对账后触发一轮到期计划、至多处理一个任务，用于一次性检查；服务运行应省略它。持续服务有独立的定时轮询协程，即使模型或工具正在等待，仍会把到期工作提交到同一个任务队列；它不会并行执行第二个 Task。

需要后台运行时，可由终端或服务管理器管理 worker，例如：

```sh
nohup hiro-agent kernel worker -w /absolute/workspace > /tmp/hiro-worker.log 2>&1 &
```

记录 shell 返回的 PID，以 `kill -TERM PID` 停止；前台可按 Ctrl+C。停止会协作取消正在运行的 Agent，等工具清理和日志结算后释放锁。已启动而没有可靠结果的工具会进入待核实状态。强制结束进程后，下次 worker 启动先对账。

## 用管理 Agent 创建永久 Bot

已有 `default` Bot 后，可以创建一个继承其模型、预算和工具权限的管理员：

```sh
hiro-agent kernel agent-manager --source default -w /absolute/workspace
hiro-agent tui --agent agent-manager -w /absolute/workspace
```

默认 ID 为 `agent-manager`、名称为「Agent 管理员」，可通过 `--bot-id`、`--name` 指定其他值。安装只创建新身份，不会覆盖同名 ID；源 Bot 必须允许写入。管理员在源工具白名单上增加 `list_agents`、`create_agent`、`send_message` 和 `read_message`，可为新 Agent 分配通信能力。模型、凭据环境变量名及运行限制从已保存配置继承，工具结果仅返回身份和权限摘要。

向管理员发送「创建一个只读代码审查员，重点检查权限边界」，模型可调用：

```json
{
  "id": "code-reviewer",
  "name": "代码审查员",
  "instructions": "静态审查代码，优先检查权限边界，结论附文件与行号。",
  "allowed_tools": ["list_files", "read_file", "search_text"],
  "permissions": "read-only"
}
```

这是 `create_agent` 的参数。该工具按现有写入规则等待审批；批准后，在同一 SQLite 事务内检查授权版本、创建者当前权限和新 ID，再保存完整配置。新 Agent 的工具必须是创建者工具的子集；省略 `permissions` 时继承创建者模式。授予 `delegate_task` 还要求创建者已启用子 Agent。只有显式授予 `create_agent` 且允许写入，新 Agent 才能再创建其他永久 Agent。

`list_agents({})` 返回工作区已有 Agent 的 ID、名称、工具白名单和权限模式，不返回模型配置或凭据。新 Bot 会在 TUI 下次刷新时出现在左侧，可以接收独立任务；创建本身不会提交任务。已有 ID、无权限或审批失效时不会写入。发生结果未知的崩溃时沿用 `needs_reconciliation`，不会自动重放创建。工具也适用于绑定 Bot 的传统 `chat` 会话；未绑定 Bot 的旧会话不提供管理工具。

代码更新后，已运行的 worker 需要重启才能加载新工具；重启 TUI 本身不会停止旧 worker。

## 清空 Agent 历史与删除 Agent

```sh
# 交互确认；加 --yes 可在脚本中显式确认
hiro-agent kernel clear-history agent-manager -w /absolute/workspace
hiro-agent kernel bot-delete code-writer -w /absolute/workspace

# 补齐默认三个 Agent；已有配置和历史不覆盖
hiro-agent kernel init-agents -w /absolute/workspace
# 空工作区可指定模型配置，或使用 HIRO_MODEL 等环境变量
hiro-agent kernel init-agents --config /path/to/model.json -w /absolute/workspace
# 离线初始化
hiro-agent kernel init-agents --mock -w /absolute/workspace
```

默认身份为 Hiro（`default`）、Agent 管理员（`agent-manager`）、代码编写员（`code-writer`）。新增身份优先继承已有 `default` 的模型与运行设置，其次使用管理员或其他已有 Agent；显式 `--config` 只影响这次新增的身份。初始化原子地创建缺失身份，不删除其他 Agent，不改变已存在默认身份的配置、权限或历史。

`clear-history` 删除目标 Bot 的持久任务、Run、审批/提问、全部绑定的传统会话及子会话 JSONL、手工记忆和任务摘要，也清理相应交接与定时触发历史。身份、模型、角色指令、工具权限、技能启用设置和定时任务定义保留；未来定时执行仍可生成新任务。`bot-delete` 进一步移除身份、专属定时任务和技能设置。两者不删除项目文件、共享技能文档或其他 Bot 的任务和日志，共享任务链仍保留已经消耗的预算。

清理命令默认要求确认，非交互脚本可加 `--yes`。执行中的任务、结果未知的待核对任务、打开的传统会话，以及相关其他 Bot 的未完成工作都会阻止清理；目标 Bot 尚未执行或等待答复的任务可随确认一并删除。TUI 中对应 `/clear`、`/delete-agent`、`/init-agents`。

在线且空闲的新版 worker 支持清理，执行与清理通过维护锁协调。若仍运行更新前的 worker，命令会提示先重启 worker。文件删除采用暂存清单，SQL 提交决定恢复还是删除日志；异常退出若留下未完成清理，后续写入会被阻止，重新执行原清理命令（必要时加 `--yes`）完成恢复。稳定的锁文件会保留，不包含对话内容。

## 查询、审批、提问和取消

```sh
hiro-agent kernel tasks --bot-id writer -w /absolute/workspace
hiro-agent kernel task TASK_ID -w /absolute/workspace
hiro-agent kernel interactions --task-id TASK_ID -w /absolute/workspace
hiro-agent kernel reply INTERACTION_ID --approve -w /absolute/workspace
hiro-agent kernel reply INTERACTION_ID --reject -w /absolute/workspace
hiro-agent kernel reply QUESTION_ID --text '按中文格式，报告覆盖最近一周' -w /absolute/workspace
hiro-agent kernel cancel TASK_ID -w /absolute/workspace
```

审批、拒绝、文本三种答复只能选择一种。交互列表包含完整提示、精确规范化操作参数、答复、消费状态和消费 Run ID。只有列入 Bot 白名单的工具可被执行；`ask_user` 也须显式列入。`read-only` 禁止所有修改工具。

写入工具首次调用时，规范化参数（含 schema 默认值）、工具名、Task/Bot/Session/Run/call 和权限版本会保存为精确操作意图。旧 Run 结束为 paused，旧模型声明的所有工具都有未执行或其他终态结果，Task 进入 `waiting_approval`。问题同样结束旧 Run，Task 进入 `waiting_input`，等待期间不占执行时间。

答复只写交互数据库，worker 启动及轮询重复读取已答复记录并幂等唤醒任务。批准后创建新 Run、写新的唯一工具声明，通过同一执行器执行保存的精确参数，然后才允许模型继续回答。授权绑定当前权限版本且只消费一次；撤权、权限修改后恢复旧白名单、参数改变和重复消费都会阻止执行。没有“自动重新询问普通审批”的兜底。拒绝、无效参数、禁用工具和失败执行均产生可查询终态。

问题答复先写 `interaction.answer` 原始日志，再消费一次性答案并加入新 Run 的用户输入。同一合法答复在已回答或已消费状态下可重复提交，不会重复执行或无限唤醒；冲突答复被拒绝。已经取消的交互不能再答复。

取消 queued/waiting Task 立即生效；running Task 设置持久取消请求，由 worker 协作停止后结算。待核实 Task 的取消请求不会擦除副作用未知状态。重新保存 Bot 配置即可更新工具权限；执行器在执行前检查最新权限，已开始的副作用不能撤销。

## 累计预算与恢复边界

Task 的模型次数、工具次数和执行时间在每个 Run 使用同一累计账本，包含失败尝试和子 Agent 用量。每次用量变化直接 checkpoint SQLite，活动期间每 100ms 保存执行时间；不通过 observer 确认持久化。终态日志前冻结本 Run 时间，避免终态写入期间心跳造成指标倒退。新 Run 只获得总预算减已用额度；审批/提问等待以及进程停止期间不计活动时间。崩溃对账以已保存 SQL 用量和真实日志重建用量的较大值为下限，不会退还已记额度。硬崩溃时最后未保存的短时间片无法精确重建，恢复采用最后持久证据；中断的已开始 Run 不自动重跑。

| 崩溃切点 | 下次启动行为 |
| --- | --- |
| SQL 已领取，Run 仍为 claimed，JSONL 无 run.started | 结束空领取并重新排队；新建 Run，不重复计费。 |
| JSONL 已有 run.ended，SQL 尚未 finish | 从相同 Run 的终态及指标结算；不重跑模型/工具，重复恢复幂等。 |
| 答复已保存，Task 仍在等待 | 从持续存在的 ready 清单重复 wake，最多一个可领取任务。 |
| tool.started 已落盘，tool.result 缺失 | 先恢复 child，再补齐工具结果为 unknown；Task 进入 needs_reconciliation，禁止自动领取。 |

worker 在调用 Agent 前先将 SQL Run 标为 started。这个标记之后、JSONL run.started 之前若崩溃，或已开始的日志后来丢失，会保守进入 `needs_reconciliation`；它与纯 claimed 空领取不同。合法空日志可识别；损坏、错误关联或忙碌日志停止 worker，不能当作不存在后重新执行。

审批授权消费之后、工具真正开始之前若崩溃，授权不返还；有日志且可确认未启动的中断 Run 安全失败，需用户检查后重新提交任务获得新审批。问题答案消费与模型输入记录之间若崩溃，同样安全失败，完整答案仍可由交互查询及 `interaction.answer` 日志取回；检查后重新提交任务。不会悄悄再发放授权或重复使用答复。

存在未知副作用时先检查外部实际结果，再明确记录人工决定：

```sh
hiro-agent kernel resolve TASK_ID fail '已确认部分写入；保留现场，另建修复任务' -w /absolute/workspace
hiro-agent kernel resolve TASK_ID cancel '已人工撤销副作用' -w /absolute/workspace
hiro-agent kernel resolve TASK_ID retry '已核实未产生副作用，允许重新执行' -w /absolute/workspace
```

只有 `needs_reconciliation` 可 resolve，说明不能为空。`retry` 是人工允许重新执行，保留原累计预算；已有取消请求的任务不能 retry。待核实的最终日志不会被删除。

## Bot 记忆、摘要和上下文压缩

```sh
hiro-agent kernel memory-add writer '报告使用中文，先列出可核实证据' -w /absolute/workspace
hiro-agent kernel memories writer --query '报告' -w /absolute/workspace
hiro-agent kernel memory writer MEMORY_ID -w /absolute/workspace
hiro-agent kernel memory-update writer MEMORY_ID '报告使用中文，并保留文件位置' --expected-revision 1 -w /absolute/workspace
hiro-agent kernel memory-delete writer MEMORY_ID -w /absolute/workspace
```

记忆必须属于真实 Bot，查询、更新、删除都核验 Bot ID；另一 Bot 的 Task 不会收到该记忆。`--query` 是区分大小写的字面子串，`--limit` 默认 100、范围 1–1000。更新可带 `--expected-revision` 拒绝覆盖别人已改过的版本；正文最多 16384 字符。模型没有新增记忆写入工具，CRUD 通过本地 CLI 或公开服务执行。

worker 在恢复后、任务执行前后和服务轮询时扫描持久终态，为 succeeded/failed/cancelled Task 生成最多 4096 字符的确定性摘要，包含真实状态、原因和结果；待审批/提问及待核实 Task 不生成终态摘要。摘要正文与消费凭证在同一 SQL 事务提交。即使进程在 Task 已结算、摘要尚未写入时退出，下次启动也会补齐；重复扫描不覆盖人工编辑。删除摘要保留其消费凭证，后续扫描不会让它复活。

每次真实模型请求重新读取当前 Bot 最新的最多 100 条记忆及已启用技能说明，不缓存上一次投影。记忆作为明确标识的参考 user 消息放在原任务之后，不能授予权限。每个新 Task 使用独立 Session；删除后后续请求不会再次注入该条记录。已有模型输出可能引用旧内容；历史审计仍如实保留当时收到的请求，不提供追溯擦除功能。参考投影不会写成会被 `derive_messages` 重新读取的原始 Session 消息，因此历史 `model.requested` 审计不会令已删记忆重新注入。

默认每次模型上下文最多 64000 个消息 JSON 序列化字符、64 条消息；这是确定的字符界限，不是模型 token 数。可以在 worker 启动时调整：

```sh
hiro-agent kernel worker -w /absolute/workspace --context-max-chars 96000 --context-max-messages 96
```

压缩保留所有 system 指令、原 Task、最新用户答复和最新完整工具组；其他历史按由近到远的完整单元裁剪，assistant 的多个调用与全部对应结果一起保留或省略。记忆按新到旧放入剩余空间，放不下的条目本次省略。原始 JSONL 不截断或重写。如果必保留内容本身超限，本 Run/Task 明确 failed，原因含“超过上下文容量”，并且不会为这次请求调用模型；请增大限制后检查现场再提交新任务。损坏的工具配对、缺失或损坏的已启用 Skill 同样明确失败。

## 按 Bot 启用文本 Skills

把技能文件保存到工作区 `.hiro-agent/skills/inspect.md`（或 `inspect/SKILL.md`），例如：

```markdown
---
name: Evidence inspection
description: 先检查证据，在报告中列出文件位置与尚未验证的事项。
---

# Evidence inspection

阅读有关文件，区分实际验证的事实与推测。
```

```sh
hiro-agent kernel skills writer -w /absolute/workspace
hiro-agent kernel skill-enable writer inspect -w /absolute/workspace
hiro-agent kernel skills writer --enabled-only -w /absolute/workspace
hiro-agent kernel skill writer inspect -w /absolute/workspace
hiro-agent kernel skill writer inspect --include-content -w /absolute/workspace
hiro-agent kernel skill-disable writer inspect -w /absolute/workspace
```

目录中的文本可由所有本地 Bot 选择，启用状态按 Bot 隔离。运行时每轮读取已启用技能的名称和说明；Bot 的 `allowed_tools` 包含 `read_skill` 时，模型可调用 `read_skill({"skill_id": "inspect"})` 读取此 Bot 已启用技能的正文。该工具是普通只读工具，使用同一个执行器并计入工具预算；未启用、其他 Bot 单独启用或已删文件都会拒绝读取。没有自动执行脚本能力。`skill` 输出说明与启用状态，带 `--include-content` 时返回已启用技能的 `{id, content}`。修改文件后下次请求读取新的说明或正文，禁用后下次请求不再注入说明、也不能再次调用工具读取。此前已读取正文的工具结果仍属于真实历史，不会被删除。文件 ID 只允许字母数字、下划线、连字符；拒绝符号链接、非普通文件、重复 ID 和越界路径。

技能说明是文本参考，声称 unrestricted、已获授权、扩大工具白名单等内容没有授权效力。所有实际调用仍通过同一个 ToolExecutor 的实时 Bot 白名单、权限模式、审批和累计预算检查。只读子 Agent 仍使用原有只读工具集合，不继承主 Task 的整份记忆投影。

从成功 Task 显式生成可编辑草稿：

```sh
hiro-agent kernel skill-draft writer SUCCEEDED_TASK_ID -w /absolute/workspace
hiro-agent kernel skill writer DRAFT_ID -w /absolute/workspace
# 阅读并编辑返回的 document.path 后，明确启用：
hiro-agent kernel skill-enable writer DRAFT_ID -w /absolute/workspace
```

草稿从真实持久 succeeded 状态和结果生成，包含来源 Bot/Task/Session；同任务重复请求返回相同 ID，崩溃后可修复未发布文件，已发布的人工编辑不会覆盖。生成操作从不自动启用技能；任务成功也不会自动创建草稿。文件被删除后，显式再次 `skill-draft` 会从最初保存的正文重建，这与记忆删除规则不同。

## 一次性与固定间隔 Routines

```sh
hiro-agent kernel routine-add writer '检查项目并在最终回答中生成报告' --at '2026-11-03T09:00:00+08:00' -w /absolute/workspace
hiro-agent kernel routine-add writer '每小时检查项目状态' --at '2026-11-03T01:00:00Z' --interval-seconds 3600 -w /absolute/workspace
hiro-agent kernel routines --bot-id writer -w /absolute/workspace
hiro-agent kernel routine ROUTINE_ID --bot-id writer -w /absolute/workspace
hiro-agent kernel routine-disable ROUTINE_ID --bot-id writer -w /absolute/workspace
hiro-agent kernel routine-enable ROUTINE_ID --bot-id writer -w /absolute/workspace
hiro-agent kernel routine-tick --max-triggers 10 -w /absolute/workspace
```

`--at` 必须包含 `Z` 或明确偏移量，存储统一为 UTC；不接受无时区的本地时间。省略 `--interval-seconds` 是一次性任务；固定间隔必须是正整数秒。`--disabled` 可创建初始停用计划。`routine` 查询同时返回 `{routine, triggers}`，包含每次计划时间和对应 Task ID；pending 的 Task ID 可能尚未确认。CLI 计划默认使用第一次实际提交时 Bot 的预算，服务 API 也可传显式预算快照。

持续 worker 每轮最多尝试 100 个触发，手动 `routine-tick` 的 `--max-triggers` 范围为 1–1000。所有触发复用 `TaskStore.submit`，source 为 routine，幂等键包含例程 ID 和固定计划时刻；重复 tick、多个进程竞争或提交后崩溃重启只得到同一个 Task。`routine-tick` 只提交，不执行模型；worker 停止时不会自动执行计划，重启后按时间补发。调度/摘要持久化失败会停止 worker 并暴露错误，避免服务继续运行却悄悄失去定时触发。

固定间隔从上次计划时间推进，不按实际轮询时间漂移。积压按最早到期优先有界补发，不跳过停机期间的中间间隔；历史积压很大时可能先于较晚计划被处理。停用保留下一计划时间，重新启用会补发停用期间的间隔。**停用仅阻止新的触发接受；已经持久化为 pending 的触发仍会提交。** 已经创建的 Task 要另外使用 `kernel cancel TASK_ID`。一次性计划提交确认后自动停用，不能再次 enable；需要重跑时创建新计划。修改时间/文本也应停用旧计划并创建新计划；触发历史目前不提供清理 API。

## 原有入口

原交互式 CLI、Session resume 和只读子 Agent 继续工作。顶层 `hiro-agent tasks --session SESSION_ID` 及 REPL `/tasks` 仍展示委派工具日志；持久 Task 队列使用 `hiro-agent kernel tasks`。observer 原 `task_id` 仍表示委派 call ID，持久身份使用独立 `persistent_task_id` 字段。

记忆、文本 Skills、Routines 和跨 Bot 交接均接入持久 Task 入口，使用统一的预算与恢复流程。

## Agent 直接通信

持久任务中的 `send_message` / `read_message` 是真实的本地 Agent 消息接口，复用下述持久收件箱、任务投递和结果回传机制。`init-agents` 新建的标准 Agent 和新安装的管理员具备通信工具；管理员创建通信角色时应显式授予 `list_agents`、`send_message`、`read_message`。已有 Agent 配置不会因更新代码而扩权，可逐个开启：

```sh
hiro-agent kernel enable-communication agent-manager -w /absolute/workspace
hiro-agent kernel enable-communication comm-test-sender -w /absolute/workspace
hiro-agent kernel enable-communication comm-test-responder -w /absolute/workspace
```

该命令保留模型、角色指令、其他权限和历史，权限实际变化时递增版本。接收消息和自动回传最终回答不需要主动发送权限。修改后重启旧 worker，在 TUI 发送新任务；传统非持久聊天暂不提供消息工具。

例如发送方调用 `send_message`：

```json
{
  "target_agent_id": "comm-test-responder",
  "correlation_id": "comm-001",
  "payload": "COMM_TEST|id=comm-001|from=comm-test-sender|to=comm-test-responder|payload=ping",
  "idempotency_key": "comm-001"
}
```

工具返回稳定 `message_id`，只代表消息已持久接受。发送方结束当前发送阶段，不能占住单 worker 轮询等待。接收 Agent 获得消息正文及真实来源、目标和关联标识，实际执行后用最终回答回复，例如 `COMM_ACK|id=comm-001|from=comm-test-responder|to=comm-test-sender|payload=pong`；系统持久保存该回答并为发送方创建回复处理任务，无须手动中继或再次调用 `send_message` 回信。

发送方在回复处理任务中调用 `read_message({"message_id": "MESSAGE_ID"})`，检查 `pending=false`、目标 Agent、`result.state` 及 `result.output` 中的标识、来源、目标和 `payload=pong`。顶层 `correlation_id` 来自原请求，不能单凭该字段判定对方已正确回显。回复缺失、失败或正文不匹配均不能报告通过。普通回复不会触发新一轮互发；主动继续通信仍消耗同一链预算与 hop 限额。消息可通过已有 `kernel handoffs`、`handoff-result` 命令检查。

管理员 → 发送方 → 应答方等多层委派中，发送方的“已发送，等待回复”属于阶段答复。上游消息保持 pending，直到下游结果及发送方的回复核验任务完成，再回传最终报告。`result.response_task_ids` 指明最终报告来自哪些实际执行任务，`result.delegated_results` 包含获准沿委派链回传的下游收发证据。管理员读取自己发给发送方的原消息即可核验这些证据，不能直接读取发送方发件箱中的消息 ID。下游任务或核验任务失败、取消、耗尽预算时，上游收到失败结果；待核实的任务继续保持 pending。多个下游分支全部处理完毕才汇总回传。

读取不存在或无权访问的消息时，`read_message` 返回 `status="unavailable"`、`error="message_not_accessible"`，不透露该消息信息，也不直接终止当前任务。Agent 应改读自己的原消息，或如实报告无法核验。已结束任务不能新增直接消息；重试已经接受的同幂等消息仍返回原消息。

离线集成验证（模型为测试实现，工具、SQLite、worker 和跨重启投递为真实路径）：

```sh
.venv/bin/pytest -q tests/test_agent_communication.py tests/test_message_delegation.py
```

## 双 Bot 收件箱、审阅与回传

为发送 Bot 的 `allowed_tools` 加入 `send_handoff` 和 `read_handoff`，再用 `bot-save` 保存。第二 Bot 也要先保存配置；例如 `reviewer` 只允许 `read_file/search_text`，说明为“检查交接请求及文件，最终回答给出审阅结论”。发送工具属于本地任务编排，和委派一样由显式白名单控制，不额外申请文件写入审批；接收 Bot 的写入仍须经过自己的白名单、权限模式和精确审批。工具不会继承发送方的文件权限，也不会发送外部网络消息。

模型可使用以下普通工具参数：

```json
{"target_bot_id": "reviewer", "text": "请审阅 report.md，指出证据不足处", "idempotency_key": "report-review-001"}
```

这是 `send_handoff` 的参数。相同源 Task 内重试时复用同一 `idempotency_key`，相同内容返回原消息，冲突内容报错。工具结果包含稳定消息 `id`、源 Bot/Task、接收 Bot、chain ID 和已接受交接序号。消息先写入 SQLite，worker 再通过统一 `TaskStore.submit` 为接收 Bot 建任务，源 Task 可以正常结束。它不会占住唯一 worker 等待第二 Bot。

第二 Bot 到达 succeeded/failed/cancelled 后，worker 将结论和失败原因保存到原发送 Bot 的结果收件箱，并为原发送 Bot 建立一个 `source="handoff_result"` 的新任务。此任务包含原消息、原 Task、目标 Task 的关联和结果正文；它与原任务使用同一 chain，但有独立 Session。发送 Bot 可调用 `read_handoff({"message_id": "MESSAGE_ID"})` 读取完整结果；尚未返回时得到 `{"pending": true}`。仅原发送 Bot 能读取返回结果。回传任务本身不会自动再回传；只有模型显式调用 `send_handoff` 才产生新交接。Bot 指令应让它在回传任务中整理审阅结论，按任务内容决定下一步。

CLI 可独立完成同一路径并观察关联；把占位 ID 替换为前一步 JSON 返回的值：

```sh
hiro-agent kernel submit writer '巡检项目，生成报告；写入批准完成后交 reviewer 审阅' --idempotency-key report-review-001 --chain-max-hops 4 --chain-model-requests 32 --chain-tool-calls 48 --chain-timeout-seconds 300 -w /absolute/workspace
hiro-agent kernel worker -w /absolute/workspace
# 查询审批并停止/重启 worker 后也可继续：
hiro-agent kernel interactions --bot-id writer -w /absolute/workspace
hiro-agent kernel reply INTERACTION_ID --approve -w /absolute/workspace
hiro-agent kernel handoffs writer --outgoing -w /absolute/workspace
hiro-agent kernel handoffs reviewer -w /absolute/workspace
hiro-agent kernel handoff-result writer MESSAGE_ID -w /absolute/workspace
hiro-agent kernel chain SOURCE_TASK_ID -w /absolute/workspace
```

也可从已完成的源任务显式发送：

```sh
hiro-agent kernel handoff-send writer SOURCE_TASK_ID reviewer '请审阅 report.md' report-review-001 -w /absolute/workspace
hiro-agent kernel handoff-sync -w /absolute/workspace
```

`handoff-sync` 只对账已保存消息/终态结果和提交任务，不调用模型；worker 平时自动对账。`handoff-result` 返回 `result` 及 `delivery_task`，因此可以区分结果已保存、回传任务 queued、已执行 succeeded，以及因预算不足 failed。原任务可通过返回结果中的 `source_task_id` 定位，第二 Bot 任务是 `target_task_id`，发送方收到结果的任务是 `delivery_task_id`。

每个新根任务自动获得持久 chain ID。默认全链预算等于根 Task 预算，默认最多接受 8 次显式交接；`submit --chain-*` 可设定根链上限。`--chain-max-hops` 计算整条链已接受的发送总数，分支、回传任务中的继续互聊也共用该上限；重试同一消息不增加次数。自动回传不会另扣发送次数，但回传任务实际模型/工具调用和活动时间计入同一链预算。任一 Bot 的新 Task、新 Run、进程重启都只能获得 Task 与 chain 两者剩余额度的较小值，不能重置或提升链上限。审批等待不计活动时间。

预算耗尽仍会保存第二 Bot 的失败结果并生成可查询的回传任务；worker 将无法执行的任务标为 failed，不会为送达结果额外调用模型。结果正文仍保存在发送 Bot 收件箱，可由 CLI 读取。接收任务处于 `needs_reconciliation` 时尚无可信终态，不回传成功或失败结论，必须先按前文人工核实。

| 交接崩溃切点 | 重启对账行为 |
| --- | --- |
| 消息已保存，接收 Task 尚未建立 | 从持久收件箱再次提交，消息 ID 生成固定任务幂等键。 |
| 接收 Task 已建立，消息尚未标记 | 重复统一 submit 返回同一个 Task，再补齐关联。 |
| 接收 Task 已结束，结果尚未保存 | 扫描真实持久终态，以唯一 message ID 写入结果消费账本。 |
| 结果已保存/回传 Task 已创建，送达标记尚未保存 | 以固定结果幂等键补建或取回同一个回传 Task，再记录送达。 |

原始 `send_handoff` 工具已经保存消息、但 JSONL 工具结果未落盘时，源 Task 仍按既有未知副作用规则停止待核实；已持久接受的消息继续对账，不重新发送同一消息。取消源 Task 不会撤销此前已接受的交接；可用 `cancel TARGET_TASK_ID` 单独取消接收任务，取消结果同样回传。旧数据库首次打开时一次性汇总已记 Task 用量，已有幂等提交继续可重试，不退还旧预算。

从仓库根目录运行离线完整演示测试，无需模型凭据或网络：

```sh
.venv/bin/pytest -q tests/test_handoff_pipeline.py::test_routine_write_review_return_survives_restart
.venv/bin/pytest -q tests/test_handoff_pipeline.py tests/test_handoffs.py tests/test_handoff_cli.py
```

第一条通过真实 Agent/工具执行器/worker/SQLite/JSONL，组合定时巡检、已有记忆和启用 Skill、报告、持久审批、重启后精确写入、第二 Bot 读取审阅，以及原发送 Bot 读取回传。其余用例覆盖自动互聊跳数/模型/工具预算、四个交接恢复窗口、旧数据迁移和独立 CLI 进程；内置 `mock` 本身不模拟这套业务流程。
