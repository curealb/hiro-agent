# 持久内核接口与所有权（A）

本文件是 B1/B2 并行开发的共同输入。实际共享类型位于 `src/hiro_agent/kernel_contracts.py`，已有单次执行契约仍在 `contracts.py`。A 只实现数据库基础、Bot 配置与执行权限；本文明确标记的 B1/B2/B3 行为由对应节点实现，不能把协议声明视为已完成的功能。

## 所有权与依赖

| 模块 / 文件 | 写入者 | 边界 |
| --- | --- | --- |
| `kernel_db.py`、`bots.py`、`kernel_contracts.py`、本文件 | A；R1 后冻结，变更由主代理协调 | 基础设施与共同 DTO/协议。B1/B2 只导入。 |
| `durable_tasks.py`、`worker.py`、`tests/test_durable_tasks.py`、`tests/test_worker.py` | B1 | Task/Run SQL、队列、单 worker、恢复状态机。不得改交互或共享运行时文件。 |
| `interactions.py`、`tests/test_interactions.py` | B2 | 提问/审批、答复、一次性授权 SQL。不得改任务 SQL 或共享运行时文件。 |
| `kernel_runtime.py`、必要的 Agent/Executor/Budget/Session/bootstrap/CLI 改动及集成测试 | B3 | 唯一运行时集成者；接入 B1/B2，维持单次 `AgentService.run`。 |
| `memory.py`、`tests/test_memory.py` | C1 | Bot 隔离记忆、终态摘要消费与纯上下文压缩。 |
| `skills.py`、`tests/test_skills.py` | C2 | Markdown Skills、按 Bot 启用和显式成功任务草稿。 |
| `routines.py`、`tests/test_routines.py` | C3 | 持久定时触发，只调用统一 Task 提交入口。 |
| `kernel_context.py`、运行时/CLI 接入及组合测试 | C4 | 每轮新鲜上下文、摘要对账与持续 worker 定时循环；不改 C1/C2/C3 权限边界。 |
| `handoffs.py` 及链预算扩展 | D | 持久双 Bot 收件箱、终态回传与链预算；仅通过 Task API 提交。 |

现有 `tasks.py` 和 CLI `tasks` 仍是委派工具调用的日志投影，不重命名、不改变含义。现有 observer `task_id` 是委派 call ID；持久 Task 必须用独立的 `persistent_task_id` 元数据字段，不得混用。

## 已实现的 SQLite 基础

```python
database = KernelDatabase.for_workspace(workspace: Path)
# 默认文件：workspace/.hiro-agent/kernel.sqlite3
KernelDatabase(path: Path)
database.connect() -> ContextManager[sqlite3.Connection]
database.transaction() -> ContextManager[sqlite3.Connection]
database.migrate(namespace: str, version: int, statements: Sequence[str]) -> None
```

每次连接启用 `foreign_keys=ON`、`synchronous=FULL`、5 秒 busy timeout；数据库使用 WAL。`transaction()` 使用 `BEGIN IMMEDIATE`，成功提交，任何异常回滚；连接在上下文退出时关闭，无需共享长期连接或 `close()`。SQL 参数必须绑定。事务内不得 `await`、等待模型/用户、写 JSONL、运行工具或调用 `executescript`（它会破坏事务边界）。SQLite 失败转换为 `PersistenceError`。

迁移按模块命名空间编号、从 1 连续递增，已应用 SQL 的 SHA-256 必须一致。B1 使用 `durable_tasks` 命名空间，B2 使用 `interactions`；不得修改 `bots` 或对方的表。构造顺序：`KernelDatabase` → `BotStore` → `TaskStore` → `InteractionStore` → `TaskWorker`。B1 Task 引用 bots；B2 交互保存 task/bot 文本 ID，不以 Task 表外键作为启动依赖，故 B2 测试只需真实 BotStore。B2 必须核验 Bot 存在、operation 与 request 的任务/Bot 一致；实际 Task 所属关系由 B3 接收请求前校验。每个模块可单独初始化自己的迁移。

SQL 是 Bot 配置、Task/Run 生命周期、累计预算、交互/授权、后续记忆/调度/收件箱的权威存储。JSONL 是模型与工具实际执行事实、完整原始消息和副作用不确定性的权威证据。两者没有跨存储原子事务。展示 observer 不得充当持久化确认，SQL 写成功也不代表工具执行成功。

## 已实现的 Bot 配置与授权

`Config.bot: BotConfig | None` 默认 `None`，旧配置/CLI 保持原路径。Bot 必须显式提供稳定 `id`（ASCII 字母数字开头，后续字母数字/下划线/连字符，最多 128 字符）。`BotConfig` 还包含 `name`、`instructions` 和规范化去重排序后的 `allowed_tools: tuple[str, ...]`；默认仅 `list_files/read_file/search_text`。空白名单表示全部禁用。BotStore 持久化完整 Config，包括模型、预算、工具限制、子 Agent 设置和原有 `permissions` 模式，凭据本身仍由环境变量读取。

```python
BotStore(database: KernelDatabase)
store.save(config: Config, *, expected_version: int | None = None) -> BotRecord
store.get(bot_id: str) -> BotRecord
store.list() -> list[BotRecord]
store.authority(bot_id: str) -> BotAuthority
store.provider(bot_id: str) -> Callable[[], BotAuthority]
```

`BotRecord`：`bot_id, config, authority, created_at, updated_at`。不可变的 `BotAuthority`：`bot_id, allowed_tools, permission_mode, permission_version, permission_digest`，并有 `allows(tool: str, *, mutating: bool = False) -> bool`。首次保存版本为 1；白名单或 permissions 模式改变递增版本，恢复到旧列表仍产生新版本；仅改名称/说明/模型不改变权限版本。摘要覆盖 Bot ID、规范化白名单及模式。`expected_version=0` 表示必须尚未存在；不匹配抛出 `ConfigurationError`。

`ToolExecutor(..., *, authority: AuthorityProvider | None = None)` 在执行端查询当前权限：已注册但禁用工具返回 `permission_denied/not_started`；未知工具返回 `unknown_tool/not_started`；均不调用 handler。schema 同时过滤，但不是安全边界。审批回调与 `tool.started` 持久写入均可能 yield，之后复查当前 authority，变化返回 `permission_changed/not_started`。已启动的调用在 JSONL `tool.started.data.authority` 记录版本与摘要。运行中已开始的副作用不能靠配置变更撤销。

`ToolExecutionContext.authority` 向子 Agent 传递相同实时权限 provider；子执行器仍只有固定只读注册表，因此子工具权限是 Bot 白名单与既有只读工具的交集。旧调用不传 provider 时行为不变。

首次 Bot Session 写 `session.bot` 绑定；已绑定 Session 不能以另一 Bot 打开。省略 `Config.bot` 续跑已绑定 Session 时，从数据库加载完整保存配置；显式传同一 Bot 的 Config 时按该配置更新 BotStore。初始 system prompt 加入 Bot instructions；已有 Session 的历史 prompt 不被重写。B3 将持久任务 Session 与同一绑定规则接轨。

## 共享数据关系与状态

- 一个 Bot 有多个 Task；每个 Task 创建一个独立根 Session，多个 Run 在此 Session 顺序执行。子 Agent Session 仍是单 Run 内的委派，不是独立持久 Task。
- Task ID、Run ID、Session ID 都是互不替代的稳定标识；领取事务预分配 Run ID，之后 `RunScope.run_id` 必须相同。
- `TaskSubmission`：`bot_id, text, idempotency_key?, source="manual", budget?, chain_id?, chain_budget?, chain_max_hops?, parent_task_id?, metadata={}`。统一入口为 `TaskStore.submit`。幂等唯一键是 `(bot_id, idempotency_key)`；None 每次创建。重复相同请求返回原任务，冲突 payload 必须报错，不能默默换任务文本。所有跨 Bot/定时/CLI 提交复用此入口。
- `TaskRecord`：`id, bot_id, session_id, text, state, revision, source, idempotency_key, chain_id, parent_task_id, metadata, budget, cumulative_metrics, active_run_id, cancel_requested, pending_operation, interaction_id, output, reason, created_at, updated_at`。`revision` 供 CAS/可审计状态变更；默认 budget 来自保存的 Bot Config。`chain_id/parent_task_id` 关联持久链；D 已实现共享累计账本，见下文。
- `DurableRunRecord`：`id, task_id, session_id, worker_id, state, metrics, result?, created_at, started_at?, finished_at?`。状态 `claimed/started/finished/uncertain`；`metrics` 是此 Run 已计入任务的累计快照。`RunClaim` 同时返回 Task 和 Run。
- Task 状态：`queued → running → succeeded/failed/cancelled/waiting_approval/waiting_input/needs_reconciliation`。答复后等待态经幂等 wake 回到 queued；恢复可把无运行证据的 claimed 重新排队。终态不会自动再次运行。`needs_reconciliation` 不可领取，必须显式 resolve。
- `OperationIntent` 是冻结对象：`id, task_id, bot_id, session_id, run_id, call_id, tool_name, arguments_json, permission_version, permission_digest`。参数先经过工具 Pydantic schema 验证并加入默认值，再用 `canonical_arguments` 排序键、紧凑 JSON、禁止 NaN/Infinity。`binding` SHA-256 覆盖 task/Bot/tool/规范参数/权限版本/权限摘要；原 Run/call ID 用于追溯，不阻止在新 Run 中匹配一次性授权。

## B1：必须实现的入口

生产实现命名 `durable_tasks.TaskStore(database: KernelDatabase, bots: BotStore)`，满足 `TaskStoreService`：

```python
submit(request: TaskSubmission) -> TaskRecord
get(task_id: str, *, bot_id: str | None = None) -> TaskRecord
list(*, bot_id: str | None = None, states: tuple[TaskState, ...] = ()) -> list[TaskRecord]
cancel(task_id: str, *, bot_id: str | None = None) -> TaskRecord
claim(worker_id: str) -> RunClaim | None
mark_started(run_id: str) -> DurableRunRecord
checkpoint(run_id: str, metrics: RunMetrics) -> TaskRecord
finish(run_id: str, result: RunResult, *, pending_operation: OperationIntent | None = None,
       interaction_id: str | None = None,
       waiting_for: Literal["approval", "question"] | None = None) -> TaskRecord
wake(task_id: str, interaction_id: str) -> TaskRecord
reconcile(evidence: RunEvidence) -> TaskRecord
runs(task_id: str) -> list[DurableRunRecord]
resolve(task_id: str, *, action: Literal["retry", "fail", "cancel"], note: str) -> TaskRecord
```

`get/cancel(bot_id=...)` 拒绝所属 Bot 不符；查询默认不自动暴露给另一 Bot 的模型。领取在一个立即事务中 CAS queued→running、插入 claimed Run、写 Task.active_run_id，之后才可以创建/执行 Run。单 worker 进程锁归 `worker.py`，建议工作区 `.hiro-agent/worker.lock`；锁冲突拒绝第二 worker，异常退出由 OS 释放。只允许一活动 Run，但取消 CLI 是另一进程的持久 `cancel_requested` 写入：queued/waiting 可直接取消，running 设置标志，由 worker 调用 runner.cancel 并结算；不能简单把 running 标为取消同时让工具继续。

`checkpoint` 接收**当前 Run 从零累计**的 metrics，按字段增量一次性计入 Task；禁止倒退和重复计费；`finish/reconcile` 复用此规则，在同一 SQL 事务内合并最后 metrics、存 RunResult、清 active_run、更新 Task。模型/工具次数含未成功尝试，父 Run 已含子调用，不能再次累加 child metrics。`elapsed_ms` 是运行期间的 active time；审批/提问等待期间 Run 已结束，不计等待时间。新 Run 的 Budget 从 `Task.budget - cumulative_metrics` 得到剩余额度，不使用旧 `continue_run()` 自动重置预算的路径。启动/请求/工具阶段的持久日志可重建尚未 checkpoint 的次数；不允许重启获得新额度。D 在这些同一事务边界附加 chain 计费。

`finish` 的 `waiting_for` 必须与 interaction/pending intent 一致：approval 要求精确 intent 和 interaction；question 要求 interaction。`wake` 仅在目标 interaction 与 Task.interaction_id 一致时转 queued；同一 wake 重复调用对 queued/running/terminal 无再次入队效果，过期交互不能唤醒新的等待。醒来仍保留 intent，交由 B3 验证当前权限并消费授权。拒绝审批同样产生答复并醒来，但不得执行该 intent。

正常终态映射为 Run `completed→succeeded`、`failed→failed`、`canceled→cancelled`。Run paused 只有携带合法 `waiting_for` 时进入等待态；预算耗尽、模型截断或其他没有可处理交互的暂停保留 reason 并结束为 failed，不自动无限重试。发现 unknown 工具结果时优先 needs_reconciliation。cancel_requested 已设定时不得因随后回答交互而重启任务。

`RunEvidence`：`run_id, session_id, started, result?, metrics, unknown_side_effect, pending_operation?, interaction_id?, waiting_for?`。`reconcile` 验证 Task/Run/Session 关联，对相同证据重复调用幂等；未知副作用优先进入 needs_reconciliation，即使合成 recovery RunResult 是 paused。`resolve` 仅对 needs_reconciliation 生效，并记录非空人工核实 note；retry 是明确决定，不是重启默认策略。

`worker.TaskWorker(database: KernelDatabase, tasks: TaskStoreService, runner: TaskRunner, *, worker_id: str | None = None, poll_interval: float = 1.0)` 提供 `async tick() -> bool`（处理一个可领取任务）、`async recover() -> list[TaskRecord]`、`async serve(stop: asyncio.Event) -> None`、`async close() -> None`。B1 用可控 fake runner 验证调度，不引入 Agent 或 CLI 的集成改动。`TaskRunner` 共享协议：

```python
async run(claim: RunClaim, checkpoint: Callable[[RunMetrics], None]) -> TaskRunOutcome
async recover(claim: RunClaim) -> RunEvidence
cancel(task_id: str) -> None
async close() -> None
```

`TaskRunOutcome` 字段为 `result, pending_operation?, interaction_id?, waiting_for?, unknown_side_effect=False`。最后一项是 B1 交付前经主代理批准的向后兼容扩展：正常运行也可能留下 outcome=unknown 的工具结果，不能只在重启恢复时识别。B3 适配器从执行证据明确设置该标志，禁止按 reason 文本猜测。worker 收到 True 时构造同 Run/Session、started=True、带结果/metrics/交互关联及 unknown_side_effect=True 的 `RunEvidence` 并调用 `reconcile`，而不调用 `finish`；任务保持 needs_reconciliation，即使 RunResult 是 canceled/paused/completed。默认 False 继续调用原 finish 接口。B3 的适配器直接、可靠地调用 checkpoint，不能依赖吞异常的 EventBus 来确认写入。

## B2：必须实现的入口

生产实现命名 `interactions.InteractionStore(database: KernelDatabase, bots: BotStore)`，满足 `InteractionService`：

```python
create(request: InteractionRequest) -> InteractionRecord
get(interaction_id: str, *, bot_id: str | None = None) -> InteractionRecord
list(*, bot_id: str | None = None, task_id: str | None = None,
     state: InteractionState | None = None) -> list[InteractionRecord]
reply(interaction_id: str, reply: InteractionReply) -> InteractionRecord
ready() -> list[InteractionRecord]
consume_authorization(interaction_id: str, operation: OperationIntent, run_id: str) -> bool
consume_answer(interaction_id: str, task_id: str, run_id: str) -> str | None
cancel_for_task(task_id: str) -> None
```

`InteractionRequest`：`task_id, bot_id, kind: approval|question, prompt, idempotency_key, operation?`。approval 必须带匹配 task/Bot 的精确 OperationIntent。唯一创建键 `(task_id, idempotency_key)`；重复相同 payload 返回原记录，冲突报错。`InteractionReply`：`approved: bool | None, text: str | None`；审批必须有 bool、提问必须有非空 text，不能把字符串 yes 当 bool。`InteractionRecord`：`id, request, state, reply?, consumed_by_run_id?, created_at, answered_at?, consumed_at?`；状态 `pending/answered/consumed/cancelled`。

答复事务只修改 B2 自己的表。首次合法答复 pending→answered；相同答复无论 answered/consumed 都幂等返回原记录，不重新发放授权；冲突答复报错；cancelled 不可答复。`ready()` 持续返回已回答未消费的记录，作为可重复读取的持久唤醒清单。B3 每次启动及 polling 调用 `ready()`，对相应 Task 执行 `wake(task_id, interaction_id)`。因此“审批已保存/Task 未唤醒”崩溃后会再次唤醒；不需要 B2 写 B1 的 SQL，也不依赖易丢失的内存通知。

一次性授权消费在 B2 一个立即事务内验证：record 属于同 Task/Bot，approved=True，state=answered，operation.binding 精确一致，数据库 Bot 当前权限版本/摘要与 intent 一致，工具仍在白名单且写入模式未收紧。成功原子写 consumed、消费 run ID 和时间并返回 True；任何不匹配或重复消费返回 False，绝不能第二次返回 True。消费失败不能自动换用普通审批 callback。若消费后尚未启动工具便崩溃，恢复不返还该授权；可明确请求新审批，或在有证据证明未执行时生成新的 intent/交互。`consume_answer` 同样一次消费，返回已保存文本或 None；`cancel_for_task` 只取消尚未消费的记录，不擦除历史授权审计。

## B3 集成顺序与崩溃窗口

挂起不是普通工具拒绝。需要保存交互与精确 intent，结束旧调用为 `not_started`，将同一 intent/interaction 的关联放入可对账的 Run checkpoint/终态记录，然后以 paused RunResult 返回 Task 层，停止继续模型循环。原 Session 的每个 assistant tool call 均有结果；Task 保存 intent 的行为不保持旧 call 未结算。

批准后先 wake，再领取**新 Run**。B3 在现有 `RunScope` 增加可选持久 task/准备动作上下文，或使用只属于 B3 的等价适配器，保持 `AgentService.run(task, scope=...)` 单次入口。必须在新 Run 内为持久 intent 产生新的唯一 tool call ID、记录合法 assistant 声明，经过同一 ToolExecutor 执行**已保存的规范化参数**；不得依靠模型重新生成相同工具调用，不得绕过 schema/白名单/版本/一次性消费检查，也不得复用已结算旧 call。允许随后再调用模型形成回答。问题答复则作为明确的新输入写入新 Run，消费记录和调用关联可追溯。

| 崩溃窗口 | 可观察证据 | 必须恢复的行为 |
| --- | --- | --- |
| SQL 已领取，JSONL Run 尚未开始 | claimed Run ID，Session 无该 run.started | 无工具能已执行；结束/回收占位 Run、幂等重新排队，保留累计预算。 |
| JSONL Run 已结束，SQL Task 未更新 | 相同 Run ID 的 run.ended 与 metrics | 读取验证后的终态，幂等 finish/reconcile；不能重新执行，不能重复计费。 |
| 审批/答复已保存，Task 尚未唤醒 | B2 ready() 仍有已答复记录 | 重复 wake，最多一个可领取任务；授权仍最多消费一次。 |
| tool.started 已落盘，tool.result 未落盘 | 恢复结算 outcome=unknown | Task→needs_reconciliation、禁止自动领取或重放；要求显式核实/resolve。 |

恢复先获得 worker/session 锁，先恢复 child delegation，再调用现有 session 恢复以维护配对；之后用真实 JSONL 构造 RunEvidence。损坏/忙碌 journal 必须阻塞相关恢复，不能当成无日志重新执行。Session 不存在与合法空 Session 是不同情况：已 started 的 SQL Run 缺日志时按 uncertain 处理。运行时持久化失败停止 worker 本轮，不把失败当成普通可重试。

R2 必须通过真实子进程证明 CLI 进程退出不停止独立 worker，并注入上述四个崩溃窗口；模块 fake runner 测试不能替代这些集成证明。

## C1/C2/C3/C4 已实现接口

`KernelRuntime(workspace, *, model_factory=None, context_max_chars=64000, context_max_messages=64)` 公开 `bots/tasks/interactions/memory/skills/routines`。其中 memory 为 `MemoryStore(database, bots, tasks)`，skills 为 `SkillStore(database, bots, tasks, workspace / ".hiro-agent" / "skills")`，routines 为 `RoutineStore(database, bots, tasks)`；三个模块各自管理独立迁移命名空间，只通过公开 Bot/Task API 读取归属和提交任务。

| 服务 | 公开接口与约束 |
| --- | --- |
| MemoryStore | `create/get/list/update/delete(bot_id, ...)` 按 Bot 隔离；update 支持 expected_revision。`remember_task(bot_id, task_id)` 与 `sync_terminal(bot_id)` 读取终态和最终 Run，原子保存摘要及唯一消费凭证。删除摘要保留凭证，重放不会复活。 |
| 纯上下文 | `prepare_context(messages, *, bot_id, current_task, memories=(), max_chars=64000, max_messages=64)` 返回 messages/char_count/omitted_messages/omitted_memories；不访问日志或数据库、不拆工具组。`ContextCapacityError` 表示必保留内容超过硬容量。 |
| SkillStore | `load/get/list/settings/enable/disable/set_enabled/describe/content/injection` 管理安全本地 Markdown 和 Bot 启用状态；说明/正文不能改变工具权限。`draft_from_task(bot_id, task_id)` 必须显式调用，验证真实 succeeded 状态，以 `(bot_id, task_id)` 保存唯一创建意图并幂等发布草稿，从不自动启用。 |
| RoutineStore | `create(RoutineCreate)`、`get/list/enable/disable/set_enabled/triggers` 管理一次性或固定正整数秒间隔计划；`tick(*, now=None, max_triggers=100)` 只提交统一 Task，单次上限 1–1000。时间必须含时区，按 UTC 规范化。 |

`create_harness(..., context: ContextService | None = None)` 提供向后兼容装配接缝，默认仍为旧 PassthroughContext。持久 Runner 提供 `KernelContext`，在每轮真实模型请求时重新读取当前 Bot 记忆与 enabled Skill 说明，再调用纯压缩函数。原始任务保持首个 user 消息；系统指令、原任务、最新用户消息及最新完整工具组受保护，记忆和历史只占剩余容量。超限错误通过 Agent 已有 HiroError 处理形成清晰 failed Task；不会削掉保护内容或通过重置预算重试。投影不写成 Session 历史消息；既有 `model.requested` 如实记录当时请求，但不参与 `derive_messages`，不会缓存或复活已删记忆。

持久 Runner 注册普通只读 `read_skill({"skill_id": ...})` 工具，闭包固定当前 Task.bot_id，调用 `skills.content` 验证已启用状态。Bot 白名单须明确包含 `read_skill`；调用仍经相同 ToolExecutor 的 schema、权限和预算流程，结果记入原始工具日志。加载正文不会执行脚本或授予其他工具权限，也不向子 Agent 扩大注册表。

KernelWorker 在 recover、tick 前后及持续服务轮询调用摘要对账；SQL Task 结算后崩溃仍可在重启补上摘要，observer 不承担持久确认。`serve` 另有受同一 worker 生命周期管理的定时协程，长 Run 等待期间也能提交到期 Task，实际执行仍由唯一 TaskWorker 串行领取。调度失败设置停止信号、协作取消活动 Run，关闭时先收敛调度和执行再释放进程锁，错误向外传播。

Routine 触发先持久接受 pending，再在事务外 submit，最后确认 Task ID 并推进计划；唯一 `(routine_id, scheduled_at)` 配合 Task 幂等键覆盖重启和竞争。固定间隔按 scheduled time 推进，有界补发全部积压。停用阻止新接受，不撤销已有 pending；pending 即使停用仍对账提交，已创建 Task 另行取消。恢复启用后补发停用期间积压；完成的一次性计划不能重新启用。

MemoryRecord 保存 `id/bot_id/content/kind/task_id/final_run_id/revision/created_at/updated_at`。正文最多 16384 个 Unicode 字符；任务摘要最多 4096 字符。记忆 CRUD 按 Bot 校验归属，revision 用于乐观并发控制；摘要与唯一消费凭证在同一事务保存，删除摘要保留凭证。

SkillDocument 保存 `id/name/description/content/path/digest`，digest 为原文件字节 SHA-256。技能文件仅接受 `<id>.md` 或 `<id>/SKILL.md`，同 ID 多文件报错；ID 为 1–128 位 ASCII 字母数字、下划线或连字符且首字符为字母数字。读取限定在初始化的技能目录内，使用目录描述符、`O_NOFOLLOW` 和普通文件检查，拒绝符号链接及 FIFO；内容必须为 UTF-8，最多 1 MiB。文件编辑在下次读取生效，启用状态按 Bot 隔离。

RoutineRecord 保存定义、当前 enabled、下次/上次计划时间和上次 Task ID；RoutineTrigger 保存 `routine_id/scheduled_at/submission/task_id`。`task_id=None` 表示持久接受后尚未确认提交，不能据此断定 Task 不存在。接受 pending、幂等提交 Task、确认 Task ID 并推进计划分别持久化，重启后对账 pending。实际 CLI 见[内核使用指南](USAGE.md)。

## D：持久交接与共享链预算

所有定时和交接任务来源只调用 TaskStore.submit；交接幂等键包含持久 message ID，按接收 Bot 隔离。完成结果保存在 Task/Run SQL 中，可重复查询终态；回传必须用持久唯一消费键幂等处理，不能只订阅瞬时 observer。D 的链预算必须与 Task checkpoint/finish 在同一数据库事务更新或使用同样可证明的幂等账本；新任务/new Run/new Bot 都不能重置 chain 上限。


`KernelRuntime.handoffs` 为 `HandoffStore(database, bots, tasks)`；命名空间 `handoffs` 保存消息与终态消费账本。公开接口：

```python
send(bot_id, source_task_id, target_bot_id, text, *, idempotency_key) -> HandoffMessage
get(bot_id, message_id) -> HandoffMessage
list(bot_id, *, outgoing=False) -> list[HandoffMessage]
result(bot_id, message_id) -> HandoffResult | None
results(bot_id) -> list[HandoffResult]
sync() -> None
```

消息包含 `id, source_bot_id, source_task_id, target_bot_id, text, idempotency_key, chain_id, hop, target_task_id?, created_at`，唯一键为 `(source_task_id, idempotency_key)`。消息与链 hops_used 增量在同一立即事务内提交，幂等重试不新增次数。源 Task 的 Bot、目标 Bot 存在、发送白名单和全链跳数必须有效；失败/取消/待核实源任务不接受新发送。消息只供源或目标 Bot 查询，结果只供源 Bot 读取。

结果包含 `message_id, source_bot_id, source_task_id, target_bot_id, target_task_id, chain_id, state, output, reason, delivery_task_id?, created_at`，message_id 唯一。Task succeeded/failed/cancelled 后持久保存结果；needs_reconciliation 不当作可信终态。收件任务使用 `source="handoff"` 和 `handoff:MESSAGE_ID` 幂等键；回传任务使用 `source="handoff_result"` 和 `handoff-result:MESSAGE_ID`，以目标任务为 parent，metadata 保留原 source_task_id/target_task_id/message_id。两次提交都只调用 TaskStore.submit，均在消息/结果写入事务之外执行。提交成功/标记失败后，幂等键关闭重启窗口；送达标记和唯一结果行就是终态消费凭证。回传任务不会自动产生另一结果回传。

Runner 注册 `send_handoff(target_bot_id, text, idempotency_key)` 和 `read_handoff(message_id)`；身份来自当前 Task 闭包，模型不能传入源 Bot/Task 或链上限。它们复用 schema、预算、执行端实时白名单及 JSONL，不给只读子 Agent 增加注册表。发送是本地任务编排，不标为文件修改；接收方执行受自己的权限约束。未知工具副作用策略不变，已接受消息不因源任务取消或未知结算而撤销。

`TaskStore.chain(task_id, *, bot_id=None) -> ChainRecord` 返回 `id/budget/max_hops/hops_used/cumulative_metrics`。新 TaskSubmission 可用 `chain_budget: BudgetConfig | None`、`chain_max_hops: int | None` 设置新根链；默认使用根 Task 预算与 8 次总发送。已存在真实 parent 的子任务强制继承其 chain_id，拒绝任何 chain 上限覆盖或另换 chain ID。已有 chain 不接受上限覆盖。历史接口允许不存在于本库的 parent_task_id 作为外部来源文本，D 保留该兼容能力；真实交接总是验证源 Task，模型无法利用此入口。

TaskStore 初始化 `task_chains` 命名空间及唯一 task_id 的 `chain_members`，为旧 Task 分配 chain_id 并只汇总一次已有 cumulative_metrics。旧 submission_json 先按当前 DTO 补齐可选默认值再比较，以兼容升级前幂等请求。每次 checkpoint/finish/reconcile 以 Run 旧快照为基准计算字段差值，Task、Run 与 chain 在同一 SQL 事务更新；父指标包含子 Agent，不再单独加子指标。新的执行 Budget 为 Task 与 chain 剩余额度逐项取 min；领取时已耗尽则 failed。链 hops_used 是全链接受发送数，不是可由新 Task 归零的局部深度；所有回传执行也计入模型/工具/活动时间。等待期间无 Run 活动时间。

KernelWorker 在摘要对账位置先对账交接，覆盖启动恢复、tick 前后和持续轮询。结果已持久保存而链无剩余额度时仍可 CLI 读取，回传 Task 将成为可观察 failed；不得为送达绕过预算执行模型。详细 CLI、停止/取消语义和离线演示见 USAGE.md。
