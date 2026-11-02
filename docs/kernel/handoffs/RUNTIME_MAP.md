# 现有运行时映射：A 契约与 B3 集成输入

范围：只读检查现有源码得到的执行边界与回归约束；未修改生产代码、未运行测试。工作区与其祖先目录未发现适用的 `AGENTS.md`。下列行号是检查时位置，后续修改可能移动。

## 文件与入口

| 关注点 | 现有位置与接口 |
| --- | --- |
| 单次执行入口 | `src/hiro_agent/contracts.py:267`：`AgentService.run(task: str, *, scope: RunScope | None = None) -> RunResult`、`continue_run()`、`cancel()`、`close()`、`status`。 |
| 装配 | `bootstrap.py:38`：`create_harness(workspace, config, session_id=None, approval=None, observer=None) -> Harness`；`Harness` 持有 runtime、agent、session、recovered。安装顺序为 session → model → permission → tools → context → 可选 subagents → agent。 |
| Agent 生命周期 | `agent.py:80`：`run` → `_invoke` → `_run`。同一 Agent/Session 同时只允许一个 Run；未传 scope 时创建新的 UUID、Budget、Cancellation。 |
| 工具执行端 | `tools.py:280`：`ToolExecutor.execute`；`tools.py:168`：`execute_many`。`ToolRegistry.restricted(names)` 同时限制 schema 和执行定义。 |
| 权限 | `permissions.py:8`：`PermissionPolicy.approve(tool, arguments, mutating)`；`contracts.py:245`：`PermissionService`。当前 `ApprovalCallback` 仅接收工具名和参数，不接收 Task/Run/权限版本。 |
| 预算 | `budget.py:12`：`Budget`；`contracts.py:197`：`RunScope`；`execution.py`：`cancellable`。 |
| JSONL | `session.py:488`：`JsonlSession`；`session.py:778`：`_recover_session`；`session.py:260`：模型消息投影 `_messages`。 |
| 子 Agent | `subagents.py:107`：`SubagentManager`；`subagents.py:287`：`recover_delegations`。 |
| 当前 CLI | `cli.py:396` 创建 harness；`cli.py:469` 根据 `/continue` 调用 `continue_run` 或 `run(text)`；运行时 SIGINT 调用 `agent.cancel`。 |
| 既有 tasks 模块 | `tasks.py:69`：`summarize_tasks` / `list_tasks` 只读投影 **delegate_task 工具调用**；它不是持久任务队列。保留其原有语义。 |

表中未写目录的文件均位于 `src/hiro_agent/`。

## 现有执行顺序与不可破坏的边界

1. `create_harness` 打开根 Session，先 `recover_delegations`，再 `recover_session`，最后补充缺少的 system prompt。指定 Session 的恢复使用 `create=False`，不能误建已删除的 Session；含 `subagent.created` 的子会话禁止直接续跑。
2. Agent 持久记录 `run.started`、`task.received` 或 `task.continued`，随后每次模型请求先记录 `model.requested`。完整 `assistant.message` 落盘后才执行工具。流式 delta 只是展示数据，不进入原始对话。
3. 工具执行依次为：预算 claim → 持久 `tool.called` → schema 校验 → 权限检查 → 取消/截止时间复查 → 持久 `tool.started` → handler → 持久 `tool.result`。工具副作用不得提前到 `tool.started` 的持久确认之前。
4. 所有已声明工具必须有终态结果，才能追加下一个 user/assistant 消息。`settle_tool_calls` 将未启动调用结算为 `not_started`，已启动但无结果调用结算为 `unknown`。恢复不能自动重放模型或工具。
5. Agent 记录 checkpoint 和 `run.ended` 后返回；返回 metrics 与持久终态记录一致。`RunResult.persistence_failed=True` 表示持久化失败，Agent 被标记为 broken，后续必须重新打开 Session。
6. `continue_run()` 只接受有已记录任务且最后 Run 未完成的情况，创建全新 Run/Budget，记录 continuation 而不重复原任务文本。若新 Run 在 `task.received` 前崩溃，不能错误续跑更早的已完成任务。
7. 模型 finish reason 为 `length` 时暂停，绝不执行该截断响应声明的工具。模型错误不会自动重试。

## A：契约与身份、权限的准确接缝

- `RunScope` 已含 `run_id`、budget、cancellation、root/parent session/run ID、role、`call_id`。已有 observer 数据中的 `task_id` 实际等于委派工具的 `call_id`；`TaskSummary.task_id` 也是该 ID。新增持久 Task 身份必须避免混用这些字段。
- `AgentService` 的主执行入口可继续作为单次 Run 边界；Task、Bot 和累计预算需要独立关系。现有 `continue_run()` 没有 scope 参数，并且默认重置预算，不能直接代表持久 Task 的累计预算续跑。
- `PermissionPolicy` 对非 mutating 工具直接批准；mutating 工具在 read-only 或无 callback 时拒绝；只有 callback 返回字面值 `True` 才批准本次调用。当前 callback 没有持久授权上下文；A 的交互契约需明确 Task 身份、规范化参数、权限版本和一次性消费信息如何到达权限执行端。
- schema 过滤之外，`ToolExecutor.execute` 的定义查找处是强制白名单的重要边界；未知工具当前返回 `unknown_tool` / `not_started`。禁用工具不得因模型提供了已知工具名就进入 handler。
- `SubagentManager` 构造时复制 `registry.restricted(READ_TOOLS)`，其中 `READ_TOOLS = (list_files, read_file, search_text)`；新 Bot 白名单需与此约束兼容，不能让子 Agent 越权，也不能把 Bot 禁用工具仍留在子 registry 中。
- `SessionEvent.type` 是字符串；目前未知的命名空间事件不进入模型投影，但只要携带非空 `run_id`，验证器仍要求引用已开始的 Run。额外元数据不能破坏工具配对规则。
- `EventBus` observer 同步调用，异常被吞掉；它适合展示，不能作为 SQLite 状态更新已经持久化的确认。

## B3：挂起、恢复与集成需要保留的行为

- 当前权限接口只能返回 bool；拒绝会产出 `permission_denied` 并继续模型循环，没有持久等待状态。挂起路径需明确退出到 Task 层，不能伪装成普通权限拒绝后让模型继续尝试。原工具仍应在 `tool.started` 前结算为未执行，精确操作意图由 Task/交互层保存。
- 批准后的新 Run 不应复用旧调用的 tool result 或绕过现有 executor；旧 `call_id` 在整个 Session 中必须唯一，已有结果不可重复落盘。授权与新的执行尝试之间的匹配/消费必须使用 A 的契约。
- `Budget` 限制模型请求、工具调用和单调时钟经过时间。子预算向各祖先累计一次，所有容量同步检查后才 charge；elapsed 不相加。权限 callback 的等待目前计入 Run timeout，因此持久审批等待需要先结束 Run，再等待；Task 累计额度应跨新 Run 保存。
- `Agent._invoke` shield 内部运行；调用方取消被转成协作取消并等待持久确认与清理。`close()` 同样等待停止。不要用外层直接取消任务的方式跳过收尾。
- `Cancellation` 保留最初中断原因，持久化失败优先于普通取消；root 预算耗尽会通知共享 token，停止队列和运行中的子 Agent。不能把 deadline 或 root 耗尽改报为用户取消。
- JSONL 使用独占非阻塞文件锁，append 只有 fsync 成功才确认；取消 append 不会撤销已排队写入；close 排空 writer 后释放锁。第一次磁盘失败使 Session 及后续排队写入全部失败。
- JSONL 只修复不完整的最后一条 JSON；必须先验证所有完整记录和引用。完整损坏记录、内部损坏、错误序列不能静默修复。
- `recover_session` 将悬空模型请求记为 interrupted，将未结束 Run 记为 paused；恢复指标根据持久记录重建，避免重复统计及把离线时间算入执行时间。SQLite 对账应读取这些持久终态，而不能把内存事件当作依据。
- `recover_delegations` 必须先于根恢复：读取子终态，校验父子双向 linkage，然后填父 completion，绝不重跑子任务。忙碌或损坏的子 journal 会阻止恢复，不应吞掉后继续执行父任务。

## 子 Agent 兼容性

- 子 Agent 只有独立历史和显式 task 文本，看不到父对话；角色仅 explorer/reviewer，执行 read-only policy，禁止递归委派。
- 父 `delegation.started` 落盘后才创建子 Session；子 `subagent.created` 保存同一 `DelegationLink`。先写子 `subagent.finished`，再写父 `delegation.finished`。子持久化失败必须阻止父后续工具。
- 子 Agent 共享父 cancellation 和祖先预算，本地 cap 只暂停本地；根 cap 会停止根任务。角色 model 与主 model 可以不同。
- 仅相邻 `delegate_task` 工具可有界并行；普通工具是执行顺序边界，必须等待之前的委派组完成。所有 child 清理和文件锁释放后，父执行才继续/返回。
- 当前 root lineage 在子创建时取父 session/run；因递归被禁止，这是现有正确行为，不能把新的跨 Bot Task 链等同于递归子 Agent。

## 优先保留的回归测试

| 测试文件 | 关键约束 |
| --- | --- |
| `tests/test_agent.py` | continuation 新预算/不重复任务、并发拒绝、审批取消、截断工具不执行、持久化 poisoning。 |
| `tests/test_tools.py` | 每次授权/default deny、执行前持久 started、返回前持久 result、受保护路径、拒绝不产生副作用。 |
| `tests/test_session.py` | fsync/锁、尾部修复范围、取消 append 仍完成、恢复幂等、恢复指标。 |
| `tests/test_cancellation.py` | durable append 期间外部取消后 Session 仍可用。 |
| `tests/test_shared_runs.py` | 祖先额度原子 claim、child deadline、清理等待、持久错误优先、终态 metrics 一致。 |
| `tests/test_subagents.py` | 工具与历史隔离、根预算共享、本地 cap、子持久错误停止父任务。 |
| `tests/test_subagent_recovery.py` / `tests/test_delegation_journal.py` | 崩溃切点只恢复不重放、忙碌/损坏子拒绝、linkage 校验、completion 与工具结果配对。 |
| `tests/test_parallel_delegation.py` | 并发上限、ordinary-tool 顺序边界、根耗尽停止 admission、并行清理、多个崩溃子只恢复一次。 |

完整项目检查命令与门槛以 `docs/kernel/REQUIREMENTS.md` 和当前 `pyproject.toml` 为准；本交接不声称任何测试已通过。
