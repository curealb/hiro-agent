# D：双 Bot 持久交接与共享链预算

2026-11-02。基准 Git revision：`81f42f90b932e61151f399d423fa9a45e55a97e7`。在 R3 PASS 后启动。实现、自有离线验证及文档由 D 节点负责；本记录不能代替全新独立 R4 的 PASS，不自行宣布最终交付。README 留给主代理在 R4 PASS 后更新。

## 文件和批准后的接口扩展

- `src/hiro_agent/handoffs.py`：持久消息、Bot 收件箱、终态结果消费、回传任务及四类重启切点对账。消息和结果都不依赖进程内回调。
- `src/hiro_agent/chains.py`：持久 chain DTO 存取、成员迁移凭证、用量增量及链预算耗尽判断。SQL 操作使用 TaskStore 传入的同一事务连接。
- `src/hiro_agent/kernel_contracts.py`：TaskSubmission 添加可选 `chain_budget/chain_max_hops`；新增 `ChainRecord`。原 Task 状态、Run/worker 协议不变。
- `src/hiro_agent/durable_tasks.py`：创建根链、已知 parent 继承、旧 Task/提交 payload 迁移兼容；checkpoint/finish/reconcile 同事务计入链；claim 同时检查链预算；新增 `chain(task_id, *, bot_id=None)`。
- `src/hiro_agent/kernel_runtime.py`：装配 HandoffStore；模型 `send_handoff/read_handoff` 工具；Run Budget 取 Task 与链剩余额度较小值；在既有 worker 对账位置扫描消息/结果。
- `src/hiro_agent/kernel_cli.py`：`handoff-send/handoffs/handoff-result/handoff-sync/chain`；根 `submit --chain-*` 限制参数。
- `tests/test_handoffs.py`：9 个事务/切点/迁移/预算/隔离用例；`tests/test_handoff_pipeline.py`：6 个真实闭环、自动互聊和工具权限用例（限定测试文件的辅助代理实现）；`tests/test_handoff_cli.py`：1 个真实独立进程 CLI 流程。
- `docs/kernel/USAGE.md`、`docs/kernel/CONTRACTS.md`：运行命令、接缝、链与回传约定；本文件为 D 实际交接记录。

公共 DTO、TaskStore 与运行时扩展在编辑前经主代理批准。没有修改既有独立验收测试、README、TaskWorker、B2 交互、C1/C2/C3 模块或依赖声明；保留原一次执行 AgentService 和只读子 Agent 工具集合。

## 消息与回传协议

```python
kernel = KernelRuntime(workspace)
message = kernel.handoffs.send(
    "writer", source_task_id, "reviewer", "请审阅 report.md", idempotency_key="review-001"
)
kernel.handoffs.sync()
message = kernel.handoffs.get("writer", message.id)
result = kernel.handoffs.result("writer", message.id)
chain = kernel.tasks.chain(source_task_id, bot_id="writer")
```

发送方身份固定来自真实 Task，Bot 白名单须包含 send_handoff；模型不能传源 Bot/Task 或 chain limits。源 Task 与消息、源 Bot 与结果均检查归属，第三 Bot 不能读取。发送的 `(source_task_id, idempotency_key)` 唯一；相同请求返回同一消息，不扣第二次 hop，冲突内容报错。持久接受和 hops_used 增量同事务提交。发送是本地编排工具，不额外索要文件审批；第二 Bot 的实际工具仍受自己的权限/审批控制。

收件箱消息先保存，再由统一 TaskStore.submit 建 `source=handoff` 的接收任务，固定幂等键 `handoff:MESSAGE_ID`。原发送 Task 可结束，不占唯一 worker 等待接收方。目标 succeeded/failed/cancelled 后，message_id 唯一的结果行保存真实终态、output、reason 和源/目标 ID；needs_reconciliation 不虚构终态。

结果保存后，统一 submit 建 `source=handoff_result` 的回传任务，固定幂等键 `handoff-result:MESSAGE_ID`，Bot 为原发送者，parent 为目标任务，metadata 保留原 source_task_id、target_task_id、message_id。回传任务与原任务有独立 Session，沿用同一 chain；实际模型读取任务文本和 read_handoff 工具中的结果，证据证明到达原发送 Bot。回传任务不会自动再生成回传；显式再 send_handoff 才产生下一消息。

`target_task_id`、`delivery_task_id` 为持久提交消费凭证。消息已保存/目标未建、目标已建/标记未存、目标结束/结果未存、结果已存/回传未建或已建/消费未标记，均用唯一消息/结果行和固定 Task 幂等键重放对账。每次启动/轮询都可重复扫描，无重复 Task、结果或 hop。

## 链与旧数据兼容

根 Task 自动以自身 ID 作为 chain ID，默认上限等于根 Task BudgetConfig，最多 8 次全链显式发送。root-only `chain_budget/chain_max_hops` 可覆盖默认值；已存在 chain 不接受覆盖。真实 parent 存在时子 Task 必须继承该链，拒绝改 chain 或任何新限制。旧 API 中不可解析为本库 Task 的 parent_task_id 仍可作为外部来源文本；真实 handoff 始终验证实际 parent，模型没有该低层入口。

`chain_members.task_id` 唯一，用于旧 Task 首次加入时汇总已有 cumulative_metrics；重复初始化不会再加。未设置 chain 的旧 Task 自动补齐，旧提交 payload 按当前 DTO 补齐可选默认值后比较，避免新增 None 字段误报幂等冲突。

Task/Run 逐字段快照差值与 chain 增量在同一 SQLite transaction 更新；finish/reconcile 复用相同逻辑，重复 checkpoint 或恢复不重扣。Run 使用 Task 与 chain 剩余模型、工具、活动时间额度的较小值，包含只读子 Agent 和回传任务；新 Bot/Task/Run/重启均不重置额度。审批等待不计执行时间。全链 hops_used 统计所有接受的显式发送，包括分支及回传任务再次发送；自动回传本身不另扣 hop。

预算耗尽仍保存结果和回传任务，worker 将无法运行的任务标记 failed。CLI 仍能读取真实结果和 delivery_task 状态，不执行预算外模型。硬崩溃未持久的最后短时间片沿用 B3 保守恢复边界；未知副作用不自动重放。send_handoff 已提交但 JSONL 工具结果丢失时，源任务仍待核实，已接受消息可继续对账；取消源任务不撤销已接受消息，可另行取消目标任务。

## 实际验证与证据

沿用 BASELINE 已验证环境。pytest 使用已授权的沙箱外执行，规避该环境 asyncio executor 退出挂起；缓存与构建产物在 /tmp。初轮旧 Task/runtime/CLI 回归 70 passed in 15.37s；新增 D 模块/闭环/进程测试合跑 16 passed in 10.13s。辅助代理补强真实巡检和报告后，其 6 项再次通过（4.38s）。无网络模型、外部消息或新依赖。

闭环实际读取本地 evidence、已存记忆和已启用 Skill，模型先发报告再请求 write_file；旧 Run 等审批结束，重启后重复批准仍只执行一次保存参数，再经 send_handoff 让 reviewer 实际 read_file 审阅，发送 Bot 的新回传 Task 通过 read_handoff 读取结论。所有链指标与三个关联 Task、原任务两个 Run 的指标之和一致；seed Task 位于独立链。

互聊模型对每个新 Task 都主动向另一个 Bot 发消息，覆盖回传 Task 再发和每轮重启；分别以 3 hops、5 model requests、3 tool calls 停止。收件箱模块另覆盖并发幂等发送、五个持久切点、失败结果、子任务改链/抬限拒绝、SQL checkpoint + JSONL recovery 最终增量、旧数据库连续三次打开和预算耗尽的可观察 failed 回传任务。CLI 测试每条命令是独立 Python 进程，覆盖提交根限制、发送/收件箱/手工对账/目标执行/结果查询/回传执行/链查询。

最终检查日志位于本目录 `D-module-tests.log`、`D-pytest.log`、`D-ruff.log`、`D-format.log`、`D-mypy.log`、`D-build.log` 和 `D-diff-check.log`；实际命令与退出码汇总于 `D-checks.json`。最终文档写入后再执行全文件 lint/format/mypy/构建检查。

| 最终必需检查 | 实际结果 |
| --- | --- |
| D 模块、完整闭环和 CLI | exit 0；16 passed in 10.08s |
| 全量 pytest（300 秒界限） | exit 0；815 passed in 149.24s |
| Ruff check / format --check | exit 0；全仓库通过，117 files already formatted |
| mypy src | exit 0；36 source files 无问题 |
| 离线 wheel + sdist 构建 | exit 0；产物均生成 |
| git diff --check | exit 0 |

构建继续使用 BASELINE 的六个已有只读缓存后端目录：

```sh
PYTHONPATH=/home/curealb/.cache/uv/archive-v0/bgLrS0yUwJ8cuofD_HC9j:/home/curealb/.cache/uv/archive-v0/rV7UoyfERUwgy4LcpvHxy:/home/curealb/.cache/uv/archive-v0/Je5sA6EHFWgYlJlNS49Vc:/home/curealb/.cache/uv/archive-v0/Xr0rqq8-K2XtfoB7jWRNG:/home/curealb/.cache/uv/archive-v0/poTN_HFt5dAAQnuvxZwQv:/home/curealb/.cache/uv/archive-v0/E_BRBr-VAGOkDwa8JEmdX PYTHONDONTWRITEBYTECODE=1 UV_CACHE_DIR=/tmp/hiro-agent-uv-cache uv build --offline --no-build-isolation --out-dir /tmp/hiro-agent-d-dist
```

普通隔离构建受既有网络环境限制；离线产物路径为 `/tmp/hiro-agent-d-dist/hiro_agent-0.1.0.tar.gz` 与 `/tmp/hiro-agent-d-dist/hiro_agent-0.1.0-py3-none-any.whl`。本节点完成实际检查后冻结，等待独立 R4；无已知待解决设计阻塞。
