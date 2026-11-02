# A 实现交接：基础契约、Bot 身份与执行权限

基准 Git revision：`81f42f90b932e61151f399d423fa9a45e55a97e7`。本节点是实现自检，不代表独立 R1 验收通过；生产代码完成后交由主代理冻结并启动 R1，未启动 B/C/D。

## 结果与改动

- 新增 `src/hiro_agent/kernel_db.py`：工作区 SQLite 路径、WAL/FULL/外键、短立即事务、按模块命名空间的顺序迁移及已应用 SQL 摘要校验；异常回滚与连接关闭。
- 新增 `src/hiro_agent/bots.py`：完整 Config 持久化、稳定 Bot ID、可跨进程重新加载的权限 provider、递增权限版本和策略摘要、可选版本 CAS。白名单重排/重复和非权限配置修改不增加版本；撤权再恢复旧列表仍增加版本。
- 新增 `src/hiro_agent/kernel_contracts.py`：Task/Run/Interaction/操作意图 DTO，任务存储、TaskRunner、交互协议，参数规范化和授权绑定。仅提供共享契约，不声称 B1/B2 功能已实现。
- `config.py` 新增可选 BotConfig；旧 Config.bot=None 默认兼容。Bot ID 必须显式配置，防止重复加载配置文件产生不同身份。
- `tools.py` 在执行端拒绝禁用/未知工具，过滤 schemas，并在审批回调与 tool.started 落盘之后复核实时权限。撤权后 handler 不执行；工具日志记录实际 authority 版本和摘要。
- `contracts.py` 的 ToolExecutionContext 添加可选 authority；`subagents.py` 向子执行器继承父 provider，结合既有只读注册表限制权限。无 Bot 的旧用法保持兼容。
- `bootstrap.py` 写入/校验 session.bot 绑定，拒绝跨 Bot 续跑。省略 Bot 配置恢复绑定 Session 时加载数据库完整配置；新 Session 初始 system prompt 追加 Bot instructions。既有 Session 原始 prompt 不修改。
- 新增 `tests/test_kernel_foundations.py`：20 项 SQLite 事务/迁移、身份/配置恢复、权限版本、真实执行端拦截、跨 await 撤权、子 Agent 权限继承、会话绑定与恢复、操作绑定测试。
- 新增 `docs/kernel/CONTRACTS.md`：文件所有权、准确函数签名/DTO、状态映射、累计预算、统一提交/幂等、持久交互唤醒清单、授权消费、四个崩溃窗口、B3 精确 intent 新 Run 执行顺序、后续 C/D 接入边界。

## 实际验证

旧基线详见 `BASELINE.md`：Ruff/format/mypy 通过；原 481 测试通过；标准隔离构建无法访问 PyPI，缓存离线后端构建通过。沙箱内 asyncio.to_thread shutdown 会挂起，已由基线最小复现确认，因此测试使用已授权的沙箱外命令；没有把挂起当通过。

| 检查 | 实际命令 | 结果 |
| --- | --- | --- |
| 新模块测试 | `timeout 120s .venv/bin/python -m pytest -q tests/test_kernel_foundations.py`（沙箱外） | exit 0，20 passed in 0.85s；由仅拥有该测试文件的辅助代理执行 |
| Ruff | `RUFF_CACHE_DIR=/tmp/hiro-agent-ruff-cache .venv/bin/ruff check .` | exit 0，All checks passed |
| 格式 | `RUFF_CACHE_DIR=/tmp/hiro-agent-ruff-cache .venv/bin/ruff format --check .` | exit 0，68 files already formatted |
| 类型 | `MYPY_CACHE_DIR=/tmp/hiro-agent-mypy-cache .venv/bin/mypy src` | exit 0，25 source files 无问题 |
| 全量测试 | `PYTHONDONTWRITEBYTECODE=1 timeout 120s .venv/bin/pytest -q -p no:cacheprovider`（沙箱外） | 最终源码状态 exit 0，501 passed in 30.30s；此前同样 501 passed in 32.02s |
| 构建 | 下方缓存后端命令 | exit 0，wheel 和 sdist 均生成 |

最终构建命令（复用基线已验证的本机缓存，不改依赖声明）：

```sh
PYTHONPATH=/home/curealb/.cache/uv/archive-v0/bgLrS0yUwJ8cuofD_HC9j:/home/curealb/.cache/uv/archive-v0/rV7UoyfERUwgy4LcpvHxy:/home/curealb/.cache/uv/archive-v0/Je5sA6EHFWgYlJlNS49Vc:/home/curealb/.cache/uv/archive-v0/Xr0rqq8-K2XtfoB7jWRNG:/home/curealb/.cache/uv/archive-v0/poTN_HFt5dAAQnuvxZwQv:/home/curealb/.cache/uv/archive-v0/E_BRBr-VAGOkDwa8JEmdX \
PYTHONDONTWRITEBYTECODE=1 UV_CACHE_DIR=/tmp/hiro-agent-uv-cache \
uv build --offline --no-build-isolation --out-dir /tmp/hiro-agent-a-dist
```

输出：`/tmp/hiro-agent-a-dist/hiro_agent-0.1.0.tar.gz`、`/tmp/hiro-agent-a-dist/hiro_agent-0.1.0-py3-none-any.whl`。普通需联网的隔离构建限制仍是环境问题；本节点实际打包验证通过。

## 下游最短启动说明

先读 `../CONTRACTS.md` 和 `../REQUIREMENTS.md`，以及实际 `kernel_contracts.py`。R1 PASS 前不启动下游。

- B1 只写 `durable_tasks.py`、`worker.py` 及自己的测试；实现 TaskStoreService 与文档指定 TaskWorker，使用真实 SQLite 和 fake TaskRunner。每次领取必须先持久 Run ID；累计 metrics 按 Run 快照差额计入一次；unknown 副作用不可领取。不要更改原 `tasks.py`。
- B2 只写 `interactions.py` 及自己的测试；实现 InteractionService。答复保存到 B2 表后由 ready() 持续提供持久唤醒清单，供 B3 幂等调用 TaskStore.wake，无须写 B1 表。授权消费必须在事务内核对当前 Bot 权限、精确 intent.binding，并且第二次返回 False。
- B3 唯一集成者：实现 TaskRunner 适配器/CLI/worker 装配，处理旧 call 结算、新 Run 内精确执行保存 intent、持续预算、真实子进程及四个崩溃窗口。不要依赖模型重新生成审批的操作。

## 未解决事项和范围限制

独立 R1 首次发现 R1-F1，修复情况见下方，须由同一独立代理复验后方可放行；后续 Task/交互/worker/记忆/Skills/routines/交接均只是契约或未来所有权，尚未实现。权限更新在下一执行检查生效，不能撤销已经开始的外部副作用。没有新增依赖，没有修改 README、REQUIREMENTS、STATUS 或任务包。

## A 修复尝试 2：R1-F1

独立报告 `docs/kernel/reviews/R1.md` 指出：`sqlite3.connect()` 在异常转换边界之外，数据库无法打开时泄漏 OperationalError。执行器因此将故障结算为普通 tool_error，而没有传播 PersistenceError 停止运行。根代理在首次 FAIL 报告完成后仅为本项修复解除冻结。

生产修改仅限 `kernel_db.py`：将连接获取、PRAGMA 设置、使用及 close 全部置于外层 `sqlite3.Error → PersistenceError` 边界，连接成功后才安排关闭；初始化的路径解析/目录创建 OSError 在原地转换为 PersistenceError。不会把连接上下文中调用方自己的任意 OSError 错当数据库故障。现有事务 BEGIN/commit/rollback 也继续在同一转换边界内，无需更改工具执行器。

自己的 `test_kernel_foundations.py` 增加 5 项真实故障测试：父路径被文件占用；数据库路径变目录或内容损坏，分别通过 connect 与 transaction 进入。断言保留原始 OSError/sqlite3.Error 为异常 cause。独立 `test_acceptance_r1.py` 和 R1 报告、清单、日志均未修改；读取并运行了其两个原始失败用例。另有只读辅助检查确认获取/关闭/路径准备同类边界，无生产或测试改动。

| 修复验证 | 实际命令 | 结果 |
| --- | --- | --- |
| 模块及独立用例 | `PYTHONDONTWRITEBYTECODE=1 timeout 120s .venv/bin/pytest -q -p no:cacheprovider tests/test_kernel_foundations.py tests/test_acceptance_r1.py`（沙箱外） | exit 0，41 passed in 2.35s |
| Ruff | `RUFF_CACHE_DIR=/tmp/hiro-agent-ruff-cache .venv/bin/ruff check .` | exit 0，All checks passed |
| 格式 | `RUFF_CACHE_DIR=/tmp/hiro-agent-ruff-cache .venv/bin/ruff format --check .` | exit 0，76 files already formatted |
| mypy | `MYPY_CACHE_DIR=/tmp/hiro-agent-mypy-cache .venv/bin/mypy src` | exit 0，25 source files 无问题 |
| 完整回归 | `PYTHONDONTWRITEBYTECODE=1 timeout 120s .venv/bin/pytest -q -p no:cacheprovider`（沙箱外） | exit 0，522 passed in 32.46s |
| 构建 | 上方缓存后端命令，输出目录改为 `/tmp/hiro-agent-a-repair-dist` | exit 0，wheel 和 sdist 均生成 |

`git diff --check` exit 0。修复后生产代码再次冻结；实现自检通过不替代同一独立 R1 代理的复验结论。
