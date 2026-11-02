# 开发与验证

使用 Python 3.12 和 uv，在仓库根目录安装锁定依赖：

```sh
uv sync --locked --managed-python
```

提交前运行与 CI 相同的检查：

```sh
uv run --locked ruff check .
uv run --locked ruff format --check .
uv run --locked mypy src
uv run --locked pytest -q
uv build
```

`tests/test_*.py` 包含模块、CLI、权限、持久化和 TUI 回归测试。`tests/acceptance_r2/`、`tests/acceptance_r3/`、`tests/acceptance_r4/` 保留真实子进程、崩溃与重启测试；其中 `process_driver.py` 提供确定性离线模型及故障注入，由测试负责启动和回收进程。测试数据使用 pytest 临时目录。

部分受限沙箱会阻塞 asyncio 的线程唤醒或退出，使异步及子进程测试挂起；遇到该环境问题时，在允许正常进程通信的环境运行完整测试。测试默认使用离线模型，不需要真实 API 凭据。

阶段任务包、交接记录、一次性验收报告、日志、哈希清单和证据归档不纳入仓库。临时实验脚本与输出放在系统临时目录；正式回归用例保留在 `tests/`。构建产物、工具缓存和本地 `.hiro-agent/` 运行数据由 `.gitignore` 排除。
