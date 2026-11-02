# Baseline and CLI/configuration map

Observed on 2026-11-02, before kernel implementation changes. This handoff records actual commands and results; no production files were edited during reconnaissance. No applicable `AGENTS.md` was found in repository or ancestor directories.

## Environment and required checks

- Project requires Python >=3.12; `.python-version` and CI select 3.12.
- Existing `.venv/bin/python`: Python 3.12.12. `uv`: 0.9.15.
- Ruff, mypy, pytest and pytest-asyncio are already installed in `.venv`; hatchling is not installed there.
- `.github/workflows/checks.yml` installs Python 3.12, runs `uv sync --locked --managed-python`, then `uv run --locked ruff check .`, `uv run --locked ruff format --check .`, `uv run --locked mypy src`, `uv run --locked pytest -q`, and `uv build`.
- Tool caches/build output were redirected to `/tmp`; existing environments and dependency declarations were left unchanged.

| Actual command | Actual result |
| --- | --- |
| `UV_CACHE_DIR=/tmp/hiro-agent-uv-cache RUFF_CACHE_DIR=/tmp/hiro-agent-ruff-cache .venv/bin/ruff check .` | Exit 0: `All checks passed!` |
| `UV_CACHE_DIR=/tmp/hiro-agent-uv-cache RUFF_CACHE_DIR=/tmp/hiro-agent-ruff-cache .venv/bin/ruff format --check .` | Exit 0: `51 files already formatted` |
| `MYPY_CACHE_DIR=/tmp/hiro-agent-mypy-cache .venv/bin/mypy src` | Exit 0: `Success: no issues found in 22 source files` |
| `PYTHONDONTWRITEBYTECODE=1 timeout 90s .venv/bin/pytest -q -p no:cacheprovider` (approved execution outside sandbox) | Exit 0: `481 passed in 30.65s` |
| `UV_CACHE_DIR=/tmp/hiro-agent-uv-cache uv build --out-dir /tmp/hiro-agent-baseline-dist` | Exit 2: isolated backend resolution could not access PyPI; exact error below. |
| Cached backend fallback below | Exit 0: wheel and source distribution built successfully. |

Standard build failure:

```text
Building source distribution...
  × Failed to build `/home/curealb/agent/hiro-agent`
  ├─▶ Failed to resolve requirements from `build-system.requires`
  ├─▶ No solution found when resolving: `hatchling>=1.26`
  ├─▶ Failed to fetch: `https://pypi.org/simple/hatchling/`
  ├─▶ Request failed after 3 retries
  ├─▶ error sending request for url (https://pypi.org/simple/hatchling/)
  ├─▶ client error (Connect)
  ├─▶ tunnel error: failed to create underlying connection
  ├─▶ tcp open error
  ╰─▶ Operation not permitted (os error 1)
```

Successful offline fallback used existing read-only cached hatchling 1.33.0 and its dependencies:

```sh
PYTHONPATH=/home/curealb/.cache/uv/archive-v0/bgLrS0yUwJ8cuofD_HC9j:/home/curealb/.cache/uv/archive-v0/rV7UoyfERUwgy4LcpvHxy:/home/curealb/.cache/uv/archive-v0/Je5sA6EHFWgYlJlNS49Vc:/home/curealb/.cache/uv/archive-v0/Xr0rqq8-K2XtfoB7jWRNG:/home/curealb/.cache/uv/archive-v0/poTN_HFt5dAAQnuvxZwQv:/home/curealb/.cache/uv/archive-v0/E_BRBr-VAGOkDwa8JEmdX \
PYTHONDONTWRITEBYTECODE=1 UV_CACHE_DIR=/tmp/hiro-agent-uv-cache \
uv build --offline --no-build-isolation --out-dir /tmp/hiro-agent-baseline-dist
```

```text
Building source distribution...
Building wheel from source distribution...
Successfully built /tmp/hiro-agent-baseline-dist/hiro_agent-0.1.0.tar.gz
Successfully built /tmp/hiro-agent-baseline-dist/hiro_agent-0.1.0-py3-none-any.whl
```

These cache paths are machine-specific, not project dependencies. Ordinary isolated build remains unverified in the network-restricted environment, while packaging itself passed using the cached backend.

## Sandbox test constraint

The initial sandboxed `PYTHONDONTWRITEBYTECODE=1 .venv/bin/pytest -q -p no:cacheprovider` stalled and was interrupted (exit 130). A bounded diagnostic, `PYTHONDONTWRITEBYTECODE=1 timeout 75s .venv/bin/pytest -q -p no:cacheprovider -o faulthandler_timeout=30 tests/test_tasks.py`, passed its first test then timed out (exit 124). Faulthandler showed an idle executor worker and the main asyncio loop waiting in `selectors.select`.

A minimal sandboxed script awaiting `asyncio.to_thread(lambda: None)` printed both before and after the await but failed to exit `asyncio.run` before an external four-second timeout. Together with the full passing suite outside the sandbox, this demonstrates environment-related asyncio wakeup/shutdown interference; it is not a baseline test failure. All diagnostic processes were stopped or reached their bounded timeout.

## Existing integration points

- `src/hiro_agent/cli.py`: Typer entrypoint (`pyproject.toml`: `hiro-agent = hiro_agent.cli:app`). Root callback and `chat` launch the REPL; `resume` selects/opens existing sessions; `sessions` and `tasks` perform read-only queries. `inherited_option()` preserves global options on subcommands. `launch()` loads JSON configuration, applies only `HIRO_BASE_URL`/`HIRO_MODEL` routing overrides, `--mock`, and `--subagents`, then calls `chat_loop()`.
- `cli.py:374` (`chat_loop`): opens a harness, wires terminal/display observers, supports `/continue`, `/tasks`, `/resume`, `/help`, `/exit`, and cooperative SIGINT cancellation. Opening/resuming displays history and waits for input; it does not automatically run the model. `/tasks` projects only the most recent `run.started` run.
- `src/hiro_agent/config.py`: Pydantic `Config` nests model/budget/tools/subagents settings plus `permissions` (`ask-write` or `read-only`). `ModelConfig.extra_body` forbids overriding protocol/credential fields. Subagent models are optional complete model configurations, not partial inheritance. `load_config()` converts parsing/validation failures into a secret-safe `ConfigurationError`. `configs/baseline.json` documents defaults.
- `src/hiro_agent/bootstrap.py:38` (`create_harness`): composition root validating workspace/session IDs, opening the journal, rejecting direct child-session resume, recovering delegations and session outcomes, creating model clients/tools/context/optional subagents/agent through dependency-aware runtime plugins. `Harness.close()` delegates to runtime disposal. Main mock mode forces child models offline; unconfigured role models share the main client. All configured credential environment names are excluded from command execution.
- `src/hiro_agent/tasks.py`: existing `TaskSummary`, `summarize_tasks(events, run_id=None)`, and `list_tasks(workspace, session_id, run_id=None)` are read-only projections of delegated tool calls from parent session journals. They are not a task scheduler. Keep these public APIs and command meanings compatible when adding kernel tasks.
- Task projection preserves declaration order, processes `delegation.started`/`delegation.finished`/`tool.result`, tracks per-child metrics and links, tolerates missing legacy model metadata, and maps open declarations in ended runs to `unknown` or `not_started`. `list_tasks()` rejects child journals. `hiro-agent tasks --session ID [--run ID]` queries all matching delegated tasks without recovery/writes.

## Compatibility tests to retain

- `tests/test_cli.py`: root/chat entrypoints, help, option inheritance, write denial/EOF, SIGINT during approval, paused resume without duplicate writes, missing sessions without new journals.
- `tests/test_cli_resume.py`, `tests/test_cli_slash_commands.py`, `tests/test_session_history.py`: existing resume and interactive behavior.
- `tests/test_tasks.py`: declaration order, run filters, child metrics/report metadata, read-only journal queries, busy/corrupt/missing journal handling, legacy metadata, unknown/not-started outcomes, pause statuses, and task progress rendering.
- `tests/test_subagent_cli.py`: public mock/subagent flags, `/tasks` plus `tasks` command, child-resume rejection, new root budget on resume without repeated writes.
- `tests/test_role_models.py`: complete role configuration validation, client routing and closure, startup cleanup, shared main client fallback, forced offline mock, credential exclusions, legacy journal compatibility.
- Existing recovery/cancellation/streaming suites remain part of the full 481-test baseline; new kernel functionality must not silently alter their behavior.
