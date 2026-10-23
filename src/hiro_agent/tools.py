"""Validated tools with durable execution records and bounded local I/O."""

from __future__ import annotations

import asyncio
import json
import os
import signal
import stat
import tempfile
from collections.abc import Callable, Coroutine, Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any, Literal, cast

from pydantic import BaseModel, Field, ValidationError

from hiro_agent.budget import Budget
from hiro_agent.config import ToolsConfig
from hiro_agent.contracts import (
    Cancellation,
    DataModel,
    Observer,
    PermissionService,
    SessionService,
    ToolCall,
    ToolExecutionContext,
    ToolResult,
)
from hiro_agent.errors import BudgetExceeded, PersistenceError, RunCancelled, ToolError
from hiro_agent.execution import cancellable
from hiro_agent.session import delegation_terminal_event

ToolHandler = Callable[[Any], Coroutine[Any, Any, Any]]
ContextualToolHandler = Callable[[Any, ToolExecutionContext], Coroutine[Any, Any, Any]]
PathArgument = Annotated[str, Field(min_length=1, max_length=4096)]
CommandArgument = Annotated[str, Field(max_length=16384)]
_IGNORED = frozenset(
    {
        ".git",
        ".hg",
        ".svn",
        ".hiro-agent",
        "__pycache__",
        ".pytest_cache",
        ".mypy_cache",
        ".ruff_cache",
        ".tox",
        ".venv",
        "venv",
        "node_modules",
        ".cache",
        "dist",
        "build",
    }
)
_ENVIRONMENT = frozenset(
    {"PATH", "HOME", "USER", "LOGNAME", "LANG", "LC_ALL", "LC_CTYPE", "TZ", "TERM", "TMPDIR"}
)


@dataclass(frozen=True)
class ToolDefinition:
    name: str
    description: str
    arguments_model: type[BaseModel]
    handler: ToolHandler | ContextualToolHandler
    mutating: bool = False
    contextual: bool = False


@dataclass(frozen=True)
class ToolOutput:
    value: Any
    ok: bool = True
    error: str | None = None
    message: str = ""
    outcome: Literal["completed", "not_started", "unknown"] = "completed"


@dataclass
class _WalkState:
    truncated: bool = False


class ToolRegistry:
    def __init__(self) -> None:
        self.definitions: dict[str, ToolDefinition] = {}

    def register(
        self,
        name: str,
        description: str,
        arguments_model: type[BaseModel],
        handler: ToolHandler,
        *,
        mutating: bool = False,
    ) -> None:
        if name in self.definitions:
            raise ValueError(f"Tool already registered: {name}")
        self.definitions[name] = ToolDefinition(
            name, description, arguments_model, handler, mutating
        )

    def register_contextual(
        self,
        name: str,
        description: str,
        arguments_model: type[BaseModel],
        handler: ContextualToolHandler,
        *,
        mutating: bool = False,
    ) -> None:
        """Register a handler receiving execution authority outside model arguments."""
        if name in self.definitions:
            raise ValueError(f"Tool already registered: {name}")
        self.definitions[name] = ToolDefinition(
            name, description, arguments_model, handler, mutating, contextual=True
        )

    def restricted(self, names: Iterable[str]) -> ToolRegistry:
        """Copy only explicitly allowed tools, for both schemas and execution."""
        allowed = frozenset(names)
        missing = allowed - self.definitions.keys()
        if missing:
            raise ValueError(f"Unknown tools in restriction: {', '.join(sorted(missing))}")
        registry = ToolRegistry()
        registry.definitions.update(
            (name, definition) for name, definition in self.definitions.items() if name in allowed
        )
        return registry

    def schemas(self) -> list[dict[str, Any]]:
        return [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.arguments_model.model_json_schema(),
                },
            }
            for tool in self.definitions.values()
        ]


class ToolExecutor:
    def __init__(
        self,
        registry: ToolRegistry,
        session: SessionService,
        permission: PermissionService,
        observer: Observer | None = None,
        *,
        max_concurrency: int = 2,
    ) -> None:
        self.registry = registry
        self.session = session
        self.permission = permission
        self.observer = observer
        if not 1 <= max_concurrency <= 8:
            raise ValueError("max_concurrency must be between 1 and 8")
        self.max_concurrency = max_concurrency

    def schemas(self) -> list[dict[str, Any]]:
        return self.registry.schemas()

    async def execute_many(
        self, calls: list[ToolCall], run_id: str, budget: Budget, cancellation: Cancellation
    ) -> list[ToolResult]:
        """Only adjacent delegations overlap; every other tool is a sequence boundary."""
        results: list[ToolResult] = []
        index = 0
        while index < len(calls):
            cancellation.check()
            budget.check()
            call = calls[index]
            if call.name != "delegate_task" or call.name not in self.registry.definitions:
                results.append(await self.execute(call, run_id, budget, cancellation))
                index += 1
                continue
            end = index + 1
            while end < len(calls) and calls[end].name == "delegate_task":
                end += 1
            results.extend(
                await self._delegation_group(calls[index:end], run_id, budget, cancellation)
            )
            index = end
        return results

    async def _delegation_group(
        self, calls: list[ToolCall], run_id: str, budget: Budget, cancellation: Cancellation
    ) -> list[ToolResult]:
        # Queued calls have no execution records or child session yet. The durable
        # assistant declaration is enough for recovery and read-only task queries.
        if self.observer is not None:
            for call in calls:
                try:
                    args = json.loads(call.arguments)
                    self.observer(
                        "delegation.queued",
                        {
                            "call_id": call.id,
                            "role": args.get("role", "unknown")
                            if isinstance(args, dict)
                            else "unknown",
                            "parent_session_id": self.session.session_id,
                            "parent_run_id": run_id,
                            "child_session_id": None,
                        },
                    )
                except Exception:
                    pass
        pending: dict[asyncio.Task[ToolResult], int] = {}
        results: dict[int, ToolResult] = {}
        next_index = 0
        failure: BaseException | None = None
        try:
            while next_index < len(calls) or pending:
                cancellation.check()
                budget.check()
                while next_index < len(calls) and len(pending) < self.max_concurrency:
                    budget.check_model_capacity()
                    pending[
                        asyncio.create_task(
                            self.execute(calls[next_index], run_id, budget, cancellation)
                        )
                    ] = next_index
                    next_index += 1
                done, _ = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
                failures: list[BaseException] = []
                for task in sorted(done, key=lambda task: pending[task]):
                    position = pending.pop(task)
                    try:
                        results[position] = task.result()
                    except BaseException as exc:
                        failures.append(exc)
                if failures:
                    failure = next(
                        (e for e in failures if isinstance(e, PersistenceError)), failures[0]
                    )
                    raise failure
        except BaseException as exc:
            failure = exc
            cancellation.cancel(exc)
        finally:
            # Cooperatively interrupt operations, never cancel an executor while
            # it is acknowledging a journal write. Await every child and lock.
            outcomes = await asyncio.gather(*pending, return_exceptions=True)
            persistence = next((e for e in outcomes if isinstance(e, PersistenceError)), None)
            if persistence is not None:
                failure = persistence
                cancellation.cancel(persistence)
        if failure is not None:
            # The token retains a timeout/root-budget reason if cleanup later
            # signals cancellation. Persistence failures have highest priority.
            cancellation.check()
            raise failure
        return [results[index] for index in range(len(calls))]

    async def _record(self, kind: str, data: dict[str, Any], run_id: str) -> None:
        await self.session.append(kind, data, run_id)
        if self.observer is not None:
            try:
                self.observer(
                    kind, {**data, "session_id": self.session.session_id, "run_id": run_id}
                )
                if kind == "tool.result":
                    progress = delegation_terminal_event(
                        self.session, ToolResult.model_validate(data)
                    )
                    if progress is not None:
                        self.observer("delegation.finished", progress)
            except PersistenceError:
                raise
            except Exception:
                # Rendering failures must not change durable execution semantics.
                pass

    async def execute(
        self, call: ToolCall, run_id: str, budget: Budget, cancellation: Cancellation
    ) -> ToolResult:
        started = False
        called = False
        accepted = False
        call_data = {"call": call.model_dump(mode="json")}
        try:
            budget.claim_tool()
            accepted = True
            await self._record("tool.called", {**call_data, "accepted": True}, run_id)
            called = True
            cancellation.check()
            definition = self.registry.definitions.get(call.name)
            if definition is None:
                result = ToolResult(
                    call_id=call.id,
                    ok=False,
                    error="unknown_tool",
                    message=f"未知工具: {call.name}",
                    outcome="not_started",
                )
            else:
                try:
                    arguments = definition.arguments_model.model_validate_json(call.arguments)
                except ValidationError:
                    # ValidationError can include secrets from the supplied arguments.
                    result = ToolResult(
                        call_id=call.id,
                        ok=False,
                        error="invalid_arguments",
                        message="工具参数不符合 schema，请检查字段名称、类型和范围。",
                        outcome="not_started",
                    )
                else:
                    approved = await cancellable(
                        self.permission.approve(
                            call.name, arguments.model_dump(mode="json"), definition.mutating
                        ),
                        cancellation,
                        budget.remaining,
                    )
                    if not approved:
                        budget.record(permission_denials=1)
                        result = ToolResult(
                            call_id=call.id,
                            ok=False,
                            error="permission_denied",
                            message="本次工具调用未获授权。",
                            outcome="not_started",
                        )
                    else:
                        budget.check()
                        cancellation.check()
                        await self._record("tool.started", call_data, run_id)
                        started = True
                        if definition.contextual:
                            context = ToolExecutionContext(
                                session=self.session,
                                run_id=run_id,
                                call_id=call.id,
                                budget=budget,
                                cancellation=cancellation,
                            )
                            operation = cast(ContextualToolHandler, definition.handler)(
                                arguments, context
                            )
                        else:
                            operation = cast(ToolHandler, definition.handler)(arguments)
                        value = await cancellable(operation, cancellation, budget.remaining)
                        output = value if isinstance(value, ToolOutput) else ToolOutput(value)
                        result = ToolResult(
                            call_id=call.id,
                            ok=output.ok,
                            value=output.value,
                            error=output.error,
                            message=output.message,
                            outcome=output.outcome,
                        )
        except PersistenceError:
            raise
        except (RunCancelled, BudgetExceeded, asyncio.CancelledError) as exc:
            if not called:
                await self._record("tool.called", {**call_data, "accepted": accepted}, run_id)
            interrupted = ToolResult(
                call_id=call.id,
                ok=False,
                error="budget_exceeded" if isinstance(exc, BudgetExceeded) else "cancelled",
                message="工具执行被中断；已开始的操作可能产生了部分副作用。"
                if started
                else "工具执行前已中断。",
                outcome="unknown" if started else "not_started",
            )
            budget.record(tool_errors=1)
            await self._record("tool.result", interrupted.model_dump(mode="json"), run_id)
            raise
        except (ToolError, OSError, UnicodeError, ValueError) as exc:
            result = ToolResult(
                call_id=call.id,
                ok=False,
                error="tool_error",
                message=str(exc),
                outcome="completed" if started else "not_started",
            )
        except Exception:
            result = ToolResult(
                call_id=call.id,
                ok=False,
                error="tool_error",
                message="工具执行失败。",
                outcome="unknown" if started else "not_started",
            )
        if not result.ok:
            budget.record(tool_errors=1)
        await self._record("tool.result", result.model_dump(mode="json"), run_id)
        return result


class ListFilesArgs(DataModel):
    path: PathArgument = "."
    recursive: bool = False
    max_results: int = Field(default=200, ge=1, le=2000)
    max_depth: int = Field(default=8, ge=1, le=32)


class ReadFileArgs(DataModel):
    path: PathArgument
    start_line: int = Field(default=1, ge=1, le=1_000_000)
    num_lines: int = Field(default=200, ge=1, le=2000)


class SearchTextArgs(DataModel):
    query: str = Field(min_length=1, max_length=1024)
    path: PathArgument = "."
    case_sensitive: bool = True
    max_matches: int = Field(default=100, ge=1, le=1000)


class WriteFileArgs(DataModel):
    path: PathArgument
    content: str = Field(max_length=8_388_608)
    create_parents: bool = False


class EditFileArgs(DataModel):
    path: PathArgument
    old_text: str = Field(min_length=1, max_length=8_388_608)
    new_text: str = Field(max_length=8_388_608)
    expected_matches: int = Field(default=1, ge=1, le=1000)


class RunCommandArgs(DataModel):
    executable: PathArgument
    args: list[CommandArgument] = Field(default_factory=list, max_length=256)
    cwd: PathArgument = "."
    timeout_seconds: float | None = Field(default=None, gt=0, le=3600)


class _Workspace:
    def __init__(self, root: Path, config: ToolsConfig) -> None:
        self.root = root.resolve(strict=True)
        if not self.root.is_dir():
            raise ValueError("工作区必须是目录。")
        self.config = config

    def path(self, raw: str, *, writing: bool = False, traversal: bool = False) -> Path:
        candidate = self.root / raw
        try:
            resolved = candidate.resolve()
        except (OSError, RuntimeError, ValueError) as exc:
            raise ToolError("路径无法解析。") from exc
        if not resolved.is_relative_to(self.root):
            raise ToolError("路径必须位于工作区内，符号链接也不能指向工作区外。")
        parts = resolved.relative_to(self.root).parts
        # Check lexical components too, so a protected directory cannot be an alias.
        lexical_parts = (
            candidate.relative_to(self.root).parts if candidate.is_relative_to(self.root) else ()
        )
        all_parts = set(parts) | set(lexical_parts)
        if ".hiro-agent" in all_parts:
            raise ToolError("工具不能访问 .hiro-agent 运行记录。")
        if writing and ".git" in all_parts:
            raise ToolError("工具不能写入 .git。")
        if traversal and _IGNORED.intersection(all_parts):
            raise ToolError("列表和搜索不访问缓存、版本控制或运行记录目录。")
        return resolved

    def read(self, path: Path) -> str:
        if not path.is_file():
            raise ToolError("路径不是可读取的普通文件。")
        if path.stat().st_size > self.config.max_file_bytes:
            raise ToolError("文件超过 max_file_bytes 限制。")
        with path.open("rb") as stream:
            content = stream.read(self.config.max_file_bytes + 1)
        if len(content) > self.config.max_file_bytes:
            raise ToolError("文件超过 max_file_bytes 限制。")
        if b"\0" in content:
            raise ToolError("不支持读取二进制文件。")
        return content.decode("utf-8")

    def write(self, path: Path, text: str, *, create_parents: bool = False) -> int:
        content = text.encode("utf-8")
        if len(content) > self.config.max_file_bytes:
            raise ToolError("写入内容超过 max_file_bytes 限制。")
        if create_parents:
            path.parent.mkdir(parents=True, exist_ok=True)
        self.path(str(path), writing=True)
        if path.exists() and not path.is_file():
            raise ToolError("目标不是普通文件。")
        temporary: str | None = None
        try:
            with tempfile.NamedTemporaryFile(
                dir=path.parent, prefix=".hiro-write-", delete=False
            ) as f:
                temporary = f.name
                f.write(content)
                f.flush()
                os.fsync(f.fileno())
            if path.exists():
                os.chmod(temporary, stat.S_IMODE(path.stat().st_mode))
            os.replace(temporary, path)
        finally:
            if temporary is not None:
                Path(temporary).unlink(missing_ok=True)
        return len(content)

    def walk(
        self, base: Path, state: _WalkState, *, recursive: bool, max_depth: int = 32
    ) -> Iterator[Path]:
        # Count every directory entry, including ignored ones, to cap traversal work.
        remaining = self.config.max_search_files
        stack = [(base, 0)]
        while stack and remaining > 0:
            directory, depth = stack.pop()
            with os.scandir(directory) as entries:
                for entry in entries:
                    remaining -= 1
                    if entry.name not in _IGNORED and not entry.is_symlink():
                        path = Path(entry.path)
                        if entry.is_dir(follow_symlinks=False):
                            yield path
                            if recursive and depth + 1 < max_depth:
                                stack.append((path, depth + 1))
                            elif recursive:
                                state.truncated = True
                        elif entry.is_file(follow_symlinks=False):
                            yield path
                    if remaining <= 0:
                        state.truncated = True
                        return


async def _stop_process(process: asyncio.subprocess.Process) -> None:
    # A group may outlive its leader, so returncode must not guard killpg.
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    await asyncio.sleep(0.15)
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    await process.wait()


async def _cleanup_process(process: asyncio.subprocess.Process) -> None:
    cleanup = asyncio.create_task(_stop_process(process))
    while not cleanup.done():
        try:
            await asyncio.shield(cleanup)
        except asyncio.CancelledError:
            # Cleanup must finish even if a second cancellation arrives.
            continue
    cleanup.result()


def build_file_tools(
    workspace: Path,
    config: ToolsConfig,
    *,
    excluded_env: Iterable[str] = (),
) -> ToolRegistry:
    """Build all six local tools. Commands require approval, not an OS sandbox."""
    fs = _Workspace(Path(workspace), config)
    excluded = set(excluded_env) | {"HIRO_API_KEY", "OPENAI_API_KEY"}
    registry = ToolRegistry()

    async def list_files(args: ListFilesArgs) -> dict[str, Any]:
        base = fs.path(args.path, traversal=True)
        if not base.is_dir():
            raise ToolError("列表路径必须是目录。")
        paths: list[str] = []
        used = 0
        truncated = False
        state = _WalkState()
        for path in fs.walk(base, state, recursive=args.recursive, max_depth=args.max_depth):
            await asyncio.sleep(0)
            display = path.relative_to(fs.root).as_posix() + ("/" if path.is_dir() else "")
            if len(paths) >= args.max_results or used + len(display) > config.max_output_chars:
                truncated = True
                break
            paths.append(display)
            used += len(display)
        return {"paths": sorted(paths), "truncated": truncated or state.truncated}

    async def read_file(args: ReadFileArgs) -> dict[str, Any]:
        path = fs.path(args.path)
        text = fs.read(path)
        lines = text.splitlines(keepends=True)
        selected = "".join(lines[args.start_line - 1 : args.start_line - 1 + args.num_lines])
        return {
            "path": path.relative_to(fs.root).as_posix(),
            "content": selected[: config.max_output_chars],
            "start_line": args.start_line,
            "total_lines": len(lines),
            "truncated": len(selected) > config.max_output_chars
            or args.start_line - 1 + args.num_lines < len(lines),
        }

    async def search_text(args: SearchTextArgs) -> dict[str, Any]:
        base = fs.path(args.path, traversal=True)
        if not base.exists():
            raise ToolError("搜索路径不存在。")
        state = _WalkState()
        candidates = fs.walk(base, state, recursive=True) if base.is_dir() else iter([base])
        matches: list[dict[str, Any]] = []
        query = args.query if args.case_sensitive else args.query.casefold()
        used = 0
        searched = 0
        truncated = False
        for path in candidates:
            await asyncio.sleep(0)
            if not path.is_file():
                continue
            searched += 1
            try:
                text = fs.read(path)
            except (OSError, ToolError, UnicodeError):
                continue
            for number, line in enumerate(text.splitlines(), start=1):
                if query not in (line if args.case_sensitive else line.casefold()):
                    continue
                name = path.relative_to(fs.root).as_posix()
                remaining = config.max_output_chars - used - len(name)
                if len(matches) >= args.max_matches or remaining <= 0:
                    truncated = True
                    break
                excerpt = line[:remaining]
                matches.append({"path": name, "line": number, "text": excerpt})
                used += len(excerpt) + len(name)
                if len(excerpt) < len(line):
                    truncated = True
                    break
            if truncated:
                break
        return {
            "matches": matches,
            "files_searched": searched,
            "truncated": truncated or state.truncated,
        }

    async def write_file(args: WriteFileArgs) -> dict[str, Any]:
        path = fs.path(args.path, writing=True)
        count = fs.write(path, args.content, create_parents=args.create_parents)
        return {"path": path.relative_to(fs.root).as_posix(), "bytes_written": count}

    async def edit_file(args: EditFileArgs) -> dict[str, Any]:
        path = fs.path(args.path, writing=True)
        text = fs.read(path)
        count = text.count(args.old_text)
        if count != args.expected_matches:
            raise ToolError(
                f"精确文本匹配 {count} 次，expected_matches 为 {args.expected_matches}；"
                "文件未修改。"
            )
        written = fs.write(path, text.replace(args.old_text, args.new_text))
        return {
            "path": path.relative_to(fs.root).as_posix(),
            "replacements": count,
            "bytes_written": written,
        }

    async def run_command(args: RunCommandArgs) -> ToolOutput:
        cwd = fs.path(args.cwd)
        if not cwd.is_dir():
            raise ToolError("命令工作目录必须是目录。")
        environment = {
            key: value
            for key, value in os.environ.items()
            if key in _ENVIRONMENT and key not in excluded
        }
        process = await asyncio.create_subprocess_exec(
            args.executable,
            *args.args,
            cwd=cwd,
            env=environment,
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            start_new_session=True,
        )
        output_limit = config.max_output_chars
        retained = 0
        truncated = False

        async def drain(stream: asyncio.StreamReader | None) -> str:
            nonlocal retained, truncated
            assert stream is not None
            chunks: list[bytes] = []
            while chunk := await stream.read(8192):
                available = max(0, output_limit - retained)
                if available:
                    chunks.append(chunk[:available])
                retained += min(len(chunk), available)
                truncated = truncated or len(chunk) > available
            return b"".join(chunks).decode("utf-8", errors="replace")

        timeout = min(
            args.timeout_seconds or config.command_timeout_seconds, config.command_timeout_seconds
        )
        completion = asyncio.gather(process.wait(), drain(process.stdout), drain(process.stderr))
        try:
            code, stdout, stderr = await asyncio.wait_for(
                asyncio.shield(completion),
                timeout=timeout,
            )
        except TimeoutError as exc:
            raise ToolError("命令执行超时，进程组已终止。") from exc
        finally:
            # Even a successful leader can leave detached children in its process group.
            # Keep draining during shutdown: paused pipes can otherwise prevent reaping.
            await _cleanup_process(process)
            await completion
        value = {"exit_code": code, "stdout": stdout, "stderr": stderr, "truncated": truncated}
        return ToolOutput(
            value,
            ok=code == 0,
            error=None if code == 0 else "command_failed",
            message="" if code == 0 else f"命令退出码为 {code}。",
        )

    registry.register(
        "list_files",
        "列出工作区文件，跳过缓存、版本控制和运行数据目录。",
        ListFilesArgs,
        list_files,
    )
    registry.register(
        "read_file", "读取有大小限制的 UTF-8 文件，可指定起始行和行数。", ReadFileArgs, read_file
    )
    registry.register(
        "search_text",
        "在工作区 UTF-8 文件中搜索字面文本，返回有限匹配。",
        SearchTextArgs,
        search_text,
    )
    registry.register(
        "write_file",
        "创建或覆盖 UTF-8 文件；需要本次调用的明确批准。",
        WriteFileArgs,
        write_file,
        mutating=True,
    )
    registry.register(
        "edit_file",
        "精确替换文本；匹配次数必须等于 expected_matches，默认 1。",
        EditFileArgs,
        edit_file,
        mutating=True,
    )
    registry.register(
        "run_command",
        "按 executable 和 args 执行命令，不解析 shell 字符串。命令可访问宿主系统，需要明确批准。",
        RunCommandArgs,
        run_command,
        mutating=True,
    )
    return registry
