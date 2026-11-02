# TUI backend handoff

`src/hiro_agent/tui_backend.py` exposes the agreed frozen `BotSummary`, `ChatEntry`,
and `Conversation` dataclasses and `TuiBackend(workspace, *, kernel=None)`. Its
methods are `list_bots`, `conversation`, `submit`, `reply`, `cancel`,
`worker_running`, and `start_worker`; `workspace` and `kernel` are public.

Conversation tasks are the newest 30 in chronological order. Entries preserve
the durable Task text even before execution, assistant messages, durable
question/approval responses, and tool activity. Task output supplies a fallback
without duplicating a recorded assistant answer. System prompts, provider
metadata, prepared-operation assistant messages, and synthetic continuation
messages are omitted. `interactions` contains only current, pending interactions
whose Task is still waiting in the matching state; historical answers appear in
`entries`. Missing/corrupt journals are reported through `Conversation.error`
without hiding other Task text or output.

`session.read_session_snapshot(workspace, session_id)` reads through no-follow
directory/file descriptors, accepts only regular files, bounds each read by the
initial byte length, and parses only newline-terminated records. It validates
schemas and existing sequence/reference invariants, takes no journal lock, and
never repairs/replays/truncates. Existing `read_session_events` remains intact.
The backend caches up to 64 display snapshots by inode/device/size/mtime/ctime;
cache entries never drive execution or authorization. Newly appended logs are
reparsed in full; there is no incremental-validation or streaming-token API.

Replies re-read the interaction with Bot ownership, then its Task and current
waiting pointer; approvals also match the saved operation. They only persist
through `InteractionStore.reply`; the worker handles wake/consumption. Cancels
use the same Task/interaction behavior as the kernel CLI.

Worker probing opens the existing OS lock without creation or unlinking.
Startup returns immediately with a detached `Popen[bytes]` logging to
`.hiro-agent/worker.log`, or `None` when an executor holds the lock. Repeated
startup on the same backend while its child is still starting returns that same
child. Concurrent frontends may launch competing children, but the existing OS
lock permits only one executor. The UI should poll the lock and child exit code
with a bounded timeout and leave successful children alive when it closes.

Validation on 2026-11-02:

- Scoped Ruff and formatting pass for backend, session, and backend tests.
- Scoped mypy passes for backend and session.
- `pytest -q -p no:cacheprovider tests/test_tui_backend.py tests/test_session.py
  tests/test_session_listing.py`: **73 passed in 2.98s**, outside the sandbox as
  required by the documented asyncio shutdown constraint.
- The real-process test executes a read-only mock Task in a temporary workspace,
  confirms it survives backend recreation, and terminates/reaps its child.
  No worker was started in the project workspace.

## CLI validation appendix

`tests/test_tui_cli.py` adds 12 passing tests for terminal-only startup before
database creation, help, inherited and overridden workspace/config options,
multiple actual Bot config imports, initial Agent selection, worker-option
forwarding, demo policy preservation, and inherited mock mode. Command tests
stub only `BotTUI.run`, so backend persistence and initial refresh remain real.

The PTY subprocess test runs the actual CLI with `--demo --no-start-worker`, sends
one message to Demo A, switches with Alt+Down, sends one to Demo B, and exits via
Ctrl+Q. It verifies two durable queued tasks with different Bot/session ownership
and confirms no worker lock or execution journals were created. Its exact child
is terminated/reaped in cleanup even on failure.

Scoped Ruff and formatting pass. Outside-sandbox command:
`pytest -q -p no:cacheprovider tests/test_tui_cli.py`: **12 passed in 1.53s**.
