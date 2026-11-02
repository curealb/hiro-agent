# Independent TUI acceptance review

Reviewed 2026-11-02. Result: **PASS after one resize fix**. No outstanding actionable findings in the reviewed scope.

The review read `bot_tui.py`, `tui_backend.py`, the additive `read_session_snapshot()` reader, the `tui` CLI command, and the existing UI/backend/CLI tests. Acceptance changes are isolated to `tests/test_tui_acceptance.py`; this reviewer did not edit production files or the existing tests.

## Finding and recheck

Mouse-clicking the Agent list in a wide terminal left the compact-pane selector on the conversation. Resizing to 42×12 then hid the still-focused list, allowing arrow keys to switch invisible rows. The implementation owner added synchronization between actual focus and the compact pane before rendering. The unchanged reproduction now passes, including switching to another Agent, returning to the composer, restoring the wide layout, and retaining the original draft.

## Independent evidence

Eight acceptance cases use the actual durable Bot/task/interaction stores and prompt-toolkit input, plus a real subprocess attached to a PTY:

- A question reply remains owned by Alpha when the next input immediately switches to Beta; a subsequent task belongs only to Beta. The unrelated Beta question remains pending.
- A delayed Alpha journal read cannot replace Beta's displayed conversation. The live writer keeps its lock, an incomplete journal tail remains byte-for-byte intact, and the running task/run are unchanged.
- Resizing after mouse focus preserves a visible, usable pane and the per-Bot draft.
- Question dialogs can accept and save answers at 80×24 and 42×12.
- Clicking the background and pressing the Agent-switch shortcut while a question dialog is open cannot move focus out of the dialog, switch Bots, or submit a task. Escape leaves the durable request unanswered.
- A background Bot's new task creates its unread dot; selecting that Bot clears the dot and displays the correct message.
- The actual `hiro-agent tui --demo` process starts a detached worker in its own session. After Ctrl+Q exits the TUI with code 0, that same worker accepts a new task, waits for the normal durable approvals, and completes it. The temporary worker is explicitly stopped by test cleanup.

Final command, executed outside the sandbox because `docs/kernel/handoffs/BASELINE.md` documents sandbox asyncio shutdown interference:

```sh
PYTHONDONTWRITEBYTECODE=1 timeout 90s .venv/bin/pytest -q -p no:cacheprovider tests/test_tui_acceptance.py
```

Result: **8 passed in 7.33s**. Scoped Ruff lint and formatting checks also pass. The final run includes the focus synchronization and task-failure explanation fixes. The root collaborator owns the full regression suite and overall packaging/type checks; these were not duplicated here. External model-provider integration was outside this acceptance scope.
