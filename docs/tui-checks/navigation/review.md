Independent navigation review completed on 2026-11-02. The confirmed permission
regression was fixed and independently rechecked; no remaining actionable findings
were identified within the scope below.

The reviewed working-tree identity is recorded in
[source-sha256.json](source-sha256.json), based on commit
`81f42f90b932e61151f399d423fa9a45e55a97e7`. After the fix, SHA-256 values for all eight
inspected source files were verified against that manifest: `cli.py`,
`ui_commands.py`, `bot_tui.py`, `tui_backend.py`, `bootstrap.py`, `bots.py`,
`config.py`, and `terminal_ui.py`, all under `src/hiro_agent/`.

The review was read-only with respect to production code. It examined default TTY
startup, traditional chat entry points, `/agent` resolution, shared completion,
`/tui` and `/chat` transitions, configuration preservation, and session ownership.
Source inspection found that navigation commands are handled before model task
execution, exact Bot IDs disambiguate duplicate names, chat harnesses close before
interface transitions, and a single `Terminal` is reused across chat visits. The
new per-Bot session map records sessions used by the local chat flow; it does not
select a worker's durable task session as the automatic TUI-to-chat target.

One P1 issue was confirmed in the original implementation. An unbound legacy chat
with `permissions="read-only"` could be associated with an existing `default` Bot
whose permissions were `ask-write`. Switching through `/tui` and `/chat` would then
reopen the legacy session using the existing Bot's configuration. The first
offline probe exercised the actual helper and harness sequence behind those
transitions: it created the saved Bot, opened and closed a legacy session, selected
the navigation identity, and reopened the session with `persist_bot=False`. It
observed `read-only` becoming `ask-write`, a new `session.bot` binding to `default`,
and the original system prompt remaining without that Bot's instructions. This
probe did not send keyboard input through a PTY or call a model.

The fix adds `navigation_source()` in `cli.py`. When an unbound source chat meets
an existing `default` identity, it creates a session-derived `chat-<hash>` identity
that preserves the source configuration. An existing generated identity is reused
only when its full configuration matches; otherwise another suffix is tried.
Creation uses `expected_version=0`, so it does not overwrite a conflicting Bot's
policy. An empty workspace can still use `default`.

Independent recheck used real offline harnesses, Bot persistence, and temporary
workspaces, with the mock adapter configured and no model or network calls. The
probes passed these assertions:

- The original collision preserves the legacy chat's read-only permissions,
  model configuration, and custom tool-call budget.
- The restored session binds to the separate generated Bot, while the existing
  `default` configuration remains equal to its original value.
- Repeating identity selection for the same legacy session reuses the matching
  generated Bot.
- An empty workspace assigns `default`.
- A conflicting generated ID causes allocation of another ID and leaves the
  conflicting Bot's configuration unchanged.

These results cover source inspection and the specific offline probes described
above. This reviewer did not run the full suite or independently verify PTY
interaction, repeated-switch resource behavior, or build/static-check results.
Those checks are recorded separately by the implementation and test owners; the
full suite was still running when this review record was written.
