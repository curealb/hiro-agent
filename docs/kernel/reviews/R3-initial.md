# R3 independent acceptance — FAIL

Date: 2026-11-02. Reviewer: fresh R3 acceptance agent. **Initial decision: FAIL; return the bounded shutdown defect to C4. D must not start until the corrected source passes R3 recheck.** No production file or pre-existing test was modified by this reviewer.

## Inspected version and method

- Git baseline: `81f42f90b932e61151f399d423fa9a45e55a97e7`. The kernel implementation is an uncommitted worktree change, so this revision alone does not identify the inspected implementation.
- [R3-initial-manifest.sha256](R3-initial-manifest.sha256) records exact bytes of 34 production modules, 44 test files, `pyproject.toml` and `uv.lock`. Manifest SHA256: `47f24261e419c0c864dc01c1e021ac7fdbedb020e154cd4e50f8182119c10dbf`.
- SHA256 of the production-only manifest lines: `81d92e98a238e17a894ee0073533a803f4a11a5450f05f4fb5e2b33ed6c006b4`. Inspected `src/hiro_agent/kernel_runtime.py`: `f8131b64006740dbc327f5628768deaa6214ccb4b4fe6af480e9e3225dd13ee0`.
- [R3-frozen-input.sha256](R3-frozen-input.sha256) was saved before creating acceptance tests. [R3-initial-hash-verification.log](R3-initial-hash-verification.log) verifies original production/tests and the final inspected manifest remained unchanged through all checks.
- Read REQUIREMENTS, CONTRACTS, R3 packet, actual source and Git diff. C4 handoff was only API/environment orientation. A separate read-only routine probe reported the shutdown issue; this reviewer reproduced it using a new actual runtime acceptance test.
- Reviewer-owned files are only `tests/acceptance_r3/` and `docs/kernel/reviews/R3*`. The process fixture replaces the model through the documented factory seam and runs the real Typer CLI, Agent, worker, SQLite services and JSONL. It also has explicit `os._exit` crash cuts surrounding public Task submission. It does not replace those services with fake implementations.
- Tests use the BASELINE-authorized execution outside the sandbox to avoid the documented asyncio executor shutdown limitation. Dependencies were unchanged. Cached offline packaging uses the existing read-only backend cache.

## Required correction

**P2: a secondary shutdown error hides the durable scheduler error.** In `src/hiro_agent/kernel_runtime.py:519`, KernelWorker saves the scheduler exception; at line 527, `await super()._shutdown()` may raise before lines 528–529 propagate the saved error. That exception is not preserved as cause, context, or a member of an exception group.

Reproduction: [test_failure_acceptance.py](../../../tests/acceptance_r3/test_failure_acceptance.py), `test_scheduler_failure_survives_a_second_failure_during_shutdown`, failing assertion at line 80. A real model request is active, the scheduling loop raises `PersistenceError("R3 scheduler primary failure")`, and runner shutdown raises `RuntimeError("R3 runner secondary close failure")` after its normal cleanup. The caller sees only:

```text
{'R3 runner secondary close failure'}
```

The test walks the entire propagated exception tree, including cause, context and exception groups. The primary error is absent. The same test first proves that the active Task is cancelled, the model has cleaned up, and a replacement worker can acquire the lock. This finding concerns lost failure information, not a claimed lock leak or duplicate task.

This violates CONTRACTS.md:165: a scheduling failure stops the worker, settles active execution and propagates the error. It is also within the R3 packet's mandatory lifecycle integration. Preserve both errors while retaining cleanup and lock-release behavior; C4 owns the correction. The reviewer did not modify production to make the test pass.

## Independent behavioral evidence

Ten parameterized acceptance cases were added; nine pass, one fails as described above. The full existing 788-test suite still passes; the only failing test in the expanded suite is the new double-failure regression.

| Required behavior | Independently executed evidence |
| --- | --- |
| Memory across Tasks, update/delete, Bot isolation | `test_cli_cross_task_memory_update_delete_restart_and_bot_isolation`: every CLI action executes in a fresh process; records are visible in real model requests, revision 2 replaces old content, foreign Bot read fails, foreign memories and task summaries are absent. |
| Deleted summaries do not reappear after restart | The same test deletes the note and both actual successful-task summaries, starts two new workers, confirms the Bot memory list remains empty, then executes another Task and confirms old task results are not injected. |
| Audit retention without cached injection | The first Task's JSONL remains byte-for-byte identical after update/delete/restart and still contains its historical `model.requested` memory. Reopened `derive_messages()` does not contain that reference. This follows the explicit deletion clarification: future injection stops; historical audit is retained. |
| Compression preserves complete tool pairs and raw logs | `test_real_compressed_run_refreshes_memory_without_changing_raw_tool_history`: real Agent performs two batches of two file reads with six-message/6500-character limits. Every request retains the original Task and intact pairs; the final projection contains only the second complete batch. The original Session still derives all four calls/results. Each captured raw-log prefix remains intact. Memory is updated then deleted during the same Run, and each later request sees the fresh state. |
| Skills description/body and authority | `test_cli_skill_body_and_description_cannot_expand_authority[allowlist/read-only]`: enabled description appears on the first request; body appears only as an actual `read_skill` result. Malicious metadata, description and body cannot authorize a write: `permission_denied/not_started`, no file, no approval record, unchanged Bot authority. Disabled/foreign Bot reads fail. |
| Successful Task skill draft | The same CLI cases explicitly create an idempotent draft from a real succeeded Task, verify result text and disabled status, and reject failed-task and foreign-task drafts. No automatic draft appears on task completion. |
| Once-only routine crash recovery | `test_cli_routine_process_crash_recovers_exactly_one_task_after_disable[accepted/submitted]`: actual processes exit 81 before Task submit or 82 after Task commit but before trigger acknowledgement. After disable/restart, the accepted pending trigger is reconciled; repeated tick/worker starts retain exactly one Task, one Run, one model request and one summary. A completed once-only schedule cannot be enabled again. |
| Fixed-interval lifecycle and dedup | `test_cli_interval_backlog_enable_disable_and_repeated_restart`: disabled schedule accepts no new trigger; enable drains three overdue hourly slots with bounded first tick; disable pauses remaining slots; re-enable resumes. Repeated fresh-process ticks/workers produce three distinct scheduled slots and exactly three succeeded Tasks. |
| Scheduling while execution is busy; shutdown | `test_live_scheduler_queues_during_model_wait_and_converges_on_shutdown[False/True]`: a due routine becomes a queued Task while the real model is waiting; only one model request executes. Normal stop and a single persistence error both cancel and settle the active Run, finish owned coroutines and release the worker lock. The second simultaneous shutdown error case fails separately. |

## Actual checks

Exact commands, exits and durations are also in [R3-initial-tests-results.json](R3-initial-tests-results.json) and [R3-initial-static-results.json](R3-initial-static-results.json). The Python orchestration process reports each subprocess exit explicitly; its own successful completion does not mean pytest passed.

| Command | Actual result / log |
| --- | --- |
| `PYTHONDONTWRITEBYTECODE=1 timeout 240s .venv/bin/pytest -q -p no:cacheprovider tests/test_memory.py tests/test_skills.py tests/test_routines.py tests/test_kernel_context_features.py tests/test_kernel_feature_cli.py tests/acceptance_r3` | **exit 1; 1 failed, 114 passed in 64.88s**; [module log](R3-initial-module-tests.log) |
| `RUFF_CACHE_DIR=/tmp/hiro-agent-r3-ruff .venv/bin/ruff check .` | exit 0; All checks passed; [log](R3-initial-ruff.log) |
| `RUFF_CACHE_DIR=/tmp/hiro-agent-r3-ruff .venv/bin/ruff format --check .` | exit 0; 109 files already formatted; [log](R3-initial-format.log) |
| `MYPY_CACHE_DIR=/tmp/hiro-agent-r3-mypy .venv/bin/mypy src` | exit 0; 34 source files checked; [log](R3-initial-mypy.log) |
| `PYTHONDONTWRITEBYTECODE=1 timeout 360s .venv/bin/pytest -q -p no:cacheprovider` | **exit 1; 1 failed, 797 passed in 152.79s**; [full log](R3-initial-pytest.log) |
| Cached offline build, exact command below | exit 0; sdist and wheel built; [log](R3-initial-build.log) |
| `git diff --check` | exit 0; [log](R3-initial-diff-check.log) |

```sh
PYTHONPATH=/home/curealb/.cache/uv/archive-v0/bgLrS0yUwJ8cuofD_HC9j:/home/curealb/.cache/uv/archive-v0/rV7UoyfERUwgy4LcpvHxy:/home/curealb/.cache/uv/archive-v0/Je5sA6EHFWgYlJlNS49Vc:/home/curealb/.cache/uv/archive-v0/Xr0rqq8-K2XtfoB7jWRNG:/home/curealb/.cache/uv/archive-v0/poTN_HFt5dAAQnuvxZwQv:/home/curealb/.cache/uv/archive-v0/E_BRBr-VAGOkDwa8JEmdX PYTHONDONTWRITEBYTECODE=1 UV_CACHE_DIR=/tmp/hiro-agent-uv-cache uv build --offline --no-build-isolation --out-dir /tmp/hiro-agent-r3-initial-dist
```

Build artifact hashes are in [R3-initial-artifacts.sha256](R3-initial-artifacts.sha256). A ZIP-content comparison independently verified that all 34 production modules in the wheel match the inspected source bytes ([log](R3-initial-package-content.log)). Network-isolated backend resolution was not claimed; the cached offline build is the actually executed packaging gate.

The preliminary nine-case run reported `9 passed in 41.58s` in [R3-initial-acceptance.log](R3-initial-acceptance.log), and the focused failure probe reported `1 failed in 0.38s` in [R3-shutdown-failure.log](R3-shutdown-failure.log). Their first shell wrappers attempted to assign zsh's read-only `status` variable after pytest completed and therefore exited 1. They are diagnostic evidence only; the module and full runs above used a corrected subprocess recorder and provide authoritative pytest exit codes.

**Decision: FAIL.** C4 must preserve the original scheduling failure when shutdown also fails, then this same independent reviewer must rerun the regression, required module/full checks and packaging against a newly recorded source manifest. No other bounded defect was found in this review.
