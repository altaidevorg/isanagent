# Execution plane: phased implementation plan

This document describes how to add **reproducible code execution** to `isanagent`: a stable agent-facing API, **capability discovery** (metadata + optional traits), **executor preflight**, and **pluggable backends** (local, Jupyter, SSH, hosted). It complements `docs/harness-implementation-plan.md`: harness Phase 6’s original “notebook / LSP / MCP / remote” bullets are **orthogonal**; **execution** is the primary path for an “AI research and engineer” agent. LSP/MCP/remote triggers can follow later.

See `AGENTS.md` for workspace/sandbox rules, actor discipline, and telemetry conventions.

## Goals

- **Agent tools** speak in stable verbs: sessions, run, cancel, outputs—**not** “Colab” or “SSH” in the tool name unless you deliberately expose provider selection.
- **`ExecutionProvider`** (core) + **`ExecutionCapabilities`** (and optional **capability traits**) describe what a backend can do.
- **Both** injection of capability summaries into context **and** hard **preflight** in the executor so unsupported operations fail fast with structured errors.
- **Long-lived kernels/subprocesses** live behind an **actor** (or dedicated task owner), not ad hoc mutexes around shared state.
- **Security**: allowlisted providers, timeouts, output/size caps, secrets only from host env—not from model tool args.

## Architecture (summary)

| Layer | Responsibility |
|--------|------------------|
| **Capabilities** | `ProviderCapabilities` / `SessionCapabilities` (versioned, serde-friendly); optional extension traits (`SshShell`, `GpuInfo`, …) for operations that are not universal. |
| **Core trait** | Session lifecycle, `run`, `cancel`, consistent error types; every adapter implements this. |
| **Executor / router** | Maps tool calls → provider; **preflight** against capabilities; applies global limits. |
| **Execution actor** | Owns child processes, kernel handles, session table; integrates with cooperative **cancellation** (`CancellationToken` / parent cancel). |
| **Tools** | Thin JSON tools; optional `BusMessage::Outbound` for slow runs; `TelemetryEvent` for run outcomes. |
| **Config** | `[harness.execution]` or `[execution.*]` gated sections: enabled providers, defaults, caps, `pip`/network policy where applicable. |
| **Skills** | `SKILL.md` under `.agents/skills/` for safe iteration, debugging, and provider-specific notes. |

---

## Phase 0 — Contracts and types (no runtime yet)

**Objective:** Lock in names, error taxonomy, and capability schema so later phases do not churn public shapes.

**Deliverables**

- Rust types: `ExecutionError`, `RunSpec` (code, timeout, cwd policy), `RunResult` (stdout/stderr, exit status, structured attachments path or inline cap).
- `ProviderCapabilities` / `SessionCapabilities` with **extensible** fields (defaults for forward compatibility).
- Core async trait `ExecutionProvider` (session create/close, run, cancel) + optional **extension traits** for non-universal features (empty impls or `Unsupported` return where appropriate).
- Design note: object-safe core trait vs enum of known providers—pick one and document it.

**Acceptance**

- Unit tests serialize/deserialize capability snapshots (JSON) with unknown fields ignored or preserved per project policy.
- Documented mapping from capabilities → **which tools are registered** or **which tool branches are valid** (preflight table).

**Depends on:** nothing (design-only PR).

**Status (implemented):** Rust module `src/execution/` (`error`, `ids`, `run`, `capabilities`, `provider`, `preflight`), exported as `isanagent::execution`. Unit tests cover capability JSON round-trip (unknown keys captured in `ProviderCapabilities.extensions`) and `allowed_optional_tool_tags`. See `execution::PREFLIGHT_MARKDOWN` for the operator-facing matrix.

---

## Phase 1 — Local provider (sandbox subprocess / REPL)

**Objective:** First real backend: run code **under workspace/sandbox policy**, with strict timeouts and output caps.

**Deliverables**

- `LocalExecutionProvider`: ephemeral `python`/`uv`/`cargo` style runs and/or a **persistent** REPL child with interrupt on cancel (start with one language, e.g. Python, if that reduces scope).
- cwd rooted in `sandbox_dir` (or configured subdir); env allowlist; no network by default if you want a safe default (configurable).
- Integration with **cancellation**: map to process kill / interrupt where OS allows (note Windows vs Unix differences; follow `AGENTS.md` Windows guidance).

**Acceptance**

- Run succeeds/fails with bounded stdout/stderr; timeout returns clear error.
- Cancel during run terminates the process without wedging the actor.
- Clippy/tests pass per repo workflow (`cargo fmt`, `cargo clippy --release -p isanagent --all-targets`, tests on Windows in release where applicable).

**Depends on:** Phase 0.

**Status (implemented):** `LocalExecutionProvider` + `LocalExecutionConfig` in `src/execution/local.rs`: sessions under `resolve_path` against `sandbox_dir`, Python (`-u -c`) or shell (`cmd /C` / `sh -c`), `wait_with_output` with per-stream byte caps, inner wall-clock timeout, overlapping runs rejected, `cancel` + PID-based `taskkill`/`kill` best-effort on Windows/Unix. Windows subprocesses use `CREATE_NO_WINDOW` on the std command. Unit tests use shell on Windows and Python on Unix for portability.

---

## Phase 2 — Execution actor + harness tools + config gate

**Objective:** Expose Phase 1 through **gated** agent tools and a single owner for sessions.

**Deliverables**

- `ExecutionActor` (or equivalent): session table, spawn/kill, message API from tools; optional persistence of session metadata (in-memory v1; SQLite later if needed).
- Tools (names illustrative): `execution_session_create`, `execution_run`, `execution_cancel`, `execution_session_close`, optional `execution_env_info`.
- Config: `[harness.execution]` with `enabled`, `default_provider`, limits (`max_output_bytes`, `max_wall_secs`, `max_sessions`), provider allowlist.
- **Capability injection**: on session create or on demand, attach a **short** capability summary to the tool result and/or inject into system/tool preamble for that chat (bounded size).
- **Preflight**: tool handler checks capabilities before calling extension APIs (e.g. no SSH tool unless provider exposes it).

**Acceptance**

- With `enabled = false`, execution tools are not registered (mirror `git_worktree` / `subagents` pattern).
- With `enabled = true`, happy path + timeout + cancel paths covered by tests (mock provider acceptable for tool routing tests).
- Telemetry hook for run duration/outcome (optional in v1, required before Phase 6 of this doc if you want analytics parity).

**Depends on:** Phase 1.

---

## Phase 3 — Jupyter / notebook-server provider

**Objective:** Real **kernel** semantics: persistent variables, interrupt, shutdown—without centering on `.ipynb` JSON editing.

**Deliverables**

- Adapter that speaks to a **user-configured** Jupyter server (URL + token from env/config—not committed secrets).
- Map core trait to kernel lifecycle (create, execute, interrupt, delete).
- Capability flags: `jupyter_kernel: true`, `supports_interrupt: true`, language list, optional GPU visibility if reported by environment.

**Acceptance**

- Against a local Jupyter instance in CI or opt-in manual test doc: one execute + interrupt + shutdown succeeds.
- Clear errors when server unreachable or token missing.

**Depends on:** Phase 0–2 (reuse actor + tools; add provider id in `execution_session_create`).

---

## Phase 4 — SSH provider + `SshShell` (or equivalent) capability trait

**Objective:** Remote workstation/HPC flows with the **same** run/cancel API where possible; SSH-specific operations gated by trait + capability bits.

**Deliverables**

- `SshExecutionProvider` implementing core trait; optional trait for interactive shell or `scp`-style staging if needed.
- Config: host, user, key path, remote base dir, strict host key policy (document risks).
- Staging policy: **bounded** copy of sandbox subset to remote (or “run only in remote cwd” without sync—document).

**Acceptance**

- Preflight: if agent calls SSH-only tool without capability, structured `Unsupported`.
- Cancel maps to channel signal / remote interrupt best-effort.

**Depends on:** Phase 0–2; Phase 3 optional (no hard dependency).

---

## Phase 5 — Hosted / Colab-shaped provider (narrow v1)

**Objective:** One hosted path with explicit **auth and limitation** documentation—not feature parity with local on day one.

**Deliverables**

- Narrow scope: e.g. “attach to existing runtime” or minimal API surface you can support reliably.
- Separate capability profile: auth method, quotas, no SSH, etc.
- Skill: user-facing steps for auth and safety.

**Acceptance**

- Fails closed when credentials absent; never logs secrets.
- Documented manual test path; automated tests mock HTTP where feasible.

**Depends on:** Phase 0–2 minimum; real value stacks after Phase 3 patterns exist.

---

## Phase 6 — Research & engineering pack

**Objective:** Make the loop **useful for ML/research**: artifacts, experiments, library-oriented skills—not just `print`.

**Deliverables**

- Artifact sink: plots/tables written under `workspace_dir` (or sandbox) with size limits; tool to list/fetch recent artifacts by id.
- Optional: structured “run manifest” (git sha, provider, pip freeze snippet) appended to telemetry or workspace log.
- Built-in **skills** (optional `always: false`): debugging, profiling, common scientific stack pointers (link out to docs; avoid dumping huge text into every prompt).

**Acceptance**

- Large outputs never blow the LLM context: truncation + “see artifact path” behavior is deterministic.
- Skills load via existing `load_skill_instructions` flow.

**Depends on:** Phase 2+ (any provider).

---

## Optional / deferred (explicitly not blocking the above)

| Item | Note |
|------|------|
| **`.ipynb` read/write tool** | Convenience only; defer or add after Phase 3 if users want file-level notebook edits. |
| **MCP / LSP / remote webhooks** | Stay in `harness-implementation-plan.md` Phase 6; can wrap execution APIs later via MCP. |
| **Cluster schedulers (Slurm, etc.)** | New capability trait + provider when SSH/local prove stable. |
| **SQLite session persistence** | Only if you need resume-after-restart for long kernels. |

---

## Cross-document alignment

- **`docs/harness-implementation-plan.md`**: Phase 6 should reference this file for **execution** scope; notebook JSON / LSP / MCP remain separate, deferrable tracks.
- **`docs/harness-handover.md`**: On resuming work, point “next execution work” to this plan.
- **`AGENTS.md`**: Update when tools/config/actor boundaries land (per project rule: keep architectural context current).

---

## Suggested PR sequence

1. Phase 0 (types + traits + docs only).  
2. Phase 1 + minimal tests.  
3. Phase 2 (actor + tools + config + preflight + bounded capability text).  
4. Phase 3 → 4 → 5 as independent verticals behind provider flags.  
5. Phase 6 as incremental quality on top of any provider.

This keeps each merge **shippable** and avoids blocking local execution on Colab or SSH.
