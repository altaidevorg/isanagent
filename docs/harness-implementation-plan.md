# Agent harness: implementation plan

This document tracks expanding the built-in tool surface toward a strong **code-agent harness**: fast discovery, precise edits, workflow control, delegation, and integrations. External reference implementations informed the shape of the catalog; naming and wiring here are **isanagent-native**.

## Principles

- All filesystem tools respect `resolve_path` and `restrict_to_workspace` (see `AGENTS.md`).
- Prefer bounded output (line/char caps), timeouts on subprocesses, and clear errors over silent truncation where possible.
- Register new tools in `src/bin/isanagent.rs` unless registration is later centralized.

## Phases

### Phase 1 — Discovery and precise edit (implemented)

| Tool | Role |
|------|------|
| `glob_files` | List paths under a base directory matching a glob pattern (bounded). |
| `search_text` | Regex search across files: ripgrep when available, Rust fallback otherwise. |
| `edit_file` (extended) | Optional `replace_all`; unified diff snippet in the result. |

### Phase 2 — Workflow

- Structured todo list tool keyed by session (`chat_id`), not a process-global buffer.
- Lightweight search over registered tool names/descriptions for large tool sets.
- Skill loading: align optional “invoke” or preview behavior with existing `load_skill_instructions` where useful.

### Phase 3 — User clarification

- Structured “question the user” path that maps to real channel behavior (terminal vs API), including how replies re-enter the agent loop.

### Phase 4 — Git worktrees

- Optional `git worktree` helpers with explicit policy for paths outside the sandbox (config-gated).

### Phase 5 — Sub-agents and plans

- Background tasks, cancellation, optional typed agent presets, and multi-step plans executed with dependency ordering. This touches `AgentLogic`, session scope, and likely a small runtime type—not only new `Tool` impls.

### Phase 6 — Notebook, LSP, MCP, remote

- Jupyter cell edits (JSON `.ipynb`).
- LSP: staged rollout (stub vs one language server vs full bridge).
- MCP: optional feature, async-friendly bridge, strict capability review.
- Remote triggers: align with existing channels/webhooks or keep as no-op until specified.

## Phase 1 acceptance

- `glob_files` returns sorted paths, caps total matches, errors on invalid glob or path escape.
- `search_text` supports `files_with_matches`, `content`, and `count`, optional glob filter and context lines; output capped; 30s timeout for `rg`.
- `edit_file` rejects identical old/new, enforces uniqueness when `replace_all` is false, and returns a truncated unified diff after success.
