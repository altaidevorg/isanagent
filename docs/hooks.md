# isanagent hooks

Hooks extend the agent for **multi-tenant observability** (async sinks) and **policy / context** (synchronous subprocess hooks). Configuration lives under **`[harness.hooks]`** in `config.toml`. Both subsystems are **off** until you set the relevant `enabled = true` flag.

Full implementation: `src/hooks/` (observation sinks, steering executor), wired from `src/agent/reasoning.rs` (`run_reasoning_loop`: `user_prompt`, `post_model`) and `src/agent/tool_dispatch.rs` (`execute_tool_call_with_activity`: `pre_tool`, `post_tool`).

## Observation (`[harness.hooks.observation]`)

When `enabled = true`, non-blocking events are written to optional **JSONL** (path relative to workspace root) and/or **HTTP webhook** (POST JSON body).

- **`metadata_keys`**: list of inbound metadata keys copied into each envelope under `hook_metadata` (for tenant routing). Only keys present on the inbound message are included.
- **`queue_capacity`**: bounded async queue (default 256); when full, additional events are dropped silently.
- **`webhook_hmac_secret`**: when set, requests include header `X-Isanagent-Hook-Signature: sha256=<hex>` (HMAC-SHA256 over the raw POST body).
- **Webhook retries**: up to **3** POST attempts; on failure, waits with **exponential backoff** (250ms base, doubling per retry, capped at 15s) plus **jitter** (derived from wall-clock time) before the next attempt, to avoid hammering a sick endpoint.

### Envelope (`schema_version` = 1)

Each line / POST body is a JSON object:

```json
{
  "schema_version": 1,
  "at": "2026-05-03T12:00:00Z",
  "channel": "api",
  "chat_id": "...",
  "thread_id": null,
  "is_subagent": false,
  "hook_metadata": { "tenant_id": "..." },
  "telemetry": { ... }
}
```

The `telemetry` value is a serialized `TelemetryEvent` from `src/bus.rs` (e.g. `ToolCall`, `ToolResult`, `AgentUsage`, `ToolCallStarted`, `ToolCallFinished`).

Emitted observation events today: **AgentUsage** (per LLM step), **ToolCall** / **ToolCallStarted** (via `log_tool_invocation_start`), **ToolResult** / **ToolCallFinished** after each tool completes, and **HookDecision** (see below) whenever a steering hook changes something.

### Hook audit trail (`HookDecision`)

When observation is enabled, every steering-hook decision that has an effect is emitted as a `HookDecision` event — an audit trail of what hooks (including plugin hooks) did to prompts, tool calls and responses. No-op hook runs are not reported.

```json
{"HookDecision": {
  "chat_id": "...", "channel": "api",
  "hook_event": "pre_tool",          // pre_tool | post_tool | user_prompt | post_model
  "source": "plugin:guard",          // config | plugin:<name>
  "command": "sh \"${CLAUDE_PLUGIN_ROOT}/check.sh\"",
  "tool_name": "exec",               // null for user_prompt / post_model
  "decision": "modify",              // block | stop | ask | modify | retry | error_blocked | error_ignored
  "reason": null,                    // block / ask / stop reason or error text (max 2 KB)
  "changed": ["args", "context"]     // modify: args | prompt | output | content | tool_calls | context
}}
```

The rewritten values themselves are not included (they can be large or sensitive); the envelope goes through the same secret redaction as other observation events. Every decision is also logged at `info` level as `hook <event> [<source>] decision=…`.

### Example

```toml
[harness.hooks.observation]
enabled = true
jsonl_path = ".system_generated/hook_events.jsonl"
webhook_url = "https://edge.example.com/v1/agent-hooks"
metadata_keys = ["tenant_id", "org_id"]
queue_capacity = 512
```

## Steering (`[harness.hooks.steering]`)

When `enabled = true`, the agent runs external **shell commands** at fixed lifecycle points. Commands receive **JSON on stdin** (UTF-8) and may answer with **JSON on stdout** to block, rewrite or annotate what the agent is doing. The contract is modelled on Claude Code hooks: the v1 fields (`decision`, `message`, `args`) keep working, and the v2 fields (`continue`, `stopReason`, `systemMessage`, `hookSpecificOutput`) add input/output rewriting.

- **`default_timeout_ms`**, **`max_stdout_bytes`**: bounds per hook (defaults 30_000 ms, 64 KiB stdout). Per-hook and default `timeout_ms` values are **clamped to at least 1000 ms**.
- **`matcher`**: optional regex. Tested against the **tool name** for `pre_tool` / `post_tool`, the **raw user prompt** (as rewritten by earlier hooks) for `user_prompt`, and the **assistant text** for `post_model`. Omitted or empty matches everything. A `post_model` response that only contains tool calls usually has empty text, so **policy hooks on `post_model` should not use a matcher**.
- **`cwd`**: optional path relative to **sandbox** (`workspace/.agents`); default is sandbox root. Paths are joined lexically under the sandbox (no escape via `..`).
- **`on_error`**: `"proceed"` (default, fail-open) or `"block"` (fail-closed). Applies when the hook itself fails: spawn error, timeout, a non-zero exit other than 2, or stdout that is not valid JSON / uses an unknown decision. **Use `"block"` for policy hooks.** Not used by `post_tool` (its failures are reported to the model as verification output).

Hooks of one event run **in order**. Each hook sees the data as rewritten by the hooks before it; the first `block` / `deny` / `continue: false` wins.

### Events

| Config key    | When it runs | Can |
|---------------|--------------|-----|
| `user_prompt` | Before the user message is stored in session memory. | block, rewrite the prompt, add context |
| `pre_tool`    | **Before** the shell / edit policy and approval prompts, then before execution. | block, rewrite arguments, force an approval prompt, add context, stop the run |
| `post_tool`   | After the tool completes or is cancelled (cancel → synthetic error payload). | append verification output, rewrite the output the model sees, mark it failed, add context, stop the run |
| `post_model`  | After every provider response, **before** it is persisted or any of its tool calls start. | rewrite the assistant text and tool calls, reject the response with feedback, stop the turn |

### Exit codes

| Exit | Meaning |
|------|---------|
| `0`  | stdout (if any) is parsed as the JSON response; empty stdout = no opinion. |
| `2`  | **Block.** stderr (or stdout when stderr is empty) is the reason, shown to the model. For `post_model` this rejects the response (see below). |
| other | Hook failure → `on_error`. |

`post_tool` keeps its verification contract: any non-JSON stdout/stderr, and any non-zero exit, is appended to the tool result (`[post-tool hook]`), so `cargo build` / `pytest` failures reach the model.

### stdin (common fields)

All payloads include: `hook_event`, `schema_version` (**2**), `channel`, `chat_id`, `thread_id`, `is_subagent`, `workspace_dir`, `sandbox_dir`, `metadata` (full inbound metadata map for the turn).

| Event | Extra fields |
|-------|--------------|
| `user_prompt` | `prompt` (raw user text), `content` (runtime context + prompt; v1 field) |
| `pre_tool` | `tool_name`, `tool_call_id`, `args` |
| `post_tool` | `tool_name`, `tool_call_id`, `args`, `result_ok`, `result`, `error` |
| `post_model` | `content`, `tool_calls` (`[{id, name, arguments}]`, arguments parsed as JSON when possible), `iteration` |

### stdout JSON

Common fields (all events):

| Field | Meaning |
|-------|---------|
| `continue: false` | Stop. `stopReason` is shown to the user. `pre_tool` / `post_tool`: abort the run. `user_prompt` / `post_model`: end the turn with an error. |
| `systemMessage` | Logged (info). |
| `decision`, `reason` / `message` | Event-specific, see below. |
| `hookSpecificOutput` | Event-specific object, see below. camelCase and snake_case keys are both accepted. |

**`pre_tool`**

| Field | Meaning |
|-------|---------|
| `hookSpecificOutput.permissionDecision` | `"allow"` (no-op), `"deny"` (skip the tool; `permissionDecisionReason` goes to the model), `"ask"` (force the user approval prompt with the reason, even when the policy would allow the call). |
| `hookSpecificOutput.updatedInput` | Replacement arguments (JSON object). |
| `hookSpecificOutput.additionalContext` | Appended to the tool result under `[pre-tool hook]`. |
| `decision` (v1) | `proceed` / `allow` / omitted, `modify` (requires `args`), `ask`, `block` / `deny`. |
| `args` (v1) | Same as `updatedInput`. |

Notes:
- **Hooks can tighten permissions, never loosen them.** The shell and edit policy evaluate the *rewritten* arguments, so a hook cannot smuggle a command past the denylist or the approval prompt. `allow` does not skip isanagent's own approval; `ask` cannot override `deny` mode, and in **unattended** sessions `ask` blocks the call (no one can answer a prompt).
- An `ask` always prompts, even if the same command / file was granted "always (this run)" earlier. For tools outside the shell / edit policy the "always" grant is keyed by a digest of the full arguments.
- When arguments were rewritten, the tool result tells the model what actually ran (the stored assistant message still holds the model's original arguments).

Example (rewrite + context):

```json
{"hookSpecificOutput": {"hookEventName": "pre_tool",
  "updatedInput": {"command": "ls -la /tmp"},
  "additionalContext": "Listing is scoped to /tmp in this tenant."}}
```

**`user_prompt`**

| Field | Meaning |
|-------|---------|
| `hookSpecificOutput.updatedPrompt` | Replacement user prompt (e.g. PII masking). The runtime context is still prepended by the agent. |
| `hookSpecificOutput.additionalContext` | Prepended to the stored message, like `inject_prefix`. |
| `decision` (v1) | `proceed` / `allow` / omitted, `inject_prefix` (prepend `message`), `block` / `deny`. |

Prefixes from several hooks are joined with newlines and placed before the runtime context.

**`post_tool`**

stdout is treated as a directive only when it is a JSON object containing at least one of `decision`, `continue`, `stopReason`, `systemMessage`, `hookSpecificOutput`; anything else is verification text (appended as before).

| Field | Meaning |
|-------|---------|
| `hookSpecificOutput.updatedOutput` | Replaces the tool output the model sees (string; other JSON is serialized). Later hooks receive the rewritten output. On an error result the error message is replaced too; an MCP App view attached to the result is dropped (it would still show the original data). |
| `hookSpecificOutput.additionalContext` | Appended under `[post-tool hook]`. |
| `decision: "block"`, `reason` | Mark the result as an error (`PolicyDenied`) with the reason appended. The tool has already run. |

**`post_model`**

| Field | Meaning |
|-------|---------|
| `hookSpecificOutput.updatedContent` | Replacement assistant text. |
| `hookSpecificOutput.updatedToolCalls` | Replacement tool-call list `[{id?, name, arguments}]`. Calls keep provider extras when `id` matches an original; calls without `id` get a fresh one; `[]` removes all calls. |
| `decision: "block"`, `reason` | Discard the response and send `reason` back to the model as feedback so it answers again (up to 3 retries per turn; the 4th consecutive rejection fails the turn). |

If a rewrite leaves **neither text nor tool calls** (e.g. `updatedToolCalls: []` on a tool-only response), the response is discarded and the model is told which calls were removed, with the same retry budget — an empty assistant message is never persisted. Hook-supplied ids that are missing, duplicated or not `^[A-Za-z0-9_-]+$` are replaced with fresh `hook_…` ids. On Gemini, new calls get the documented `skip_thought_signature_validator` signature when the original response was signed.

Reasoning / thinking content is never passed to or modified by hooks.

### Plugin hooks (`hooks.json`)

Agent Plugins can ship hooks in `dev.altai.isanagent/hooks.json` (or `hooks.json` at the plugin root, or under `com.google.antigravity/` / `com.github.copilot/`). Because a plugin hook runs code from the plugin, **plugin hooks only run when steering is enabled and the plugin is allow-listed**:

```toml
[harness.hooks.steering]
enabled = true
plugin_hooks = ["guard", "pii-mask"]   # or ["*"] for every installed plugin
```

Plugins whose hooks are skipped are listed in the startup log. Config hooks run first, then plugin hooks in plugin-name order; the same chaining, `on_error` and policy rules apply, and audit events carry `source = "plugin:<name>"`.

Two `hooks.json` shapes are accepted (optionally wrapped in a top-level `"hooks"` object):

```json
{"hooks": {
  "PreToolUse": [{"matcher": "^exec$", "hooks": [
    {"type": "command", "command": "sh \"${CLAUDE_PLUGIN_ROOT}/check.sh\"", "timeout": 10}
  ]}],
  "UserPromptSubmit": [{"hooks": [{"type": "command", "command": "python3 \"$PLUGIN_ROOT/mask.py\""}]}]
}}
```

```json
{"pre_tool":   [{"matcher": "^exec$", "command": "sh \"$ISANAGENT_PLUGIN_ROOT/check.sh\"", "on_error": "block"}],
 "post_model": [{"command": "python3 \"$ISANAGENT_PLUGIN_ROOT/policy.py\"", "timeout_ms": 5000}]}
```

- Events: `pre_tool` / `PreToolUse`, `post_tool` / `PostToolUse`, `user_prompt` / `UserPromptSubmit`, `post_model` / `PostModel`. Other events (e.g. `Stop`, `SessionStart`) and hook types other than `"command"` are skipped with a warning.
- `timeout` is in seconds (Claude Code); `timeout_ms` in milliseconds. `matcher: "*"` matches everything.
- Matchers are regexes over **isanagent tool names** (`exec`, `read_file`, `write_file`, `edit_file`, …), not Claude Code's (`Bash`, `Edit`, …).
- The hook runs in the agent sandbox (like config hooks) with `ISANAGENT_PLUGIN_ROOT`, `CLAUDE_PLUGIN_ROOT` and `PLUGIN_ROOT` set to the absolute plugin directory; reference plugin files through them (Unix shells expand `${CLAUDE_PLUGIN_ROOT}`).

### Security

- Steering commands run with the same trust model as arbitrary workspace tooling: they can read stdin payloads that may contain prompts, tool arguments and tool output. They run in the scrubbed safe environment (no host credentials).
- Prefer `on_error = "block"`, dedicated service accounts, timeouts, and minimal `cwd` directories for policy hooks.
- On Windows, commands run under `cmd /C`; on Unix under `sh -c`.

### Example

```toml
[harness.hooks.steering]
enabled = true
default_timeout_ms = 5000

[[harness.hooks.steering.pre_tool]]
matcher = "^exec$"
command = "python .agents/hooks/pre_exec.py"
on_error = "block"

[[harness.hooks.steering.user_prompt]]
command = "python .agents/hooks/mask_pii.py"

[[harness.hooks.steering.post_tool]]
matcher = "^(read_file|exec)$"
command = "python .agents/hooks/redact_secrets.py"

[[harness.hooks.steering.post_model]]
command = "python .agents/hooks/response_policy.py"
```

Note: TOML array-of-tables `[[harness.hooks.steering.pre_tool]]` is used because each event is a list of hook definitions.

### Migrating from schema v1

- `schema_version` on stdin is now `2`; all v1 stdin fields are still sent.
- **Behaviour changes:**
  - `pre_tool` hooks now run *before* the shell / edit policy and the approval prompt (previously the policy checked the pre-rewrite command). Side effects of a `pre_tool` hook therefore happen before the user approves.
  - `pre_tool`: a top-level `args` object now rewrites the arguments with any `decision` (or none), not only with `modify`.
  - Exit code `2` from `pre_tool` / `user_prompt` / `post_model` now blocks (previously any non-zero exit was ignored). In `post_tool` it is still just verification output.
  - `post_tool` stdout that is a JSON object with a contract key (`decision`, `continue`, `stopReason`, `systemMessage`, `hookSpecificOutput`) is now acted on instead of being shown to the model as text.
  - The `user_prompt` matcher now tests the raw prompt instead of runtime context + prompt.
  - stderr of every hook is now captured (bounded by `max_stdout_bytes`) and used as the block reason for exit 2.
- Known limits: `updatedPrompt` changes what the model sees and what is stored in session memory, but the raw inbound message is still written to the conversation log (`conversation.jsonl`) and used by internal heuristics, so it is not a complete PII scrub. A result that is waiting on `ask_user` skips post_tool decisions. There are no `stop`, `session_start` or `pre_compact` events yet.
