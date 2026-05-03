# isanagent hooks

Hooks extend the agent for **multi-tenant observability** (async sinks) and **policy / context** (synchronous subprocess hooks). Configuration lives under **`[harness.hooks]`** in `config.toml`. Both subsystems are **off** until you set the relevant `enabled = true` flag.

Full implementation: `src/hooks/` (observation sinks, steering executor), wired from `src/agent/mod.rs` (`run_reasoning_loop`, `execute_tool_call_with_activity`).

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

Emitted observation events today: **AgentUsage** (per LLM step), **ToolCall** / **ToolCallStarted** (via `log_tool_invocation_start`), **ToolResult** / **ToolCallFinished** after each tool completes.

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

When `enabled = true`, the agent runs external **shell commands** at fixed lifecycle points. Commands receive **JSON on stdin** (UTF-8). On success (exit code 0), **stdout** may contain a small JSON object to influence behavior.

- **`default_timeout_ms`**, **`max_stdout_bytes`**: bounds per hook (defaults 30_000 ms, 64 KiB stdout). Per-hook and default `timeout_ms` values are **clamped to at least 1000 ms** (and the subprocess wait uses the same floor) so hooks stay usable under load.
- **`matcher`**: optional regex against **tool name** for `pre_tool` / `post_tool`; omitted or empty matches all tools.
- **`cwd`**: optional path relative to **sandbox** (`workspace/.agents`); default is sandbox root. Paths are joined lexically under the sandbox (no escape via `..`).

### Events

| Config key       | When it runs | stdin `hook_event` |
|-----------------|--------------|-------------------|
| `user_prompt`   | After runtime context is prepended to the user message, **before** it is stored in session memory. | `user_prompt` |
| `pre_tool`      | After built-in shell policy checks, **before** `execute_tool_scoped`. | `pre_tool` |
| `post_tool`     | After the tool call completes or is cancelled (cancel → synthetic error payload). | `post_tool` |

### stdin (common fields)

All steering payloads include at least:

- `hook_event`, `schema_version` (1)
- `channel`, `chat_id`, `thread_id`, `is_subagent`
- `workspace_dir`, `sandbox_dir` (strings)
- `metadata` (full inbound metadata map for the turn)

Tool events add: `tool_name`, `tool_call_id` (when known), `args` (JSON). `post_tool` adds `result_ok`, `result`, `error`.

### stdout JSON (steering decisions)

**`pre_tool`**

| `decision`   | Meaning |
|-------------|---------|
| `proceed` / `allow` / omitted | Continue; optional `args` object replaces tool arguments. |
| `modify`    | Same as proceed with **required** `args` (replaces tool arguments). |
| `block` / `deny` | Abort tool; `message` is returned to the model as the tool error. |

Example block:

```json
{"decision": "block", "message": "exec blocked by tenant policy"}
```

Example modify:

```json
{"decision": "modify", "args": {"path": "README.md"}}
```

**`user_prompt`**

| `decision`        | Meaning |
|------------------|---------|
| `proceed` / `allow` / omitted | No change. |
| `inject_prefix`  | Prepend `message` to the contextualized user text (after runtime prefix). Multiple hooks append in order. |
| `block` / `deny` | End the turn with an error; `message` is surfaced as the reasoning loop error. |

**`post_tool`**: stdout is ignored (advisory / logging only).

### Security

- Steering commands run with the same trust model as arbitrary workspace tooling: they can read stdin payloads that may contain prompts and tool arguments. Use **`metadata_keys`** for observation instead of duplicating secrets in webhooks where possible.
- Prefer dedicated service accounts, timeouts, and minimal `cwd` directories.
- On Windows, commands run under `cmd /C`; on Unix under `sh -c`.

### Example

```toml
[harness.hooks.steering]
enabled = true
default_timeout_ms = 5000

[[harness.hooks.steering.pre_tool]]
matcher = "^exec$"
command = "python .agents/hooks/pre_exec.py"

[[harness.hooks.steering.user_prompt]]
command = "python .agents/hooks/user_prompt.py"
```

Note: TOML array-of-tables `[[harness.hooks.steering.pre_tool]]` is used because `pre_tool` is a list of hook definitions.
