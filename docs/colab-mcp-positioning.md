# Colab MCP: positioning and extending beyond `ExecutionProvider`

This document explains what [googlecolab/colab-mcp](https://github.com/googlecolab/colab-mcp) actually is, which capabilities are **dynamic** (not fixed in the Python repo), and how `isanagent` can expose Colab-specific features without abandoning the universal execution trait.

## What Colab MCP is architecturally

From upstream `session.py`:

1. **Local FastMCP process** — stdio server (`uvx git+https://github.com/googlecolab/colab-mcp`).
2. **`open_colab_browser_connection`** — injected tool that opens Colab in the browser and waits (up to ~60s) for a UI session.
3. **`FastMCPProxy`** — after connect, **most tools are proxied from the Colab front-end** over a WebSocket, not defined as static Python `@tool` functions in the repo.

So the authoritative catalog of “what Colab MCP can do” is **`tools/list` after the browser session is live** (and it can change when connectivity changes; upstream sends `notifications/tools/list_changed`).

Implication: positioning `colab_mcp` purely as “a Python runner behind `ExecutionProvider::run`” is correct for **code execution**, but **incomplete** for product framing: users and agents should understand Colab MCP as **“remote Colab session control over MCP,”** where execution is one subset of tools.

## Typical tool families (discovered at runtime)

Exact names and schemas vary by Colab build. The following families appear in practice and in debugging (`tmp_colab_mcp_probe.py`):

| Area | Representative tools / behavior |
|------|----------------------------------|
| **Session / UI** | `open_colab_browser_connection`, connection state, `tools/list` refresh after connect |
| **Notebook** | `add_code_cell`, `run_code_cell`, `get_cells`, `update_cell`, indices like `cellIndex`, `newCellId` / `cellId` |
| **Runtime / hardware** | Tools related to runtime type, GPU/TPU, memory (names vary; often surfaced as notebook or “session” operations) |
| **Drive / files** | Mount Google Drive, upload/download or path helpers (when exposed as MCP tools by Colab) |
| **Other** | Anything the Colab UI adds to the proxied tool list |

Treat this table as **families**, not a frozen API contract.

## How to position `colab_mcp` in `isanagent`

**Primary story (today):** `colab_mcp` is an **execution backend** for Python in a real Colab notebook: `execution_session_create` / `execution_run` map to MCP `tools/call` with schema-driven discovery (direct execute vs notebook cell flow).

**Secondary story (roadmap):** the same MCP connection is a **Colab session adapter** — optional agent or operator features that call **other** MCP tools with arguments derived from `tools/list` schemas.

## Integration strategy: trait-first, extensions explicit

`ExecutionProvider` is intentionally minimal; `provider.rs` already documents the pattern: **optional capabilities live in separate traits or capability maps**, not as extra methods on the core trait.

Recommended layers (in order of preference):

### 1. Keep `ExecutionProvider` for “run code” only

- Stable contract for the agent: sessions, `run`, timeouts, journals.
- Colab-specific quirks (cell ids, `cellIndex`, post-connect tool discovery) stay inside `ColabMcpExecutionProvider`.

### 2. Advertise Colab extras in `ProviderCapabilities` / session extensions

- At `create_session`, snapshot **high-level** facts: execution mode (`direct` vs `notebook_cells`), tool names chosen, optional `tools/list` **summary** (names only, capped) for operator visibility.
- Avoid dumping huge schemas into every session unless needed for debugging.

### 3. Optional **narrow trait** for “MCP tool bridge” (preferred if multiple backends need it)

Introduce something like `McpSessionTools` (name TBD) implemented only by `ColabMcpExecutionProvider` (and maybe future Jupyter MCP):

- `async fn call_named_tool(session_id, tool_name, arguments: Value) -> Result<Value, ExecutionError>`
- `fn list_known_tool_names(&self, session_id) -> Option<Vec<String>>` (from last `tools/list` cache)

The harness or a dedicated **`colab_mcp_*` agent tool** can downcast / match on `provider_id` and call this. This preserves **trait-based** structure without polluting `ExecutionProvider`.

### 4. Dedicated agent tools (special case, still clean UX)

Implemented in `isanagent`:

- **`colab_mcp_tool_call`** — passthrough `tools/call` for an existing Colab MCP execution session when config registers it for `default_provider = "colab_mcp"`. Use `list_cached_tool_names: true` to read the cached `tools/list` names. Prefer **`execution_run`** for Python.

Optional future: `colab_mcp_tools_list` with richer schema hints (today the cache + `execution_env_info` colab note suffice for many flows).

This stays a **tool-registry** concern, not an `ExecutionProvider` method.

### 5. Full “MCP client in the agent” (only if needed)

Register colab-mcp as a **separate MCP server** in addition to the execution harness. Duplicates process management and auth UX; use only if you need full MCP semantics (resources, prompts) independent of execution.

## When to extend `ExecutionProvider` vs not

| Need | Extend `ExecutionProvider`? |
|------|-----------------------------|
| Run code / cells / stdout | No — already covered |
| Interrupt / cancel remote kernel | Yes, if Colab exposes it reliably through MCP |
| “Install package” mapped to `%pip` or tool | Maybe — could be `PackageOperations` or Colab-specific trait |
| Mount Drive, pick GPU, upload file | Prefer **separate trait or agent tool**, not `run()` overloads |

Overloading `RunSpec` with Colab-only fields tends to rot; prefer **explicit tools** or a **small extension trait**.

## Upstream client expectations

The Colab MCP README states that clients should support **`notifications/tools/list_changed`**. `isanagent` now marks the MCP client dirty on that notification, **refreshes `tools/list` into a per-session cache** before the next `execution_run` or `colab_mcp_tool_call`, and records tool names for `list_cached_tool_names` (session capabilities extensions are unchanged for stability).

## References

- [colab-mcp README](https://github.com/googlecolab/colab-mcp/blob/main/README.md) — client requirements, setup.
- [colab-mcp `session.py`](https://github.com/googlecolab/colab-mcp/blob/main/src/colab_mcp/session.py) — proxy architecture, `open_colab_browser_connection`, `send_tool_list_changed`.
- `tmp_colab_mcp_probe.py` in this repo — stdio protocol and execution-path probing.
