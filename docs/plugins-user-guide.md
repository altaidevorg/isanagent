# Agent Plugins 1.0 User Guide

`isanagent` natively implements the **Agent Plugins 1.0 Specification** ([agent-plugins.org](https://agent-plugins.org)), an open, vendor-neutral standard jointly published and maintained by **Microsoft / GitHub, Google, AWS, OpenAI, Anysphere (Cursor), and Vercel**.

Agent Plugins 1.0 standardizes how **Agent Skills**, **Model Context Protocol (MCP) servers**, **declarative subagents**, **rules**, and **lifecycle hooks** are packaged, distributed, and discovered across modern AI agent runtimes and IDEs.

---

## 1. Package Structure

An Agent Plugin is a self-contained directory containing a root manifest (`plugin.json`), optional standardized components (`skills/`, `mcp.json`), and client extension directories (`dev.altai.isanagent/`):

```text
my-plugin/
├── plugin.json                 # Required: Root manifest targeting Agent Plugins 1.0 schema
├── mcp.json                    # Optional: Standard MCP server declarations
├── skills/                     # Optional: Standard portable Agent Skills
│   └── data-analysis/
│       ├── SKILL.md
│       ├── scripts/
│       └── references/
├── dev.altai.isanagent/         # Client Extension Namespace (for isanagent & altai-app)
│   ├── agents/                 # Declarative subagents
│   │   └── researcher/
│   │       └── AGENT.md
│   ├── rules/                  # Behavioral markdown rules
│   │   └── code_style.md
│   └── hooks.json              # Lifecycle hooks configuration
├── LICENSE
└── README.md
```

---

## 2. Manifest (`plugin.json`)

The manifest identifies the plugin and provides metadata. It follows a closed schema where client-specific extensions belong under the `extensions` map:

```json
{
  "$schema": "https://agent-plugins.org/schemas/1.0.0/plugin.schema.json",
  "name": "ml-engineer",
  "version": "1.0.0",
  "description": "Autonomous ML engineering, GPU benchmarking, and dataset analysis plugin",
  "author": {
    "name": "Altai Labs",
    "url": "https://altai.dev"
  },
  "homepage": "https://altai.dev/plugins/ml-engineer",
  "repository": "https://github.com/altaidevorg/plugin-ml-engineer",
  "license": "Apache-2.0",
  "keywords": ["ml", "gpu", "jax", "triton", "benchmarking"],
  "extensions": {
    "dev.altai.isanagent": {
      "min_version": "0.13.0",
      "overlay_prompt": "Always verify GPU environment and PyTorch/JAX dependencies before launching compute runs."
    }
  }
}
```

### Manifest Fields

| Field | Type | Description |
| :--- | :--- | :--- |
| **`$schema`** | `string` | Canonical schema: `https://agent-plugins.org/schemas/1.0.0/plugin.schema.json` |
| **`name`** | `string` | Unique, human-readable plugin identifier. |
| **`version`** | `string` | Semantic version string (e.g. `1.0.0`). |
| **`description`** | `string` | Short summary of plugin capabilities. |
| **`author`** | `object`/`string` | Author details (`name`, `email`, `url`). |
| **`homepage`** | `string` | Documentation or product URL. |
| **`repository`** | `string` | Source repository Git URL. |
| **`license`** | `string` | SPDX license identifier (e.g. `Apache-2.0`, `MIT`). |
| **`keywords`** | `string[]` | Discovery tags. |
| **`extensions`** | `object` | Client-specific configurations keyed by reverse-domain namespace. |

---

## 3. Reverse-Domain Client Extensions (`dev.altai.isanagent`)

The Agent Plugins 1.0 specification standardizes skills and MCP servers for cross-vendor portability. Advanced capabilities (declarative subagents, prompt overlays, rules, hooks) live in **reverse-domain extension namespaces**:

- **Namespace for `isanagent`**: `dev.altai.isanagent` (and alias `dev.altai`).
- **Filesystem directory**: `<plugin-root>/dev.altai.isanagent/`.
- **Manifest extension block**: `extensions["dev.altai.isanagent"]`.

> **Cross-Client Compatibility**: `isanagent` also automatically discovers subagents, rules, and hooks placed under `com.google.antigravity/` and `com.github.copilot/`, allowing plugins built for Google Antigravity or VS Code Copilot to run smoothly in `isanagent`.

---

## 4. Declarative Subagents (`AGENT.md`)

Subagents defined in plugins are placed under `dev.altai.isanagent/agents/<agent_name>/AGENT.md` (or root `agents/<agent_name>/AGENT.md`).

Each `AGENT.md` uses YAML frontmatter for metadata followed by Markdown instructions for the subagent's system prompt:

```yaml
---
name: researcher
description: Deep research subagent with arXiv, Hugging Face, and primary-source web extraction
mode: subagent
temperature: 0.1
max_iterations: 15
color: "#2196F3"
allowed_tools:
  - web_search
  - web_fetch
  - arxiv_search
  - arxiv_fetch
  - read_file
  - search_text
  - glob_files
  - list_dir
  - search_memory
  - fetch_memory_by_date
  - todo_write
  - recall_tool_result
---

# Deep Research Specialist

You are a focused sub-task researcher and literature synthesizer.
Follow this systematic workflow:
1. **Discovery**: Use `web_search` and `arxiv_search` to shortlist candidate papers and official repositories.
2. **Primary Sources**: Fetch full texts with `web_fetch` or `arxiv_fetch`.
3. **Cross-Check**: Verify findings across at least two independent sources.
4. **Synthesis**: Present structured findings with exact citations, hyperparameters, and datasets.
```

### Frontmatter Configuration

| Key | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `name` | `string` | directory name | Identifier used in `subagent_spawn(agent="...")`. |
| `description` | `string` | `""` | Description injected into coordinator prompts. |
| `mode` | `string` | `"subagent"` | Mode: `"subagent"` or specialized worker mode. |
| `temperature` | `float` | config default | LLM temperature override. |
| `max_iterations` | `int` | config default | Turn iteration ceiling. |
| `allowed_tools` | `string[]` | full allowlist | Strict tool subset available to this subagent. |
| `color` | `string` | `"#888888"` | Hex color code for TUI and UI clients. |

---

## 5. Model Context Protocol (`mcp.json`)

Declare MCP servers at the plugin root in `mcp.json`:

```json
{
  "$schema": "https://agent-plugins.org/schemas/1.0.0/mcp.schema.json",
  "mcpServers": {
    "git-mcp": {
      "type": "stdio",
      "command": "git-mcp-server",
      "args": ["--read-only"]
    }
  }
}
```

### Tool names

Every MCP tool is exposed to the model as `{server}__{tool}` — `chem__rdkit_mol_weight` for tool `rdkit_mol_weight` of server `chem`. Server and tool can therefore never shadow a built-in tool (`read_file`, `exec`, …) or another server's tool. Providers only accept `[A-Za-z0-9_-]` up to 64 characters, so other characters become `_`, and a longer name keeps a readable prefix plus a stable 8-character hash. The server still receives its own tool name in `tools/call`.

Use the exposed name wherever isanagent refers to a tool: `[harness.permissions.tools]` (a trailing `*` covers a whole server: `"chem__*" = "ask"`), sub-agent `allowed_tools`, and a skill's `allowed-tools`. Two plugins declaring the same server name: the second is skipped with a warning.

### Variables, environment and sandbox

`command`, `args`, `env` and `cwd` expand `${VAR}` and `${VAR:-default}`:

| Variable | Value |
| :--- | :--- |
| `${PLUGIN_DIR}` (also `${CLAUDE_PLUGIN_ROOT}`, `${CLAUDE_PLUGIN_DIR}`) | The plugin directory. |
| `${SANDBOX_DIR}` | The agent sandbox (where `read_file` / `write_file` work). |
| any other name | The host environment variable. |

The server process starts from the scrubbed safe environment: it inherits nothing secret by default. A credential-looking variable (`*_KEY`, `*_TOKEN`, `*_SECRET`, …) expands only when the user lists it in `config.toml`, so a plugin cannot read the host's provider keys on its own:

```toml
[harness.mcp]
env_passthrough = ["NCBI_API_KEY"]
```

Every server also gets `ISANAGENT_SANDBOX_DIR`, and answers to MCP `roots/list` name the sandbox, so file-reading tools can resolve relative paths against the same directory the agent writes to. The server's working directory stays the plugin directory unless `cwd` says otherwise (`"cwd": "${SANDBOX_DIR}"`). These tell the server where the sandbox is; they do not confine it: an MCP server is a normal process with the user's permissions.

### Robustness

- Servers start concurrently. Each must answer `initialize` and `tools/list` within the startup timeout (default 60 s); one that fails or times out is logged and skipped without delaying the others. A server whose first start installs packages (`uv run` resolving RDKit, …) should raise `startupTimeoutSecs`.
- Each `tools/call` has a deadline (default 300 s). On timeout, or when the user stops the turn, the server receives `notifications/cancelled`.
- stdout must carry JSON-RPC only. Lines that are not JSON (a library printing at import time) are skipped with a warning rather than failing the call, but servers should send logs to stderr. stderr is captured to the isanagent log (`debug`), and its last lines are attached to errors when a server exits.

### Large servers: filters and deferred tools

Every tool schema is sent with every model request. For a server with a hundred tools that is tens of thousands of tokens per turn, so isanagent offers two controls:

- **Filters** — `includeTools` / `excludeTools` (original tool names, trailing `*` allowed) drop tools a plugin does not need, for example tools that duplicate isanagent's own sandboxed file and code-execution tools.
- **Deferred tools** — a deferred tool is registered but its schema stays out of requests until the session loads it. The `search_tools` description lists deferred tools by name; a search loads the tools it returns (`select:a,b` loads exact names), and `load_skill_instructions` loads the tools named by the skill's `allowed-tools` frontmatter. Tools named by an always-on skill are never deferred. By default (`[harness.mcp] defer_tools = "auto"`) all MCP tools are deferred once their schemas exceed about 10k tokens (`defer_threshold_chars`, default 40000). Deferral needs `search_tools` (builtin tools enabled).

Per-server isanagent extensions in `mcp.json` (other clients ignore them):

```json
{
  "mcpServers": {
    "chem": {
      "command": "uv",
      "args": ["run", "--project", "${PLUGIN_DIR}", "chem-mcp"],
      "env": { "NCBI_API_KEY": "${NCBI_API_KEY}" },
      "startupTimeoutSecs": 600,
      "toolTimeoutSecs": 120,
      "excludeTools": ["read_file", "write_file", "list_files", "python_interpreter"],
      "deferTools": true
    }
  }
}
```

A skill that should bring its tools along:

```markdown
---
name: rdkit-descriptors
description: Compute molecular descriptors with RDKit.
allowed-tools: chem__rdkit_*
---
```

The user can override any of this per server in `config.toml` (keys are the `mcpServers` names):

```toml
[harness.mcp]
startup_timeout_secs = 60
tool_timeout_secs = 300
env_passthrough = ["NCBI_API_KEY"]
defer_tools = "auto"            # auto | always | never

[harness.mcp.servers.chem]
enabled = true
startup_timeout_secs = 900
exclude_tools = ["kegg_*"]      # added to the server's excludeTools
include_tools = ["rdkit_*", "stats_*"]
defer_tools = false
env_passthrough = ["CHEMSPIDER_API_KEY"]
```

### MCP Apps

If a tool sets `_meta.ui.resourceUri` to a `ui://` HTML resource (`text/html;profile=mcp-app`), the embedded web UI renders that resource in a sandboxed iframe after the tool runs. The model still receives only the tool's text result. The iframe can call tools on the same server and can send a follow-up chat message. Tools whose `_meta.ui.visibility` does not include `model` are hidden from the LLM and remain callable from the widget.

---

## 5a. Lifecycle Hooks (`hooks.json`)

Plugins can ship steering hooks in `dev.altai.isanagent/hooks.json`, in Claude Code format (`PreToolUse`, `PostToolUse`, `UserPromptSubmit` with `{"type": "command"}` entries) or isanagent format (`pre_tool`, `post_tool`, `user_prompt`, `post_model`). They run only when the operator enables steering and allow-lists the plugin:

```toml
[harness.hooks.steering]
enabled = true
plugin_hooks = ["my-plugin"]   # or ["*"]
```

Commands find plugin files through `$ISANAGENT_PLUGIN_ROOT` / `$CLAUDE_PLUGIN_ROOT`. See [`hooks.md`](./hooks.md#plugin-hooks-hooksjson) for the full contract.

---

## 6. Hierarchical Discovery

`isanagent` discovers Agent Plugins automatically at startup:

1. **User Global Root**: `~/.agent-plugins/` (the canonical Agent Plugins directory) and `~/.isanagent/plugins/`.
2. **Workspace Roots**:
   - `<workspace>/.agents/plugins/`
   - `<workspace>/.isanagent/plugins/`
   - `<workspace>/plugins/`

Workspace plugins cleanly override global plugins when names collide.

---

## 7. CLI Management Commands

Manage plugins using the `isanagent plugin` CLI (with `isanagent pack` maintained as an alias):

### Install a Plugin
```powershell
# Install from Git repository into workspace (.agents/plugins/<name>)
isanagent plugin install https://github.com/altaidevorg/plugin-ml-engineer

# Install globally to ~/.agent-plugins
isanagent plugin install https://github.com/altaidevorg/plugin-ml-engineer --global

# Install with custom local name
isanagent plugin install altaidevorg/plugin-ml-engineer --name ml-tools

# Install from a local directory, e.g. a plugin you are developing
isanagent plugin install ./my-plugin
isanagent plugin install ~/src/plugin-ml-engineer --global
```

A local directory is copied, not linked: edit the source and install again (after `isanagent plugin remove <name>`) to pick up changes. The copy is validated before it is moved into place, like a Git install. If a local directory exists with the same name as an `owner/repo` shorthand, the directory wins; use a full URL to install from GitHub instead.

### List Installed Plugins
```powershell
isanagent plugin list
```
*Output:*
```text
Installed Agent Plugins (2):
  • ml-engineer (v1.0.0) - Autonomous ML engineering, GPU benchmarking, and dataset analysis plugin
    └─ Agents: .agents/plugins/ml-engineer/dev.altai.isanagent/agents
    └─ Skills: .agents/plugins/ml-engineer/skills
    └─ MCP: .agents/plugins/ml-engineer/mcp.json
  • web-scout (v0.2.1) - Web extraction and PDF parsing tools
    └─ Skills: .agents/plugins/web-scout/skills
```

### Remove a Plugin
```powershell
# Remove from workspace
isanagent plugin remove ml-engineer

# Remove from global directory
isanagent plugin remove ml-engineer --global
```
