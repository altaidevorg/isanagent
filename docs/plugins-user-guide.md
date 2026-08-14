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
```

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
