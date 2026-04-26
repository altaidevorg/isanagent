use serde::{Deserialize, Serialize};

/// Local stdin/stdout chat. When `enabled` is omitted, defaults to `true`.
///
/// `max_iterations` / `max_tool_output_chars` here are used only when the root-level keys are
/// unset (see [`AppConfig::resolved_max_iterations`]) — prefer root keys above `[terminal]` in TOML.
#[derive(Debug, Deserialize, Serialize, Clone, Default)]
pub struct TerminalConfig {
    pub enabled: Option<bool>,
    pub max_iterations: Option<usize>,
    pub max_tool_output_chars: Option<usize>,
}

/// OpenSSH-style remote exec over TCP (`default_provider = "ssh"`).
#[derive(Debug, Deserialize, Serialize, Clone, Default)]
pub struct SshExecutionConfig {
    /// Remote hostname or IP (required for `ssh`).
    pub host: Option<String>,
    /// TCP port (default 22).
    pub port: Option<u16>,
    /// Remote login name (required for `ssh`).
    pub user: Option<String>,
    /// Path to a private key file (OpenSSH PEM). Tilde expansion applied. Optional if
    /// **`SSH_PASSWORD`** is set in the environment.
    pub identity_file: Option<String>,
    /// Absolute path on the **remote** host used as `cd` before running code (required).
    pub remote_workdir: Option<String>,
    /// Remote Python interpreter for `language: python` (default `python3`).
    pub remote_python: Option<String>,
    /// When true (default), `check_server_key` accepts any host key (**MITM risk**). When false,
    /// host key verification fails until strict known-hosts support exists.
    pub accept_unknown_host_keys: Option<bool>,
}

/// Jupyter Server / Lab HTTP + kernel WebSocket (`default_provider = "jupyter"`).
#[derive(Debug, Deserialize, Serialize, Clone, Default)]
pub struct JupyterExecutionConfig {
    /// e.g. `http://127.0.0.1:8888` (no trailing path).
    pub base_url: Option<String>,
    /// Optional token (prefer env `JUPYTER_TOKEN` in production; avoid committing secrets).
    pub token: Option<String>,
    /// Kernel spec name for `POST /api/kernels` (default `python3`).
    pub kernel_name: Option<String>,
    /// Optional server-side notebook path template for Contents API sync (e.g. `isanagent/{session_id}.ipynb`).
    /// `{session_id}` is replaced with the **sanitized** isanagent session id. Each `execution_run` appends a code cell.
    pub notebook_sync_path_template: Option<String>,
}

/// Google Colab MCP bridge over stdio (`default_provider = "colab_mcp"`).
#[derive(Debug, Deserialize, Serialize, Clone, Default)]
pub struct ColabMcpExecutionConfig {
    /// Command to launch the MCP server (default `uvx`).
    pub command: Option<String>,
    /// Command args (default `["git+https://github.com/googlecolab/colab-mcp"]`).
    pub args: Option<Vec<String>>,
    /// Optional working directory for the MCP process.
    pub cwd: Option<String>,
    /// Session-start timeout for MCP init + tools/list handshake (default 30, clamped 5–300).
    pub startup_timeout_secs: Option<u64>,
    /// Tool used to trigger browser connection (default `open_colab_browser_connection`).
    pub connect_tool_name: Option<String>,
    /// Optional explicit execution tool name; when unset, auto-detected from tools/list.
    pub execute_tool_name: Option<String>,
    /// Preferred code argument keys to try in order (default `["code","source","cell","input"]`).
    pub execute_code_arg_keys: Option<Vec<String>>,
    /// When true, register agent tool `colab_mcp_tool_call` for allowlisted MCP tools (default true).
    pub extra_mcp_tool_call_enabled: Option<bool>,
    /// Glob patterns matched against MCP tool names (e.g. `mount_*`). Default `["*"]`.
    pub extra_mcp_tool_allowlist: Option<Vec<String>>,
}

/// Code execution harness (`execution_*` tools). On by default; set `[harness.execution] enabled = false` to disable.
#[derive(Debug, Deserialize, Serialize, Clone, Default)]
pub struct ExecutionHarnessConfig {
    /// When `Some(false)`, execution tools are not registered. Omitted or `Some(true)` keeps them on.
    pub enabled: Option<bool>,
    /// Provider id: `local` (subprocess), `jupyter` (remote kernel), or `ssh` (remote exec).
    pub default_provider: Option<String>,
    /// Max combined stdout+stderr bytes per run (default 262_144).
    pub max_output_bytes: Option<usize>,
    /// Upper bound on per-run `timeout_secs` (default 3600, clamped 1–86400).
    pub max_wall_secs: Option<u64>,
    /// Default `timeout_secs` when `execution_run` / `execution_run_background` omit it (default 600 when unset, clamped to `max_wall_secs`).
    pub default_execution_timeout_secs: Option<u64>,
    /// Short bound after which a synchronous run (`execution_run`, `colab_mcp_tool_call`)
    /// auto-promotes to a background job and returns a `job_id` envelope.
    /// Default 120 when unset; clamped to `5..=max_wall_secs`. Set to `0` to disable
    /// auto-promotion (synchronous calls then run up to their full `timeout_secs`).
    pub auto_promote_after_secs: Option<u64>,
    /// Max concurrent sessions (default 32, clamped 1–256).
    pub max_sessions: Option<usize>,
    /// If set and non-empty, only these provider ids may be constructed (e.g. `["local"]`).
    pub allowed_providers: Option<Vec<String>>,
    /// Interpreter for `language: python` (default `python`) — local provider and `execution_env_info`.
    pub python_executable: Option<String>,
    /// Local Python only: **`repl`** (default) keeps one interpreter per session so variables survive
    /// across `execution_run` calls; **`subprocess`** spawns a fresh `python -u -` per run (legacy).
    pub local_python_mode: Option<String>,
    /// Local Python runtime backend: `uv_managed` (default) or `system` (`uv`, `uv-managed` aliases accepted).
    pub local_python_runtime: Option<String>,
    /// Binary used when `local_python_runtime = "uv_managed"` (default `uv`).
    pub uv_binary: Option<String>,
    /// Python version request for `uv venv --python` (default `3.11`).
    pub uv_python: Option<String>,
    /// Optional package specs installed into the managed env (`uv pip install ...`) on first creation.
    pub uv_requirements: Option<Vec<String>>,
    /// Max bytes per execution artifact file (default 4MiB, clamped 64KiB–64MiB).
    pub artifact_max_file_bytes: Option<usize>,
    /// Max total bytes for all artifacts in one `execution_run` (default 32MiB).
    pub artifact_max_total_bytes_per_run: Option<usize>,
    /// Max artifact files per run (default 64, clamped 1–256).
    pub artifact_max_files_per_run: Option<usize>,
    /// Required when `default_provider = "jupyter"`.
    pub jupyter: Option<JupyterExecutionConfig>,
    /// Required when `default_provider = "ssh"`.
    pub ssh: Option<SshExecutionConfig>,
    /// Required when `default_provider = "colab_mcp"`.
    pub colab_mcp: Option<ColabMcpExecutionConfig>,
}

/// Sub-agent / task harness. Disabled unless `[harness.subagents] enabled = true`.
#[derive(Debug, Deserialize, Serialize, Clone, Default)]
pub struct SubagentHarnessConfig {
    pub enabled: Option<bool>,
    /// When true (default), cancelling the parent chat's reasoning (new inbound or `Cancel`) also
    /// cancels in-flight sub-agent tasks that were spawned from that chat.
    pub cancel_children_on_parent_cancel: Option<bool>,
    /// If set and non-empty, sub-agents may only call these tool names (main chat is unaffected).
    pub allowed_tools: Option<Vec<String>>,
    /// Max concurrent tasks per process (default 32, clamped 1–256).
    pub max_tasks: Option<usize>,
    /// Max seconds `subagent_spawn` may block when `wait` is true (default 300, clamped 10–3600).
    pub max_wait_secs: Option<u64>,
}

/// HF ml-intern–style ML policy overlay + optional autonomy hints (see `assets/ml_engineer_overlay.md`).
#[derive(Debug, Deserialize, Serialize, Clone, Default)]
pub struct MlEngineerHarnessConfig {
    /// Append ML engineer policy to the compiled system prompt (default: false).
    pub enabled: Option<bool>,
    /// Append research-oriented instructions to **sub-agent** system prompts when `enabled` (default: true when enabled).
    pub subagent_research_overlay: Option<bool>,
    /// When true, if an inbound message sets no metadata override, autonomous sessions may still use config default (see inbound metadata `isanagent_autonomous_forbid_final_without_tools`).
    pub forbid_final_without_tools: Option<bool>,
}

/// Shell safety policy for `exec` tool decisions before command execution.
#[derive(Debug, Deserialize, Serialize, Clone, Default)]
pub struct ShellPolicyConfig {
    /// Interactive default mode: `ask`, `deny`, or `allow` (default `ask`).
    pub mode: Option<String>,
    /// Unattended/autonomous mode: `ask`, `deny`, or `allow` (default `deny`).
    pub unattended_default: Option<String>,
    /// Extra lowercase substrings that should require approval in `ask` mode.
    pub interactive_requires_approval_for: Option<Vec<String>>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ShellPolicyMode {
    Ask,
    Deny,
    Allow,
}

#[derive(Debug, Clone)]
pub struct ResolvedShellPolicy {
    pub interactive_mode: ShellPolicyMode,
    pub unattended_mode: ShellPolicyMode,
    pub approval_patterns: Vec<String>,
}

/// Optional harness features (see `docs/harness-implementation-plan.md`).
#[derive(Debug, Deserialize, Serialize, Clone, Default)]
pub struct HarnessConfig {
    pub git_worktree: Option<GitWorktreeConfig>,
    /// Background sub-agents, task tools, and optional plan execution (Phase 5).
    pub subagents: Option<SubagentHarnessConfig>,
    /// Shell command policy (`exec`), including approval-vs-deny behavior.
    pub shell_policy: Option<ShellPolicyConfig>,
    /// Local / future execution providers (`execution_*` tools). See `docs/execution-implementation-plan.md`.
    pub execution: Option<ExecutionHarnessConfig>,
    /// ML engineer prompt overlay and related defaults.
    pub ml_engineer: Option<MlEngineerHarnessConfig>,
}

/// Git worktree helpers (`git_worktree` tool). Disabled unless `[harness.git_worktree] enabled = true`.
#[derive(Debug, Deserialize, Serialize, Clone, Default)]
pub struct GitWorktreeConfig {
    /// Register the `git_worktree` tool when true (default: false).
    pub enabled: Option<bool>,
    /// When false (default), worktree paths must satisfy the same sandbox boundary as other tools
    /// whenever `restrict_to_workspace` is true. When true, worktree paths may resolve outside the
    /// sandbox (e.g. a host temp directory) after canonicalization.
    pub allow_path_outside_sandbox: Option<bool>,
}

#[derive(Debug, Deserialize, Serialize, Default, Clone)]
pub struct AppConfig {
    pub restrict_to_workspace: Option<bool>,
    pub provider: Option<ProviderConfig>,
    pub api: Option<ApiConfig>,
    pub slack: Option<SlackConfig>,
    pub email: Option<EmailConfig>,
    pub terminal: Option<TerminalConfig>,
    pub max_iterations: Option<usize>,
    /// When true (default), detect repeated identical tool calls and inject a corrective user message.
    pub doom_loop_enabled: Option<bool>,
    pub max_tool_output_chars: Option<usize>,
    /// Max characters returned by `web_search` / `web_fetch` (default 50_000). Separate from
    /// `max_tool_output_chars`, which caps tool output when passed to the model.
    pub max_web_tool_output_chars: Option<usize>,
    /// Wall-clock limit in seconds for the `search_text` ripgrep subprocess (default 30, clamped 1–3600).
    pub search_text_ripgrep_timeout_secs: Option<u64>,
    pub memory: Option<MemoryConfig>,
    pub multi_tenant_edge: Option<MultiTenantEdgeConfig>,
    /// When `enabled`, `web_search` / `web_fetch` use [Jina Reader](https://r.jina.ai/) and search (`s.jina.ai`).
    pub jina: Option<JinaConfig>,
    pub harness: Option<HarnessConfig>,
}

/// Optional Jina Reader / Search backend for web tools (see https://jina.ai/reader ).
#[derive(Debug, Deserialize, Serialize, Clone, Default)]
pub struct JinaConfig {
    pub enabled: Option<bool>,
    pub api_key: Option<String>,
}

/// Resolved settings when `[jina].enabled = true` (for wiring into tools).
#[derive(Clone, Debug, Default)]
pub struct JinaWebBackend {
    pub api_key: Option<String>,
}

/// Heuristics to avoid sending obvious template values as `Authorization: Bearer`.
/// Language-agnostic: rejects non-ASCII (Jina keys are ASCII), angle-bracket templates, and
/// common README placeholder tokens (ASCII substrings only).
fn jina_api_key_looks_like_placeholder(s: &str) -> bool {
    let t = s.trim();
    if t.is_empty() || t.starts_with('<') {
        return true;
    }
    if !t.is_ascii() {
        return true;
    }
    let lower = t.to_ascii_lowercase();
    if lower == "changethis" {
        return true;
    }
    ["optional", "placeholder", "replace_me", "replaceme"]
        .iter()
        .any(|pat| lower.contains(pat))
}

fn parse_shell_policy_mode(raw: Option<&str>, default_mode: ShellPolicyMode) -> ShellPolicyMode {
    let Some(value) = raw else {
        return default_mode;
    };
    match value.trim().to_ascii_lowercase().as_str() {
        "allow" => ShellPolicyMode::Allow,
        "deny" => ShellPolicyMode::Deny,
        "ask" => ShellPolicyMode::Ask,
        _ => default_mode,
    }
}

impl AppConfig {
    /// Whether the stdin/stdout terminal channel is active (`[terminal].enabled`, default `true`).
    pub fn terminal_enabled(&self) -> bool {
        self.terminal
            .as_ref()
            .and_then(|t| t.enabled)
            .unwrap_or(true)
    }

    /// Root `max_iterations`, or `[terminal].max_iterations` when root is unset (common TOML mistake).
    pub fn resolved_max_iterations(&self) -> Option<usize> {
        self.max_iterations
            .or_else(|| self.terminal.as_ref().and_then(|t| t.max_iterations))
    }

    /// Root `max_tool_output_chars`, or `[terminal].max_tool_output_chars` when root is unset.
    pub fn resolved_max_tool_output_chars(&self) -> Option<usize> {
        self.max_tool_output_chars
            .or_else(|| self.terminal.as_ref().and_then(|t| t.max_tool_output_chars))
    }

    /// At least one inbound channel other than terminal (API, Slack, or Email).
    pub fn has_non_terminal_inbound_channel(&self) -> bool {
        let api_on = self
            .api
            .as_ref()
            .is_some_and(|a| a.enabled.unwrap_or(false));
        let slack_on = self
            .slack
            .as_ref()
            .is_some_and(|s| s.enabled.unwrap_or(false));
        let email_on = self
            .email
            .as_ref()
            .is_some_and(|e| e.enabled.unwrap_or(false));
        api_on || slack_on || email_on
    }

    /// Returns `Some` when `[jina].enabled` is true so tools should call r.jina.ai / s.jina.ai.
    pub fn jina_web_backend(&self) -> Option<JinaWebBackend> {
        let j = self.jina.as_ref()?;
        if !j.enabled.unwrap_or(false) {
            return None;
        }
        let api_key = j
            .api_key
            .as_ref()
            .map(|s| s.trim().to_string())
            .filter(|s| !jina_api_key_looks_like_placeholder(s));
        Some(JinaWebBackend { api_key })
    }

    /// Upper bound for `web_search` / `web_fetch` response bodies.
    pub fn effective_max_web_tool_output_chars(&self) -> usize {
        const DEFAULT: usize = 50_000;
        self.max_web_tool_output_chars.unwrap_or(DEFAULT)
    }

    /// Timeout for `search_text` when using ripgrep (workspace default; per-call override in tool args).
    pub fn effective_search_text_ripgrep_timeout_secs(&self) -> u64 {
        const DEFAULT: u64 = 30;
        const MIN: u64 = 1;
        const MAX: u64 = 3600;
        self.search_text_ripgrep_timeout_secs
            .unwrap_or(DEFAULT)
            .clamp(MIN, MAX)
    }

    /// ml-intern-style doom loop detection before each LLM call (default: enabled).
    pub fn doom_loop_enabled(&self) -> bool {
        self.doom_loop_enabled.unwrap_or(true)
    }

    /// When true, `git_worktree` is registered (see `[harness.git_worktree]` in config).
    pub fn git_worktree_tool_enabled(&self) -> bool {
        self.harness
            .as_ref()
            .and_then(|h| h.git_worktree.as_ref())
            .and_then(|g| g.enabled)
            .unwrap_or(false)
    }

    /// When true with `git_worktree_tool_enabled`, worktree paths may lie outside the sandbox.
    pub fn git_worktree_allow_path_outside_sandbox(&self) -> bool {
        self.harness
            .as_ref()
            .and_then(|h| h.git_worktree.as_ref())
            .and_then(|g| g.allow_path_outside_sandbox)
            .unwrap_or(false)
    }

    /// `[harness.subagents] enabled = true` registers task / spawn / plan tools.
    pub fn subagent_harness_enabled(&self) -> bool {
        self.harness
            .as_ref()
            .and_then(|h| h.subagents.as_ref())
            .and_then(|s| s.enabled)
            .unwrap_or(false)
    }

    /// Default true: parent chat cancel also cancels that chat's sub-agent tasks.
    pub fn subagent_cancel_children_on_parent_cancel(&self) -> bool {
        self.harness
            .as_ref()
            .and_then(|h| h.subagents.as_ref())
            .and_then(|s| s.cancel_children_on_parent_cancel)
            .unwrap_or(true)
    }

    pub fn subagent_allowed_tools_set(&self) -> Option<std::collections::HashSet<String>> {
        let v = self
            .harness
            .as_ref()
            .and_then(|h| h.subagents.as_ref())
            .and_then(|s| s.allowed_tools.as_ref())?;
        if v.is_empty() {
            return None;
        }
        Some(v.iter().cloned().collect())
    }

    pub fn subagent_max_tasks(&self) -> usize {
        const DEFAULT: usize = 32;
        const MIN: usize = 1;
        const MAX: usize = 256;
        self.harness
            .as_ref()
            .and_then(|h| h.subagents.as_ref())
            .and_then(|s| s.max_tasks)
            .unwrap_or(DEFAULT)
            .clamp(MIN, MAX)
    }

    pub fn subagent_max_wait_secs(&self) -> u64 {
        const DEFAULT: u64 = 300;
        const MIN: u64 = 10;
        const MAX: u64 = 3600;
        self.harness
            .as_ref()
            .and_then(|h| h.subagents.as_ref())
            .and_then(|s| s.max_wait_secs)
            .unwrap_or(DEFAULT)
            .clamp(MIN, MAX)
    }

    /// When false under `[harness.execution]`, execution tools are not registered. Otherwise on (including when the table is omitted).
    pub fn execution_harness_enabled(&self) -> bool {
        match self.harness.as_ref().and_then(|h| h.execution.as_ref()) {
            None => true,
            Some(e) => e.enabled.unwrap_or(true),
        }
    }

    /// `[harness.ml_engineer] enabled = true` appends ML policy overlay to the system prompt.
    pub fn ml_engineer_harness_enabled(&self) -> bool {
        self.harness
            .as_ref()
            .and_then(|h| h.ml_engineer.as_ref())
            .and_then(|m| m.enabled)
            .unwrap_or(false)
    }

    /// When ML harness is on, append research instructions to sub-agent system prompts (default true).
    pub fn ml_engineer_subagent_research_overlay(&self) -> bool {
        if !self.ml_engineer_harness_enabled() {
            return false;
        }
        self.harness
            .as_ref()
            .and_then(|h| h.ml_engineer.as_ref())
            .and_then(|m| m.subagent_research_overlay)
            .unwrap_or(true)
    }

    /// Config default for forbidding a final assistant message with no tool calls (overridable per inbound metadata).
    pub fn ml_engineer_forbid_final_without_tools(&self) -> bool {
        self.harness
            .as_ref()
            .and_then(|h| h.ml_engineer.as_ref())
            .and_then(|m| m.forbid_final_without_tools)
            .unwrap_or(false)
    }

    pub fn resolved_shell_policy(&self) -> ResolvedShellPolicy {
        let shell_cfg = self.harness.as_ref().and_then(|h| h.shell_policy.as_ref());
        let interactive_mode = parse_shell_policy_mode(
            shell_cfg.and_then(|s| s.mode.as_deref()),
            ShellPolicyMode::Ask,
        );
        let unattended_mode = parse_shell_policy_mode(
            shell_cfg.and_then(|s| s.unattended_default.as_deref()),
            ShellPolicyMode::Deny,
        );
        let mut approval_patterns = vec![
            "rm -rf".to_string(),
            "rm -fr".to_string(),
            "del /f".to_string(),
            "del /q".to_string(),
            "rmdir /s".to_string(),
            "git clean -fd".to_string(),
            "git reset --hard".to_string(),
        ];
        if let Some(extra) = shell_cfg
            .and_then(|s| s.interactive_requires_approval_for.as_ref())
            .filter(|v| !v.is_empty())
        {
            approval_patterns.extend(
                extra
                    .iter()
                    .map(|s| s.trim().to_ascii_lowercase())
                    .filter(|s| !s.is_empty()),
            );
        }
        approval_patterns.sort();
        approval_patterns.dedup();
        ResolvedShellPolicy {
            interactive_mode,
            unattended_mode,
            approval_patterns,
        }
    }

    /// Short lines for `[RUNTIME CONTEXT]` (token-frugal). Built in the binary and passed into the agent.
    pub fn runtime_harness_summary_lines(&self) -> Vec<String> {
        let mut lines = Vec::new();
        let shell_policy = self.resolved_shell_policy();
        let os_family = std::env::consts::OS;
        let shell_family = if cfg!(windows) {
            "powershell_or_cmd"
        } else {
            "sh_or_bash"
        };
        lines.push(format!(
            "host_os={} shell_family={} path_separator={}",
            os_family,
            shell_family,
            std::path::MAIN_SEPARATOR
        ));
        lines.push(format!(
            "execution_harness_enabled={}",
            self.execution_harness_enabled()
        ));
        if self.execution_harness_enabled() {
            lines.push(format!(
                "execution_default_provider={}",
                self.execution_default_provider()
            ));
            lines.push(format!(
                "execution_max_wall_secs={}",
                self.execution_max_wall_secs()
            ));
            lines.push(format!(
                "execution_default_run_timeout_secs={}",
                self.execution_default_run_timeout_secs()
            ));
            lines.push(format!(
                "execution_auto_promote_after_secs={}",
                self.execution_auto_promote_after_secs()
            ));
            lines.push(format!(
                "execution_max_output_bytes={}",
                self.execution_max_output_bytes()
            ));
            lines.push(format!(
                "execution_artifact_caps=file:{} total_per_run:{} max_files:{}",
                self.execution_artifact_max_file_bytes(),
                self.execution_artifact_max_total_bytes_per_run(),
                self.execution_artifact_max_files_per_run()
            ));
        }
        lines.push(format!(
            "subagent_harness_enabled={}",
            self.subagent_harness_enabled()
        ));
        if self.subagent_harness_enabled() {
            lines.push(format!("subagent_max_tasks={}", self.subagent_max_tasks()));
            lines.push(format!(
                "subagent_max_wait_secs={}",
                self.subagent_max_wait_secs()
            ));
            let allow = self.subagent_allowed_tools_set();
            lines.push(format!(
                "subagent_allowlist_active={} (count={})",
                allow.is_some(),
                allow.map(|s| s.len()).unwrap_or(0)
            ));
        }
        lines.push(format!(
            "ml_engineer_harness_enabled={}",
            self.ml_engineer_harness_enabled()
        ));
        lines.push(format!(
            "ml_engineer_forbid_final_without_tools_default={}",
            self.ml_engineer_forbid_final_without_tools()
        ));
        lines.push(format!(
            "shell_policy_mode_interactive={:?} shell_policy_mode_unattended={:?} shell_policy_approval_patterns={}",
            shell_policy.interactive_mode,
            shell_policy.unattended_mode,
            shell_policy.approval_patterns.len()
        ));
        lines
    }

    pub fn execution_default_provider(&self) -> String {
        self.harness
            .as_ref()
            .and_then(|h| h.execution.as_ref())
            .and_then(|e| e.default_provider.as_ref())
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
            .unwrap_or_else(|| "colab_mcp".to_string())
    }

    /// True iff the user explicitly set `[harness.execution].default_provider` to a non-empty
    /// string. Used by `build_execution_harness` to decide between auto-pick (implicit fallback
    /// happened to be misconfigured) and hard-fail (user pinned a now-pruned provider).
    pub fn execution_default_provider_explicit(&self) -> bool {
        self.harness
            .as_ref()
            .and_then(|h| h.execution.as_ref())
            .and_then(|e| e.default_provider.as_ref())
            .map(|s| !s.trim().is_empty())
            .unwrap_or(false)
    }

    /// Returns the user-configured allowed-providers list (trimmed, non-empty entries).
    /// `None` means "no restriction" (all implemented providers may be tried).
    pub fn execution_allowed_providers(&self) -> Option<Vec<String>> {
        let raw = self
            .harness
            .as_ref()
            .and_then(|h| h.execution.as_ref())
            .and_then(|e| e.allowed_providers.as_ref())?;
        let cleaned: Vec<String> = raw
            .iter()
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
            .collect();
        if cleaned.is_empty() {
            None
        } else {
            Some(cleaned)
        }
    }

    pub fn execution_max_output_bytes(&self) -> usize {
        const DEFAULT: usize = 256 * 1024;
        const MIN: usize = 4096;
        const MAX: usize = 16 * 1024 * 1024;
        self.harness
            .as_ref()
            .and_then(|h| h.execution.as_ref())
            .and_then(|e| e.max_output_bytes)
            .unwrap_or(DEFAULT)
            .clamp(MIN, MAX)
    }

    pub fn execution_max_wall_secs(&self) -> u64 {
        const DEFAULT: u64 = 3600;
        const MIN: u64 = 1;
        const MAX: u64 = 86400;
        self.harness
            .as_ref()
            .and_then(|h| h.execution.as_ref())
            .and_then(|e| e.max_wall_secs)
            .unwrap_or(DEFAULT)
            .clamp(MIN, MAX)
    }

    /// Default per-run wall clock when the model omits `timeout_secs` (1..=max_wall_secs).
    pub fn execution_default_run_timeout_secs(&self) -> u64 {
        let cap = self.execution_max_wall_secs();
        const FALLBACK: u64 = 600;
        let v = self
            .harness
            .as_ref()
            .and_then(|h| h.execution.as_ref())
            .and_then(|e| e.default_execution_timeout_secs)
            .unwrap_or(FALLBACK);
        v.clamp(1, cap)
    }

    /// Short bound after which a synchronous run auto-promotes to a background job
    /// (`execution_run`, `colab_mcp_tool_call`).
    ///
    /// Returns `0` when auto-promote is explicitly disabled (`auto_promote_after_secs = 0`).
    /// Otherwise returns the configured value clamped to `5..=max_wall_secs`, defaulting to
    /// `min(120, max_wall_secs)` when unset.
    pub fn execution_auto_promote_after_secs(&self) -> u64 {
        const FALLBACK: u64 = 120;
        const MIN_NONZERO: u64 = 5;
        let cap = self.execution_max_wall_secs();
        let raw = self
            .harness
            .as_ref()
            .and_then(|h| h.execution.as_ref())
            .and_then(|e| e.auto_promote_after_secs);
        match raw {
            Some(0) => 0,
            Some(v) => v.clamp(MIN_NONZERO, cap),
            None => FALLBACK.min(cap).max(MIN_NONZERO.min(cap)),
        }
    }

    pub fn execution_max_sessions(&self) -> usize {
        const DEFAULT: usize = 32;
        const MIN: usize = 1;
        const MAX: usize = 256;
        self.harness
            .as_ref()
            .and_then(|h| h.execution.as_ref())
            .and_then(|e| e.max_sessions)
            .unwrap_or(DEFAULT)
            .clamp(MIN, MAX)
    }

    pub fn execution_python_executable(&self) -> String {
        self.execution_python_executable_configured()
            .unwrap_or_else(|| "python".to_string())
    }

    /// Raw configured python executable (trimmed), if set.
    pub fn execution_python_executable_configured(&self) -> Option<String> {
        self.harness
            .as_ref()
            .and_then(|h| h.execution.as_ref())
            .and_then(|e| e.python_executable.as_ref())
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
    }

    /// Local provider Python: persistent REPL per session (default) vs one subprocess per run.
    pub fn execution_local_python_repl_enabled(&self) -> bool {
        let raw = self
            .harness
            .as_ref()
            .and_then(|h| h.execution.as_ref())
            .and_then(|e| e.local_python_mode.as_deref())
            .map(str::trim)
            .filter(|s| !s.is_empty());
        match raw {
            None => true,
            Some(s) => {
                let lower = s.to_ascii_lowercase();
                !matches!(
                    lower.as_str(),
                    "subprocess"
                        | "fresh"
                        | "stateless"
                        | "one_shot"
                        | "oneshot"
                        | "no"
                        | "false"
                        | "0"
                )
            }
        }
    }

    /// Local provider Python runtime backend.
    pub fn execution_local_python_runtime(&self) -> String {
        self.harness
            .as_ref()
            .and_then(|h| h.execution.as_ref())
            .and_then(|e| e.local_python_runtime.as_ref())
            .map(|s| s.trim().to_ascii_lowercase().replace('-', "_"))
            .filter(|s| !s.is_empty())
            .unwrap_or_else(|| "uv_managed".to_string())
    }

    /// Binary for UV-managed local Python runtime.
    pub fn execution_uv_binary(&self) -> String {
        self.harness
            .as_ref()
            .and_then(|h| h.execution.as_ref())
            .and_then(|e| e.uv_binary.as_ref())
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
            .unwrap_or_else(|| "uv".to_string())
    }

    /// Requested Python version for UV-managed local Python runtime.
    pub fn execution_uv_python(&self) -> String {
        self.harness
            .as_ref()
            .and_then(|h| h.execution.as_ref())
            .and_then(|e| e.uv_python.as_ref())
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
            .unwrap_or_else(|| "3.11".to_string())
    }

    /// Optional package specs installed once for UV-managed local runtime.
    pub fn execution_uv_requirements(&self) -> Vec<String> {
        self.harness
            .as_ref()
            .and_then(|h| h.execution.as_ref())
            .and_then(|e| e.uv_requirements.as_ref())
            .map(|items| {
                items
                    .iter()
                    .map(|s| s.trim().to_string())
                    .filter(|s| !s.is_empty())
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default()
    }

    /// Caps for Phase 6 execution artifacts (Jupyter `display_data` materialization).
    pub fn execution_artifact_max_file_bytes(&self) -> usize {
        const DEFAULT: usize = 4 * 1024 * 1024;
        const MIN: usize = 64 * 1024;
        const MAX: usize = 64 * 1024 * 1024;
        self.harness
            .as_ref()
            .and_then(|h| h.execution.as_ref())
            .and_then(|e| e.artifact_max_file_bytes)
            .unwrap_or(DEFAULT)
            .clamp(MIN, MAX)
    }

    pub fn execution_artifact_max_total_bytes_per_run(&self) -> usize {
        const DEFAULT: usize = 32 * 1024 * 1024;
        const MIN: usize = 256 * 1024;
        const MAX: usize = 128 * 1024 * 1024;
        self.harness
            .as_ref()
            .and_then(|h| h.execution.as_ref())
            .and_then(|e| e.artifact_max_total_bytes_per_run)
            .unwrap_or(DEFAULT)
            .clamp(MIN, MAX)
    }

    pub fn execution_artifact_max_files_per_run(&self) -> usize {
        const DEFAULT: usize = 64;
        const MIN: usize = 1;
        const MAX: usize = 256;
        self.harness
            .as_ref()
            .and_then(|h| h.execution.as_ref())
            .and_then(|e| e.artifact_max_files_per_run)
            .unwrap_or(DEFAULT)
            .clamp(MIN, MAX)
    }

    /// When `allowed_providers` is missing or empty, any implemented provider id is allowed.
    pub fn execution_provider_allowed(&self, provider_id: &str) -> bool {
        let Some(list) = self
            .harness
            .as_ref()
            .and_then(|h| h.execution.as_ref())
            .and_then(|e| e.allowed_providers.as_ref())
        else {
            return true;
        };
        if list.is_empty() {
            return true;
        }
        list.iter().any(|s| s == provider_id)
    }

    /// `JUPYTER_TOKEN` env wins over `[harness.execution.jupyter].token`.
    pub fn execution_jupyter_token(&self) -> Option<String> {
        let from_env = std::env::var("JUPYTER_TOKEN")
            .ok()
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty());
        if from_env.is_some() {
            return from_env;
        }
        self.harness
            .as_ref()
            .and_then(|h| h.execution.as_ref())
            .and_then(|e| e.jupyter.as_ref())
            .and_then(|j| j.token.as_ref())
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
    }

    pub fn execution_jupyter_base_url(&self) -> Option<String> {
        self.harness
            .as_ref()
            .and_then(|h| h.execution.as_ref())
            .and_then(|e| e.jupyter.as_ref())
            .and_then(|j| j.base_url.as_ref())
            .map(|s| s.trim().trim_end_matches('/').to_string())
            .filter(|s| !s.is_empty())
    }

    pub fn execution_jupyter_kernel_name(&self) -> String {
        self.harness
            .as_ref()
            .and_then(|h| h.execution.as_ref())
            .and_then(|e| e.jupyter.as_ref())
            .and_then(|j| j.kernel_name.as_ref())
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
            .unwrap_or_else(|| "python3".to_string())
    }

    pub fn execution_jupyter_notebook_sync_path_template(&self) -> Option<String> {
        self.harness
            .as_ref()
            .and_then(|h| h.execution.as_ref())
            .and_then(|e| e.jupyter.as_ref())
            .and_then(|j| j.notebook_sync_path_template.as_ref())
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
    }

    pub fn execution_ssh_host(&self) -> Option<String> {
        self.harness
            .as_ref()
            .and_then(|h| h.execution.as_ref())
            .and_then(|e| e.ssh.as_ref())
            .and_then(|s| s.host.as_ref())
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
    }

    pub fn execution_ssh_port(&self) -> u16 {
        const DEFAULT: u16 = 22;
        self.harness
            .as_ref()
            .and_then(|h| h.execution.as_ref())
            .and_then(|e| e.ssh.as_ref())
            .and_then(|s| s.port)
            .unwrap_or(DEFAULT)
    }

    pub fn execution_ssh_user(&self) -> Option<String> {
        self.harness
            .as_ref()
            .and_then(|h| h.execution.as_ref())
            .and_then(|e| e.ssh.as_ref())
            .and_then(|s| s.user.as_ref())
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
    }

    /// Expanded filesystem path to a private key, when configured.
    pub fn execution_ssh_identity_file(&self) -> Option<String> {
        let raw = self
            .harness
            .as_ref()
            .and_then(|h| h.execution.as_ref())
            .and_then(|e| e.ssh.as_ref())
            .and_then(|s| s.identity_file.as_ref())?;
        let t = raw.trim();
        if t.is_empty() {
            return None;
        }
        let expanded = shellexpand::tilde(t).into_owned();
        if expanded.trim().is_empty() {
            return None;
        }
        Some(expanded)
    }

    pub fn execution_ssh_remote_workdir(&self) -> Option<String> {
        self.harness
            .as_ref()
            .and_then(|h| h.execution.as_ref())
            .and_then(|e| e.ssh.as_ref())
            .and_then(|s| s.remote_workdir.as_ref())
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
    }

    pub fn execution_ssh_remote_python(&self) -> String {
        self.harness
            .as_ref()
            .and_then(|h| h.execution.as_ref())
            .and_then(|e| e.ssh.as_ref())
            .and_then(|s| s.remote_python.as_ref())
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
            .unwrap_or_else(|| "python3".to_string())
    }

    pub fn execution_ssh_accept_unknown_host_keys(&self) -> bool {
        self.harness
            .as_ref()
            .and_then(|h| h.execution.as_ref())
            .and_then(|e| e.ssh.as_ref())
            .and_then(|s| s.accept_unknown_host_keys)
            .unwrap_or(true)
    }

    pub fn execution_colab_mcp_command(&self) -> String {
        self.harness
            .as_ref()
            .and_then(|h| h.execution.as_ref())
            .and_then(|e| e.colab_mcp.as_ref())
            .and_then(|c| c.command.as_ref())
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
            .unwrap_or_else(|| "uvx".to_string())
    }

    pub fn execution_colab_mcp_args(&self) -> Vec<String> {
        self.harness
            .as_ref()
            .and_then(|h| h.execution.as_ref())
            .and_then(|e| e.colab_mcp.as_ref())
            .and_then(|c| c.args.as_ref())
            .map(|items| {
                items
                    .iter()
                    .map(|s| s.trim().to_string())
                    .filter(|s| !s.is_empty())
                    .collect::<Vec<_>>()
            })
            .filter(|v| !v.is_empty())
            .unwrap_or_else(|| vec!["git+https://github.com/googlecolab/colab-mcp".to_string()])
    }

    pub fn execution_colab_mcp_cwd(&self) -> Option<String> {
        self.harness
            .as_ref()
            .and_then(|h| h.execution.as_ref())
            .and_then(|e| e.colab_mcp.as_ref())
            .and_then(|c| c.cwd.as_ref())
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
    }

    pub fn execution_colab_mcp_startup_timeout_secs(&self) -> u64 {
        const DEFAULT: u64 = 30;
        const MIN: u64 = 5;
        const MAX: u64 = 300;
        self.harness
            .as_ref()
            .and_then(|h| h.execution.as_ref())
            .and_then(|e| e.colab_mcp.as_ref())
            .and_then(|c| c.startup_timeout_secs)
            .unwrap_or(DEFAULT)
            .clamp(MIN, MAX)
    }

    pub fn execution_colab_mcp_connect_tool_name(&self) -> String {
        self.harness
            .as_ref()
            .and_then(|h| h.execution.as_ref())
            .and_then(|e| e.colab_mcp.as_ref())
            .and_then(|c| c.connect_tool_name.as_ref())
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
            .unwrap_or_else(|| "open_colab_browser_connection".to_string())
    }

    pub fn execution_colab_mcp_execute_tool_name(&self) -> Option<String> {
        self.harness
            .as_ref()
            .and_then(|h| h.execution.as_ref())
            .and_then(|e| e.colab_mcp.as_ref())
            .and_then(|c| c.execute_tool_name.as_ref())
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
    }

    pub fn execution_colab_mcp_execute_code_arg_keys(&self) -> Vec<String> {
        self.harness
            .as_ref()
            .and_then(|h| h.execution.as_ref())
            .and_then(|e| e.colab_mcp.as_ref())
            .and_then(|c| c.execute_code_arg_keys.as_ref())
            .map(|items| {
                items
                    .iter()
                    .map(|s| s.trim().to_string())
                    .filter(|s| !s.is_empty())
                    .collect::<Vec<_>>()
            })
            .filter(|v| !v.is_empty())
            .unwrap_or_else(|| {
                vec![
                    "code".to_string(),
                    "source".to_string(),
                    "cell".to_string(),
                    "input".to_string(),
                ]
            })
    }

    pub fn execution_colab_mcp_extra_mcp_tool_call_enabled(&self) -> bool {
        self.harness
            .as_ref()
            .and_then(|h| h.execution.as_ref())
            .and_then(|e| e.colab_mcp.as_ref())
            .and_then(|c| c.extra_mcp_tool_call_enabled)
            .unwrap_or(true)
    }

    pub fn execution_colab_mcp_extra_mcp_tool_allowlist(&self) -> Vec<String> {
        self.harness
            .as_ref()
            .and_then(|h| h.execution.as_ref())
            .and_then(|e| e.colab_mcp.as_ref())
            .and_then(|c| c.extra_mcp_tool_allowlist.as_ref())
            .map(|items| {
                items
                    .iter()
                    .map(|s| s.trim().to_string())
                    .filter(|s| !s.is_empty())
                    .collect::<Vec<_>>()
            })
            .filter(|v| !v.is_empty())
            .unwrap_or_else(|| vec!["*".to_string()])
    }
}

#[derive(Debug, Deserialize, Serialize, Clone, Default)]
pub struct MemoryConfig {
    pub enabled: Option<bool>,
    pub short_term_threshold_turns: Option<usize>,
    pub short_term_threshold_tokens: Option<usize>,
    pub short_term_threshold_mins: Option<u64>,
    pub long_term_interval_mins: Option<u64>,
    pub max_recent_summaries: Option<usize>,
    pub long_term_threshold_summaries: Option<usize>,
}

#[derive(Debug, Deserialize, Serialize, Clone, Default)]
pub struct MultiTenantEdgeConfig {
    pub activity_heartbeat_enabled: Option<bool>,
    pub cron_scheduling_enabled: Option<bool>,
}

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct ApiConfig {
    pub enabled: Option<bool>,
    pub port: u16,
    pub serve_ui: Option<bool>,
    pub bind_address: Option<String>,
}

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct ProviderConfig {
    pub model_name: String,
    pub api_key_env: String,
    pub base_url: String,
}

#[derive(Debug, Deserialize, Serialize, Clone, Copy, PartialEq, Eq, Default, clap::ValueEnum)]
#[serde(rename_all = "lowercase")]
#[clap(rename_all = "lower")]
pub enum SlackMode {
    #[default]
    Webhook,
    Socket,
}

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct SlackConfig {
    pub enabled: Option<bool>,
    pub mode: Option<SlackMode>,
    pub app_token: Option<String>,
    pub bot_token: String,
    pub signing_secret: Option<String>,
    pub webhook_port: Option<u16>,
    pub webhook_path: Option<String>,
    pub reply_in_thread: Option<bool>,
    pub reaction_emoji: Option<String>,
}

impl SlackConfig {
    pub fn mode(&self) -> SlackMode {
        self.mode.unwrap_or_default()
    }
}

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct EmailConfig {
    pub enabled: Option<bool>,
    pub imap_host: String,
    pub imap_port: u16,
    pub imap_username: String,
    pub imap_password: String,
    pub smtp_host: String,
    pub smtp_port: u16,
    pub email_address: String,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn harness_execution_toml_roundtrip() {
        let s = r#"
[harness.execution]
enabled = true
default_provider = "local"
max_wall_secs = 90
max_output_bytes = 8192
allowed_providers = ["local"]
python_executable = "python3"
"#;
        let c: AppConfig = toml::from_str(s).expect("parse");
        assert!(c.execution_harness_enabled());
        assert_eq!(c.execution_max_wall_secs(), 90);
        assert_eq!(c.execution_default_run_timeout_secs(), 90);
        assert_eq!(c.execution_max_output_bytes(), 8192);
        assert!(c.execution_provider_allowed("local"));
        assert!(!c.execution_provider_allowed("jupyter"));
        assert_eq!(c.execution_python_executable(), "python3");
    }

    #[test]
    fn harness_execution_on_by_default_without_harness_section() {
        let s = "restrict_to_workspace = true\n";
        let c: AppConfig = toml::from_str(s).expect("parse");
        assert!(c.execution_harness_enabled());
        assert_eq!(c.execution_default_provider(), "colab_mcp");
    }

    #[test]
    fn harness_execution_explicit_disabled() {
        let s = r#"
restrict_to_workspace = true
[harness.execution]
enabled = false
"#;
        let c: AppConfig = toml::from_str(s).expect("parse");
        assert!(!c.execution_harness_enabled());
    }

    #[test]
    fn harness_execution_defaults_when_only_enabled() {
        let s = r#"
[harness.execution]
enabled = true
"#;
        let c: AppConfig = toml::from_str(s).expect("parse");
        assert_eq!(c.execution_max_wall_secs(), 3600);
        assert_eq!(c.execution_default_run_timeout_secs(), 600);
        assert_eq!(c.execution_local_python_runtime(), "uv_managed");
        assert_eq!(c.execution_default_provider(), "colab_mcp");
    }

    #[test]
    fn harness_execution_default_run_timeout_respects_max_wall() {
        let s = r#"
[harness.execution]
enabled = true
max_wall_secs = 120
default_execution_timeout_secs = 9999
"#;
        let c: AppConfig = toml::from_str(s).expect("parse");
        assert_eq!(c.execution_default_run_timeout_secs(), 120);
        let s2 = r#"
[harness.execution]
enabled = true
max_wall_secs = 600
default_execution_timeout_secs = 45
"#;
        let c2: AppConfig = toml::from_str(s2).expect("parse");
        assert_eq!(c2.execution_default_run_timeout_secs(), 45);
    }

    #[test]
    fn harness_execution_artifact_limits_toml() {
        let s = r#"
[harness.execution]
enabled = true
artifact_max_file_bytes = 100000
artifact_max_total_bytes_per_run = 500000
artifact_max_files_per_run = 10
"#;
        let c: AppConfig = toml::from_str(s).expect("parse");
        assert_eq!(c.execution_artifact_max_file_bytes(), 100_000);
        assert_eq!(c.execution_artifact_max_total_bytes_per_run(), 500_000);
        assert_eq!(c.execution_artifact_max_files_per_run(), 10);
    }

    #[test]
    fn doom_loop_enabled_toml() {
        let s = r#"doom_loop_enabled = false"#;
        let c: AppConfig = toml::from_str(s).expect("parse");
        assert!(!c.doom_loop_enabled());
        assert!(AppConfig::default().doom_loop_enabled());
    }

    #[test]
    fn resolved_max_iterations_falls_back_to_terminal_table() {
        let s = r#"
[terminal]
enabled = true
max_iterations = 999
"#;
        let c: AppConfig = toml::from_str(s).expect("parse");
        assert_eq!(c.resolved_max_iterations(), Some(999));
    }

    #[test]
    fn resolved_max_iterations_root_wins_over_terminal() {
        let s = r#"
max_iterations = 12
[terminal]
max_iterations = 999
"#;
        let c: AppConfig = toml::from_str(s).expect("parse");
        assert_eq!(c.resolved_max_iterations(), Some(12));
    }

    #[test]
    fn harness_execution_jupyter_toml() {
        let s = r#"
[harness.execution]
enabled = true
default_provider = "jupyter"
allowed_providers = ["jupyter", "local"]

[harness.execution.jupyter]
base_url = "http://127.0.0.1:8888"
token = "testtoken"
kernel_name = "python3"
notebook_sync_path_template = "scratch/{session_id}.ipynb"
"#;
        let c: AppConfig = toml::from_str(s).expect("parse");
        assert_eq!(c.execution_default_provider(), "jupyter");
        assert!(c.execution_provider_allowed("jupyter"));
        assert_eq!(
            c.execution_jupyter_base_url().as_deref(),
            Some("http://127.0.0.1:8888")
        );
        assert_eq!(c.execution_jupyter_token().as_deref(), Some("testtoken"));
        assert_eq!(c.execution_jupyter_kernel_name(), "python3");
        assert_eq!(
            c.execution_jupyter_notebook_sync_path_template().as_deref(),
            Some("scratch/{session_id}.ipynb")
        );
    }

    #[test]
    fn harness_execution_ssh_toml() {
        let s = r#"
[harness.execution]
enabled = true
default_provider = "ssh"
allowed_providers = ["ssh", "local"]

[harness.execution.ssh]
host = "10.0.0.5"
port = 2222
user = "dev"
identity_file = "~/.ssh/id_ed25519"
remote_workdir = "/tmp/isanagent-exec"
remote_python = "python3"
accept_unknown_host_keys = false
"#;
        let c: AppConfig = toml::from_str(s).expect("parse");
        assert_eq!(c.execution_default_provider(), "ssh");
        assert!(c.execution_provider_allowed("ssh"));
        assert_eq!(c.execution_ssh_host().as_deref(), Some("10.0.0.5"));
        assert_eq!(c.execution_ssh_port(), 2222);
        assert_eq!(c.execution_ssh_user().as_deref(), Some("dev"));
        assert!(c
            .execution_ssh_identity_file()
            .expect("identity")
            .replace('\\', "/")
            .ends_with("/.ssh/id_ed25519"));
        assert_eq!(
            c.execution_ssh_remote_workdir().as_deref(),
            Some("/tmp/isanagent-exec")
        );
        assert_eq!(c.execution_ssh_remote_python(), "python3");
        assert!(!c.execution_ssh_accept_unknown_host_keys());
    }

    #[test]
    fn harness_execution_uv_local_runtime_toml() {
        let s = r#"
[harness.execution]
enabled = true
local_python_runtime = "uv_managed"
uv_binary = "uvx"
uv_python = "3.12"
uv_requirements = ["numpy", "pandas>=2.2"]
"#;
        let c: AppConfig = toml::from_str(s).expect("parse");
        assert_eq!(c.execution_local_python_runtime(), "uv_managed");
        assert_eq!(c.execution_uv_binary(), "uvx");
        assert_eq!(c.execution_uv_python(), "3.12");
        assert_eq!(
            c.execution_uv_requirements(),
            vec!["numpy".to_string(), "pandas>=2.2".to_string()]
        );
    }

    #[test]
    fn harness_execution_colab_mcp_toml() {
        let s = r#"
[harness.execution]
enabled = true
default_provider = "colab_mcp"
allowed_providers = ["colab_mcp", "local"]

[harness.execution.colab_mcp]
command = "uvx"
args = ["git+https://github.com/googlecolab/colab-mcp"]
startup_timeout_secs = 45
connect_tool_name = "open_colab_browser_connection"
execute_tool_name = "execute_python"
execute_code_arg_keys = ["code", "source"]
extra_mcp_tool_call_enabled = true
extra_mcp_tool_allowlist = ["get_*", "mount_drive"]
"#;
        let c: AppConfig = toml::from_str(s).expect("parse");
        assert_eq!(c.execution_default_provider(), "colab_mcp");
        assert!(c.execution_provider_allowed("colab_mcp"));
        assert_eq!(c.execution_colab_mcp_command(), "uvx");
        assert_eq!(
            c.execution_colab_mcp_args(),
            vec!["git+https://github.com/googlecolab/colab-mcp".to_string()]
        );
        assert_eq!(c.execution_colab_mcp_startup_timeout_secs(), 45);
        assert_eq!(
            c.execution_colab_mcp_connect_tool_name(),
            "open_colab_browser_connection"
        );
        assert_eq!(
            c.execution_colab_mcp_execute_tool_name().as_deref(),
            Some("execute_python")
        );
        assert_eq!(
            c.execution_colab_mcp_execute_code_arg_keys(),
            vec!["code".to_string(), "source".to_string()]
        );
        assert!(c.execution_colab_mcp_extra_mcp_tool_call_enabled());
        assert_eq!(
            c.execution_colab_mcp_extra_mcp_tool_allowlist(),
            vec!["get_*".to_string(), "mount_drive".to_string()]
        );
    }

    #[test]
    fn harness_execution_colab_mcp_extra_tools_defaults_enabled() {
        let s = r#"
[harness.execution]
enabled = true
default_provider = "colab_mcp"
"#;
        let c: AppConfig = toml::from_str(s).expect("parse");
        assert!(c.execution_colab_mcp_extra_mcp_tool_call_enabled());
        assert_eq!(
            c.execution_colab_mcp_extra_mcp_tool_allowlist(),
            vec!["*".to_string()]
        );
    }

    #[test]
    fn shell_policy_defaults_and_overrides() {
        let c = AppConfig::default();
        let p = c.resolved_shell_policy();
        assert_eq!(p.interactive_mode, ShellPolicyMode::Ask);
        assert_eq!(p.unattended_mode, ShellPolicyMode::Deny);
        assert!(!p.approval_patterns.is_empty());

        let s = r#"
[harness.shell_policy]
mode = "deny"
unattended_default = "allow"
interactive_requires_approval_for = ["terraform destroy"]
"#;
        let c2: AppConfig = toml::from_str(s).expect("parse");
        let p2 = c2.resolved_shell_policy();
        assert_eq!(p2.interactive_mode, ShellPolicyMode::Deny);
        assert_eq!(p2.unattended_mode, ShellPolicyMode::Allow);
        assert!(p2
            .approval_patterns
            .iter()
            .any(|s| s.contains("terraform destroy")));
    }
}
