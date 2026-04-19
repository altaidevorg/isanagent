use serde::{Deserialize, Serialize};

/// Local stdin/stdout chat. When `enable` is omitted, defaults to `true` (legacy behavior).
#[derive(Debug, Deserialize, Serialize, Clone, Default)]
pub struct TerminalConfig {
    pub enable: Option<bool>,
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
}

/// Code execution harness (`execution_*` tools). Disabled unless `[harness.execution] enabled = true`.
#[derive(Debug, Deserialize, Serialize, Clone, Default)]
pub struct ExecutionHarnessConfig {
    /// Register execution tools when true (default: false).
    pub enabled: Option<bool>,
    /// Provider id: `local` (subprocess), `jupyter` (remote kernel), or `ssh` (remote exec).
    pub default_provider: Option<String>,
    /// Max combined stdout+stderr bytes per run (default 262_144).
    pub max_output_bytes: Option<usize>,
    /// Upper bound on per-run `timeout_secs` (default 300, clamped 1–86400).
    pub max_wall_secs: Option<u64>,
    /// Max concurrent sessions (default 32, clamped 1–256).
    pub max_sessions: Option<usize>,
    /// If set and non-empty, only these provider ids may be constructed (e.g. `["local"]`).
    pub allowed_providers: Option<Vec<String>>,
    /// Interpreter for `language: python` (default `python`) — local provider and `execution_env_info`.
    pub python_executable: Option<String>,
    /// Required when `default_provider = "jupyter"`.
    pub jupyter: Option<JupyterExecutionConfig>,
    /// Required when `default_provider = "ssh"`.
    pub ssh: Option<SshExecutionConfig>,
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

/// Optional harness features (see `docs/harness-implementation-plan.md`).
#[derive(Debug, Deserialize, Serialize, Clone, Default)]
pub struct HarnessConfig {
    pub git_worktree: Option<GitWorktreeConfig>,
    /// Background sub-agents, task tools, and optional plan execution (Phase 5).
    pub subagents: Option<SubagentHarnessConfig>,
    /// Local / future execution providers (`execution_*` tools). See `docs/execution-implementation-plan.md`.
    pub execution: Option<ExecutionHarnessConfig>,
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

impl AppConfig {
    /// Whether the stdin/stdout terminal channel is active (`[terminal].enable`, default `true`).
    pub fn terminal_enabled(&self) -> bool {
        self.terminal
            .as_ref()
            .and_then(|t| t.enable)
            .unwrap_or(true)
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

    /// `[harness.execution] enabled = true` registers `execution_*` tools.
    pub fn execution_harness_enabled(&self) -> bool {
        self.harness
            .as_ref()
            .and_then(|h| h.execution.as_ref())
            .and_then(|e| e.enabled)
            .unwrap_or(false)
    }

    pub fn execution_default_provider(&self) -> String {
        self.harness
            .as_ref()
            .and_then(|h| h.execution.as_ref())
            .and_then(|e| e.default_provider.as_ref())
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
            .unwrap_or_else(|| "local".to_string())
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
        const DEFAULT: u64 = 300;
        const MIN: u64 = 1;
        const MAX: u64 = 86400;
        self.harness
            .as_ref()
            .and_then(|h| h.execution.as_ref())
            .and_then(|e| e.max_wall_secs)
            .unwrap_or(DEFAULT)
            .clamp(MIN, MAX)
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
        self.harness
            .as_ref()
            .and_then(|h| h.execution.as_ref())
            .and_then(|e| e.python_executable.as_ref())
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
            .unwrap_or_else(|| "python".to_string())
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
max_wall_secs = 90
max_output_bytes = 8192
allowed_providers = ["local"]
python_executable = "python3"
"#;
        let c: AppConfig = toml::from_str(s).expect("parse");
        assert!(c.execution_harness_enabled());
        assert_eq!(c.execution_max_wall_secs(), 90);
        assert_eq!(c.execution_max_output_bytes(), 8192);
        assert!(c.execution_provider_allowed("local"));
        assert!(!c.execution_provider_allowed("jupyter"));
        assert_eq!(c.execution_python_executable(), "python3");
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
}
