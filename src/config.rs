use serde::{Deserialize, Serialize};

/// Local stdin/stdout chat. When `enable` is omitted, defaults to `true` (legacy behavior).
#[derive(Debug, Deserialize, Serialize, Clone, Default)]
pub struct TerminalConfig {
    pub enable: Option<bool>,
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
