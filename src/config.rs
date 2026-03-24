use serde::{Deserialize, Serialize};

#[derive(Debug, Deserialize, Serialize, Default, Clone)]
pub struct AppConfig {
    pub restrict_to_workspace: Option<bool>,
    pub provider: Option<ProviderConfig>,
    pub api: Option<ApiConfig>,
    pub slack: Option<SlackConfig>,
    pub email: Option<EmailConfig>,
    pub max_iterations: Option<usize>,
    pub max_tool_output_chars: Option<usize>,
    /// Max characters returned by `web_search` / `web_fetch` (default 50_000). Separate from
    /// `max_tool_output_chars`, which caps tool output when passed to the model.
    pub max_web_tool_output_chars: Option<usize>,
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
}

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct ApiConfig {
    pub enabled: Option<bool>,
    pub port: u16,
}

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct ProviderConfig {
    pub model_name: String,
    pub api_key_env: String,
    pub base_url: String,
}

#[derive(Debug, Deserialize, Serialize, Clone, Copy, PartialEq, Eq, Default)]
#[serde(rename_all = "lowercase")]
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
