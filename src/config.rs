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
    pub memory: Option<MemoryConfig>,
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
