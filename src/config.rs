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

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct SlackConfig {
    pub enabled: Option<bool>,
    pub app_token: String,
    pub bot_token: String,
    pub reply_in_thread: Option<bool>,
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
