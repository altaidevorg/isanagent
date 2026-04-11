use log::{debug, info};
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::time::Duration;

/// Suffix after the human-readable runtime context line on user messages (see `agent` injection).
/// API and previews strip through this marker so wording inside the `[RUNTIME CONTEXT]` line can evolve.
pub const RUNTIME_CONTEXT_END_SUFFIX: &str = "\n---ISANAGENT_RUNTIME_CONTEXT_END---\n\n";

/// Regex pattern for stripping `<redacted_thinking>...</redacted_thinking>` from model output.
/// Shared by the agent (outbound cleanup) and the HTTP API transcript builder.
pub const REDACTED_THINKING_STRIP_PATTERN: &str =
    r"(?s)<redacted_thinking>.*?</redacted_thinking>\s*";

// --- Multimodal Content Types ---

/// A single part within a multimodal message content array.
/// Follows the OpenAI content part schema.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ContentPart {
    /// Plain text content.
    Text { text: String },
    /// An image referenced by URL or base64 data URI.
    ImageUrl { image_url: ImageUrl },
}

/// Image reference inside a content part.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ImageUrl {
    /// An https URL or a base64 data URI (`data:<media_type>;base64,<data>`).
    pub url: String,
    /// Optional detail hint: "auto", "low", or "high".
    #[serde(skip_serializing_if = "Option::is_none")]
    pub detail: Option<String>,
}

/// The value of a `ChatMessage.content` field, which may be plain text or a
/// list of multimodal content parts (following the OpenAI spec).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(untagged)]
pub enum MessageContent {
    /// A plain-text string – used for system, assistant, and tool messages.
    Text(String),
    /// An ordered list of content parts – used for multimodal user messages.
    Parts(Vec<ContentPart>),
}

impl MessageContent {
    /// Returns the concatenated plain-text of all text parts (or the string itself).
    pub fn text_content(&self) -> String {
        match self {
            MessageContent::Text(s) => s.clone(),
            MessageContent::Parts(parts) => parts
                .iter()
                .filter_map(|p| match p {
                    ContentPart::Text { text } => Some(text.as_str()),
                    ContentPart::ImageUrl { .. } => None,
                })
                .collect::<Vec<_>>()
                .join("\n\n"),
        }
    }
}

impl std::fmt::Display for MessageContent {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.text_content())
    }
}

// --- Data Structures ---

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMessage {
    pub role: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<MessageContent>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ToolCallRequest>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCallFunction {
    pub name: String,
    pub arguments: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCallRequest {
    pub id: String,
    #[serde(rename = "type")]
    pub tool_type: String, // Usually "function"
    #[serde(skip_serializing_if = "Option::is_none")]
    pub extra_content: Option<serde_json::Value>,
    pub function: ToolCallFunction,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenUsage {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub total_tokens: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LLMResponse {
    pub content: String,
    pub tool_calls: Option<Vec<ToolCallRequest>>,
    pub reasoning_content: Option<String>,
    pub usage: Option<TokenUsage>,
}

impl ChatMessage {
    pub fn user(content: &str) -> Self {
        Self {
            role: "user".to_string(),
            content: Some(MessageContent::Text(content.to_string())),
            name: None,
            tool_calls: None,
            tool_call_id: None,
        }
    }

    /// Create a user message with both a text portion and additional multimodal attachments.
    /// If `attachments` is empty, this is equivalent to `user(text)`.
    pub fn user_multimodal(text: &str, attachments: &[ContentPart]) -> Self {
        if attachments.is_empty() {
            return Self::user(text);
        }
        let mut parts = vec![ContentPart::Text {
            text: text.to_string(),
        }];
        parts.extend_from_slice(attachments);
        Self {
            role: "user".to_string(),
            content: Some(MessageContent::Parts(parts)),
            name: None,
            tool_calls: None,
            tool_call_id: None,
        }
    }

    pub fn system(content: &str) -> Self {
        Self {
            role: "system".to_string(),
            content: Some(MessageContent::Text(content.to_string())),
            name: None,
            tool_calls: None,
            tool_call_id: None,
        }
    }

    pub fn assistant(content: &str) -> Self {
        Self {
            role: "assistant".to_string(),
            content: Some(MessageContent::Text(content.to_string())),
            name: None,
            tool_calls: None,
            tool_call_id: None,
        }
    }

    pub fn tool(content: &str, tool_call_id: &str) -> Self {
        Self {
            role: "tool".to_string(),
            content: Some(MessageContent::Text(content.to_string())),
            name: None,
            tool_calls: None,
            tool_call_id: Some(tool_call_id.to_string()),
        }
    }
}

// --- Error Handling ---

#[derive(thiserror::Error, Debug)]
pub enum LLMError {
    #[error("HTTP Request Failed: {0}")]
    RequestError(#[from] reqwest::Error),
    #[error("API Error: {0}")]
    ApiError(String),
    #[error("Parsing Error: {0}")]
    ParseError(#[from] serde_json::Error),
    #[error("No content in response")]
    NoContent,
}

// --- Client ---

#[derive(Clone)]
pub struct LLMClient {
    base_url: String,
    api_key: String,
    model: String,
    temperature: f32,
    timeout: Duration,
    client: reqwest::Client,
}

pub fn build_reqwest_client() -> reqwest::Client {
    reqwest::Client::builder()
        .no_proxy()
        .build()
        .expect("failed to build reqwest client")
}

impl LLMClient {
    /// Create a new client with default settings (temp=0.7, timeout=30s).
    pub fn new_openai_compatible(base_url: &str, api_key: &str, model: &str) -> Self {
        // Ensure base_url ends with slash if needed, or handle path joining correctly
        // For simplicity, we assume user gives enough of the path or we append standardized paths.
        // However, OpenAI-compatible APIs often vary in suffix.
        // We'll treat `base_url` as the full endpoint URL for completions for maximum flexibility,
        // OR we can default to adding "/chat/completions" if the user provided base domain.
        // Let's go with robust: User provides full endpoint or we construct.
        // Actually, simplest usage: base_url is the helper.
        Self {
            base_url: base_url.to_string(),
            api_key: api_key.to_string(),
            model: model.to_string(),
            temperature: 0.7,
            timeout: Duration::from_secs(30),
            client: build_reqwest_client(),
        }
    }

    /// Set connection timeout.
    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = timeout;
        self
    }

    /// Set generation temperature.
    pub fn with_temperature(mut self, temperature: f32) -> Self {
        self.temperature = temperature;
        self
    }

    /// Send a chat completion request with history.
    pub async fn chat(
        &self,
        messages: &[ChatMessage],
        tools: Option<serde_json::Value>,
    ) -> Result<LLMResponse, LLMError> {
        debug!(
            "Sending chat request to {} with {} messages",
            self.model,
            messages.len()
        );

        // Construct body for OpenAI-compatible
        let mut body = json!({
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature
        });

        if let Some(t) = tools {
            if let Some(obj) = body.as_object_mut() {
                obj.insert("tools".to_string(), t);
                obj.insert("tool_choice".to_string(), json!("auto"));
            }
        }

        // Normally "openai compatible" implies base_url is something like https://api.openai.com/v1
        // and we append /chat/completions.
        // But some services give you the exact endpoint.
        // We'll look for "completions" in the URL. If missing, we append /chat/completions?
        // Let's stick to the convention in the example: The URL passed IS the endpoint.
        let url = &self.base_url;

        let res = self
            .client
            .post(url)
            .header("Authorization", format!("Bearer {}", self.api_key))
            .header("Content-Type", "application/json")
            .timeout(self.timeout)
            .json(&body)
            .send()
            .await?;

        let status = res.status();
        if !status.is_success() {
            let text = res.text().await.unwrap_or_default();
            return Err(LLMError::ApiError(format!("Status {}: {}", status, text)));
        }

        let raw_text = res
            .text()
            .await
            .map_err(|e| LLMError::ApiError(e.to_string()))?;
        let json_resp: serde_json::Value =
            serde_json::from_str(&raw_text).map_err(LLMError::ParseError)?;

        let content_val = &json_resp["choices"][0]["message"]["content"];
        let content = if content_val.is_null() {
            "".to_string()
        } else {
            content_val.as_str().ok_or(LLMError::NoContent)?.to_string()
        };

        // Parse tool calls
        let tool_calls_val = &json_resp["choices"][0]["message"]["tool_calls"];
        let tool_calls = if tool_calls_val.is_null() {
            None
        } else {
            serde_json::from_value::<Vec<ToolCallRequest>>(tool_calls_val.clone()).ok()
        };

        let reasoning_content = json_resp["choices"][0]["message"]["reasoning_content"]
            .as_str()
            .map(|s| s.to_string());

        let usage = json_resp.get("usage").map(|usage_obj| TokenUsage {
            prompt_tokens: usage_obj["prompt_tokens"].as_u64().unwrap_or(0) as u32,
            completion_tokens: usage_obj["completion_tokens"].as_u64().unwrap_or(0) as u32,
            total_tokens: usage_obj["total_tokens"].as_u64().unwrap_or(0) as u32,
        });

        let finish_reason = json_resp["choices"][0]["finish_reason"]
            .as_str()
            .unwrap_or("unknown")
            .to_string();

        if usage
            .as_ref()
            .map(|u| u.completion_tokens == 0)
            .unwrap_or(false)
            || content.trim().is_empty()
        {
            debug!(
                "LLM returned empty content or zero completion tokens. finish_reason={} raw_response_len={}",
                finish_reason,
                raw_text.len()
            );
        }

        info!(
            "LLM Response received ({} chars content, tool_calls: {}, reasoning: {}, usage: {})",
            content.len(),
            tool_calls.as_ref().map(|v| v.len()).unwrap_or(0),
            reasoning_content.is_some(),
            usage.is_some()
        );

        Ok(LLMResponse {
            content,
            tool_calls,
            reasoning_content,
            usage,
        })
    }

    /// Simple one-shot prompt.
    pub async fn ask(&self, prompt: &str) -> Result<LLMResponse, LLMError> {
        let messages = vec![ChatMessage::user(prompt)];
        self.chat(&messages, None).await
    }

    /// One-shot with system instruction.
    pub async fn ask_with_system(&self, system: &str, user: &str) -> Result<LLMResponse, LLMError> {
        let messages = vec![ChatMessage::system(system), ChatMessage::user(user)];
        self.chat(&messages, None).await
    }
}

/// Resolves `agent_path` relative to `sandbox_dir`, ensuring the resulting
/// canonical path stays within the sandbox boundary.
///
/// Returns `None` when:
/// - the path escapes the sandbox via `../` traversal or absolute references
///   outside the sandbox directory, or
/// - the path does not exist on disk (canonicalization requires existence).
pub fn resolve_path(sandbox_dir: &std::path::Path, agent_path: &str) -> Option<std::path::PathBuf> {
    // Canonicalize the sandbox root first so we can fail fast before touching
    // any user-supplied path data.
    let sandbox_canonical = sandbox_dir.canonicalize().ok()?;

    let raw = std::path::Path::new(agent_path);

    // Absolute paths are only allowed when they live inside the sandbox.
    // Relative paths are joined against the sandbox root first.
    let joined = if raw.is_absolute() {
        raw.to_path_buf()
    } else {
        sandbox_canonical.join(raw)
    };

    // `canonicalize` resolves symlinks and `..` components and requires the
    // path to exist – so a non-existent file naturally returns `None`.
    let canonical = joined.canonicalize().ok()?;

    if canonical.starts_with(&sandbox_canonical) {
        Some(canonical)
    } else {
        None
    }
}

/// Robustly extracts a JSON object from a raw LLM text response.
/// Intended to handle markdown formatting (` ```json ... ``` `)
/// or conversational wrappers around the core `{ ... }` payload.
pub fn extract_json_from_llm_response(text: &str) -> Option<serde_json::Value> {
    // Attempt 1: Look for explicit markdown JSON blocks
    if let Some(start_idx) = text.find("```json") {
        let block_content = &text[start_idx + 7..];
        if let Some(end_idx) = block_content.find("```") {
            let json_candidate = &block_content[..end_idx].trim();
            if let Ok(val) = serde_json::from_str::<serde_json::Value>(json_candidate) {
                return Some(val);
            }
        }
    }

    // Attempt 2: Naive bracket matching as fallback
    if let Some(json_start) = text.find('{') {
        if let Some(json_end) = text.rfind('}') {
            let json_candidate = &text[json_start..=json_end].trim();
            if let Ok(val) = serde_json::from_str::<serde_json::Value>(json_candidate) {
                return Some(val);
            }
        }
    }

    None
}
