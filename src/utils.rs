use log::{debug, info};
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::path::{Component, Path, PathBuf};
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
    /// DeepSeek-style chain-of-thought block surfaced by reasoning models (e.g. `deepseek-v4-pro`).
    /// Sent back unchanged on subsequent assistant turns so providers that key on it can chain
    /// reasoning across turns. Skipped from serialization when unset, so OpenAI-compatible
    /// providers that ignore the field see no change.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning_content: Option<String>,
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
            reasoning_content: None,
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
            reasoning_content: None,
        }
    }

    pub fn system(content: &str) -> Self {
        Self {
            role: "system".to_string(),
            content: Some(MessageContent::Text(content.to_string())),
            name: None,
            tool_calls: None,
            tool_call_id: None,
            reasoning_content: None,
        }
    }

    pub fn assistant(content: &str) -> Self {
        Self {
            role: "assistant".to_string(),
            content: Some(MessageContent::Text(content.to_string())),
            name: None,
            tool_calls: None,
            tool_call_id: None,
            reasoning_content: None,
        }
    }

    pub fn tool(content: &str, tool_call_id: &str) -> Self {
        Self {
            role: "tool".to_string(),
            content: Some(MessageContent::Text(content.to_string())),
            name: None,
            tool_calls: None,
            tool_call_id: Some(tool_call_id.to_string()),
            reasoning_content: None,
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

impl LLMError {
    /// Heuristic: should the caller retry this error after a backoff?
    /// True for network/IO errors and HTTP 429 / 5xx. False for parse errors and 4xx
    /// (those need user/config attention; retrying just hammers the upstream).
    pub fn is_transient(&self) -> bool {
        match self {
            LLMError::RequestError(_) => true,
            LLMError::ApiError(msg) => {
                let m = msg.to_lowercase();
                m.contains("status 5") || m.contains("status 429") || m.contains("rate limit")
            }
            LLMError::ParseError(_) | LLMError::NoContent => false,
        }
    }
}

/// Produce a user-friendly error message for HTTP API errors.
pub fn format_api_error(status: u16, body: &str, base_url: &str, model: &str) -> String {
    // Parse JSON; Gemini wraps errors in an array: [{...}] — unwrap to the first element.
    let parsed = serde_json::from_str::<serde_json::Value>(body)
        .ok()
        .map(|v| {
            if let Some(arr) = v.as_array() {
                arr.first().cloned().unwrap_or(v)
            } else {
                v
            }
        });

    // Try to extract a message from JSON error body
    let msg = parsed.as_ref().and_then(|v| {
        v.get("error")
            .and_then(|e| e.get("message"))
            .and_then(|m| m.as_str().map(|s| s.to_string()))
    });

    // Also try to extract error code/status string (e.g. "PERMISSION_DENIED")
    let error_code = parsed.as_ref().and_then(|v| {
        // Gemini: {"error":{"code":403,"status":"PERMISSION_DENIED"}}
        // OpenAI: {"error":{"code":"model_not_found","type":"invalid_request_error"}}
        let err = v.get("error")?;
        err.get("status")
            .or_else(|| err.get("code"))
            .and_then(|c| {
                if c.is_string() {
                    c.as_str().map(|s| s.to_string())
                } else {
                    None // skip numeric codes, we already have `status`
                }
            })
            .or_else(|| {
                err.get("type")
                    .and_then(|t| t.as_str().map(|s| s.to_string()))
            })
    });

    let code_tag = error_code
        .as_deref()
        .map(|c| format!(" [{}]", c))
        .unwrap_or_default();

    match status {
        401 => {
            let hint = if base_url.contains("openrouter") {
                "Check your OPENROUTER_API_KEY."
            } else if base_url.contains("openai") || base_url.contains("api.openai.com") {
                "Check your OPENAI_API_KEY."
            } else if base_url.contains("anthropic") {
                "Check your ANTHROPIC_API_KEY."
            } else if base_url.contains("googleapis") || base_url.contains("generativelanguage") {
                "Check your GEMINI_API_KEY."
            } else if base_url.contains("deepseek") {
                "Check your DEEPSEEK_API_KEY."
            } else {
                "Check that the correct API key is set for this provider."
            };
            format!(
                "({}{}) Authentication failed for model '{}'. {}",
                status, code_tag, model, hint
            )
        }
        403 => {
            let detail = msg
                .as_deref()
                .map(|m| format!(" {}", m))
                .unwrap_or_default();
            format!(
                "({}{}) Access denied for model '{}'.{} Your API key may not have permission to use this model.",
                status, code_tag, model, detail
            )
        }
        404 => format!(
            "({}{}) Model '{}' not found at {}. It may not exist or is not available on your plan.",
            status, code_tag, model, base_url
        ),
        429 => format!(
            "({}{}) Rate limit exceeded for model '{}'. Try again in a moment.",
            status, code_tag, model
        ),
        _ if status >= 500 => format!(
            "({}{}) Server error from provider while using model '{}'. Try again later.",
            status, code_tag, model
        ),
        _ => {
            let detail = msg.unwrap_or_else(|| body.chars().take(200).collect());
            format!(
                "({}{}) API error for model '{}': {}",
                status, code_tag, model, detail
            )
        }
    }
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
            let friendly = format_api_error(status.as_u16(), &text, &self.base_url, &self.model);
            return Err(LLMError::ApiError(friendly));
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

/// Joins `relative` under `root`, applying `.` / `..` lexically without traversing above `root`.
///
/// Extra `..` at the sandbox root are ignored (clamped), so `list_dir("..")` stays inside the
/// workspace instead of canonicalizing to the parent directory and tripping the boundary check
/// (common on Windows with `\\?\`-prefixed paths).
///
/// Callers must pass [`Path::is_absolute`] == false for `relative` (absolute paths should skip this).
pub fn join_lexically_under_root(root: &Path, relative: &Path) -> Result<PathBuf, String> {
    let mut out = root.to_path_buf();
    for comp in relative.components() {
        match comp {
            Component::CurDir => {}
            Component::ParentDir => {
                if out != root {
                    out.pop();
                }
            }
            Component::Normal(name) => out.push(name),
            Component::RootDir | Component::Prefix(_) => {
                return Err(format!(
                    "path component {comp:?} is not allowed in a sandbox-relative path"
                ));
            }
        }
    }
    Ok(out)
}

/// When the tool workspace is already `.../workspace` and the model passes `workspace/foo`, strip
/// the redundant first segment so resolution targets `.../workspace/foo` instead of
/// `.../workspace/workspace/foo`.
pub fn normalize_sandbox_relative_input(workspace_dir: &Path, path: &str) -> PathBuf {
    let trimmed = path.trim();
    let raw = Path::new(trimmed);
    if raw.is_absolute() {
        return raw.to_path_buf();
    }
    let Some(ws_leaf) = workspace_dir.file_name().and_then(|n| n.to_str()) else {
        return raw.to_path_buf();
    };
    let mut it = raw.components().peekable();
    if matches!(
        it.peek(),
        Some(Component::Normal(first))
            if first
                .to_str()
                .is_some_and(|s| s.eq_ignore_ascii_case(ws_leaf))
    ) {
        it.next();
        let rest: PathBuf = it.collect();
        return if rest.as_os_str().is_empty() {
            PathBuf::from(".")
        } else {
            rest
        };
    }
    raw.to_path_buf()
}

/// Resolves `agent_path` relative to `sandbox_dir`, ensuring the resulting
/// canonical path stays within the sandbox boundary.
///
/// Returns `None` when:
/// - the path escapes the sandbox via `../` traversal or absolute references
///   outside the sandbox directory, or
/// - the path does not exist on disk (canonicalization requires existence).
pub fn resolve_path(sandbox_dir: &Path, agent_path: &str) -> Option<PathBuf> {
    // Canonicalize the sandbox root first so we can fail fast before touching
    // any user-supplied path data.
    let sandbox_canonical = sandbox_dir.canonicalize().ok()?;

    let raw = Path::new(agent_path);

    // Absolute paths are only allowed when they live inside the sandbox.
    // Relative paths: strip redundant `workspace/`-style prefix, then join with `..` clamped.
    let joined = if raw.is_absolute() {
        raw.to_path_buf()
    } else {
        let rel = normalize_sandbox_relative_input(&sandbox_canonical, agent_path);
        join_lexically_under_root(&sandbox_canonical, &rel).ok()?
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

/// Truncate a `String` to at most `max_bytes` bytes without splitting a multi-byte
/// UTF-8 character, then append `suffix`. The truncation point is adjusted so the
/// **total** result (truncated content + suffix) stays within `max_bytes`.
pub fn truncate_utf8_safe(s: &mut String, max_bytes: usize, suffix: &str) {
    if s.len() <= max_bytes {
        return;
    }
    let mut end = max_bytes.saturating_sub(suffix.len());
    while end > 0 && !s.is_char_boundary(end) {
        end -= 1;
    }
    s.truncate(end);
    s.push_str(suffix);
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn assistant_message_serializes_reasoning_content_when_set() {
        let mut msg = ChatMessage::assistant("OK");
        msg.reasoning_content = Some("step 1 ... step 2".to_string());
        let v = serde_json::to_value(&msg).expect("serialize");
        assert_eq!(v["role"], "assistant");
        assert_eq!(v["content"], "OK");
        assert_eq!(v["reasoning_content"], "step 1 ... step 2");
    }

    #[test]
    fn assistant_message_skips_reasoning_content_when_unset() {
        let msg = ChatMessage::assistant("hi");
        let s = serde_json::to_string(&msg).expect("serialize");
        assert!(
            !s.contains("reasoning_content"),
            "unset reasoning_content must be skipped: {s}"
        );
    }

    #[test]
    fn chat_message_round_trips_reasoning_content() {
        let mut msg = ChatMessage::user("hi");
        msg.reasoning_content = Some("rc".to_string());
        let s = serde_json::to_string(&msg).unwrap();
        let back: ChatMessage = serde_json::from_str(&s).unwrap();
        assert_eq!(back.reasoning_content.as_deref(), Some("rc"));
    }

    #[test]
    fn llm_error_transience_classification() {
        assert!(LLMError::ApiError("Status 500: ...".into()).is_transient());
        assert!(LLMError::ApiError("Status 503 service unavailable".into()).is_transient());
        assert!(LLMError::ApiError("Status 429 too many requests".into()).is_transient());
        assert!(LLMError::ApiError("rate limit exceeded".into()).is_transient());
        assert!(!LLMError::ApiError("Status 400 bad request".into()).is_transient());
        assert!(!LLMError::ApiError("Status 401 unauthorized".into()).is_transient());
        assert!(!LLMError::NoContent.is_transient());
    }
}
