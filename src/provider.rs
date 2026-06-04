use crate::traits::Provider;
use crate::utils::{
    build_reqwest_client, ChatMessage, LLMClient, LLMError, LLMResponse, MessageContent,
    TokenUsage, ToolCallFunction, ToolCallRequest,
};

/// Create a boxed [`Provider`] for the given provider name.
///
/// Routes `"anthropic"` to [`AnthropicProvider`] (Messages API) and everything else to
/// [`OpenAIProvider`] (OpenAI-compatible chat completions). Temperature is set to 0.3.
pub fn create_provider(
    provider_name: &str,
    base_url: &str,
    api_key: &str,
    model_name: &str,
) -> Box<dyn Provider> {
    if provider_name == "anthropic" {
        Box::new(AnthropicProvider::new(base_url, api_key, model_name).with_temperature(0.3))
    } else {
        let client =
            LLMClient::new_openai_compatible(base_url, api_key, model_name).with_temperature(0.3);
        Box::new(OpenAIProvider::new(client))
    }
}
use async_trait::async_trait;
use log::debug;
use serde_json::{json, Value};
use std::time::Duration;

/// Placeholder provider used when no API key is configured at startup.
/// Returns an error directing the user to configure a key or use `/model`.
#[derive(Clone)]
pub struct NoKeyProvider;

#[async_trait]
impl Provider for NoKeyProvider {
    async fn chat(
        &self,
        _messages: &[ChatMessage],
        _tools: Option<Value>,
    ) -> Result<LLMResponse, LLMError> {
        Err(LLMError::ApiError(
            "No API key configured. Use /model to select a provider, or add api_key to config.toml."
                .to_string(),
        ))
    }
}

/// A Provider implementation that wraps the existing LLMClient (OpenAI-compatible protocol).
#[derive(Clone)]
pub struct OpenAIProvider {
    client: LLMClient,
}

impl OpenAIProvider {
    pub fn new(client: LLMClient) -> Self {
        Self { client }
    }
}

#[async_trait]
impl Provider for OpenAIProvider {
    async fn chat(
        &self,
        messages: &[ChatMessage],
        tools: Option<serde_json::Value>,
    ) -> Result<LLMResponse, LLMError> {
        self.client.chat(messages, tools).await
    }
}

/// A Provider implementation for the Anthropic Messages API.
/// Translates between isanagent's internal OpenAI-format messages and Anthropic's format.
#[derive(Clone)]
pub struct AnthropicProvider {
    api_key: String,
    model: String,
    base_url: String,
    temperature: f32,
    max_tokens: u32,
    client: reqwest::Client,
}

impl AnthropicProvider {
    pub fn new(base_url: &str, api_key: &str, model: &str) -> Self {
        // Use a higher default for newer models that support longer output
        let max_tokens = if model.contains("opus") || model.contains("sonnet") {
            16384
        } else {
            8192
        };
        Self {
            api_key: api_key.to_string(),
            model: model.to_string(),
            base_url: base_url.to_string(),
            temperature: 0.3,
            max_tokens,
            client: build_reqwest_client(),
        }
    }

    pub fn with_temperature(mut self, temperature: f32) -> Self {
        self.temperature = temperature;
        self
    }

    pub fn with_max_tokens(mut self, max_tokens: u32) -> Self {
        self.max_tokens = max_tokens;
        self
    }

    /// Convert OpenAI-format tool definitions to Anthropic format.
    fn convert_tools(tools: &Value) -> Value {
        // OpenAI: [{ "type": "function", "function": { "name", "description", "parameters" } }]
        // Anthropic: [{ "name", "description", "input_schema" }]
        if let Some(arr) = tools.as_array() {
            let converted: Vec<Value> = arr
                .iter()
                .filter_map(|t| {
                    let func = t.get("function")?;
                    Some(json!({
                        "name": func.get("name")?,
                        "description": func.get("description").unwrap_or(&json!("")),
                        "input_schema": func.get("parameters").unwrap_or(&json!({"type": "object", "properties": {}}))
                    }))
                })
                .collect();
            json!(converted)
        } else {
            json!([])
        }
    }

    /// Convert internal ChatMessage list to Anthropic's format.
    /// Returns (system_prompt, messages_array).
    fn convert_messages(messages: &[ChatMessage]) -> (Option<String>, Vec<Value>) {
        let mut system_parts: Vec<String> = Vec::new();
        let mut anthropic_messages: Vec<Value> = Vec::new();

        for msg in messages {
            match msg.role.as_str() {
                "system" => {
                    if let Some(ref content) = msg.content {
                        system_parts.push(content.text_content());
                    }
                }
                "user" => {
                    let content = match &msg.content {
                        Some(MessageContent::Text(s)) => json!(s),
                        Some(MessageContent::Parts(parts)) => {
                            let anthropic_parts: Vec<Value> = parts
                                .iter()
                                .map(|p| match p {
                                    crate::utils::ContentPart::Text { text } => {
                                        json!({"type": "text", "text": text})
                                    }
                                    crate::utils::ContentPart::ImageUrl { image_url } => {
                                        // Convert data URI to Anthropic's base64 format
                                        if image_url.url.starts_with("data:") {
                                            let parts: Vec<&str> =
                                                image_url.url.splitn(2, ',').collect();
                                            if parts.len() == 2 {
                                                let media_type = parts[0]
                                                    .trim_start_matches("data:")
                                                    .trim_end_matches(";base64");
                                                json!({
                                                    "type": "image",
                                                    "source": {
                                                        "type": "base64",
                                                        "media_type": media_type,
                                                        "data": parts[1]
                                                    }
                                                })
                                            } else {
                                                json!({"type": "text", "text": "[image]"})
                                            }
                                        } else {
                                            // Anthropic only supports base64 image sources;
                                            // non-data URIs fall back to a text placeholder.
                                            json!({"type": "text", "text": format!("[image: {}]", image_url.url)})
                                        }
                                    }
                                })
                                .collect();
                            json!(anthropic_parts)
                        }
                        None => json!(""),
                    };
                    anthropic_messages.push(json!({"role": "user", "content": content}));
                }
                "assistant" => {
                    let mut content_blocks: Vec<Value> = Vec::new();

                    // Add text content if present
                    if let Some(ref content) = msg.content {
                        let text = content.text_content();
                        if !text.is_empty() {
                            content_blocks.push(json!({"type": "text", "text": text}));
                        }
                    }

                    // Add tool_use blocks if present
                    if let Some(ref tool_calls) = msg.tool_calls {
                        for tc in tool_calls {
                            let input: Value =
                                serde_json::from_str(&tc.function.arguments).unwrap_or(json!({}));
                            content_blocks.push(json!({
                                "type": "tool_use",
                                "id": tc.id,
                                "name": tc.function.name,
                                "input": input
                            }));
                        }
                    }

                    if content_blocks.is_empty() {
                        content_blocks.push(json!({"type": "text", "text": ""}));
                    }

                    anthropic_messages
                        .push(json!({"role": "assistant", "content": content_blocks}));
                }
                "tool" => {
                    // Anthropic expects tool results as user messages with tool_result content
                    let result_text = msg
                        .content
                        .as_ref()
                        .map(|c| c.text_content())
                        .unwrap_or_default();
                    let tool_call_id = msg.tool_call_id.as_deref().unwrap_or("");

                    let mut tool_result = json!({
                        "type": "tool_result",
                        "tool_use_id": tool_call_id,
                        "content": result_text
                    });
                    // Surface Anthropic's native failure flag so the model gets a structured
                    // error signal, not just an "Error:" text prefix. Only emit on failure
                    // (Anthropic treats an absent flag as success).
                    if msg.is_error == Some(true) {
                        tool_result["is_error"] = json!(true);
                    }
                    anthropic_messages.push(json!({
                        "role": "user",
                        "content": [tool_result]
                    }));
                }
                _ => {}
            }
        }

        // Merge consecutive same-role messages (Anthropic requires alternating roles)
        let merged = Self::merge_consecutive_roles(anthropic_messages);

        let system = if system_parts.is_empty() {
            None
        } else {
            Some(system_parts.join("\n\n"))
        };
        (system, merged)
    }

    /// Merge consecutive messages with the same role into one message.
    fn merge_consecutive_roles(messages: Vec<Value>) -> Vec<Value> {
        let mut result: Vec<Value> = Vec::new();
        for msg in messages {
            let role = msg["role"].as_str().unwrap_or("").to_string();
            if let Some(last) = result.last_mut() {
                if last["role"].as_str() == Some(&role) {
                    // Merge content arrays
                    let existing = last["content"].clone();
                    let new_content = msg["content"].clone();
                    let mut merged_content = match existing {
                        Value::Array(arr) => arr,
                        Value::String(s) => vec![json!({"type": "text", "text": s})],
                        _ => vec![],
                    };
                    match new_content {
                        Value::Array(arr) => merged_content.extend(arr),
                        Value::String(s) => merged_content.push(json!({"type": "text", "text": s})),
                        _ => {}
                    }
                    last["content"] = json!(merged_content);
                    continue;
                }
            }
            result.push(msg);
        }
        result
    }
}

#[async_trait]
impl Provider for AnthropicProvider {
    fn context_window_tokens(&self) -> Option<usize> {
        // PR-3: Anthropic publishes per-model input windows. Match on the
        // family in the model id; default 200k for current Claude (3.x/4.x)
        // Opus/Sonnet/Haiku and 100k for Claude 2.x. Unknown models return
        // `None` so the trigger check falls back to the absolute threshold.
        let m = self.model.to_lowercase();
        if m.contains("opus") || m.contains("sonnet") || m.contains("haiku") {
            Some(200_000)
        } else if m.contains("claude-2") {
            Some(100_000)
        } else {
            None
        }
    }

    async fn chat(
        &self,
        messages: &[ChatMessage],
        tools: Option<Value>,
    ) -> Result<LLMResponse, LLMError> {
        debug!(
            "AnthropicProvider: sending request to {} with {} messages",
            self.model,
            messages.len()
        );

        let (system, anthropic_messages) = Self::convert_messages(messages);

        let mut body = json!({
            "model": self.model,
            "messages": anthropic_messages,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature
        });

        if let Some(system_text) = system {
            // PR-6: mark the system block as cacheable so repeated summarizations
            // (and any other repeat call within a ~5 min window) hit the prompt
            // cache. Cache writes cost ~25% more on the first call; cache reads
            // cost ~10% of normal input tokens, so break-even is one reuse — system
            // prompts in a multi-turn agent reuse far more than that. Switching
            // from `string` to a single-block array form is required because
            // `cache_control` is a block-level marker.
            body["system"] = json!([
                {
                    "type": "text",
                    "text": system_text,
                    "cache_control": {"type": "ephemeral"}
                }
            ]);
        }

        if let Some(ref t) = tools {
            let anthropic_tools = Self::convert_tools(t);
            if let Some(arr) = anthropic_tools.as_array() {
                if !arr.is_empty() {
                    body["tools"] = anthropic_tools;
                }
            }
        }

        let res = self
            .client
            .post(&self.base_url)
            .header("x-api-key", &self.api_key)
            .header("anthropic-version", "2023-06-01")
            .header("Content-Type", "application/json")
            .timeout(Duration::from_secs(120))
            .json(&body)
            .send()
            .await?;

        let status = res.status();
        if !status.is_success() {
            let text = res.text().await.unwrap_or_default();
            // PR-4: classify context-length overflow as a typed error.
            if status == reqwest::StatusCode::BAD_REQUEST
                && LLMError::looks_like_context_overflow(&text)
            {
                return Err(LLMError::ContextOverflow {
                    tokens_attempted: 0,
                    max: None,
                });
            }
            let friendly =
                crate::utils::format_api_error(status.as_u16(), &text, &self.base_url, &self.model);
            return Err(LLMError::ApiError(friendly));
        }

        let raw_text = res
            .text()
            .await
            .map_err(|e| LLMError::ApiError(e.to_string()))?;
        let json_resp: Value = serde_json::from_str(&raw_text).map_err(LLMError::ParseError)?;

        // Parse Anthropic response format
        let mut content_text = String::new();
        let mut tool_calls: Vec<ToolCallRequest> = Vec::new();

        if let Some(content_blocks) = json_resp["content"].as_array() {
            for block in content_blocks {
                match block["type"].as_str() {
                    Some("text") => {
                        if let Some(text) = block["text"].as_str() {
                            if !content_text.is_empty() {
                                content_text.push('\n');
                            }
                            content_text.push_str(text);
                        }
                    }
                    Some("tool_use") => {
                        let id = block["id"].as_str().unwrap_or("").to_string();
                        let name = block["name"].as_str().unwrap_or("").to_string();
                        let input = block["input"].clone();
                        tool_calls.push(ToolCallRequest {
                            id,
                            tool_type: "function".to_string(),
                            extra_content: None,
                            function: ToolCallFunction {
                                name,
                                arguments: serde_json::to_string(&input).unwrap_or_default(),
                            },
                        });
                    }
                    _ => {}
                }
            }
        }

        // Check if response was truncated due to max_tokens
        let stop_reason = json_resp["stop_reason"].as_str().unwrap_or("");
        if stop_reason == "max_tokens" {
            log::warn!(
                "Anthropic response truncated (stop_reason=max_tokens, max_tokens={}). \
                 Tool calls may be incomplete.",
                self.max_tokens
            );
        }

        let usage = json_resp.get("usage").map(|u| {
            // PR-6.1: surface Anthropic's cache stats so eval tooling can verify
            // the PR-6 system-prompt cache_control is actually hitting. `total_tokens`
            // here is just input+output; cache reads/creations are reported separately.
            let cache_read = u["cache_read_input_tokens"].as_u64().unwrap_or(0) as u32;
            let cache_creation = u["cache_creation_input_tokens"].as_u64().unwrap_or(0) as u32;
            TokenUsage {
                prompt_tokens: u["input_tokens"].as_u64().unwrap_or(0) as u32,
                completion_tokens: u["output_tokens"].as_u64().unwrap_or(0) as u32,
                total_tokens: (u["input_tokens"].as_u64().unwrap_or(0)
                    + u["output_tokens"].as_u64().unwrap_or(0))
                    as u32,
                cache_read_tokens: cache_read,
                cache_creation_tokens: cache_creation,
            }
        });

        Ok(LLMResponse {
            content: content_text,
            tool_calls: if tool_calls.is_empty() {
                None
            } else {
                Some(tool_calls)
            },
            reasoning_content: None,
            usage,
        })
    }
}

#[cfg(test)]
mod is_error_tests {
    use super::AnthropicProvider;
    use crate::utils::ChatMessage;

    /// Collect every `tool_result` content block across the converted Anthropic messages.
    fn tool_result_blocks(msgs: &[serde_json::Value]) -> Vec<serde_json::Value> {
        let mut blocks = Vec::new();
        for m in msgs {
            if let Some(content) = m.get("content").and_then(|c| c.as_array()) {
                for b in content {
                    if b.get("type").and_then(|t| t.as_str()) == Some("tool_result") {
                        blocks.push(b.clone());
                    }
                }
            }
        }
        blocks
    }

    #[test]
    fn anthropic_tool_result_sets_is_error_only_on_failure() {
        let msgs = vec![
            ChatMessage::tool_with_error("Error: boom", "call_1", Some("exec"), true),
            ChatMessage::tool_with_error("ok output", "call_2", Some("exec"), false),
            ChatMessage::tool("legacy output", "call_3", Some("exec")), // is_error == None
        ];
        let (_system, anthropic) = AnthropicProvider::convert_messages(&msgs);
        let blocks = tool_result_blocks(&anthropic);
        assert_eq!(
            blocks.len(),
            3,
            "expected 3 tool_result blocks, got {anthropic:?}"
        );

        let by_id = |id: &str| {
            blocks
                .iter()
                .find(|b| b["tool_use_id"] == id)
                .unwrap_or_else(|| panic!("missing tool_result for {id}"))
        };
        // Failure -> native is_error: true.
        assert_eq!(
            by_id("call_1").get("is_error"),
            Some(&serde_json::json!(true))
        );
        // Success and legacy(None) -> NO is_error key (Anthropic treats absence as success).
        assert!(by_id("call_2").get("is_error").is_none());
        assert!(by_id("call_3").get("is_error").is_none());
    }

    #[test]
    fn is_error_is_never_serialized_to_openai_wire() {
        // The OpenAI-compatible request serializes ChatMessage directly; `is_error` must NOT
        // appear (strict endpoints reject unknown message fields).
        let msg = ChatMessage::tool_with_error("Error: boom", "call_1", Some("exec"), true);
        let v = serde_json::to_value(&msg).expect("serialize");
        assert!(
            v.get("is_error").is_none(),
            "is_error leaked onto the OpenAI-compatible wire: {v}"
        );
        // Also assert against the real request-body shape (LLMClient::chat serializes
        // `{"messages": [...]}` directly), not just the bare struct.
        let body = serde_json::json!({ "messages": [msg] });
        assert!(
            body["messages"][0].get("is_error").is_none(),
            "is_error leaked inside the messages array: {body}"
        );
    }
}
