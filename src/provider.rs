use crate::traits::Provider;
use crate::utils::{
    build_reqwest_client, ChatMessage, LLMClient, LLMError, LLMResponse, MessageContent,
    TokenUsage, ToolCallFunction, ToolCallRequest,
};
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
        Self {
            api_key: api_key.to_string(),
            model: model.to_string(),
            base_url: base_url.to_string(),
            temperature: 0.3,
            max_tokens: 8192,
            client: build_reqwest_client(),
        }
    }

    pub fn with_temperature(mut self, temperature: f32) -> Self {
        self.temperature = temperature;
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
                                            json!({
                                                "type": "image",
                                                "source": {
                                                    "type": "url",
                                                    "url": image_url.url
                                                }
                                            })
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

                    anthropic_messages.push(json!({
                        "role": "user",
                        "content": [{
                            "type": "tool_result",
                            "tool_use_id": tool_call_id,
                            "content": result_text
                        }]
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
                        Value::String(s) => {
                            merged_content.push(json!({"type": "text", "text": s}))
                        }
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
            body["system"] = json!(system_text);
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

        let usage = json_resp.get("usage").map(|u| TokenUsage {
            prompt_tokens: u["input_tokens"].as_u64().unwrap_or(0) as u32,
            completion_tokens: u["output_tokens"].as_u64().unwrap_or(0) as u32,
            total_tokens: (u["input_tokens"].as_u64().unwrap_or(0)
                + u["output_tokens"].as_u64().unwrap_or(0)) as u32,
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
