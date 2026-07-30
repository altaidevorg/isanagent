use crate::traits::Provider;
use crate::utils::{
    build_reqwest_client, ChatMessage, ContentPart, Document, LLMClient, LLMError, LLMResponse,
    MessageContent, TokenUsage, ToolCallFunction, ToolCallRequest,
};

/// Live credentials for the active LLM session (updated on `/model` switch).
#[derive(Clone, Debug)]
pub struct ProviderCredentials {
    pub provider_name: String,
    pub base_url: String,
    pub api_key: String,
    pub model_name: String,
}

impl ProviderCredentials {
    pub fn empty() -> Self {
        Self {
            provider_name: String::new(),
            base_url: String::new(),
            api_key: String::new(),
            model_name: String::new(),
        }
    }

    pub fn is_usable(&self) -> bool {
        !self.api_key.is_empty() && !self.model_name.is_empty()
    }
}

/// Create a boxed [`Provider`] for the given provider name.
///
/// Routes `"anthropic"` to [`AnthropicProvider`] (Messages API) and everything else to
/// [`OpenAIProvider`] (OpenAI-compatible chat completions). Temperature defaults to 0.3.
pub fn create_provider(
    provider_name: &str,
    base_url: &str,
    api_key: &str,
    model_name: &str,
) -> Box<dyn Provider> {
    provider_for_agent(
        &ProviderCredentials {
            provider_name: provider_name.to_string(),
            base_url: base_url.to_string(),
            api_key: api_key.to_string(),
            model_name: model_name.to_string(),
        },
        None,
        None,
    )
}

/// Build a provider for a sub-agent, optionally overriding model and temperature.
pub fn provider_for_agent(
    creds: &ProviderCredentials,
    model_override: Option<&str>,
    temperature_override: Option<f32>,
) -> Box<dyn Provider> {
    let model = model_override.unwrap_or(&creds.model_name);
    let temp = temperature_override.unwrap_or(0.3);
    if creds.provider_name == "anthropic" {
        Box::new(
            AnthropicProvider::new(&creds.base_url, &creds.api_key, model).with_temperature(temp),
        )
    } else if creds.provider_name == "gemini"
        && creds.base_url.contains("generativelanguage.googleapis.com")
    {
        Box::new(GeminiProvider::new(&creds.base_url, &creds.api_key, model).with_temperature(temp))
    } else {
        let client = LLMClient::new_openai_compatible(&creds.base_url, &creds.api_key, model)
            .with_temperature(temp);
        Box::new(OpenAIProvider::new(client))
    }
}
use async_trait::async_trait;
use base64::{engine::general_purpose::STANDARD as BASE64_STANDARD, Engine as _};
use log::debug;
use serde_json::{json, Value};
use std::collections::HashMap;
use std::sync::Arc;
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

/// Deterministic in-process provider for host/CLI smoke tests.
#[derive(Clone)]
pub struct ScriptedProvider {
    responses: Arc<std::sync::Mutex<std::collections::VecDeque<String>>>,
    fallback: String,
}

impl ScriptedProvider {
    pub fn new(responses: Vec<String>) -> Self {
        let fallback = responses
            .last()
            .cloned()
            .unwrap_or_else(|| "scripted-ok".to_string());
        Self {
            responses: Arc::new(std::sync::Mutex::new(responses.into())),
            fallback,
        }
    }
}

#[async_trait]
impl Provider for ScriptedProvider {
    async fn chat(
        &self,
        _messages: &[ChatMessage],
        _tools: Option<Value>,
    ) -> Result<LLMResponse, LLMError> {
        let content = {
            let mut queue = self
                .responses
                .lock()
                .map_err(|_| LLMError::ApiError("scripted provider lock poisoned".into()))?;
            queue.pop_front().unwrap_or_else(|| self.fallback.clone())
        };
        Ok(LLMResponse {
            content,
            tool_calls: None,
            reasoning_content: None,
            usage: Some(crate::utils::TokenUsage {
                prompt_tokens: 1,
                completion_tokens: 1,
                total_tokens: 2,
                cache_read_tokens: 0,
                cache_creation_tokens: 0,
            }),
        })
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

/// Convert a document attachment into text for APIs that do not support IsanAgent's
/// internal `document` content part. Keeping this conversion at the provider boundary
/// means strict OpenAI-compatible endpoints never receive an unknown content-part type.
fn document_text_fallback(document: &Document) -> String {
    let label = document.name.as_deref().unwrap_or("attached document");

    if document.media_type != "application/pdf" {
        return format!(
            "[Document attachment: {label} ({})]\n\nThis document format cannot be sent natively to the selected model.",
            document.media_type
        );
    }

    let extracted = BASE64_STANDARD
        .decode(&document.data)
        .map_err(|error| format!("invalid base64 payload: {error}"))
        .and_then(|pdf_bytes| crate::utils::extract_markdown_from_pdf_bytes(&pdf_bytes));

    match extracted {
        Ok(text) if !text.trim().is_empty() => format!("[PDF attachment: {label}]\n\n{text}"),
        Ok(_) | Err(_) => format!(
            "[PDF attachment: {label}]\n\nThe PDF could not be converted to text for the selected model."
        ),
    }
}

/// Replace document parts with text before they reach an OpenAI-compatible API.
fn messages_with_document_text_fallback(messages: &[ChatMessage]) -> Vec<ChatMessage> {
    messages
        .iter()
        .cloned()
        .map(|mut message| {
            if let Some(MessageContent::Parts(parts)) = &mut message.content {
                for part in parts {
                    if let ContentPart::Document { document } = part {
                        *part = ContentPart::Text {
                            text: document_text_fallback(document),
                        };
                    }
                }
            }
            message
        })
        .collect()
}

#[async_trait]
impl Provider for OpenAIProvider {
    async fn chat(
        &self,
        messages: &[ChatMessage],
        tools: Option<serde_json::Value>,
    ) -> Result<LLMResponse, LLMError> {
        let messages = messages_with_document_text_fallback(messages);
        self.client.chat(&messages, tools).await
    }
}

/// Native Gemini GenerateContent provider.
///
/// Gemini's OpenAI-compatible endpoint is useful for text chat, but does not expose the
/// first-class PDF input supported by Gemini. This provider uses the documented Gemini wire
/// format so document attachments retain visual understanding instead of falling back to text.
#[derive(Clone)]
pub struct GeminiProvider {
    api_key: String,
    model: String,
    base_url: String,
    temperature: f32,
    client: reqwest::Client,
}

impl GeminiProvider {
    pub fn new(openai_compatible_url: &str, api_key: &str, model: &str) -> Self {
        let origin = openai_compatible_url
            .split("/v1beta/")
            .next()
            .unwrap_or(openai_compatible_url)
            .trim_end_matches('/');
        Self {
            api_key: api_key.to_string(),
            model: model.to_string(),
            base_url: format!("{origin}/v1beta/models/{model}:generateContent"),
            temperature: 0.3,
            client: build_reqwest_client(),
        }
    }

    pub fn with_temperature(mut self, temperature: f32) -> Self {
        self.temperature = temperature;
        self
    }

    fn convert_tools(tools: &Value) -> Value {
        let declarations: Vec<Value> = tools
            .as_array()
            .into_iter()
            .flatten()
            .filter_map(|tool| tool.get("function").cloned())
            .collect();
        json!([{"functionDeclarations": declarations}])
    }

    fn data_uri_part(url: &str) -> Value {
        let Some((prefix, data)) = url.split_once(',') else {
            return json!({"text": format!("[image: {url}]")});
        };
        let Some(media_type) = prefix
            .strip_prefix("data:")
            .and_then(|value| value.strip_suffix(";base64"))
        else {
            return json!({"text": format!("[image: {url}]")});
        };
        json!({"inlineData": {"mimeType": media_type, "data": data}})
    }

    /// Convert internal history to Gemini's Content/Part schema.
    fn convert_messages(messages: &[ChatMessage]) -> (Option<Value>, Vec<Value>) {
        let mut system_parts = Vec::new();
        let mut contents = Vec::new();
        let mut tool_names = HashMap::<String, String>::new();

        for message in messages {
            match message.role.as_str() {
                "system" => {
                    if let Some(content) = &message.content {
                        system_parts.push(content.text_content());
                    }
                }
                "user" => {
                    let parts = match &message.content {
                        Some(MessageContent::Text(text)) => vec![json!({"text": text})],
                        Some(MessageContent::Parts(parts)) => parts
                            .iter()
                            .map(|part| match part {
                                ContentPart::Text { text } => json!({"text": text}),
                                ContentPart::ImageUrl { image_url }
                                    if image_url.url.starts_with("data:") =>
                                {
                                    Self::data_uri_part(&image_url.url)
                                }
                                ContentPart::ImageUrl { image_url } => {
                                    json!({"text": format!("[image: {}]", image_url.url)})
                                }
                                ContentPart::Document { document } => json!({
                                    "inlineData": {
                                        "mimeType": document.media_type,
                                        "data": document.data,
                                    }
                                }),
                            })
                            .collect(),
                        None => vec![json!({"text": ""})],
                    };
                    contents.push(json!({"role": "user", "parts": parts}));
                }
                "assistant" => {
                    let mut parts = Vec::new();
                    if let Some(content) = &message.content {
                        let text = content.text_content();
                        if !text.is_empty() {
                            parts.push(json!({"text": text}));
                        }
                    }
                    if let Some(calls) = &message.tool_calls {
                        for call in calls {
                            tool_names.insert(call.id.clone(), call.function.name.clone());
                            let args = serde_json::from_str::<Value>(&call.function.arguments)
                                .unwrap_or_else(|_| json!({}));
                            let function_call = json!({
                                "name": call.function.name,
                                "args": args,
                                "id": call.id,
                            });
                            let mut part = json!({"functionCall": function_call});
                            if let Some(extra) = &call.extra_content {
                                if let Some(signature) = extra
                                    .get("thoughtSignature")
                                    .or_else(|| extra.get("thought_signature"))
                                {
                                    part["thoughtSignature"] = signature.clone();
                                }
                            }
                            parts.push(part);
                        }
                    }
                    if parts.is_empty() {
                        parts.push(json!({"text": ""}));
                    }
                    contents.push(json!({"role": "model", "parts": parts}));
                }
                "tool" => {
                    let id = message.tool_call_id.as_deref().unwrap_or("");
                    let name = message
                        .name
                        .as_deref()
                        .or_else(|| tool_names.get(id).map(String::as_str))
                        .unwrap_or("unknown_tool");
                    let result = message
                        .content
                        .as_ref()
                        .map(MessageContent::text_content)
                        .unwrap_or_default();
                    let mut response = json!({"result": result});
                    if message.is_error == Some(true) {
                        response["error"] = json!(true);
                    }
                    contents.push(json!({
                        "role": "user",
                        "parts": [{"functionResponse": {"name": name, "response": response, "id": id}}]
                    }));
                }
                _ => {}
            }
        }

        let system = (!system_parts.is_empty())
            .then(|| json!({"parts": [{"text": system_parts.join("\n\n")} ]}));
        (system, contents)
    }
}

#[async_trait]
impl Provider for GeminiProvider {
    async fn chat(
        &self,
        messages: &[ChatMessage],
        tools: Option<Value>,
    ) -> Result<LLMResponse, LLMError> {
        let (system_instruction, contents) = Self::convert_messages(messages);
        let mut body = json!({
            "contents": contents,
            "generationConfig": {"temperature": self.temperature},
        });
        if let Some(system_instruction) = system_instruction {
            body["systemInstruction"] = system_instruction;
        }
        if let Some(tools) = tools {
            let tools = Self::convert_tools(&tools);
            if tools[0]["functionDeclarations"]
                .as_array()
                .is_some_and(|declarations| !declarations.is_empty())
            {
                body["tools"] = tools;
            }
        }

        let res = self
            .client
            .post(&self.base_url)
            .header("x-goog-api-key", &self.api_key)
            .header("Content-Type", "application/json")
            .timeout(Duration::from_secs(120))
            .json(&body)
            .send()
            .await?;
        let status = res.status();
        if !status.is_success() {
            let text = res.text().await.unwrap_or_default();
            if status == reqwest::StatusCode::BAD_REQUEST
                && LLMError::looks_like_context_overflow(&text)
            {
                return Err(LLMError::ContextOverflow {
                    tokens_attempted: 0,
                    max: None,
                });
            }
            return Err(LLMError::ApiError(crate::utils::format_api_error(
                status.as_u16(),
                &text,
                &self.base_url,
                &self.model,
            )));
        }

        let response: Value = res.json().await?;
        let parts = response["candidates"][0]["content"]["parts"]
            .as_array()
            .ok_or(LLMError::NoContent)?;
        let mut text_parts = Vec::new();
        let mut tool_calls = Vec::new();
        for (index, part) in parts.iter().enumerate() {
            if let Some(text) = part.get("text").and_then(Value::as_str) {
                text_parts.push(text);
            }
            if let Some(call) = part.get("functionCall") {
                let name = call["name"].as_str().unwrap_or("unknown_tool").to_string();
                let id = call["id"]
                    .as_str()
                    .map(str::to_string)
                    .unwrap_or_else(|| format!("gemini_call_{index}"));
                let mut extra_content = serde_json::Map::new();
                if let Some(signature) = part
                    .get("thoughtSignature")
                    .or_else(|| part.get("thought_signature"))
                {
                    extra_content.insert("thought_signature".to_string(), signature.clone());
                }
                tool_calls.push(ToolCallRequest {
                    id,
                    tool_type: "function".to_string(),
                    extra_content: (!extra_content.is_empty())
                        .then_some(Value::Object(extra_content)),
                    function: ToolCallFunction {
                        name,
                        arguments: call
                            .get("args")
                            .cloned()
                            .unwrap_or_else(|| json!({}))
                            .to_string(),
                    },
                });
            }
        }

        let usage = response.get("usageMetadata").map(|usage| TokenUsage {
            prompt_tokens: usage["promptTokenCount"].as_u64().unwrap_or(0) as u32,
            completion_tokens: usage["candidatesTokenCount"].as_u64().unwrap_or(0) as u32,
            total_tokens: usage["totalTokenCount"].as_u64().unwrap_or(0) as u32,
            cache_read_tokens: usage["cachedContentTokenCount"].as_u64().unwrap_or(0) as u32,
            cache_creation_tokens: 0,
        });
        Ok(LLMResponse {
            content: text_parts.join(""),
            tool_calls: (!tool_calls.is_empty()).then_some(tool_calls),
            reasoning_content: None,
            usage,
        })
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
    fn convert_messages(model: &str, messages: &[ChatMessage]) -> (Option<String>, Vec<Value>) {
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
                                    crate::utils::ContentPart::Document { document } => {
                                        // Native document blocks are only supported by Sonnet.
                                        // Other Anthropic models must receive ordinary text;
                                        // otherwise the Messages API rejects the request.
                                        if model.to_ascii_lowercase().contains("sonnet")
                                            && document.media_type == "application/pdf"
                                        {
                                            let mut block = json!({
                                                "type": "document",
                                                "source": {
                                                    "type": "base64",
                                                    "media_type": document.media_type,
                                                    "data": document.data,
                                                }
                                            });
                                            if let Some(name) = &document.name {
                                                block["title"] = json!(name);
                                            }
                                            block
                                        } else {
                                            json!({
                                                "type": "text",
                                                "text": document_text_fallback(document)
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

        let (system, anthropic_messages) = Self::convert_messages(&self.model, messages);

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
    use super::{messages_with_document_text_fallback, AnthropicProvider, GeminiProvider};
    use crate::utils::{
        ChatMessage, ContentPart, Document, MessageContent, ToolCallFunction, ToolCallRequest,
    };

    fn pdf_message() -> ChatMessage {
        ChatMessage::user_multimodal(
            "Please summarize this file.",
            &[ContentPart::Document {
                document: Document {
                    data: "not-valid-pdf-data".to_string(),
                    media_type: "application/pdf".to_string(),
                    name: Some("report.pdf".to_string()),
                },
            }],
        )
    }

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
        let (_system, anthropic) = AnthropicProvider::convert_messages("claude-3-5-sonnet", &msgs);
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

    #[test]
    fn anthropic_uses_native_pdf_blocks_only_for_sonnet() {
        let messages = vec![pdf_message()];

        let (_system, sonnet) =
            AnthropicProvider::convert_messages("claude-3-5-sonnet-latest", &messages);
        assert_eq!(
            sonnet[0]["content"][1]["type"],
            serde_json::json!("document"),
            "Sonnet should retain the native PDF block"
        );

        let (_system, haiku) = AnthropicProvider::convert_messages("claude-3-5-haiku", &messages);
        assert_eq!(
            haiku[0]["content"][1]["type"],
            serde_json::json!("text"),
            "models without native document support must receive text"
        );
        assert!(
            haiku[0]["content"][1]["text"]
                .as_str()
                .is_some_and(|text| text.contains("report.pdf")),
            "the fallback should identify the attached document"
        );
    }

    #[test]
    fn openai_compatible_messages_never_include_document_parts() {
        let messages = messages_with_document_text_fallback(&[pdf_message()]);
        let Some(MessageContent::Parts(parts)) = &messages[0].content else {
            panic!("expected multimodal parts");
        };

        assert!(
            parts
                .iter()
                .all(|part| !matches!(part, ContentPart::Document { .. })),
            "OpenAI-compatible request must not contain IsanAgent document parts"
        );
        assert!(matches!(parts[1], ContentPart::Text { .. }));
    }

    #[test]
    fn gemini_keeps_pdf_as_native_inline_data() {
        let messages = vec![ChatMessage::system("Follow the document."), pdf_message()];
        let (system, contents) = GeminiProvider::convert_messages(&messages);

        assert_eq!(
            system.expect("system instruction")["parts"][0]["text"],
            serde_json::json!("Follow the document.")
        );
        assert_eq!(
            contents[0]["role"],
            serde_json::json!("user"),
            "PDF must remain attached to the user turn"
        );
        assert_eq!(
            contents[0]["parts"][1]["inlineData"]["mimeType"],
            serde_json::json!("application/pdf")
        );
        assert_eq!(
            contents[0]["parts"][1]["inlineData"]["data"],
            serde_json::json!("not-valid-pdf-data")
        );
    }

    #[test]
    fn gemini_preserves_tool_call_ids_and_results() {
        let mut assistant = ChatMessage::assistant("");
        assistant.tool_calls = Some(vec![ToolCallRequest {
            id: "gemini-call-1".to_string(),
            tool_type: "function".to_string(),
            extra_content: Some(serde_json::json!({"thought_signature": "signed"})),
            function: ToolCallFunction {
                name: "read_file".to_string(),
                arguments: "{\"path\":\"README.md\"}".to_string(),
            },
        }]);
        let messages = vec![
            assistant,
            ChatMessage::tool("contents", "gemini-call-1", Some("read_file")),
        ];

        let (_system, contents) = GeminiProvider::convert_messages(&messages);
        assert_eq!(
            contents[0]["role"],
            serde_json::json!("model"),
            "assistant turns are Gemini model turns"
        );
        assert_eq!(
            contents[0]["parts"][0]["functionCall"]["id"],
            serde_json::json!("gemini-call-1")
        );
        assert_eq!(
            contents[0]["parts"][0]["thoughtSignature"],
            serde_json::json!("signed")
        );
        assert!(
            contents[0]["parts"][0]["functionCall"]
                .get("thoughtSignature")
                .is_none(),
            "thought signatures belong to the surrounding Gemini Part, not FunctionCall"
        );
        assert_eq!(
            contents[1]["parts"][0]["functionResponse"]["name"],
            serde_json::json!("read_file")
        );
        assert_eq!(
            contents[1]["parts"][0]["functionResponse"]["id"],
            serde_json::json!("gemini-call-1")
        );
    }

    #[test]
    fn gemini_uses_native_generate_content_endpoint() {
        let provider = GeminiProvider::new(
            "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions",
            "test-key",
            "gemini-2.5-flash",
        );
        assert_eq!(
            provider.base_url,
            "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent"
        );
    }
}
