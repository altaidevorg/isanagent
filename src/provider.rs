use crate::traits::{FinishReason, Provider, StreamChunk};
use crate::utils::{
    build_reqwest_client, drive_sse, ChatMessage, ContentPart, Document, LLMClient, LLMError,
    LLMResponse, MessageContent, SseStep, TokenUsage, ToolCallFunction, ToolCallRequest,
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
    let raw_model = model_override.unwrap_or(&creds.model_name);
    let model = raw_model.strip_prefix("models/").unwrap_or(raw_model);
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
            "No API key configured. Use /key <api_key> to set one now, /model to switch \
             providers, or add api_key to config.toml."
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

    async fn stream(
        &self,
        messages: &[ChatMessage],
        tools: Option<serde_json::Value>,
        sink: tokio::sync::mpsc::Sender<crate::traits::StreamChunk>,
    ) -> Result<LLMResponse, LLMError> {
        let messages = messages_with_document_text_fallback(messages);
        self.client.chat_stream(&messages, tools, sink).await
    }
}

/// Gemini `Schema` is a subset of JSON Schema. MCP servers (FastMCP/Pydantic) emit
/// keys such as `additionalProperties` that `generateContent` rejects as
/// `INVALID_ARGUMENT`. Strip unsupported keys recursively while keeping the
/// documented Gemini fields (`anyOf`, `properties`, `items`, …).
fn sanitize_gemini_schema(value: &mut Value) {
    const STRIP: &[&str] = &[
        "additionalProperties",
        "$schema",
        "$id",
        "$ref",
        "$defs",
        "definitions",
        "unevaluatedProperties",
        "prefixItems",
        "dependentRequired",
        "dependentSchemas",
        "if",
        "then",
        "else",
        "not",
        "oneOf",
        "allOf",
        "uniqueItems",
        "exclusiveMinimum",
        "exclusiveMaximum",
        "contentMediaType",
        "contentEncoding",
    ];
    match value {
        Value::Object(map) => {
            for key in STRIP {
                map.remove(*key);
            }
            for child in map.values_mut() {
                sanitize_gemini_schema(child);
            }
        }
        Value::Array(items) => {
            for item in items {
                sanitize_gemini_schema(item);
            }
        }
        _ => {}
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
}

impl GeminiProvider {
    pub fn new(openai_compatible_url: &str, api_key: &str, model: &str) -> Self {
        let origin = openai_compatible_url
            .split("/v1beta/")
            .next()
            .unwrap_or(openai_compatible_url)
            .trim_end_matches('/');
        let clean_model = model.strip_prefix("models/").unwrap_or(model);
        Self {
            api_key: api_key.to_string(),
            model: clean_model.to_string(),
            base_url: format!("{origin}/v1beta/models/{clean_model}:generateContent"),
            temperature: 0.3,
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
            .map(|mut func| {
                if let Some(parameters) = func.get_mut("parameters") {
                    sanitize_gemini_schema(parameters);
                }
                func
            })
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

impl GeminiProvider {
    /// `generateContent` request body shared by [`Provider::chat`] and [`Provider::stream`].
    fn request_body(&self, messages: &[ChatMessage], tools: Option<Value>) -> Value {
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
        body
    }

    /// `:streamGenerateContent?alt=sse` sibling of the `:generateContent` endpoint.
    fn stream_url(&self) -> String {
        match self.base_url.strip_suffix(":generateContent") {
            Some(prefix) => format!("{prefix}:streamGenerateContent?alt=sse"),
            None => format!("{}?alt=sse", self.base_url),
        }
    }

    async fn post(&self, url: &str, body: &Value) -> Result<reqwest::Response, LLMError> {
        // A new client each call. The shared pool kept a stalled connection, and
        // the next chat — including New chat — waited on it until the process restarted.
        let client = build_reqwest_client();
        Ok(client
            .post(url)
            .header("x-goog-api-key", &self.api_key)
            .header("Content-Type", "application/json")
            .timeout(Duration::from_secs(45))
            .json(body)
            .send()
            .await?)
    }

    fn classify_http_error(&self, status: reqwest::StatusCode, text: &str) -> LLMError {
        if status == reqwest::StatusCode::BAD_REQUEST && LLMError::looks_like_context_overflow(text)
        {
            return LLMError::ContextOverflow {
                tokens_attempted: 0,
                max: None,
            };
        }
        LLMError::ApiError(crate::utils::format_api_error(
            status.as_u16(),
            text,
            &self.base_url,
            &self.model,
        ))
    }

    /// Convert one `GenerateContentResponse` into an [`LLMResponse`].
    ///
    /// Parts marked `thought: true` are model reasoning summaries: they go to
    /// `reasoning_content`, never into the visible reply.
    fn parse_response(response: &Value) -> Result<LLMResponse, LLMError> {
        let parts = response["candidates"][0]["content"]["parts"]
            .as_array()
            .ok_or(LLMError::NoContent)?;
        let mut text_parts = Vec::new();
        let mut thought_parts = Vec::new();
        let mut tool_calls = Vec::new();
        for (index, part) in parts.iter().enumerate() {
            if let Some(text) = part.get("text").and_then(Value::as_str) {
                if part.get("thought").and_then(Value::as_bool) == Some(true) {
                    thought_parts.push(text);
                } else {
                    text_parts.push(text);
                }
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
            reasoning_content: (!thought_parts.is_empty()).then(|| thought_parts.join("")),
            usage,
        })
    }
}

/// Map a Gemini `finishReason` to the typed stream finish reason.
fn gemini_finish_reason(reason: Option<&str>, has_tool_calls: bool) -> FinishReason {
    match reason {
        _ if has_tool_calls => FinishReason::ToolCalls,
        None | Some("STOP") | Some("FINISH_REASON_UNSPECIFIED") => FinishReason::Stop,
        Some("MAX_TOKENS") => FinishReason::Length,
        Some(
            "SAFETY" | "RECITATION" | "BLOCKLIST" | "PROHIBITED_CONTENT" | "SPII" | "IMAGE_SAFETY",
        ) => FinishReason::ContentFilter,
        Some(other) => FinishReason::Other(other.to_string()),
    }
}

/// Folds `streamGenerateContent` SSE chunks back into one `GenerateContentResponse`, so the
/// final [`LLMResponse`] comes from the same [`GeminiProvider::parse_response`] as `chat()`.
///
/// Consecutive text parts of the same kind (visible vs `thought`) are coalesced the way the
/// non-streaming endpoint returns them; `functionCall` parts are kept verbatim (including
/// their `thoughtSignature`, which Gemini requires to be echoed back on the next turn).
#[derive(Default)]
struct GeminiStreamAccumulator {
    /// `None` until a chunk carries `candidates[0].content.parts`; mirrors the non-streaming
    /// `NoContent` error for blocked prompts / empty candidates.
    parts: Option<Vec<Value>>,
    candidate: serde_json::Map<String, Value>,
    top_level: serde_json::Map<String, Value>,
}

impl GeminiStreamAccumulator {
    fn push_data(
        &mut self,
        data: &str,
        out: &mut Vec<StreamChunk>,
        provider: &GeminiProvider,
    ) -> Result<SseStep, LLMError> {
        if data.is_empty() || data == "[DONE]" {
            return Ok(SseStep::Continue);
        }
        let mut value: Value = serde_json::from_str(data).map_err(LLMError::ParseError)?;
        if let Some(err) = value.get("error") {
            let status = err.get("code").and_then(Value::as_u64).unwrap_or(500) as u16;
            let status = reqwest::StatusCode::from_u16(status)
                .unwrap_or(reqwest::StatusCode::INTERNAL_SERVER_ERROR);
            return Err(provider.classify_http_error(status, data));
        }
        let Some(obj) = value.as_object_mut() else {
            return Ok(SseStep::Continue);
        };
        let candidate = obj
            .remove("candidates")
            .and_then(|c| match c {
                Value::Array(mut arr) if !arr.is_empty() => Some(arr.swap_remove(0)),
                _ => None,
            })
            .and_then(|c| match c {
                Value::Object(map) => Some(map),
                _ => None,
            });
        // usageMetadata, promptFeedback, modelVersion, responseId: last value wins
        // (Gemini reports cumulative usage on each chunk).
        for (key, v) in std::mem::take(obj) {
            self.top_level.insert(key, v);
        }
        let Some(mut candidate) = candidate else {
            return Ok(SseStep::Continue);
        };
        if let Some(Value::Object(mut content)) = candidate.remove("content") {
            if let Some(Value::Array(parts)) = content.remove("parts") {
                let acc = self.parts.get_or_insert_with(Vec::new);
                for part in parts {
                    Self::push_part(acc, part, out);
                }
            }
        }
        for (key, v) in candidate {
            self.candidate.insert(key, v);
        }
        Ok(SseStep::Continue)
    }

    fn push_part(acc: &mut Vec<Value>, part: Value, out: &mut Vec<StreamChunk>) {
        let is_text =
            part.get("text").is_some_and(Value::is_string) && part.get("functionCall").is_none();
        if !is_text {
            if let Some(call) = part.get("functionCall") {
                out.push(StreamChunk::ToolCallDelta {
                    id: call["id"].as_str().unwrap_or_default().to_string(),
                    name: call["name"].as_str().map(str::to_string),
                    args_delta: call.get("args").map(Value::to_string).unwrap_or_default(),
                });
            }
            acc.push(part);
            return;
        }
        let thought = part.get("thought").and_then(Value::as_bool) == Some(true);
        let text = part["text"].as_str().unwrap_or_default().to_string();
        if !text.is_empty() {
            out.push(if thought {
                StreamChunk::ReasoningDelta(text.clone())
            } else {
                StreamChunk::TextDelta(text.clone())
            });
        }
        let merge_into_last = acc.last().is_some_and(|prev| {
            prev.get("text").is_some_and(Value::is_string)
                && prev.get("functionCall").is_none()
                && (prev.get("thought").and_then(Value::as_bool) == Some(true)) == thought
        });
        match acc.last_mut() {
            Some(Value::Object(prev)) if merge_into_last => {
                let mut merged = prev
                    .get("text")
                    .and_then(Value::as_str)
                    .unwrap_or_default()
                    .to_string();
                merged.push_str(&text);
                if let Value::Object(fields) = part {
                    prev.extend(fields);
                }
                prev.insert("text".to_string(), Value::String(merged));
            }
            _ => acc.push(part),
        }
    }

    /// The equivalent non-streaming `GenerateContentResponse`.
    fn into_response_value(self) -> Value {
        let mut response = Value::Object(self.top_level);
        let mut candidate = self.candidate;
        if let Some(parts) = self.parts {
            candidate.insert(
                "content".to_string(),
                json!({"role": "model", "parts": parts}),
            );
        }
        if !candidate.is_empty() {
            response["candidates"] = json!([Value::Object(candidate)]);
        }
        response
    }
}

/// HTTP statuses on a streaming request that mean "this endpoint does not do streaming"
/// (a proxy without the streaming route, or a server rejecting the stream parameter). The
/// request is then repeated on the non-streaming endpoint, which reports the real error, if any.
fn stream_rejected(status: reqwest::StatusCode) -> bool {
    matches!(
        status,
        reqwest::StatusCode::BAD_REQUEST
            | reqwest::StatusCode::NOT_FOUND
            | reqwest::StatusCode::METHOD_NOT_ALLOWED
            | reqwest::StatusCode::NOT_IMPLEMENTED
    )
}

#[async_trait]
impl Provider for GeminiProvider {
    async fn chat(
        &self,
        messages: &[ChatMessage],
        tools: Option<Value>,
    ) -> Result<LLMResponse, LLMError> {
        let body = self.request_body(messages, tools);
        let res = self.post(&self.base_url, &body).await?;
        let status = res.status();
        if !status.is_success() {
            let text = res.text().await.unwrap_or_default();
            return Err(self.classify_http_error(status, &text));
        }

        let response: Value = res.json().await?;
        Self::parse_response(&response)
    }

    /// Streams over `:streamGenerateContent?alt=sse`. Each SSE event is a partial
    /// `GenerateContentResponse`; text parts are forwarded as they arrive and the chunks are
    /// folded back into the response `chat()` would have returned. A rejected streaming request
    /// (400/404/405/501, other than context overflow) is repeated through `chat()`.
    async fn stream(
        &self,
        messages: &[ChatMessage],
        tools: Option<Value>,
        sink: tokio::sync::mpsc::Sender<StreamChunk>,
    ) -> Result<LLMResponse, LLMError> {
        let body = self.request_body(messages, tools.clone());
        let mut res = self.post(&self.stream_url(), &body).await?;
        let status = res.status();
        if !status.is_success() {
            let text = res.text().await.unwrap_or_default();
            let err = self.classify_http_error(status, &text);
            if stream_rejected(status) && !matches!(err, LLMError::ContextOverflow { .. }) {
                debug!(
                    "Gemini streaming request rejected (HTTP {status}); retrying without stream"
                );
                let resp = self.chat(messages, tools).await?;
                crate::utils::emit_whole_response(&sink, &resp).await;
                return Ok(resp);
            }
            return Err(err);
        }

        let mut acc = GeminiStreamAccumulator::default();
        if crate::utils::is_event_stream(&res) {
            drive_sse(&mut res, &sink, LLMError::RequestError, |event, out| {
                acc.push_data(&event.data, out, self)
            })
            .await?;
        } else {
            // A server that ignored `alt=sse` answers with a JSON array of chunks (or a
            // single response object).
            let value: Value = res.json().await?;
            let chunks = match value {
                Value::Array(chunks) => chunks,
                other => vec![other],
            };
            let mut out = Vec::new();
            for chunk in chunks {
                acc.push_data(&chunk.to_string(), &mut out, self)?;
            }
            for c in out {
                let _ = sink.send(c).await;
            }
        }

        let response = acc.into_response_value();
        let resp = Self::parse_response(&response)?;
        if let Some(usage) = &resp.usage {
            let _ = sink.send(StreamChunk::Usage(usage.clone())).await;
        }
        let reason = gemini_finish_reason(
            response["candidates"][0]["finishReason"].as_str(),
            resp.tool_calls.is_some(),
        );
        let _ = sink.send(StreamChunk::Finish(reason)).await;
        Ok(resp)
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

impl AnthropicProvider {
    /// Messages API request body shared by [`Provider::chat`] and [`Provider::stream`].
    fn request_body(&self, messages: &[ChatMessage], tools: Option<&Value>) -> Value {
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

        if let Some(t) = tools {
            let anthropic_tools = Self::convert_tools(t);
            if let Some(arr) = anthropic_tools.as_array() {
                if !arr.is_empty() {
                    body["tools"] = anthropic_tools;
                }
            }
        }
        body
    }

    async fn post(&self, body: &Value) -> Result<reqwest::Response, LLMError> {
        let client = build_reqwest_client();
        Ok(client
            .post(&self.base_url)
            .header("x-api-key", &self.api_key)
            .header("anthropic-version", "2023-06-01")
            .header("Content-Type", "application/json")
            .timeout(Duration::from_secs(120))
            .json(body)
            .send()
            .await?)
    }

    fn classify_http_error(&self, status: reqwest::StatusCode, text: &str) -> LLMError {
        // PR-4: classify context-length overflow as a typed error.
        if status == reqwest::StatusCode::BAD_REQUEST && LLMError::looks_like_context_overflow(text)
        {
            return LLMError::ContextOverflow {
                tokens_attempted: 0,
                max: None,
            };
        }
        let friendly =
            crate::utils::format_api_error(status.as_u16(), text, &self.base_url, &self.model);
        LLMError::ApiError(friendly)
    }

    /// Convert a Messages API `message` object into an [`LLMResponse`].
    fn parse_message(&self, json_resp: &Value) -> LLMResponse {
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

        LLMResponse {
            content: content_text,
            tool_calls: if tool_calls.is_empty() {
                None
            } else {
                Some(tool_calls)
            },
            reasoning_content: None,
            usage,
        }
    }
}

/// HTTP status Anthropic uses for each error `type`, so an `error` event inside a stream is
/// classified (transient or not) exactly like the same error returned as an HTTP response.
fn anthropic_error_status(error_type: Option<&str>) -> reqwest::StatusCode {
    match error_type {
        Some("invalid_request_error") => reqwest::StatusCode::BAD_REQUEST,
        Some("authentication_error") => reqwest::StatusCode::UNAUTHORIZED,
        Some("permission_error") => reqwest::StatusCode::FORBIDDEN,
        Some("not_found_error") => reqwest::StatusCode::NOT_FOUND,
        Some("request_too_large") => reqwest::StatusCode::PAYLOAD_TOO_LARGE,
        Some("rate_limit_error") => reqwest::StatusCode::TOO_MANY_REQUESTS,
        Some("overloaded_error") => {
            reqwest::StatusCode::from_u16(529).unwrap_or(reqwest::StatusCode::SERVICE_UNAVAILABLE)
        }
        _ => reqwest::StatusCode::INTERNAL_SERVER_ERROR,
    }
}

/// Folds a Messages API SSE stream (`message_start`, `content_block_*`, `message_delta`,
/// `message_stop`, `error`) back into the `message` object the non-streaming endpoint
/// returns, so the final [`LLMResponse`] comes from the same [`AnthropicProvider::parse_message`].
/// Thinking blocks keep their `thinking` text and `signature`.
#[derive(Default)]
struct AnthropicStreamAccumulator {
    message: serde_json::Map<String, Value>,
    blocks: Vec<Value>,
    /// Raw `input_json_delta` fragments per block index, parsed on `content_block_stop`.
    partial_json: HashMap<usize, String>,
    /// Text blocks seen so far, to emit the `\n` that `parse_message` puts between them.
    text_blocks: usize,
}

impl AnthropicStreamAccumulator {
    fn push_data(
        &mut self,
        data: &str,
        out: &mut Vec<StreamChunk>,
        provider: &AnthropicProvider,
    ) -> Result<SseStep, LLMError> {
        if data.is_empty() {
            return Ok(SseStep::Continue);
        }
        let value: Value = serde_json::from_str(data).map_err(LLMError::ParseError)?;
        let index = value["index"].as_u64().map(|i| i as usize);
        match value["type"].as_str() {
            Some("message_start") => {
                if let Value::Object(message) = value["message"].clone() {
                    self.message = message;
                }
            }
            Some("content_block_start") => {
                let (Some(index), Some(block)) = (index, value.get("content_block")) else {
                    return Ok(SseStep::Continue);
                };
                while self.blocks.len() <= index {
                    self.blocks.push(Value::Null);
                }
                let mut block = block.clone();
                match block["type"].as_str() {
                    Some("text") => {
                        self.text_blocks += 1;
                        if self.text_blocks > 1 {
                            out.push(StreamChunk::TextDelta("\n".to_string()));
                        }
                        let text = block["text"].as_str().unwrap_or_default();
                        if !text.is_empty() {
                            out.push(StreamChunk::TextDelta(text.to_string()));
                        }
                    }
                    Some("tool_use") => {
                        out.push(StreamChunk::ToolCallDelta {
                            id: block["id"].as_str().unwrap_or_default().to_string(),
                            name: block["name"].as_str().map(str::to_string),
                            args_delta: String::new(),
                        });
                        // `input` arrives as `input_json_delta` fragments; the start event
                        // carries an empty placeholder object.
                        self.partial_json.insert(index, String::new());
                        if block.get("input").is_none() {
                            block["input"] = json!({});
                        }
                    }
                    _ => {}
                }
                self.blocks[index] = block;
            }
            Some("content_block_delta") => {
                let Some(block) = index.and_then(|i| self.blocks.get_mut(i)) else {
                    return Ok(SseStep::Continue);
                };
                let delta = &value["delta"];
                match delta["type"].as_str() {
                    Some("text_delta") => {
                        let text = delta["text"].as_str().unwrap_or_default();
                        let mut full = block["text"].as_str().unwrap_or_default().to_string();
                        full.push_str(text);
                        block["text"] = Value::String(full);
                        if !text.is_empty() {
                            out.push(StreamChunk::TextDelta(text.to_string()));
                        }
                    }
                    Some("input_json_delta") => {
                        let fragment = delta["partial_json"].as_str().unwrap_or_default();
                        if let Some(i) = index {
                            self.partial_json.entry(i).or_default().push_str(fragment);
                        }
                        out.push(StreamChunk::ToolCallDelta {
                            id: block["id"].as_str().unwrap_or_default().to_string(),
                            name: None,
                            args_delta: fragment.to_string(),
                        });
                    }
                    Some("thinking_delta") => {
                        let text = delta["thinking"].as_str().unwrap_or_default();
                        let mut full = block["thinking"].as_str().unwrap_or_default().to_string();
                        full.push_str(text);
                        block["thinking"] = Value::String(full);
                        if !text.is_empty() {
                            out.push(StreamChunk::ReasoningDelta(text.to_string()));
                        }
                    }
                    Some("signature_delta") => {
                        let sig = delta["signature"].as_str().unwrap_or_default();
                        let mut full = block["signature"].as_str().unwrap_or_default().to_string();
                        full.push_str(sig);
                        block["signature"] = Value::String(full);
                    }
                    _ => {}
                }
            }
            Some("content_block_stop") => {
                if let Some(i) = index {
                    self.finish_block(i)?;
                }
            }
            Some("message_delta") => {
                if let Some(Value::Object(delta)) = value.get("delta") {
                    for (key, v) in delta {
                        self.message.insert(key.clone(), v.clone());
                    }
                }
                // `message_delta.usage` holds cumulative counts; merge over `message_start`.
                if let Some(Value::Object(usage)) = value.get("usage") {
                    let entry = self
                        .message
                        .entry("usage".to_string())
                        .or_insert_with(|| json!({}));
                    if let Value::Object(existing) = entry {
                        for (key, v) in usage {
                            if !v.is_null() {
                                existing.insert(key.clone(), v.clone());
                            }
                        }
                    }
                }
            }
            Some("message_stop") => return Ok(SseStep::Done),
            Some("error") => {
                let status = anthropic_error_status(value["error"]["type"].as_str());
                return Err(provider.classify_http_error(status, data));
            }
            // `ping` and unknown future event types carry nothing to assemble.
            _ => {}
        }
        Ok(SseStep::Continue)
    }

    /// Parse the accumulated `input_json_delta` fragments of a finished `tool_use` block.
    fn finish_block(&mut self, index: usize) -> Result<(), LLMError> {
        let Some(raw) = self.partial_json.remove(&index) else {
            return Ok(());
        };
        if let Some(block) = self.blocks.get_mut(index) {
            block["input"] = if raw.trim().is_empty() {
                json!({})
            } else {
                serde_json::from_str(&raw).map_err(LLMError::ParseError)?
            };
        }
        Ok(())
    }

    /// The equivalent non-streaming `message` object.
    fn into_message(mut self) -> Result<Value, LLMError> {
        // A stream cut before `content_block_stop` still yields whatever JSON parses.
        let pending: Vec<usize> = self.partial_json.keys().copied().collect();
        for index in pending {
            self.finish_block(index)?;
        }
        let blocks: Vec<Value> = self.blocks.into_iter().filter(|b| !b.is_null()).collect();
        self.message
            .insert("content".to_string(), Value::Array(blocks));
        Ok(Value::Object(self.message))
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

        let body = self.request_body(messages, tools.as_ref());
        let res = self.post(&body).await?;

        let status = res.status();
        if !status.is_success() {
            let text = res.text().await.unwrap_or_default();
            return Err(self.classify_http_error(status, &text));
        }

        let raw_text = res
            .text()
            .await
            .map_err(|e| LLMError::ApiError(e.to_string()))?;
        let json_resp: Value = serde_json::from_str(&raw_text).map_err(LLMError::ParseError)?;
        Ok(self.parse_message(&json_resp))
    }

    /// Streams with `"stream": true`, forwarding text deltas as they arrive and folding the
    /// events back into the response `chat()` would have returned. A rejected streaming
    /// request (400/404/405/501, other than context overflow) is repeated through `chat()`.
    async fn stream(
        &self,
        messages: &[ChatMessage],
        tools: Option<Value>,
        sink: tokio::sync::mpsc::Sender<StreamChunk>,
    ) -> Result<LLMResponse, LLMError> {
        let mut body = self.request_body(messages, tools.as_ref());
        body["stream"] = json!(true);
        let mut res = self.post(&body).await?;

        let status = res.status();
        if !status.is_success() {
            let text = res.text().await.unwrap_or_default();
            let err = self.classify_http_error(status, &text);
            if stream_rejected(status) && !matches!(err, LLMError::ContextOverflow { .. }) {
                debug!(
                    "Anthropic streaming request rejected (HTTP {status}); retrying without stream"
                );
                let resp = self.chat(messages, tools).await?;
                crate::utils::emit_whole_response(&sink, &resp).await;
                return Ok(resp);
            }
            return Err(err);
        }

        if !crate::utils::is_event_stream(&res) {
            // A proxy that ignored `stream` answers with the plain message object.
            let raw_text = res
                .text()
                .await
                .map_err(|e| LLMError::ApiError(e.to_string()))?;
            let json_resp: Value = serde_json::from_str(&raw_text).map_err(LLMError::ParseError)?;
            let resp = self.parse_message(&json_resp);
            crate::utils::emit_whole_response(&sink, &resp).await;
            return Ok(resp);
        }

        let mut acc = AnthropicStreamAccumulator::default();
        drive_sse(
            &mut res,
            &sink,
            |e| LLMError::ApiError(e.to_string()),
            |event, out| acc.push_data(&event.data, out, self),
        )
        .await?;

        let message = acc.into_message()?;
        let resp = self.parse_message(&message);
        if let Some(usage) = &resp.usage {
            let _ = sink.send(StreamChunk::Usage(usage.clone())).await;
        }
        let reason = match message["stop_reason"].as_str() {
            _ if resp.tool_calls.is_some() => FinishReason::ToolCalls,
            None | Some("end_turn") | Some("stop_sequence") => FinishReason::Stop,
            Some("tool_use") => FinishReason::ToolCalls,
            Some("max_tokens") => FinishReason::Length,
            Some("refusal") => FinishReason::ContentFilter,
            Some(other) => FinishReason::Other(other.to_string()),
        };
        let _ = sink.send(StreamChunk::Finish(reason)).await;
        Ok(resp)
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
        let msg = ChatMessage::tool_with_error("Error: boom", "call_1", Some("exec"), true)
            .with_mcp_app("view-1".to_string(), "ui://app/dashboard".to_string());
        let v = serde_json::to_value(&msg).expect("serialize");
        assert!(
            v.get("is_error").is_none(),
            "is_error leaked onto the OpenAI-compatible wire: {v}"
        );
        assert!(
            v.get("mcp_app").is_none(),
            "mcp_app leaked onto the OpenAI-compatible wire: {v}"
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
    fn gemini_strips_mcp_additional_properties_from_tool_schemas() {
        let openai_tools = serde_json::json!([{
            "type": "function",
            "function": {
                "name": "list_documents",
                "description": "List indexed documents",
                "parameters": {
                    "type": "object",
                    "additionalProperties": false,
                    "properties": {
                        "category": {
                            "anyOf": [{"type": "string"}, {"type": "null"}],
                            "default": null
                        },
                        "hits": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "additionalProperties": true
                            }
                        }
                    }
                }
            }
        }]);
        let gemini_tools = GeminiProvider::convert_tools(&openai_tools);
        let params = &gemini_tools[0]["functionDeclarations"][0]["parameters"];
        assert!(params.get("additionalProperties").is_none());
        assert!(params["properties"]["hits"]["items"]
            .get("additionalProperties")
            .is_none());
        assert_eq!(
            params["properties"]["category"]["anyOf"][0]["type"],
            serde_json::json!("string")
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

#[cfg(test)]
mod stream_tests {
    use super::{
        AnthropicProvider, AnthropicStreamAccumulator, GeminiProvider, GeminiStreamAccumulator,
    };
    use crate::traits::StreamChunk;
    use crate::utils::{LLMError, LLMResponse, SseDecoder, SseStep};
    use serde_json::{json, Value};

    fn gemini() -> GeminiProvider {
        GeminiProvider::new(
            "https://generativelanguage.googleapis.com/v1beta/openai/",
            "k",
            "gemini-2.5-flash",
        )
    }

    fn anthropic() -> AnthropicProvider {
        AnthropicProvider::new(
            "https://api.anthropic.com/v1/messages",
            "k",
            "claude-sonnet-4",
        )
    }

    /// Feed a recorded SSE body through the shared decoder in small byte slices (to exercise
    /// split lines) and return every emitted chunk plus the final step.
    fn feed<F>(body: &str, mut push: F) -> Result<(Vec<StreamChunk>, SseStep), LLMError>
    where
        F: FnMut(&str, &mut Vec<StreamChunk>) -> Result<SseStep, LLMError>,
    {
        let mut decoder = SseDecoder::default();
        let mut events = Vec::new();
        for piece in body.as_bytes().chunks(7) {
            events.extend(decoder.push(piece));
        }
        events.extend(decoder.finish());
        let mut chunks = Vec::new();
        let mut step = SseStep::Continue;
        for event in &events {
            step = push(&event.data, &mut chunks)?;
            if step == SseStep::Done {
                break;
            }
        }
        Ok((chunks, step))
    }

    fn text_deltas(chunks: &[StreamChunk]) -> Vec<String> {
        chunks
            .iter()
            .filter_map(|c| match c {
                StreamChunk::TextDelta(t) => Some(t.clone()),
                _ => None,
            })
            .collect()
    }

    fn reasoning(chunks: &[StreamChunk]) -> String {
        chunks
            .iter()
            .filter_map(|c| match c {
                StreamChunk::ReasoningDelta(t) => Some(t.as_str()),
                _ => None,
            })
            .collect()
    }

    fn as_json(resp: &LLMResponse) -> Value {
        serde_json::to_value(resp).expect("serializable")
    }

    fn gemini_stream(body: &str) -> Result<(Vec<StreamChunk>, Value, LLMResponse), LLMError> {
        let provider = gemini();
        let mut acc = GeminiStreamAccumulator::default();
        let (chunks, _) = feed(body, |data, out| acc.push_data(data, out, &provider))?;
        let value = acc.into_response_value();
        let resp = GeminiProvider::parse_response(&value)?;
        Ok((chunks, value, resp))
    }

    fn sse(events: &[Value]) -> String {
        events
            .iter()
            .map(|e| format!("data: {e}\r\n\r\n"))
            .collect()
    }

    #[test]
    fn gemini_stream_text_only_matches_non_streaming() {
        let body = sse(&[
            json!({"candidates":[{"content":{"role":"model","parts":[{"text":"Hel"}]}}],"usageMetadata":{"promptTokenCount":4,"totalTokenCount":4}}),
            json!({"candidates":[{"content":{"role":"model","parts":[{"text":"lo, "}]}}]}),
            json!({"candidates":[{"content":{"role":"model","parts":[{"text":"world"}]},"finishReason":"STOP"}],"usageMetadata":{"promptTokenCount":4,"candidatesTokenCount":3,"totalTokenCount":7,"cachedContentTokenCount":2}}),
        ]);
        let (chunks, value, resp) = gemini_stream(&body).expect("stream");
        assert_eq!(text_deltas(&chunks), vec!["Hel", "lo, ", "world"]);
        assert_eq!(
            value["candidates"][0]["content"]["parts"]
                .as_array()
                .map(Vec::len),
            Some(1)
        );
        assert_eq!(value["candidates"][0]["finishReason"], "STOP");

        let non_streaming = json!({
            "candidates":[{"content":{"role":"model","parts":[{"text":"Hello, world"}]},"finishReason":"STOP"}],
            "usageMetadata":{"promptTokenCount":4,"candidatesTokenCount":3,"totalTokenCount":7,"cachedContentTokenCount":2}
        });
        let expected = GeminiProvider::parse_response(&non_streaming).expect("parse");
        assert_eq!(as_json(&resp), as_json(&expected));
        let usage = resp.usage.expect("usage");
        assert_eq!(
            (
                usage.prompt_tokens,
                usage.completion_tokens,
                usage.total_tokens,
                usage.cache_read_tokens
            ),
            (4, 3, 7, 2)
        );
        assert!(resp.tool_calls.is_none());
    }

    #[test]
    fn gemini_stream_function_call_keeps_thought_signature() {
        let body = sse(&[
            json!({"candidates":[{"content":{"role":"model","parts":[{"text":"Let me check."}]}}]}),
            json!({"candidates":[{"content":{"role":"model","parts":[
                {"functionCall":{"name":"read_file","args":{"path":"a.txt"}},"thoughtSignature":"c2lnLTE="},
                {"functionCall":{"name":"list_dir","args":{"path":"."},"id":"call-xyz"}}
            ]},"finishReason":"STOP"}],"usageMetadata":{"promptTokenCount":10,"candidatesTokenCount":5,"totalTokenCount":15}}),
        ]);
        let (chunks, _, resp) = gemini_stream(&body).expect("stream");
        assert_eq!(text_deltas(&chunks), vec!["Let me check."]);
        assert!(chunks.iter().any(
            |c| matches!(c, StreamChunk::ToolCallDelta { name: Some(n), .. } if n == "read_file")
        ));

        let non_streaming = json!({"candidates":[{"content":{"role":"model","parts":[
            {"text":"Let me check."},
            {"functionCall":{"name":"read_file","args":{"path":"a.txt"}},"thoughtSignature":"c2lnLTE="},
            {"functionCall":{"name":"list_dir","args":{"path":"."},"id":"call-xyz"}}
        ]},"finishReason":"STOP"}],"usageMetadata":{"promptTokenCount":10,"candidatesTokenCount":5,"totalTokenCount":15}});
        let expected = GeminiProvider::parse_response(&non_streaming).expect("parse");
        assert_eq!(as_json(&resp), as_json(&expected));

        let calls = resp.tool_calls.expect("tool calls");
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0].id, "gemini_call_1");
        assert_eq!(calls[0].function.name, "read_file");
        assert_eq!(calls[0].function.arguments, r#"{"path":"a.txt"}"#);
        assert_eq!(
            calls[0].extra_content,
            Some(json!({"thought_signature": "c2lnLTE="}))
        );
        assert_eq!(calls[1].id, "call-xyz");
        assert!(calls[1].extra_content.is_none());

        // The signature is echoed back on the next request exactly as received.
        let history = vec![crate::utils::ChatMessage {
            tool_calls: Some(calls),
            ..crate::utils::ChatMessage::assistant("Let me check.")
        }];
        let (_, contents) = GeminiProvider::convert_messages(&history);
        assert_eq!(contents[0]["parts"][1]["thoughtSignature"], "c2lnLTE=");
    }

    #[test]
    fn gemini_stream_thought_parts_are_reasoning_not_text() {
        let body = sse(&[
            json!({"candidates":[{"content":{"role":"model","parts":[{"text":"Thinking about ","thought":true}]}}]}),
            json!({"candidates":[{"content":{"role":"model","parts":[{"text":"it.","thought":true}]}}]}),
            json!({"candidates":[{"content":{"role":"model","parts":[{"text":"Answer: 42"}]}}]}),
            json!({"candidates":[{"content":{"role":"model","parts":[{"text":"","thoughtSignature":"c2ln"}]},"finishReason":"STOP"}],
                   "usageMetadata":{"promptTokenCount":3,"candidatesTokenCount":4,"thoughtsTokenCount":9,"totalTokenCount":16}}),
        ]);
        let (chunks, value, resp) = gemini_stream(&body).expect("stream");
        assert_eq!(text_deltas(&chunks), vec!["Answer: 42"]);
        assert_eq!(reasoning(&chunks), "Thinking about it.");
        assert_eq!(resp.content, "Answer: 42");
        assert_eq!(
            resp.reasoning_content.as_deref(),
            Some("Thinking about it.")
        );
        assert_eq!(resp.usage.expect("usage").total_tokens, 16);
        // Coalesced the way generateContent returns it: one thought part, one text part.
        let parts = value["candidates"][0]["content"]["parts"]
            .as_array()
            .expect("parts");
        assert_eq!(parts.len(), 2);
        assert_eq!(parts[1]["thoughtSignature"], "c2ln");
    }

    #[test]
    fn gemini_stream_blocked_prompt_is_no_content_like_chat() {
        let body = sse(&[
            json!({"promptFeedback":{"blockReason":"SAFETY"},"usageMetadata":{"promptTokenCount":3,"totalTokenCount":3}}),
        ]);
        let err = gemini_stream(&body).expect_err("blocked");
        assert!(matches!(err, LLMError::NoContent));
        let non_streaming = json!({"promptFeedback":{"blockReason":"SAFETY"}});
        assert!(matches!(
            GeminiProvider::parse_response(&non_streaming),
            Err(LLMError::NoContent)
        ));
    }

    #[test]
    fn gemini_stream_error_event_is_classified_like_http_error() {
        let body = sse(&[
            json!({"candidates":[{"content":{"role":"model","parts":[{"text":"partial"}]}}]}),
            json!({"error":{"code":503,"message":"The model is overloaded.","status":"UNAVAILABLE"}}),
        ]);
        let err = gemini_stream(&body).expect_err("error");
        assert!(err.is_transient(), "{err}");
        assert!(err.to_string().contains("(503"), "{err}");
    }

    #[test]
    fn gemini_stream_url_targets_stream_generate_content() {
        assert_eq!(
            gemini().stream_url(),
            "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:streamGenerateContent?alt=sse"
        );
    }

    fn anthropic_stream(
        body: &str,
    ) -> Result<(Vec<StreamChunk>, SseStep, Value, LLMResponse), LLMError> {
        let provider = anthropic();
        let mut acc = AnthropicStreamAccumulator::default();
        let (chunks, step) = feed(body, |data, out| acc.push_data(data, out, &provider))?;
        let message = acc.into_message()?;
        let resp = provider.parse_message(&message);
        Ok((chunks, step, message, resp))
    }

    fn named_sse(events: &[Value]) -> String {
        events
            .iter()
            .map(|e| format!("event: {}\ndata: {e}\n\n", e["type"].as_str().unwrap_or("")))
            .collect()
    }

    fn message_start() -> Value {
        json!({"type":"message_start","message":{"id":"msg_1","type":"message","role":"assistant","model":"claude-sonnet-4","content":[],"stop_reason":null,"stop_sequence":null,
               "usage":{"input_tokens":25,"cache_read_input_tokens":7,"cache_creation_input_tokens":3,"output_tokens":1}}})
    }

    #[test]
    fn anthropic_stream_text_matches_non_streaming() {
        let body = named_sse(&[
            message_start(),
            json!({"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}),
            json!({"type":"ping"}),
            json!({"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Hello"}}),
            json!({"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":" there"}}),
            json!({"type":"content_block_stop","index":0}),
            json!({"type":"message_delta","delta":{"stop_reason":"end_turn","stop_sequence":null},"usage":{"output_tokens":12}}),
            json!({"type":"message_stop"}),
        ]);
        let (chunks, step, _, resp) = anthropic_stream(&body).expect("stream");
        assert_eq!(step, SseStep::Done);
        assert_eq!(text_deltas(&chunks), vec!["Hello", " there"]);

        let non_streaming = json!({"id":"msg_1","type":"message","role":"assistant","content":[{"type":"text","text":"Hello there"}],
            "stop_reason":"end_turn","usage":{"input_tokens":25,"cache_read_input_tokens":7,"cache_creation_input_tokens":3,"output_tokens":12}});
        let expected = anthropic().parse_message(&non_streaming);
        assert_eq!(as_json(&resp), as_json(&expected));
        let usage = resp.usage.expect("usage");
        assert_eq!(
            (
                usage.prompt_tokens,
                usage.completion_tokens,
                usage.total_tokens,
                usage.cache_read_tokens,
                usage.cache_creation_tokens
            ),
            (25, 12, 37, 7, 3)
        );
    }

    #[test]
    fn anthropic_stream_tool_use_with_split_input_json() {
        let body = named_sse(&[
            message_start(),
            json!({"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}),
            json!({"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Reading it."}}),
            json!({"type":"content_block_stop","index":0}),
            json!({"type":"content_block_start","index":1,"content_block":{"type":"tool_use","id":"toolu_01","name":"read_file","input":{}}}),
            json!({"type":"content_block_delta","index":1,"delta":{"type":"input_json_delta","partial_json":""}}),
            json!({"type":"content_block_delta","index":1,"delta":{"type":"input_json_delta","partial_json":"{\"pa"}}),
            json!({"type":"content_block_delta","index":1,"delta":{"type":"input_json_delta","partial_json":"th\": \"src/ma"}}),
            json!({"type":"content_block_delta","index":1,"delta":{"type":"input_json_delta","partial_json":"in.rs\"}"}}),
            json!({"type":"content_block_stop","index":1}),
            json!({"type":"content_block_start","index":2,"content_block":{"type":"tool_use","id":"toolu_02","name":"list_dir","input":{}}}),
            json!({"type":"content_block_stop","index":2}),
            json!({"type":"message_delta","delta":{"stop_reason":"tool_use"},"usage":{"output_tokens":40}}),
            json!({"type":"message_stop"}),
        ]);
        let (chunks, _, _, resp) = anthropic_stream(&body).expect("stream");
        assert_eq!(text_deltas(&chunks), vec!["Reading it."]);
        let args: String = chunks
            .iter()
            .filter_map(|c| match c {
                StreamChunk::ToolCallDelta { id, args_delta, .. } if id == "toolu_01" => {
                    Some(args_delta.as_str())
                }
                _ => None,
            })
            .collect();
        assert_eq!(args, r#"{"path": "src/main.rs"}"#);

        let non_streaming = json!({"content":[
            {"type":"text","text":"Reading it."},
            {"type":"tool_use","id":"toolu_01","name":"read_file","input":{"path":"src/main.rs"}},
            {"type":"tool_use","id":"toolu_02","name":"list_dir","input":{}}
        ],"stop_reason":"tool_use","usage":{"input_tokens":25,"cache_read_input_tokens":7,"cache_creation_input_tokens":3,"output_tokens":40}});
        let expected = anthropic().parse_message(&non_streaming);
        assert_eq!(as_json(&resp), as_json(&expected));
        let calls = resp.tool_calls.expect("tool calls");
        assert_eq!(calls[0].function.arguments, r#"{"path":"src/main.rs"}"#);
        assert_eq!(calls[1].function.arguments, "{}");
    }

    #[test]
    fn anthropic_stream_thinking_and_signature() {
        let body = named_sse(&[
            message_start(),
            json!({"type":"content_block_start","index":0,"content_block":{"type":"thinking","thinking":"","signature":""}}),
            json!({"type":"content_block_delta","index":0,"delta":{"type":"thinking_delta","thinking":"Let me "}}),
            json!({"type":"content_block_delta","index":0,"delta":{"type":"thinking_delta","thinking":"think."}}),
            json!({"type":"content_block_delta","index":0,"delta":{"type":"signature_delta","signature":"EqQBCgIYAhIM"}}),
            json!({"type":"content_block_stop","index":0}),
            json!({"type":"content_block_start","index":1,"content_block":{"type":"text","text":""}}),
            json!({"type":"content_block_delta","index":1,"delta":{"type":"text_delta","text":"First."}}),
            json!({"type":"content_block_stop","index":1}),
            json!({"type":"content_block_start","index":2,"content_block":{"type":"text","text":""}}),
            json!({"type":"content_block_delta","index":2,"delta":{"type":"text_delta","text":"Second."}}),
            json!({"type":"content_block_stop","index":2}),
            json!({"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":30}}),
            json!({"type":"message_stop"}),
        ]);
        let (chunks, _, message, resp) = anthropic_stream(&body).expect("stream");
        assert_eq!(reasoning(&chunks), "Let me think.");
        // Streamed text equals the assembled content (blocks joined by "\n", as in chat()).
        assert_eq!(text_deltas(&chunks).concat(), resp.content);
        assert_eq!(resp.content, "First.\nSecond.");
        assert_eq!(
            message["content"][0],
            json!({"type":"thinking","thinking":"Let me think.","signature":"EqQBCgIYAhIM"})
        );
        let expected = anthropic().parse_message(&json!({"content":[
            {"type":"thinking","thinking":"Let me think.","signature":"EqQBCgIYAhIM"},
            {"type":"text","text":"First."},{"type":"text","text":"Second."}
        ],"stop_reason":"end_turn","usage":{"input_tokens":25,"cache_read_input_tokens":7,"cache_creation_input_tokens":3,"output_tokens":30}}));
        assert_eq!(as_json(&resp), as_json(&expected));
    }

    #[test]
    fn anthropic_stream_error_event_is_classified_like_http_error() {
        let body = named_sse(&[
            message_start(),
            json!({"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}),
            json!({"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Par"}}),
            json!({"type":"error","error":{"type":"overloaded_error","message":"Overloaded"}}),
        ]);
        let err = anthropic_stream(&body).expect_err("error");
        assert!(err.is_transient(), "{err}");
        assert!(
            err.to_string().contains("(529 [overloaded_error])"),
            "{err}"
        );

        let body = named_sse(&[
            json!({"type":"error","error":{"type":"invalid_request_error","message":"prompt is too long: 250000 tokens > 200000 maximum"}}),
        ]);
        let err = anthropic_stream(&body).expect_err("error");
        assert!(matches!(err, LLMError::ContextOverflow { .. }), "{err}");
    }
}
