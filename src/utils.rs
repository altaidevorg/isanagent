use serde::{Deserialize, Serialize};
use serde_json::json;
use std::time::Duration;
use log::{info, debug};

// --- Data Structures ---

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMessage {
    pub role: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
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
            content: Some(content.to_string()),
            name: None,
            tool_calls: None,
            tool_call_id: None,
        }
    }

    pub fn system(content: &str) -> Self {
        Self {
            role: "system".to_string(),
            content: Some(content.to_string()),
            name: None,
            tool_calls: None,
            tool_call_id: None,
        }
    }

    pub fn assistant(content: &str) -> Self {
        Self {
            role: "assistant".to_string(),
            content: Some(content.to_string()),
            name: None,
            tool_calls: None,
            tool_call_id: None,
        }
    }

    pub fn tool(content: &str, tool_call_id: &str) -> Self {
        Self {
            role: "tool".to_string(),
            content: Some(content.to_string()),
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
            client: reqwest::Client::new(),
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
    pub async fn chat(&self, messages: &[ChatMessage], tools: Option<serde_json::Value>) -> Result<LLMResponse, LLMError> {
        debug!("Sending chat request to {} with {} messages", self.model, messages.len());

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

        let res = self.client.post(url)
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

        let raw_text = res.text().await.map_err(|e| LLMError::ApiError(e.to_string()))?;
        let json_resp: serde_json::Value = serde_json::from_str(&raw_text).map_err(LLMError::ParseError)?;
        
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

        let usage = if let Some(usage_obj) = json_resp.get("usage") {
            Some(TokenUsage {
                prompt_tokens: usage_obj["prompt_tokens"].as_u64().unwrap_or(0) as u32,
                completion_tokens: usage_obj["completion_tokens"].as_u64().unwrap_or(0) as u32,
                total_tokens: usage_obj["total_tokens"].as_u64().unwrap_or(0) as u32,
            })
        } else {
            None
        };

        let finish_reason = json_resp["choices"][0]["finish_reason"]
            .as_str()
            .unwrap_or("unknown")
            .to_string();

        if usage.as_ref().map(|u| u.completion_tokens == 0).unwrap_or(false) || content.trim().is_empty() {
            eprintln!("\n[DIAGNOSTIC] LLM returned 0 completion tokens or empty content.\nFinish reason: {}\nRaw response: {}\n", finish_reason, raw_text);
        }

        info!("LLM Response received ({} chars content, tool_calls: {}, reasoning: {}, usage: {})", 
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
        let messages = vec![
            ChatMessage::system(system),
            ChatMessage::user(user)
        ];
        self.chat(&messages, None).await
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
