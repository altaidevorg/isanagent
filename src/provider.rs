use crate::traits::Provider;
use crate::utils::{ChatMessage, LLMClient, LLMError, LLMResponse};
use async_trait::async_trait;

/// A Provider implementation that wraps the existing LLMClient
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
