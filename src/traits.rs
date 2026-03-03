use async_trait::async_trait;
use serde_json::Value;

// --- Trait Definitions ---

/// A Provider abstracts the generation capabilities of an LLM.
#[async_trait]
pub trait Provider: Send + Sync {
    /// Send a chat completion request with a list of messages.
    /// Messages are defined in the specific Provider implementation, 
    /// or we can use a generic `crate::utils::ChatMessage`.
    async fn chat(&self, messages: &[crate::utils::ChatMessage]) -> Result<crate::utils::LLMResponse, crate::utils::LLMError>;
}

/// A Memory abstracts the context storage capabilities of an Agent.
#[async_trait]
pub trait Memory: Send + Sync {
    /// Add a user message to the memory
    async fn add_user_message(&mut self, content: &str) -> Result<(), String>;
    
    /// Add an assistant message to the memory
    async fn add_assistant_message(&mut self, content: &str) -> Result<(), String>;

    /// Add a system message to the memory (usually at initialization)
    async fn add_system_message(&mut self, content: &str) -> Result<(), String>;

    /// Retrieve the recent context as a list of messages.
    async fn get_context(&self) -> Result<Vec<crate::utils::ChatMessage>, String>;

    /// Clear the current memory context
    async fn clear(&mut self) -> Result<(), String>;
}

/// A Tool definition that can be executed by the Agent.
#[async_trait]
pub trait Tool: Send + Sync {
    /// Name of the tool, used by the LLM to call it.
    fn name(&self) -> &str;

    /// Description of what the tool does.
    fn description(&self) -> &str;

    /// JSON Schema of the arguments expected by the tool.
    fn parameters(&self) -> Value;

    /// Execute the tool with the given JSON arguments.
    async fn execute(&self, args: Value) -> Result<String, String>;
}
