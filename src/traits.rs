use async_trait::async_trait;
use serde_json::Value;

// --- Trait Definitions ---

/// A Provider abstracts the generation capabilities of an LLM.
#[async_trait]
pub trait Provider: Send + Sync + dyn_clone::DynClone {
    /// Send a chat completion request with a list of messages.
    /// Messages are defined in the specific Provider implementation,
    /// or we can use a generic `crate::utils::ChatMessage`.
    async fn chat(
        &self,
        messages: &[crate::utils::ChatMessage],
        tools: Option<serde_json::Value>,
    ) -> Result<crate::utils::LLMResponse, crate::utils::LLMError>;

    /// PR-3: the model's input context window in tokens, if known. Used by
    /// `effective_compaction_threshold` to fire compaction at a fraction of the
    /// window rather than at an absolute count — so Sonnet (200k) and Opus (1M)
    /// behave appropriately with a single configured percentage.
    ///
    /// Default returns `None`; provider impls should override when they can
    /// determine the window from their model name. Adding the method to the
    /// trait via a default keeps Phase 0.0b's additive contract intact.
    fn context_window_tokens(&self) -> Option<usize> {
        None
    }
}

dyn_clone::clone_trait_object!(Provider);

/// A Memory abstracts the context storage capabilities of an Agent.
#[async_trait]
pub trait Memory: Send + Sync {
    /// Add a message to the memory
    async fn add_message(&mut self, message: crate::utils::ChatMessage) -> Result<(), String>;

    /// Retrieve the recent context as a list of messages.
    async fn get_context(&self) -> Result<Vec<crate::utils::ChatMessage>, String>;

    /// Retrieve the context since the last reflection/summary.
    async fn get_context_since_reflection(&self) -> Result<Vec<crate::utils::ChatMessage>, String>;

    /// Clear the current memory context
    async fn clear(&mut self) -> Result<(), String>;

    /// Clear the current memory context, keeping the most recent N messages.
    async fn clear_keep_last(&mut self, keep_last: usize) -> Result<(), String>;
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
