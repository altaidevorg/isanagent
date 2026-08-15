use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_json::Value;

/// A bounded, user-facing description of a pending file mutation.
///
/// The agent dispatcher creates this before asking for approval. `base_fingerprint`
/// identifies the exact file state the user reviewed; mutation tools re-check it
/// immediately before writing so an intervening edit is detected.
#[derive(Debug, Clone)]
pub struct MutationPreview {
    pub path: String,
    pub diff: String,
    pub diff_truncated: bool,
    pub base_fingerprint: String,
}

/// Machine-readable outcome of one tool dispatch. The reasoning loop and
/// telemetry must use this status instead of inferring success from prose.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ToolResultStatus {
    Success,
    Error,
}

/// Stable root-cause categories produced at the central tool boundary.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ToolErrorCode {
    InvalidToolArguments,
    NotFound,
    NotAllowed,
    PolicyDenied,
    ExecutionFailed,
    NonZeroExit,
    LegacyReportedFailure,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ToolResultError {
    pub code: ToolErrorCode,
    pub message: String,
}

/// Canonical result returned by the central executor boundary.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ToolResult {
    pub status: ToolResultStatus,
    pub content: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<ToolResultError>,
    /// True only while an old `Result<String, String>` is crossing the registry
    /// adapter. A natively typed result is never reclassified from its text.
    #[serde(skip)]
    legacy: bool,
}

impl ToolResult {
    pub fn success(content: impl Into<String>) -> Self {
        Self {
            status: ToolResultStatus::Success,
            content: content.into(),
            error: None,
            legacy: false,
        }
    }

    pub fn error(code: ToolErrorCode, message: impl Into<String>) -> Self {
        let message = message.into();
        Self::error_with_content(code, message.clone(), format!("Error: {message}"))
    }

    pub fn error_with_content(
        code: ToolErrorCode,
        message: impl Into<String>,
        content: impl Into<String>,
    ) -> Self {
        Self {
            status: ToolResultStatus::Error,
            content: content.into(),
            error: Some(ToolResultError {
                code,
                message: message.into(),
            }),
            legacy: false,
        }
    }

    pub fn is_error(&self) -> bool {
        self.status == ToolResultStatus::Error
    }

    pub fn error_code(&self) -> Option<ToolErrorCode> {
        self.error.as_ref().map(|error| error.code)
    }

    pub fn into_legacy_result(self) -> Result<String, String> {
        match self.status {
            ToolResultStatus::Success => Ok(self.content),
            ToolResultStatus::Error => Err(self
                .error
                .map(|error| error.message)
                .unwrap_or(self.content)),
        }
    }

    pub(crate) fn from_legacy(result: Result<String, String>) -> Self {
        match result {
            Ok(content) => Self {
                status: ToolResultStatus::Success,
                content,
                error: None,
                legacy: true,
            },
            Err(message) => Self {
                status: ToolResultStatus::Error,
                content: format!("Error: {message}"),
                error: Some(ToolResultError {
                    code: ToolErrorCode::ExecutionFailed,
                    message,
                }),
                legacy: true,
            },
        }
    }

    pub(crate) fn is_legacy(&self) -> bool {
        self.legacy
    }

    pub(crate) fn mark_normalized(&mut self) {
        self.legacy = false;
    }
}

// --- Trait Definitions ---

/// Reasons why an LLM generation finished.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum FinishReason {
    Stop,
    ToolCalls,
    Length,
    ContentFilter,
    Other(String),
}

/// Typed incremental chunk emitted during streaming LLM generation.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "type", content = "data", rename_all = "snake_case")]
pub enum StreamChunk {
    /// Incremental model text output delta.
    TextDelta(String),
    /// Incremental reasoning/thinking token delta.
    ReasoningDelta(String),
    /// Tool call invocation start or argument delta.
    ToolCallDelta {
        id: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        name: Option<String>,
        args_delta: String,
    },
    /// Final token usage metadata.
    Usage(crate::utils::TokenUsage),
    /// Stream finished event.
    Finish(FinishReason),
}

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

    /// Send a streaming chat completion request, forwarding chunks into `sink`.
    /// Default implementation calls `chat()` and emits one full `TextDelta` and `Finish`.
    async fn stream(
        &self,
        messages: &[crate::utils::ChatMessage],
        tools: Option<serde_json::Value>,
        sink: tokio::sync::mpsc::Sender<StreamChunk>,
    ) -> Result<crate::utils::LLMResponse, crate::utils::LLMError> {
        let resp = self.chat(messages, tools).await?;
        if !resp.content.is_empty() {
            let _ = sink
                .send(StreamChunk::TextDelta(resp.content.clone()))
                .await;
        }
        if let Some(ref reasoning) = resp.reasoning_content {
            let _ = sink
                .send(StreamChunk::ReasoningDelta(reasoning.clone()))
                .await;
        }
        if let Some(ref usage) = resp.usage {
            let _ = sink.send(StreamChunk::Usage(usage.clone())).await;
        }
        let _ = sink.send(StreamChunk::Finish(FinishReason::Stop)).await;
        Ok(resp)
    }

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

/// Execution concurrency classification for metadata-driven tool scheduling.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ExecutionMode {
    /// Safe to execute concurrently alongside other read-only tools (e.g. read_file, search_text, glob_files).
    Parallel,
    /// Standard mutating tool executed in strict sequential order (e.g. write_file, edit_file).
    Serial,
    /// Environmental barrier that requires all preceding in-flight tools to finish before execution (e.g. git_worktree).
    Barrier,
    /// Asynchronous, long-running job managed out of band (e.g. exec_background).
    Background,
}

/// Metadata policy governing tool scheduling, execution mode, and default timeout caps.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ToolPolicy {
    pub execution_mode: ExecutionMode,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub timeout_secs: Option<u64>,
}

impl Default for ToolPolicy {
    fn default() -> Self {
        Self {
            execution_mode: ExecutionMode::Serial,
            timeout_secs: None,
        }
    }
}

impl ToolPolicy {
    pub fn parallel() -> Self {
        Self {
            execution_mode: ExecutionMode::Parallel,
            timeout_secs: None,
        }
    }

    pub fn serial() -> Self {
        Self {
            execution_mode: ExecutionMode::Serial,
            timeout_secs: None,
        }
    }

    pub fn barrier() -> Self {
        Self {
            execution_mode: ExecutionMode::Barrier,
            timeout_secs: None,
        }
    }

    pub fn background() -> Self {
        Self {
            execution_mode: ExecutionMode::Background,
            timeout_secs: None,
        }
    }
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

    /// Returns the execution policy (concurrency mode, timeout) for this tool.
    fn policy(&self) -> ToolPolicy {
        ToolPolicy::default()
    }

    /// Execute the tool with the given JSON arguments.
    async fn execute(&self, args: Value) -> Result<String, String>;

    /// Return a preview for a file mutation, if this tool performs one.
    ///
    /// The default keeps existing tools source-compatible and lets the central
    /// dispatcher apply a single policy to the small set of mutation tools.
    async fn preview_mutation(&self, _args: &Value) -> Result<Option<MutationPreview>, String> {
        Ok(None)
    }

    /// Execute a mutation after the dispatcher has obtained user approval.
    ///
    /// Mutation tools override this to validate `approved_preview` before
    /// writing. Non-mutation tools retain the normal execution path.
    async fn execute_with_approved_mutation(
        &self,
        args: Value,
        _approved_preview: Option<&MutationPreview>,
    ) -> Result<String, String> {
        self.execute(args).await
    }

    /// Canonical typed execution hook used by the central dispatcher. Existing
    /// tools inherit one compatibility adapter around the legacy mutation-aware
    /// method; migrated tools override this hook and their typed status is final.
    async fn execute_with_approved_mutation_typed(
        &self,
        args: Value,
        approved_preview: Option<&MutationPreview>,
    ) -> ToolResult {
        ToolResult::from_legacy(
            self.execute_with_approved_mutation(args, approved_preview)
                .await,
        )
    }
}
