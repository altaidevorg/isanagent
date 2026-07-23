use crate::utils::ContentPart;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Inbound metadata: synthetic user message enqueued when a background execution job finishes (`execution_jobs`).
pub const METADATA_SYNTHETIC_JOB_FOLLOWUP: &str = "isanagent_synthetic_job_followup";

/// Inbound metadata: synthetic user message enqueued when a cron job fires.
pub const METADATA_SYNTHETIC_CRON_TRIGGER: &str = "isanagent_synthetic_cron_trigger";
/// Inbound metadata: synthetic user message enqueued when a subagent task finishes.
pub const METADATA_SYNTHETIC_SUBAGENT_COMPLETION: &str = "isanagent_synthetic_subagent_completion";
/// Inbound metadata: hint to the agent to avoid finishing without using tools.
pub const METADATA_AUTONOMOUS_FORBID_FINAL_WITHOUT_TOOLS: &str =
    "isanagent_autonomous_forbid_final_without_tools";
/// Inbound metadata: synthetic message used to resume a background job from a notification action.
pub const METADATA_SYNTHETIC_BACKGROUND_RESUME: &str = "isanagent_synthetic_background_resume";
/// Inbound metadata: the ID of the background job being resumed.
pub const METADATA_BACKGROUND_JOB_ID: &str = "isanagent_background_job_id";
/// Inbound metadata: the ID of the clarification ticket being replied to.
pub const METADATA_CLARIFICATION_TICKET_ID: &str = "clarification_ticket_id";
/// Trusted caller-provided identifier for one foreground reasoning run.
pub const METADATA_RUN_ID: &str = "isanagent_run_id";

/// An inbound message received from a Channel (e.g. Slack, Email).
//
// NOTE: `#[non_exhaustive]` is deferred — `src/main.rs` (a separate Cargo crate from the lib)
// constructs this struct directly during background-job recovery. Adopting the marker requires
// either a builder or a constructor helper; tracked as a Phase 0.0b follow-up
// (see docs/public-api-surface.md §9.3).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InboundMessage {
    pub channel: String,
    pub sender_id: String,
    pub chat_id: String,
    pub thread_id: Option<String>,
    /// The plain-text portion of the message content.
    pub content: String,
    /// Optional multimodal attachments (e.g. images) accompanying the text content.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub attachments: Vec<ContentPart>,
    #[serde(default)]
    pub metadata: HashMap<String, serde_json::Value>,
}

pub fn get_background_job_id(
    metadata: &std::collections::HashMap<String, serde_json::Value>,
) -> Option<String> {
    metadata
        .get(METADATA_BACKGROUND_JOB_ID)
        .and_then(|v| v.as_str())
        .map(|s| s.to_string())
}

/// Session key for clarification routing and tool execution context ([`crate::tool_runtime::ToolExecCtx`]).
///
/// Format must stay aligned with [`crate::tool_runtime::ToolExecCtx`]: `channel:chat_id:thread`,
/// using an empty thread segment when `thread_id` is missing.
pub fn clarification_session_key(channel: &str, chat_id: &str, thread_id: Option<&str>) -> String {
    let thread_part = thread_id.unwrap_or("");
    format!("{}:{}:{}", channel, chat_id, thread_part)
}

impl InboundMessage {
    pub fn clarification_session_key(&self) -> String {
        clarification_session_key(&self.channel, &self.chat_id, self.thread_id.as_deref())
    }
}

/// An outbound message from the Agent to a Channel.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[non_exhaustive]
pub struct OutboundMessage {
    pub channel: String,
    pub chat_id: String,
    pub thread_id: Option<String>,
    pub content: String,
    #[serde(default)]
    pub metadata: HashMap<String, serde_json::Value>,
}

/// Why a compaction event fired. Drives the eval harness's "trigger reason" bucket.
///
/// Currently only `TurnLimit`, `TokenLimit`, and `BothLimits` are emitted by the
/// in-loop auto-compaction path (see [`src/agent/mod.rs`](../src/agent/mod.rs) auto-compaction
/// check). Future PRs will add additional triggers: `Manual` (PR-5 trigger API),
/// `AgentSelf` (PR-10 self-compaction tool), `Overflow400` (PR-4 context-overflow recovery).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[non_exhaustive]
pub enum CompactionTrigger {
    /// `user_turns >= short_term_threshold_turns` and tokens still under limit.
    TurnLimit,
    /// `approx_tokens >= short_term_threshold_tokens` and turns still under limit.
    TokenLimit,
    /// Both thresholds met in the same check.
    BothLimits,
    /// PR-4: the provider returned a context-overflow error (e.g. HTTP 400 with
    /// "input is too long"). Threshold-based compaction failed to fire in time —
    /// usually because the per-call window is tighter than the configured
    /// thresholds, or a tool call returned an unexpectedly large output.
    Overflow400,
    /// PR-5: caller-driven trigger — `AgentLogic::trigger_compaction`, the
    /// `BusMessage::TriggerCompaction` variant, or a CLI `/compact` command.
    Manual,
    /// PR-10: the agent itself called the `compact_context` tool to free up
    /// context. Distinguished from `Manual` so eval tooling can measure how
    /// often (and how usefully) the model decides to compact unprompted.
    AgentSelf,
}

/// Which reflection loop produced the event. Short-term operates per-thread on
/// recent messages; long-term aggregates summaries across threads into `MEMORY.md`.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[non_exhaustive]
pub enum ReflectionKind {
    ShortTerm,
    LongTerm,
}

/// Specific telemetry events for deep Agent observability.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[non_exhaustive]
pub enum TelemetryEvent {
    ToolCall {
        chat_id: String,
        /// Originating channel (`terminal`, `api`, …) for routing live notifications.
        #[serde(default)]
        channel: String,
        tool_name: String,
        args: String,
        /// LLM-supplied stable id for this invocation. Used by the terminal UI to mutate
        /// the pending cell in place when the matching `ToolResult` arrives. Optional for
        /// backwards compat with synthetic emit sites that have no upstream id.
        #[serde(default)]
        tool_call_id: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        background_job_id: Option<String>,
    },
    ToolResult {
        chat_id: String,
        #[serde(default)]
        channel: String,
        tool_name: String,
        result: String,
        /// True when the tool returned `Err` (failed) rather than `Ok`.
        #[serde(default)]
        is_error: bool,
        #[serde(default)]
        tool_call_id: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        background_job_id: Option<String>,
    },
    AgentThought {
        chat_id: String,
        thought: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        background_job_id: Option<String>,
    },
    AgentUsage {
        chat_id: String,
        model: String,
        prompt_tokens: u32,
        completion_tokens: u32,
        total_tokens: u32,
        /// PR-6.1: tokens served from the provider's prompt cache. `0` when the
        /// provider doesn't expose this. `#[serde(default)]` for backward compat.
        #[serde(default)]
        cache_read_tokens: u32,
        /// PR-6.1: tokens written to the provider's prompt cache on this call.
        #[serde(default)]
        cache_creation_tokens: u32,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        background_job_id: Option<String>,
    },
    ToolCallStarted {
        chat_id: String,
        tool_name: String,
        args: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        tool_call_id: Option<String>,
        background_job_id: Option<String>,
    },
    ToolCallFinished {
        chat_id: String,
        tool_name: String,
        result: String,
        /// Authoritative executor status. Defaults to success so telemetry
        /// persisted before this field was added still deserializes.
        #[serde(default)]
        is_error: bool,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        tool_call_id: Option<String>,
        background_job_id: Option<String>,
    },
    /// Mid–tool-call status (e.g. uv-managed Python env setup); not a tool result.
    ToolProgress {
        chat_id: String,
        #[serde(default)]
        channel: String,
        tool_name: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        tool_call_id: Option<String>,
        message: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        background_job_id: Option<String>,
    },
    CronTrigger {
        job_id: String,
        message: String,
    },
    /// One completed `execution_run` (no code body, no secrets).
    ExecutionRunFinished {
        chat_id: String,
        #[serde(default)]
        channel: String,
        provider_id: String,
        session_id: String,
        exit_code: Option<i32>,
        duration_ms: u64,
        stdout_len: usize,
        stderr_len: usize,
        artifact_count: usize,
        #[serde(skip_serializing_if = "Option::is_none")]
        git_head: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        description: Option<String>,
    },
    /// Background `execution_run_background` job reached a terminal state.
    ExecutionJobFinished {
        chat_id: String,
        #[serde(default)]
        channel: String,
        job_id: String,
        session_id: String,
        provider_id: String,
        /// `completed`, `failed`, `cancelled`, or `timeout`.
        status: String,
        duration_ms: u64,
        exit_code: Option<i32>,
        stdout_len: usize,
        stderr_len: usize,
        artifact_count: usize,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        description: Option<String>,
    },
    /// A sub-agent task was started from a parent chat (for UI / audit filtering).
    SubagentSpawned {
        parent_chat_id: String,
        child_chat_id: String,
        task_id: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        display_name: Option<String>,
        /// Named agent type (e.g. "researcher", "coder"). None for legacy generic spawns.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        agent_name: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        background_job_id: Option<String>,
    },
    /// Sub-agent task reached a terminal state (also persisted in SQLite).
    SubagentFinished {
        parent_chat_id: String,
        child_chat_id: String,
        task_id: String,
        /// `completed`, `failed`, or `cancelled`.
        status: String,
        /// Named agent type. None for legacy generic spawns.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        agent_name: Option<String>,
    },
    /// Shell policy decision before executing `exec`.
    ShellPolicyDecision {
        chat_id: String,
        #[serde(default)]
        channel: String,
        /// `ask`, `deny`, `allow`
        mode: String,
        /// `approval_requested`, `approval_granted`, `approval_denied`, `blocked`
        decision: String,
        /// Redacted command preview for diagnostics.
        command_preview: String,
    },
    /// Non-blocking signal that a grep/cat/wc-style shell pipeline was attempted.
    ShellGrepLikeDetected {
        chat_id: String,
        #[serde(default)]
        channel: String,
        command_preview: String,
    },
    /// Research-depth correction was injected (search without source fetch).
    ResearchDepthNudge {
        chat_id: String,
        #[serde(default)]
        channel: String,
        reason: String,
    },
    BackgroundJobUpdated {
        job_id: String,
        chat_id: String,
        #[serde(default)]
        channel: String,
        state: String,
        kind: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        detail: Option<String>,
    },
    NotificationCreated {
        notification_id: String,
        chat_id: String,
        #[serde(default)]
        channel: String,
        kind: String,
        title: String,
    },
    NotificationUpdated {
        notification_id: String,
        chat_id: String,
        #[serde(default)]
        channel: String,
        state: String,
    },
    // --- Phase 0 (compaction overhaul) telemetry ---
    /// Auto-compaction fired for a chat. `tokens_before` / `turns_before` are the
    /// values measured at the trigger check; `reason` indicates which threshold tripped.
    /// `tokens_after_preprocess` is the approximate token count of the transcript
    /// after preprocessing (image stripping, tool-result truncation) — added in PR-1.
    /// `#[serde(default)]` lets older `conversation.jsonl` blobs without this field
    /// continue to deserialize, so historical data can still be replayed.
    CompactionTriggered {
        chat_id: String,
        reason: CompactionTrigger,
        tokens_before: u32,
        turns_before: u32,
        #[serde(default)]
        tokens_after_preprocess: u32,
    },
    /// Compaction produced a summary and persisted the new reflection cursor.
    /// `wall_ms` is the duration from `CompactionTriggered` to this event.
    /// `section_completeness` is the fraction of required slots populated in the
    /// summary — `0.0` until PR-2 (sectional template) lands.
    CompactionCompleted {
        chat_id: String,
        tokens_before: u32,
        tokens_after: u32,
        wall_ms: u64,
        summary_bytes: u32,
        section_completeness: f32,
    },
    /// Compaction did not produce a usable summary (provider error, cancellation,
    /// or unparseable response). `tokens_at_failure` is the context size measured
    /// at the matching `CompactionTriggered`.
    CompactionFailed {
        chat_id: String,
        reason: String,
        tokens_at_failure: u32,
    },
    /// An idle reflection cycle began. `chat_id` is the thread id for short-term
    /// reflection and `None` for global long-term aggregation.
    /// `inputs_consumed` is the count of messages (short-term) or summaries (long-term)
    /// fed into the cycle.
    ReflectionStarted {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        chat_id: Option<String>,
        kind: ReflectionKind,
        inputs_consumed: u32,
    },
    /// Reflection cycle finished successfully. `output_bytes` is the size of the
    /// produced artifact (a summary's text or the rewritten `MEMORY.md`).
    ReflectionCompleted {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        chat_id: Option<String>,
        kind: ReflectionKind,
        output_bytes: u32,
        wall_ms: u64,
    },
    /// PR-7: the `recall_tool_result` tool was called to re-materialize a
    /// tool result that had been compacted out of the active context.
    /// Frequent recalls suggest the compaction was over-aggressive (the
    /// agent needed content we threw away); low/zero recalls mean the swap
    /// is paying for itself without rework.
    ToolResultRefetch {
        chat_id: String,
        tool_call_id: String,
    },
}

/// Log severity levels for verbose diagnostics.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum LogLevel {
    Trace,
    Debug,
    Info,
    Warn,
    Error,
}

impl std::fmt::Display for LogLevel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LogLevel::Trace => write!(f, "TRACE"),
            LogLevel::Debug => write!(f, "DEBUG"),
            LogLevel::Info => write!(f, "INFO"),
            LogLevel::Warn => write!(f, "WARN"),
            LogLevel::Error => write!(f, "ERROR"),
        }
    }
}

/// Control messages used internally by the LoggingActor.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum LoggerControlMessage {
    Flush,
    Flushed,
}

/// Structured log event for verbose diagnostics and 100% traceability.
/// All actors send these to the LoggingActor via BusMessage::Log.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[non_exhaustive]
pub struct LogEvent {
    /// ISO 8601 timestamp
    pub timestamp: String,
    /// Severity level
    pub level: LogLevel,
    /// Log source actor name: "AgentLogic", "SlackChannel", "ReflectionEngine", etc.
    pub source: String,
    /// Optional logger target/module for macro-based runtime logs
    #[serde(skip_serializing_if = "Option::is_none")]
    pub target: Option<String>,
    /// Optional chat/session context for tracing
    #[serde(skip_serializing_if = "Option::is_none")]
    pub chat_id: Option<String>,
    /// Human-readable log message
    pub message: String,
    /// Optional file path for source location
    #[serde(skip_serializing_if = "Option::is_none")]
    pub file: Option<String>,
    /// Optional source line for traceability
    #[serde(skip_serializing_if = "Option::is_none")]
    pub line: Option<u32>,
    /// Optional structured metadata (JSON key-value)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metadata: Option<serde_json::Value>,
}

impl LogEvent {
    /// Create a new LogEvent with the current timestamp.
    pub fn new(level: LogLevel, source: &str, message: &str) -> Self {
        Self {
            timestamp: chrono::Local::now().to_rfc3339(),
            level,
            source: source.to_string(),
            target: None,
            chat_id: None,
            message: message.to_string(),
            file: None,
            line: None,
            metadata: None,
        }
    }

    pub fn trace(source: &str, message: &str) -> Self {
        Self::new(LogLevel::Trace, source, message)
    }
    pub fn debug(source: &str, message: &str) -> Self {
        Self::new(LogLevel::Debug, source, message)
    }
    pub fn info(source: &str, message: &str) -> Self {
        Self::new(LogLevel::Info, source, message)
    }
    pub fn warn(source: &str, message: &str) -> Self {
        Self::new(LogLevel::Warn, source, message)
    }
    pub fn error(source: &str, message: &str) -> Self {
        Self::new(LogLevel::Error, source, message)
    }

    /// Attach a chat_id for session-level tracing.
    pub fn with_chat_id(mut self, chat_id: &str) -> Self {
        self.chat_id = Some(chat_id.to_string());
        self
    }

    /// Attach a logger target, usually the module path or explicit target.
    pub fn with_target(mut self, target: &str) -> Self {
        self.target = Some(target.to_string());
        self
    }

    /// Attach a source location.
    pub fn with_location(mut self, file: &str, line: Option<u32>) -> Self {
        self.file = Some(file.to_string());
        self.line = line;
        self
    }

    /// Attach arbitrary structured metadata.
    pub fn with_metadata(mut self, metadata: serde_json::Value) -> Self {
        self.metadata = Some(metadata);
        self
    }

    /// Format as a single human-readable log line for the verbose log file.
    pub fn format_line(&self) -> String {
        let chat_part = self
            .chat_id
            .as_deref()
            .map(|id| format!(" [chat:{}]", redact_chat_id(id)))
            .unwrap_or_default();
        let target_part = self
            .target
            .as_deref()
            .map(|target| format!(" [target:{}]", target))
            .unwrap_or_default();
        let location_part = match (self.file.as_deref(), self.line) {
            (Some(file), Some(line)) => format!(" [{}:{}]", file, line),
            (Some(file), None) => format!(" [{}]", file),
            _ => String::new(),
        };
        let meta_part = self
            .metadata
            .as_ref()
            .map(|m| format!(" {}", m))
            .unwrap_or_default();
        format!(
            "{} [{}] [{}]{}{}{} {}",
            self.timestamp,
            self.level,
            self.source,
            target_part,
            chat_part,
            location_part,
            self.message
        ) + &meta_part
    }
}

fn redact_chat_id(chat_id: &str) -> String {
    if let Some((local, domain)) = chat_id.split_once('@') {
        let first = local.chars().next().unwrap_or('*');
        return format!("{}***@{}", first, domain);
    }
    chat_id.to_string()
}

#[cfg(test)]
mod tests {
    use super::{clarification_session_key, InboundMessage, LogEvent, LogLevel, TelemetryEvent};
    use crate::tool_runtime::ToolExecCtx;

    #[test]
    fn clarification_session_key_matches_tool_exec_ctx() {
        let k = clarification_session_key("slack", "C123", Some("t1"));
        assert_eq!(
            k,
            ToolExecCtx::new("slack", "C123", Some("t1".to_string())).session_key
        );
        let k2 = clarification_session_key("terminal", "u1", None);
        assert_eq!(k2, ToolExecCtx::new("terminal", "u1", None).session_key);
        let inbound = InboundMessage {
            channel: "api".to_string(),
            sender_id: "s".to_string(),
            chat_id: "x".to_string(),
            thread_id: None,
            content: "".to_string(),
            attachments: vec![],
            metadata: Default::default(),
        };
        assert_eq!(inbound.clarification_session_key(), "api:x:");
    }

    #[test]
    fn legacy_tool_completion_defaults_to_success_status() {
        let encoded = serde_json::json!({
            "ToolCallFinished": {
                "chat_id": "chat-1",
                "tool_name": "read_file",
                "result": "Error: literal file content",
                "background_job_id": null
            }
        });
        let event: TelemetryEvent =
            serde_json::from_value(encoded).expect("deserialize legacy telemetry");

        assert!(matches!(
            event,
            TelemetryEvent::ToolCallFinished {
                is_error: false,
                ..
            }
        ));
    }

    #[test]
    fn format_line_masks_email_chat_ids() {
        let line = LogEvent {
            timestamp: "2026-03-12T00:00:00+03:00".to_string(),
            level: LogLevel::Info,
            source: "EmailChannel".to_string(),
            target: None,
            chat_id: Some("umut@altai.dev".to_string()),
            message: "processed email".to_string(),
            file: None,
            line: None,
            metadata: None,
        }
        .format_line();

        assert!(line.contains("[chat:u***@altai.dev]"));
        assert!(!line.contains("[chat:umut@altai.dev]"));
    }
}

/// A wrapper used to distinguish routing intents inside the Agent network.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum RunFailureKind {
    ProviderRetriesExhausted,
    Provider,
    Tool,
    Protocol,
    Persistence,
    Internal,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum RunStuckReason {
    DoomLoop,
    RepeatedRootCause,
    NoProgress,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash)]
#[serde(rename_all = "snake_case")]
pub enum RunBudgetLimit {
    LlmTurns,
    WallTime,
    Tokens,
    ProviderRetries,
    ContextRecoveries,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct RunBudgetSnapshot {
    pub iterations_used: usize,
    pub iterations_limit: usize,
    #[serde(default)]
    pub elapsed_ms: u64,
    #[serde(default)]
    pub elapsed_limit_ms: u64,
    #[serde(default)]
    pub tokens_used: u64,
    #[serde(default)]
    pub tokens_limit: u64,
    #[serde(default)]
    pub provider_retries_used: u32,
    #[serde(default)]
    pub provider_retries_limit: u32,
    #[serde(default)]
    pub context_recoveries_used: u32,
    #[serde(default)]
    pub context_recoveries_limit: u32,
    #[serde(default)]
    pub no_progress_turns: usize,
    #[serde(default)]
    pub repeated_root_cause_failures: usize,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub exhausted_limit: Option<RunBudgetLimit>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum RunBudgetWarningReason {
    ApproachingLimit { limit: RunBudgetLimit },
    RepeatedRootCause { failures: usize },
    NoProgress { turns: usize },
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct RunBudgetWarning {
    pub reason: RunBudgetWarningReason,
    pub budget: RunBudgetSnapshot,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum RunOutcome {
    Completed,
    Failed {
        failure: RunFailureKind,
        retryable: bool,
    },
    Cancelled,
    Stuck {
        reason: RunStuckReason,
    },
    BudgetExhausted {
        budget: RunBudgetSnapshot,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum RunLifecycleEvent {
    Started {
        run_id: String,
        chat_id: String,
    },
    Warning {
        run_id: String,
        chat_id: String,
        warning: RunBudgetWarning,
    },
    /// A previously emitted non-terminal budget warning was resolved by measurable progress
    /// (for example a successful tool result that clears a repeated-root-cause latch).
    WarningCleared {
        run_id: String,
        chat_id: String,
    },
    Terminated {
        run_id: String,
        chat_id: String,
        outcome: RunOutcome,
    },
}

/// A wrapper used to distinguish routing intents inside the Agent network.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[non_exhaustive]
pub enum BusMessage {
    Inbound(InboundMessage),
    Outbound(OutboundMessage),
    Telemetry(TelemetryEvent),
    /// Verbose structured log event for file-based diagnostics.
    Log(LogEvent),
    /// Typed foreground reasoning-run lifecycle signal.
    RunLifecycle(RunLifecycleEvent),
    /// Internal control flow for deterministic logger flush/shutdown.
    LoggerControl(LoggerControlMessage),
    /// Signal to interrupt an active reasoning loop for a specific chat.
    Cancel(String),
    /// Signal to interrupt one exact active reasoning run. Hosts with a
    /// lifecycle coordinator must use this instead of chat-only cancellation
    /// so a delayed stop cannot cancel a later run for the same chat.
    CancelRun {
        chat_id: String,
        run_id: String,
    },
    /// Apply new user direction to one exact active run at its next safe
    /// boundary, after the provider or current tool call completes.
    Steer {
        chat_id: String,
        run_id: String,
        content: String,
    },
    /// Signal to promote the current in-flight synchronous tool call (if any) to
    /// a background `ExecutionJobManager` job for the given chat. Triggered by
    /// the `/background` slash command.
    PromoteSyncToBackground(String),
    /// TUI `/new`, past-session resume, and startup: updates which `chat_id` receives
    /// terminal-scoped thought/progress telemetry (see main outbound router).
    SetTerminalSessionChat {
        chat_id: String,
    },
    /// Runtime model/provider switch triggered by the `/model` slash command.
    SwitchModel {
        provider_name: String,
        model_name: String,
        base_url: String,
        api_key: String,
    },
    /// PR-5: caller-driven compaction request for a specific chat session.
    /// `session_key` is `format!("{channel}:{chat_id}:{thread}")` — construct
    /// via [`clarification_session_key`]. `focus_instructions`, when present,
    /// is appended to the sectional summary prompt as a `FOCUS:` block so the
    /// summarizer can prioritize certain content (e.g. "drop the file-listing
    /// exploration, keep the API design decisions").
    ///
    /// Manual triggers are no-ops while a reasoning turn is already in flight
    /// for the same `chat_id` — see [`crate::agent::AgentLogic::trigger_compaction`].
    /// `trigger` distinguishes a `Manual` caller-API trigger from `AgentSelf` (the
    /// `compact_context` tool); other `CompactionTrigger` variants are reserved
    /// for the in-loop paths and should not appear here. `None` defaults to `Manual`
    /// so older serialized payloads still deserialize cleanly.
    TriggerCompaction {
        session_key: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        focus_instructions: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        trigger: Option<CompactionTrigger>,
    },
    /// TUI `/skills add <repo_url> [skill_name]` command: triggers remote skill installation.
    InstallSkill {
        repo_url: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        skill_name: Option<String>,
    },
}

#[cfg(test)]
mod run_lifecycle_tests {
    use super::{
        RunBudgetLimit, RunBudgetSnapshot, RunBudgetWarning, RunBudgetWarningReason,
        RunFailureKind, RunLifecycleEvent, RunOutcome, RunStuckReason,
    };

    #[test]
    fn lifecycle_events_round_trip_for_every_terminal_outcome() {
        let outcomes = vec![
            RunOutcome::Completed,
            RunOutcome::Failed {
                failure: RunFailureKind::ProviderRetriesExhausted,
                retryable: true,
            },
            RunOutcome::Cancelled,
            RunOutcome::Stuck {
                reason: RunStuckReason::RepeatedRootCause,
            },
            RunOutcome::BudgetExhausted {
                budget: RunBudgetSnapshot {
                    iterations_used: 24,
                    iterations_limit: 24,
                    ..RunBudgetSnapshot::default()
                },
            },
        ];

        for outcome in outcomes {
            let event = RunLifecycleEvent::Terminated {
                run_id: "run-123".to_string(),
                chat_id: "chat-456".to_string(),
                outcome,
            };
            let encoded = serde_json::to_string(&event).expect("serialize lifecycle event");
            let decoded: RunLifecycleEvent =
                serde_json::from_str(&encoded).expect("deserialize lifecycle event");
            assert_eq!(decoded, event);
        }
    }

    #[test]
    fn started_lifecycle_event_round_trips() {
        let event = RunLifecycleEvent::Started {
            run_id: "run-123".to_string(),
            chat_id: "chat-456".to_string(),
        };

        let encoded = serde_json::to_string(&event).expect("serialize lifecycle event");
        let decoded: RunLifecycleEvent =
            serde_json::from_str(&encoded).expect("deserialize lifecycle event");
        assert_eq!(decoded, event);
    }

    #[test]
    fn budget_warning_round_trips_without_becoming_terminal() {
        let event = RunLifecycleEvent::Warning {
            run_id: "run-123".to_string(),
            chat_id: "chat-456".to_string(),
            warning: RunBudgetWarning {
                reason: RunBudgetWarningReason::ApproachingLimit {
                    limit: RunBudgetLimit::Tokens,
                },
                budget: RunBudgetSnapshot {
                    iterations_used: 12,
                    iterations_limit: 50,
                    tokens_used: 4_000_000,
                    tokens_limit: 5_000_000,
                    ..RunBudgetSnapshot::default()
                },
            },
        };

        let encoded = serde_json::to_string(&event).expect("serialize lifecycle warning");
        assert!(encoded.contains("\"type\":\"warning\""));
        let decoded: RunLifecycleEvent =
            serde_json::from_str(&encoded).expect("deserialize lifecycle warning");
        assert_eq!(decoded, event);
    }

    #[test]
    fn budget_warning_cleared_round_trips() {
        let event = RunLifecycleEvent::WarningCleared {
            run_id: "run-123".to_string(),
            chat_id: "chat-456".to_string(),
        };
        let encoded = serde_json::to_string(&event).expect("serialize warning_cleared");
        assert!(encoded.contains("\"type\":\"warning_cleared\""));
        let decoded: RunLifecycleEvent =
            serde_json::from_str(&encoded).expect("deserialize warning_cleared");
        assert_eq!(decoded, event);
    }
}
