use crate::utils::ContentPart;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Inbound metadata: synthetic user message enqueued when a background execution job finishes (`execution_jobs`).
pub const METADATA_SYNTHETIC_JOB_FOLLOWUP: &str = "isanagent_synthetic_job_followup";

/// Inbound metadata: synthetic user message enqueued when a cron job fires.
pub const METADATA_SYNTHETIC_CRON_TRIGGER: &str = "isanagent_synthetic_cron_trigger";

/// An inbound message received from a Channel (e.g. Slack, Email).
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
    pub metadata: HashMap<String, serde_json::Value>,
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
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OutboundMessage {
    pub channel: String,
    pub chat_id: String,
    pub thread_id: Option<String>,
    pub content: String,
    pub metadata: HashMap<String, serde_json::Value>,
}

/// Specific telemetry events for deep Agent observability.
#[derive(Debug, Clone, Serialize, Deserialize)]
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
    },
    ToolResult {
        chat_id: String,
        #[serde(default)]
        channel: String,
        tool_name: String,
        result: String,
        #[serde(default)]
        tool_call_id: Option<String>,
    },
    AgentThought {
        chat_id: String,
        thought: String,
    },
    AgentUsage {
        chat_id: String,
        model: String,
        prompt_tokens: u32,
        completion_tokens: u32,
        total_tokens: u32,
    },
    ToolCallStarted {
        chat_id: String,
        tool_name: String,
        args: String,
    },
    ToolCallFinished {
        chat_id: String,
        tool_name: String,
        result: String,
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
    },
    /// Sub-agent task reached a terminal state (also persisted in SQLite).
    SubagentFinished {
        parent_chat_id: String,
        child_chat_id: String,
        task_id: String,
        /// `completed`, `failed`, or `cancelled`.
        status: String,
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
    use super::{clarification_session_key, InboundMessage, LogEvent, LogLevel};
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
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum BusMessage {
    Inbound(InboundMessage),
    Outbound(OutboundMessage),
    Telemetry(TelemetryEvent),
    /// Verbose structured log event for file-based diagnostics.
    Log(LogEvent),
    /// Internal control flow for deterministic logger flush/shutdown.
    LoggerControl(LoggerControlMessage),
    /// Signal to interrupt an active reasoning loop for a specific chat.
    Cancel(String),
    /// Signal to promote the current in-flight synchronous tool call (if any) to
    /// a background `ExecutionJobManager` job for the given chat. Triggered by
    /// the `/background` slash command.
    PromoteSyncToBackground(String),
    /// TUI `/new`, past-session resume, and startup: updates which `chat_id` receives
    /// terminal-scoped thought/progress telemetry (see main outbound router).
    SetTerminalSessionChat {
        chat_id: String,
    },
}
