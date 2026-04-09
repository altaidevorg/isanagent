use serde::{Serialize, Deserialize};
use std::collections::HashMap;
use crate::utils::ContentPart;

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
    },
    ToolResult {
        chat_id: String,
        #[serde(default)]
        channel: String,
        tool_name: String,
        result: String,
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
    CronTrigger {
        job_id: String,
        message: String,
    }
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
            LogLevel::Info  => write!(f, "INFO"),
            LogLevel::Warn  => write!(f, "WARN"),
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

    pub fn trace(source: &str, message: &str) -> Self { Self::new(LogLevel::Trace, source, message) }
    pub fn debug(source: &str, message: &str) -> Self { Self::new(LogLevel::Debug, source, message) }
    pub fn info(source: &str, message: &str) -> Self { Self::new(LogLevel::Info, source, message) }
    pub fn warn(source: &str, message: &str) -> Self { Self::new(LogLevel::Warn, source, message) }
    pub fn error(source: &str, message: &str) -> Self { Self::new(LogLevel::Error, source, message) }

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
        let chat_part = self.chat_id.as_deref()
            .map(|id| format!(" [chat:{}]", redact_chat_id(id)))
            .unwrap_or_default();
        let target_part = self.target.as_deref()
            .map(|target| format!(" [target:{}]", target))
            .unwrap_or_default();
        let location_part = match (self.file.as_deref(), self.line) {
            (Some(file), Some(line)) => format!(" [{}:{}]", file, line),
            (Some(file), None) => format!(" [{}]", file),
            _ => String::new(),
        };
        let meta_part = self.metadata.as_ref()
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
    use super::{LogEvent, LogLevel};

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
}
