use serde::{Serialize, Deserialize};
use std::collections::HashMap;

/// An inbound message received from a Channel (e.g. Slack, Email).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InboundMessage {
    pub channel: String,
    pub sender_id: String,
    pub chat_id: String,
    pub thread_id: Option<String>,
    pub content: String,
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
        tool_name: String,
        args: String,
    },
    ToolResult {
        chat_id: String,
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

/// A wrapper used to distinguish routing intents inside the Agent network.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum BusMessage {
    Inbound(InboundMessage),
    Outbound(OutboundMessage),
    Telemetry(TelemetryEvent),
}
