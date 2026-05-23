//! PR-7: `recall_tool_result` built-in tool.
//!
//! Re-materializes a previously cached tool result that was compacted out of
//! the active conversation. Cache is populated by the agent on every tool-result
//! add ([src/agent/mod.rs](../src/agent/mod.rs)) and queried via
//! `MemoryMessage::FetchToolResult`. Each successful recall emits a
//! `TelemetryEvent::ToolResultRefetch` so eval tooling can measure how often
//! the swap-and-recall cycle pays off versus generating rework.

use async_trait::async_trait;
use serde_json::Value;
use tokio::sync::mpsc::Sender;

use crate::bus::{BusMessage, TelemetryEvent};
use crate::memory::{MemoryMessage, SharedReply};
use crate::tool_runtime::current_tool_exec_ctx;
use crate::traits::Tool;
use crate::NodeHandle;

/// Tool definition. Constructed in `main.rs` with the memory node and the
/// outbound bus sender (for refetch telemetry).
pub struct RecallToolResultTool {
    pub memory_node: NodeHandle<MemoryMessage>,
    pub outbound_tx: Sender<BusMessage>,
}

#[async_trait]
impl Tool for RecallToolResultTool {
    fn name(&self) -> &str {
        "recall_tool_result"
    }

    fn description(&self) -> &str {
        "Retrieve the full content of an earlier tool result that has been compacted out \
         of the active conversation. Pass the tool_call_id printed inside the \
         `[Tool result archived. …]` placeholder. Returns an error if no result is \
         cached for that id (it was never cached, or the cache has been cleared). \
         Use sparingly — every recall undoes part of the compaction's win."
    }

    fn parameters(&self) -> Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "tool_call_id": {
                    "type": "string",
                    "description": "The LLM-supplied id of the tool call whose result you want to retrieve. Copy it verbatim from the archived placeholder text."
                }
            },
            "required": ["tool_call_id"]
        })
    }

    async fn execute(&self, args: Value) -> Result<String, String> {
        let tool_call_id = args
            .get("tool_call_id")
            .and_then(|v| v.as_str())
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
            .ok_or_else(|| {
                "Missing or empty 'tool_call_id' (string). Copy it verbatim from the \
                 [Tool result archived. …] placeholder."
                    .to_string()
            })?;

        let (tx, rx) = tokio::sync::oneshot::channel();
        self.memory_node
            .send_packet(MemoryMessage::FetchToolResult {
                tool_call_id: tool_call_id.clone(),
                reply: SharedReply::new(tx),
            })
            .await
            .map_err(|e| format!("memory bus send: {}", e))?;
        let fetched = rx
            .await
            .map_err(|_| "memory actor channel closed".to_string())?
            .map_err(|e| format!("fetch tool result: {}", e))?;

        match fetched {
            Some(full_content) => {
                // Emit refetch telemetry on every successful recall. `chat_id`
                // comes from the tool-exec context when available; an empty
                // string is the documented fallback used by other tools.
                let chat_id = current_tool_exec_ctx()
                    .map(|c| c.chat_id)
                    .unwrap_or_default();
                let _ = self
                    .outbound_tx
                    .send(BusMessage::Telemetry(TelemetryEvent::ToolResultRefetch {
                        chat_id,
                        tool_call_id: tool_call_id.clone(),
                    }))
                    .await;
                Ok(full_content)
            }
            None => Err(format!(
                "No cached tool result for id={tool_call_id}. The result was never cached \
                 (older session, or a tool that bypassed the agent's cache hook), or the \
                 cache has been cleared."
            )),
        }
    }
}
